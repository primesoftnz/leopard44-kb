"""Zone and inventory item CRUD, item-as-chunk write path, and location history.

Provisions scope guard: ``quantity`` is at-last-physical-check, never auto-decremented.
This module owns zone/item CRUD + item-as-chunk emit; the find/locate verbs and the
CLI surface live in Plan 04 (Wave 3).

Requirements: ZONE-01, ZONE-04, ZONE-05, ZONE-06, INV-01, INV-02, INV-03, D-08.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from leopard44_kb.answer import select_generation_model
from leopard44_kb.paths import validate_path

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES: frozenset[str] = frozenset({
    "spare",
    "provision",
    "safety",
    "tool",
    "toy",
})

VALID_SIDES: frozenset[str] = frozenset({"port", "stbd", "centre", "both"})
VALID_FORE_AFT: frozenset[str] = frozenset({"fwd", "mid", "aft"})

EXTRACT_TEMPERATURE: float = 0.05
EXTRACT_NUM_PREDICT: int = 150

# Strips leading ```json (or ```) and trailing ``` code fences.
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$",
    re.DOTALL,
)

ZONE_DESC_PROMPT: str = """\
You are a boat layout assistant. Given a storage zone name and its position (side, fore-aft,
vertical index), write a concise one-sentence description of where it is on the boat.
Keep it under 15 words. Return ONLY a JSON object: {"vertical_desc": "..."}.
Example: {"vertical_desc": "Lower shelf of the port aft settee locker, below the berth."}
"""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ZoneModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    label: str
    side: Optional[str] = None
    fore_aft: Optional[str] = None
    vertical_index: Optional[float] = None
    vertical_desc: Optional[str] = None
    area: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("side", mode="before")
    @classmethod
    def normalise_side(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        lower = str(v).lower().strip()
        if lower in VALID_SIDES:
            return lower
        return None

    @field_validator("fore_aft", mode="before")
    @classmethod
    def normalise_fore_aft(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        lower = str(v).lower().strip()
        if lower in VALID_FORE_AFT:
            return lower
        return None


class ItemModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    category: str
    aliases: Optional[str] = None
    brand: Optional[str] = None
    model_number: Optional[str] = None
    quantity: Optional[float] = None
    metadata: dict = Field(default_factory=dict)
    notes: Optional[str] = None

    @field_validator("category", mode="before")
    @classmethod
    def normalise_category(cls, v: object) -> str:
        lower = str(v).lower().strip()
        if lower in VALID_CATEGORIES:
            return lower
        raise ValueError(
            f"Invalid category {v!r}. Must be one of {sorted(VALID_CATEGORIES)}."
        )


# ---------------------------------------------------------------------------
# Ollama call (copied from maintenance.py)
# ---------------------------------------------------------------------------


def call_extract_json(prompt_text: str, system_prompt: str) -> dict:
    """Non-streaming Ollama /api/chat call returning a parsed JSON dict.

    Strips Markdown code fences (```json ... ```) from the response content
    before json.loads so a fenced LLM response parses correctly.

    Raises:
        RuntimeError: Ollama unreachable, model not pulled, HTTP error, or
            non-JSON response after fence-stripping. Never raises on a
            malformed-but-present content payload (that is handled by
            the caller's sanitizer).
    """
    model, _ = select_generation_model()
    # Read at call time so monkeypatch.setenv("OLLAMA_HOST", ...) works in tests.
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"

    try:
        response = httpx.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": EXTRACT_TEMPERATURE,
                    "num_predict": EXTRACT_NUM_PREDICT,
                },
            },
            timeout=60.0,
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{model}' not found — run `ollama pull {model}` then retry."
            )
        response.raise_for_status()
        content = response.json()["message"]["content"]

        # Strip code fences before json.loads.
        m = _FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()
        else:
            content = content.strip()

        return json.loads(content)

    except httpx.ConnectError:
        raise RuntimeError(
            "Ollama not reachable at :11434 — run `ollama serve` and "
            f"`ollama pull {model}`, then retry."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Ollama extraction timed out after 60s. "
            f"Check `ollama ps` for model status, then retry."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Ollama /api/chat returned HTTP {exc.response.status_code}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Ollama returned non-JSON response: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Never-raise sanitizer for zone description AI response
# ---------------------------------------------------------------------------


def _sanitize_zone_desc(raw: object) -> str:
    """Return a string vertical_desc from the AI response dict; never raises.

    Contract:
    - Non-dict raw -> return ""
    - Missing "vertical_desc" key -> return ""
    - Non-string value -> return ""
    - String value -> return it
    """
    if not isinstance(raw, dict):
        return ""
    val = raw.get("vertical_desc")
    if not isinstance(val, str):
        return ""
    return val


# ---------------------------------------------------------------------------
# Sub-slot validation helper
# ---------------------------------------------------------------------------


def _validate_sub_slot(conn: sqlite3.Connection, zone_id: int, sub_slot: object) -> None:
    """Validate that sub_slot is legal for the given zone.

    - sub_slot is None -> always OK (zone without sub-slot assignment)
    - zone has NO grid -> raise ValueError ("zone has no sub-slot grid")
    - zone has a grid -> assert (row, col) exists; raise ValueError on miss

    Args:
        conn: Open SQLite connection.
        zone_id: Zone id to validate against.
        sub_slot: None, or a dict/JSON-parseable dict with 'row' and 'col' keys.
    """
    if sub_slot is None:
        return

    # Parse sub_slot to get row/col
    if isinstance(sub_slot, str):
        try:
            sub_slot = json.loads(sub_slot)
        except Exception:
            raise ValueError(f"sub_slot is not valid JSON: {sub_slot!r}")

    if not isinstance(sub_slot, dict):
        raise ValueError(f"sub_slot must be a dict, got {type(sub_slot)!r}")

    row = sub_slot.get("row")
    col = sub_slot.get("col")

    # Check whether the zone has ANY sub-slot grid rows
    grid_count = conn.execute(
        "SELECT COUNT(*) FROM zone_sub_slots WHERE zone_id = ?",
        (zone_id,),
    ).fetchone()[0]

    if grid_count == 0:
        raise ValueError(
            f"Zone {zone_id} has no sub-slot grid; sub_slot must be NULL."
        )

    # Zone has a grid — verify the specific (row, col) exists
    slot_row = conn.execute(
        "SELECT id FROM zone_sub_slots WHERE zone_id = ? AND row_num = ? AND col_num = ?",
        (zone_id, row, col),
    ).fetchone()

    if slot_row is None:
        raise ValueError(
            f"Sub-slot (row={row}, col={col}) does not exist in zone {zone_id}."
        )


# ---------------------------------------------------------------------------
# Zone CRUD
# ---------------------------------------------------------------------------


def create_zone(
    conn: sqlite3.Connection,
    name: str,
    label: str,
    side: Optional[str] = None,
    fore_aft: Optional[str] = None,
    vertical_index: Optional[float] = None,
    vertical_desc: Optional[str] = None,
    area: Optional[str] = None,
    notes: Optional[str] = None,
    use_ai: bool = False,
) -> int:
    """Insert a zone row and return the new zone id.

    Args:
        conn: Open SQLite connection.
        name: Slug name, e.g. "stbd-aft-cabin". Must be UNIQUE.
        label: Display name, e.g. "Stbd aft cabin".
        side: One of port/stbd/centre/both or None.
        fore_aft: One of fwd/mid/aft or None.
        vertical_index: REAL orderable position; None = unspecified.
        vertical_desc: Descriptive text. When None AND use_ai=True, generated by Ollama.
            When explicitly provided, the AI call is skipped regardless of use_ai.
        area: Grouping tag, e.g. "cockpit".
        notes: Free text notes.
        use_ai: Default False (offline-safe). When True and vertical_desc is None,
            calls Ollama to generate a one-sentence vertical_desc. The CLI layer
            (Plan 04) is responsible for opting into this; tests and scripts run
            with the default False to avoid Ollama dependency.

    Returns:
        The integer id of the new zone row.

    Raises:
        sqlite3.IntegrityError: if a zone with the same name already exists.
        RuntimeError: if use_ai=True and Ollama is unreachable.
    """
    # AI-default zone description: only when use_ai=True AND no explicit desc
    if use_ai and vertical_desc is None:
        prompt = (
            f"Zone name: {name}. "
            f"Side: {side or 'unspecified'}. "
            f"Fore-aft: {fore_aft or 'unspecified'}. "
            f"Vertical index: {vertical_index if vertical_index is not None else 'unspecified'}."
        )
        raw = call_extract_json(prompt, ZONE_DESC_PROMPT)
        vertical_desc = _sanitize_zone_desc(raw) or None

    with conn:
        conn.execute(
            "INSERT INTO zones (name, label, side, fore_aft, vertical_index, "
            "vertical_desc, area, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, label, side, fore_aft, vertical_index, vertical_desc, area, notes),
        )
        zone_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    return zone_id


def create_sub_slots(
    conn: sqlite3.Connection,
    zone_id: int,
    rows: int,
    cols: int,
    row_labels: Optional[list[str]] = None,
    col_labels: Optional[list[str]] = None,
) -> int:
    """Create a rows×cols sub-slot grid for the given zone.

    Args:
        conn: Open SQLite connection.
        zone_id: Zone to attach the grid to.
        rows: Number of rows (1-based row_num values 1..rows).
        cols: Number of columns (1-based col_num values 1..cols).
        row_labels: Optional list of length `rows` for row_label text.
        col_labels: Optional list of length `cols` for col_label text.

    Returns:
        The count of rows inserted (rows * cols).

    Raises:
        sqlite3.IntegrityError: if any (zone_id, row_num, col_num) already exists.
    """
    count = 0
    with conn:
        for r in range(1, rows + 1):
            rl = row_labels[r - 1] if row_labels and r - 1 < len(row_labels) else None
            for c in range(1, cols + 1):
                cl = col_labels[c - 1] if col_labels and c - 1 < len(col_labels) else None
                conn.execute(
                    "INSERT INTO zone_sub_slots (zone_id, row_num, col_num, row_label, col_label) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (zone_id, r, c, rl, cl),
                )
                count += 1
    return count


# ---------------------------------------------------------------------------
# Chunk content builder
# ---------------------------------------------------------------------------


def _build_chunk_content(item: dict, zone: Optional[dict]) -> str:
    """Build the embeddable rich-descriptor string for an item chunk.

    Design goal: dense term coverage for BM25 (exact tokens like part numbers,
    brand names) AND semantic coverage (natural-language description of the item
    and its category) so both KNN and BM25 paths find it.

    Example output:
      "[Item] Scrabble (toy, board game) — saloon port locker, Shelf 1 Section A
       Aliases: board game, word game"

    Args:
        item: Dict with keys name, category, aliases, brand, model_number,
              metadata (JSON string or dict), notes, current_sub_slot (JSON or None).
        zone: Dict with at minimum a 'label' key, or None for location-unknown.

    Returns:
        The rich descriptor string.
    """
    parts = [f"[Item] {item['name']}"]

    cat = item["category"]
    alias_str = item.get("aliases") or ""
    if alias_str:
        parts.append(f"({cat}, {alias_str})")
    else:
        parts.append(f"({cat})")

    # Zone + sub-slot
    if zone:
        loc = zone["label"]
        sub = item.get("current_sub_slot")
        if sub:
            s = json.loads(sub) if isinstance(sub, str) else sub
            row_label = s.get("row_label") or f"row {s.get('row', '')}".strip()
            col_label = s.get("col_label") or f"col {s.get('col', '')}".strip()
            loc_suffix = f"{row_label} {col_label}".strip()
            if loc_suffix:
                loc = f"{loc}, {loc_suffix}"
        parts.append(f"— {loc}")
    else:
        parts.append("— location unknown")

    # Brand + model
    if item.get("brand"):
        parts.append(f"Brand: {item['brand']}")
    if item.get("model_number"):
        parts.append(f"Model: {item['model_number']}")

    # Category-specific fields from metadata JSON
    meta: dict = {}
    raw_meta = item.get("metadata")
    if raw_meta:
        if isinstance(raw_meta, str):
            try:
                meta = json.loads(raw_meta)
            except Exception:
                pass
        elif isinstance(raw_meta, dict):
            meta = raw_meta

    if cat == "spare" and meta.get("part_number"):
        parts.append(f"Part: {meta['part_number']}")
    elif cat == "provision" and meta.get("best_before"):
        parts.append(f"Best before: {meta['best_before']}")
    elif cat == "safety" and meta.get("expiry"):
        parts.append(f"Expiry: {meta['expiry']}")

    if item.get("notes"):
        parts.append(item["notes"])

    # Aliases line for BM25 term coverage
    if alias_str:
        parts.append(f"Aliases: {alias_str}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Item CRUD
# ---------------------------------------------------------------------------


def create_item(
    conn: sqlite3.Connection,
    name: str,
    category: str,
    zone_id: Optional[int] = None,
    aliases: Optional[str] = None,
    brand: Optional[str] = None,
    model_number: Optional[str] = None,
    current_sub_slot: Optional[dict] = None,
    metadata: Optional[dict] = None,
    notes: Optional[str] = None,
    quantity: Optional[float] = None,
    repo_root: Optional[Path] = None,
) -> int:
    """Insert an item row, write a vessel-layer chunk, and return the item id.

    Atomic/orphan-safe: failures during embed or file write roll back the item
    row and unlink any partially-written ITEM-{id}.md file.

    Step order (locked — do NOT reorder):
      1. Validate category (raises ValueError before any write)
      2. Validate sub_slot against zone grid (raises ValueError before any write)
      3. Fetch zone row for chunk content
      4. Build chunk content + content_hash
      5. Validate vessel path (raises ValueError BEFORE any row insert)
      6. Insert items row, capture item_id
      7. Try: write md_path, embed, store_source_and_chunks
         Except: unlink md_path, DELETE items row, re-raise
      8. Update chunk_source_id on items row

    Args:
        conn: Open SQLite connection.
        name: Item name.
        category: One of spare/provision/safety/tool/toy.
        zone_id: Current zone; None = location unknown.
        aliases: Comma-separated synonyms.
        brand: Brand name.
        model_number: Model or part number identifier.
        current_sub_slot: Dict with 'row' and 'col' (and optional labels), or None.
        metadata: Category-specific metadata dict (e.g. {"part_number": "22-41016"}).
        notes: Free text notes.
        quantity: Quantity at last physical check.
        repo_root: Repository root for writing data/inventory/ITEM-{id}.md.
                   Defaults to Path.cwd().

    Returns:
        The integer id of the new items row.

    Raises:
        ValueError: Invalid category, path escape, or invalid sub_slot.
        RuntimeError: Ollama unreachable during embedding.
        sqlite3.IntegrityError: DB constraint violation.
    """
    root = repo_root if repo_root is not None else Path.cwd()

    # Normalise aliases: accept list[str] or str; store as comma-separated string or None
    if isinstance(aliases, list):
        aliases = ", ".join(str(a) for a in aliases) if aliases else None

    # Step 1: Validate category via ItemModel BEFORE any write
    # (raises ValueError for invalid category like 'gizmo')
    ItemModel(name=name, category=category)

    # Step 2: Validate sub_slot BEFORE any row insert
    if current_sub_slot is not None:
        if zone_id is None:
            raise ValueError(
                "current_sub_slot requires a zone_id — cannot assign a sub-slot to an item "
                "with no zone."
            )
        _validate_sub_slot(conn, zone_id, current_sub_slot)

    # Step 3: Fetch zone row if zone_id given
    zone = None
    if zone_id is not None:
        row = conn.execute("SELECT * FROM zones WHERE id = ?", (zone_id,)).fetchone()
        if row is not None:
            zone = dict(row)

    # Step 4: Build chunk content and content_hash
    sub_slot_json = json.dumps(current_sub_slot) if current_sub_slot is not None else None
    meta_json = json.dumps(metadata or {})
    item_dict = {
        "name": name,
        "category": category,
        "aliases": aliases,
        "brand": brand,
        "model_number": model_number,
        "metadata": meta_json,
        "notes": notes,
        "current_sub_slot": sub_slot_json,
    }
    content = _build_chunk_content(item_dict, zone)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Step 5: Validate the vessel path BEFORE inserting.
    # Validate the parent directory (data/inventory/) — we don't know item_id yet.
    # This is intentionally performed before the INSERT so a path escape leaves no orphan row.
    #
    # Two-layer path guard (T-08-07):
    # (a) validate_path("vessel", inv_dir, root) — catches '..' traversal and symlinks
    #     inside data/ whose target resolves outside repo_root/data/.
    # (b) resolved_inv.relative_to(resolved_root) — catches data/ itself being a
    #     symlink to a sibling directory; validate_path cannot detect this because
    #     it resolves expected_root through the same symlink.
    inv_dir = root / "data" / "inventory"
    # Resolve paths BEFORE mkdir so a symlink escape does not create the target directory
    resolved_root = root.resolve()
    resolved_inv = (root / "data").resolve() / "inventory"
    try:
        resolved_inv.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"Inventory path {resolved_inv} escapes repo_root {resolved_root} — "
            "suspected symlink attack on data/ directory."
        )
    validate_path("vessel", inv_dir, root)
    # mkdir only after both guards pass
    inv_dir.mkdir(parents=True, exist_ok=True)

    # Step 6: INSERT the items row
    with conn:
        conn.execute(
            "INSERT INTO items (name, aliases, brand, model_number, category, "
            "current_zone_id, current_sub_slot, quantity, metadata, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name, aliases, brand, model_number, category,
                zone_id, sub_slot_json, quantity, meta_json, notes,
            ),
        )
        item_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    md_path = inv_dir / f"ITEM-{item_id}.md"

    # Step 7: Write md + embed + store — roll back on any failure
    try:
        # Write the markdown front-matter + content
        front_matter = (
            f"---\n"
            f"item_id: {item_id}\n"
            f"zone_id: {zone_id if zone_id is not None else 'null'}\n"
            f"category: {category}\n"
            f"source_type: inventory_item\n"
            f"layer: vessel\n"
            f"---\n\n"
        )
        md_path.write_text(front_matter + content, encoding="utf-8")

        # Lazy imports — keep module-level imports light (heavy ingest stack)
        from leopard44_kb.ingest.embedder import embed_texts, select_model
        from leopard44_kb.ingest.writer import store_source_and_chunks

        model_name, model_version = select_model()
        embedding = embed_texts([content], model_name)[0]

        path_str = str(md_path.relative_to(root))
        chunks = [
            {
                "content": content,
                "section_path": "",
                "section_ordinal": 0,
                "embedding": embedding,
                "metadata": {"item_id": item_id, "zone_id": zone_id},
            }
        ]
        store_source_and_chunks(
            conn,
            "vessel",
            path_str,
            "inventory_item",
            content_hash,
            chunks,
            model_name,
            model_version,
            title=f"Inventory: {name}",
        )

    except Exception:
        # Orphan-safety: unlink md file (if written) and roll back the items row
        md_path.unlink(missing_ok=True)
        with conn:
            conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        raise

    # Step 8: Update chunk_source_id (conn already in scope — no import needed)
    path_str = str(md_path.relative_to(root))
    src_row = conn.execute(
        "SELECT id FROM sources WHERE layer = 'vessel' AND path = ?",
        (path_str,),
    ).fetchone()
    if src_row is not None:
        with conn:
            conn.execute(
                "UPDATE items SET chunk_source_id = ? WHERE id = ?",
                (src_row[0], item_id),
            )

    return item_id


def update_item_location(
    conn: sqlite3.Connection,
    item_id: int,
    zone_id: int,
    sub_slot: Optional[dict] = None,
    repo_root: Optional[Path] = None,
) -> None:
    """Move an item to a new zone/sub-slot, recording location history (D-08).

    No-op guard: if the new zone_id and sub_slot exactly match the current values,
    returns immediately without appending to history or re-embedding.

    Args:
        conn: Open SQLite connection.
        item_id: Item to update.
        zone_id: New zone id.
        sub_slot: New sub-slot dict, or None.
        repo_root: Repository root for re-writing ITEM-{id}.md.

    Raises:
        ValueError: If sub_slot is invalid for the zone's grid.
    """
    root = repo_root if repo_root is not None else Path.cwd()

    # Read current state
    row = conn.execute(
        "SELECT current_zone_id, current_sub_slot, location_history, "
        "       name, category, aliases, brand, model_number, quantity, metadata, notes "
        "FROM items WHERE id = ?",
        (item_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Item {item_id} not found.")

    current_zone_id = row["current_zone_id"]
    current_sub_slot_raw = row["current_sub_slot"]

    # Parse current_sub_slot for no-op comparison
    current_sub_slot_parsed: Optional[dict] = None
    if current_sub_slot_raw is not None:
        try:
            current_sub_slot_parsed = json.loads(current_sub_slot_raw)
        except Exception:
            current_sub_slot_parsed = None

    # No-op guard (finding 8): same zone + same sub_slot -> return early
    if current_zone_id == zone_id and current_sub_slot_parsed == sub_slot:
        return

    # Validate sub_slot BEFORE making any changes
    if sub_slot is not None:
        _validate_sub_slot(conn, zone_id, sub_slot)

    # Parse location_history
    history_raw = row["location_history"]
    try:
        history: list = json.loads(history_raw)
    except Exception:
        history = []

    # Prepend prior location (newest-first)
    ts = datetime.now(timezone.utc).isoformat()
    history.insert(0, {
        "zone_id": current_zone_id,
        "sub_slot": current_sub_slot_parsed,
        "ts": ts,
    })

    sub_slot_json = json.dumps(sub_slot) if sub_slot is not None else None
    history_json = json.dumps(history)

    # Update items row
    with conn:
        conn.execute(
            "UPDATE items SET current_zone_id = ?, current_sub_slot = ?, "
            "location_history = ? WHERE id = ?",
            (zone_id, sub_slot_json, history_json, item_id),
        )

    # Re-embed: rebuild descriptor and re-write ITEM-{id}.md, then re-store on same path
    # (store_source_and_chunks handles the vec-first delete internally for same path)
    zone = None
    zone_row = conn.execute("SELECT * FROM zones WHERE id = ?", (zone_id,)).fetchone()
    if zone_row is not None:
        zone = dict(zone_row)

    item_dict = {
        "name": row["name"],
        "category": row["category"],
        "aliases": row["aliases"],
        "brand": row["brand"],
        "model_number": row["model_number"],
        "metadata": row["metadata"],
        "notes": row["notes"],
        "current_sub_slot": sub_slot_json,
    }
    content = _build_chunk_content(item_dict, zone)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    inv_dir = root / "data" / "inventory"

    # Two-layer path guard (mirrors create_item Step 5 — T-08-07):
    # (a) validate_path catches '..' traversal and symlinks inside data/.
    # (b) relative_to check catches data/ itself being a symlink to a sibling directory.
    resolved_root = root.resolve()
    resolved_inv = (root / "data").resolve() / "inventory"
    try:
        resolved_inv.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"Inventory path {resolved_inv} escapes repo_root {resolved_root} — "
            "suspected symlink attack on data/ directory."
        )
    validate_path("vessel", inv_dir, root)
    inv_dir.mkdir(parents=True, exist_ok=True)

    md_path = inv_dir / f"ITEM-{item_id}.md"

    front_matter = (
        f"---\n"
        f"item_id: {item_id}\n"
        f"zone_id: {zone_id}\n"
        f"category: {row['category']}\n"
        f"source_type: inventory_item\n"
        f"layer: vessel\n"
        f"---\n\n"
    )
    md_path.write_text(front_matter + content, encoding="utf-8")

    from leopard44_kb.ingest.embedder import embed_texts, select_model
    from leopard44_kb.ingest.writer import store_source_and_chunks

    model_name, model_version = select_model()
    embedding = embed_texts([content], model_name)[0]

    path_str = str(md_path.relative_to(root))
    chunks = [
        {
            "content": content,
            "section_path": "",
            "section_ordinal": 0,
            "embedding": embedding,
            "metadata": {"item_id": item_id, "zone_id": zone_id},
        }
    ]
    store_source_and_chunks(
        conn,
        "vessel",
        path_str,
        "inventory_item",
        content_hash,
        chunks,
        model_name,
        model_version,
        title=f"Inventory: {row['name']}",
    )

    # Update chunk_source_id in case it changed
    src_row = conn.execute(
        "SELECT id FROM sources WHERE layer = 'vessel' AND path = ?",
        (path_str,),
    ).fetchone()
    if src_row is not None:
        with conn:
            conn.execute(
                "UPDATE items SET chunk_source_id = ? WHERE id = ?",
                (src_row[0], item_id),
            )


# ---------------------------------------------------------------------------
# Find / locate / list — read paths (Plan 04, Wave 3)
# ---------------------------------------------------------------------------


def _enrich_item(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    """Return a structured enrichment dict for an items row.

    Given an items row (sqlite3.Row or dict-like), return:
      {
        "item": dict(row),
        "zone": <zones row dict or None>,
        "sub_slot": <parsed current_sub_slot JSON dict or None>,
        "history": <parsed location_history JSON list>,
      }

    All json.loads calls are guarded — malformed stored JSON returns the
    safe fallback (None or []) rather than raising.
    """
    item_dict = dict(row)

    # Resolve zone from current_zone_id
    zone: Optional[dict] = None
    zone_id = item_dict.get("current_zone_id")
    if zone_id is not None:
        zone_row = conn.execute(
            "SELECT * FROM zones WHERE id = ?", (zone_id,)
        ).fetchone()
        if zone_row is not None:
            zone = dict(zone_row)

    # Parse current_sub_slot (never-raise)
    sub_slot: Optional[dict] = None
    raw_sub = item_dict.get("current_sub_slot")
    if raw_sub is not None:
        try:
            parsed = json.loads(raw_sub) if isinstance(raw_sub, str) else raw_sub
            sub_slot = parsed if isinstance(parsed, dict) else None
        except Exception:
            sub_slot = None

    # Parse location_history (never-raise)
    history: list = []
    raw_hist = item_dict.get("location_history")
    if raw_hist is not None:
        try:
            parsed_hist = json.loads(raw_hist) if isinstance(raw_hist, str) else raw_hist
            history = parsed_hist if isinstance(parsed_hist, list) else []
        except Exception:
            history = []

    return {
        "item": item_dict,
        "zone": zone,
        "sub_slot": sub_slot,
        "history": history,
    }


def _escape_like(s: str, escape_char: str = "\\") -> str:
    """Escape LIKE metacharacters (%, _) so user input matches literally."""
    return (
        s.replace(escape_char, escape_char * 2)
        .replace("%", escape_char + "%")
        .replace("_", escape_char + "_")
    )


def find_item(conn: sqlite3.Connection, query: str) -> list[dict]:
    """Return all items whose name, aliases, brand, or model_number LIKE %query%.

    Parameterised LIKE — no f-string SQL. Results are ordered by items.name.
    Each result is enriched via _enrich_item (zone label + sub-slot + history).

    Args:
        conn: Open SQLite connection.
        query: Search string; partial matches are returned.

    Returns:
        List of enrichment dicts from _enrich_item (possibly empty).
    """
    escaped = _escape_like(query)
    pattern = f"%{escaped}%"
    rows = conn.execute(
        "SELECT * FROM items "
        "WHERE items.name LIKE ? ESCAPE '\\' OR items.aliases LIKE ? ESCAPE '\\' "
        "   OR items.brand LIKE ? ESCAPE '\\' OR items.model_number LIKE ? ESCAPE '\\' "
        "ORDER BY items.name",
        (pattern, pattern, pattern, pattern),
    ).fetchall()
    return [_enrich_item(conn, r) for r in rows]


def locate_item(
    conn: sqlite3.Connection,
    query: str,
    pool: Optional[int] = None,
) -> dict:
    """Hybrid structured-first + semantic-fallback item location (D-10).

    Step 1 (structured): find_item(conn, query) — LIKE on name/aliases/brand/model_number.
      If any rows match, return ALL matches as candidates (caller sees len(items) > 1
      when the query is ambiguous). Chunks remain empty because the structured record
      is authoritative.

    Step 2 (semantic fallback — only on Step 1 miss): retrieve(conn, query,
      layers=["vessel"], n=5, pool=20), keep only chunks whose metadata carries
      "item_id", resolve EACH item_id back to the items row via _enrich_item,
      de-duplicate by item_id. The location is read from the item record —
      never from LLM/chunk text (Pitfall 5 option a; D-10 contract).

    Args:
        conn: Open SQLite connection.
        query: Search string (natural language or exact name).
        pool: Override for the retrieve() pool parameter (default 20).

    Returns:
        {
          "found": bool,
          "items": list[dict],   # enriched structured item records
          "chunks": list[dict],  # retrieve() chunks (populated only in fallback)
        }
    """
    # Step 1 — structured LIKE match
    rows = find_item(conn, query)
    if rows:
        return {"found": True, "items": rows, "chunks": []}

    # Step 2 — semantic fallback via retrieve()
    from leopard44_kb.retrieve import retrieve  # lazy import (avoids circular + heavy dep)

    retrieve_pool = pool if pool is not None else 20
    chunks, _below_floor = retrieve(conn, query, layers=["vessel"], n=5, pool=retrieve_pool)

    # Keep only chunks that carry an item_id in metadata
    item_chunks: list[dict] = []
    for chunk in chunks:
        raw_meta = chunk.get("metadata", "{}")
        try:
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
        except Exception:
            meta = {}
        if isinstance(meta, dict) and "item_id" in meta:
            item_chunks.append(chunk)

    # Resolve each item_id back to the structured item record via _enrich_item
    seen_ids: set = set()
    enriched_items: list[dict] = []
    for chunk in item_chunks:
        raw_meta = chunk.get("metadata", "{}")
        try:
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
        except Exception:
            meta = {}
        item_id_val = meta.get("item_id")
        if item_id_val is None or item_id_val in seen_ids:
            continue
        seen_ids.add(item_id_val)
        item_row = conn.execute(
            "SELECT * FROM items WHERE id = ?", (item_id_val,)
        ).fetchone()
        if item_row is not None:
            enriched_items.append(_enrich_item(conn, item_row))

    found = bool(enriched_items or item_chunks)
    return {"found": found, "items": enriched_items, "chunks": item_chunks}


def list_items(
    conn: sqlite3.Connection,
    zone: Optional[str] = None,
    category: Optional[str] = None,
    text: Optional[str] = None,
) -> list[dict]:
    """Return all items matching the supplied filters (AND-combined), ordered by name.

    Args:
        conn: Open SQLite connection.
        zone: Zone slug to filter by (zones.name = ?); None = no zone filter.
        category: Category string to filter by (items.category = ?); None = no filter.
        text: Free-text LIKE filter on items.name / items.aliases; None = no filter.

    Returns:
        List of plain dicts with item fields plus a 'zone_label' key (or None
        when the item has no zone). Results ordered by items.name.
    """
    clauses: list[str] = []
    params: list = []

    base = "SELECT items.*, zones.label AS zone_label FROM items LEFT JOIN zones ON zones.id = items.current_zone_id"

    if zone is not None:
        # Filter by zone slug — requires the zone to exist
        clauses.append("zones.name = ?")
        params.append(zone)

    if category is not None:
        clauses.append("items.category = ?")
        params.append(category)

    if text is not None:
        escaped_text = _escape_like(text)
        pattern = f"%{escaped_text}%"
        clauses.append("(items.name LIKE ? ESCAPE '\\' OR items.aliases LIKE ? ESCAPE '\\')")
        params.extend([pattern, pattern])

    sql = base
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY items.name"

    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def delete_item(
    conn: sqlite3.Connection,
    item_id: int,
    repo_root: Optional[Path] = None,
) -> None:
    """Delete an item row, its chunk (vec_chunks-first), and its ITEM-{id}.md file.

    NULL-safe: when chunk_source_id IS NULL (prior partial-failure state), skips
    the vec_chunks/sources delete entirely and does not crash.

    MANDATORY: always unlinks ITEM-{id}.md (missing_ok=True) to prevent disk bloat.

    vec_chunks-first ordering (T-08-09):
        DELETE FROM vec_chunks WHERE source_id=? FIRST
        DELETE FROM sources WHERE id=?           SECOND (cascades chunks + FTS)

    Args:
        conn: Open SQLite connection.
        item_id: Item to delete.
        repo_root: Repository root to resolve ITEM-{id}.md path. Defaults to Path.cwd().

    Raises:
        ValueError: If item_id not found.
    """
    root = repo_root if repo_root is not None else Path.cwd()

    # Look up chunk_source_id and resolve md_path
    row = conn.execute(
        "SELECT chunk_source_id FROM items WHERE id = ?",
        (item_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Item {item_id} not found.")

    chunk_source_id = row["chunk_source_id"]

    # Resolve md_path:
    # - If chunk_source_id is set, try to get path from sources
    # - Otherwise fall back to deterministic path
    md_path: Optional[Path] = None
    if chunk_source_id is not None:
        src_row = conn.execute(
            "SELECT path FROM sources WHERE id = ?",
            (chunk_source_id,),
        ).fetchone()
        if src_row is not None:
            md_path = root / src_row["path"]

    if md_path is None:
        # Deterministic fallback
        md_path = root / "data" / "inventory" / f"ITEM-{item_id}.md"

    with conn:
        if chunk_source_id is not None:
            # vec_chunks FIRST (vec0 has no FK cascade — Pitfall 1)
            conn.execute(
                "DELETE FROM vec_chunks WHERE source_id = ?",
                (chunk_source_id,),
            )
            # sources DELETE cascades to chunks (ON DELETE CASCADE) + FTS via trigger
            conn.execute(
                "DELETE FROM sources WHERE id = ?",
                (chunk_source_id,),
            )
        # Always delete the items row
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))

    # MANDATORY: unlink the ITEM-{id}.md file (finding 2)
    md_path.unlink(missing_ok=True)
