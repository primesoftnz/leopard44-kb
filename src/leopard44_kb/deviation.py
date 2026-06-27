"""Factory-deviation log: NL extraction, dual-write core, and vessel-layer chunk.

DEV-01: create_deviation writes a deviations row + a vessel-layer markdown chunk
(source_type='deviation') in one atomic step with orphan-safe rollback.
DEV-02: deviation chunks surface in the SAME RRF path as maintenance-log entries;
no retrieve.py change is needed because 'deviation' is an authoritative source_type.

Non-streaming Ollama /api/chat extraction mirrors the maintenance.py ladder exactly.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from leopard44_kb.answer import select_generation_model
from leopard44_kb.paths import validate_path

# ---------------------------------------------------------------------------
# Module constants — reuse LOCKED values from maintenance.py (never change)
# ---------------------------------------------------------------------------

EXTRACT_TEMPERATURE: float = 0.05
EXTRACT_NUM_PREDICT: int = 150

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DEVIATION_SYSTEM_PROMPT: str = """\
You are a vessel deviation log assistant. Extract structured fields from the text.
Return ONLY a JSON object with these fields:
  component    (string — the boat system or part that differs from factory spec, required)
  factory_spec (string — what the factory originally installed; null if not mentioned)
  as_built     (string — what is actually installed / how it was modified; null if not mentioned)
  reason       (string — why the change was made; null if not mentioned)
  date_noted   (string — date this deviation was noted or installed; null if not mentioned)
  notes        (string — any additional context; null if not mentioned)

Return only the JSON object with no other text.\
"""

DEVIATION_RETRY_PROMPT: str = """\
You are a vessel deviation log assistant. The previous extraction was incomplete.
Try again to extract ALL fields from the text below.
Return ONLY a JSON object with exactly these keys:
component, factory_spec, as_built, reason, date_noted, notes.
Missing values must be null, not absent.\
"""

# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class DeviationExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    component: str
    factory_spec: Optional[str] = None
    as_built: Optional[str] = None
    reason: Optional[str] = None
    date_noted: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("component", mode="before")
    @classmethod
    def component_not_empty(cls, v: object) -> str:
        s = str(v).strip() if v is not None else ""
        if not s:
            raise ValueError("component must be a non-empty string")
        return s


# ---------------------------------------------------------------------------
# NOTE (Phase 11, intentional duplication):
# call_extract_json + _FENCE_RE are copied verbatim from maintenance.py rather
# than shared. See 11-02-PLAN DRIFT NOTE — consolidate into a shared extraction
# helper in a later phase.
# ---------------------------------------------------------------------------

# Strips leading ```json (or ```) and trailing ``` code fences.
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$",
    re.DOTALL,
)


def call_extract_json(prompt_text: str, system_prompt: str) -> dict:
    """Non-streaming Ollama /api/chat call returning a parsed JSON dict.

    Strips Markdown code fences (```json ... ```) from the response content
    before json.loads so a fenced LLM response parses correctly.

    Raises:
        RuntimeError: Ollama unreachable, model not pulled, HTTP error, or
            non-JSON response after fence-stripping. Never raises on a
            malformed-but-present content payload (that is handled by
            extract_fields' sanitizer).
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
# Never-raise sanitizer for deviation extraction
# ---------------------------------------------------------------------------


def _sanitize_payload(raw: object) -> dict:
    """Return a dict guaranteed to satisfy DeviationExtraction.model_validate.

    Never raises on content. Contract:
    - Non-dict raw -> start from {}
    - component: coerce non-str to "other"; guarantee present
    - All optional fields: coerce non-str to None; never raise
    """
    if not isinstance(raw, dict):
        safe: dict = {}
    else:
        safe = {
            k: raw[k]
            for k in ("component", "factory_spec", "as_built", "reason", "date_noted", "notes")
            if k in raw
        }

    # Optional string fields: must be str or None
    for key in ("factory_spec", "as_built", "reason", "date_noted", "notes"):
        val = safe.get(key)
        if val is not None and not isinstance(val, str):
            safe[key] = None

    # component: required field — coerce non-str to "other" and guarantee present
    comp = safe.get("component")
    if not isinstance(comp, str) or not comp.strip():
        safe["component"] = "other"
    safe.setdefault("component", "other")

    return safe


# ---------------------------------------------------------------------------
# extract_fields
# ---------------------------------------------------------------------------


def extract_fields(text: str) -> DeviationExtraction:
    """Extract structured deviation fields from free-text using the local LLM.

    Re-prompts once on ValidationError. If both calls fail validation, runs the
    strict never-raise sanitizer so a poor extraction never loses the entry.

    Args:
        text: Natural-language deviation description text.

    Returns:
        DeviationExtraction with extracted fields.

    Raises:
        RuntimeError: Only if Ollama itself is unreachable or returns a transport
            error. Malformed-but-present LLM content never raises.
    """
    raw = call_extract_json(text, DEVIATION_SYSTEM_PROMPT)
    try:
        result = DeviationExtraction.model_validate(raw)
    except Exception:
        # One retry with the explicit-field retry prompt.
        raw2 = call_extract_json(text, DEVIATION_RETRY_PROMPT)
        try:
            result = DeviationExtraction.model_validate(raw2)
        except Exception:
            # Strict sanitizer — never raises on content.
            safe = _sanitize_payload(raw2)
            try:
                result = DeviationExtraction.model_validate(safe)
            except Exception:
                # Last-resort guarantee: a minimal valid object.
                result = DeviationExtraction(component="other")

    return result


# ---------------------------------------------------------------------------
# Chunk content builder
# ---------------------------------------------------------------------------


def _build_chunk_content(extraction: DeviationExtraction, zone: Optional[dict]) -> str:
    """Build the embeddable rich-descriptor string for a deviation chunk.

    Design goal: dense term coverage for BM25 (component name, factory/as-built
    part IDs) AND semantic coverage (natural-language description) so both KNN
    and BM25 paths find it. Mirrors inventory._build_chunk_content pattern.

    Args:
        extraction: Validated DeviationExtraction.
        zone: Dict with at minimum a 'label' key, or None for location-unknown.

    Returns:
        The rich descriptor string prefixed '[Deviation]'.
    """
    parts = [f"[Deviation] {extraction.component}"]

    if extraction.factory_spec:
        parts.append(f"factory: {extraction.factory_spec}")
    if extraction.as_built:
        parts.append(f"as-built: {extraction.as_built}")
    if extraction.reason:
        parts.append(f"reason: {extraction.reason}")
    if extraction.date_noted:
        parts.append(f"noted: {extraction.date_noted}")

    # Zone location
    if zone:
        parts.append(f"— {zone['label']}")
    else:
        parts.append("— location unknown")

    if extraction.notes:
        parts.append(extraction.notes)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# create_deviation — dual-write core
# ---------------------------------------------------------------------------


def create_deviation(
    conn: sqlite3.Connection,
    extraction: Optional[DeviationExtraction] = None,
    original_text: Optional[str] = None,
    *,
    component: Optional[str] = None,
    factory_spec: Optional[str] = None,
    as_built: Optional[str] = None,
    reason: Optional[str] = None,
    date_noted: Optional[str] = None,
    notes: Optional[str] = None,
    zone_id: Optional[int] = None,
    repo_root: Optional[Path] = None,
) -> int:
    """Insert a deviations row, write DEV-{id}.md, embed, and store a vessel-layer chunk.

    Atomic/orphan-safe: failures during embed or file write roll back the deviations
    row and unlink any partially-written DEV-{id}.md file.

    Accepts either a DeviationExtraction object (positional) or keyword field args
    (component, factory_spec, etc.) — both calling conventions supported.

    Step order (locked — mirrors inventory.create_item; do NOT reorder):
      1. Build/validate extraction object from args
      2. Validate zone_id FK existence (BEFORE any insert)
      3. Fetch zone row for chunk content
      4. Build chunk content + content_hash
      5. Two-layer path guard + validate_path("vessel", dir) BEFORE insert
      6. INSERT deviations row, capture id
      7. Try: write DEV-{id}.md, embed, store_source_and_chunks
         Except: unlink md_path, DELETE deviations row, re-raise
      8. UPDATE deviations.chunk_source_id

    Args:
        conn: Open SQLite connection.
        extraction: Pre-validated DeviationExtraction object (optional).
        original_text: Original free-text entry used as file body (optional).
        component: Component name (used if extraction is None).
        factory_spec: Factory specification (used if extraction is None).
        as_built: As-built specification (used if extraction is None).
        reason: Reason for deviation (used if extraction is None).
        date_noted: Date the deviation was noted (used if extraction is None).
        notes: Additional notes (used if extraction is None).
        zone_id: Optional zone FK. None = location unknown. Non-existent id raises ValueError.
        repo_root: Repository root for writing data/deviations/DEV-{id}.md.
                   Defaults to Path.cwd().

    Returns:
        The integer id of the new deviations row.

    Raises:
        ValueError: Non-existent zone_id, path escape, or missing component.
        RuntimeError: Ollama unreachable during embedding.
        sqlite3.IntegrityError: DB constraint violation.
    """
    root = repo_root if repo_root is not None else Path.cwd()

    # Step 1: Build extraction from args if not provided as object
    if extraction is None:
        if not component:
            raise ValueError("component is required for create_deviation")
        extraction = DeviationExtraction(
            component=component,
            factory_spec=factory_spec,
            as_built=as_built,
            reason=reason,
            date_noted=date_noted,
            notes=notes,
        )

    # Step 2: Validate zone_id FK existence BEFORE any insert
    if zone_id is not None:
        zone_exists = conn.execute(
            "SELECT id FROM zones WHERE id = ?", (zone_id,)
        ).fetchone()
        if zone_exists is None:
            raise ValueError(f"zone {zone_id} not found")

    # Step 3: Fetch zone row for chunk content
    zone: Optional[dict] = None
    if zone_id is not None:
        zone_row = conn.execute(
            "SELECT * FROM zones WHERE id = ?", (zone_id,)
        ).fetchone()
        if zone_row is not None:
            zone = dict(zone_row)

    # Step 4: Build chunk content (the retrievable [Deviation] descriptor).
    # content_hash is computed from the written FILE below (step 7), NOT from this
    # descriptor, so a later `l44 ingest data/` re-ingest of the unchanged file
    # no-ops (file_content_hash match) instead of re-storing it — which would
    # otherwise downgrade source_type and drop the deviation_id chunk metadata.
    content = _build_chunk_content(extraction, zone)

    # Step 5: Two-layer path guard + validate_path BEFORE insert
    dev_dir = root / "data" / "deviations"
    # Two-layer path guard (mirrors inventory.create_item T-08-07):
    # (a) relative_to check catches data/ itself being a symlink to a sibling dir
    # (b) validate_path catches '..' traversal and symlinks inside data/
    resolved_root = root.resolve()
    resolved_dev = (root / "data").resolve() / "deviations"
    try:
        resolved_dev.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"Deviations path {resolved_dev} escapes repo_root {resolved_root} — "
            "suspected symlink attack on data/ directory."
        )
    validate_path("vessel", dev_dir, root)
    # mkdir only after both guards pass
    dev_dir.mkdir(parents=True, exist_ok=True)

    # Step 6: INSERT the deviations row
    with conn:
        conn.execute(
            "INSERT INTO deviations "
            "(component, factory_spec, as_built, reason, date_noted, zone_id, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                extraction.component,
                extraction.factory_spec,
                extraction.as_built,
                extraction.reason,
                extraction.date_noted,
                zone_id,
                extraction.notes,
            ),
        )
        deviation_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    md_path = dev_dir / f"DEV-{deviation_id}.md"

    # Step 7: Write md + embed + store — roll back on any failure
    # Step 8 (chunk_source_id back-link) is INSIDE this try block so any failure
    # there triggers the same orphan-safe cleanup (D-16 Item 1 fix).
    try:
        # Write the markdown front-matter + body.
        # Item 3 fix (D-16): always write the [Deviation] descriptor as the file body
        # (NOT original_text) so parse_deviation_entry returns the same content shape
        # on re-ingest — making the file self-consistent with the embedded chunk.
        # original_text is accepted for API compatibility but is not used as the body.
        zone_id_yaml = zone_id if zone_id is not None else "null"
        front_matter = (
            f"---\n"
            f"deviation_id: {deviation_id}\n"
            f"zone_id: {zone_id_yaml}\n"
            f"source_type: deviation\n"
            f"layer: vessel\n"
            f"---\n\n"
        )
        md_path.write_text(front_matter + content, encoding="utf-8")

        # Lazy imports — keep module-level imports light (heavy ingest stack)
        from leopard44_kb.ingest.embedder import embed_texts, select_model
        from leopard44_kb.ingest.writer import file_content_hash, store_source_and_chunks

        # Hash the actual written FILE (front-matter + descriptor body) so a later
        # re-ingest of an unchanged deviation file no-ops (content_hash match)
        # instead of re-storing — preserving source_type and chunk metadata.
        content_hash = file_content_hash(md_path)

        model_name, model_version = select_model()
        embedding = embed_texts([content], model_name)[0]

        path_str = str(md_path.relative_to(root))
        chunks = [
            {
                "content": content,
                "section_path": "",
                "section_ordinal": 0,
                "embedding": embedding,
                "metadata": {"deviation_id": deviation_id, "zone_id": zone_id},
            }
        ]
        store_source_and_chunks(
            conn,
            "vessel",
            path_str,
            "deviation",
            content_hash,
            chunks,
            model_name,
            model_version,
            title=f"Deviation: {extraction.component}",
        )

        # Step 8: Update chunk_source_id on the deviations row.
        # INSIDE the try block (D-16 Item 1 fix) so a failure here triggers the same
        # md-unlink + deviations-row DELETE + source-row DELETE rollback below.
        src_row = conn.execute(
            "SELECT id FROM sources WHERE layer = 'vessel' AND path = ?",
            (path_str,),
        ).fetchone()
        if src_row is not None:
            with conn:
                conn.execute(
                    "UPDATE deviations SET chunk_source_id = ? WHERE id = ?",
                    (src_row[0], deviation_id),
                )

    except Exception:
        # Orphan-safety: unlink md file (if written), delete the deviations row,
        # and delete the source row (cascade-deletes its chunks) if store already ran.
        # path_str is always computable from md_path/root (both defined before the try).
        md_path.unlink(missing_ok=True)
        with conn:
            conn.execute("DELETE FROM deviations WHERE id = ?", (deviation_id,))
        _cleanup_path = str(md_path.relative_to(root))
        try:
            with conn:
                conn.execute(
                    "DELETE FROM sources WHERE layer = 'vessel' AND path = ?",
                    (_cleanup_path,),
                )
        except Exception:
            pass  # best-effort: source not yet stored, or already gone
        raise

    return deviation_id
