"""Fail-soft commit for the Leopard 44 KB capture pipeline (H1, CAP-03).

CONNECTED surface — never imported by leopard44_kb.web (offline guarantee
enforced by tests/test_capture_import_boundary.py).

commit_capture performs the dual-write: inventory item row first (always
durable), then optional photo store (fail-soft — a photo I/O error downgrades
cleanly to photo_path=NULL with a warning, never aborts the capture).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# The strict inventory category enum (mirrors cli.VALID_CATEGORIES_CLI).
VALID_CATEGORIES = ("provision", "safety", "spare", "tool", "toy")

# Curated synonym map for the free-form strings the vision model returns.
# Keys are exact lower-cased phrases (NOT substrings) so ordinary boat vocabulary
# cannot misfire the way the old `"tool" in raw` / `"water" in raw` substring
# checks did (e.g. "toolbox of spares" → tool, "watermaker spares" → provision).
# Anything not matched exactly here OR in the enum falls back to "spare", the
# correct default for the vast majority of boat hardware/parts (WR-05).
_CATEGORY_SYNONYMS: dict[str, str] = {
    # provisions / consumables
    "food": "provision",
    "provisions": "provision",
    "drink": "provision",
    "water": "provision",
    "galley": "provision",
    "consumable": "provision",
    "consumables": "provision",
    # safety gear
    "flare": "safety",
    "flares": "safety",
    "lifejacket": "safety",
    "life jacket": "safety",
    "pfd": "safety",
    "epirb": "safety",
    "first aid": "safety",
    "first aid kit": "safety",
    "fire extinguisher": "safety",
    "liferaft": "safety",
    "life raft": "safety",
    "harness": "safety",
    "man overboard": "safety",
    # tools
    "tool": "tool",
    "tools": "tool",
    "toolbox": "tool",
    "hand tool": "tool",
    "power tool": "tool",
    # toys
    "toy": "toy",
    "toys": "toy",
    "game": "toy",
    "games": "toy",
}


def normalize_category(raw_category: Optional[str]) -> str:
    """Map a free-form vision category string to a valid inventory enum value.

    Resolution order (WR-05 — exact matches only, no fragile `in` substring tests):
      1. Exact enum match (case-insensitive) → that enum value.
      2. Exact curated-synonym match (case-insensitive) → mapped enum value.
      3. Otherwise → "spare" (the safe default for boat hardware/parts).

    Args:
        raw_category: The free-form category the vision model returned (or None).

    Returns:
        One of VALID_CATEGORIES.
    """
    norm = (raw_category or "").strip().lower()
    if norm in VALID_CATEGORIES:
        return norm
    if norm in _CATEGORY_SYNONYMS:
        return _CATEGORY_SYNONYMS[norm]
    return "spare"


@dataclass
class CommitResult:
    """Result of a successful capture commit.

    Attributes:
        item_id: The created items row id.
        photo_path: Repo-relative path to the stored photo, or None if the
                    photo was not stored (fail-soft H1).
        warning: A human-readable warning string if the photo store failed,
                 or None if everything succeeded.
    """
    item_id: int
    photo_path: Optional[str]
    warning: Optional[str]


def commit_capture(
    conn: sqlite3.Connection,
    result: dict,
    photo_src: Path,
    zone_id: Optional[int],
    repo_root: Path,
) -> CommitResult:
    """Write an inventory item (durable) then OPTIONALLY store the photo (fail-soft).

    Fail-soft contract (H1):
      Step 1: Call inventory.create_item with the confirmed fields — this is
              the primary durable write (item row + vessel-layer chunk).
      Step 2: The photo is OPTIONAL. Inside a try/except, call the
              capture-package store helper (capture_pkg.store_item_photo) which
              processes and writes the photo file, then UPDATEs photo_path in
              the items row.
      Step 3: On ANY photo store/processing exception, do NOT raise and do NOT
              set photo_path (it stays NULL). Return the failure reason as a
              warning string. The item row ALWAYS survives.

    Never leaves a dangling/orphan photo_path. Never aborts the capture over a
    photo I/O error.

    Args:
        conn: Open SQLite connection (with migrations applied).
        result: Normalized vision result dict (from identify_item_for_cli).
        photo_src: Source photo path (already validated before this call).
        zone_id: Resolved zone id (None if unknown/ambiguous).
        repo_root: Repository root (for item chunk write + photo dest).

    Returns:
        CommitResult(item_id, photo_path|None, warning|None)
    """
    import leopard44_kb.inventory as _inv
    import leopard44_kb.capture as _capture

    # Normalize the free-form vision category to a valid inventory enum value.
    # Vision returns strings like "deck hardware" / "adhesive" / "raw-water pump"
    # that may not match the strict enum; normalize_category resolves on exact
    # enum/synonym match and otherwise defaults to "spare" (WR-05).
    raw_category_display = (result.get("category") or "").strip()
    category = normalize_category(raw_category_display)

    # Build notes from key_properties + zone_reasoning if present.
    # WR-03: when the free-form vision category was COERCED (it is not the value
    # being stored), preserve the original in notes so the labelling intent is not
    # silently lost — the owner reviewed the coerced value in the table, but the
    # original phrasing stays recoverable.
    parts = []
    if raw_category_display and raw_category_display.lower() != category:
        parts.append(f"Vision category: {raw_category_display} (stored as {category})")
    key_props = result.get("key_properties") or []
    if key_props:
        parts.append("Properties: " + ", ".join(str(p) for p in key_props))
    zone_reasoning = result.get("zone_reasoning") or ""
    if zone_reasoning:
        parts.append(zone_reasoning)
    notes = "; ".join(parts) if parts else None

    # Step 1: Create the inventory item — this is the durable write.
    # If create_item raises, the capture aborts with the real error (not a photo issue).
    item_id = _inv.create_item(
        conn,
        name=result.get("item") or "Unknown item",
        category=category,
        zone_id=zone_id,
        brand=result.get("brand"),
        model_number=result.get("model"),
        notes=notes,
        repo_root=repo_root,
    )

    # Step 2/3: OPTIONAL photo store (fail-soft H1).
    # Call capture_pkg.store_item_photo (monkeypatch-safe for tests).
    # The real implementation in __init__.py processes the photo file and
    # UPDATEs photo_path; on any failure the item row survives with NULL photo_path.
    photo_path: Optional[str] = None
    warning: Optional[str] = None

    try:
        path_str = _capture.store_item_photo(conn, item_id, photo_src, repo_root)
        photo_path = path_str  # returned by the real store helper
    except Exception as exc:  # noqa: BLE001 — explicit fail-soft per H1
        # Photo store failed — the item row is already committed (step 1).
        # Leave photo_path NULL; capture the reason as a warning.
        warning = f"photo not saved: {exc}"

    return CommitResult(item_id=item_id, photo_path=photo_path, warning=warning)
