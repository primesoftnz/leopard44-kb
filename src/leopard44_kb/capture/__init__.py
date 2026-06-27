"""Leopard 44 KB capture package — OFFLINE BOUNDARY.

This package handles photo capture and local/cloud vision identification of
onboard items. It is a separate surface from the offline query path.

OFFLINE BOUNDARY: This package must NEVER be imported by leopard44_kb.web or
leopard44_kb.answer. The capture/ package may make cloud calls (when explicitly
requested with --cloud), which would break the zero-outbound guarantee of
the serve path if imported there.

The two surfaces (capture and serve/query) are kept strictly separate so
cloud calls in the capture path never compromise the offline `serve` guarantee.

Public symbols exported for CLI + tests (monkeypatch-safe):
  identify_item_for_cli(image_path, zones, cloud=False) → dict
  store_item_photo(conn, item_id, photo_src, repo_root) → str | None
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


def identify_item_for_cli(
    image_path: str,
    zones: list,
    cloud: bool = False,
) -> dict:
    """Identify an item from a photo, loading zones from the provided list.

    Thin module-level wrapper around vision.identify_item so that the CLI
    and tests can monkeypatch `leopard44_kb.capture.identify_item_for_cli` as a
    single symbol (same monkeypatch-safety pattern as deviation_add_cmd).

    Args:
        image_path: Path to the validated photo file.
        zones: List of zone name strings (loaded by the CLI from the DB).
        cloud: If True, use the Anthropic cloud model (H3 consent gate).

    Returns:
        Normalized vision result dict (see vision._normalize_result).
    """
    from leopard44_kb.capture.vision import identify_item
    return identify_item(image_path, zones=zones, cloud=cloud)


def store_item_photo(
    conn: sqlite3.Connection,
    item_id: int,
    photo_src: Path,
    repo_root: Path,
) -> Optional[str]:
    """Process and store a photo for an inventory item; UPDATE photo_path in the DB.

    Thin module-level wrapper around photo.store_item_photo so that the CLI's
    commit_capture can be monkeypatched in tests via `leopard44_kb.capture.store_item_photo`.

    Steps:
      1. Call photo.store_item_photo(item_id, photo_src, repo_root) to process
         (resize/GPS-strip) and write the file; returns the repo-relative path.
      2. UPDATE items SET photo_path = <path> WHERE id = item_id.
      3. Commit the connection.

    Args:
        conn: Open SQLite connection (with migrations applied).
        item_id: The inventory item row ID.
        photo_src: Source photo path.
        repo_root: Repository root (for path validation + dest resolution).

    Returns:
        Repo-relative photo_path string on success.

    Raises:
        Any exception from photo.store_item_photo (I/O, path-escape, etc.) —
        the caller (commit_capture) catches these and downgrades to fail-soft.
    """
    from leopard44_kb.capture.photo import store_item_photo as _store
    path_str = _store(item_id, photo_src, repo_root)
    try:
        conn.execute(
            "UPDATE items SET photo_path = ? WHERE id = ?",
            (path_str, item_id),
        )
        conn.commit()
    except Exception:
        # WR-01: the processed file is already on disk but the row still points
        # at nothing (photo_path stays NULL once commit_capture downgrades to
        # fail-soft). Unlink the orphan so a failed dual-write leaves neither a
        # row reference NOR a dangling file that a later capture would overwrite.
        (Path(repo_root) / path_str).unlink(missing_ok=True)
        raise
    return path_str
