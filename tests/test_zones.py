# RED state until Wave 2 (Plan 02) lands. Import of leopard44_kb.inventory fails at
# collection time because inventory.py does not yet exist — a collection-time
# ModuleNotFoundError is the EXPECTED Wave-0 state, consistent with the Phase 1-5
# import-at-collection RED convention. The module-top import is deliberate.
"""Tests for ZONE-01/04/05/06 — zone model, schema constraints, sub-slot grid,
and seed migration.

Per-requirement verification map source:
  .planning/phases/08-zone-taxonomy-inventory-core/08-VALIDATION.md
"""
from __future__ import annotations

import sqlite3

import pytest

import leopard44_kb.inventory as inv  # collection-time ModuleNotFoundError = expected RED state


# ---------------------------------------------------------------------------
# ZONE-01: create_zone — model + vessel-layer + unique constraint
# ---------------------------------------------------------------------------


def test_create_zone(empty_db):
    """create_zone inserts a row and returns an int id; row fields match inputs."""
    # Use a name not present in the seed data to avoid UNIQUE collision with migration 002.
    zone_id = inv.create_zone(
        empty_db,
        "test-stbd-aft-cabin",
        "Test stbd aft cabin",
        side="stbd",
        fore_aft="aft",
        area="aft-cabin",
    )
    assert isinstance(zone_id, int), f"Expected int zone_id, got {type(zone_id)}"

    row = empty_db.execute(
        "SELECT * FROM zones WHERE id = ?", (zone_id,)
    ).fetchone()
    assert row is not None, "Expected a zones row for the returned id"
    assert row["side"] == "stbd", f"Expected side='stbd', got {row['side']!r}"
    assert row["fore_aft"] == "aft", f"Expected fore_aft='aft', got {row['fore_aft']!r}"


def test_create_zone_duplicate(empty_db):
    """Two zones with the same name slug raise sqlite3.IntegrityError (UNIQUE)."""
    inv.create_zone(empty_db, "bow-locker", "Bow locker")
    with pytest.raises(sqlite3.IntegrityError):
        inv.create_zone(empty_db, "bow-locker", "Bow locker duplicate")


def test_create_zone_vessel_layer(empty_db, tmp_path):
    """create_zone writes only to the DB — no files under shared/, no shared-layer artifact."""
    inv.create_zone(
        empty_db,
        "port-cockpit-locker",
        "Port cockpit locker",
        side="port",
        fore_aft="aft",
        area="cockpit",
    )
    # Zones are DB rows only — no disk writes outside data/
    shared_dir = tmp_path / "shared"
    if shared_dir.exists():
        shared_files = list(shared_dir.rglob("*"))
        assert shared_files == [], (
            f"create_zone must not write to shared/; found: {shared_files}"
        )


# ---------------------------------------------------------------------------
# ZONE-04: vertical_index — REAL type allows between-insert ordering
# ---------------------------------------------------------------------------


def test_vertical_index_between(empty_db):
    """vertical_index is REAL; a zone at 1.5 sorts between zones at 1.0 and 2.0."""
    # Assert the column type is REAL (not INTEGER)
    col_info = empty_db.execute("PRAGMA table_info(zones)").fetchall()
    vi_col = next((c for c in col_info if c["name"] == "vertical_index"), None)
    assert vi_col is not None, "zones table must have a vertical_index column"
    assert vi_col["type"].upper() == "REAL", (
        f"vertical_index must be REAL, got {vi_col['type']!r}"
    )

    inv.create_zone(empty_db, "lower-shelf", "Lower shelf", vertical_index=1.0)
    inv.create_zone(empty_db, "upper-shelf", "Upper shelf", vertical_index=2.0)
    inv.create_zone(empty_db, "mid-shelf", "Mid shelf", vertical_index=1.5)

    rows = empty_db.execute(
        "SELECT name FROM zones WHERE name IN ('lower-shelf','mid-shelf','upper-shelf') "
        "ORDER BY vertical_index"
    ).fetchall()
    names = [r["name"] for r in rows]
    assert names == ["lower-shelf", "mid-shelf", "upper-shelf"], (
        f"Expected ORDER BY vertical_index = [lower, mid, upper], got {names}"
    )


def test_vertical_desc_present(empty_db, fake_zone_ai):
    """use_ai=True yields a non-null vertical_desc; use_ai=False (default) leaves it NULL."""
    # With AI: vertical_desc should be filled by the stub
    zone_id_ai = inv.create_zone(
        empty_db, "port-locker-ai", "Port locker AI", use_ai=True
    )
    row_ai = empty_db.execute(
        "SELECT vertical_desc FROM zones WHERE id = ?", (zone_id_ai,)
    ).fetchone()
    assert row_ai["vertical_desc"] is not None, (
        "create_zone with use_ai=True must produce a non-null vertical_desc"
    )

    # Without AI (default): vertical_desc should be NULL; no AI call made
    zone_id_no_ai = inv.create_zone(
        empty_db, "stbd-locker-no-ai", "Stbd locker no AI"
        # use_ai defaults to False — no Ollama call
    )
    row_no_ai = empty_db.execute(
        "SELECT vertical_desc FROM zones WHERE id = ?", (zone_id_no_ai,)
    ).fetchone()
    assert row_no_ai["vertical_desc"] is None, (
        "create_zone without use_ai=True must leave vertical_desc NULL"
    )


# ---------------------------------------------------------------------------
# ZONE-05: sub-slot grid — create_sub_slots inserts correct count + UNIQUE
# ---------------------------------------------------------------------------


def test_sub_slot_grid(empty_db):
    """create_sub_slots(3, 2) inserts 6 rows; UNIQUE(zone_id,row,col) holds; no-grid zone has 0 rows."""
    zone_id = inv.create_zone(empty_db, "saloon-starboard", "Saloon starboard")
    inserted = inv.create_sub_slots(empty_db, zone_id, rows=3, cols=2)
    assert inserted == 6, f"Expected 6 sub-slot rows, got {inserted}"

    count = empty_db.execute(
        "SELECT COUNT(*) FROM zone_sub_slots WHERE zone_id = ?", (zone_id,)
    ).fetchone()[0]
    assert count == 6, f"Expected 6 zone_sub_slots rows, got {count}"

    # UNIQUE constraint: re-inserting same grid slot raises IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        empty_db.execute(
            "INSERT INTO zone_sub_slots (zone_id, row_num, col_num) VALUES (?, ?, ?)",
            (zone_id, 1, 1),
        )

    # No-grid zone: zero sub-slot rows
    zone_no_grid = inv.create_zone(empty_db, "open-shelf", "Open shelf")
    count_no_grid = empty_db.execute(
        "SELECT COUNT(*) FROM zone_sub_slots WHERE zone_id = ?", (zone_no_grid,)
    ).fetchone()[0]
    assert count_no_grid == 0, (
        f"Zone with no sub-slots should have 0 zone_sub_slots rows, got {count_no_grid}"
    )


# ---------------------------------------------------------------------------
# ZONE-06: seed migration — at least 20 zones seeded; anchor-locker present
# ---------------------------------------------------------------------------


def test_seed_zones_present(empty_db):
    """Fresh-migrated DB has >= 20 seeded zones; 'anchor-locker' has a non-null vertical_desc."""
    count = empty_db.execute("SELECT COUNT(*) FROM zones").fetchone()[0]
    assert count >= 20, f"Expected >= 20 seeded zones, got {count}"

    anchor = empty_db.execute(
        "SELECT vertical_desc FROM zones WHERE name = 'anchor-locker'"
    ).fetchone()
    assert anchor is not None, "Expected 'anchor-locker' zone in seed data"
    assert anchor["vertical_desc"] is not None, (
        "anchor-locker must have a non-null vertical_desc in seed data"
    )
