# RED state until Waves 2 and 3 land. Import of leopard44_kb.inventory fails at
# collection time because inventory.py does not yet exist — a collection-time
# ModuleNotFoundError is the EXPECTED Wave-0 state, consistent with the Phase 1-5
# import-at-collection RED convention. The module-top import is deliberate.
"""Tests for INV-01/02/03/04 item model, item-as-chunk, location history (D-08),
and locate/find (D-10).

Per-requirement verification map source:
  .planning/phases/08-zone-taxonomy-inventory-core/08-VALIDATION.md
"""
from __future__ import annotations

import json
import sqlite3

import pytest

import leopard44_kb.inventory as inv  # collection-time ModuleNotFoundError = expected RED state


# ---------------------------------------------------------------------------
# INV-01: create_item — model + vessel-layer write + category constraint
# ---------------------------------------------------------------------------


def test_item_add(fake_embedder, tmp_path, seeded_zone_db):
    """create_item returns an int id; items row exists with correct category;
    ITEM-{id}.md is written under tmp_path/data/inventory."""
    conn = seeded_zone_db
    zone_row = conn.execute("SELECT id FROM zones WHERE name='saloon-port-locker'").fetchone()
    zone_id = zone_row["id"]

    item_id = inv.create_item(
        conn, "Scrabble", "toy", zone_id=zone_id, repo_root=tmp_path
    )
    assert isinstance(item_id, int), f"Expected int item_id, got {type(item_id)}"

    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    assert row is not None, "Expected an items row for the returned id"
    assert row["category"] == "toy", f"Expected category='toy', got {row['category']!r}"

    md_path = tmp_path / "data" / "inventory" / f"ITEM-{item_id}.md"
    assert md_path.exists(), f"Expected ITEM-{item_id}.md at {md_path}"


def test_item_vessel_layer(fake_embedder, tmp_path, empty_db):
    """Path-escape attempt raises ValueError AND leaves no orphan items row."""
    # create_item must call validate_path("vessel", ...) before writing
    # A path_escape scenario: assert the md file lands under tmp_path/data/inventory
    # and not elsewhere; verify no orphan row if validate_path raises
    item_id = inv.create_item(
        empty_db, "SafeItem", "tool", repo_root=tmp_path
    )
    md_path = tmp_path / "data" / "inventory" / f"ITEM-{item_id}.md"
    assert md_path.exists(), f"ITEM-{item_id}.md must land under data/inventory"
    # Verify it is NOT outside the expected path
    assert str(md_path).startswith(str(tmp_path)), (
        f"ITEM file must be under repo_root tmp_path, got {md_path}"
    )

    # Simulate path-escape by providing a manipulated repo_root whose
    # data/inventory resolves outside — validate_path("vessel") must raise ValueError
    # and leave no items row for that name
    before_count = empty_db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    import os
    from pathlib import Path
    # Use a path that, after join, could escape via os.path tricks.
    # validate_path in leopard44_kb.paths resolves symlinks and checks containment.
    escape_root = tmp_path / "escape_root"
    escape_root.mkdir(parents=True, exist_ok=True)
    # Create a symlink that points outside escape_root
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    escape_link = escape_root / "data"
    try:
        escape_link.symlink_to(outside_dir)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported on this platform")

    with pytest.raises(ValueError):
        inv.create_item(empty_db, "EscapeAttempt", "tool", repo_root=escape_root)

    after_count = empty_db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    assert after_count == before_count, (
        f"Path-escape attempt must not leave an orphan items row; "
        f"before={before_count}, after={after_count}"
    )


def test_item_category_check(empty_db, tmp_path):
    """create_item with an invalid category raises (IntegrityError or ValueError)."""
    with pytest.raises((sqlite3.IntegrityError, ValueError)):
        inv.create_item(empty_db, "Gizmo", "gizmo", repo_root=tmp_path)


# ---------------------------------------------------------------------------
# INV-02: category-appropriate metadata in chunk content
# ---------------------------------------------------------------------------


def test_spare_chunk_content(fake_embedder, tmp_path, empty_db):
    """spare item with part_number in metadata: chunk content contains part number."""
    item_id = inv.create_item(
        empty_db,
        "impeller",
        "spare",
        metadata={"part_number": "22-41016"},
        repo_root=tmp_path,
    )
    # Check either the _build_chunk_content result or the written .md file
    md_path = tmp_path / "data" / "inventory" / f"ITEM-{item_id}.md"
    if md_path.exists():
        content = md_path.read_text()
    else:
        # Fall back to checking chunk content via DB
        row = empty_db.execute(
            "SELECT c.content FROM chunks c "
            "JOIN sources s ON s.id = c.source_id "
            "JOIN items i ON i.chunk_source_id = s.id "
            "WHERE i.id = ?",
            (item_id,),
        ).fetchone()
        content = row["content"] if row else ""
    assert "22-41016" in content, (
        f"Expected part_number '22-41016' in chunk content, got: {content[:200]!r}"
    )


def test_provision_chunk_content(fake_embedder, tmp_path, empty_db):
    """provision item with best_before: chunk content contains the date string."""
    item_id = inv.create_item(
        empty_db,
        "Canned tomatoes",
        "provision",
        metadata={"best_before": "2027-06-01"},
        repo_root=tmp_path,
    )
    md_path = tmp_path / "data" / "inventory" / f"ITEM-{item_id}.md"
    if md_path.exists():
        content = md_path.read_text()
    else:
        row = empty_db.execute(
            "SELECT c.content FROM chunks c "
            "JOIN sources s ON s.id = c.source_id "
            "JOIN items i ON i.chunk_source_id = s.id "
            "WHERE i.id = ?",
            (item_id,),
        ).fetchone()
        content = row["content"] if row else ""
    assert "2027-06-01" in content, (
        f"Expected best_before '2027-06-01' in chunk content, got: {content[:200]!r}"
    )


def test_safety_chunk_content(fake_embedder, tmp_path, empty_db):
    """safety item with expiry: chunk content contains the expiry date."""
    item_id = inv.create_item(
        empty_db,
        "EPIRB",
        "safety",
        metadata={"expiry": "2026-11-01"},
        repo_root=tmp_path,
    )
    md_path = tmp_path / "data" / "inventory" / f"ITEM-{item_id}.md"
    if md_path.exists():
        content = md_path.read_text()
    else:
        row = empty_db.execute(
            "SELECT c.content FROM chunks c "
            "JOIN sources s ON s.id = c.source_id "
            "JOIN items i ON i.chunk_source_id = s.id "
            "WHERE i.id = ?",
            (item_id,),
        ).fetchone()
        content = row["content"] if row else ""
    assert "2026-11-01" in content, (
        f"Expected expiry '2026-11-01' in chunk content, got: {content[:200]!r}"
    )


# ---------------------------------------------------------------------------
# INV-03: item-as-chunk — chunk metadata, retrieval, delete no orphan
# ---------------------------------------------------------------------------


def test_chunk_metadata_fields(fake_embedder, tmp_path, seeded_zone_db):
    """After create_item with a zone, the chunk metadata JSON carries item_id and zone_id."""
    conn = seeded_zone_db
    zone_row = conn.execute("SELECT id FROM zones WHERE name='saloon-port-locker'").fetchone()
    zone_id = zone_row["id"]

    item_id = inv.create_item(
        conn, "Scrabble metadata test", "toy", zone_id=zone_id, repo_root=tmp_path
    )
    src_row = conn.execute(
        "SELECT chunk_source_id FROM items WHERE id = ?", (item_id,)
    ).fetchone()
    assert src_row is not None, "items row must exist after create_item"
    src_id = src_row["chunk_source_id"]
    assert src_id is not None, "chunk_source_id must be non-null after successful create_item"

    chunk_row = conn.execute(
        "SELECT metadata FROM chunks WHERE source_id = ?", (src_id,)
    ).fetchone()
    assert chunk_row is not None, "A chunk row must exist for the created item's source"
    meta = json.loads(chunk_row["metadata"])
    assert "item_id" in meta, f"chunk metadata must contain 'item_id', got {meta}"
    assert "zone_id" in meta, f"chunk metadata must contain 'zone_id', got {meta}"
    assert meta["item_id"] == item_id, (
        f"chunk metadata item_id {meta['item_id']} must match item_id {item_id}"
    )
    assert meta["zone_id"] == zone_id, (
        f"chunk metadata zone_id {meta['zone_id']} must match zone_id {zone_id}"
    )


def test_item_chunk_retrieval(fake_embedder, tmp_path, seeded_zone_db):
    """After create_item, retrieve() returns at least one chunk whose metadata carries item_id."""
    from leopard44_kb.retrieve import retrieve

    conn = seeded_zone_db
    zone_row = conn.execute("SELECT id FROM zones WHERE name='saloon-port-locker'").fetchone()
    zone_id = zone_row["id"]

    item_id = inv.create_item(
        conn, "Scrabble", "toy", zone_id=zone_id, repo_root=tmp_path
    )
    chunks, _below_floor = retrieve(conn, "Scrabble", layers=["vessel"], n=5)
    assert len(chunks) >= 1, "retrieve() must return at least one chunk for the created item"

    # At least one chunk must carry the item_id in metadata
    found_item = False
    for chunk in chunks:
        meta_str = chunk.get("metadata", "{}")
        meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
        if meta.get("item_id") == item_id:
            found_item = True
            break
    assert found_item, (
        f"No chunk returned by retrieve() carries item_id={item_id}; "
        f"metadata in chunks: {[c.get('metadata') for c in chunks]}"
    )


def test_delete_item_no_orphan(fake_embedder, tmp_path, seeded_zone_db):
    """delete_item: vec_chunks row gone after delete; ITEM-{id}.md file unlinked (no orphans)."""
    conn = seeded_zone_db
    zone_row = conn.execute("SELECT id FROM zones WHERE name='saloon-port-locker'").fetchone()
    zone_id = zone_row["id"]

    item_id = inv.create_item(
        conn, "Drill", "tool", zone_id=zone_id, repo_root=tmp_path
    )
    src_row = conn.execute(
        "SELECT chunk_source_id FROM items WHERE id = ?", (item_id,)
    ).fetchone()
    src_id = src_row["chunk_source_id"]

    # vec_chunks row must exist before delete
    vc_before = conn.execute(
        "SELECT 1 FROM vec_chunks WHERE source_id = ?", (src_id,)
    ).fetchone()
    assert vc_before is not None, "vec_chunks row must exist before delete_item"

    md_path = tmp_path / "data" / "inventory" / f"ITEM-{item_id}.md"

    inv.delete_item(conn, item_id, repo_root=tmp_path)

    # vec_chunks row must be gone (no orphan)
    vc_after = conn.execute(
        "SELECT 1 FROM vec_chunks WHERE source_id = ?", (src_id,)
    ).fetchone()
    assert vc_after is None, (
        f"vec_chunks row for source_id={src_id} must be deleted after delete_item"
    )

    # ITEM-{id}.md must be unlinked
    assert not md_path.exists(), (
        f"ITEM-{item_id}.md must be unlinked after delete_item; path={md_path}"
    )


def test_delete_item_null_chunk_source(fake_embedder, tmp_path, empty_db):
    """delete_item on an items row with chunk_source_id=NULL does not raise; items row gone."""
    # Simulate a prior partial-failure state: insert item with no chunk
    with empty_db:
        empty_db.execute(
            "INSERT INTO items (name, category, chunk_source_id) VALUES (?, ?, NULL)",
            ("BrokenItem", "tool"),
        )
    item_id = empty_db.execute(
        "SELECT id FROM items WHERE name='BrokenItem'"
    ).fetchone()["id"]

    # delete_item must not raise when chunk_source_id is NULL
    inv.delete_item(empty_db, item_id, repo_root=tmp_path)

    row = empty_db.execute("SELECT id FROM items WHERE id = ?", (item_id,)).fetchone()
    assert row is None, f"items row must be gone after delete_item; got {dict(row) if row else None}"


def test_create_item_embed_failure_no_orphan(monkeypatch, tmp_path, empty_db):
    """Embedding failure during create_item leaves no orphan: no items row, no ITEM-*.md file."""
    import leopard44_kb.ingest.embedder as emb

    # Stub embed_texts to raise (simulates Ollama down)
    monkeypatch.setattr(
        emb,
        "embed_texts",
        lambda texts, model: (_ for _ in ()).throw(RuntimeError("Ollama unavailable")),
    )
    monkeypatch.setattr(emb, "select_model", lambda: ("nomic-embed-text:v1.5", "v1.5"))

    with pytest.raises(Exception):
        inv.create_item(empty_db, "OrphanTest", "spare", repo_root=tmp_path)

    # No orphan items row
    orphan_row = empty_db.execute(
        "SELECT id FROM items WHERE name='OrphanTest'"
    ).fetchone()
    assert orphan_row is None, (
        "create_item must not leave an orphan items row on embed failure"
    )

    # No orphan ITEM-*.md file
    inv_dir = tmp_path / "data" / "inventory"
    if inv_dir.exists():
        md_files = list(inv_dir.glob("ITEM-*.md"))
        assert md_files == [], (
            f"create_item must not leave orphan ITEM-*.md files on embed failure; found: {md_files}"
        )


# ---------------------------------------------------------------------------
# D-08: update_item_location — history push + no-op deduplication
# ---------------------------------------------------------------------------


def test_update_pushes_history(fake_embedder, tmp_path, empty_db):
    """update_item_location from zone A to zone B: current_zone_id=B, history references A."""
    with empty_db:
        empty_db.execute(
            "INSERT INTO zones (name, label) VALUES ('zone-a', 'Zone A')"
        )
        empty_db.execute(
            "INSERT INTO zones (name, label) VALUES ('zone-b', 'Zone B')"
        )
    zone_a = empty_db.execute("SELECT id FROM zones WHERE name='zone-a'").fetchone()["id"]
    zone_b = empty_db.execute("SELECT id FROM zones WHERE name='zone-b'").fetchone()["id"]

    item_id = inv.create_item(
        empty_db, "Flashlight", "tool", zone_id=zone_a, repo_root=tmp_path
    )
    inv.update_item_location(empty_db, item_id, zone_b, repo_root=tmp_path)

    row = empty_db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    assert row["current_zone_id"] == zone_b, (
        f"current_zone_id must be zone_b ({zone_b}) after update, got {row['current_zone_id']}"
    )

    history = json.loads(row["location_history"])
    assert isinstance(history, list) and len(history) > 0, (
        f"location_history must be a non-empty list after update, got {row['location_history']!r}"
    )
    # Newest-first: first element references the previous zone (zone_a)
    assert history[0].get("zone_id") == zone_a, (
        f"First history entry must reference zone_a ({zone_a}), got {history[0]}"
    )


def test_update_location_noop(fake_embedder, tmp_path, seeded_zone_db):
    """update_item_location to same zone+sub_slot returns early; history NOT appended."""
    conn = seeded_zone_db
    zone_row = conn.execute("SELECT id FROM zones WHERE name='saloon-port-locker'").fetchone()
    zone_id = zone_row["id"]

    item_id = inv.create_item(
        conn, "Noop test item", "toy", zone_id=zone_id, repo_root=tmp_path
    )
    before_row = conn.execute("SELECT location_history FROM items WHERE id = ?", (item_id,)).fetchone()
    before_history = json.loads(before_row["location_history"])

    # Update to same zone (no-op)
    inv.update_item_location(conn, item_id, zone_id, repo_root=tmp_path)

    after_row = conn.execute("SELECT location_history FROM items WHERE id = ?", (item_id,)).fetchone()
    after_history = json.loads(after_row["location_history"])

    assert len(after_history) == len(before_history), (
        f"No-op update must not append to location_history; "
        f"before={len(before_history)}, after={len(after_history)}"
    )


# ---------------------------------------------------------------------------
# ZONE-05: sub-slot validation on items
# ---------------------------------------------------------------------------


def test_sub_slot_validation(fake_embedder, tmp_path, empty_db):
    """create_item sub_slot validation: valid slot ok; invalid slot raises ValueError;
    no-grid zone rejects non-NULL sub_slot."""
    with empty_db:
        empty_db.execute(
            "INSERT INTO zones (name, label) VALUES ('grid-zone', 'Grid zone')"
        )
        empty_db.execute(
            "INSERT INTO zones (name, label) VALUES ('flat-zone', 'Flat zone')"
        )
    grid_zone = empty_db.execute("SELECT id FROM zones WHERE name='grid-zone'").fetchone()["id"]
    flat_zone = empty_db.execute("SELECT id FROM zones WHERE name='flat-zone'").fetchone()["id"]

    # Create a 3x2 sub-slot grid
    inv.create_sub_slots(empty_db, grid_zone, rows=3, cols=2)

    # Valid sub_slot: (row=1, col=1) exists in the 3x2 grid
    item_id = inv.create_item(
        empty_db, "Slot item", "tool", zone_id=grid_zone,
        current_sub_slot={"row": 1, "col": 1}, repo_root=tmp_path
    )
    assert item_id is not None, "create_item with a valid sub_slot must succeed"

    # Invalid sub_slot: (row=5, col=5) not in the 3x2 grid
    with pytest.raises(ValueError):
        inv.create_item(
            empty_db, "Invalid slot item", "tool", zone_id=grid_zone,
            current_sub_slot={"row": 5, "col": 5}, repo_root=tmp_path
        )

    # No-grid zone: non-NULL sub_slot raises ValueError
    with pytest.raises(ValueError):
        inv.create_item(
            empty_db, "No-grid slot item", "tool", zone_id=flat_zone,
            current_sub_slot={"row": 1, "col": 1}, repo_root=tmp_path
        )


# ---------------------------------------------------------------------------
# INV-04 / D-10: find_item + locate_item — structured result
# ---------------------------------------------------------------------------


def test_find_item_exact(fake_embedder, tmp_path, seeded_zone_db):
    """find_item returns a list with the item enriched with zone label."""
    conn = seeded_zone_db
    zone_row = conn.execute("SELECT id FROM zones WHERE name='saloon-port-locker'").fetchone()
    zone_id = zone_row["id"]

    inv.create_item(conn, "Scrabble", "toy", zone_id=zone_id, repo_root=tmp_path)

    results = inv.find_item(conn, "scrabble")
    assert isinstance(results, list), f"find_item must return a list, got {type(results)}"
    assert len(results) >= 1, "find_item must return at least one result for 'scrabble'"

    item = results[0]
    assert "zone" in item or "zone_label" in item, (
        f"find_item result must be enriched with zone info, got keys: {list(item.keys())}"
    )


def test_locate_item_semantic(fake_embedder, tmp_path, seeded_zone_db):
    """locate_item falls through to retrieve() for non-exact query; resolves item_id from
    chunk metadata; returns structured location (not LLM text)."""
    conn = seeded_zone_db
    zone_row = conn.execute("SELECT id FROM zones WHERE name='saloon-port-locker'").fetchone()
    zone_id = zone_row["id"]

    inv.create_item(
        conn, "Scrabble", "toy", zone_id=zone_id,
        aliases=["board game", "word game"], repo_root=tmp_path
    )
    result = inv.locate_item(conn, "board game")

    assert isinstance(result, dict), f"locate_item must return a dict, got {type(result)}"
    assert "found" in result, f"locate_item result must have 'found' key, got {list(result.keys())}"
    assert result["found"] is True, (
        f"locate_item must find the item for 'board game', got found={result['found']}"
    )
    assert "items" in result, f"locate_item result must have 'items' key, got {list(result.keys())}"
    assert len(result["items"]) >= 1, "locate_item must return at least one item"

    # The location must come from the structured item record, not LLM text
    item_result = result["items"][0]
    assert "zone" in item_result, (
        f"locate_item item result must carry 'zone' from the item record, got keys: {list(item_result.keys())}"
    )


def test_locate_structured_result(fake_embedder, tmp_path, seeded_zone_db):
    """locate_item for an exact match returns structured current location (zone label + sub-slot)
    read from the item record — not free LLM text."""
    conn = seeded_zone_db
    zone_row = conn.execute("SELECT id FROM zones WHERE name='saloon-port-locker'").fetchone()
    zone_id = zone_row["id"]

    inv.create_item(conn, "Compass", "tool", zone_id=zone_id, repo_root=tmp_path)
    result = inv.locate_item(conn, "Compass")

    assert result.get("found") is True, (
        f"locate_item must find 'Compass', got found={result.get('found')}"
    )
    item_result = result["items"][0]
    zone_info = item_result.get("zone")
    assert zone_info is not None, (
        "locate_item result item must carry structured 'zone' from the item record"
    )
    # zone must be a structured value (dict or label string), not an empty result
    assert zone_info, f"zone info must be truthy (non-empty), got {zone_info!r}"


def test_locate_multiple_candidates(fake_embedder, tmp_path, empty_db):
    """locate_item returns multiple candidates when the query matches more than one item."""
    with empty_db:
        empty_db.execute(
            "INSERT INTO zones (name, label) VALUES ('fore-cabin', 'Fore cabin')"
        )
        empty_db.execute(
            "INSERT INTO zones (name, label) VALUES ('aft-cabin', 'Aft cabin')"
        )
    zone_fore = empty_db.execute("SELECT id FROM zones WHERE name='fore-cabin'").fetchone()["id"]
    zone_aft = empty_db.execute("SELECT id FROM zones WHERE name='aft-cabin'").fetchone()["id"]

    inv.create_item(empty_db, "Scrabble", "toy", zone_id=zone_fore, repo_root=tmp_path)
    inv.create_item(empty_db, "Scrabble Junior", "toy", zone_id=zone_aft, repo_root=tmp_path)

    result = inv.locate_item(empty_db, "Scrabble")
    assert isinstance(result.get("items"), list), "locate_item must return an 'items' list"
    assert len(result["items"]) > 1, (
        f"locate_item must return multiple candidates for ambiguous 'Scrabble' query, "
        f"got {len(result['items'])} candidate(s)"
    )
