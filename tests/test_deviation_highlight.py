"""RED tests for DEV-02 / VIS-01: blue deviation zone_highlight discriminator,
CSS class, and deviation-precedence over inventory on a shared zone.

Per-requirement verification map source:
  .planning/phases/11-factory-deviation-log/11-VALIDATION.md

Nyquist discipline:
- leopard44_kb.deviation is NEVER imported at module top (body-import only where needed).
- create_app is NEVER imported at module top (body-import inside each test).
- All tests collect cleanly (zero ERROR collecting lines).
- Tests are RED until 11-03 adds the kind="deviation" discriminator + CSS class +
  deviation-precedence to the query SSE pipeline.
"""
from __future__ import annotations

import json
import sqlite3
import struct

import pytest
import sqlite_vec

from leopard44_kb.schema import apply_migrations


# ---------------------------------------------------------------------------
# Module-level SSE parsing helper (no leopard44_kb.web import — collection-safe).
# Identical pattern to test_web_query.py._parse_sse_events.
# ---------------------------------------------------------------------------


def _parse_sse_events(client, method: str, url: str, **kwargs) -> list[tuple[str, str]]:
    """Stream a request and return ordered list of (event_name, data) tuples."""
    events: list[tuple[str, str]] = []
    current_event = "message"
    pending_data: list[str] = []
    with client.stream(method, url, **kwargs) as resp:
        for line in resp.iter_lines():
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: "):
                pending_data.append(line[6:])
            elif line == "":
                if pending_data:
                    events.append((current_event, "\n".join(pending_data)))
                    pending_data = []
                    current_event = "message"
    return events


def _bootstrap_highlight_db(db_path) -> None:
    """Bootstrap a file-backed test DB for zone_highlight tests."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# (a) zone_highlight event carries kind="deviation" discriminator
# ---------------------------------------------------------------------------


def test_zone_highlight_kind_deviation(monkeypatch, fake_embedder, tmp_path):
    """zone_highlight SSE event for a deviation chunk carries kind="deviation".

    Seeding:
      - A zone with geometry set
      - A deviation row referencing that zone
      - A deviation chunk with metadata.deviation_id pointing to that deviation

    Asserts a zone_highlight event with JSON payload containing kind="deviation".
    Goes GREEN when 11-03 adds the kind discriminator to the zone_highlight emission.
    """
    db_path = tmp_path / "test_dev_highlight.db"
    _bootstrap_highlight_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")

    # Set geometry on zone id=1 (32 seed zones from 002_inventory.sql)
    geometry_json = "[[0.1,0.2],[0.5,0.2],[0.5,0.8],[0.1,0.8]]"
    conn.execute(
        "UPDATE zones SET geometry = ?, schematic_image = 'page_001.png' WHERE id = 1",
        (geometry_json,),
    )

    # Insert a deviation row referencing zone id=1
    conn.execute(
        "INSERT INTO deviations(id, component, zone_id) VALUES (1, 'windlass', 1)"
    )

    # Seed a deviation chunk with metadata.deviation_id = 1
    conn.execute(
        "INSERT INTO sources(id, layer, source_type, path, content_hash, title) "
        "VALUES (200, 'vessel', 'deviation', 'data/deviations/DEV-1.md', 'hd1', 'Windlass deviation')"
    )
    conn.execute(
        "INSERT INTO chunks(id, source_id, layer, ordinal, section_path, page_start, page_end, "
        "content, content_hash, anchor_key, embedding_model, embedding_model_version, metadata) "
        "VALUES (200, 200, 'vessel', 0, 'Deviations', 0, 0, "
        "'windlass replaced with Maxwell 1000W instead of factory Muir 1200W', "
        "'hdc1', 'akdev1', 'm', 'v', '{\"deviation_id\": 1}')"
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id, layer, source_id, embedding_model, is_active, embedding) "
        "VALUES (200, 'vessel', 200, 'm', 1, ?)",
        (struct.pack("384f", *([0.1] * 384)),),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until 11-03
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "windlass Maxwell", "layer": "vessel"},
    )

    highlight_events = [(name, data) for name, data in events if name == "zone_highlight"]
    assert len(highlight_events) >= 1, (
        f"Expected at least one zone_highlight event; got events: {[n for n,_ in events]!r}"
    )

    # At least one zone_highlight for the deviation zone must have kind="deviation"
    found_deviation_kind = False
    for _, raw_data in highlight_events:
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            continue
        if payload.get("kind") == "deviation":
            found_deviation_kind = True
            break

    assert found_deviation_kind, (
        f"Expected a zone_highlight with kind='deviation'; got payloads: "
        f"{[json.loads(d) for _, d in highlight_events]!r}"
    )


# ---------------------------------------------------------------------------
# (b) app.css contains .zone-highlight--deviation CSS selector
# ---------------------------------------------------------------------------


def test_css_contains_deviation_highlight_class():
    """app.css contains the literal '.zone-highlight--deviation' selector.

    Goes GREEN when 11-03 adds the blue deviation CSS class.
    """
    from pathlib import Path

    css_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "leopard44_kb" / "web" / "static" / "app.css"
    )
    assert css_path.exists(), f"app.css not found at {css_path}"
    css_content = css_path.read_text(encoding="utf-8")
    assert ".zone-highlight--deviation" in css_content, (
        f"Expected '.zone-highlight--deviation' selector in app.css; "
        f"did not find it. CSS file has {len(css_content)} chars."
    )


# ---------------------------------------------------------------------------
# (c) app.js applies .zone-highlight--deviation class when kind === 'deviation'
# ---------------------------------------------------------------------------


def test_js_applies_deviation_class():
    """app.js source contains the string 'zone-highlight--deviation'.

    Goes GREEN when 11-03 updates appendZoneHighlight to branch on kind.
    """
    from pathlib import Path

    js_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "leopard44_kb" / "web" / "static" / "app.js"
    )
    assert js_path.exists(), f"app.js not found at {js_path}"
    js_content = js_path.read_text(encoding="utf-8")
    assert "zone-highlight--deviation" in js_content, (
        f"Expected 'zone-highlight--deviation' string in app.js; "
        f"did not find it. JS file has {len(js_content)} chars."
    )


# ---------------------------------------------------------------------------
# (d) DEVIATION-PRECEDENCE: when both an inventory item AND a deviation map
#     to the same zone, the zone_highlight carries kind="deviation" (blue wins)
# ---------------------------------------------------------------------------


def test_deviation_precedence_over_inventory_on_shared_zone(
    monkeypatch, fake_embedder, tmp_path
):
    """When both an inventory item AND a deviation resolve to the same zone,
    the zone_highlight for that zone carries kind="deviation" (blue wins tie-break).

    Seeding:
      - Zone id=1 with geometry
      - An inventory item chunk with metadata.item_id=1 (item in zone 1)
      - A deviation chunk with metadata.deviation_id=1 (deviation in zone 1)
      Both chunks contain the same query term so both are retrieved.

    Asserts:
      - Exactly one zone_highlight for zone 1
      - That zone_highlight carries kind="deviation" (deviation wins)

    Goes GREEN when 11-03 implements deviation-precedence in the zone_highlight logic.
    """
    db_path = tmp_path / "test_precedence.db"
    _bootstrap_highlight_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")

    # Set geometry on zone id=1
    geometry_json = "[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]"
    conn.execute(
        "UPDATE zones SET geometry = ?, schematic_image = 'page_001.png' WHERE id = 1",
        (geometry_json,),
    )

    # Insert an inventory item in zone 1
    conn.execute(
        "INSERT INTO items(id, name, category, current_zone_id) "
        "VALUES (50, 'Test item shared zone', 'spare', 1)"
    )

    # Insert a deviation in zone 1
    conn.execute(
        "INSERT INTO deviations(id, component, zone_id) VALUES (50, 'windlass', 1)"
    )

    # Seed inventory item chunk
    conn.execute(
        "INSERT INTO sources(id, layer, source_type, path, content_hash, title) "
        "VALUES (301, 'vessel', 'text', 'data/inventory/ITEM-50.md', 'hi50', 'Item 50')"
    )
    conn.execute(
        "INSERT INTO chunks(id, source_id, layer, ordinal, section_path, page_start, page_end, "
        "content, content_hash, anchor_key, embedding_model, embedding_model_version, metadata) "
        "VALUES (301, 301, 'vessel', 0, 'Items', 0, 0, "
        "'ZZSHAREDZONE anchor chain spare stored here', "
        "'hitem50', 'akitem50', 'm', 'v', '{\"item_id\": 50}')"
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id, layer, source_id, embedding_model, is_active, embedding) "
        "VALUES (301, 'vessel', 301, 'm', 1, ?)",
        (struct.pack("384f", *([0.1] * 384)),),
    )

    # Seed deviation chunk
    conn.execute(
        "INSERT INTO sources(id, layer, source_type, path, content_hash, title) "
        "VALUES (302, 'vessel', 'deviation', 'data/deviations/DEV-50.md', 'hd50', 'Windlass dev')"
    )
    conn.execute(
        "INSERT INTO chunks(id, source_id, layer, ordinal, section_path, page_start, page_end, "
        "content, content_hash, anchor_key, embedding_model, embedding_model_version, metadata) "
        "VALUES (302, 302, 'vessel', 0, 'Deviations', 0, 0, "
        "'ZZSHAREDZONE windlass factory deviation replacement Maxwell', "
        "'hdev50', 'akdev50', 'm', 'v', '{\"deviation_id\": 50}')"
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id, layer, source_id, embedding_model, is_active, embedding) "
        "VALUES (302, 'vessel', 302, 'm', 1, ?)",
        (struct.pack("384f", *([0.1] * 384)),),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until 11-03
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "ZZSHAREDZONE", "layer": "vessel"},
    )

    highlight_events = [(name, data) for name, data in events if name == "zone_highlight"]

    # There must be exactly one zone_highlight for zone 1 (de-duplication must fire)
    zone1_highlights = []
    for _, raw_data in highlight_events:
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            continue
        if payload.get("zone_id") == 1:
            zone1_highlights.append(payload)

    assert len(zone1_highlights) == 1, (
        f"Expected exactly one zone_highlight for zone_id=1; got {zone1_highlights!r}"
    )
    assert zone1_highlights[0].get("kind") == "deviation", (
        f"Expected kind='deviation' (blue wins tie-break); "
        f"got kind={zone1_highlights[0].get('kind')!r}"
    )
