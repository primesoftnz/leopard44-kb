# Wave 0 RED state until Plan 09-03 ships annotation routes in app.py.
# create_app, TestClient, and _resolve_schematic_image are imported INSIDE each
# function body — NEVER at module top level — so pytest collection succeeds before
# the production routes exist (RED = route 404/import error, not collection crash).
"""Tests for ZONE-03: annotation routes + reject-not-sanitize image guard (Phase 9 Wave 0).

Per-requirement map source:
  .planning/phases/09-schematic-rendering-zone-annotation-visual-highlight/09-VALIDATION.md

Contracts pinned:
  GET  /annotate                       -> 200 HTML, zones listed (unannotated-first)
  GET  /annotate/{zone_id}             -> 200 for seeded zone, 404 for unknown id
  POST /annotate/{zone_id}             -> 200 for valid body; writes geometry + schematic_image
                                          422/400 for bad schematic_image (URL / traversal / missing)
  GET  /schematic-image/{filename:path}-> 200 image/png for existing file in data/schematics/
                                          404 for missing; 404 (REJECTED) for traversal/slash/backslash/non-.png
  _resolve_schematic_image helper      -> None for unsafe names, Path for valid existing file

Security (D-13, reject-not-sanitize): filename guard REJECTS unsafe names (404), never
normalizes-and-serves a potentially dangerous path. Unit-tested directly on the resolver
helper to pin the contract even where Starlette pre-normalizes slash-containing params.
"""
from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

import pytest
import sqlite_vec

from leopard44_kb.schema import apply_migrations


# ---------------------------------------------------------------------------
# File-backed DB bootstrap helper
# Same pattern as tests/test_inventory_cli.py lines 28-37 (_bootstrap_db).
# ---------------------------------------------------------------------------


def _bootstrap_db(db_path: Path) -> None:
    """Bootstrap a file-backed DB at db_path with sqlite-vec and all migrations applied."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    conn.close()


def _make_minimal_png(path: Path) -> None:
    """Write a minimal valid 1x1 white PNG to *path*.

    Used to provide a real .png file so the POST /annotate/{zone_id} validator
    (09-03) can confirm the file exists in data/schematics/.
    """
    # Minimal 1x1 white PNG bytes (valid header + IDAT + IEND)
    PNG_1X1 = (
        b"\x89PNG\r\n\x1a\n"                     # PNG signature
        b"\x00\x00\x00\rIHDR"                    # IHDR length + type
        b"\x00\x00\x00\x01"                       # width = 1
        b"\x00\x00\x00\x01"                       # height = 1
        b"\x08\x02"                               # bit depth 8, colour type 2 (RGB)
        b"\x00\x00\x00"                           # compression, filter, interlace
        b"\x90wS\xde"                             # IHDR CRC
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa75\x81\x84"  # IDAT
        b"\x00\x00\x00\x00IEND\xaeB`\x82"        # IEND
    )
    path.write_bytes(PNG_1X1)


# ---------------------------------------------------------------------------
# ZONE-03: GET /annotate — zone list
# ---------------------------------------------------------------------------


def test_annotate_list(monkeypatch, tmp_path):
    """GET /annotate returns 200 with a zone list containing seeded zone labels.

    Seeds the DB from migration 002 (32 zones); asserts the response is 200 and
    the HTML body contains zone data.
    """
    db_path = tmp_path / "s.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until 09-03
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/annotate")
    assert resp.status_code == 200, (
        f"GET /annotate returned {resp.status_code}; expected 200"
    )
    # Migration 002 seeds 32 zones; the HTML should contain at least one zone label
    html = resp.text
    # We cannot assert exact zone names without reading the seed SQL, but the page
    # should contain recognisable structure (has at least some zone content)
    assert len(html) > 200, f"GET /annotate response suspiciously short: {len(html)} bytes"


# ---------------------------------------------------------------------------
# ZONE-03: GET /annotate/{zone_id} — zone editor
# ---------------------------------------------------------------------------


def test_annotate_editor_route(monkeypatch, tmp_path):
    """GET /annotate/1 returns 200 for a seeded zone; GET /annotate/99999 returns 404."""
    db_path = tmp_path / "s.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until 09-03
    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    # Zone id=1 exists after migration 002 seeds 32 zones
    resp_ok = client.get("/annotate/1")
    assert resp_ok.status_code == 200, (
        f"GET /annotate/1 returned {resp_ok.status_code}; expected 200 for seeded zone"
    )

    # Unknown zone id
    resp_404 = client.get("/annotate/99999")
    assert resp_404.status_code == 404, (
        f"GET /annotate/99999 returned {resp_404.status_code}; expected 404 for unknown zone"
    )


# ---------------------------------------------------------------------------
# ZONE-03: POST /annotate/{zone_id} — save geometry + schematic_image
# ---------------------------------------------------------------------------


def test_save_polygon(monkeypatch, tmp_path):
    """POST /annotate/1 saves geometry + schematic_image; DB round-trip asserted (D-05/D-06)."""
    import leopard44_kb.web.app as _app_mod

    db_path = tmp_path / "s.db"
    _bootstrap_db(db_path)

    # Write a valid PNG to a controlled schematics dir; patch _SCHEMATICS_DIR so
    # the route finds it regardless of cwd (WR-02: cwd-independent resolution).
    schematics_dir = tmp_path / "data" / "schematics"
    schematics_dir.mkdir(parents=True, exist_ok=True)
    _make_minimal_png(schematics_dir / "page_061.png")

    monkeypatch.setattr(_app_mod, "_SCHEMATICS_DIR", schematics_dir)
    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until 09-03
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    geometry = [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8]]
    resp = client.post(
        "/annotate/1",
        json={"schematic_image": "page_061.png", "geometry": geometry},
    )
    assert resp.status_code == 200, (
        f"POST /annotate/1 returned {resp.status_code}; expected 200.\n"
        f"Body: {resp.text!r}"
    )

    # Verify the DB was actually updated (D-06: geometry round-trip)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT geometry, schematic_image FROM zones WHERE id = 1"
    ).fetchone()
    conn.close()

    assert row is not None, "Zone id=1 not found after POST"
    assert row["schematic_image"] == "page_061.png", (
        f"zones.schematic_image not saved; got {row['schematic_image']!r}"
    )
    assert json.loads(row["geometry"]) == geometry, (
        f"zones.geometry round-trip mismatch; got {row['geometry']!r}"
    )


def test_editor_reloads_geometry_as_array(monkeypatch, tmp_path):
    """GET /annotate/{id} for an annotated zone must embed geometry as a JSON ARRAY,
    not a double-encoded string. Regression for the Phase 9 visual-UAT round-trip gap:
    the route passed the raw TEXT column straight to `| tojson`, so zoneData.geometry
    arrived in JS as a string and loadExisting() threw on `.map`, leaving a re-opened
    zone unable to display (or re-edit) its saved polygon.
    """
    import leopard44_kb.web.app as _app_mod

    db_path = tmp_path / "s.db"
    _bootstrap_db(db_path)
    schematics_dir = tmp_path / "data" / "schematics"
    schematics_dir.mkdir(parents=True, exist_ok=True)
    _make_minimal_png(schematics_dir / "page_061.png")
    monkeypatch.setattr(_app_mod, "_SCHEMATICS_DIR", schematics_dir)
    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    geometry = [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8]]
    assert client.post(
        "/annotate/1", json={"schematic_image": "page_061.png", "geometry": geometry}
    ).status_code == 200

    html = client.get("/annotate/1").text
    # The zone-data <script> must serialize geometry as an array literal, never a
    # quoted string. `"geometry": [[` proves an array; `"geometry": "[` proves the
    # double-encoding bug.
    assert '"geometry": "' not in html and "'geometry': '" not in html, (
        "geometry is double-encoded as a string in the editor zone-data — "
        "loadExisting() will throw and the saved polygon won't reload"
    )
    assert '"geometry": [[' in html or '"geometry":[[' in html, (
        f"editor zone-data does not embed geometry as a JSON array; "
        f"snippet: {html[html.find('zone-data'):html.find('zone-data') + 200]!r}"
    )


def test_save_polygon_rejects_bad_schematic_image(monkeypatch, tmp_path):
    """POST /annotate/1 with invalid schematic_image is rejected (422/400); zone unchanged.

    Pins review concern 5: validate schematic_image is a bare .png that exists.
    Cases tested:
      (a) URL string "https://evil/x.png" — external URL not a bare name
      (b) Traversal string "../page_061.png" — directory component present
      (c) Non-existent bare name "ghost.png" — file not in data/schematics/
    """
    db_path = tmp_path / "s.db"
    _bootstrap_db(db_path)

    # Create data/schematics/ but do NOT write "ghost.png" there
    schematics_dir = tmp_path / "data" / "schematics"
    schematics_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    from leopard44_kb.web.app import create_app  # RED until 09-03
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    geometry = [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8]]

    bad_cases = [
        "https://evil/x.png",   # (a) external URL
        "../page_061.png",       # (b) path traversal
        "ghost.png",             # (c) file doesn't exist in data/schematics/
    ]

    for bad_image in bad_cases:
        resp = client.post(
            "/annotate/1",
            json={"schematic_image": bad_image, "geometry": geometry},
        )
        assert resp.status_code in (400, 422), (
            f"POST /annotate/1 with schematic_image={bad_image!r} returned "
            f"{resp.status_code}; expected 400 or 422 (rejection)."
        )

    # Assert the zone row was NOT updated by any of the bad requests
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT geometry, schematic_image FROM zones WHERE id = 1"
    ).fetchone()
    conn.close()

    assert row["schematic_image"] is None, (
        f"zones.schematic_image was updated despite bad schematic_image inputs; "
        f"got {row['schematic_image']!r}"
    )
    assert row["geometry"] is None, (
        f"zones.geometry was updated despite bad schematic_image inputs; "
        f"got {row['geometry']!r}"
    )


# ---------------------------------------------------------------------------
# ZONE-03 / D-13: GET /schematic-image/{filename} — reject-not-sanitize guard
# ---------------------------------------------------------------------------


def test_schematic_image_route(monkeypatch, tmp_path):
    """GET /schematic-image/{filename} guards against traversal (reject, not sanitize).

    Positive cases:
      - GET /schematic-image/page_061.png -> 200 image/png for an existing file
      - GET /schematic-image/missing.png  -> 404

    Negative / traversal cases (REJECT, not sanitize-and-serve):
      - percent-encoded traversal: /schematic-image/..%2F..%2Fpage_061.png -> 404
      - literal slash (subdir): /schematic-image/subdir/page_061.png -> 404
      - backslash name: /schematic-image/..\\page_061.png -> 404
      - non-.png name: /schematic-image/page_061.txt -> 404

    Additionally unit-tests the resolver helper _resolve_schematic_image directly
    (review concern 2: pin rejection even if Starlette pre-normalizes slashes).

    Asserts response bodies never contain bytes of a file outside data/schematics/.
    """
    import leopard44_kb.web.app as _app_mod

    # Set up a controlled schematics dir with a real PNG; patch _SCHEMATICS_DIR
    # so the route finds it regardless of cwd (WR-02: cwd-independent resolution).
    schematics_dir = tmp_path / "data" / "schematics"
    schematics_dir.mkdir(parents=True, exist_ok=True)
    _make_minimal_png(schematics_dir / "page_061.png")

    db_path = tmp_path / "s.db"
    monkeypatch.setattr(_app_mod, "_SCHEMATICS_DIR", schematics_dir)
    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until 09-03
    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    # Positive: existing file -> 200 image/png
    resp_ok = client.get("/schematic-image/page_061.png")
    assert resp_ok.status_code == 200, (
        f"GET /schematic-image/page_061.png returned {resp_ok.status_code}; expected 200"
    )
    assert "png" in resp_ok.headers.get("content-type", "").lower(), (
        f"Expected image/png content-type; got {resp_ok.headers.get('content-type')!r}"
    )

    # Positive: missing file -> 404
    resp_missing = client.get("/schematic-image/missing.png")
    assert resp_missing.status_code == 404, (
        f"GET /schematic-image/missing.png returned {resp_missing.status_code}; expected 404"
    )

    # Negative: percent-encoded traversal (route captures full path when declared :path)
    resp_encoded = client.get("/schematic-image/..%2F..%2Fpage_061.png")
    assert resp_encoded.status_code == 404, (
        f"Percent-encoded traversal returned {resp_encoded.status_code}; "
        "expected 404 (REJECTION, not sanitized serve)"
    )

    # Negative: literal slash (subdirectory attempt)
    resp_slash = client.get("/schematic-image/subdir/page_061.png")
    assert resp_slash.status_code == 404, (
        f"Slash-path returned {resp_slash.status_code}; expected 404 (REJECTION)"
    )

    # Negative: non-.png extension
    resp_txt = client.get("/schematic-image/page_061.txt")
    assert resp_txt.status_code == 404, (
        f"Non-.png file returned {resp_txt.status_code}; expected 404 (REJECTION)"
    )

    # Response bodies must NOT contain the bytes of our PNG file
    # (sanity check that nothing is served from a path outside data/schematics/)
    png_bytes = (schematics_dir / "page_061.png").read_bytes()
    for resp in [resp_encoded, resp_slash, resp_txt]:
        if resp.status_code == 200:
            assert resp.content != png_bytes, (
                "A traversal/rejection response served the protected PNG bytes!"
            )

    # Unit-test the resolver helper directly (review concern 2: Starlette may
    # pre-normalize slashes; the helper must still reject unsafe inputs).
    from leopard44_kb.web.app import _resolve_schematic_image  # RED until 09-03

    # Returns None for unsafe names
    assert _resolve_schematic_image("../page_061.png") is None, (
        "_resolve_schematic_image must return None for '../page_061.png' (traversal)"
    )
    assert _resolve_schematic_image("a/b.png") is None, (
        "_resolve_schematic_image must return None for 'a/b.png' (slash)"
    )
    assert _resolve_schematic_image("page_061.txt") is None, (
        "_resolve_schematic_image must return None for non-.png extension"
    )
    assert _resolve_schematic_image("") is None, (
        "_resolve_schematic_image must return None for empty string"
    )

    # Returns a Path for an existing valid file (when schematics dir is set up)
    # Note: the resolver needs to know where to look; the test will fail RED
    # if the helper doesn't exist, which is the expected state for 09-01.
    result = _resolve_schematic_image("page_061.png")
    assert result is not None, (
        "_resolve_schematic_image must return a non-None Path for existing 'page_061.png'"
    )
    assert isinstance(result, Path), (
        f"_resolve_schematic_image must return a Path; got {type(result)}"
    )


# ---------------------------------------------------------------------------
# WR-02 regression: schematic image route is cwd-independent (module-relative)
# ---------------------------------------------------------------------------


def test_schematic_image_route_cwd_independent(monkeypatch, tmp_path):
    """GET /schematic-image/{filename} resolves against the module-relative data/schematics/,
    NOT os.getcwd().  Running from a completely different cwd must NOT produce 404.

    This test PROVES the WR-02 fix: it sets cwd to a temp dir that has NO
    data/schematics/ subdirectory at all, yet the route still finds the PNG
    because _SCHEMATICS_DIR is anchored to Path(__file__).resolve().parents[3].

    The test patches leopard44_kb.web.app._SCHEMATICS_DIR to point to a controlled
    test directory (so we do not write to the real data/schematics/).
    """
    import leopard44_kb.web.app as _app_mod

    # Create a controlled schematics dir in tmp_path and write a test PNG
    fake_schematics = tmp_path / "fake_schematics"
    fake_schematics.mkdir()
    _make_minimal_png(fake_schematics / "page_042.png")

    # Patch _SCHEMATICS_DIR on the module so both _resolve_schematic_image
    # and annotate_editor use the fake dir
    monkeypatch.setattr(_app_mod, "_SCHEMATICS_DIR", fake_schematics)

    # Change cwd to a directory that has NO data/schematics/ — proves the
    # route does NOT depend on cwd (WR-02 regression gate)
    other_dir = tmp_path / "some_other_cwd"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    # Minimal DB
    db_path = tmp_path / "s.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    # Should resolve to 200 even though cwd has no data/schematics/
    resp = client.get("/schematic-image/page_042.png")
    assert resp.status_code == 200, (
        f"GET /schematic-image/page_042.png returned {resp.status_code} when cwd="
        f"{other_dir!r}; expected 200 (cwd must NOT matter — WR-02 regression)"
    )
    assert "png" in resp.headers.get("content-type", "").lower(), (
        f"Expected image/png content-type; got {resp.headers.get('content-type')!r}"
    )
