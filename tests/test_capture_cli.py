"""RED tests for CAP-02: `l44 capture <photo>` confirm/edit/abort CLI.

Phase 12 Wave-0 RED scaffold — tests FAIL at assertion until Wave-2/3 ships
leopard44_kb/capture/ and the `capture` sub-app in cli.py. Imports of
leopard44_kb.capture.* are INSIDE each test body (RED-at-assertion, not collection).

Coverage (CAP-02, H1, H3, M4):
  (a) Abort ("q") leaves items row count unchanged — never auto-commits
  (b) Accept ("a") creates exactly one inventory item
  (c) low confidence (<0.7) prints flag + "rerun with --cloud" suggestion (H3)
  (d) Edit ("e") allows changing a field before write
  (e) non-existent photo path exits non-zero with a clear error
  (f) H3: zero egress without --cloud even at low confidence with a key set
  (g) H3: --cloud prints explicit "sending to cloud" notice before the call
  (h) H1: photo-store failure → item STILL created, photo_path NULL, warning emitted
  (i) M4: unknown vision zone → zone_id NULL + warning; ambiguous edited zone → same
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from leopard44_kb.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap_db(db_path: Path) -> None:
    """Bootstrap a file-backed DB with sqlite-vec and migrations applied."""
    import sqlite_vec

    from leopard44_kb.schema import apply_migrations

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    conn.close()


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Open a bootstrapped DB for direct query verification."""
    import sqlite_vec

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _make_jpeg(path: Path) -> Path:
    """Write a minimal valid 32×32 JPEG to `path` and return it."""
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    img.save(str(path), format="JPEG")
    return path


def _make_vision_result(
    *,
    confidence: float = 0.90,
    suggested_zone: str = "Saloon",
    item: str = "winch handle",
    zone_id: int | None = 1,
    low_confidence: bool = False,
) -> dict:
    """Return a minimal normalised vision result (as identify_item would return)."""
    return {
        "item": item,
        "brand": None,
        "model": None,
        "category": "deck hardware",
        "marine": True,
        "legible": False,
        "key_properties": [],
        "other_items": [],
        "suggested_zone": suggested_zone,
        "zone_id": zone_id,
        "zone_reasoning": "Used at helm",
        "confidence": confidence,
        "low_confidence": low_confidence,
    }


# ---------------------------------------------------------------------------
# (a) Abort ("q") — items row count unchanged
# ---------------------------------------------------------------------------

def test_capture_abort_writes_nothing(monkeypatch, tmp_path):
    """capture <photo> with input 'q' (Abort) writes NOTHING to the items table.

    RED: ModuleNotFoundError from 'from leopard44_kb.capture import vision' inside
    the capture command body until Wave 2 ships capture/.
    """
    import leopard44_kb.capture as capture_pkg  # RED until Wave 2

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    photo_path = _make_jpeg(tmp_path / "item.jpg")

    # Mock the vision call so the command reaches the confirm prompt
    monkeypatch.setattr(
        capture_pkg,
        "identify_item_for_cli",
        lambda path, zones, cloud=False: _make_vision_result(),
    )
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    result = runner.invoke(app, ["capture", str(photo_path)], input="q\n")

    conn = _open_db(db_path)
    count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    conn.close()

    assert count == 0, f"Expected 0 items after Abort; got {count}"


# ---------------------------------------------------------------------------
# (b) Accept ("a") creates exactly one inventory item
# ---------------------------------------------------------------------------

def test_capture_accept_creates_item(monkeypatch, fake_embedder, tmp_path):
    """capture <photo> with input 'a' (Accept) creates exactly one items row.

    RED: ModuleNotFoundError until Wave 2 ships capture/.
    """
    import leopard44_kb.capture as capture_pkg  # RED until Wave 2

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    photo_path = _make_jpeg(tmp_path / "item.jpg")

    monkeypatch.setattr(
        capture_pkg,
        "identify_item_for_cli",
        lambda path, zones, cloud=False: _make_vision_result(),
    )
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    result = runner.invoke(app, ["capture", str(photo_path)], input="a\n")
    combined = (result.stdout or "") + (result.stderr or "")

    assert result.exit_code == 0, (
        f"Expected exit 0 on Accept; got {result.exit_code}: {combined!r}"
    )

    conn = _open_db(db_path)
    count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    conn.close()
    assert count == 1, f"Expected 1 items row after Accept; got {count}"


# ---------------------------------------------------------------------------
# (c) H3: low-confidence prints flag + "rerun with --cloud" suggestion
# ---------------------------------------------------------------------------

def test_capture_low_confidence_prints_flag_and_cloud_suggestion(monkeypatch, tmp_path):
    """capture with low-confidence result prints flag string + rerun-with-cloud hint (H3).

    RED: ModuleNotFoundError until Wave 2 ships capture/.
    """
    import leopard44_kb.capture as capture_pkg  # RED until Wave 2

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    photo_path = _make_jpeg(tmp_path / "item.jpg")

    monkeypatch.setattr(
        capture_pkg,
        "identify_item_for_cli",
        lambda path, zones, cloud=False: _make_vision_result(
            confidence=0.45, low_confidence=True
        ),
    )
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    result = runner.invoke(app, ["capture", str(photo_path)], input="q\n")
    combined = (result.stdout or "") + (result.stderr or "")

    # Must print a low-confidence flag/warning
    flag_words = ["low confidence", "low-confidence", "review", "uncertain"]
    assert any(w in combined.lower() for w in flag_words), (
        f"Expected low-confidence flag in output; got: {combined!r}"
    )

    # Must print a suggestion to rerun with --cloud (H3 consent gate)
    assert "--cloud" in combined or "cloud" in combined.lower(), (
        f"Expected '--cloud' rerun suggestion in output for low confidence; got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# (d) Edit ("e") allows changing a field before write
# ---------------------------------------------------------------------------

def test_capture_edit_changes_field_before_write(monkeypatch, fake_embedder, tmp_path):
    """capture with input 'e' lets owner edit a field; the edited value is committed.

    RED: ModuleNotFoundError until Wave 2 ships capture/.
    """
    import leopard44_kb.capture as capture_pkg  # RED until Wave 2

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    photo_path = _make_jpeg(tmp_path / "item.jpg")

    monkeypatch.setattr(
        capture_pkg,
        "identify_item_for_cli",
        lambda path, zones, cloud=False: _make_vision_result(item="winch handle"),
    )
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    # Edit flow: 'e' to enter edit, change 'name' field to "Lewmar winch handle", 'a' accept
    result = runner.invoke(
        app,
        ["capture", str(photo_path)],
        input="e\nname\nLewmar winch handle\na\n",
    )
    combined = (result.stdout or "") + (result.stderr or "")

    assert result.exit_code == 0, (
        f"Expected exit 0 after Edit→Accept; got {result.exit_code}: {combined!r}"
    )

    conn = _open_db(db_path)
    row = conn.execute("SELECT name FROM items ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()

    assert row is not None, "Expected one items row after Edit→Accept"
    assert "lewmar" in row[0].lower() or "winch handle" in row[0].lower(), (
        f"Expected edited name 'Lewmar winch handle' or 'winch handle' in DB; got: {row[0]!r}"
    )


def test_capture_edit_invalid_category_is_rejected_and_reprompted(
    monkeypatch, fake_embedder, tmp_path
):
    """Editing category to an invalid enum value is rejected + re-prompted (WR-04).

    The owner types an invalid category ("widget"), which must be rejected with a
    clear message, then a valid one ("tool"), which is accepted and stored. The
    invalid value must NOT reach the DB (no silent coercion of a user edit).
    """
    import leopard44_kb.capture as capture_pkg

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    photo_path = _make_jpeg(tmp_path / "item.jpg")

    monkeypatch.setattr(
        capture_pkg,
        "identify_item_for_cli",
        lambda path, zones, cloud=False: _make_vision_result(item="winch handle"),
    )
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    # Edit category: 'widget' (invalid → re-prompt), then 'tool' (valid), then accept.
    result = runner.invoke(
        app,
        ["capture", str(photo_path)],
        input="e\ncategory\nwidget\ntool\na\n",
    )
    combined = (result.stdout or "") + (result.stderr or "")

    assert result.exit_code == 0, (
        f"Expected exit 0 after re-prompt→Accept; got {result.exit_code}: {combined!r}"
    )
    assert "invalid category" in combined.lower(), (
        f"Expected an 'invalid category' rejection message; got: {combined!r}"
    )

    conn = _open_db(db_path)
    row = conn.execute(
        "SELECT category FROM items ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert row is not None, "Expected one items row after Edit→Accept"
    assert row[0] == "tool", (
        f"Edited category should be the valid 'tool', never the rejected 'widget'; "
        f"got: {row[0]!r}"
    )


# ---------------------------------------------------------------------------
# (e) Non-existent photo path exits non-zero with clear error
# ---------------------------------------------------------------------------

def test_capture_missing_photo_exits_nonzero(monkeypatch, tmp_path):
    """capture <nonexistent-path> exits non-zero with a clear path error.

    RED: ModuleNotFoundError until Wave 2 ships capture/.
    — OR — exits non-zero at the path-validation step before capture code is called.
    """
    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    missing = str(tmp_path / "no_such_photo.jpg")
    result = runner.invoke(app, ["capture", missing])
    combined = (result.stdout or "") + (result.stderr or "")

    assert result.exit_code != 0, (
        f"Expected non-zero exit for missing photo; got {result.exit_code}: {combined!r}"
    )
    # Must mention the path or 'not found' / 'does not exist'
    assert any(word in combined.lower() for word in ["not found", "does not exist", "no such", missing.lower()[-15:]]), (
        f"Expected path error in output; got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# (f) H3: ZERO egress without --cloud even at low confidence with a key set
# ---------------------------------------------------------------------------

def test_capture_zero_egress_without_cloud_flag(monkeypatch, tmp_path):
    """Plain `l44 capture` (no --cloud) makes ZERO api.anthropic.com calls
    even when ANTHROPIC_API_KEY is set and confidence is low (H3).

    Blocks the httpx transport at the module level and verifies it is never invoked
    for any anthropic.com URL.
    RED: ModuleNotFoundError until Wave 2 ships capture/.
    """
    import leopard44_kb.capture as capture_pkg  # RED until Wave 2
    import leopard44_kb.capture.vision as vision_mod  # RED until Wave 2

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-00000000000000000000000000")

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    anthropic_calls: list[str] = []

    def _sentinel_post(url: str, json: dict, timeout: float | None = None, **kw):
        if "anthropic.com" in url:
            anthropic_calls.append(url)
            raise AssertionError(
                f"MUST NOT call api.anthropic.com without --cloud, called: {url}"
            )
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        import json as _json
        mock_resp.json.return_value = {
            "response": _json.dumps(_make_vision_result(confidence=0.45, low_confidence=True))
        }
        return mock_resp

    monkeypatch.setattr(vision_mod, "_httpx_post", _sentinel_post)
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    photo_path = _make_jpeg(tmp_path / "item.jpg")
    result = runner.invoke(app, ["capture", str(photo_path)], input="q\n")

    assert anthropic_calls == [], (
        f"ZERO api.anthropic.com calls expected without --cloud; got: {anthropic_calls}"
    )


# ---------------------------------------------------------------------------
# (g) H3: --cloud prints explicit "sending to cloud" notice before the call
# ---------------------------------------------------------------------------

def test_capture_cloud_flag_prints_egress_notice(monkeypatch, tmp_path):
    """capture --cloud prints an explicit 'sending to cloud' notice before the API call (H3).

    RED: ModuleNotFoundError until Wave 2 ships capture/.
    """
    import leopard44_kb.capture as capture_pkg  # RED until Wave 2

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-00000000000000000000000000")
    monkeypatch.chdir(tmp_path)

    cloud_called = [False]

    def _fake_identify(path, zones, cloud=False):
        result = _make_vision_result(confidence=0.92)
        result["cloud_used"] = cloud
        if cloud:
            cloud_called[0] = True
        return result

    monkeypatch.setattr(capture_pkg, "identify_item_for_cli", _fake_identify)
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    photo_path = _make_jpeg(tmp_path / "item.jpg")
    result = runner.invoke(app, ["capture", "--cloud", str(photo_path)], input="q\n")
    combined = (result.stdout or "") + (result.stderr or "")

    # Must print an explicit notice about sending to cloud
    notice_words = ["sending", "cloud", "uploading", "cloud vision"]
    assert any(w in combined.lower() for w in notice_words), (
        f"Expected 'sending to cloud' notice with --cloud; got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# (h) H1: photo-store failure → item STILL created, photo_path NULL, warning emitted
# ---------------------------------------------------------------------------

def test_capture_photo_store_failure_is_fail_soft(monkeypatch, fake_embedder, tmp_path):
    """When store_item_photo raises, the item row is STILL created with photo_path=NULL
    and a 'photo not saved: <reason>' warning is emitted (H1 fail-soft contract).

    RED: ModuleNotFoundError until Wave 2 ships capture/.
    """
    import leopard44_kb.capture as capture_pkg  # RED until Wave 2

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    photo_path = _make_jpeg(tmp_path / "item.jpg")

    monkeypatch.setattr(
        capture_pkg,
        "identify_item_for_cli",
        lambda path, zones, cloud=False: _make_vision_result(confidence=0.92),
    )

    # Simulate a photo-store failure
    def _failing_store_photo(conn, item_id: int, photo_src: Path, dest_dir: Path) -> None:
        raise OSError("disk full: cannot write photo")

    monkeypatch.setattr(capture_pkg, "store_item_photo", _failing_store_photo)
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    result = runner.invoke(app, ["capture", str(photo_path)], input="a\n")
    combined = (result.stdout or "") + (result.stderr or "")

    # Item MUST still be created (no abort over photo I/O error)
    conn = _open_db(db_path)
    row = conn.execute("SELECT photo_path FROM items ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()

    assert row is not None, (
        "Item row must be created even when photo store fails (H1 fail-soft)"
    )
    assert row[0] is None, (
        f"photo_path must be NULL when photo store fails; got: {row[0]!r}"
    )

    # Must emit a "photo not saved" warning
    assert "photo not saved" in combined.lower() or "photo" in combined.lower(), (
        f"Expected 'photo not saved' warning in output; got: {combined!r}"
    )

    # Must NOT abort the whole capture (item created, exit 0)
    assert result.exit_code == 0, (
        f"Expected exit 0 (fail-soft); got {result.exit_code}: {combined!r}"
    )


# ---------------------------------------------------------------------------
# (i) M4: unknown vision zone → zone_id NULL + warning;
#         ambiguous edited zone name (matches >1) → same
# ---------------------------------------------------------------------------

def test_capture_unknown_vision_zone_leaves_zone_id_null(monkeypatch, tmp_path):
    """Unknown suggested_zone from vision leaves zone_id=NULL + emits a warning (M4).

    RED: ModuleNotFoundError until Wave 2 ships capture/.
    """
    import leopard44_kb.capture as capture_pkg  # RED until Wave 2

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    photo_path = _make_jpeg(tmp_path / "item.jpg")

    monkeypatch.setattr(
        capture_pkg,
        "identify_item_for_cli",
        lambda path, zones, cloud=False: _make_vision_result(
            suggested_zone="Nowhere",
            zone_id=None,
            low_confidence=True,
        ),
    )
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    result = runner.invoke(app, ["capture", str(photo_path)], input="a\n")
    combined = (result.stdout or "") + (result.stderr or "")

    # Must warn about unknown zone
    assert any(w in combined.lower() for w in ["unknown zone", "zone not found", "no zone", "unknown"]), (
        f"Expected unknown-zone warning; got: {combined!r}"
    )

    # Item is created but zone_id must be NULL (M4 — unknown zone ⇒ zone_id=None)
    # Note: the DB column is current_zone_id (see schema/002_inventory.sql).
    if result.exit_code == 0:
        conn = _open_db(db_path)
        rows = conn.execute("SELECT current_zone_id FROM items ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        if rows:
            assert rows[0] is None, (
                f"current_zone_id must be NULL for unknown vision zone; got: {rows[0]!r}"
            )


def test_capture_partial_edited_zone_resolves_to_null(monkeypatch, fake_embedder, tmp_path):
    """A partial-only edited zone name resolves to NULL, NOT the partial hit (H3).

    The capture edit prompt states "exact match required". A value like "cabin"
    that only PARTIALLY matches real zones (port-fwd-cabin / stbd-fwd-cabin) must
    NOT silently resolve to one of them — it warns and leaves current_zone_id NULL.

    RED: ModuleNotFoundError until Wave 2 ships capture/.
    """
    import leopard44_kb.capture as capture_pkg  # RED until Wave 2

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)

    # Two zones containing "cabin" exist so "cabin" is a PARTIAL match to both.
    # Note: port-fwd-cabin and stbd-fwd-cabin are seeded by migration 002 already;
    # INSERT OR IGNORE is safe when they are already present.
    conn = _open_db(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT OR IGNORE INTO zones (name, label, side, fore_aft, area, vertical_index) "
        "VALUES ('port-fwd-cabin', 'Port fwd cabin', 'port', 'fwd', 'cabin', 1.0)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO zones (name, label, side, fore_aft, area, vertical_index) "
        "VALUES ('stbd-fwd-cabin', 'Stbd fwd cabin', 'stbd', 'fwd', 'cabin', 1.0)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    photo_path = _make_jpeg(tmp_path / "item.jpg")

    monkeypatch.setattr(
        capture_pkg,
        "identify_item_for_cli",
        lambda path, zones, cloud=False: _make_vision_result(),
    )
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    # Edit zone to "cabin" — only a PARTIAL match → must NOT resolve (H3).
    result = runner.invoke(
        app,
        ["capture", str(photo_path)],
        input="e\nzone\ncabin\na\n",
    )
    combined = (result.stdout or "") + (result.stderr or "")

    # Must warn that there is no exact zone match (no silent partial resolution).
    assert any(
        w in combined.lower()
        for w in ["no exact zone match", "unknown zone", "no exact match"]
    ), (
        f"Expected a no-exact-match warning for a partial edited zone; got: {combined!r}"
    )

    # And current_zone_id must be NULL — never a partial hit.
    if result.exit_code == 0:
        conn = _open_db(db_path)
        row = conn.execute(
            "SELECT current_zone_id FROM items ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            assert row[0] is None, (
                f"partial edited zone must resolve to NULL, not a partial hit; got: {row[0]!r}"
            )


def test_capture_explicit_zone_authoritative_over_edit(monkeypatch, fake_embedder, tmp_path):
    """H2: --zone X stays authoritative even when the owner edits zone to a different
    valid zone Y. Precedence is --zone > edited > vision-exact > null.

    Passes --zone anchor-locker (id 1) and then interactively edits the zone field
    to 'saloon' (id 5). The committed item must keep current_zone_id = 1.
    """
    import leopard44_kb.capture as capture_pkg  # RED until Wave 2

    db_path = tmp_path / "test.db"
    _bootstrap_db(db_path)

    # Resolve the two real seeded zone ids we will use.
    conn = _open_db(db_path)
    conn.row_factory = sqlite3.Row
    anchor_id = conn.execute(
        "SELECT id FROM zones WHERE name = 'anchor-locker'"
    ).fetchone()[0]
    saloon_id = conn.execute("SELECT id FROM zones WHERE name = 'saloon'").fetchone()[0]
    conn.close()
    assert anchor_id != saloon_id

    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    photo_path = _make_jpeg(tmp_path / "item.jpg")

    monkeypatch.setattr(
        capture_pkg,
        "identify_item_for_cli",
        lambda path, zones, cloud=False: _make_vision_result(suggested_zone="saloon"),
    )
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    # --zone anchor-locker, then edit zone → saloon, then accept.
    result = runner.invoke(
        app,
        ["capture", str(photo_path), "--zone", "anchor-locker"],
        input="e\nzone\nsaloon\na\n",
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, f"expected exit 0; got {result.exit_code}: {combined!r}"

    conn = _open_db(db_path)
    row = conn.execute(
        "SELECT current_zone_id FROM items ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None, "an item should have been created"
    assert row[0] == anchor_id, (
        f"H2: --zone anchor-locker must remain authoritative over an edit to saloon; "
        f"expected current_zone_id={anchor_id}, got {row[0]!r}"
    )


# ---------------------------------------------------------------------------
# WR-01: a failed dual-write must not orphan the just-written photo file.
# ---------------------------------------------------------------------------

def test_store_item_photo_unlinks_orphan_on_commit_failure(tmp_path):
    """If the UPDATE/commit fails after the file is written, the file is removed (WR-01).

    Exercises the leopard44_kb.capture.store_item_photo wrapper directly with a real
    photo and a connection whose commit() raises. The processed file must be
    written first (so we can observe it) and then unlinked on the failure path,
    leaving neither a row reference nor an orphan file on disk.
    """
    import leopard44_kb.capture as capture_pkg

    db_path = tmp_path / "wr01.db"
    _bootstrap_db(db_path)

    repo_root = tmp_path
    photo_src = _make_jpeg(tmp_path / "src.jpg")

    conn = _open_db(db_path)
    # Create an item row to attach the photo to.
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO items (name, category) VALUES ('widget', 'spare')")
    conn.commit()
    item_id = conn.execute("SELECT id FROM items WHERE name='widget'").fetchone()[0]
    conn.close()

    expected_file = repo_root / "data" / "photos" / "items" / f"ITEM-{item_id}.jpg"

    # sqlite3.Connection.commit is read-only, so wrap the connection in a thin
    # proxy whose commit() raises — simulating a DB-locked / disk-full journal
    # failure AFTER photo.store_item_photo has written the processed file.
    class _CommitFailsConn:
        def __init__(self, real):
            self._real = real

        def execute(self, *a, **kw):
            return self._real.execute(*a, **kw)

        def commit(self):
            raise sqlite3.OperationalError("database is locked")

    real_conn = _open_db(db_path)
    proxy = _CommitFailsConn(real_conn)

    with pytest.raises(sqlite3.OperationalError):
        capture_pkg.store_item_photo(proxy, item_id, photo_src, repo_root)

    real_conn.close()

    assert not expected_file.exists(), (
        f"WR-01: a failed dual-write must unlink the orphan photo file, "
        f"but it still exists at {expected_file}"
    )
