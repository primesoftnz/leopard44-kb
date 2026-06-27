# RED state until Plan 04 (see VALIDATION.md). Imports from leopard44_kb.* will fail until production code lands.
"""Tests for SCHEMA-02: l44 sources --layer subcommand via typer CliRunner."""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from leopard44_kb.cli import app

# mix_stderr is not supported in typer 0.26 — stderr is merged into result.output
# (result.stdout). Tests that check combined output work correctly with this setup.
runner = CliRunner()


def test_sources_empty_db(monkeypatch, tmp_path):
    """sources --layer shared against empty DB exits 0 with empty stdout."""
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    result = runner.invoke(app, ["sources", "--layer", "shared"])
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert result.stdout.strip() == "", f"Expected empty stdout, got: {result.stdout!r}"


def test_sources_filters_by_layer(monkeypatch, tmp_path):
    """sources --layer shared returns only shared-layer rows."""
    import sqlite3
    import sqlite_vec
    from leopard44_kb.store import open_db
    from leopard44_kb.schema import apply_migrations

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))

    # Seed the database with 2 shared + 1 vessel source
    conn = open_db(db_path)
    apply_migrations(conn)
    conn.execute(
        "INSERT INTO sources (layer, source_type, path, content_hash, ingested_at) "
        "VALUES ('shared', 'markdown', 'shared/a.md', 'h1', CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO sources (layer, source_type, path, content_hash, ingested_at) "
        "VALUES ('shared', 'markdown', 'shared/b.md', 'h2', CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO sources (layer, source_type, path, content_hash, ingested_at) "
        "VALUES ('vessel', 'markdown', 'data/logs/engine.md', 'h3', CURRENT_TIMESTAMP)"
    )
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["sources", "--layer", "shared"])
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    assert len(lines) == 2, f"Expected 2 lines for shared layer, got {len(lines)}: {result.stdout!r}"


def test_sources_bad_layer_rejected():
    """sources --layer garbage exits non-zero and mentions valid layer values."""
    result = runner.invoke(app, ["sources", "--layer", "garbage"])
    assert result.exit_code != 0, "Expected non-zero exit for invalid layer"
    combined = (result.stdout or "") + (result.stderr or "")
    assert any(v in combined for v in ("shared", "vessel", "community", "garbage")), (
        f"Expected error to mention valid layers or rejected value, got: {combined!r}"
    )


def test_sources_missing_layer_rejected():
    """sources without --layer exits non-zero (required option)."""
    result = runner.invoke(app, ["sources"])
    assert result.exit_code != 0, "Expected non-zero exit when --layer is missing"


def test_ingest_no_longer_a_stub(monkeypatch, tmp_path):
    """ingest command is real in Phase 2 — does NOT say 'Not yet implemented'.

    A valid vessel path that fails due to missing file or layer mismatch exits
    with code 1 (partial failure) rather than code 2 (stub). The key invariant
    is that the old stub message is gone.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))

    # Pass a path that doesn't exist — ingest_file will raise, exit code 1.
    # Default layer 'vessel' is used (D-06) so no --layer required.
    result = runner.invoke(app, ["ingest", str(tmp_path / "nonexistent.md")])
    # Exit code must NOT be 2 (the stub code); 1 or 0 are both acceptable.
    assert result.exit_code != 2, (
        f"ingest still behaves as a stub (exit 2); got: {result.output!r}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Not yet implemented" not in combined, (
        f"Stub message still present after Phase 2 replacement: {combined!r}"
    )



def test_add_no_longer_a_stub_sources():
    """add command is real in Phase 4 — does NOT say 'Not yet implemented'.

    A non-TTY invocation without --yes exits with code 1 ('no terminal for review')
    rather than code 2 (stub). The key invariant is that the old stub message is gone.
    Mirrors test_ingest_no_longer_a_stub and test_add_no_longer_a_stub (test_maintenance_cli.py).
    """
    result = runner.invoke(app, ["add", "replaced port impeller"])
    # Exit code must NOT be 2 (the stub code); 1 is expected for non-TTY without --yes.
    assert result.exit_code != 2, (
        f"add still behaves as a stub (exit 2); got: {result.output!r}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Not yet implemented" not in combined, (
        f"Stub message still present after Phase 4 replacement: {combined!r}"
    )


def test_serve_no_longer_a_stub_sources(monkeypatch, tmp_path):
    """serve command is real in Phase 5 — does NOT say 'Not yet implemented'.

    Mirrors test_ingest_no_longer_a_stub and test_add_no_longer_a_stub_sources.
    uvicorn.run is patched to a no-op so the test does not block.
    Exit code must NOT be 2 (the stub code); 0 is expected after real replacement.
    """
    import uvicorn

    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 2, (
        f"serve still behaves as a stub (exit 2); got: {result.output!r}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Not yet implemented" not in combined, (
        f"Stub message still present after Phase 5 replacement: {combined!r}"
    )
