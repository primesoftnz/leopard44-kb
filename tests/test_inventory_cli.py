# RED state until Wave 3 (Plan 03) lands. Import of leopard44_kb.inventory fails at
# collection time because inventory.py does not yet exist — a collection-time
# ModuleNotFoundError is the EXPECTED Wave-0 state, consistent with the Phase 1-5
# import-at-collection RED convention. The module-top import is deliberate.
"""Tests for INV-01/04 and zone/item CLI surface (zone add/list, item add/list/find/locate).

Per-requirement verification map source:
  .planning/phases/08-zone-taxonomy-inventory-core/08-VALIDATION.md
CLI naming (finding 11): item command group is `item` (NOT `inv`); zone group is `zone`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlite_vec
from typer.testing import CliRunner

from leopard44_kb.cli import app
from leopard44_kb.schema import apply_migrations

import leopard44_kb.inventory as inv  # collection-time ModuleNotFoundError = expected RED state

runner = CliRunner()


def _bootstrap_db(db_path: Path) -> None:
    """Bootstrap a file-backed DB at db_path with sqlite-vec and migrations applied."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    conn.close()


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Open the bootstrapped DB for direct queries in tests."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# ZONE-01: zone add / zone list CLI surface
# ---------------------------------------------------------------------------


def test_zone_add(monkeypatch, fake_embedder, tmp_path):
    """zone add exits 0 and creates a zones row."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "inventory").mkdir(parents=True, exist_ok=True)
    _bootstrap_db(db_path)

    result = runner.invoke(
        app,
        ["zone", "add", "saloon-port-locker", "Saloon port locker",
         "--side", "port", "--no-ai", "--yes"],
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 for zone add; got {result.exit_code}: {combined!r}"
    )

    conn = _open_db(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM zones WHERE name='saloon-port-locker'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "zones row must exist after zone add"


def test_zone_list(monkeypatch, fake_embedder, tmp_path):
    """zone list exits 0 and stdout contains the zone label after a zone add."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "inventory").mkdir(parents=True, exist_ok=True)
    _bootstrap_db(db_path)

    runner.invoke(
        app,
        ["zone", "add", "saloon-port-locker", "Saloon port locker",
         "--side", "port", "--no-ai", "--yes"],
    )
    result = runner.invoke(app, ["zone", "list"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 for zone list; got {result.exit_code}: {combined!r}"
    )
    assert "Saloon port locker" in combined, (
        f"zone list stdout must contain zone label 'Saloon port locker', got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# INV-01: item add CLI surface
# ---------------------------------------------------------------------------


def test_item_add_cli(monkeypatch, fake_embedder, tmp_path):
    """item add exits 0; items row exists; ITEM-*.md is written under data/inventory."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "inventory").mkdir(parents=True, exist_ok=True)
    _bootstrap_db(db_path)

    # First add a zone so the item can reference it
    runner.invoke(
        app,
        ["zone", "add", "saloon-port-locker", "Saloon port locker",
         "--side", "port", "--no-ai", "--yes"],
    )

    result = runner.invoke(
        app,
        ["item", "add", "Scrabble", "--category", "toy",
         "--zone", "saloon-port-locker", "--yes"],
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 for item add; got {result.exit_code}: {combined!r}"
    )

    conn = _open_db(db_path)
    try:
        row = conn.execute("SELECT id FROM items WHERE name='Scrabble'").fetchone()
    finally:
        conn.close()
    assert row is not None, "items row must exist after item add"

    md_files = list((tmp_path / "data" / "inventory").glob("ITEM-*.md"))
    assert len(md_files) >= 1, (
        f"Expected at least one ITEM-*.md under data/inventory/, found: {md_files}"
    )


def test_item_add_unknown_zone(monkeypatch, fake_embedder, tmp_path):
    """item add with --zone that doesn't exist exits 1 and reports 'zone not found'; no item row."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "inventory").mkdir(parents=True, exist_ok=True)
    _bootstrap_db(db_path)

    result = runner.invoke(
        app,
        ["item", "add", "Widget", "--category", "tool",
         "--zone", "does-not-exist", "--yes"],
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 1, (
        f"Expected exit 1 for unknown zone; got {result.exit_code}: {combined!r}"
    )
    assert "zone not found" in combined.lower(), (
        f"Expected 'zone not found' in output, got: {combined!r}"
    )

    conn = _open_db(db_path)
    try:
        row = conn.execute("SELECT id FROM items WHERE name='Widget'").fetchone()
    finally:
        conn.close()
    assert row is None, "No items row must be written when zone is not found"


# ---------------------------------------------------------------------------
# INV-04: item list — zone and category filters
# ---------------------------------------------------------------------------


def test_item_list_zone_filter(monkeypatch, fake_embedder, tmp_path):
    """item list --zone returns only items in the specified zone."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "inventory").mkdir(parents=True, exist_ok=True)
    _bootstrap_db(db_path)

    # Add two zones
    runner.invoke(
        app, ["zone", "add", "saloon-port-locker", "Saloon port locker", "--no-ai", "--yes"]
    )
    runner.invoke(
        app, ["zone", "add", "fore-cabin", "Fore cabin", "--no-ai", "--yes"]
    )

    # Add items in different zones
    runner.invoke(
        app,
        ["item", "add", "Scrabble", "--category", "toy",
         "--zone", "saloon-port-locker", "--yes"],
    )
    runner.invoke(
        app,
        ["item", "add", "Anchor chain", "--category", "tool",
         "--zone", "fore-cabin", "--yes"],
    )

    result = runner.invoke(app, ["item", "list", "--zone", "saloon-port-locker"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 for item list --zone; got {result.exit_code}: {combined!r}"
    )
    assert "Scrabble" in combined, (
        f"item list --zone saloon-port-locker must contain 'Scrabble', got: {combined!r}"
    )
    assert "Anchor chain" not in combined, (
        f"item list --zone saloon-port-locker must NOT contain 'Anchor chain', got: {combined!r}"
    )


def test_item_list_category_filter(monkeypatch, fake_embedder, tmp_path):
    """item list --category returns only items of the specified category."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "inventory").mkdir(parents=True, exist_ok=True)
    _bootstrap_db(db_path)

    runner.invoke(
        app, ["zone", "add", "saloon-port-locker", "Saloon port locker", "--no-ai", "--yes"]
    )

    runner.invoke(
        app,
        ["item", "add", "Scrabble", "--category", "toy",
         "--zone", "saloon-port-locker", "--yes"],
    )
    runner.invoke(
        app,
        ["item", "add", "Impeller", "--category", "spare",
         "--zone", "saloon-port-locker", "--yes"],
    )

    result = runner.invoke(app, ["item", "list", "--category", "spare"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 for item list --category; got {result.exit_code}: {combined!r}"
    )
    assert "Impeller" in combined, (
        f"item list --category spare must contain 'Impeller', got: {combined!r}"
    )
    assert "Scrabble" not in combined, (
        f"item list --category spare must NOT contain 'Scrabble', got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# INV-04 / D-10: item find + item locate CLI surface
# ---------------------------------------------------------------------------


def test_item_find_cli(monkeypatch, fake_embedder, tmp_path):
    """item find exits 0 and stdout names the zone."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "inventory").mkdir(parents=True, exist_ok=True)
    _bootstrap_db(db_path)

    runner.invoke(
        app, ["zone", "add", "saloon-port-locker", "Saloon port locker", "--no-ai", "--yes"]
    )
    runner.invoke(
        app,
        ["item", "add", "Scrabble", "--category", "toy",
         "--zone", "saloon-port-locker", "--yes"],
    )

    result = runner.invoke(app, ["item", "find", "scrabble"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 for item find; got {result.exit_code}: {combined!r}"
    )
    # Output must reference the zone
    assert "saloon" in combined.lower() or "port locker" in combined.lower(), (
        f"item find output must name the zone, got: {combined!r}"
    )


def test_item_locate_cli(monkeypatch, fake_embedder, tmp_path):
    """item locate exits 0 and shows stow location from structured item record (not LLM text)."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "inventory").mkdir(parents=True, exist_ok=True)
    _bootstrap_db(db_path)

    runner.invoke(
        app, ["zone", "add", "saloon-port-locker", "Saloon port locker", "--no-ai", "--yes"]
    )
    runner.invoke(
        app,
        ["item", "add", "Scrabble", "--category", "toy",
         "--zone", "saloon-port-locker", "--aliases", "board game",
         "--yes"],
    )

    result = runner.invoke(app, ["item", "locate", "board game"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 for item locate; got {result.exit_code}: {combined!r}"
    )
    # Output must show a stow location from the structured item record
    assert "saloon" in combined.lower() or "port locker" in combined.lower(), (
        f"item locate output must show zone from structured item record, got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# CR-01 regression: zone/item commands must self-migrate on a raw DB
# (DB has sqlite-vec loaded but apply_migrations has NOT been called yet)
# ---------------------------------------------------------------------------


def _raw_db(db_path: Path) -> None:
    """Create a file-backed DB with sqlite-vec loaded but NO schema migrations applied.

    This simulates a pre-v1.1 database on disk — the exact condition that would
    have caused CR-01 ('no such table: zones') before the apply_migrations fix.
    """
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # Deliberately do NOT call apply_migrations — raw schema v0
    conn.close()


def test_zone_list_self_migrates(monkeypatch, tmp_path):
    """zone list must not crash on a pre-v1.1 DB (CR-01 regression)."""
    db_path = tmp_path / "raw.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _raw_db(db_path)

    result = runner.invoke(app, ["zone", "list"])
    combined = (result.stdout or "") + (result.stderr or "")
    # Must exit 0 (empty list is fine); must NOT crash with OperationalError
    assert result.exit_code == 0, (
        f"zone list must self-migrate on a raw DB (CR-01); "
        f"got exit {result.exit_code}: {combined!r}"
    )
    assert "OperationalError" not in combined and "no such table" not in combined, (
        f"zone list must not crash with missing-table error on raw DB: {combined!r}"
    )


def test_item_list_self_migrates(monkeypatch, tmp_path):
    """item list must not crash on a pre-v1.1 DB (CR-01 regression)."""
    db_path = tmp_path / "raw.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _raw_db(db_path)

    result = runner.invoke(app, ["item", "list"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"item list must self-migrate on a raw DB (CR-01); "
        f"got exit {result.exit_code}: {combined!r}"
    )
    assert "OperationalError" not in combined and "no such table" not in combined, (
        f"item list must not crash with missing-table error on raw DB: {combined!r}"
    )


def test_item_find_self_migrates(monkeypatch, tmp_path):
    """item find must not crash on a pre-v1.1 DB (CR-01 regression)."""
    db_path = tmp_path / "raw.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _raw_db(db_path)

    result = runner.invoke(app, ["item", "find", "anything"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"item find must self-migrate on a raw DB (CR-01); "
        f"got exit {result.exit_code}: {combined!r}"
    )
    assert "OperationalError" not in combined and "no such table" not in combined, (
        f"item find must not crash with missing-table error on raw DB: {combined!r}"
    )
