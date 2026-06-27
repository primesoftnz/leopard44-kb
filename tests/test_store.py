# RED state until Plan 02 (see VALIDATION.md). Imports from leopard44_kb.* will fail until production code lands.
"""Tests for open_db() pragmas, sqlite-vec extension load, WAL mode."""
from __future__ import annotations

import importlib

import pytest

from leopard44_kb.store import open_db, DEFAULT_DB_PATH


def test_fk_pragma_is_on(tmp_path):
    """open_db() connection has PRAGMA foreign_keys = ON."""
    conn = open_db(tmp_path / "store.db")
    try:
        result = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert result == 1, f"Expected foreign_keys=1, got {result}"
    finally:
        conn.close()


def test_extension_loaded(tmp_path):
    """open_db() connection has sqlite-vec loaded (vec_version() works)."""
    conn = open_db(tmp_path / "store.db")
    try:
        row = conn.execute("SELECT vec_version()").fetchone()
        assert row is not None
        version = row[0]
        assert isinstance(version, str) and len(version) > 0, (
            f"Expected non-empty vec_version, got {version!r}"
        )
    finally:
        conn.close()


def test_journal_mode_wal(tmp_path):
    """open_db() sets WAL journal mode for on-disk databases."""
    conn = open_db(tmp_path / "store.db")
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].lower() == "wal", f"Expected WAL, got {row[0]!r}"
    finally:
        conn.close()


def test_default_db_path_uses_xdg_or_env(monkeypatch, tmp_path):
    """L44_DB env override is respected by DEFAULT_DB_PATH."""
    override_path = str(tmp_path / "override.db")
    monkeypatch.setenv("L44_DB", override_path)

    import leopard44_kb.store as store_mod
    importlib.reload(store_mod)

    from pathlib import Path
    assert store_mod.DEFAULT_DB_PATH == Path(override_path), (
        f"Expected DEFAULT_DB_PATH={override_path!r}, got {store_mod.DEFAULT_DB_PATH!r}"
    )
