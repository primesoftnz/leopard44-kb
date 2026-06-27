# RED state until Plan 02 (see VALIDATION.md). Imports from leopard44_kb.* will fail until production code lands.
"""Tests for SCHEMA-01 / SCHEMA-04: schema introspection, migration idempotency, FTS5 sync."""
from __future__ import annotations

import re

import pytest

from leopard44_kb.schema import apply_migrations


def test_chunks_has_layer_column(empty_db):
    """SCHEMA-01: chunks table has a NOT NULL layer column with CHECK constraint."""
    rows = empty_db.execute("PRAGMA table_info(chunks)").fetchall()
    col_names = {row["name"]: row for row in rows}
    assert "layer" in col_names, "layer column missing from chunks"
    assert col_names["layer"]["notnull"] == 1, "layer must be NOT NULL"

    # Verify the CHECK constraint is expressed in the DDL
    sql_row = empty_db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone()
    assert sql_row is not None
    ddl = sql_row["sql"]
    # CHECK (layer IN ('shared', 'vessel', 'community')) — tolerates whitespace variation
    assert re.search(
        r"CHECK\s*\(\s*layer\s+IN\s*\(\s*'shared'\s*,\s*'vessel'\s*,\s*'community'\s*\)\s*\)",
        ddl,
        re.IGNORECASE,
    ), f"Expected CHECK constraint on layer not found in DDL:\n{ddl}"


def test_vec_chunks_has_layer_partition(empty_db):
    """SCHEMA-01: vec_chunks virtual table has layer as partition key with float[384]."""
    sql_row = empty_db.execute(
        "SELECT sql FROM sqlite_master WHERE name='vec_chunks'"
    ).fetchone()
    assert sql_row is not None, "vec_chunks table missing"
    ddl = sql_row["sql"].lower()
    assert "layer" in ddl, "layer column missing from vec_chunks"
    assert "partition key" in ddl, "layer must be declared as partition key"
    assert "float[384]" in ddl, "embedding must be float[384] (matryoshka dim lock)"


def test_chunks_has_embedding_model_columns(empty_db):
    """SCHEMA-04: chunks has NOT NULL embedding_model and embedding_model_version columns."""
    rows = empty_db.execute("PRAGMA table_info(chunks)").fetchall()
    col_names = {row["name"]: row for row in rows}

    assert "embedding_model" in col_names, "embedding_model column missing from chunks"
    assert col_names["embedding_model"]["notnull"] == 1, "embedding_model must be NOT NULL"

    assert "embedding_model_version" in col_names, "embedding_model_version column missing"
    assert col_names["embedding_model_version"]["notnull"] == 1, "embedding_model_version must be NOT NULL"


def test_migrations_idempotent(empty_db):
    """Calling apply_migrations twice on same DB must not error or duplicate rows."""
    # First call is already done by the empty_db fixture
    # Call again
    apply_migrations(empty_db)

    row = empty_db.execute("SELECT MAX(version) FROM schema_version").fetchone()
    # schema_version grows with each new migration; assert the current latest (v4 after 004_deviations.sql)
    assert row[0] == 4, f"Expected schema_version=4, got {row[0]}"

    count = empty_db.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    assert count == 4, f"Expected 4 rows in schema_version, got {count}"


def test_self_fk_uses_set_null(empty_db):
    """D-02: supersedes_chunk_id and annotates_chunk_id are ON DELETE SET NULL."""
    fk_rows = empty_db.execute("PRAGMA foreign_key_list(chunks)").fetchall()
    self_fks = [r for r in fk_rows if r["table"] == "chunks"]
    assert len(self_fks) == 2, f"Expected 2 self-FKs on chunks, got {len(self_fks)}: {self_fks}"
    for fk in self_fks:
        assert fk["on_delete"].upper() == "SET NULL", (
            f"FK {fk['from']} must be ON DELETE SET NULL, got {fk['on_delete']!r}"
        )


def test_migration_003(tmp_path):
    """D-05 / Phase 9 Wave 0: schema migration 003 adds schematic_image to zones.

    Bootstraps a file-backed DB via apply_migrations (same pattern as
    test_inventory_cli.py lines 28-37), then asserts:
      (a) zones table has a column named schematic_image
      (b) schema_version has a row WHERE version = 3 EXACTLY
          (not MAX(version) >= 3 — per review concern, avoids masking a broken 003)

    RED until 09-02 ships schema/003_schematic_image.sql.
    """
    import sqlite3
    import sqlite_vec

    db_path = tmp_path / "migration_003.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    conn.close()

    # Re-open for assertion queries
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # (a) zones table must include a schematic_image column
    col_info = conn.execute("PRAGMA table_info(zones)").fetchall()
    col_names = {row["name"] for row in col_info}
    assert "schematic_image" in col_names, (
        f"zones table is missing schematic_image column after migration 003. "
        f"Columns present: {col_names!r}"
    )

    # (b) schema_version must have a row WHERE version = 3 exactly
    row = conn.execute(
        "SELECT 1 FROM schema_version WHERE version = 3"
    ).fetchone()
    assert row is not None, (
        "schema_version has no row WHERE version = 3. "
        "Migration 003 must INSERT INTO schema_version(version) VALUES (3)."
    )

    conn.close()


def test_fts_triggers_keep_in_sync(empty_db):
    """FTS5 external-content triggers keep fts_chunks in sync with chunks."""
    # Need a sources row first (FK constraint)
    empty_db.execute(
        "INSERT INTO sources (layer, source_type, path, content_hash, ingested_at) "
        "VALUES ('shared', 'markdown', 'shared/test.md', 'abc123', CURRENT_TIMESTAMP)"
    )
    source_id = empty_db.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Insert a chunk — the chunks_ai trigger should populate fts_chunks
    empty_db.execute(
        "INSERT INTO chunks "
        "(source_id, layer, ordinal, content, content_hash, anchor_key, "
        " embedding_model, embedding_model_version) "
        "VALUES (?, 'shared', 1, 'hello world test content', 'h1', 'ak1', "
        "        'all-minilm', 'v1')",
        (source_id,),
    )
    empty_db.commit()

    # INSERT trigger: row appears in FTS
    results = empty_db.execute(
        "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH 'hello'"
    ).fetchall()
    assert len(results) == 1, "Expected 1 FTS hit after INSERT"

    chunk_id = results[0][0]

    # UPDATE trigger: old content gone, new content searchable
    empty_db.execute(
        "UPDATE chunks SET content = 'goodbye updated content' WHERE id = ?",
        (chunk_id,),
    )
    empty_db.commit()

    old_results = empty_db.execute(
        "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH 'hello'"
    ).fetchall()
    assert len(old_results) == 0, "Old FTS token should be gone after UPDATE"

    new_results = empty_db.execute(
        "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH 'goodbye'"
    ).fetchall()
    assert len(new_results) == 1, "New FTS token should appear after UPDATE"

    # DELETE trigger: row gone from FTS
    empty_db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    empty_db.commit()

    deleted_results = empty_db.execute(
        "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH 'goodbye'"
    ).fetchall()
    assert len(deleted_results) == 0, "FTS entry should be gone after DELETE"
