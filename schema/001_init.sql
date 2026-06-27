-- Leopard 44 KB v1 schema — full DDL
-- Executed by leopard44_kb.schema.apply_migrations() via sqlite3.Connection.executescript()
-- DO NOT use Alembic op.execute() directives or any non-SQL syntax here.

PRAGMA foreign_keys = ON;

-- =============================================================================
-- Section 1: sources
-- =============================================================================

-- SCHEMA-03 (vessel-layer path restriction) is enforced by leopard44_kb.paths.validate_path()
-- at the ingest boundary, not by a DB constraint. See Pitfall 1 in 01-RESEARCH.md.
-- AUTOINCREMENT prevents SQLite from recycling deleted rowids.
-- This is required so that after a re-ingest (delete+reinsert), vec_chunks rows
-- belonging to the NEW source have a distinct source_id from the deleted OLD source.
-- Without AUTOINCREMENT, SQLite recycles the lowest unused rowid, making orphan
-- detection impossible in a single-source database (plan 02-02).
CREATE TABLE sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    layer           TEXT NOT NULL CHECK (layer IN ('shared','vessel','community')),
    source_type     TEXT NOT NULL,
    path            TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    title           TEXT,
    original_uri    TEXT,
    version         TEXT,
    ingested_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_metadata TEXT
);

CREATE UNIQUE INDEX idx_sources_layer_path ON sources(layer, path);

-- =============================================================================
-- Section 2: chunks
-- =============================================================================

CREATE TABLE chunks (
    id                       INTEGER PRIMARY KEY,
    source_id                INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    layer                    TEXT NOT NULL CHECK (layer IN ('shared','vessel','community')),
    ordinal                  INTEGER NOT NULL,
    section_path             TEXT,
    page_start               INTEGER,
    page_end                 INTEGER,
    char_offset_start        INTEGER,
    char_offset_end          INTEGER,
    token_count              INTEGER,
    content                  TEXT NOT NULL,
    content_hash             TEXT NOT NULL,
    -- anchor_key = sha256(source.path + '\n' + section_path + '\n' + ordinal_within_section).
    -- Phase 2 computes it; see Pitfall 3 in 01-RESEARCH.md.
    -- ordinal is relative to section_path, not to the whole source, so that upstream
    -- inserts into one section do not shift anchor_keys in sibling sections.
    anchor_key TEXT NOT NULL,
    supersedes_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    annotates_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    embedding_model TEXT NOT NULL,
    embedding_model_version TEXT NOT NULL,
    is_active                BOOLEAN NOT NULL DEFAULT 1,
    created_at               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata                 TEXT          -- JSON blob
);

CREATE INDEX idx_chunks_source     ON chunks(source_id);
CREATE INDEX idx_chunks_layer      ON chunks(layer);
CREATE INDEX idx_chunks_anchor     ON chunks(anchor_key);
CREATE INDEX idx_chunks_supersedes ON chunks(supersedes_chunk_id);
CREATE INDEX idx_chunks_annotates  ON chunks(annotates_chunk_id);
CREATE UNIQUE INDEX idx_chunks_source_ordinal ON chunks(source_id, ordinal);

-- =============================================================================
-- Section 3: vec_chunks (vec0 virtual table)
-- =============================================================================

-- layer is a partition key — sqlite-vec shards the index internally so that
-- KNN queries with WHERE layer='vessel' only scan the vessel partition.
-- embedding float[384]: dimension locked at 384 per RESEARCH.md Pattern 3
-- (Matryoshka lock — all-MiniLM-L6-v2 native 384; nomic-embed-text v1.5
-- truncated to 384 via Ollama dimensions=384 + re-normalise).
-- chunk_id INTEGER PRIMARY KEY shares the same value as chunks.id (set
-- explicitly on INSERT in Phase 2). fts_chunks.rowid is kept in sync by
-- the three AI/AD/AU triggers below — no manual sync needed.
CREATE VIRTUAL TABLE vec_chunks USING vec0(
    chunk_id        INTEGER PRIMARY KEY,
    layer           TEXT partition key,
    source_id       INTEGER,
    embedding_model TEXT,
    is_active       INTEGER,
    embedding       float[384]
);

-- =============================================================================
-- Section 4: fts_chunks (FTS5 external-content table + 3 sync triggers)
-- =============================================================================

-- External-content FTS5: content lives in chunks.content; FTS5 holds only the
-- inverted index. Do NOT use content='' (contentless) — that prevents text
-- reconstruction from FTS5 alone. See Anti-Patterns in 01-RESEARCH.md.
CREATE VIRTUAL TABLE fts_chunks USING fts5(content, content='chunks', content_rowid='id');

-- AFTER INSERT: new chunk content enters the FTS index.
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
END;

-- AFTER DELETE: remove chunk content from FTS index.
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, content) VALUES('delete', old.id, old.content);
END;

-- AFTER UPDATE: remove old content and re-insert new content in FTS index.
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
END;

-- =============================================================================
-- Section 5: schema_version
-- =============================================================================

CREATE TABLE schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO schema_version(version) VALUES (1);
