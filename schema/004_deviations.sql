-- schema/004_deviations.sql
-- Factory-deviation log table (Phase 11: DEV-01/DEV-02).
-- Executed by leopard44_kb.schema.apply_migrations() via sqlite3.Connection.executescript()
-- DO NOT use Alembic op.execute() directives or any non-SQL syntax here.
--
-- NOTE: PRAGMA foreign_keys is a connection-level pragma set once in open_db().
-- Re-asserting it here via executescript() forces an implicit COMMIT that breaks
-- the with-conn: atomicity guarantee in apply_migrations(). Omit it here.
--
-- AUTOINCREMENT prevents SQLite from recycling deleted rowids.
-- chunk metadata carries deviation_id and zone_id cross-references, so a recycled
-- deviation_id would make old chunk metadata silently point at a new deviation.
-- Same rationale as zones.id and items.id (see 002_inventory.sql).

CREATE TABLE deviations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    component       TEXT NOT NULL,          -- the system/part that differs from factory
    factory_spec    TEXT,                   -- how the L44 left the factory
    as_built        TEXT,                   -- the actual current state
    reason          TEXT,                   -- why it differs
    date_noted      TEXT,                   -- when the deviation was first observed (free-form date)
    zone_id         INTEGER REFERENCES zones(id) ON DELETE SET NULL,
                                            -- OPTIONAL schematic location (NULL = no zone known)
                                            -- mirrors items.current_zone_id EXACTLY — optional FK
                                            -- + ON DELETE SET NULL, so deleting a referenced zone
                                            -- preserves the deviation with location unknown rather
                                            -- than failing the delete on a FK violation.
    -- chunk_source_id FKs sources(id) (the chunk's source row), NOT chunks(id); kept identical
    -- to items.chunk_source_id — same name AND ON DELETE SET NULL — so source cleanup/re-ingest
    -- nulls the link rather than breaking.
    chunk_source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    notes           TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_deviations_zone ON deviations(zone_id);
