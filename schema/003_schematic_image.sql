-- schema/003_schematic_image.sql
-- Adds schematic_image column to zones (Phase 9: D-05 one-zone-one-image).
-- zones.geometry was reserved in 002_inventory.sql and stays as-is.
-- ALTER TABLE ADD COLUMN is NOT idempotent in SQLite — the version guard in
-- apply_migrations() (schema.py lines 56-58) is the sole protection.
-- DO NOT add IF NOT EXISTS — that is a SQLite syntax error.
--
-- NOTE: PRAGMA foreign_keys is a connection-level pragma set once in open_db().
-- Re-asserting it here via executescript() forces an implicit COMMIT that breaks
-- the with-conn: atomicity guarantee in apply_migrations(). Omit it here.

ALTER TABLE zones ADD COLUMN schematic_image TEXT;  -- NULL until set by editor

INSERT INTO schema_version(version) VALUES (3);
