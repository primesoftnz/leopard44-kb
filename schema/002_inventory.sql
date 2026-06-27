-- Leopard 44 KB v1.1 inventory schema — zones, zone_sub_slots, items
-- Executed by leopard44_kb.schema.apply_migrations() via sqlite3.Connection.executescript()
-- DO NOT use Alembic op.execute() directives or any non-SQL syntax here.

PRAGMA foreign_keys = ON;

-- =============================================================================
-- Section 1: zones
-- =============================================================================

-- Named storage zones for the vessel.
-- vertical_index: REAL so zones can be inserted between existing ones without
--   renumbering (e.g. 1.5 between 1 and 2). NULL = unspecified.
-- area: free grouping tag (e.g. "cockpit", "saloon", "port-hull", "starboard-hull", "bilge")
-- geometry: NULL in Phase 8 (filled in Phase 9 schematic polygon editor)
-- color_hint: NULL in Phase 8 (filled in Phase 9 schematic annotation)
--
-- AUTOINCREMENT prevents SQLite from recycling deleted rowids.
-- This is required because chunks.metadata carries zone_id and item_id references.
-- A recycled zone_id would make old chunk metadata silently point at a new zone.
-- Without AUTOINCREMENT, SQLite recycles the lowest unused rowid, making orphan
-- detection unreliable after a zone delete+reinsert cycle.
CREATE TABLE zones (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,       -- slug, e.g. "stbd-aft-cabin-hanging-locker"
    label           TEXT NOT NULL,              -- display name "Stbd aft cabin hanging locker"
    side            TEXT CHECK (side IN ('port','stbd','centre','both') OR side IS NULL),
    fore_aft        TEXT CHECK (fore_aft IN ('fwd','mid','aft') OR fore_aft IS NULL),
    vertical_index  REAL,                       -- D-01 orderable; NULL = unspecified
    vertical_desc   TEXT,                       -- D-01 e.g. "lower shelf"; AI-generated default
    area            TEXT,                       -- D-04 grouping tag e.g. "cockpit"
    geometry        TEXT,                       -- Phase 9: JSON [[x,y],...] polygon; NULL in P8
    color_hint      TEXT,                       -- Phase 9: hex color; NULL in P8
    notes           TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_zones_area     ON zones(area);
CREATE INDEX idx_zones_side     ON zones(side);
CREATE INDEX idx_zones_fore_aft ON zones(fore_aft);

-- =============================================================================
-- Section 2: zone_sub_slots
-- =============================================================================

-- Optional sub-slot grid per zone (D-02).
-- A zone with no sub-slots holds items directly (zero rows for that zone_id).
-- row_label/col_label: e.g. "Shelf 1", "Section A" — owner-set at grid definition time.
-- ON DELETE CASCADE: removing a zone removes all its sub-slot grid rows.
CREATE TABLE zone_sub_slots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id     INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    row_num     INTEGER NOT NULL,   -- 1-based (shelf 1, shelf 2, shelf 3)
    col_num     INTEGER NOT NULL,   -- 1-based (section A=1, section B=2)
    row_label   TEXT,               -- "Shelf 1"
    col_label   TEXT,               -- "Section A"
    notes       TEXT,
    UNIQUE (zone_id, row_num, col_num)
);

CREATE INDEX idx_sub_slots_zone ON zone_sub_slots(zone_id);

-- =============================================================================
-- Section 3: items
-- =============================================================================

-- Inventory items — each assigned to a zone (optional).
-- current_zone_id: FK to zones; NULL = location unknown / not yet assigned.
-- current_sub_slot: JSON {"row": 1, "col": 2, "row_label": "Shelf 1", "col_label": "Section A"}
--   or NULL if zone has no grid or slot unspecified.
-- location_history: JSON array of {"zone_id": N, "sub_slot": ..., "ts": "..."} objects,
--   newest-first. Populated by Phase 10 voice-location-update; empty array at creation.
-- metadata: JSON blob for category-specific fields (D-06/D-07):
--   spare:     {"part_number": "22-41016"}
--   provision: {"quantity": 3, "unit": "each", "best_before": "2027-06-01"}
--   safety:    {"expiry": "2026-11-01", "last_inspected": "2025-11-01"}
--   tool/toy:  {} or any owner-added fields
-- photo_path: relative to repo root (data/photos/items/ITEM-{id}.jpg); populated by Phase 12
-- chunk_source_id: FK to sources.id for the vessel-layer chunk this item is embedded as.
--   Set after first ingest; NULL until then. Used for delete+re-embed on item update.
--
-- quantity = at-last-physical-check, not real-time; system never decrements automatically.
--
-- AUTOINCREMENT prevents SQLite from recycling deleted rowids.
-- Recycled item ids would corrupt chunks.metadata item_id cross-references (same
-- rationale as zones.id and sources.id AUTOINCREMENT).
CREATE TABLE items (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    aliases             TEXT,                   -- comma-separated synonyms for richer FTS
    brand               TEXT,
    model_number        TEXT,
    category            TEXT NOT NULL CHECK (
                            category IN ('spare','provision','safety','tool','toy')
                        ),
    current_zone_id     INTEGER REFERENCES zones(id) ON DELETE SET NULL,
    current_sub_slot    TEXT,                   -- JSON {"row", "col", "row_label", "col_label"} or NULL
    location_history    TEXT NOT NULL DEFAULT '[]',  -- JSON array, newest-first
    quantity            REAL,                   -- at-last-physical-check; never auto-decremented
    metadata            TEXT NOT NULL DEFAULT '{}',  -- JSON blob, category-specific
    photo_path          TEXT,                   -- data/photos/items/ITEM-{id}.jpg (Phase 12)
    notes               TEXT,
    chunk_source_id     INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- JSON integrity backstop (finding 5): DB-level guards against malformed JSON
    -- being written via a raw UPDATE/INSERT, even if the Python layer skips json.dumps.
    -- json_valid() is available — the project's sqlite-vec runtime ships JSON1.
    CHECK (json_valid(metadata)),
    CHECK (json_valid(location_history)),
    CHECK (current_sub_slot IS NULL OR json_valid(current_sub_slot))
);

CREATE INDEX idx_items_zone     ON items(current_zone_id);
CREATE INDEX idx_items_category ON items(category);
CREATE INDEX idx_items_name     ON items(name);

-- =============================================================================
-- Section 4: items_au trigger
-- =============================================================================

-- Trigger: update items.updated_at on each UPDATE (mirrors chunks_au trigger shape).
CREATE TRIGGER items_au AFTER UPDATE ON items BEGIN
    UPDATE items SET updated_at = CURRENT_TIMESTAMP WHERE id = new.id;
END;

-- =============================================================================
-- Section 5: L44 seed zones
-- =============================================================================

-- STARTER DATA ONLY — these INSERT OR IGNORE rows are the default L44 zone vocabulary
-- extracted from the Owner's Manual. Important mutability note:
--
--   Later corrections to a seed zone's label or vertical_desc in this migration
--   will NOT reach DBs that already ran migration v2.  The version guard in
--   apply_migrations() skips a re-run, and INSERT OR IGNORE skips rows whose
--   slug already exists.  To update an existing seed zone on an already-migrated
--   DB, use the app's zone-edit command — not this migration file.
--
-- These 32 zones represent the canonical named locations referenced throughout
-- the L44 Owner's Manual systems sections (gas, bilge, electrical, etc.).
-- The owner adds vessel-specific zones on top via "l44 zone add".
INSERT OR IGNORE INTO zones (name, label, side, fore_aft, area, vertical_desc) VALUES
  ('anchor-locker',
   'Anchor locker',
   'centre', 'fwd', 'foredeck',
   'Below the bow deck hatch, houses anchor chain and windlass'),
  ('port-foredeck-locker',
   'Port foredeck locker',
   'port', 'fwd', 'foredeck',
   'Port side foredeck hatch locker, houses fresh water tank (390L)'),
  ('stbd-foredeck-locker',
   'Stbd foredeck locker',
   'stbd', 'fwd', 'foredeck',
   'Stbd side foredeck hatch locker, houses fresh water tank (390L)'),
  ('port-genset-locker',
   'Port genset locker',
   'port', 'fwd', 'foredeck',
   'Foredeck locker housing the generator'),
  ('saloon',
   'Saloon',
   'centre', 'mid', 'saloon',
   'Main living and dining area amidships'),
  ('galley-locker-under-sink',
   'Galley locker under sink',
   'centre', 'mid', 'saloon',
   'Under-sink locker with windlass breaker and battery switch access'),
  ('saloon-seat-port',
   'Port saloon seat locker',
   'port', 'mid', 'saloon',
   'Under the port saloon seat, houses genset starter battery'),
  ('saloon-seat-stbd',
   'Stbd saloon seat locker',
   'stbd', 'mid', 'saloon',
   'Under the stbd saloon seat, general storage'),
  ('nav-station',
   'Nav station',
   NULL, 'mid', 'saloon',
   'Navigation station with electronics and chart storage'),
  ('cockpit-gas-locker',
   'Cockpit gas locker',
   'stbd', 'aft', 'cockpit',
   'Stbd aft cockpit locker for LPG cylinders'),
  ('cockpit-battery-locker',
   'Aft cockpit battery locker',
   'centre', 'aft', 'cockpit',
   'Aft cockpit locker housing house batteries'),
  ('cockpit-liferaft-locker',
   'Liferaft locker',
   'centre', 'aft', 'cockpit',
   'Cockpit locker for liferaft and emergency tiller'),
  ('port-engine-room',
   'Port engine room',
   'port', 'mid', 'engine-room',
   'Below saloon floor to port, accessed via floor hatch'),
  ('stbd-engine-room',
   'Stbd engine room',
   'stbd', 'mid', 'engine-room',
   'Below saloon floor to stbd, accessed via floor hatch'),
  ('port-aft-cabin',
   'Port aft cabin',
   'port', 'aft', 'aft-cabin',
   'Port hull aft cabin with berth and holding tank seacock'),
  ('stbd-aft-cabin',
   'Stbd aft cabin',
   'stbd', 'aft', 'aft-cabin',
   'Stbd hull aft cabin with berth and fridge/freezer access'),
  ('port-aft-cabin-hanging-locker',
   'Port aft cabin hanging locker',
   'port', 'aft', 'aft-cabin',
   'Hanging locker in port aft cabin, lower shelf has engine battery switch'),
  ('stbd-aft-cabin-hanging-locker',
   'Stbd aft cabin hanging locker',
   'stbd', 'aft', 'aft-cabin',
   'Hanging locker in stbd aft cabin, has engine battery switch and fluxgate compass'),
  ('port-fwd-cabin',
   'Port fwd cabin',
   'port', 'fwd', 'fwd-cabin',
   'Port hull forward cabin with V-berth'),
  ('stbd-fwd-cabin',
   'Stbd fwd cabin',
   'stbd', 'fwd', 'fwd-cabin',
   'Stbd hull forward cabin with V-berth'),
  ('port-fwd-cabin-locker',
   'Port fwd cabin locker',
   'port', 'fwd', 'fwd-cabin',
   'Storage locker in port forward cabin'),
  ('stbd-fwd-cabin-locker',
   'Stbd fwd cabin locker',
   'stbd', 'fwd', 'fwd-cabin',
   'Storage locker in stbd forward cabin'),
  ('port-hull-aft-lazarette',
   'Port hull aft lazarette',
   'port', 'aft', 'lazarette',
   'Below-deck stern stowage in port hull'),
  ('stbd-hull-aft-lazarette',
   'Stbd hull aft lazarette',
   'stbd', 'aft', 'lazarette',
   'Below-deck stern stowage in stbd hull'),
  ('bilge-port-keel-sump',
   'Port keel sump',
   'port', 'mid', 'bilge',
   'Port keel sump housing manual and electric bilge pumps'),
  ('bilge-stbd-keel-sump',
   'Stbd keel sump',
   'stbd', 'mid', 'bilge',
   'Stbd keel sump housing manual and electric bilge pumps'),
  ('port-companionway',
   'Port companionway',
   'port', 'mid', 'companionway',
   'Port hull companionway, manual bilge pump handle stowed here'),
  ('stbd-companionway',
   'Stbd companionway',
   'stbd', 'mid', 'companionway',
   'Stbd hull companionway, manual bilge pump handle stowed here'),
  ('cockpit-general',
   'Cockpit (general)',
   'centre', 'aft', 'cockpit',
   'Main cockpit area with helm, winches, and sheet leads'),
  ('deck-port',
   'Port deck',
   'port', NULL, 'deck',
   'Port side deck for sail hardware, cleats, and running rigging'),
  ('deck-stbd',
   'Stbd deck',
   'stbd', NULL, 'deck',
   'Stbd side deck for sail hardware, cleats, and running rigging'),
  ('transom',
   'Transom',
   'centre', 'aft', 'deck',
   'Transom area for davits, RIB storage, and emergency tiller deck plates');

-- =============================================================================
-- Section 6: schema_version
-- =============================================================================

INSERT INTO schema_version(version) VALUES (2);
