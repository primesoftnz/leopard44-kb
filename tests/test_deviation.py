"""RED tests for DEV-01: DeviationExtraction model, create_deviation dual-write,
zone_id FK enforcement, and deterministic unified retrieval (DEV-02 criterion 2).

Per-requirement verification map source:
  .planning/phases/11-factory-deviation-log/11-VALIDATION.md

Nyquist discipline: leopard44_kb.deviation is NEVER imported at module top.
Every test that needs the module body-imports it so:
  - Collection always succeeds (no ERROR collecting this file)
  - Tests FAIL (ModuleNotFoundError) until 11-02 creates deviation.py
  - Exception: assertions (f) and (g) are GREEN after Task 1 (004 migration exists,
    raw DB seed + PRAGMA foreign_keys=ON need no deviation module)
"""
from __future__ import annotations

import sqlite3
import struct

import pytest

from tests._corpus import pack


# ---------------------------------------------------------------------------
# (a) DeviationExtraction validates required component
# ---------------------------------------------------------------------------


def test_deviation_extraction_requires_component():
    """DeviationExtraction raises ValidationError when component is absent or empty.

    Goes GREEN when 11-02 creates leopard44_kb.deviation with the DeviationExtraction model.
    """
    import leopard44_kb.deviation as dev  # body-import: FAILS with ModuleNotFoundError until 11-02

    with pytest.raises(Exception):  # pydantic ValidationError or ValueError
        dev.DeviationExtraction(component="")  # empty component must be rejected


# ---------------------------------------------------------------------------
# (b) extract_fields returns a DeviationExtraction
# ---------------------------------------------------------------------------


def test_extract_fields_returns_deviation_extraction(monkeypatch):
    """extract_fields returns a DeviationExtraction when call_extract_json returns valid dict.

    Goes GREEN when 11-02 creates leopard44_kb.deviation with extract_fields + call_extract_json seam.
    """
    import leopard44_kb.deviation as dev  # body-import: FAILS with ModuleNotFoundError until 11-02

    raw = {
        "component": "windlass",
        "factory_spec": "12V Muir 1200W",
        "as_built": "12V Maxwell 1000W",
        "reason": "replacement after failure",
        "date_noted": "2024-01-10",
    }
    monkeypatch.setattr(dev, "call_extract_json", lambda prompt, sys_prompt: raw)

    result = dev.extract_fields("windlass replaced with Maxwell 1000W")
    assert result.component == "windlass", f"Expected component='windlass', got {result.component!r}"
    assert result.factory_spec == "12V Muir 1200W", (
        f"Expected factory_spec='12V Muir 1200W', got {result.factory_spec!r}"
    )
    assert result.as_built == "12V Maxwell 1000W", (
        f"Expected as_built='12V Maxwell 1000W', got {result.as_built!r}"
    )


# ---------------------------------------------------------------------------
# (c) create_deviation dual-write: deviations row + vessel-layer sources chunk
# ---------------------------------------------------------------------------


def test_create_deviation_dual_write(fake_embedder, ingest_db, tmp_path):
    """create_deviation inserts a deviations row AND a vessel-layer chunk (source_type='deviation').

    Asserts:
      - A deviations row exists with the correct component
      - A sources row with layer='vessel', source_type='deviation' exists
      - The sources row content embeds component and factory_spec

    Goes GREEN when 11-02 creates leopard44_kb.deviation.create_deviation.
    """
    import leopard44_kb.deviation as dev  # body-import: FAILS with ModuleNotFoundError until 11-02

    deviation_id = dev.create_deviation(
        ingest_db,
        component="windlass",
        factory_spec="12V Muir 1200W",
        as_built="12V Maxwell 1000W",
        reason="replacement after failure",
        date_noted="2024-01-10",
        repo_root=tmp_path,
    )
    assert isinstance(deviation_id, int), f"Expected int deviation_id, got {type(deviation_id)}"

    row = ingest_db.execute(
        "SELECT * FROM deviations WHERE id = ?", (deviation_id,)
    ).fetchone()
    assert row is not None, f"Expected a deviations row for id={deviation_id}"
    assert row["component"] == "windlass", (
        f"Expected component='windlass', got {row['component']!r}"
    )

    # Vessel-layer chunk: a sources row with layer='vessel', source_type='deviation'
    src_row = ingest_db.execute(
        "SELECT * FROM sources WHERE layer='vessel' AND source_type='deviation'"
    ).fetchone()
    assert src_row is not None, (
        "Expected a sources row with layer='vessel', source_type='deviation' after create_deviation"
    )
    # Content should embed component and/or factory_spec
    assert "windlass" in (src_row["title"] or "") or "windlass" in (src_row["path"] or ""), (
        f"Expected 'windlass' to appear in sources title or path; got title={src_row['title']!r}, "
        f"path={src_row['path']!r}"
    )


# ---------------------------------------------------------------------------
# (d) create_deviation with a non-existent zone_id raises ValueError
# ---------------------------------------------------------------------------


def test_create_deviation_rejects_invalid_zone_id(fake_embedder, ingest_db, tmp_path):
    """create_deviation with a non-existent zone_id raises ValueError (app-layer guard).

    Goes GREEN when 11-02 creates leopard44_kb.deviation.create_deviation with zone validation.
    """
    import leopard44_kb.deviation as dev  # body-import: FAILS with ModuleNotFoundError until 11-02

    with pytest.raises(ValueError):
        dev.create_deviation(
            ingest_db,
            component="engine",
            zone_id=999999,  # bogus FK — must be rejected before DB insert
            repo_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# (e) create_deviation with zone_id=None succeeds (zone is optional)
# ---------------------------------------------------------------------------


def test_create_deviation_zone_optional(fake_embedder, ingest_db, tmp_path):
    """create_deviation with zone_id=None succeeds — zone is optional (NULL allowed).

    Goes GREEN when 11-02 creates leopard44_kb.deviation.create_deviation.
    """
    import leopard44_kb.deviation as dev  # body-import: FAILS with ModuleNotFoundError until 11-02

    deviation_id = dev.create_deviation(
        ingest_db,
        component="anchor chain",
        factory_spec="6mm G4 50m",
        as_built="8mm G4 55m",
        zone_id=None,
        repo_root=tmp_path,
    )
    assert isinstance(deviation_id, int), (
        f"Expected int deviation_id with zone_id=None, got {type(deviation_id)}"
    )
    row = ingest_db.execute(
        "SELECT zone_id FROM deviations WHERE id = ?", (deviation_id,)
    ).fetchone()
    assert row is not None, f"Expected deviations row for id={deviation_id}"
    assert row["zone_id"] is None, (
        f"Expected zone_id=None in deviations row, got {row['zone_id']!r}"
    )


# ---------------------------------------------------------------------------
# (f) FK enforcement at the DB layer — GREEN after Task 1 (004 migration exists)
#
# This test needs NO deviation.py — it exercises the DB schema directly.
# It is GREEN as soon as 004_deviations.sql is applied and PRAGMA foreign_keys=ON
# is active (ingest_db does this via store.py's open_db() sequence).
# ---------------------------------------------------------------------------


def test_deviations_zone_fk_enforced_at_db_layer(ingest_db):
    """A raw INSERT with a bogus zone_id raises sqlite3.IntegrityError under PRAGMA foreign_keys=ON.

    Proves the FK is ENFORCED at the DB layer (not merely declared) — GREEN after Task 1.
    No deviation.py is required; this test only exercises the 004 schema.
    """
    with pytest.raises(sqlite3.IntegrityError):
        ingest_db.execute(
            "INSERT INTO deviations(component, zone_id) VALUES ('test-component', 999999)"
        )


# ---------------------------------------------------------------------------
# (g) DETERMINISTIC UNIFIED RETRIEVAL — GREEN after Task 1 (004 migration exists)
#
# Proves that a deviation chunk AND a maintenance_entry chunk BOTH surface in one
# retrieve() call. Uses fake_embedder to neutralise KNN nondeterminism.
# Seeds raw sources+chunks+vec_chunks ONLY (FTS via chunks_ai trigger, no manual
# fts_chunks insert — mirrors _corpus.seed_corpus exactly).
# GREEN immediately after Task 1 because: (1) the deviations source_type is a free
# TEXT column on sources (no schema constraint blocks it), (2) the retrieve() pipeline
# already handles mixed source_type rows, (3) no deviation.py is required.
# ---------------------------------------------------------------------------


def test_deterministic_unified_retrieval(fake_embedder, ingest_db):
    """A deviation chunk AND a maintenance_entry chunk both surface in one retrieve().

    Seed: two vessel chunks both containing the nonce "ZZQNONCE42" — one with
    source_type='deviation', one with source_type='maintenance_entry'. Uses
    fake_embedder so every vector is [0.1]*384 (KNN neutral). Call retrieve()
    with generous n/pool. Resolve source_type via the chunks⋈sources JOIN.
    Assert BOTH source_types appear in the results.

    GREEN after Task 1. Does NOT require deviation.py (raw INSERT seeds the DB directly).
    Grep contract: does NOT insert into fts_chunks (chunks_ai trigger handles FTS).
    """
    from leopard44_kb.retrieve import retrieve

    conn = ingest_db

    # --- Raw seed: sources → chunks → vec_chunks ONLY (no fts_chunks insert) ---
    # Source 1: deviation layer
    conn.execute(
        "INSERT INTO sources(id, layer, source_type, path, content_hash, title) "
        "VALUES (101, 'vessel', 'deviation', 'data/deviations/DEV-1.md', 'hdev1', 'Windlass deviation')"
    )
    # Source 2: maintenance_entry layer
    conn.execute(
        "INSERT INTO sources(id, layer, source_type, path, content_hash, title) "
        "VALUES (102, 'vessel', 'maintenance_entry', 'data/logs/maint/2024-01-10-windlass.md', 'hmaint1', 'Windlass maintenance')"
    )

    # Chunk 1: deviation — content contains the unique nonce ZZQNONCE42
    conn.execute(
        "INSERT INTO chunks(id, source_id, layer, ordinal, section_path, page_start, page_end, "
        "content, content_hash, anchor_key, embedding_model, embedding_model_version) "
        "VALUES (101, 101, 'vessel', 0, 'Deviations', 0, 0, "
        "'ZZQNONCE42 windlass replaced with Maxwell 1000W instead of factory Muir 1200W', "
        "'hdc1', 'akdev1', 'm', 'v')"
    )
    # Chunk 2: maintenance_entry — content also contains the unique nonce ZZQNONCE42
    conn.execute(
        "INSERT INTO chunks(id, source_id, layer, ordinal, section_path, page_start, page_end, "
        "content, content_hash, anchor_key, embedding_model, embedding_model_version) "
        "VALUES (102, 102, 'vessel', 0, 'Maintenance', 0, 0, "
        "'ZZQNONCE42 windlass serviced 2024-01-10 anchor chain', "
        "'hmc1', 'akmaint1', 'm', 'v')"
    )

    # vec_chunks: both in the fake_embedder [0.1]*384 direction (KNN neutral)
    # NO INSERT INTO fts_chunks — the chunks_ai trigger populates FTS automatically
    for cid, src_id in [(101, 101), (102, 102)]:
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id, layer, source_id, embedding_model, is_active, embedding) "
            "VALUES (?, 'vessel', ?, 'm', 1, ?)",
            (cid, src_id, pack([0.1] * 384)),
        )
    conn.commit()

    # Call retrieve with generous n/pool so source-diversity cap cannot drop either row
    chunks, is_below_floor = retrieve(conn, "ZZQNONCE42", ["vessel"], n=20, pool=50)
    assert not is_below_floor, "Expected is_below_floor=False for the nonce term"
    assert len(chunks) >= 2, f"Expected at least 2 chunks for nonce ZZQNONCE42, got {len(chunks)}"

    # Resolve source_type via the chunks⋈sources JOIN (same as _apply_source_diversity_cap)
    chunk_ids = tuple(c["id"] for c in chunks)
    rows = conn.execute(
        f"SELECT c.id, s.source_type FROM chunks c JOIN sources s ON s.id = c.source_id "
        f"WHERE c.id IN ({','.join('?' for _ in chunk_ids)})",
        chunk_ids,
    ).fetchall()
    returned_source_types = {r["source_type"] for r in rows}

    assert "deviation" in returned_source_types, (
        f"Expected source_type='deviation' in results; got source_types={returned_source_types!r}"
    )
    assert "maintenance_entry" in returned_source_types, (
        f"Expected source_type='maintenance_entry' in results; got source_types={returned_source_types!r}"
    )


# ---------------------------------------------------------------------------
# Re-ingest safety (code-review HIGH): a broad `l44 ingest data/` must not
# downgrade a generated deviation file to plain markdown and drop its metadata.
# ---------------------------------------------------------------------------


def test_detect_source_type_recognizes_deviations_tree():
    """Files under data/deviations/ are detected as 'deviation', not 'markdown'."""
    from pathlib import Path

    from leopard44_kb.ingest import _detect_source_type

    assert _detect_source_type(Path("data/deviations/DEV-1.md")) == "deviation"
    # An unrelated .md outside the canonical tree stays markdown.
    assert _detect_source_type(Path("data/notes/deviations-summary.md")) == "markdown"


def test_parse_deviation_entry_extracts_metadata(tmp_path):
    """parse_deviation_entry pulls deviation_id + zone_id from front-matter into chunk metadata."""
    from leopard44_kb.ingest.text_md import parse_deviation_entry

    f = tmp_path / "DEV-7.md"
    f.write_text(
        "---\ndeviation_id: 7\nzone_id: 3\nsource_type: deviation\nlayer: vessel\n---\n\n"
        "Windlass replaced with a Maxwell 1000W.\n",
        encoding="utf-8",
    )
    chunk = parse_deviation_entry(f, str(f))[0]
    assert chunk["metadata"]["deviation_id"] == 7
    assert chunk["metadata"]["zone_id"] == 3
    assert "Windlass" in chunk["content"]

    # zone_id: null in the front-matter coerces to None (location unknown).
    f2 = tmp_path / "DEV-8.md"
    f2.write_text(
        "---\ndeviation_id: 8\nzone_id: null\nsource_type: deviation\nlayer: vessel\n---\n\nNo zone.\n",
        encoding="utf-8",
    )
    meta2 = parse_deviation_entry(f2, str(f2))[0]["metadata"]
    assert meta2["deviation_id"] == 8
    assert meta2["zone_id"] is None


def test_deviation_reingest_preserves_type_and_metadata(
    monkeypatch, fake_embedder, ingest_db, tmp_path
):
    """`l44 ingest data/` re-ingesting DEV-{id}.md must NOT downgrade it (code-review HIGH).

    source_type stays 'deviation', the deviation_id chunk metadata (which drives the blue
    highlight) survives, and — because create_deviation now hashes the actual file — an
    unchanged re-ingest is a no-op rather than a destructive re-store.
    """
    import json

    import leopard44_kb.deviation as dev
    from leopard44_kb.ingest import ingest_file

    deviation_id = dev.create_deviation(
        ingest_db,
        component="windlass",
        factory_spec="12V Muir 1200W",
        as_built="12V Maxwell 1000W",
        repo_root=tmp_path,
    )
    rel = f"data/deviations/DEV-{deviation_id}.md"
    assert (tmp_path / rel).exists()

    # Replicate `l44 ingest data/` from the repo root: relative path, as ingest_cmd stores it.
    monkeypatch.chdir(tmp_path)
    result = ingest_file(rel, layer="vessel", conn=ingest_db)
    assert result == "no-op", f"Expected an unchanged re-ingest to no-op; got {result!r}"

    src = ingest_db.execute(
        "SELECT source_type FROM sources WHERE layer='vessel' AND path=?", (rel,)
    ).fetchone()
    assert src is not None and src["source_type"] == "deviation", (
        f"Re-ingest downgraded the deviation source_type; got {src and src['source_type']!r}"
    )

    meta_row = ingest_db.execute(
        "SELECT c.metadata FROM chunks c JOIN sources s ON s.id = c.source_id "
        "WHERE s.path = ? AND s.layer = 'vessel'",
        (rel,),
    ).fetchone()
    assert meta_row is not None and meta_row["metadata"] is not None
    assert json.loads(meta_row["metadata"]).get("deviation_id") == deviation_id


def test_deviation_files_are_vessel_only_on_ingest(
    monkeypatch, fake_embedder, ingest_db, tmp_path
):
    """Deviation entries are owner-private — ingest must refuse a non-vessel layer."""
    import leopard44_kb.deviation as dev
    from leopard44_kb.ingest import ingest_file

    deviation_id = dev.create_deviation(ingest_db, component="windlass", repo_root=tmp_path)
    rel = f"data/deviations/DEV-{deviation_id}.md"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="vessel-only"):
        ingest_file(rel, layer="shared", conn=ingest_db)


# ---------------------------------------------------------------------------
# D-16 regression tests (Phase 13, Plan 02)
# Items 1–4: LOW-severity edge cases from the Phase 11 cross-AI code review.
# Items 1/3/4 are RED before Task 2 fixes; item 2 asserts an already-valid invariant.
# ---------------------------------------------------------------------------


class _BacklinkFailConn:
    """SQLite connection wrapper that raises on the chunk_source_id UPDATE (Step 8).

    Used to test item 1 atomicity: if the back-link UPDATE fails, the whole
    create_deviation should roll back (deviations row + md file + source + chunks).
    """

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)

    def execute(self, sql: str, parameters: tuple = ()) -> object:
        if "UPDATE deviations SET chunk_source_id" in sql:
            raise sqlite3.OperationalError("simulated Step-8 chunk_source_id failure")
        return self._real.execute(sql, parameters)

    def __enter__(self) -> "_BacklinkFailConn":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        if exc_type is None:
            self._real.commit()
        else:
            self._real.rollback()
        return False


def test_create_deviation_atomic_on_backlink_failure(fake_embedder, ingest_db, tmp_path):
    """Item 1: Step-8 chunk_source_id failure rolls back deviations row, md file, and chunks.

    RED until Item 1 fix: Step 8 moved inside the try/except cleanup block, and cleanup
    extended to also delete the source row (cascade-deleting its chunks).
    """
    import leopard44_kb.deviation as dev

    wrapped = _BacklinkFailConn(ingest_db)

    with pytest.raises(sqlite3.OperationalError, match="chunk_source_id"):
        dev.create_deviation(
            wrapped,
            component="windlass",
            factory_spec="12V Muir 1200W",
            repo_root=tmp_path,
        )

    # No orphan deviations row.
    row = ingest_db.execute(
        "SELECT id FROM deviations WHERE component = 'windlass'"
    ).fetchone()
    assert row is None, (
        f"Orphan deviations row found after Step-8 failure; dict={dict(row)!r}"
    )

    # No orphan md file.
    dev_dir = tmp_path / "data" / "deviations"
    md_files = list(dev_dir.glob("DEV-*.md")) if dev_dir.exists() else []
    assert len(md_files) == 0, f"Orphan md file(s) left after Step-8 failure: {md_files}"

    # No orphan source row or chunks.
    src = ingest_db.execute(
        "SELECT id FROM sources WHERE source_type = 'deviation'"
    ).fetchone()
    assert src is None, (
        f"Orphan source row found after Step-8 failure: id={src['id'] if src else None}"
    )


def test_detect_source_type_deviation_fails_closed_on_any_path(tmp_path):
    """WR-01: deviation detection fails CLOSED for ANY data/deviations/ path.

    A deviation file can legitimately live outside the package _REPO_ROOT (a second
    clone, a relocated/symlinked data/ dir — exactly what ingest_file's repo-root walk
    supports). The detector must classify it as 'deviation' so the vessel-only layer
    guard fires and PRIVATE deviation content cannot be ingested as --layer community.
    This supersedes the earlier _REPO_ROOT-anchored narrowing, which returned 'markdown'
    for such paths and opened a privacy hole. Detection mirrors the maintenance_entry
    rule (consecutive data/<kind> parts anywhere in the path).
    """
    from pathlib import Path

    from leopard44_kb.ingest import _detect_source_type

    # An absolute path under tmp_path (NOT the real repo root) with data/deviations parts
    # must STILL be detected as 'deviation' (fail closed → vessel-only).
    non_repo_path = tmp_path / "data" / "deviations" / "DEV-1.md"
    result = _detect_source_type(non_repo_path)
    assert result == "deviation", (
        f"Expected 'deviation' (fail-closed) for any data/deviations path {non_repo_path}; "
        f"got {result!r}. The vessel-only layer guard must fire regardless of repo root."
    )

    # A plain relative path data/deviations/... is also 'deviation' (normal CLI usage).
    assert _detect_source_type(Path("data/deviations/DEV-1.md")) == "deviation", (
        "Relative path data/deviations/DEV-1.md must return 'deviation'"
    )

    # An unrelated .md NOT under a consecutive data/deviations sequence stays 'markdown'.
    assert _detect_source_type(Path("data/notes/deviations-summary.md")) == "markdown", (
        "data/notes/deviations-summary.md has no consecutive data/deviations parts → markdown"
    )


def test_create_deviation_file_body_matches_chunk_content(fake_embedder, ingest_db, tmp_path):
    """Item 3: The written md file body matches the embedded chunk content shape.

    RED until Item 3 fix: create_deviation must write the [Deviation] descriptor as the
    file body (instead of original_text) so parse_deviation_entry returns the same shape.
    Currently when original_text is supplied, the file body differs from the descriptor.
    """
    import leopard44_kb.deviation as dev
    from leopard44_kb.ingest.text_md import parse_deviation_entry

    deviation_id = dev.create_deviation(
        ingest_db,
        component="windlass",
        factory_spec="12V Muir 1200W",
        as_built="12V Maxwell 1000W",
        # Supply original_text to expose the descriptor-vs-body mismatch.
        original_text="The windlass was swapped out during the 2022 refit.",
        repo_root=tmp_path,
    )

    # Chunk content embedded by create_deviation (the [Deviation] descriptor).
    row = ingest_db.execute(
        "SELECT c.content FROM chunks c JOIN sources s ON s.id = c.source_id "
        "WHERE s.source_type = 'deviation' AND s.layer = 'vessel'"
    ).fetchone()
    assert row is not None, "No deviation chunk found in DB after create_deviation"
    embedded_content = row["content"]

    # Chunk content returned by re-parsing the written file.
    md_path = tmp_path / "data" / "deviations" / f"DEV-{deviation_id}.md"
    assert md_path.exists(), f"Deviation md file not found: {md_path}"
    reparsed = parse_deviation_entry(md_path, str(md_path))
    assert len(reparsed) == 1
    reparsed_content = reparsed[0]["content"]

    # They must match so a re-ingest produces the same chunk content.
    assert embedded_content == reparsed_content, (
        f"Content mismatch — embedded: {embedded_content!r} vs re-parsed: {reparsed_content!r}. "
        "The file body must match the descriptor so re-ingest is self-consistent."
    )


def test_deviation_reingest_preserves_highlight_via_metadata(
    monkeypatch, fake_embedder, ingest_db, tmp_path
):
    """Item 2: Blue highlight still resolves after a changed-file re-ingest.

    After re-ingest, store_source_and_chunks deletes+reinserts the source, causing
    ON DELETE SET NULL to null deviations.chunk_source_id (a known stale field).
    The blue zone highlight MUST still resolve via chunks.metadata.deviation_id, which
    parse_deviation_entry restores from the front-matter on re-ingest.

    This test asserts the invariant that drives blue-highlight resolution — it is expected
    to be GREEN even before the other D-16 fixes (the invariant already holds).
    """
    import json

    import leopard44_kb.deviation as dev
    from leopard44_kb.ingest import ingest_file

    deviation_id = dev.create_deviation(
        ingest_db,
        component="engine-raw-water-pump",
        factory_spec="Yanmar OEM",
        as_built="Jabsco aftermarket",
        repo_root=tmp_path,
    )

    rel = f"data/deviations/DEV-{deviation_id}.md"
    md_path = tmp_path / rel

    # Modify the file body so re-ingest sees a changed hash (triggers re-store).
    original_text = md_path.read_text(encoding="utf-8")
    md_path.write_text(original_text + "\n\n(Updated note added.)\n", encoding="utf-8")

    # Re-ingest the changed file.
    monkeypatch.chdir(tmp_path)
    result = ingest_file(rel, layer="vessel", conn=ingest_db)
    assert result == "ok", f"Expected re-ingest of changed file to return 'ok'; got {result!r}"

    # Blue highlight must still resolve via chunks.metadata.deviation_id.
    meta_row = ingest_db.execute(
        "SELECT c.metadata FROM chunks c JOIN sources s ON s.id = c.source_id "
        "WHERE s.path = ? AND s.layer = 'vessel'",
        (rel,),
    ).fetchone()
    assert meta_row is not None, "No chunk found for re-ingested deviation file"
    meta = json.loads(meta_row["metadata"])
    assert meta.get("deviation_id") == deviation_id, (
        f"Expected deviation_id={deviation_id} in chunk metadata after re-ingest; got {meta!r}"
    )

    # Acknowledge chunk_source_id may be NULL (known stale field after re-store).
    # The highlight invariant above (metadata.deviation_id) is what matters for correctness.
    dev_row = ingest_db.execute(
        "SELECT chunk_source_id FROM deviations WHERE id = ?", (deviation_id,)
    ).fetchone()
    assert dev_row is not None, f"Deviations row missing for id={deviation_id}"
    # chunk_source_id == NULL after re-store is documented acceptable behaviour.
    # No assertion on its value — the metadata invariant above is authoritative.
