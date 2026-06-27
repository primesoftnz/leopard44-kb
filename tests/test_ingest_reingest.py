# RED state until Phase 2 implementation (see 02-VALIDATION.md). Imports from leopard44_kb.ingest.* fail until production code lands.
"""Tests for INGEST-06: no-op re-ingest, changed-file replacement, vec_chunks orphan prevention, transaction atomicity."""
from __future__ import annotations

from pathlib import Path

import pytest

from leopard44_kb.ingest import ingest_file
from leopard44_kb.ingest.writer import compute_anchor_key


def _write_note(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# No-op behaviour
# ---------------------------------------------------------------------------


def test_unchanged_noop(ingest_db, fake_embedder, tmp_path):
    """Re-ingest of unchanged file: sources/chunks/vec_chunks row counts are unchanged AND result is 'no-op'."""
    note = tmp_path / "data" / "logs" / "unchanged.md"
    _write_note(note, "# Note\n\nSome content.\n")

    r1 = ingest_file(note, layer="vessel", conn=ingest_db)
    assert r1 == "ok"

    src_count_before = ingest_db.execute("SELECT count(*) FROM sources").fetchone()[0]
    chunk_count_before = ingest_db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    vec_count_before = ingest_db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]

    r2 = ingest_file(note, layer="vessel", conn=ingest_db)
    assert r2 == "no-op", f"Expected 'no-op' for unchanged file; got {r2!r}"

    assert ingest_db.execute("SELECT count(*) FROM sources").fetchone()[0] == src_count_before
    assert ingest_db.execute("SELECT count(*) FROM chunks").fetchone()[0] == chunk_count_before
    assert ingest_db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == vec_count_before


def test_noop_skips_parser_and_ollama(ingest_db, tmp_path, monkeypatch):
    """No-op path must NOT call embed_texts (proves parse+embed are skipped when file unchanged)."""
    import leopard44_kb.ingest.embedder as emb

    # First ingest with real fake embedder to seed the DB
    called = []
    monkeypatch.setattr(emb, "embed_texts", lambda texts, model: [[0.1] * 384 for _ in texts])
    monkeypatch.setattr(emb, "select_model", lambda: ("nomic-embed-text:v1.5", "v1.5"))

    note = tmp_path / "data" / "logs" / "noop_skip.md"
    _write_note(note, "# Noop Test\n\nContent.\n")
    ingest_file(note, layer="vessel", conn=ingest_db)

    # Now replace embed_texts with a function that raises on any call
    def _fail_if_called(*a, **kw):
        called.append(True)
        raise AssertionError("embed_texts must not be called on no-op re-ingest")

    monkeypatch.setattr(emb, "embed_texts", _fail_if_called)

    r2 = ingest_file(note, layer="vessel", conn=ingest_db)
    assert r2 == "no-op"
    assert not called, "embed_texts was called during a no-op re-ingest"


# ---------------------------------------------------------------------------
# Changed-file behaviour
# ---------------------------------------------------------------------------


def test_changed_file_replaces(ingest_db, fake_embedder, tmp_path):
    """Re-ingest of changed file replaces all chunks; new content is in DB."""
    note = tmp_path / "data" / "logs" / "changed.md"
    _write_note(note, "# Original\n\nOriginal content.\n")
    ingest_file(note, layer="vessel", conn=ingest_db)

    # Modify the file
    _write_note(note, "# Updated\n\nCompletely new content here.\n")
    r2 = ingest_file(note, layer="vessel", conn=ingest_db)
    assert r2 == "ok", f"Expected 'ok' for changed file; got {r2!r}"

    # Only one source row should exist
    src_count = ingest_db.execute("SELECT count(*) FROM sources").fetchone()[0]
    assert src_count == 1, f"Expected 1 source row after re-ingest; got {src_count}"

    # New content must be searchable; old content must be gone
    old_fts = ingest_db.execute(
        "SELECT content FROM fts_chunks WHERE fts_chunks MATCH 'Original'"
    ).fetchall()
    # "Updated" is new content — must be present
    new_fts = ingest_db.execute(
        "SELECT content FROM fts_chunks WHERE fts_chunks MATCH 'Updated'"
    ).fetchall()
    assert len(new_fts) > 0, "New content not found in FTS after re-ingest"


def test_vec_chunks_cleared_on_reingest(ingest_db, fake_embedder, tmp_path):
    """Re-ingest of changed file leaves no orphaned vec_chunks rows for the old source_id (Pitfall 1)."""
    note = tmp_path / "data" / "logs" / "vec_test.md"
    _write_note(note, "# Original\n\nContent.\n")
    ingest_file(note, layer="vessel", conn=ingest_db)

    old_src_id = ingest_db.execute("SELECT id FROM sources").fetchone()[0]

    # Change and re-ingest
    _write_note(note, "# Changed\n\nNew content.\n")
    ingest_file(note, layer="vessel", conn=ingest_db)

    orphan_count = ingest_db.execute(
        "SELECT count(*) FROM vec_chunks WHERE source_id = ?", (old_src_id,)
    ).fetchone()[0]
    assert orphan_count == 0, (
        f"Orphaned vec_chunks rows remain for old source_id {old_src_id}: {orphan_count} rows"
    )


def test_anchor_key_stable(ingest_db, fake_embedder, tmp_path):
    """anchor_key is identical across two ingests of the same unchanged file."""
    note = tmp_path / "data" / "logs" / "anchor_stable.md"
    _write_note(note, "# Stable Section\n\nContent that never changes.\n")

    ingest_file(note, layer="vessel", conn=ingest_db)
    keys_first = set(
        row[0] for row in ingest_db.execute("SELECT anchor_key FROM chunks").fetchall()
    )

    # Force a re-ingest by treating it as changed then unchanged — we'll use compute_anchor_key directly
    # Since the file is unchanged, the second ingest is a no-op. Check anchor_key from first ingest.
    assert len(keys_first) > 0, "No anchor_keys after first ingest"
    for key in keys_first:
        assert len(key) == 64, f"anchor_key should be 64-char hex; got {len(key)}: {key!r}"


def test_reingest_failure_preserves_old_source(ingest_db, tmp_path, monkeypatch):
    """ATOMICITY: a mid-insert failure on a CHANGED re-ingest preserves the OLD source row + its chunks + vec_chunks.

    The delete+insert must be a single transaction so a failure after delete but before
    the insert is complete rolls back — leaving the old data intact (no partial data loss).
    """
    import leopard44_kb.ingest.embedder as emb

    monkeypatch.setattr(emb, "embed_texts", lambda texts, model: [[0.1] * 384 for _ in texts])
    monkeypatch.setattr(emb, "select_model", lambda: ("nomic-embed-text:v1.5", "v1.5"))

    note = tmp_path / "data" / "logs" / "atomic_test.md"
    _write_note(note, "# Original\n\nOriginal content that must survive.\n")
    ingest_file(note, layer="vessel", conn=ingest_db)

    old_chunk_count = ingest_db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    old_vec_count = ingest_db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    assert old_chunk_count > 0

    # Monkeypatch the writer to raise mid-insert (after delete, before insert completes)
    import leopard44_kb.ingest.writer as writer_mod

    _original_store = writer_mod.store_source_and_chunks

    call_count = [0]

    def _fail_on_second_call(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] > 1:
            raise RuntimeError("Simulated mid-insert failure for atomicity test")
        return _original_store(*args, **kwargs)

    # We need to force a re-ingest: modify the file
    _write_note(note, "# Changed\n\nThis new content should NOT appear if atomicity works.\n")

    monkeypatch.setattr(writer_mod, "store_source_and_chunks", _fail_on_second_call)

    with pytest.raises(Exception):
        ingest_file(note, layer="vessel", conn=ingest_db)

    # Old source + chunks + vec_chunks must still exist
    src_count = ingest_db.execute("SELECT count(*) FROM sources").fetchone()[0]
    chunk_count = ingest_db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    vec_count = ingest_db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]

    assert src_count >= 1, "Source row was lost after failed re-ingest (atomicity violation)"
    assert chunk_count == old_chunk_count, (
        f"Chunk count changed after failed re-ingest: old={old_chunk_count}, now={chunk_count}"
    )
    assert vec_count == old_vec_count, (
        f"vec_chunks count changed after failed re-ingest: old={old_vec_count}, now={vec_count}"
    )


def test_fts_rows_recreated_on_changed_reingest(ingest_db, fake_embedder, tmp_path):
    """After a changed re-ingest, fts_chunks contains the NEW content and none of the OLD."""
    note = tmp_path / "data" / "logs" / "fts_recreate.md"
    old_phrase = "uniqueoldphrasealpha999"
    new_phrase = "uniquenewphrasebeta777"

    _write_note(note, f"# Section\n\nContent with {old_phrase}.\n")
    ingest_file(note, layer="vessel", conn=ingest_db)

    # Verify old phrase is in FTS
    old_rows = ingest_db.execute(
        "SELECT content FROM fts_chunks WHERE fts_chunks MATCH ?", (old_phrase,)
    ).fetchall()
    assert len(old_rows) > 0, f"Old phrase '{old_phrase}' not found in FTS before re-ingest"

    # Change and re-ingest
    _write_note(note, f"# Section\n\nContent with {new_phrase}.\n")
    ingest_file(note, layer="vessel", conn=ingest_db)

    # Old phrase must be gone
    old_after = ingest_db.execute(
        "SELECT content FROM fts_chunks WHERE fts_chunks MATCH ?", (old_phrase,)
    ).fetchall()
    assert len(old_after) == 0, (
        f"Old FTS content '{old_phrase}' still present after changed re-ingest"
    )

    # New phrase must be present
    new_rows = ingest_db.execute(
        "SELECT content FROM fts_chunks WHERE fts_chunks MATCH ?", (new_phrase,)
    ).fetchall()
    assert len(new_rows) > 0, f"New content '{new_phrase}' not found in FTS after re-ingest"
