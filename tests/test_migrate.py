"""Tests for leopard44_kb.migrate (Plan 13-03, D-15 re-layer migration).

PROBE (test_probe_vec_chunks_partition_key_update):
    Empirically verifies that sqlite-vec 0.1.9 does NOT support in-place UPDATE of a
    partition-key column (vec0 partition key is immutable). The probe also confirms that
    DELETE + re-INSERT correctly moves a row to the new partition. This determines
    which internal path migrate.py uses: DELETE + re-INSERT (not in-place UPDATE).
    This test PASSES in both RED and GREEN state — it documents a platform fact.

RED tests (test_whatsapp_vessel_candidates, test_relayer_sources_to_community,
           test_relayer_idempotency, test_safe_default_no_auto_promote,
           test_rollback_on_vec_failure):
    All import from leopard44_kb.migrate, which does not exist until Task 2.
    They fail with ImportError/ModuleNotFoundError in RED state and pass in GREEN.
"""
from __future__ import annotations

import sqlite3
import struct

import pytest
import sqlite_vec

from leopard44_kb.schema import apply_migrations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pack(v: list) -> bytes:
    """Pack a 384-float list into sqlite-vec binary format (little-endian f32)."""
    return struct.pack("384f", *v)


def _make_conn() -> sqlite3.Connection:
    """Return an in-memory connection with sqlite-vec loaded and schema migrated."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    return conn


def _seed_wa_source(conn: sqlite3.Connection) -> tuple[int, int]:
    """Insert one whatsapp/vessel source + chunk + vec_chunks row.

    Returns (source_id, chunk_id). Uses explicit IDs (10) so tests can reference
    them without querying the DB.
    """
    with conn:
        conn.execute(
            "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
            "VALUES (10,'vessel','whatsapp','data/whatsapp/owners.txt','hwwa','Owners WhatsApp')"
        )
        conn.execute(
            "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,"
            "content,content_hash,anchor_key,embedding_model,embedding_model_version) "
            "VALUES (10,10,'vessel',0,'Chat',"
            "'L44 length is about 43 feet','hwwc','ak10','m','v')"
        )
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
            "VALUES (10,'vessel',10,'m',1,?)",
            (_pack([0.2] * 384),),
        )
    return 10, 10


def _knn_ids(conn: sqlite3.Connection, layer: str, query_vec: list) -> list[int]:
    """Return chunk_ids from a KNN query scoped to the given layer."""
    rows = conn.execute(
        "SELECT chunk_id FROM vec_chunks WHERE embedding MATCH ? "
        "AND layer=? AND is_active=1 AND k=5",
        (_pack(query_vec), layer),
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# PROBE: is sqlite-vec partition-key UPDATE supported? (always passes)
# ---------------------------------------------------------------------------


def test_probe_vec_chunks_partition_key_update():
    """PROBE: confirm sqlite-vec partition-key immutability and DELETE+re-INSERT correctness.

    DOCUMENTED FINDING (2026-06-27, sqlite-vec 0.1.9):
    -------------------------------------------------------
    1. In-place UPDATE of vec_chunks.layer does NOT move the row to the new partition.
       After `UPDATE vec_chunks SET layer='community' WHERE chunk_id=10`, the row
       still appears only in the 'vessel' partition for KNN queries (or UPDATE raises).
       Conclusion: partition keys are IMMUTABLE after insert.

    2. DELETE + re-INSERT with the new layer value DOES correctly move the row.
       After DELETE + INSERT with layer='community', the row appears in 'community'
       KNN and is absent from 'vessel' KNN.
       Conclusion: DELETE+re-INSERT is the correct migration path.

    CONSEQUENCE: migrate.py relayer_sources_to_community() MUST use the
    DELETE+re-INSERT approach (with vectors held in a TEMP TABLE to avoid loading
    3k+ vectors into Python memory during the transaction).

    This test ALWAYS PASSES — it documents platform facts and enforces that the
    expected behavior has not changed in the installed sqlite-vec version.
    """
    conn = _make_conn()
    _seed_wa_source(conn)

    query_vec = [0.2] * 384

    # Pre-condition: row exists in vessel partition
    assert 10 in _knn_ids(conn, "vessel", query_vec), (
        "Precondition failed: chunk_id=10 must be in 'vessel' partition before probe"
    )

    # ---- FINDING 1: in-place UPDATE does NOT move the row ----
    try:
        conn.execute("UPDATE vec_chunks SET layer='community' WHERE chunk_id=10")
        conn.commit()
        update_raised = False
    except sqlite3.OperationalError:
        update_raised = True

    # The critical assertion: after UPDATE attempt, chunk_id=10 must NOT appear in
    # 'community' KNN. This confirms the partition key is immutable and that migrate.py
    # MUST use DELETE+re-INSERT (not in-place UPDATE).
    community_ids_after_update = _knn_ids(conn, "community", query_vec)
    assert 10 not in community_ids_after_update, (
        f"PROBE: chunk_id=10 unexpectedly appeared in 'community' KNN after UPDATE "
        f"(update_raised={update_raised}). "
        "In-place partition-key UPDATE may be supported in this sqlite-vec version. "
        "Review migrate.py implementation — UPDATE path may be viable."
    )

    # ---- FINDING 2: DELETE + re-INSERT DOES move the row ----
    # Snapshot the embedding first (simulating what the TEMP TABLE does)
    row = conn.execute(
        "SELECT chunk_id, source_id, embedding_model, is_active, embedding "
        "FROM vec_chunks WHERE chunk_id=10 AND layer='vessel'"
    ).fetchone()

    # Note: after UPDATE (which no-oped), the row is still in 'vessel' partition.
    # If UPDATE raised, row is also still in 'vessel'. In both cases we can proceed.
    if row is None:
        # Update somehow deleted the row — re-seed for the re-insert test
        _seed_wa_source(conn)
        row = conn.execute(
            "SELECT chunk_id, source_id, embedding_model, is_active, embedding "
            "FROM vec_chunks WHERE chunk_id=10 AND layer='vessel'"
        ).fetchone()

    assert row is not None, "Precondition: vessel row must exist for DELETE+re-INSERT test"

    with conn:
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id=10 AND layer='vessel'")
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id, layer, source_id, embedding_model, is_active, embedding) "
            "VALUES (?, 'community', ?, ?, ?, ?)",
            (row["chunk_id"], row["source_id"], row["embedding_model"], row["is_active"], row["embedding"]),
        )

    # After DELETE+re-INSERT: row MUST appear in 'community' and NOT in 'vessel'
    community_ids_after_reinsert = _knn_ids(conn, "community", query_vec)
    vessel_ids_after_reinsert = _knn_ids(conn, "vessel", query_vec)

    assert 10 in community_ids_after_reinsert, (
        "PROBE FINDING 2 FAILED: chunk_id=10 did NOT appear in 'community' KNN after "
        "DELETE+re-INSERT. The migration mechanism itself is broken — investigate sqlite-vec."
    )
    assert 10 not in vessel_ids_after_reinsert, (
        "PROBE FINDING 2 FAILED: chunk_id=10 still appears in 'vessel' KNN after "
        "DELETE+re-INSERT. Row was not properly removed from the old partition."
    )

    conn.close()


# ---------------------------------------------------------------------------
# RED: whatsapp_vessel_candidates
# ---------------------------------------------------------------------------


def test_relayer_whatsapp_vessel_candidates():
    """RED: whatsapp_vessel_candidates lists whatsapp/vessel sources with chunk counts.

    Fails with ImportError until Task 2 creates migrate.py.
    """
    conn = _make_conn()
    _seed_wa_source(conn)

    from leopard44_kb.migrate import whatsapp_vessel_candidates  # noqa: PLC0415 — lazy import (RED until Task 2)

    candidates = whatsapp_vessel_candidates(conn)
    assert len(candidates) == 1, f"Expected 1 candidate, got {candidates}"
    cand = candidates[0]
    assert cand["source_id"] == 10, f"Expected source_id=10, got {cand}"
    assert cand["chunk_count"] == 1, f"Expected chunk_count=1, got {cand}"
    assert "path" in cand, "Candidate dict must include 'path'"

    conn.close()


# ---------------------------------------------------------------------------
# RED: relayer_sources_to_community — re-layers all three tables
# ---------------------------------------------------------------------------


def test_relayer_sources_to_community():
    """RED: re-layering a whatsapp/vessel source moves sources, chunks, AND vec_chunks.

    Fails with ImportError until Task 2 creates migrate.py.
    After migration:
    - sources.layer == 'community'
    - chunks.layer == 'community'
    - KNN query scoped to layer='community' finds the chunk
    - KNN query scoped to layer='vessel' does NOT find the chunk
    """
    conn = _make_conn()
    _seed_wa_source(conn)

    from leopard44_kb.migrate import relayer_sources_to_community  # noqa: PLC0415

    moved = relayer_sources_to_community(conn, [10])
    assert moved == 1, f"Expected 1 source moved, got {moved}"

    # sources.layer updated
    src = conn.execute("SELECT layer FROM sources WHERE id=10").fetchone()
    assert src["layer"] == "community", f"sources.layer must be 'community', got {src['layer']}"

    # chunks.layer updated
    chk = conn.execute("SELECT layer FROM chunks WHERE id=10").fetchone()
    assert chk["layer"] == "community", f"chunks.layer must be 'community', got {chk['layer']}"

    # vec_chunks: community KNN finds the chunk
    query_vec = [0.2] * 384
    community_ids = _knn_ids(conn, "community", query_vec)
    assert 10 in community_ids, (
        f"vec_chunks row must appear in 'community' KNN after re-layer; got {community_ids}"
    )

    # vec_chunks: vessel KNN no longer finds the chunk
    vessel_ids = _knn_ids(conn, "vessel", query_vec)
    assert 10 not in vessel_ids, (
        f"vec_chunks row must NOT appear in 'vessel' KNN after re-layer; got {vessel_ids}"
    )

    conn.close()


# ---------------------------------------------------------------------------
# RED: idempotency — second call is a no-op
# ---------------------------------------------------------------------------


def test_relayer_idempotency():
    """RED: calling relayer_sources_to_community twice on the same source is a no-op.

    First call moves the source. Second call (source already in 'community') must:
    - Return 0 (nothing moved)
    - Leave sources/chunks/vec_chunks unchanged in community state
    - Not raise any error (safe to re-run)

    Fails with ImportError until Task 2 creates migrate.py.
    """
    conn = _make_conn()
    _seed_wa_source(conn)

    from leopard44_kb.migrate import relayer_sources_to_community  # noqa: PLC0415

    moved_first = relayer_sources_to_community(conn, [10])
    assert moved_first == 1, f"First call must move 1 source; got {moved_first}"

    # Second call: already community, nothing to do
    moved_second = relayer_sources_to_community(conn, [10])
    assert moved_second == 0, (
        f"Second call must be a no-op (return 0); got {moved_second}"
    )

    # State still consistent after second call
    src = conn.execute("SELECT layer FROM sources WHERE id=10").fetchone()
    assert src["layer"] == "community", "sources.layer must remain 'community' after idempotent call"

    query_vec = [0.2] * 384
    community_ids = _knn_ids(conn, "community", query_vec)
    assert 10 in community_ids, "vec_chunks row must still be in 'community' partition"

    conn.close()


# ---------------------------------------------------------------------------
# RED: safe-default — apply_migrations does NOT auto-promote
# ---------------------------------------------------------------------------


def test_safe_default_no_auto_promote():
    """RED: apply_migrations / opening a store does NOT auto-promote whatsapp sources.

    D-15 privacy requirement: the source-diversity cap and any schema migration runner
    must NEVER auto-promote a vessel-layer WhatsApp export to the community scope.
    Only an EXPLICIT relayer_sources_to_community() call moves rows.

    Also tests that relayer_sources_to_community(conn, []) is a strict no-op.

    Partially passes in RED (apply_migrations check is pure SQL logic), then
    fails with ImportError when importing migrate for the explicit call test.
    """
    conn = _make_conn()
    _seed_wa_source(conn)

    # After seeding (migrations already applied), source remains 'vessel'
    src = conn.execute("SELECT layer FROM sources WHERE id=10").fetchone()
    assert src["layer"] == "vessel", (
        "apply_migrations MUST NOT auto-promote whatsapp/vessel sources to community"
    )

    chk = conn.execute("SELECT layer FROM chunks WHERE id=10").fetchone()
    assert chk["layer"] == "vessel", (
        "apply_migrations MUST NOT change chunks.layer for whatsapp sources"
    )

    # Explicit call with empty list: must move nothing
    from leopard44_kb.migrate import relayer_sources_to_community  # noqa: PLC0415

    moved = relayer_sources_to_community(conn, [])
    assert moved == 0, f"Empty source list must move nothing; got {moved}"

    # Source still in vessel after explicit empty call
    src = conn.execute("SELECT layer FROM sources WHERE id=10").fetchone()
    assert src["layer"] == "vessel", (
        "relayer_sources_to_community(conn, []) must not move any rows"
    )

    conn.close()


# ---------------------------------------------------------------------------
# RED: deterministic rollback on vec_chunks failure
# ---------------------------------------------------------------------------


def test_rollback_on_vec_failure(monkeypatch):
    """RED: if the vec_chunks move step fails, the whole transaction rolls back.

    The migrate.py implementation executes all steps (UPDATE chunks, UPDATE sources,
    vec DELETE+re-INSERT) inside a single `with conn:` transaction. Any exception
    inside that block causes SQLite to roll back ALL changes — sources and chunks
    remain in 'vessel' and the store is unchanged (safe to re-run).

    The rollback is tested by monkeypatching migrate._move_vec_chunks to raise an
    OperationalError mid-transaction. In RED state, the import fails before the patch.

    Fails with ImportError until Task 2 creates migrate.py.
    """
    import leopard44_kb.migrate as migrate_mod  # noqa: PLC0415 — ImportError until Task 2

    conn = _make_conn()
    _seed_wa_source(conn)

    def _fail(*args, **kwargs):
        raise sqlite3.OperationalError("Simulated vec_chunks move failure")

    # Patch the internal vec move helper so the failure happens inside the transaction
    monkeypatch.setattr(migrate_mod, "_move_vec_chunks", _fail, raising=False)

    from leopard44_kb.migrate import relayer_sources_to_community  # noqa: PLC0415

    with pytest.raises(Exception):
        relayer_sources_to_community(conn, [10])

    # After rollback: all metadata must remain in 'vessel'
    src = conn.execute("SELECT layer FROM sources WHERE id=10").fetchone()
    assert src["layer"] == "vessel", (
        "sources.layer must be rolled back to 'vessel' after vec failure"
    )

    chk = conn.execute("SELECT layer FROM chunks WHERE id=10").fetchone()
    assert chk["layer"] == "vessel", (
        "chunks.layer must be rolled back to 'vessel' after vec failure"
    )

    # vec_chunks must also be unchanged: still in 'vessel', not in 'community'
    query_vec = [0.2] * 384
    vessel_ids = _knn_ids(conn, "vessel", query_vec)
    community_ids = _knn_ids(conn, "community", query_vec)
    assert 10 in vessel_ids, (
        "vec_chunks row must remain in 'vessel' partition after rollback"
    )
    assert 10 not in community_ids, (
        "vec_chunks row must NOT appear in 'community' partition after rollback"
    )

    conn.close()
