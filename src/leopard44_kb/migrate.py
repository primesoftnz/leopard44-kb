"""Explicit data migration for the Leopard 44 KB (D-15: WhatsApp vessel→community re-layer).

DESIGN RATIONALE
----------------
The schema auto-migration runner (schema.py:apply_migrations) executes only SQL files
from schema/NNN_*.sql via executescript — it is a pure-SQL, auto-running mechanism.
Wiring a vessel→community re-layer into that auto-runner would:

  (a) Auto-promote ANY future private WhatsApp export ingested into 'vessel' the next
      time the store opens — a privacy leak (T-13-33).
  (b) Require embedding vectors (3k+ 384-dim floats) to round-trip through SQL only,
      with no clean Python-level transaction boundary.

Therefore the re-layer is implemented HERE as an EXPLICIT, owner-invoked migration,
driven by `l44 migrate relayer-whatsapp --source <id|path>`.

The discriminator is the OWNER'S EXPLICIT per-source selection at migrate time.
Default behaviour: nothing moves. Private WhatsApp exports stay 'vessel' unless named.

SQLITE-VEC PARTITION KEY CONSTRAINT (RESEARCH Pitfall 4, Task 1 probe 2026-06-27)
-----------------------------------------------------------------------------------
vec_chunks uses `layer TEXT partition key` (sqlite-vec vec0). Partition keys are
immutable after INSERT — in-place `UPDATE vec_chunks SET layer = 'community'`
does NOT move the row to the new partition (confirmed by probe in tests/test_migrate.py).

Migration must therefore use:
  1. CREATE TEMP TABLE _wa_vec: snapshot the (chunk_id, source_id, embedding_model,
     is_active, embedding) tuples BEFORE deleting, avoiding 3k+ vectors in Python memory.
  2. DELETE FROM vec_chunks WHERE chunk_id IN (SELECT chunk_id FROM _wa_vec)
  3. INSERT INTO vec_chunks ... SELECT ..., 'community', ... FROM _wa_vec
  4. UPDATE chunks SET layer = 'community' WHERE source_id IN (...)
  5. UPDATE sources SET layer = 'community' WHERE id IN (...)

All five steps run inside a single `with conn:` block — any failure rolls back everything
deterministically. The store is safe to re-run (idempotent guard: only rows still in
'vessel' are touched).
"""
from __future__ import annotations

import sqlite3
from typing import Sequence


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def whatsapp_vessel_candidates(conn: sqlite3.Connection) -> list[dict]:
    """Return a list of whatsapp sources currently in the 'vessel' layer.

    Each entry is a dict with keys: source_id, path, chunk_count.
    Used by `l44 migrate relayer-whatsapp` (no args) to list candidates safely
    before any move is attempted.

    Only sources with source_type='whatsapp' AND layer='vessel' are returned.
    Sources already in 'community' are excluded (idempotency: nothing to move).
    """
    rows = conn.execute(
        "SELECT s.id AS source_id, s.path, COUNT(c.id) AS chunk_count "
        "FROM sources s "
        "LEFT JOIN chunks c ON c.source_id = s.id "
        "WHERE s.source_type = 'whatsapp' AND s.layer = 'vessel' "
        "GROUP BY s.id, s.path "
        "ORDER BY s.id"
    ).fetchall()
    return [dict(r) for r in rows]


def _move_vec_chunks(conn: sqlite3.Connection, source_ids: Sequence[int]) -> None:
    """Move vec_chunks rows from 'vessel' to 'community' partition for the given source_ids.

    Implements the DELETE+re-INSERT approach required because vec0 partition keys
    are immutable after insert (RESEARCH Pitfall 4, confirmed by Task 1 probe).

    The TEMP TABLE _wa_vec avoids loading 3k+ 384-dim vectors into Python memory:
    - CREATE TEMP TABLE snapshots chunk_id, source_id, embedding_model, is_active,
      and the raw embedding blob from vec_chunks before any delete.
    - DELETE removes the 'vessel' partition rows.
    - INSERT re-inserts from the TEMP TABLE with layer='community'.

    This function is called INSIDE a `with conn:` transaction in
    relayer_sources_to_community — any failure here causes the caller's
    transaction to roll back, leaving sources/chunks unchanged.

    This function is exposed as a module-level name so tests can monkeypatch it
    to simulate mid-transaction vec_chunks failure (deterministic rollback test).
    """
    if not source_ids:
        return

    # Step 1: Snapshot affected vec_chunks rows into Python memory.
    # RATIONALE: We cannot use `CREATE TEMP TABLE ... AS SELECT` inside the caller's
    # `with conn:` block because sqlite3.Connection.executescript() issues an implicit
    # COMMIT before running — that would break the enclosing transaction.
    # Instead we snapshot via a normal SELECT (fetchall), then DELETE, then re-INSERT.
    # Memory cost: one Python tuple per affected vec_chunk row (chunk_id int, source_id int,
    # embedding_model str, is_active int, embedding bytes). For 3,241 WhatsApp chunks this
    # is ~3,241 × (8 + 8 + ~20 + 4 + 384×4) bytes ≈ 3,241 × 1,572 bytes ≈ 5 MB — safe.
    ids_ph = ",".join("?" * len(source_ids))
    rows = conn.execute(
        f"SELECT chunk_id, source_id, embedding_model, is_active, embedding "
        f"FROM vec_chunks "
        f"WHERE source_id IN ({ids_ph}) AND layer = 'vessel'",
        list(source_ids),
    ).fetchall()

    if not rows:
        return  # Nothing to move (already community or no vec rows)

    # Step 2: DELETE the 'vessel' rows from vec_chunks
    chunk_ids = [r["chunk_id"] for r in rows]
    chunk_ph = ",".join("?" * len(chunk_ids))
    conn.execute(
        f"DELETE FROM vec_chunks WHERE chunk_id IN ({chunk_ph}) AND layer = 'vessel'",
        chunk_ids,
    )

    # Step 3: re-INSERT with layer='community', preserving all other columns
    for row in rows:
        conn.execute(
            "INSERT INTO vec_chunks"
            "(chunk_id, layer, source_id, embedding_model, is_active, embedding) "
            "VALUES (?, 'community', ?, ?, ?, ?)",
            (row["chunk_id"], row["source_id"], row["embedding_model"],
             row["is_active"], row["embedding"]),
        )


def relayer_sources_to_community(
    conn: sqlite3.Connection,
    source_ids: Sequence[int],
) -> int:
    """Re-layer the specified sources (and their chunks/vec_chunks) from 'vessel' to 'community'.

    Idempotent: only acts on sources whose layer is still 'vessel'. A source already
    in 'community' is skipped (second call returns 0 — safe to re-run).

    Args:
        conn: Open SQLite connection with sqlite-vec loaded and schema migrated.
        source_ids: List of source IDs to re-layer. Pass [] to move nothing (no-op).

    Returns:
        Number of sources actually moved (0 if all were already in 'community').

    Transaction safety:
        All metadata + vec_chunks moves execute inside a single `with conn:` block.
        If any step fails (especially vec_chunks re-insert), the ENTIRE transaction
        rolls back — sources.layer and chunks.layer revert to 'vessel', and the
        vec_chunks rows are either restored to 'vessel' or the delete never happened.
        The store is always in a consistent state after a failure.
    """
    if not source_ids:
        return 0

    # Guard: only act on sources still in 'vessel' (idempotency filter)
    placeholders = ",".join("?" * len(source_ids))
    vessel_rows = conn.execute(
        f"SELECT id FROM sources WHERE id IN ({placeholders}) AND layer = 'vessel'",
        list(source_ids),
    ).fetchall()
    vessel_ids = [r["id"] for r in vessel_rows]

    if not vessel_ids:
        return 0  # All already in community (or don't exist) — no-op

    ids_list = list(vessel_ids)
    ids_ph = ",".join("?" * len(ids_list))

    with conn:
        # Step A: move vec_chunks (most likely to fail — keep first so rollback is clean)
        # This calls the monkeypatcha-able internal helper, enabling the rollback test.
        _move_vec_chunks(conn, ids_list)

        # Step B: update chunks.layer
        conn.execute(
            f"UPDATE chunks SET layer = 'community' WHERE source_id IN ({ids_ph})",
            ids_list,
        )

        # Step C: update sources.layer
        conn.execute(
            f"UPDATE sources SET layer = 'community' WHERE id IN ({ids_ph})",
            ids_list,
        )

    return len(ids_list)
