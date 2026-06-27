"""Hybrid retrieval: sqlite-vec KNN + FTS5 BM25, RRF fusion, suppression ranking.

This module implements the retrieval half of the query engine:
  - FTS5-safe query sanitization (V5 input validation, T-03-03)
  - detect_exact_token_query: exact-token detection for D-05 gating (review fix #3)
  - knn_search: sqlite-vec KNN with layer partition filter (T-03-04)
  - bm25_search: FTS5 BM25 with hyphen-safe quoting and layer filter (T-03-03)
  - rrf_fuse: Reciprocal Rank Fusion of KNN + BM25 lists
  - apply_d05_fts_slot: D-05 exact-token-gated guaranteed FTS slot
  - find_suppressed_chunks: pool-wide D-01/D-02 vessel-addendum suppression (review fix #1)
  - fetch_chunk_metadata: chunk + source metadata JOIN
  - is_below_relevance_floor: D-07 relevance gate
  - most_common_embedding_model: embedding-model mismatch helper (review fix #2)
  - retrieve: full pipeline orchestrator

retrieve() has NO dependency on leopard44_kb.answer — refusal is signalled via the
([], True) return; cli.py renders REFUSAL_MESSAGE (review fix #6 keeps it
single-sourced in answer.py).
"""
from __future__ import annotations

import re
import sqlite3
import struct
import sys

import leopard44_kb.ingest.embedder as _emb
from leopard44_kb.store import open_db  # noqa: F401 — re-exported for callers

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Relevance floor: refuse (D-07) when best KNN L2 distance exceeds this value
# AND BM25 has no results. For unit vectors (Ollama normalizes), L2 > 1.0
# corresponds to cosine_sim < 0.5 (orthogonal). Exposed as a tunable constant
# (Assumption A1 — should be validated against a populated corpus).
REFUSAL_DISTANCE_FLOOR = 1.0

# BM25 strength floor for D-05 gating (review fix #3): if the BM25 top score
# is at or below (more negative than) this value, treat the query as having a
# "strong BM25 hit" and guarantee the FTS slot even for natural-language queries.
# bm25() returns NEGATIVE floats — more negative = stronger match.
# Default: -5.0 (a moderately strong hit). Tunable (cite D-05: "exact token
# that BM25 matches strongly").
BM25_STRENGTH_FLOOR: float = -5.0

# Layer-authority multiplier (D-15, Plan 13-03).
# Applied to accumulated RRF scores before the final top-N sort so that
# authoritative sources (shared manuals/spec sheets) outrank community
# chatter (WhatsApp owners' feed) for factual queries, and community chatter
# outranks vessel-layer personal notes.
#
# Rationale: shared=1.2 gives factory-manual / spec-sheet chunks a 20% bonus,
# community=1.0 is neutral, vessel=0.9 slightly discounts personal maintenance
# notes when competing with authoritative content. The exact-token D-05 slot
# guarantee is applied AFTER weighting, so a strong BM25 hit on a part number
# still forces the exact-match chunk into the top-N regardless of layer.
#
# These weights are deliberately modest (ratio 1.2 : 1.0 : 0.9) so that a
# highly-relevant vessel or community chunk can still rank above a generic
# shared chunk — the multiplier tips ties, not absolute ordering.
LAYER_AUTHORITY: dict[str, float] = {
    "shared": 1.2,
    "community": 1.0,
    "vessel": 0.9,
}


# ---------------------------------------------------------------------------
# pack_embedding — re-declared here for module independence
# (same 3-line body as src/leopard44_kb/ingest/writer.py lines 44-46)
# ---------------------------------------------------------------------------


def pack_embedding(vec: list[float]) -> bytes:
    """Pack a float list into sqlite-vec's binary format (little-endian IEEE 754 floats)."""
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Task 1: sanitize_fts5_query, detect_exact_token_query, knn_search,
#          bm25_search, rrf_fuse, apply_d05_fts_slot
# ---------------------------------------------------------------------------


def sanitize_fts5_query(user_query: str) -> str:
    """Convert raw user query to a safe FTS5 MATCH expression (V5 input validation).

    Each whitespace-delimited token is double-quoted to:
      1. Prevent FTS5 operator injection (AND/OR/NOT/parens from user input).
      2. Treat hyphenated tokens like '4JH45-W001' as a single phrase (Pitfall 1).

    Tokens are joined with explicit ``OR`` rather than whitespace. FTS5 treats
    space-separated terms as an implicit AND, which silently broke the lexical
    half of hybrid retrieval for natural-language questions: "where are the bilge
    pumps" required a SINGLE chunk containing every one of those words and matched
    nothing, so BM25 returned zero and retrieval fell back to KNN-only (which then
    drifted to semantically-adjacent-but-wrong chunks). With OR, BM25 ranks chunks
    by their matching terms — and since FTS5's bm25() applies IDF weighting, rare
    domain terms ("liferaft", "windlass", "12.98") dominate the score while common
    stopwords ("where", "the", "is") contribute ~nothing. A single-token query is
    unaffected (no join). [Fixed 2026-06-17 — field UAT surfaced the AND bug.]

    Internal double-quotes within tokens are escaped as '""'.
    Empty / whitespace-only input returns "" (bm25_search guards against this).

    Verified: '"4JH45-W001"' matches correctly; bare '4JH45-W001' raises
    OperationalError in FTS5. [VERIFIED: local FTS5 probe 2026-05-29]
    """
    words = user_query.split()
    if not words:
        return ""
    return " OR ".join('"' + w.replace('"', '""') + '"' for w in words)


def detect_exact_token_query(user_query: str) -> bool:
    """Return True if the query looks like an exact-token query (D-05 gating, review fix #3).

    Exact-token queries contain at least one of:
      - A digit (part numbers, model codes, dates, measurements)
      - A hyphen or slash embedded inside an alphanumeric token (e.g. 4JH45-W001, P/N)
      - An uppercase-alphanumeric model-code token: a token containing both
        an uppercase letter AND a digit (e.g. "4JH45", "W001", "22-41016" prefix)
      - A double-quoted phrase (user explicitly requesting exact match)

    False for purely lowercase natural-language sentences with no digits/hyphens/slashes.

    Rationale (D-05 context): D-05 says "exact token that BM25 matches strongly".
    Only apply the guaranteed FTS slot when the query contains tokens a user would
    type to find a specific part, code, or quoted phrase — not for free-form NL.
    """
    # Double-quoted phrase in the query
    if '"' in user_query:
        return True
    # Any digit
    if re.search(r"\d", user_query):
        return True
    # Uppercase-alphanumeric token: contains at least one uppercase letter
    # (model codes like "4JH45", "W001", "NL M673L3")
    for token in user_query.split():
        stripped = re.sub(r"[^A-Za-z0-9]", "", token)
        if stripped and re.search(r"[A-Z]", stripped):
            return True
    # Hyphen or slash inside an alphanumeric token (P/N, part-number style)
    if re.search(r"[A-Za-z0-9][-/][A-Za-z0-9]", user_query):
        return True
    return False


def knn_search(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    layers: list[str],
    k: int = 20,
) -> list[tuple[int, float]]:
    """Return (chunk_id, l2_distance) pairs ordered by distance ASC.

    Distance metric: L2 (Euclidean). For unit vectors (Ollama normalizes),
    L2 ordering is equivalent to cosine distance ordering.
    Distance 0.0 = identical; ~1.414 = orthogonal (cosine_sim = 0).

    Layer filter is applied inside the SQL (partition key, not post-filter)
    so --layer shared cannot surface vessel chunks (T-03-04 V4 access control).
    [VERIFIED: local sqlite-vec probe 2026-05-29, v0.1.9]
    """
    query_bytes = pack_embedding(query_embedding)

    if not layers:
        # No layer filter: search all layers
        # `k` is sqlite-vec's reserved KNN-limit pseudo-column (vec0), NOT a
        # column of vec_chunks. `AND k = ?` binds the KNN result count, it is
        # not a column filter; keep it parameterized.
        rows = conn.execute(
            "SELECT v.chunk_id, v.distance "
            "FROM vec_chunks v "
            "WHERE v.embedding MATCH ? AND v.is_active = 1 AND k = ? "
            "ORDER BY v.distance",
            (query_bytes, k),
        ).fetchall()
    else:
        # Dynamic IN clause — build exactly len(layers) placeholders (Pitfall 2)
        placeholders = ",".join("?" * len(layers))
        # `k` is sqlite-vec's reserved KNN-limit pseudo-column (vec0), NOT a
        # column of vec_chunks. `AND k = ?` binds the KNN result count, it is
        # not a column filter; keep it parameterized.
        rows = conn.execute(
            f"SELECT v.chunk_id, v.distance "
            f"FROM vec_chunks v "
            f"WHERE v.embedding MATCH ? "
            f"AND v.layer IN ({placeholders}) "
            f"AND v.is_active = 1 "
            f"AND k = ? "
            f"ORDER BY v.distance",
            [query_bytes] + layers + [k],
        ).fetchall()

    return [(r["chunk_id"], r["distance"]) for r in rows]


def bm25_search(
    conn: sqlite3.Connection,
    user_query: str,
    layers: list[str],
    pool: int = 20,
) -> list[tuple[int, float]]:
    """Return (chunk_id, bm25_score) pairs ordered by relevance (lowest/most-negative = best).

    bm25() returns NEGATIVE floats. Empty list if no FTS5 matches.

    All user input is routed through sanitize_fts5_query before the MATCH
    expression (V5 input validation — T-03-03). A try/except OperationalError
    provides a defensive fallback if sanitization somehow fails.
    [VERIFIED: local FTS5 probe 2026-05-29]
    """
    safe_query = sanitize_fts5_query(user_query)
    if not safe_query:
        return []

    try:
        if not layers:
            rows = conn.execute(
                "SELECT f.rowid AS chunk_id, bm25(fts_chunks) AS score "
                "FROM fts_chunks f "
                "JOIN chunks c ON c.id = f.rowid "
                "WHERE fts_chunks MATCH ? "
                "AND c.is_active = 1 "
                "ORDER BY bm25(fts_chunks) "
                "LIMIT ?",
                (safe_query, pool),
            ).fetchall()
        else:
            placeholders = ",".join("?" * len(layers))
            rows = conn.execute(
                f"SELECT f.rowid AS chunk_id, bm25(fts_chunks) AS score "
                f"FROM fts_chunks f "
                f"JOIN chunks c ON c.id = f.rowid "
                f"WHERE fts_chunks MATCH ? "
                f"AND c.layer IN ({placeholders}) "
                f"AND c.is_active = 1 "
                f"ORDER BY bm25(fts_chunks) "
                f"LIMIT ?",
                [safe_query] + layers + [pool],
            ).fetchall()
    except sqlite3.OperationalError:
        # Defensive fallback: FTS5 syntax error despite sanitization (rare).
        return []

    return [(r["chunk_id"], r["score"]) for r in rows]


def rrf_fuse(
    knn_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse two ranked lists with Reciprocal Rank Fusion. Returns (chunk_id, rrf_score) DESC.

    RRF score formula: score(chunk) = Σ 1 / (k + rank) across both lists.
    k=60 is the standard RRF constant; it dampens the influence of very high ranks.

    A chunk appearing in both KNN and BM25 lists accumulates score from both,
    so it ranks higher than a chunk appearing in only one list.
    [VERIFIED: local Python test 2026-05-29]
    """
    scores: dict[int, float] = {}
    for rank, (cid, _) in enumerate(knn_results, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, (cid, _) in enumerate(bm25_results, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def apply_d05_fts_slot(
    fused: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    n: int,
    user_query: str = "",
) -> list[int]:
    """Apply D-05: guarantee BM25 rank-1 has a slot in the final top-N.

    Gating (review fix #3 — exact-token-aware D-05):
    The forced slot fires ONLY when:
      (a) detect_exact_token_query(user_query) is True  — query contains a part
          number, model code, digit, or quoted phrase, OR
      (b) bm25_results[0][1] <= BM25_STRENGTH_FLOOR  — the top BM25 hit is
          strongly matching even for a natural-language query.

    Without this gate, a weak BM25 hit on a NL query could displace a better
    semantic result (D-05 as written: "exact token that BM25 matches strongly").

    No-ops:
      - bm25_results is empty
      - BM25 rank-1 already in top-N fused IDs
      - Neither gating condition fires
    """
    # WR-05 guard: n must be >= 1. A non-positive n would make top_n_ids[: n - 1]
    # slice from the tail (n=0 → [:-1], n=-1 → [:-2]), silently corrupting the
    # forced-slot result. ask_cmd validates --top-k >= 1, but guard here too.
    if n <= 0:
        return []

    top_n_ids = [cid for cid, _ in fused[:n]]

    if not bm25_results:
        return top_n_ids  # nothing to guarantee

    fts_top_id = bm25_results[0][0]
    if fts_top_id in top_n_ids:
        return top_n_ids  # already present: no-op

    # Check gating conditions
    exact_token = detect_exact_token_query(user_query) if user_query else False
    strong_bm25 = bm25_results[0][1] <= BM25_STRENGTH_FLOOR

    if exact_token or strong_bm25:
        # Force BM25 top-1 into the last slot
        return top_n_ids[: n - 1] + [fts_top_id]

    # Neither condition: leave top-N unchanged (NL query, weak BM25 hit)
    return top_n_ids


# ---------------------------------------------------------------------------
# Task 2: find_suppressed_chunks, fetch_chunk_metadata,
#          is_below_relevance_floor, most_common_embedding_model
# ---------------------------------------------------------------------------


def find_suppressed_chunks(
    conn: sqlite3.Connection,
    pool_ids: list[int],
) -> dict[int, dict]:
    """Return a dict mapping suppressed shared chunk IDs to suppression info.

    Operates over the FULL candidate pool (pool_ids, k≈20), NOT just the top-N
    (review fix #1 — pool-wide suppression). This ensures a vessel override
    just outside top-N is detected and can be pulled into context by retrieve().

    Returns:
        {shared_chunk_id: {'vessel_chunk_id': int, 'relation': 'supersedes'|'annotates'}}

    Empty dict when:
      - pool_ids is empty (guard)
      - all supersedes_chunk_id / annotates_chunk_id FKs are NULL (correct no-op)

    Only is_active=1 vessel chunks suppress. supersedes takes precedence over
    annotates when the same shared_id appears in both queries.

    Pitfall 6 (double-binding): pool_ids appears in TWO IN clauses per query
    (vessel side: id IN (...), FK target: supersedes_chunk_id IN (...)). Pass
    args + args as parameters.
    [VERIFIED: local sqlite3 test 2026-05-29]
    """
    if not pool_ids:
        return {}

    placeholders = ",".join("?" * len(pool_ids))
    args = list(pool_ids)
    suppressed: dict[int, dict] = {}

    # Supersedes links: vessel chunks that supersede a shared chunk in the pool
    rows = conn.execute(
        f"SELECT id AS vessel_chunk_id, supersedes_chunk_id AS shared_id "
        f"FROM chunks "
        f"WHERE id IN ({placeholders}) "
        f"AND layer = 'vessel' "
        f"AND is_active = 1 "
        f"AND supersedes_chunk_id IS NOT NULL "
        f"AND supersedes_chunk_id IN ({placeholders})",
        args + args,  # double-bind: pool_ids used twice (Pitfall 6)
    ).fetchall()
    for r in rows:
        suppressed[r["shared_id"]] = {
            "vessel_chunk_id": r["vessel_chunk_id"],
            "relation": "supersedes",
        }

    # Annotates links: vessel chunks that annotate a shared chunk in the pool.
    # supersedes takes precedence — skip if shared_id already mapped.
    rows = conn.execute(
        f"SELECT id AS vessel_chunk_id, annotates_chunk_id AS shared_id "
        f"FROM chunks "
        f"WHERE id IN ({placeholders}) "
        f"AND layer = 'vessel' "
        f"AND is_active = 1 "
        f"AND annotates_chunk_id IS NOT NULL "
        f"AND annotates_chunk_id IN ({placeholders})",
        args + args,
    ).fetchall()
    for r in rows:
        if r["shared_id"] not in suppressed:  # supersedes wins
            suppressed[r["shared_id"]] = {
                "vessel_chunk_id": r["vessel_chunk_id"],
                "relation": "annotates",
            }

    return suppressed


def fetch_chunk_metadata(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
) -> list[dict]:
    """Fetch full chunk metadata for the given IDs, preserving caller-specified order.

    JOINs chunks → sources to include title and path (needed for D-06 citation rendering).
    SQL IN clause has no guaranteed order, so results are re-sorted by the input id list.
    [VERIFIED: local sqlite3 pattern 2026-05-29]
    """
    if not chunk_ids:
        return []

    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT c.id, c.layer, c.content, c.section_path, "
        f"       c.page_start, c.page_end, c.anchor_key, c.metadata, "
        f"       s.title, s.path "
        f"FROM chunks c "
        f"JOIN sources s ON s.id = c.source_id "
        f"WHERE c.id IN ({placeholders})",
        chunk_ids,
    ).fetchall()

    # Restore caller-specified order (SQL IN has no guaranteed order)
    by_id = {r["id"]: dict(r) for r in rows}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


def is_below_relevance_floor(
    knn_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
) -> bool:
    """Return True if retrieval quality is too low to ground an answer (D-07).

    Refuses when:
      - No chunks retrieved at all (empty corpus or no match).
      - Best KNN L2 distance > REFUSAL_DISTANCE_FLOOR AND no BM25 results.
        (BM25 results = exact-token match always qualifies; distance check skipped.)

    Threshold REFUSAL_DISTANCE_FLOOR = 1.0 (Assumption A1 — tunable; see module constant).
    """
    if not knn_results and not bm25_results:
        return True
    # WR-07 fix: only a STRONG BM25 hit qualifies as exact-token grounding.
    # A weak/any-token FTS hit (e.g. a stopword match from sanitize_fts5_query
    # quoting every whitespace token) must NOT defeat the distance floor —
    # otherwise D-07 refusal is bypassed for irrelevant matches. Mirrors the
    # D-05 strength gate (apply_d05_fts_slot / BM25_STRENGTH_FLOOR).
    if bm25_results and bm25_results[0][1] <= BM25_STRENGTH_FLOOR:
        return False  # strong exact-token match always qualifies
    best_distance = knn_results[0][1] if knn_results else 999.0
    return best_distance > REFUSAL_DISTANCE_FLOOR


def most_common_embedding_model(conn: sqlite3.Connection) -> str | None:
    """Return the modal embedding_model string from chunks, or None if no chunks exist.

    Used by retrieve() to detect embedding-model mismatch at query time (review fix #2).
    A mismatch means KNN distances may be meaningless (different vector spaces).
    [VERIFIED: pattern from 03-RESEARCH.md Pitfall 5]
    """
    # WHERE is_active = 1: reflect the same active corpus the KNN actually
    # queries (CR-01 filters vec_chunks to is_active = 1); a retired chunk's
    # stale embedding_model must not drive the mismatch warning (WR-08).
    # ORDER BY ... embedding_model ASC: deterministic tiebreaker so a mid-
    # migration tie does not flip-flop the warning (WR-06).
    row = conn.execute(
        "SELECT embedding_model, COUNT(*) AS c "
        "FROM chunks "
        "WHERE is_active = 1 "
        "GROUP BY embedding_model "
        "ORDER BY c DESC, embedding_model ASC "
        "LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return row["embedding_model"]


# ---------------------------------------------------------------------------
# retrieve() orchestrator
# ---------------------------------------------------------------------------


def retrieve(
    conn: sqlite3.Connection,
    question: str,
    layers: list[str],
    n: int = 5,
    pool: int = 20,
) -> tuple[list[dict], bool]:
    """Full hybrid retrieval pipeline. Returns (chunks, is_below_floor).

    chunks: list of dicts with keys:
        id, layer, content, section_path, page_start, page_end,
        anchor_key, title, path, _suppression_note ('' or non-empty flag)
    is_below_floor: True → caller should render D-07 refusal message.

    Pipeline:
      1. Emit soft stderr warning if corpus embed model ≠ current select_model()
         (review fix #2). Does NOT raise — query continues.
      2. Embed the question using the same model/dim as ingest (select_model).
      3. KNN + BM25 in the layer-scoped candidate pool.
      4. Relevance floor check → early ([], True) return on empty/distant corpus.
      5. RRF fusion → LAYER_AUTHORITY weighting → D-05 exact-token-gated slot.
         (Source-diversity cap retired in D-15; layer-authority weighting is the
         principled replacement — shared > community > vessel for factual queries.)
      6. Pool-wide suppression detection (review fix #1: full pool, not just top-N).
      7. Pool-wide pull-in: vessel superseder/annotator outside top-N → pulled in.
      8. Compute final suppression map restricted to working set.
      9. Reorder: non-suppressed first (down-rank suppressed to end). Cap at n.
      10. Fetch metadata + annotate _suppression_note.

    retrieve() has NO import of leopard44_kb.answer. Refusal is signalled via ([], True);
    cli.py renders REFUSAL_MESSAGE (review fix #6).
    """
    # ------------------------------------------------------------------ Step 1
    # Embedding-model mismatch warning (review fix #2 / Pitfall 5).
    # Compare current select_model() against what the corpus was embedded with.
    # Call via module reference so test monkeypatching of embedder works
    # (Phase 2 lesson: module-level import binding).
    model, _ = _emb.select_model()
    corpus_model = most_common_embedding_model(conn)
    if corpus_model is not None and corpus_model != model:
        # Soft yellow warning — do NOT raise. User may legitimately test on a
        # lower-RAM machine. KNN may be degraded but we still return results.
        print(
            f"\033[33mWARNING: embedding model mismatch — "
            f"corpus stored '{corpus_model}', current select_model() returns '{model}'. "
            f"KNN semantic retrieval may be degraded.\033[0m",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ Step 2
    # Embed the query using the SAME 384-dim path as ingest (Anti-Pattern: do NOT
    # use a different model — query and chunk vectors must share the same space).
    query_vec = _emb.embed_texts([question], model)[0]

    # ------------------------------------------------------------------ Step 3
    # KNN + BM25 in parallel (both are fast; sequential is fine)
    knn = knn_search(conn, query_vec, layers, k=pool)
    bm25 = bm25_search(conn, question, layers, pool=pool)

    # ------------------------------------------------------------------ Step 4
    # Relevance floor check BEFORE fusion — avoids wasted work on empty corpus.
    if is_below_relevance_floor(knn, bm25):
        return [], True

    # ------------------------------------------------------------------ Step 5
    # RRF fusion → layer-authority weighting → D-05 exact-token-gated FTS slot.
    #
    # The source-diversity cap (CHATTER_SOURCE_TYPES / _apply_source_diversity_cap)
    # is RETIRED (D-15, Plan 13-03).  The principled replacement is a per-layer
    # multiplier applied to the accumulated RRF scores before the final sort.
    # This means shared (factory manuals / spec sheets) outranks community
    # (WhatsApp owners' feed, after re-layer) which outranks vessel for factual
    # queries, without artificially capping any source type.
    #
    # The D-05 guaranteed FTS slot fires AFTER the authority sort so that an
    # exact part/model-code BM25 hit is still forced into the top-N regardless
    # of its layer weight (T-13-34 Codex MEDIUM — exact-token preservation).
    fused = rrf_fuse(knn, bm25)

    # Apply LAYER_AUTHORITY multiplier: fetch chunk layers via a single JOIN,
    # scale each chunk's accumulated RRF score, then re-sort descending.
    if fused:
        fused_ids = [cid for cid, _ in fused]
        layer_ph = ",".join("?" * len(fused_ids))
        layer_rows = conn.execute(
            f"SELECT id, layer FROM chunks WHERE id IN ({layer_ph})",
            fused_ids,
        ).fetchall()
        id_to_layer: dict[int, str] = {r["id"]: r["layer"] for r in layer_rows}
        fused = sorted(
            [
                # IN-05: a fused id absent from the chunks JOIN (a genuine orphan, e.g. a
                # vec_chunks/fts row with no chunks parent) gets the NEUTRAL 1.0
                # multiplier — not the vessel 0.9 — so a data inconsistency is neither
                # silently down-weighted nor hidden.
                (cid, score * LAYER_AUTHORITY.get(id_to_layer.get(cid), 1.0))
                for cid, score in fused
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    top_n_ids = apply_d05_fts_slot(fused, bm25, n=n, user_query=question)

    # ------------------------------------------------------------------ Step 6
    # Pool-wide suppression detection (review fix #1).
    # Build the FULL pool id list: union of all fused ids + any BM25-only ids,
    # capped at pool size. This is larger than top_n_ids.
    fused_pool_ids = [cid for cid, _ in fused[:pool]]
    bm25_ids = [cid for cid, _ in bm25]
    # union while preserving fused order (fused_pool_ids first, bm25_ids appended)
    seen: set[int] = set(fused_pool_ids)
    pool_ids: list[int] = list(fused_pool_ids)
    for cid in bm25_ids:
        if cid not in seen:
            pool_ids.append(cid)
            seen.add(cid)
    pool_suppression = find_suppressed_chunks(conn, pool_ids)

    # ------------------------------------------------------------------ Step 7
    # Pool-wide pull-in (review fix #1, Consensus fix #1):
    # For any shared_id in top_n_ids that is a suppression KEY (shared chunk flagged
    # as having a vessel override), ensure its vessel_chunk_id is in the working set.
    # If absent, pull it in by replacing the weakest UNRELATED chunk (to honour n cap).
    # Symmetrically, for each vessel chunk in top_n_ids that is a vessel_chunk_id in
    # pool_suppression, optionally pull its linked shared baseline in too.

    working_ids: list[int] = list(top_n_ids)

    # Collect all (shared_id, vessel_chunk_id) pairs relevant to the working set
    pull_pairs: list[tuple[int, int]] = []
    for shared_id, info in pool_suppression.items():
        vcid = info["vessel_chunk_id"]
        # Case A: shared_id in top-N, vessel superseder outside top-N → pull vessel in
        if shared_id in working_ids and vcid not in working_ids:
            pull_pairs.append((shared_id, vcid))
        # Case B: vessel chunk in top-N, its shared baseline outside top-N → pull shared in
        if vcid in working_ids and shared_id not in working_ids:
            pull_pairs.append((vcid, shared_id))

    for _anchor, newcomer in pull_pairs:
        if newcomer in working_ids:
            continue  # already present (may have been pulled in by a prior iteration)
        if len(working_ids) < n:
            # Cap not yet reached — just append
            working_ids.append(newcomer)
        else:
            # Cap reached: evict the weakest chunk that is NOT part of any suppression pair.
            # "Weakest" = last in the current working set (lowest RRF/fused rank).
            # Suppression-pair IDs = all keys + all vessel_chunk_ids in pool_suppression.
            suppression_ids: set[int] = set()
            for sid, info in pool_suppression.items():
                suppression_ids.add(sid)
                suppression_ids.add(info["vessel_chunk_id"])
            # Find the last chunk not in any suppression pair
            evict_idx = None
            for i in range(len(working_ids) - 1, -1, -1):
                if working_ids[i] not in suppression_ids:
                    evict_idx = i
                    break
            if evict_idx is not None:
                working_ids[evict_idx] = newcomer
            else:
                # All remaining slots are suppression pairs — append anyway (overflow by 1)
                # rather than silently drop the linked counterpart.
                working_ids.append(newcomer)

    # ------------------------------------------------------------------ Step 8
    # Final suppression map restricted to chunks actually in the working set.
    # Only annotate pairs where BOTH the shared chunk AND its vessel counterpart
    # are present in context (so the annotation is meaningful).
    working_set: set[int] = set(working_ids)
    suppressed = {
        sid: info
        for sid, info in pool_suppression.items()
        if sid in working_set and info["vessel_chunk_id"] in working_set
    }

    # ------------------------------------------------------------------ Step 9
    # Reorder: non-suppressed IDs first, suppressed IDs last (D-02 down-rank).
    # Cap to n.
    non_suppressed = [cid for cid in working_ids if cid not in suppressed]
    suppressed_ids_ordered = [cid for cid in working_ids if cid in suppressed]
    ordered_ids = (non_suppressed + suppressed_ids_ordered)[:n]

    # ------------------------------------------------------------------ Step 10
    # Fetch metadata and annotate suppression notes.
    chunks = fetch_chunk_metadata(conn, ordered_ids)
    for chunk in chunks:
        info = suppressed.get(chunk["id"])
        if info:
            relation = info["relation"]
            chunk["_suppression_note"] = f" [NOTE: {relation}d by vessel note]"
        else:
            chunk["_suppression_note"] = ""

    return chunks, False
