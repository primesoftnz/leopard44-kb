"""Tests for QUERY-01..05, D-01..D-05, D-15, Pitfall 5: KNN, FTS5, RRF, suppression,
layer scoping, layer-authority weighting.

Per-requirement verification map source: .planning/phases/03-query-engine/03-VALIDATION.md
Review fixes covered: #1 (pool-wide suppression pull-in), #2 (embed-model mismatch warning),
#3 (exact-token D-05 gating + detect_exact_token_query).
D-15 (Plan 13-03): LAYER_AUTHORITY weighting, source-diversity cap retired.
"""
from __future__ import annotations

import pytest

from leopard44_kb.retrieve import (
    apply_d05_fts_slot,
    bm25_search,
    detect_exact_token_query,
    fetch_chunk_metadata,
    find_suppressed_chunks,
    is_below_relevance_floor,
    knn_search,
    most_common_embedding_model,
    retrieve,
    rrf_fuse,
    sanitize_fts5_query,
)


# ---------------------------------------------------------------------------
# QUERY-04 / Pitfall 1: FTS5 sanitization
# ---------------------------------------------------------------------------


def test_sanitize_hyphenated():
    """sanitize_fts5_query wraps hyphenated part numbers in double-quotes."""
    result = sanitize_fts5_query("4JH45-W001")
    assert result == '"4JH45-W001"', f"Expected '\"4JH45-W001\"', got {result!r}"


def test_fts_hyphenated_part_number(retrieval_db):
    """bm25_search with a hyphenated part number does not raise OperationalError."""
    results = bm25_search(retrieval_db, "4JH45-W001", [], pool=20)
    # Results may be empty or populated; what matters is no exception is raised.
    assert isinstance(results, list)


def test_fts_exact_match(retrieval_db):
    """bm25_search with an exact part-number token finds the chunk containing it."""
    results = bm25_search(retrieval_db, "4JH45-W001", [], pool=20)
    ids = [r[0] for r in results]
    # Chunk 1 and 3 both contain '4JH45-W001' in their content.
    assert len(ids) > 0, "Expected at least one BM25 hit for '4JH45-W001'"
    assert 1 in ids or 3 in ids, f"Expected chunk 1 or 3 in BM25 results: {ids}"


# ---------------------------------------------------------------------------
# QUERY-04: RRF fusion
# ---------------------------------------------------------------------------


def test_rrf_fusion_scores():
    """A chunk appearing in both KNN and BM25 lists outranks a single-list chunk."""
    knn = [(1, 0.1), (2, 0.5)]  # chunk 1 rank-1, chunk 2 rank-2
    bm25 = [(1, -5.0), (3, -2.0)]  # chunk 1 rank-1, chunk 3 rank-2

    fused = rrf_fuse(knn, bm25)
    fused_ids = [cid for cid, _ in fused]

    # Chunk 1 appears in both lists: should rank first.
    assert fused_ids[0] == 1, f"Chunk in both lists should rank first; got {fused_ids}"
    # Chunk 2 (KNN only) and chunk 3 (BM25 only) have lower score than chunk 1.
    chunk1_score = dict(fused)[1]
    for cid in (2, 3):
        if cid in dict(fused):
            assert chunk1_score > dict(fused)[cid], f"Chunk 1 score should exceed chunk {cid}"


# ---------------------------------------------------------------------------
# QUERY-04 / D-05: Forced FTS slot + exact-token gating (review fixes #3)
# ---------------------------------------------------------------------------


def test_d05_fts_slot_guarantee():
    """D-05 happy path: exact-token query forces BM25 rank-1 into top-N even if RRF excludes it.

    Simulates a scenario where chunk 99 is BM25 rank-1 but absent from the fused top-N.
    With an exact-token user_query, apply_d05_fts_slot forces it into the last slot.
    """
    # Fused top-2: chunks 1 and 2 (no chunk 99)
    fused = [(1, 0.032), (2, 0.016), (99, 0.008)]
    bm25 = [(99, -10.0), (1, -5.0)]  # BM25 rank-1 is chunk 99

    # Exact-token query containing a part number
    top_n = apply_d05_fts_slot(fused, bm25, n=2, user_query="4JH45-W001")
    assert 99 in top_n, f"D-05 should force chunk 99 into top-2; got {top_n}"


def test_d05_exact_token_gating_negative():
    """Review fix #3: natural-language query does NOT trigger D-05 forced slot.

    When detect_exact_token_query is False, a weak BM25 hit must NOT displace
    a stronger semantic result. The returned top-N stays unchanged.
    """
    # Fused top-2: chunks 1 and 2 (chunk 99 just misses)
    fused = [(1, 0.032), (2, 0.016), (99, 0.008)]
    bm25 = [(99, -2.0), (1, -1.8)]  # BM25 rank-1 is chunk 99 (weak hit)

    # Natural-language query: no digits/hyphens/slashes/uppercase-alphanumeric
    top_n = apply_d05_fts_slot(fused, bm25, n=2, user_query="why does the cooling system lose pressure")
    assert 99 not in top_n, f"D-05 should NOT fire for NL query; got {top_n}"
    assert top_n == [1, 2], f"Top-N should be unchanged; got {top_n}"


def test_detect_exact_token_query():
    """Review fix #3: detect_exact_token_query identifies exact-token queries correctly."""
    # True cases: digits, hyphens, slashes, uppercase alphanumeric, quoted terms
    assert detect_exact_token_query("4JH45") is True, "Standalone model code"
    assert detect_exact_token_query("P/N 22-41016") is True, "Part number with slash and hyphen"
    assert detect_exact_token_query('"impeller"') is True, "Quoted term"

    # False cases: natural language queries
    assert detect_exact_token_query("why does the cooling system lose pressure") is False
    assert detect_exact_token_query("what is the recommended oil type") is False


# ---------------------------------------------------------------------------
# QUERY-03: Layer scoping
# ---------------------------------------------------------------------------


def test_layer_scoping_shared(retrieval_db, fake_embedder):
    """knn_search with layers=['shared'] returns only shared-layer chunks."""
    query_vec = [0.1] * 384
    results = knn_search(retrieval_db, query_vec, layers=["shared"], k=10)
    ids = [r[0] for r in results]
    # Chunk 3 is vessel — must not appear
    assert 3 not in ids, f"Vessel chunk 3 leaked into shared-scoped KNN: {ids}"
    # Chunks 1 and 2 are shared — at least one should appear
    assert len(ids) > 0, "Expected at least one shared chunk from KNN"


def test_layer_scoping_vessel(retrieval_db, fake_embedder):
    """knn_search with layers=['vessel'] returns only vessel-layer chunks."""
    query_vec = [0.1] * 384
    results = knn_search(retrieval_db, query_vec, layers=["vessel"], k=10)
    ids = [r[0] for r in results]
    # Chunks 1 and 2 are shared — must not appear
    assert 1 not in ids and 2 not in ids, f"Shared chunks leaked into vessel-scoped KNN: {ids}"


def test_layer_all_returns_both(retrieval_db, fake_embedder):
    """knn_search with empty layers (all) returns chunks from both layers."""
    query_vec = [0.1] * 384
    results = knn_search(retrieval_db, query_vec, layers=[], k=20)
    ids = [r[0] for r in results]
    layers_found = set()
    # The result set must contain at least one shared and one vessel chunk.
    for cid in ids:
        if cid in (1, 2):
            layers_found.add("shared")
        elif cid == 3:
            layers_found.add("vessel")
    assert "shared" in layers_found, f"No shared chunks found with all-layer KNN: {ids}"
    assert "vessel" in layers_found, f"No vessel chunks found with all-layer KNN: {ids}"


# ---------------------------------------------------------------------------
# QUERY-05: Top-k cap
# ---------------------------------------------------------------------------


def test_top_k_cap(retrieval_db, fake_embedder):
    """retrieve() result count never exceeds the requested n."""
    chunks, below_floor = retrieve(retrieval_db, "impeller interval", [], n=2)
    assert len(chunks) <= 2, f"Expected at most 2 chunks, got {len(chunks)}"


# ---------------------------------------------------------------------------
# D-01: Suppression detection
# ---------------------------------------------------------------------------


def test_suppression_null_fks_noop(empty_db):
    """find_suppressed_chunks returns empty dict when all chunks have NULL FKs.

    Operates over the full candidate pool (review fix #1 semantics: pool-wide).
    """
    # empty_db has no chunks at all; passing arbitrary IDs returns {}
    result = find_suppressed_chunks(empty_db, [1, 2, 3])
    assert result == {}, f"Expected empty dict when no suppression links: {result}"


def test_suppression_detects_supersede(retrieval_db):
    """find_suppressed_chunks over the pool detects chunk 3 supersedes chunk 1.

    Pool contains all three chunks (pool-wide search, review fix #1).
    """
    pool_ids = [1, 2, 3]
    result = find_suppressed_chunks(retrieval_db, pool_ids)
    # Chunk 3 supersedes chunk 1 → shared chunk 1 should be flagged
    assert 1 in result, f"Expected chunk 1 to be flagged as suppressed: {result}"
    assert result[1]["vessel_chunk_id"] == 3
    assert result[1]["relation"] == "supersedes"


# ---------------------------------------------------------------------------
# D-02: Suppression down-ranking
# ---------------------------------------------------------------------------


def test_suppressed_chunk_down_ranked(retrieval_db, fake_embedder):
    """Suppressed shared chunk is present but ordered after non-suppressed chunks.

    chunk 1 is superseded by vessel chunk 3. When both appear in results,
    chunk 1 (suppressed) should appear after chunk 2 (not suppressed).
    """
    chunks, below_floor = retrieve(retrieval_db, "impeller interval", [], n=5)
    ids = [c["id"] for c in chunks]

    # IN-01: assert the precondition so the ordering check is always exercised —
    # a vacuous pass (chunk 1 or 2 missing) would not prove the down-ranking.
    assert 1 in ids and 2 in ids, (
        f"Expected both chunk 1 (suppressed) and chunk 2 in results; got ids={ids}"
    )
    pos_1 = ids.index(1)
    pos_2 = ids.index(2)
    assert pos_2 < pos_1, (
        f"Non-suppressed chunk 2 (pos {pos_2}) should precede "
        f"suppressed chunk 1 (pos {pos_1}); order: {ids}"
    )


# ---------------------------------------------------------------------------
# D-01 / review fix #1: Pool-wide suppression pull-in
# ---------------------------------------------------------------------------


def test_suppression_pull_in_from_pool(retrieval_db, fake_embedder):
    """Review fix #1: vessel superseder outside fused top-N is pulled into context.

    retrieval_db gives chunk 3 (vessel, supersedes chunk 1) a far vector that is
    unlikely to land in the fused top-N when n is small. retrieve() should still
    pull chunk 3 into the returned chunks alongside shared chunk 1, and chunk 1
    should carry a non-empty _suppression_note.
    """
    # Use n=2 so chunk 3's far vector may be excluded from raw fusion top-2
    chunks, below_floor = retrieve(retrieval_db, "impeller interval", [], n=2)
    ids = [c["id"] for c in chunks]

    # IN-01: assert the precondition so the pull-in check is always exercised —
    # this is the exact test meant to prove review fix #1.
    assert 1 in ids, (
        f"Expected suppressed shared chunk 1 in results to trigger pull-in; got ids={ids}"
    )
    assert 3 in ids, (
        f"Vessel superseder (chunk 3) should be pulled into context "
        f"alongside suppressed shared chunk 1; got ids={ids}"
    )
    # Chunk 1 must carry a suppression note
    chunk1 = next(c for c in chunks if c["id"] == 1)
    assert chunk1.get("_suppression_note"), (
        f"Suppressed chunk 1 should have a non-empty _suppression_note: {chunk1}"
    )


# ---------------------------------------------------------------------------
# Pitfall 5 / review fix #2: Embedding-model mismatch warning
# ---------------------------------------------------------------------------


def test_embed_model_mismatch_warning(retrieval_db, fake_embedder, capsys, monkeypatch):
    """Review fix #2: mismatch between current select_model and stored embedding_model
    emits a warning to stderr and still returns results (soft warning, not a raise).

    retrieval_db stores embedding_model='m' for all chunks.
    We monkeypatch select_model to return a different name so mismatch fires.
    """
    import leopard44_kb.ingest.embedder as emb

    # Override select_model to return a model name that differs from 'm'
    monkeypatch.setattr(emb, "select_model", lambda: ("nomic-embed-text:v1.5", "v1.5"))

    # retrieve() should succeed (not raise)
    chunks, below_floor = retrieve(retrieval_db, "impeller interval", [], n=5)
    assert isinstance(chunks, list), "retrieve() should return a list despite model mismatch"

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # A warning containing the differing model name should appear on stderr
    assert "nomic-embed-text" in combined or "mismatch" in combined.lower(), (
        f"Expected a mismatch warning on stderr; got stdout={captured.out!r} stderr={captured.err!r}"
    )


# ---------------------------------------------------------------------------
# QUERY-01 / D-07: Refusal conditions
# ---------------------------------------------------------------------------


def test_empty_db_refusal(empty_db, fake_embedder):
    """retrieve on an empty DB returns ([], True) — D-07 refusal fires."""
    chunks, below_floor = retrieve(empty_db, "impeller interval", [], n=5)
    assert chunks == [], f"Expected empty chunk list, got {chunks}"
    assert below_floor is True, "Expected below_floor=True for empty DB"


def test_low_relevance_refusal():
    """is_below_relevance_floor returns True when best KNN distance exceeds floor and no BM25."""
    # Simulate very distant KNN results (distance > 1.0 = below cosine 0.5 for unit vectors)
    knn_far = [(1, 1.2), (2, 1.35)]
    bm25_empty: list = []
    assert is_below_relevance_floor(knn_far, bm25_empty) is True, (
        "Expected refusal when best KNN distance > floor and BM25 empty"
    )


# ---------------------------------------------------------------------------
# review fix #2: most_common_embedding_model helper
# ---------------------------------------------------------------------------


def test_most_common_embedding_model_returns_stored(retrieval_db):
    """most_common_embedding_model returns the model stored in the corpus."""
    model = most_common_embedding_model(retrieval_db)
    assert model == "m", f"Expected 'm' (stored model); got {model!r}"


def test_most_common_embedding_model_empty_db(empty_db):
    """most_common_embedding_model returns None on empty DB."""
    model = most_common_embedding_model(empty_db)
    assert model is None, f"Expected None for empty DB; got {model!r}"


# ---------------------------------------------------------------------------
# CR-01 regression: is_active=0 chunks must never surface in retrieval
# ---------------------------------------------------------------------------


def test_inactive_chunk_excluded_from_retrieval(retrieval_db, fake_embedder):
    """CR-01: a deactivated (is_active=0) chunk must not appear in KNN, BM25, or retrieve().

    Seeds chunk 4 at the exact fake_embedder query direction ([0.1]*384 → L2 distance 0),
    so it WOULD be the top KNN hit and a strong BM25 match if the is_active filter were
    missing. find_suppressed_chunks already requires is_active=1; knn_search/bm25_search
    must filter it too, or retired/superseded content leaks into a cited answer.
    """
    import struct

    def pack(v):
        return struct.pack("384f", *v)

    # Inactive shared chunk 4: same content/direction as the active impeller chunk,
    # but is_active=0 in BOTH chunks and vec_chunks.
    retrieval_db.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,"
        "page_end,content,content_hash,anchor_key,embedding_model,embedding_model_version,"
        "is_active) "
        "VALUES (4,1,'shared',2,'Cooling',47,47,"
        "'Replace the impeller every 200 hours. Part 4JH45-W001.','h4','ak4','m','v',0)"
    )
    retrieval_db.execute(
        "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
        "VALUES (4,'shared',1,'m',0,?)",
        (pack([0.1] * 384),),
    )
    retrieval_db.commit()

    # KNN: chunk 4 sits at distance 0 but is inactive — must be absent.
    query_vec = [0.1] * 384
    knn_ids = [cid for cid, _ in knn_search(retrieval_db, query_vec, [], k=20)]
    assert 4 not in knn_ids, f"Inactive chunk 4 leaked into KNN results: {knn_ids}"

    # BM25: 'impeller' matches chunk 4's content but it is inactive — must be absent.
    bm25_ids = [cid for cid, _ in bm25_search(retrieval_db, "impeller interval", [], pool=20)]
    assert 4 not in bm25_ids, f"Inactive chunk 4 leaked into BM25 results: {bm25_ids}"

    # End-to-end: retrieve() must never return the inactive chunk.
    chunks, _ = retrieve(retrieval_db, "impeller interval", [], n=5)
    ids = [c["id"] for c in chunks]
    assert 4 not in ids, f"Inactive chunk 4 leaked into retrieve() output: {ids}"


# ---------------------------------------------------------------------------
# BM25 OR-join fix (2026-06-17): natural-language questions must produce lexical
# hits. FTS5 treats space-separated terms as AND, which silently killed BM25 for
# any multi-word question (field UAT: "where is the liferaft" missed the manual's
# stowage section because BM25 returned nothing and KNN drifted).
# ---------------------------------------------------------------------------


def test_sanitize_multi_token_uses_or():
    """Multi-token queries join with OR so BM25 ranks by matching terms, not all-or-nothing."""
    assert sanitize_fts5_query("bilge pumps") == '"bilge" OR "pumps"'
    assert sanitize_fts5_query("where is the liferaft") == '"where" OR "is" OR "the" OR "liferaft"'
    # Single token is unchanged (no join) — preserves the hyphenated-part-number contract.
    assert sanitize_fts5_query("4JH45-W001") == '"4JH45-W001"'


def test_bm25_natural_language_query_returns_hits(retrieval_db):
    """An NL question with stopwords + a rare term matches the relevant chunk (was 0 under AND)."""
    # retrieval_db chunk 2 = "Change engine oil every 100 hours." (shared, Lubrication)
    ids = [cid for cid, _ in bm25_search(retrieval_db, "how often should I change the engine oil", [], pool=20)]
    assert len(ids) > 0, "NL query must produce BM25 hits after the OR fix"
    assert 2 in ids, f"Expected the oil-change chunk (id=2) in BM25 results: {ids}"


# ---------------------------------------------------------------------------
# D-15: Layer-authority weighting (Plan 13-03 RED tests)
# ---------------------------------------------------------------------------
# These tests import LAYER_AUTHORITY from retrieve.py, which does not yet
# exist — they fail with ImportError in RED state (Task 1). Once Task 3
# defines LAYER_AUTHORITY and applies the per-layer multiplier in retrieve(),
# all assertions should pass (GREEN state).
# ---------------------------------------------------------------------------


def _seed_authority_db(conn):
    """Seed a shared spec chunk + a community chat chunk for authority tests.

    Community chunk has a BETTER raw KNN embedding distance (closer to fake_embedder
    query direction [0.1]*384) than the shared chunk.  Without layer-authority
    weighting, the community chunk would rank first.  With LAYER_AUTHORITY
    (shared=1.2, community=1.0), the shared chunk wins despite the KNN disadvantage.
    """
    import struct
    def _pack(v):
        return struct.pack("384f", *v)

    conn.execute(
        "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
        "VALUES (30,'shared','pdf','shared/spec.pdf','hs30','Spec Sheet')"
    )
    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,"
        "content,content_hash,anchor_key,embedding_model,embedding_model_version) "
        "VALUES (30,30,'shared',0,'Specs',"
        "'LOA 12.98 m overall length Leopard 44 specification','hsc30','ak30','m','v')"
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
        "VALUES (30,'shared',30,'m',1,?)",
        (_pack([0.2] * 384),),  # further from query [0.1]*384 → worse KNN rank
    )
    conn.execute(
        "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
        "VALUES (31,'community','md','community/chat.md','hc31','Owners Chat')"
    )
    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,"
        "content,content_hash,anchor_key,embedding_model,embedding_model_version) "
        "VALUES (31,31,'community',0,'Chat',"
        "'Leopard 44 length in metres overall length maybe','hcc31','ak31','m','v')"
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
        "VALUES (31,'community',31,'m',1,?)",
        (_pack([0.11] * 384),),  # closer to query [0.1]*384 → better KNN rank
    )
    conn.commit()


def test_layer_authority_ordering():
    """RED: LAYER_AUTHORITY constant exists with shared > community > vessel weights.

    Fails with ImportError until Task 3 defines LAYER_AUTHORITY in retrieve.py.
    In GREEN: asserts the expected hierarchy and that all three keys are present.
    """
    from leopard44_kb.retrieve import LAYER_AUTHORITY  # noqa: PLC0415  # ImportError until Task 3

    assert "shared" in LAYER_AUTHORITY, "LAYER_AUTHORITY must define 'shared' weight"
    assert "community" in LAYER_AUTHORITY, "LAYER_AUTHORITY must define 'community' weight"
    assert "vessel" in LAYER_AUTHORITY, "LAYER_AUTHORITY must define 'vessel' weight"
    assert LAYER_AUTHORITY["shared"] > LAYER_AUTHORITY["community"], (
        f"shared ({LAYER_AUTHORITY['shared']}) must outrank community ({LAYER_AUTHORITY['community']})"
    )
    assert LAYER_AUTHORITY["community"] > LAYER_AUTHORITY["vessel"], (
        f"community ({LAYER_AUTHORITY['community']}) must outrank vessel ({LAYER_AUTHORITY['vessel']})"
    )


def test_layer_authority_shared_over_community(empty_db, fake_embedder):
    """RED: a shared spec chunk outranks a community chat chunk with better raw KNN score.

    The community chunk is CLOSER to the fake_embedder query vector ([0.11]*384 vs
    [0.2]*384 for the shared chunk), so without LAYER_AUTHORITY weighting the
    community chunk would rank first.  With LAYER_AUTHORITY (shared=1.2,
    community=1.0), the shared chunk wins.

    Fails with ImportError until Task 3 defines LAYER_AUTHORITY in retrieve.py.
    """
    from leopard44_kb.retrieve import LAYER_AUTHORITY  # noqa: PLC0415  # ImportError until Task 3

    _seed_authority_db(empty_db)

    chunks, below_floor = retrieve(empty_db, "length in metres Leopard 44", [], n=3)
    assert not below_floor, "Should not refuse query with seeded relevant chunks"
    assert len(chunks) > 0, "Expected at least one chunk returned"

    ids = [c["id"] for c in chunks]
    layers = [c["layer"] for c in chunks]

    # Shared spec chunk (id=30) must outrank community chat chunk (id=31)
    assert 30 in ids, f"Shared spec chunk (id=30) must appear in results; got {ids}"
    assert ids[0] == 30, (
        f"Shared spec (id=30, layer=shared, LAYER_AUTHORITY={LAYER_AUTHORITY['shared']}) "
        f"must rank first over community chat (id=31, LAYER_AUTHORITY={LAYER_AUTHORITY['community']}); "
        f"got ranking: {list(zip(ids, layers))}"
    )


def test_exact_part_retrieval_after_weighting(retrieval_db, fake_embedder):
    """RED: exact part/model-code query still surfaces the exact-match chunk after layer weighting.

    retrieval_db chunk 1 is shared-layer and contains '4JH45-W001' (exact part number).
    With LAYER_AUTHORITY (shared=1.2), chunk 1 benefits from the multiplier AND from D-05
    exact-token forcing.  This test serves as a regression anchor: if layer-authority
    weighting is implemented in a way that demotes exact-match chunks, this fails.

    Fails with ImportError until Task 3 defines LAYER_AUTHORITY in retrieve.py.
    In GREEN: asserts both that LAYER_AUTHORITY values are sane AND that the exact-match
    chunk appears in results.
    """
    from leopard44_kb.retrieve import LAYER_AUTHORITY  # noqa: PLC0415  # ImportError until Task 3

    # Sanity: exact-match chunk is shared-layer (benefits from LAYER_AUTHORITY multiplier)
    assert LAYER_AUTHORITY["shared"] >= 1.0, (
        f"shared LAYER_AUTHORITY ({LAYER_AUTHORITY['shared']}) must be >= 1.0 "
        "to not demote authoritative content"
    )

    chunks, below_floor = retrieve(retrieval_db, "4JH45-W001", [], n=3)
    assert not below_floor, "Exact part number query must not be refused"
    ids = [c["id"] for c in chunks]
    assert 1 in ids, (
        f"Exact-match chunk 1 (4JH45-W001, shared layer) must appear in results "
        f"after layer-authority weighting; got {ids}"
    )
