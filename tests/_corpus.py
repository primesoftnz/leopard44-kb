"""Shared retrieval-test seed corpus (IN-05 de-duplication).

The 3-chunk corpus and the ``pack`` embedding helper were previously copy-pasted
between the ``retrieval_db`` fixture in conftest.py and ``_seed_db`` in
test_query_cli.py, risking silent drift between the in-memory unit-test path and
the file-backed CLI integration path (IN-06 was a live instance of exactly that
risk). This module defines the corpus once; both call sites import ``seed_corpus``.

The connection is assumed to already have sqlite-vec loaded and the schema
migrated — the caller owns connection bootstrap (in-memory vs file-backed).

Corpus per 03-RESEARCH.md lines 1067-1117:
  source id=1: shared 'Yanmar 4JH45 Manual' (pdf)
  source id=2: vessel 'Maintenance Log' (log)
  chunk id=1: shared, impeller interval (p.47), content contains '4JH45-W001'
  chunk id=2: shared, oil change interval (p.55)
  chunk id=3: vessel, supersedes chunk 1 (owner note, no page)

vec_chunks vectors (review fix #1):
  chunk 1 and 2: near the fake_embedder query direction ([0.1]*384)
  chunk 3: far-direction vector ([1.0] + [0.0]*383) so it falls OUTSIDE the fused
           top-N when k is small, enabling the pool-wide pull-in test.

All three rows share embedding_model='m' so the mismatch test can monkeypatch
select_model to a different value and assert a warning is emitted.

include_inactive=True (IN-06) appends a 4th chunk (id=4, source id=3) whose
content WOULD match the impeller query but is marked is_active=0 in vec_chunks.
This lets the CLI integration suite exercise the CR-01 is_active filter
end-to-end (ask -> retrieve -> render) and assert the inactive source never
appears in the rendered Sources block.
"""
from __future__ import annotations

import sqlite3
import struct


def pack(v: list) -> bytes:
    """Pack a 384-float embedding into sqlite-vec's little-endian f32 blob."""
    return struct.pack("384f", *v)


def seed_corpus(conn: sqlite3.Connection, *, include_inactive: bool = False) -> None:
    """Seed the canonical 3-chunk retrieval corpus into an already-bootstrapped conn.

    The connection must already have sqlite-vec loaded and the schema migrated.
    Commits before returning. When ``include_inactive`` is True, also seeds an
    inactive (is_active=0) chunk whose content matches the impeller query, for the
    CR-01 filter regression at the CLI level (IN-06).
    """
    # Sources
    conn.execute(
        "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
        "VALUES (1,'shared','pdf','shared/yanmar.pdf','h1','Yanmar 4JH45 Manual')"
    )
    conn.execute(
        "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
        "VALUES (2,'vessel','log','data/docs/log.md','h2','Maintenance Log')"
    )

    # Shared chunk id=1: impeller interval (factory)
    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,"
        "page_end,content,content_hash,anchor_key,embedding_model,embedding_model_version) "
        "VALUES (1,1,'shared',0,'Cooling',47,47,"
        "'Replace the impeller every 200 hours. Part 4JH45-W001.','h','ak1','m','v')"
    )
    # Shared chunk id=2: oil change (factory, no override)
    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,"
        "page_end,content,content_hash,anchor_key,embedding_model,embedding_model_version) "
        "VALUES (2,1,'shared',1,'Lubrication',55,55,"
        "'Change engine oil every 100 hours.','h2','ak2','m','v')"
    )
    # Vessel chunk id=3: supersedes shared chunk id=1
    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,"
        "page_end,content,content_hash,anchor_key,embedding_model,embedding_model_version,"
        "supersedes_chunk_id) "
        "VALUES (3,2,'vessel',0,'Cooling',NULL,NULL,"
        "'Owner note: impeller replaced 2024-03-15 at 150h, used 4JH45-W001 from Burnsco.',"
        "'h3','ak3','m','v',1)"
    )

    # vec_chunks: chunk 1 and 2 cluster near the fake_embedder query direction ([0.1]*384).
    # chunk 3 (vessel superseder) is in a FAR direction — only first component non-zero —
    # so it can be driven outside the fused top-N to exercise the pool-wide pull-in test.
    far_vec = [1.0] + [0.0] * 383  # orthogonal to [0.1]*384 direction

    for cid, layer, src_id, v in [
        (1, "shared", 1, [0.1] * 384),
        (2, "shared", 1, [0.5] * 384),
        (3, "vessel", 2, far_vec),
    ]:
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
            "VALUES (?,?,?,'m',1,?)",
            (cid, layer, src_id, pack(v)),
        )

    if include_inactive:
        # IN-06: a deactivated chunk whose content WOULD match the impeller query.
        # Its vec row is is_active=0, so the CR-01 KNN/BM25 filter must exclude it
        # end-to-end. Distinct source (id=3) so we can assert its title/path never
        # appears in the rendered Sources block.
        conn.execute(
            "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
            "VALUES (3,'shared','pdf','shared/stale_impeller.pdf','h4','Stale Impeller Note')"
        )
        conn.execute(
            "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,"
            "page_end,content,content_hash,anchor_key,embedding_model,embedding_model_version,"
            "is_active) "
            "VALUES (4,3,'shared',0,'Cooling',12,12,"
            "'STALE: replace the impeller every 999 hours. Part 4JH45-W001.',"
            "'h5','ak4','m','v',0)"
        )
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
            "VALUES (4,'shared',3,'m',0,?)",
            (pack([0.1] * 384),),  # same direction as chunk 1 — would rank highly if active
        )

    conn.commit()
