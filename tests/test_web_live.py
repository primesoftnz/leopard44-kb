# Live Ollama integration test — excluded from the default run by the 'live' marker.
# Run: uv run pytest tests/test_web_live.py -m live
"""Live end-to-end smoke test for the web UI query endpoint (UI-01 + UI-02).

Requires: Ollama running + qwen2.5:7b-instruct-q4_K_M (or 3b variant) pulled.
Excluded from the default pytest run by `addopts = -m "not live"` in pyproject.toml.
create_app is imported INSIDE the test function body (collection-safe RED per review fix).
"""
from __future__ import annotations

import sqlite3
import struct
import time

import pytest
import sqlite_vec
from leopard44_kb.schema import apply_migrations


@pytest.mark.live
def test_web_query_end_to_end_live(tmp_path, monkeypatch):
    """Live smoke test: POST /query to a real TestClient with a real Ollama backend.

    Bootstraps a file-backed DB with a real embedding (mirrors test_query_live.py),
    creates the FastAPI app via create_app(), streams a query, asserts:
    - at least one 'token' event arrives
    - wall-clock elapsed < 10.0s (QUERY-06 warm-model budget)

    Run: uv run pytest tests/test_web_live.py -m live
    """
    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    db_path = tmp_path / "live_test.db"
    monkeypatch.setenv("L44_DB", str(db_path))

    def pack(v: list) -> bytes:
        return struct.pack("384f", *v)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)

    # Insert a single shared source + chunk with a known fact
    conn.execute(
        "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
        "VALUES (1,'shared','log','shared/test.md','h1','Live Web Test Doc')"
    )

    # Use real Ollama embedding for the chunk (mirrors test_query_live.py)
    from leopard44_kb.ingest.embedder import embed_texts, select_model

    model, _ = select_model()
    fact_text = "The Yanmar 4JH45 engine impeller should be replaced every 200 hours."
    embeddings = embed_texts([fact_text], model)
    embedding = embeddings[0]

    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,"
        "page_end,content,content_hash,anchor_key,embedding_model,embedding_model_version) "
        "VALUES (1,1,'shared',0,'Engine',1,1,?,?,'ak1',?,?)",
        (fact_text, "h1", model, "live"),
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
        "VALUES (1,'shared',1,?,1,?)",
        (model, pack(embedding)),
    )
    conn.commit()
    conn.close()

    # Stream a query via TestClient and measure wall-clock time
    start = time.monotonic()

    token_events: list[str] = []
    current_event = "message"

    with TestClient(create_app()).stream(
        "POST", "/query",
        json={"question": "What is the impeller replacement interval?", "layer": "all"},
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: "):
                if current_event == "token":
                    token_events.append(line[6:])
                current_event = "message"

    elapsed = time.monotonic() - start

    assert token_events, (
        f"Expected at least one 'token' event from live /query; got none. "
        f"Elapsed: {elapsed:.1f}s. Ensure Ollama is running and the model is pulled."
    )
    assert elapsed < 10.0, (
        f"QUERY-06: expected warm-model latency < 10s; got {elapsed:.1f}s. "
        f"Ensure qwen2.5:7b-instruct-q4_K_M is already loaded (warm model)."
    )
