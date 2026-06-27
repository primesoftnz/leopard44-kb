# Live Ollama integration test — excluded from the default run by the 'live' marker.
# Run: uv run pytest tests/test_query_live.py -m live
"""Live end-to-end smoke test for the ask command (QUERY-06).

Requires: Ollama running + qwen2.5:7b-instruct-q4_K_M (or 3b variant) pulled.
Excluded from the default pytest run by `addopts = -m "not live"` in pyproject.toml.
"""
from __future__ import annotations

import sqlite3
import struct
import time

import pytest
import sqlite_vec
from typer.testing import CliRunner

from leopard44_kb.cli import app
from leopard44_kb.schema import apply_migrations

runner = CliRunner()


@pytest.mark.live
def test_ask_end_to_end_live(tmp_path, monkeypatch):
    """Live smoke test: ingest a tiny fixture with real Ollama, run l44 ask, assert
    exit 0 + a citation block + wall-clock latency < 10s on a warm model (QUERY-06).

    Run: uv run pytest tests/test_query_live.py -m live
    """
    # Bootstrap a file-backed DB with a tiny shared corpus (no ingest pipeline needed)
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
        "VALUES (1,'shared','log','shared/test.md','h1','Live Test Doc')"
    )

    # We need a real embedding for the chunk — use Ollama embed to get one
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

    # Invoke ask and measure wall-clock time
    start = time.monotonic()
    result = runner.invoke(app, ["ask", "What is the impeller replacement interval?"])
    elapsed = time.monotonic() - start

    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 from live ask; got {result.exit_code}: {combined}"
    )
    assert "Sources:" in combined, f"Expected citation block in live output: {combined!r}"
    assert elapsed < 10.0, (
        f"QUERY-06: expected warm-model latency < 10s; got {elapsed:.1f}s. "
        f"Ensure qwen2.5:7b-instruct-q4_K_M is already loaded (warm model)."
    )
