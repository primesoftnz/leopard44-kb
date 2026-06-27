# RED state until Phase 2 implementation (see 02-VALIDATION.md). Imports from leopard44_kb.ingest.* fail until production code lands.
"""Tests for Ollama embedder failure modes, dimension assertions, and model version storage (02-VALIDATION.md).

Embedder failure-mode tests monkeypatch httpx.post directly — they exercise
embed_texts' own error handling, so they MUST NOT use the fake_embedder fixture.
"""
from __future__ import annotations

import pytest

from leopard44_kb.ingest.embedder import embed_texts, select_model


def test_ollama_unreachable_hard_fail(monkeypatch):
    """D-09: embed_texts raises RuntimeError with setup hint when Ollama is unreachable."""
    import httpx

    def _raise(*a, **kw):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "post", _raise)
    with pytest.raises(RuntimeError, match=r"ollama serve"):
        embed_texts(["test text"], "nomic-embed-text:v1.5")


def test_model_version_stored(ingest_db, fake_embedder, tmp_path):
    """Integration: ingest via fake_embedder; assert chunks.embedding_model + embedding_model_version are set per chunk."""
    from pathlib import Path
    from leopard44_kb.ingest import ingest_file

    data_dir = tmp_path / "data" / "logs"
    data_dir.mkdir(parents=True)
    note = data_dir / "model_version_test.md"
    note.write_text("# Test\nSome content to embed.\n")

    result = ingest_file(note, layer="vessel", conn=ingest_db)
    assert result == "ok"

    rows = ingest_db.execute(
        "SELECT embedding_model, embedding_model_version FROM chunks"
    ).fetchall()
    assert len(rows) > 0, "No chunks stored after ingest"
    for row in rows:
        assert row["embedding_model"] == "nomic-embed-text:v1.5"
        assert row["embedding_model_version"] == "v1.5"


def test_embed_timeout_raises(monkeypatch):
    """embed_texts raises RuntimeError (not httpx.TimeoutException) when Ollama times out."""
    import httpx

    def _timeout(*a, **kw):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "post", _timeout)
    with pytest.raises(RuntimeError, match=r"timeout|retry|Ollama"):
        embed_texts(["test"], "nomic-embed-text:v1.5")


def test_embed_non_2xx_raises(monkeypatch):
    """embed_texts raises RuntimeError on a non-2xx HTTP response (e.g. 500 Internal Error)."""
    import httpx

    class _FakeResp:
        status_code = 500
        text = "Internal Server Error"

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "500 error", request=None, response=self
            )

        def json(self):
            return {}

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResp())
    with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
        embed_texts(["test"], "nomic-embed-text:v1.5")


def test_embed_404_model_missing(monkeypatch):
    """embed_texts raises RuntimeError naming 'ollama pull <model>' when model returns 404 not found."""
    import httpx

    class _FakeResp:
        status_code = 404
        text = "model not found"

        def raise_for_status(self):
            pass

        def json(self):
            return {}

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResp())
    with pytest.raises(RuntimeError, match=r"ollama pull"):
        embed_texts(["test"], "nomic-embed-text:v1.5")


def test_embed_malformed_json_raises(monkeypatch):
    """embed_texts raises RuntimeError when the response body is non-JSON or missing 'embeddings' key."""
    import httpx

    class _FakeResp:
        status_code = 200
        text = "not json at all"

        def raise_for_status(self):
            pass

        def json(self):
            # Return dict missing the 'embeddings' key
            return {"result": "something else"}

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResp())
    with pytest.raises(RuntimeError, match=r"embeddings|malformed|response"):
        embed_texts(["test"], "nomic-embed-text:v1.5")


def test_embed_count_mismatch_raises(monkeypatch):
    """embed_texts raises RuntimeError when len(embeddings) != len(texts)."""
    import httpx

    class _FakeResp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            # Return only 1 embedding for 2 input texts
            return {"embeddings": [[0.1] * 384]}

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResp())
    with pytest.raises(RuntimeError, match=r"count|mismatch|expected"):
        embed_texts(["text one", "text two"], "nomic-embed-text:v1.5")


def test_dimension_mismatch_rejected(monkeypatch):
    """A returned vector whose len != 384 raises RuntimeError BEFORE any sqlite-vec insert.

    This is the key 384-dimension guard from the Codex review — the assertion must
    fire in embed_texts (or at the earliest point before storage), not in sqlite-vec.
    """
    import httpx

    class _FakeResp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            # Return 100-dim vector instead of 384
            return {"embeddings": [[0.5] * 100]}

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResp())
    with pytest.raises(RuntimeError, match=r"384|dimension"):
        embed_texts(["test"], "nomic-embed-text:v1.5")


def test_select_model_config_first(monkeypatch, tmp_path):
    """select_model() returns the embedding model from config when config file exists.

    Uses an 8gb config on a machine that likely has >=14GB RAM — so the RAM-based
    fallback would return nomic-embed-text:v1.5, but the config says all-minilm:latest.
    This proves the config-first branch is active, not the RAM fallback.

    RED until 06-02 retrofits select_model() to be config-first (D-03).
    """
    import json

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "tier": "8gb",
        "generation_model": "qwen2.5:3b-instruct-q4_K_M",
        "embedding_model": "all-minilm:latest",
        "embedding_model_version": "latest",
    }))
    monkeypatch.setenv("L44_CONFIG", str(cfg))
    from leopard44_kb.ingest.embedder import select_model
    model, version = select_model()
    assert model == "all-minilm:latest", (
        f"Expected config-first select_model() to return 'all-minilm:latest' (from 8gb config), "
        f"got {model!r} — the config-first branch is not active; RAM fallback returned a different model"
    )
    assert version == "latest", (
        f"Expected config-first select_model() version 'latest', got {version!r}"
    )


def test_empty_text_list_handled(monkeypatch):
    """embed_texts([]) returns [] without calling Ollama (no network call)."""
    import httpx

    called = []

    def _should_not_be_called(*a, **kw):
        called.append(True)
        raise AssertionError("Ollama should not be called for empty input")

    monkeypatch.setattr(httpx, "post", _should_not_be_called)
    result = embed_texts([], "nomic-embed-text:v1.5")
    assert result == [], f"Expected [], got {result!r}"
    assert not called, "Ollama was called for empty input list"
