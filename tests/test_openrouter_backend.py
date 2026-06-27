"""Hosted generation backend (Phase 7 alpha) — OpenRouter / OpenAI-compatible SSE.

The offline product always uses Ollama; this backend is env-gated via
L44_LLM_BACKEND=openrouter and exists only for the connected public alpha
(the DC VM's AVX-less CPU can't run a 7B fast enough — 07-03 latency gate).
Embeddings are unaffected here (they always stay on local nomic).
"""
from __future__ import annotations

import httpx
import pytest

from leopard44_kb.answer import (
    DEFAULT_OPENROUTER_MODEL,
    select_generation_model,
    stream_generate,
)


class _FakeSSEResponse:
    """Mimics httpx.stream(...)'s context-managed streaming response for OpenAI SSE."""

    def __init__(self, status_code: int = 200, lines: list[str] | None = None, body: bytes = b""):
        self.status_code = status_code
        self._lines = lines or []
        self._body = body

    def iter_lines(self):
        yield from self._lines

    def iter_bytes(self):
        yield self._body

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_select_generation_model_openrouter_default(monkeypatch):
    """Backend=openrouter returns the default hosted model + 'openrouter' label."""
    monkeypatch.setenv("L44_LLM_BACKEND", "openrouter")
    model, label = select_generation_model()
    assert model == DEFAULT_OPENROUTER_MODEL
    assert label == "openrouter"


def test_select_generation_model_openrouter_override(monkeypatch):
    """L44_LLM_MODEL overrides the hosted model name."""
    monkeypatch.setenv("L44_LLM_BACKEND", "openrouter")
    monkeypatch.setenv("L44_LLM_MODEL", "anthropic/claude-haiku-4.5")
    model, label = select_generation_model()
    assert model == "anthropic/claude-haiku-4.5"
    assert label == "openrouter"


def test_openrouter_streams_tokens(monkeypatch):
    """stream_generate parses OpenAI SSE deltas, skips keep-alives, yields tokens."""
    monkeypatch.setenv("L44_LLM_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-cap-key")
    lines = [
        ": OPENROUTER PROCESSING",  # keep-alive comment — must be skipped
        'data: {"choices":[{"delta":{"content":"The hull"}}]}',
        "",  # blank line — must be skipped
        'data: {"choices":[{"delta":{"content":" is 13.5m"}}]}',
        'data: {"choices":[{"delta":{"content":" [1]"}}]}',
        'data: {"choices":[],"usage":{"completion_tokens":7}}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _FakeSSEResponse(lines=lines))

    tokens = list(stream_generate("google/gemini-2.5-flash", "sys", "user"))
    assert "".join(tokens) == "The hull is 13.5m [1]"


def test_openrouter_sends_bearer_key_and_model(monkeypatch):
    """The Authorization bearer + model land in the request, key never defaulted."""
    monkeypatch.setenv("L44_LLM_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-123")
    captured: dict = {}

    def _fake_stream(method, url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeSSEResponse(lines=["data: [DONE]"])

    monkeypatch.setattr(httpx, "stream", _fake_stream)
    list(stream_generate("google/gemini-2.5-flash", "sys", "user"))

    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-secret-123"
    assert captured["json"]["model"] == "google/gemini-2.5-flash"
    assert captured["json"]["stream"] is True


def test_openrouter_missing_key_hard_fail(monkeypatch):
    """Backend=openrouter with no key raises a clear RuntimeError (no silent fallback)."""
    monkeypatch.setenv("L44_LLM_BACKEND", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match=r"OPENROUTER_API_KEY"):
        list(stream_generate("google/gemini-2.5-flash", "sys", "user"))


def test_openrouter_http_error_hard_fail(monkeypatch):
    """A 402 (credit cap hit) surfaces as RuntimeError with the status code."""
    monkeypatch.setenv("L44_LLM_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **kw: _FakeSSEResponse(status_code=402, body=b'{"error":"insufficient credits"}'),
    )
    with pytest.raises(RuntimeError, match=r"HTTP 402"):
        list(stream_generate("google/gemini-2.5-flash", "sys", "user"))


def test_offline_default_does_not_use_openrouter(monkeypatch):
    """With no backend env set, stream_generate uses the Ollama path (hard-fail hint)."""
    monkeypatch.delenv("L44_LLM_BACKEND", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def _raise(*a, **kw):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "stream", _raise)
    # Ollama-path error mentions 'ollama serve', proving openrouter was NOT taken.
    with pytest.raises(RuntimeError, match=r"ollama serve"):
        list(stream_generate("qwen2.5:7b-instruct-q4_K_M", "sys", "user"))
