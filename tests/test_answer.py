# RED state until Phase 3 Wave 2 implementation. Imports from leopard44_kb.answer fail until
# production code lands in plan 03-03. Collection failure with ModuleNotFoundError is expected.
"""Tests for QUERY-01/02, D-06/D-07/D-08/D-09: generation model selection, grounding prompt,
citation validation, citation rendering, streaming order, hard-fail error handling.

Per-requirement verification map source: .planning/phases/03-query-engine/03-VALIDATION.md
Review fixes covered: #5 (num_predict cap), #6 (REFUSAL_MESSAGE single source of truth).
"""
from __future__ import annotations

import pytest

from leopard44_kb.answer import (
    DEFAULT_NUM_PREDICT,
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    build_user_message,
    render_citation_block,
    select_generation_model,
    stream_generate,
    validate_citations,
)


# ---------------------------------------------------------------------------
# D-08: Generation model selection
# ---------------------------------------------------------------------------


def test_gen_model_16gb():
    """select_generation_model(16) returns qwen2.5:7b-instruct-q4_K_M."""
    model, label = select_generation_model(16)
    assert model == "qwen2.5:7b-instruct-q4_K_M", f"Unexpected model for 16GB: {model!r}"


def test_gen_model_8gb():
    """select_generation_model(8) returns qwen2.5:3b-instruct-q4_K_M (smaller, faster)."""
    model, label = select_generation_model(8)
    assert model == "qwen2.5:3b-instruct-q4_K_M", f"Unexpected model for 8GB: {model!r}"


# ---------------------------------------------------------------------------
# QUERY-02 / D-06: Citation block rendering
# ---------------------------------------------------------------------------


def test_select_generation_model_config_first(monkeypatch, tmp_path):
    """select_generation_model() returns the model from config when config file exists.

    Called with NO args — exercises the config-first branch introduced in D-03
    without disturbing existing positional ram_gb tests.
    RED until 06-02 retrofits select_generation_model() to be config-first.
    """
    import json

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "tier": "gpu",
        "generation_model": "qwen2.5:14b-instruct-q4_K_M",
        "embedding_model": "nomic-embed-text:v1.5",
        "embedding_model_version": "v1.5",
    }))
    monkeypatch.setenv("L44_CONFIG", str(cfg))
    from leopard44_kb.answer import select_generation_model
    model, tier = select_generation_model()
    assert model == "qwen2.5:14b-instruct-q4_K_M", (
        f"Expected config-first select_generation_model() to return "
        f"'qwen2.5:14b-instruct-q4_K_M', got {model!r}"
    )


def test_citation_block_from_metadata():
    """render_citation_block builds from chunk dicts, not LLM text."""
    chunks = [
        {"layer": "shared", "title": "Yanmar 4JH45 Manual", "path": "shared/yanmar.pdf",
         "page_start": 47, "page_end": 47, "content": "...", "_suppression_note": ""},
        {"layer": "vessel", "title": "Maintenance Log", "path": "data/docs/log.md",
         "page_start": None, "page_end": None, "content": "...", "_suppression_note": ""},
    ]
    block = render_citation_block(chunks)
    assert "Sources:" in block, f"Expected 'Sources:' header in block: {block!r}"
    assert "[1]" in block and "[2]" in block, f"Expected [1] and [2] in block: {block!r}"
    assert "Yanmar 4JH45 Manual" in block, f"Expected source title in block: {block!r}"
    assert "Maintenance Log" in block, f"Expected source title in block: {block!r}"


def test_citation_format():
    """Citation block follows 'layer: source, p.N' format for shared and vessel chunks."""
    chunks = [
        {"layer": "shared", "title": "Yanmar 4JH45 Manual", "path": "shared/yanmar.pdf",
         "page_start": 47, "page_end": 47, "content": "...", "_suppression_note": ""},
        {"layer": "vessel", "title": "Maintenance Log", "path": "data/docs/log.md",
         "page_start": None, "page_end": None, "content": "...", "_suppression_note": ""},
    ]
    block = render_citation_block(chunks)
    # Shared entry must include layer prefix and page reference
    assert "shared:" in block, f"Expected 'shared:' prefix: {block!r}"
    assert "p.47" in block, f"Expected 'p.47' page ref: {block!r}"
    # Vessel entry must include layer prefix
    assert "vessel:" in block, f"Expected 'vessel:' prefix: {block!r}"


# ---------------------------------------------------------------------------
# QUERY-02 / review fix #4: Citation validation
# ---------------------------------------------------------------------------


def test_citation_validation_out_of_range():
    """validate_citations returns out-of-range [n] numbers."""
    invalid = validate_citations("see [9]", num_chunks=2)
    assert 9 in invalid, f"Expected [9] flagged as out-of-range; got {invalid}"


def test_citation_validation_in_range_ok():
    """validate_citations returns empty list when all citations are valid."""
    invalid = validate_citations("see [1] and [2]", num_chunks=2)
    assert invalid == [], f"Expected no invalid citations; got {invalid}"


# ---------------------------------------------------------------------------
# D-02 / review fix #6: Suppression note in build_user_message
# ---------------------------------------------------------------------------


def test_suppression_note_in_context():
    """build_user_message includes the _suppression_note when set on a chunk."""
    chunks = [
        {"layer": "shared", "title": "Yanmar 4JH45 Manual", "path": "shared/yanmar.pdf",
         "page_start": 47, "page_end": 47, "content": "Replace the impeller every 200 hours.",
         "_suppression_note": " [NOTE: superseded by vessel note]"},
    ]
    msg = build_user_message("What is the impeller interval?", chunks)
    assert "[NOTE: superseded by vessel note]" in msg, (
        f"Expected suppression note in user message: {msg!r}"
    )


# ---------------------------------------------------------------------------
# QUERY-02 / review fix #6: REFUSAL_MESSAGE single source of truth
# ---------------------------------------------------------------------------


def test_refusal_message_single_source():
    """REFUSAL_MESSAGE is defined in answer.py and is a non-empty string.

    retrieve.py and cli.py must import this constant from answer.py (review fix #6),
    not define their own copy.
    """
    assert isinstance(REFUSAL_MESSAGE, str), "REFUSAL_MESSAGE must be a str"
    assert len(REFUSAL_MESSAGE) > 0, "REFUSAL_MESSAGE must be non-empty"


# ---------------------------------------------------------------------------
# review fix #5: DEFAULT_NUM_PREDICT constant
# ---------------------------------------------------------------------------


def test_default_num_predict_is_75():
    """DEFAULT_NUM_PREDICT is 75 — retained as the conservative resolution floor."""
    assert DEFAULT_NUM_PREDICT == 75, (
        f"Expected DEFAULT_NUM_PREDICT=75 as the fallback floor; got {DEFAULT_NUM_PREDICT}"
    )


# ---------------------------------------------------------------------------
# select_num_predict: tier-scaled generation cap (ask-truncation fix 2026-06-09)
# ---------------------------------------------------------------------------


def test_select_num_predict_maps_config_tier_keys(monkeypatch):
    """The config-first path returns tier keys ('8gb'/'16gb'/'gpu') → tier caps."""
    from leopard44_kb.answer import select_num_predict
    from leopard44_kb.config import TIER_NUM_PREDICT

    monkeypatch.delenv("L44_NUM_PREDICT", raising=False)
    for tier, expected in TIER_NUM_PREDICT.items():
        # model arg is irrelevant when the tier key matches directly
        assert select_num_predict(tier, "qwen2.5:7b-instruct-q4_K_M") == expected


def test_select_num_predict_maps_ram_fallback_version_labels(monkeypatch):
    """The RAM-detection path returns version labels — map via the model tag."""
    from leopard44_kb.answer import select_num_predict
    from leopard44_kb.config import TIER_NUM_PREDICT

    monkeypatch.delenv("L44_NUM_PREDICT", raising=False)
    assert select_num_predict("3b-q4_K_M", "qwen2.5:3b-instruct-q4_K_M") == TIER_NUM_PREDICT["8gb"]
    assert select_num_predict("7b-q4_K_M", "qwen2.5:7b-instruct-q4_K_M") == TIER_NUM_PREDICT["16gb"]
    assert select_num_predict("unknown", "qwen2.5:14b-instruct-q4_K_M") == TIER_NUM_PREDICT["gpu"]


def test_select_num_predict_unknown_falls_back_to_default(monkeypatch):
    """An unresolvable tier/model returns the conservative DEFAULT_NUM_PREDICT floor."""
    from leopard44_kb.answer import select_num_predict

    monkeypatch.delenv("L44_NUM_PREDICT", raising=False)
    assert select_num_predict("unknown", "some-exotic-model:latest") == DEFAULT_NUM_PREDICT


def test_select_num_predict_env_override_wins(monkeypatch):
    """L44_NUM_PREDICT overrides the tier cap unconditionally (on-demand knob)."""
    from leopard44_kb.answer import select_num_predict

    monkeypatch.setenv("L44_NUM_PREDICT", "900")
    # Even a known tier with a smaller cap is overridden.
    assert select_num_predict("8gb", "qwen2.5:3b-instruct-q4_K_M") == 900
    # Negative sentinels pass through (Ollama: -1 = infinite, -2 = fill context).
    monkeypatch.setenv("L44_NUM_PREDICT", "-1")
    assert select_num_predict("gpu", "qwen2.5:14b-instruct-q4_K_M") == -1


# ---------------------------------------------------------------------------
# QUERY-01 / hard-fail error handling
# ---------------------------------------------------------------------------


def test_ollama_unreachable_hard_fail(monkeypatch):
    """stream_generate raises RuntimeError with 'ollama serve' hint when Ollama unreachable."""
    import httpx

    def _raise(*a, **kw):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "stream", _raise)
    with pytest.raises(RuntimeError, match=r"ollama serve"):
        list(stream_generate("qwen2.5:7b-instruct-q4_K_M", "sys", "user"))


def test_model_not_pulled(monkeypatch):
    """stream_generate raises RuntimeError with 'ollama pull' hint when model not found (404)."""
    import httpx

    class _FakeResponse:
        status_code = 404

        def iter_bytes(self):
            yield b'{"error":"model \'qwen2.5:7b-instruct-q4_K_M\' not found"}'

        def iter_lines(self):
            yield '{"error":"model not found"}'

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _FakeResponse())
    with pytest.raises(RuntimeError, match=r"ollama pull"):
        list(stream_generate("qwen2.5:7b-instruct-q4_K_M", "sys", "user"))
