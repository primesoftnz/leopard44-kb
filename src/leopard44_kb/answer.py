"""LLM generation: model selection, grounding prompt, streaming, citation validation.

This module is the canonical owner of REFUSAL_MESSAGE and DEFAULT_NUM_PREDICT (review fixes
#5 and #6). retrieve.py and cli.py import these constants from here; they do NOT define
their own copies.

Note on floor constants: REFUSAL_DISTANCE_FLOOR (the relevance-threshold numeric) lives in
retrieve.py (retrieval-side decision). The refusal TEXT lives here in REFUSAL_MESSAGE.

Pitfall 7 (first-query latency): The first `l44 ask` after machine restart or Ollama
idle timeout triggers model load (~3-8s before the first token appears). QUERY-06 (<10s)
applies to warm-model queries where the model is already loaded.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Generator

import httpx

from leopard44_kb.ingest.embedder import detect_ram_gb

# ---------------------------------------------------------------------------
# Module-level constants — answer.py is the CANONICAL owner (review fix #6).
# ---------------------------------------------------------------------------

# Conservative fallback cap, used only when the active tier can't be resolved to
# a TIER_NUM_PREDICT entry (unknown/hand-edited config, exotic model tag). Known
# tiers get a larger, capability-scaled cap via select_num_predict() — the flat 75
# here truncated real answers at ~2 bullets, so it survives only as the floor.
# Original rationale: at 8-12 tok/s on a 16GB CPU, 75 tokens kept generation within
# the QUERY-06 <10s warm-model budget (review fix #5).
DEFAULT_NUM_PREDICT: int = 75

DEFAULT_TEMPERATURE: float = 0.15

# Ollama context window: fits 5 chunks × ~200 tokens + prompt overhead + answer.
NUM_CTX: int = 4096

# ---------------------------------------------------------------------------
# Hosted generation backend (Phase 7 public alpha) — env-gated.
#
# The OFFLINE product always uses Ollama (the default). The hosted backend exists
# ONLY for connected deployments where the host CPU cannot run a 7B model fast
# enough (see Phase 7 latency gate — AVX-less Westmere constraint).
# It is selected purely via environment so it has ZERO effect on a normal offline
# install: set L44_LLM_BACKEND=openrouter to route generation to an
# OpenAI-compatible hosted endpoint. Embeddings ALWAYS stay local (nomic) — the
# store is built with nomic-embed-text:v1.5 and a hosted embedder would break
# vector search.
DEFAULT_OPENROUTER_MODEL: str = "google/gemini-2.5-flash"
DEFAULT_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
# Hosted gen is fast, so we are not bound by the QUERY-06 CPU latency cap (75);
# allow a fuller answer for the public-alpha trial users. Override with
# L44_LLM_MAX_TOKENS.
DEFAULT_HOSTED_MAX_TOKENS: int = 512


def _llm_backend() -> str:
    """Generation backend: 'ollama' (default, offline) or 'openrouter' (hosted alpha).

    Read at call time (not import time) so the systemd env and tests can switch it.
    """
    return os.environ.get("L44_LLM_BACKEND", "ollama").strip().lower()

# Strict grounding system prompt (D-06, D-07). {n_chunks} is formatted by the caller.
SYSTEM_PROMPT: str = """\
You are a vessel knowledge assistant. Answer ONLY using the provided context chunks.
Do not use general knowledge; if the context does not contain the answer, say so.

For every claim you make, add an inline citation marker like [1] or [2] that refers
to the numbered context chunk below. Only use citation numbers that exist in the
provided context (1 through {n_chunks}). Do not invent sources.

Keep your answer concise and factual. Focus on actionable information.\
"""

# The SINGLE definition of the D-07 refusal text across the codebase.
# retrieve.py imports this; cli.py imports this. Neither module re-declares it.
REFUSAL_MESSAGE: str = (
    "I don't have information about that in your knowledge base. "
    "Try ingesting more documents or rephrasing your question."
)

# ---------------------------------------------------------------------------
# Citation regex — module-level for efficiency.
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(\d+)\]")


# ---------------------------------------------------------------------------
# D-08: Generation model selection
# ---------------------------------------------------------------------------


def select_generation_model(ram_gb: float | None = None) -> tuple[str, str]:
    """Return (ollama_model_tag, version_label) for the generation model.

    Config-first (D-01/D-03): reads L44_CONFIG at call time. If a config file
    exists and contains 'generation_model', that value is returned without touching
    RAM detection — preserving the install-time tier choice (including GPU/14B).

    NOTE: The GPU tier (qwen2.5:14b-instruct-q4_K_M) is ONLY reachable via config.
    RAM detection never selects 14B — it returns 7b at >=14GB, 3b below. Users
    wanting the 14B model must run setup with --tier gpu to write the config.

    When config is absent, falls back to the RAM-based path (behaviour unchanged):
      RAM >= 14 GB → qwen2.5:7b-instruct-q4_K_M (4.7 GB, ~8-12 tok/s CPU)
      RAM <  14 GB → qwen2.5:3b-instruct-q4_K_M (1.9 GB, faster CPU inference)

    Model tags verified against ollama.com/library/qwen2.5/tags, 2026-05-29
    (Assumption A3 — document exact tags in setup.sh; Phase 6 checks availability).

    Args:
        ram_gb: Override RAM amount in GB for the fallback RAM path. If None and
                no config is present, detect_ram_gb() is called. Has no effect
                when a config file exists (config-first takes precedence).

    Returns:
        Tuple of (ollama_model_tag, version_label).
    """
    # Phase 7 alpha: when the hosted backend is selected, generation runs on a
    # remote OpenAI-compatible model, not an Ollama tag. Return that model name
    # (env-overridable) so app.py/cli.py pass it straight through to
    # stream_generate. Offline installs never set this var and fall through.
    if _llm_backend() == "openrouter":
        return (
            os.environ.get("L44_LLM_MODEL", DEFAULT_OPENROUTER_MODEL),
            "openrouter",
        )
    # D-01/D-03: config-first; RAM fallback only when absent
    from leopard44_kb.config import load_config
    cfg = load_config()
    if cfg and "generation_model" in cfg:
        return (cfg["generation_model"], cfg.get("tier", "unknown"))
    # NOTE: GPU tier (14B) only reached via config; RAM detection never selects 14B
    if ram_gb is None:
        ram_gb = detect_ram_gb()
    if ram_gb >= 14:
        return ("qwen2.5:7b-instruct-q4_K_M", "7b-q4_K_M")
    return ("qwen2.5:3b-instruct-q4_K_M", "3b-q4_K_M")


def select_num_predict(tier_label: str, model: str) -> int:
    """Resolve the Ollama generation token cap (num_predict) for the active tier.

    The flat 75-token DEFAULT_NUM_PREDICT truncated real answers at ~2 bullets.
    This scales the cap with tier capability so CPU installs still get a complete
    answer (at some latency cost) and GPU installs get a fuller one.

    Resolution order (all read at call time):
      1. L44_NUM_PREDICT env override — wins unconditionally. A negative value
         is passed straight through (Ollama treats -1 = infinite, -2 = fill-ctx),
         so this is the on-demand "give me a full answer" knob.
      2. TIER_NUM_PREDICT[tier_label] — the config-first path returns tier keys
         ('8gb'/'16gb'/'gpu') directly.
      3. Model-tag heuristic — the RAM-detection path returns version labels
         ('3b-q4_K_M'/'7b-q4_K_M'), not tier keys, so map 3b/7b/14b in the
         tier_label+model blob to the matching tier cap.
      4. DEFAULT_NUM_PREDICT — conservative floor when nothing else matches.

    Note: the openrouter backend ignores num_predict entirely (it uses
    L44_LLM_MAX_TOKENS / DEFAULT_HOSTED_MAX_TOKENS), so this only affects the
    offline Ollama path.

    Args:
        tier_label: The second element of select_generation_model()'s return.
        model: The Ollama model tag (first element of select_generation_model()).

    Returns:
        The num_predict token cap to pass to stream_generate.
    """
    env = os.environ.get("L44_NUM_PREDICT")
    if env is not None and env.strip():
        return int(env)  # operator override; let a typo raise ValueError loudly

    from leopard44_kb.config import TIER_NUM_PREDICT

    if tier_label in TIER_NUM_PREDICT:
        return TIER_NUM_PREDICT[tier_label]

    blob = f"{tier_label} {model}"
    if "14b" in blob:
        return TIER_NUM_PREDICT["gpu"]
    if "7b" in blob:
        return TIER_NUM_PREDICT["16gb"]
    if "3b" in blob:
        return TIER_NUM_PREDICT["8gb"]
    return DEFAULT_NUM_PREDICT


# ---------------------------------------------------------------------------
# Prompt assembly (D-06, D-07)
# ---------------------------------------------------------------------------


def build_user_message(question: str, chunks: list[dict]) -> str:
    """Assemble the numbered-context user message for the LLM.

    Each chunk is numbered [1]..[N]. The header line includes layer, title/path,
    optional page reference, and any _suppression_note from the retrieval layer
    (D-02 annotation reaches the LLM context).

    Args:
        question: The user's natural-language question.
        chunks: List of chunk dicts from fetch_chunk_metadata, with keys:
            layer, title (optional), path, page_start (optional),
            _suppression_note (optional).

    Returns:
        Formatted string: context header + numbered chunks + question.
    """
    lines = [f"Context chunks (cite these as [1]–[{len(chunks)}]):"]
    for i, chunk in enumerate(chunks, 1):
        layer = chunk["layer"]
        title = chunk.get("title") or chunk.get("path", "unknown")
        page = chunk.get("page_start")
        # page_start is a nullable INTEGER; page 0 (0-indexed/cover) is valid,
        # so test against None, not truthiness (WR-04).
        loc = f", p.{page}" if page is not None else ""
        note = chunk.get("_suppression_note", "")
        header = f"[{i}] {layer}: {title}{loc}{note}"
        lines.append(f"\n{header}\n{chunk['content']}")
    lines.append(f"\nQuestion: {question}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# QUERY-02 / D-06: Citation validation + rendering
# ---------------------------------------------------------------------------


def validate_citations(text: str, num_chunks: int) -> list[int]:
    """Return citation numbers in text that are out of range [1, num_chunks].

    Empty list means all inline [n] markers are valid. This is the seam cli.py
    uses (03-04) to strip/warn about out-of-range markers before rendering
    the sources block. Side-effect-free — does not modify the text.

    Args:
        text: The LLM-generated answer text containing [n] markers.
        num_chunks: The total number of chunks that were provided as context.

    Returns:
        List of invalid citation numbers (not in [1, num_chunks]). May contain
        duplicates if the same invalid number appears multiple times.
    """
    all_nums = [int(m) for m in _CITATION_RE.findall(text)]
    return [n for n in all_nums if not (1 <= n <= num_chunks)]


def render_citation_block(chunks: list[dict]) -> str:
    """Render the citation sources block from retrieved chunk metadata.

    All metadata comes from the DB (chunk dicts), NOT from LLM text. This is
    the D-06 anti-hallucination guarantee: the LLM cannot fabricate a source
    because the block is code-rendered from the database.

    Format per CONTEXT.md D-06 success criterion:
        [1] shared: Yanmar 4JH45 Manual, p.47
        [2] vessel: Maintenance Log

    Args:
        chunks: List of chunk dicts (from fetch_chunk_metadata) in display order.
            Required keys: layer, title (optional), path (optional),
            page_start (optional), page_end (optional),
            _suppression_note (optional).

    Returns:
        Formatted citation block string starting with "\\n---\\nSources:".
    """
    lines = ["\n---\nSources:"]
    for i, chunk in enumerate(chunks, 1):
        layer = chunk["layer"]
        title = chunk.get("title") or chunk.get("path", "unknown source")
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")

        # page_start/page_end are nullable INTEGERs; page 0 is valid, so the
        # sentinel for "no page" is None, not falsy (WR-04).
        if page_start is not None and page_end is not None and page_start != page_end:
            loc = f", p.{page_start}–{page_end}"
        elif page_start is not None:
            loc = f", p.{page_start}"
        else:
            loc = ""

        suppression = chunk.get("_suppression_note", "")
        lines.append(f"[{i}] {layer}: {title}{loc}{suppression}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# D-09: Ollama /api/chat streaming with hard-fail
# ---------------------------------------------------------------------------


def _warn_malformed_lines(count: int) -> None:
    """Emit a single soft yellow stderr note for dropped malformed NDJSON lines.

    Mirrors the soft-warning convention in retrieve.py (yellow, stderr, no raise).
    Streaming resilience is preserved — the answer is still produced; this only
    makes mid-stream JSON-decode drops observable so a truncated answer is not
    silent.
    """
    plural = "line" if count == 1 else "lines"
    print(
        f"\033[33mWARNING: skipped {count} malformed stream {plural} from Ollama — "
        f"the answer may be truncated.\033[0m",
        file=sys.stderr,
    )


def stream_generate(
    model: str,
    system_prompt: str,
    user_message: str,
    num_predict: int = DEFAULT_NUM_PREDICT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Generator[str, None, dict]:
    """Backend dispatcher — yields generation tokens from the selected backend.

    Default (offline): Ollama /api/chat. When L44_LLM_BACKEND=openrouter
    (Phase 7 hosted alpha): an OpenAI-compatible /chat/completions SSE stream.
    Both yield token strings and return a stats dict, so app.py/cli.py call this
    one function unchanged.
    """
    if _llm_backend() == "openrouter":
        return _stream_generate_openrouter(
            model, system_prompt, user_message, temperature=temperature
        )
    return _stream_generate_ollama(
        model,
        system_prompt,
        user_message,
        num_predict=num_predict,
        temperature=temperature,
    )


def _stream_generate_openrouter(
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Generator[str, None, dict]:
    """Stream tokens from an OpenAI-compatible hosted endpoint (OpenRouter).

    Hard-fails to RuntimeError on any error (mirrors the Ollama path). The API
    key is read at call time from OPENROUTER_API_KEY and never logged. Parses the
    OpenAI SSE wire format: `data: {json}` lines with choices[0].delta.content,
    terminated by `data: [DONE]`; `:`-comment keep-alive lines are skipped.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "L44_LLM_BACKEND=openrouter but OPENROUTER_API_KEY is not set "
            "(expected in the systemd EnvironmentFile, never committed)."
        )
    base_url = os.environ.get(
        "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
    ).rstrip("/")
    max_tokens = int(
        os.environ.get("L44_LLM_MAX_TOKENS", str(DEFAULT_HOSTED_MAX_TOKENS))
    )
    url = base_url + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter attribution (optional but recommended).
        "HTTP-Referer": os.environ.get(
            "OPENROUTER_REFERER", "https://github.com/primesoftnz/leopard44-kb"
        ),
        "X-Title": os.environ.get("OPENROUTER_TITLE", "Leopard 44 KB"),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        with httpx.stream(
            "POST", url, headers=headers, json=payload, timeout=120.0
        ) as response:
            if response.status_code >= 400:
                body = b"".join(response.iter_bytes())
                raise RuntimeError(
                    f"OpenRouter returned HTTP {response.status_code}: "
                    f"{body[:300].decode(errors='replace')}"
                )

            stats: dict = {"model": model, "eval_count": 0, "total_duration_ns": 0}
            malformed_lines = 0
            for line in response.iter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue  # skip blank lines and `:`-comment keep-alives
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                choices = data.get("choices") or []
                if choices:
                    token = (choices[0].get("delta") or {}).get("content") or ""
                    if token:
                        yield token
                usage = data.get("usage")
                if usage:
                    stats["eval_count"] = usage.get(
                        "completion_tokens", stats["eval_count"]
                    )
            if malformed_lines:
                _warn_malformed_lines(malformed_lines)
            return stats

    except httpx.ConnectError:
        raise RuntimeError(
            f"OpenRouter not reachable at {base_url} — check the VM's outbound "
            "network/DNS, then retry."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            "OpenRouter generation timed out after 120s. Retry, or check the "
            "OpenRouter status / your key's credit cap."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"OpenRouter /chat/completions returned HTTP {exc.response.status_code}"
        ) from exc


def _stream_generate_ollama(
    model: str,
    system_prompt: str,
    user_message: str,
    num_predict: int = DEFAULT_NUM_PREDICT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Generator[str, None, dict]:
    """Stream generation tokens from Ollama /api/chat.

    Yields individual token strings as they arrive from the NDJSON stream
    (print each immediately to give live output — D-09). On the done=True
    line, returns a stats dict via generator return value (Python 3.7+):
        {"model": str, "eval_count": int, "total_duration_ns": int}

    Note (Pitfall 7): The first query after model load adds ~3-8s before the
    first token appears. QUERY-06 (<10s) applies to warm-model queries.

    The num_predict parameter defaults to DEFAULT_NUM_PREDICT (75) — the
    QUERY-06 latency cap. Callers must not scale it unboundedly (review fix #5).
    The 120s httpx timeout is a secondary DoS guard (T-03-08).

    OLLAMA_HOST is read at call time (not import time) so monkeypatch.setenv
    works correctly in tests.

    Args:
        model: Ollama model tag (e.g. 'qwen2.5:7b-instruct-q4_K_M').
        system_prompt: Grounding instruction for the LLM.
        user_message: Numbered context + question assembled by build_user_message.
        num_predict: Token generation cap (default 75 per QUERY-06 budget).
        temperature: Sampling temperature (default 0.15 for factual answers).

    Yields:
        str: Individual token strings from message.content (not data["response"]).

    Returns:
        dict: Final stats {"model", "eval_count", "total_duration_ns"} via
            StopIteration.value when the generator is exhausted.

    Raises:
        RuntimeError: Ollama unreachable, model not pulled, or HTTP error.
            All httpx exceptions are converted to RuntimeError (mirrors
            embedder.py hard-fail pattern).
    """
    # Read at call time — enables monkeypatch.setenv("OLLAMA_HOST", ...) in tests.
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"

    try:
        with httpx.stream(
            "POST",
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "num_ctx": NUM_CTX,
                },
            },
            timeout=120.0,
        ) as response:
            # 404 model-missing check BEFORE raise_for_status (mirrors embedder.py pattern).
            if response.status_code == 404:
                body = b"".join(response.iter_bytes())
                raise RuntimeError(
                    f"Ollama model '{model}' not found — run "
                    f"`ollama pull {model}` then retry.\n"
                    f"(Error: {body[:200].decode(errors='replace')})"
                )
            response.raise_for_status()

            stats: dict = {}
            # IN-04: count malformed NDJSON lines so silent mid-stream drops are
            # observable. We deliberately `continue` (do NOT raise) to preserve
            # streaming resilience, but surface a single soft warning at the end
            # if any line failed to parse — otherwise a garbled Ollama line yields
            # a truncated answer with no signal.
            malformed_lines = 0
            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue

                # Pitfall 3: /api/chat uses message.content, NOT data["response"].
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token

                if data.get("done"):
                    if malformed_lines:
                        _warn_malformed_lines(malformed_lines)
                    stats = {
                        "model": data.get("model", model),
                        "eval_count": data.get("eval_count", 0),
                        "total_duration_ns": data.get("total_duration", 0),
                    }
                    return stats  # Python 3.7+ generator return value

            # Stream ended without a `done` line (truncated upstream). Still
            # surface any malformed-line count so the truncation is observable.
            if malformed_lines:
                _warn_malformed_lines(malformed_lines)

    except httpx.ConnectError:
        raise RuntimeError(
            "Ollama not reachable at :11434 — run `ollama serve` and "
            f"`ollama pull {model}`, then retry."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Ollama generation timed out after 120s. The model may still be "
            f"loading. Retry, or check `ollama ps` for model status."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Ollama /api/chat returned HTTP {exc.response.status_code}"
        ) from exc
