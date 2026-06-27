"""Ollama embedding wrapper — single call path for all tiers per D-08/D-09."""
from __future__ import annotations

import os

import httpx

# Dimension locked at 384 per RESEARCH.md Pattern 3 (Matryoshka lock).
# All tiers produce 384-dim vectors: all-MiniLM-L6-v2 native; nomic-embed-text v1.5
# truncated via Ollama dimensions=384 + re-normalise.
EMBED_DIM = 384


def detect_ram_gb() -> float:
    """Return total RAM in GB. Cross-platform.

    Platform probe order: Linux /proc/meminfo → macOS sysctl hw.memsize →
    Windows ctypes GlobalMemoryStatusEx → 8.0 conservative fallback.
    """
    # Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024 / 1024
    except (FileNotFoundError, ValueError):
        pass
    # macOS
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out) / 1024**3
    except Exception:
        pass
    # Windows
    try:
        import ctypes

        class _MEM(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEM()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return stat.ullTotalPhys / 1024**3
    except Exception:
        pass
    return 8.0  # conservative fallback — selects all-minilm tier


def select_model() -> tuple[str, str]:
    """Return (ollama_model_name, version_tag) for the embedding model.

    Config-first (D-03/D-04): reads L44_CONFIG at call time. If a config file
    exists and contains 'embedding_model', that value is returned without touching
    RAM detection — pinning the embedding model to the install-time choice and
    protecting the 384-dim lock from silent re-selection on RAM changes.

    When config is absent, falls back to detect_ram_gb() (behaviour unchanged):
      RAM >= 14 GB → nomic-embed-text:v1.5 (higher quality, 270 MB)
      RAM <  14 GB → all-minilm:latest (minimum footprint, native 384-dim)
    """
    # D-03/D-04: config-first; RAM fallback only when absent
    from leopard44_kb.config import load_config
    cfg = load_config()
    if cfg and "embedding_model" in cfg:
        return (cfg["embedding_model"], cfg.get("embedding_model_version", "unknown"))
    # Fallback: RAM-based autodetect (unchanged)
    gb = detect_ram_gb()
    if gb >= 14:
        return ("nomic-embed-text:v1.5", "v1.5")
    return ("all-minilm:latest", "latest")


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    """Embed a batch of texts via Ollama /api/embed. Hard-fail if Ollama unreachable (D-09).

    Args:
        texts: List of text strings to embed.
        model: Ollama model name (e.g. 'nomic-embed-text:v1.5').

    Returns:
        List of 384-dim float lists, one per input text.

    Raises:
        RuntimeError: On any of: Ollama unreachable, timeout, non-2xx response,
            404 model-missing, malformed/invalid JSON, missing 'embeddings' key,
            count mismatch (len(embeddings) != len(texts)), or any vector len != 384.
    """
    if not texts:
        return []

    # Read OLLAMA_HOST at call time (not import time) — env may be patched in tests.
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/embed"

    try:
        r = httpx.post(
            url,
            json={"model": model, "input": texts, "dimensions": EMBED_DIM},
            timeout=60.0,
        )
    except httpx.ConnectError:
        raise RuntimeError(
            "Ollama not reachable at :11434 — run `ollama serve` and "
            "`ollama pull nomic-embed-text:v1.5` (or `ollama pull all-minilm`), "
            "then retry."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            "Ollama embed timed out after 60s at :11434 — the model may still be "
            "loading; retry, or use a smaller tier model."
        )

    # Handle 404 model-missing before generic raise_for_status.
    if r.status_code == 404 and "not found" in r.text.lower():
        raise RuntimeError(
            f"Ollama model '{model}' not found — run `ollama pull {model}` then retry."
        )

    # Raise on any other non-2xx; convert httpx error to RuntimeError for consistency.
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body_excerpt = r.text[:200] if r.text else "(empty body)"
        raise RuntimeError(
            f"Ollama /api/embed returned HTTP {r.status_code}: {body_excerpt}"
        ) from exc

    # Parse JSON; require the 'embeddings' key.
    try:
        data = r.json()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama returned malformed JSON from /api/embed: {exc}"
        ) from exc

    if "embeddings" not in data:
        raise RuntimeError(
            "Ollama response missing 'embeddings' key — got keys: "
            f"{list(data.keys())!r}. Check that you are using /api/embed "
            "(the legacy singular endpoint uses a different response key)."
        )

    embeddings: list[list[float]] = data["embeddings"]

    # Validate count matches input.
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama embed count mismatch: expected {len(texts)} embeddings "
            f"but got {len(embeddings)}."
        )

    # Validate dimension of every vector BEFORE returning (384 assert — Codex review HIGH).
    for i, vec in enumerate(embeddings):
        if len(vec) != EMBED_DIM:
            raise RuntimeError(
                f"Ollama returned a {len(vec)}-dim vector at index {i}; "
                f"expected {EMBED_DIM}. Check that the model honours "
                f"dimensions={EMBED_DIM} (or use a native 384-dim model)."
            )

    return embeddings
