"""Config file read/write for Leopard 44 KB model-tier selection (D-03/D-04).

Provides load_config / write_config backed by a JSON file alongside store.db.
The L44_CONFIG env var controls the path and is read at *call time* — not
import time — so tests can monkeypatch.setenv without module reload.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Canonical tier table (D-01/D-02/D-04).
# All embeddings are 384-dim per the SCHEMA-04 Matryoshka lock.
#
#   tier  -> (generation_model, embedding_model, embedding_model_version)
#
TIER_MODELS: dict[str, tuple[str, str, str]] = {
    "8gb":  ("qwen2.5:3b-instruct-q4_K_M",  "all-minilm:latest",     "latest"),
    "16gb": ("qwen2.5:7b-instruct-q4_K_M",  "nomic-embed-text:v1.5", "v1.5"),
    "gpu":  ("qwen2.5:14b-instruct-q4_K_M", "nomic-embed-text:v1.5", "v1.5"),
}

# Per-tier generation length cap (Ollama num_predict). The original flat 75-token
# cap (DEFAULT_NUM_PREDICT, QUERY-06 <10s CPU budget) truncated real answers at
# ~2 bullets. These tier-scaled caps trade some CPU latency for a complete answer:
# the 8gb/16gb tiers run on CPU so larger caps cost wall-clock, while the gpu tier
# can afford a much fuller answer cheaply. Override any tier with L44_NUM_PREDICT
# (see answer.select_num_predict). Keys mirror TIER_MODELS.
TIER_NUM_PREDICT: dict[str, int] = {
    "8gb":  256,
    "16gb": 384,
    "gpu":  512,
}

def _config_path() -> Path:
    """Resolve the config file path at call time from L44_CONFIG or XDG default."""
    return Path(
        os.environ.get("L44_CONFIG")
        or (Path.home() / ".local" / "share" / "leopard44-kb" / "config.json")
    )


def load_config() -> dict | None:
    """Return the parsed config dict, or None if the config file is absent.

    Reads L44_CONFIG at call time so monkeypatch.setenv works in tests.
    Propagates json.JSONDecodeError on malformed JSON (fail-closed per project
    philosophy — a corrupt config is surfaced loudly rather than silently
    selecting the wrong model tier).

    Raises:
        ValueError: If the stored tier/model combination is internally inconsistent
            (e.g. tier='16gb' but embedding_model='all-minilm:latest'). This guards
            against hand-edited configs that would silently select the wrong tier.
        json.JSONDecodeError: If the file exists but contains invalid JSON.
    """
    p = _config_path()
    if not p.exists():
        return None
    with open(p) as f:
        cfg = json.load(f)

    # Schema validation: if a tier key is present, it must be a known tier AND
    # both model fields must match TIER_MODELS for that tier. A mismatch (or an
    # unknown tier) means the file was hand-edited inconsistently and would cause
    # a silent model/384-dim violation (D-04).
    tier = cfg.get("tier")
    if tier is not None:
        if tier not in TIER_MODELS:
            raise ValueError(
                f"Config tier '{tier}' is not a known tier "
                f"(expected one of {sorted(TIER_MODELS)}). "
                f"The config.json appears to have been hand-edited. "
                f"Re-run setup to regenerate a valid config."
            )
        expected_gen, expected_emb, expected_emb_ver = TIER_MODELS[tier]
        for field, expected in (
            ("embedding_model", expected_emb),
            ("generation_model", expected_gen),
        ):
            stored = cfg.get(field)
            if stored is not None and stored != expected:
                raise ValueError(
                    f"Config tier '{tier}' expects {field} "
                    f"'{expected}' but found '{stored}'. "
                    f"The config.json appears to have been hand-edited inconsistently. "
                    f"Re-run setup to regenerate a valid config."
                )

    return cfg


def write_config(tier: str) -> Path:
    """Write config.json for the given tier. Creates the parent directory if absent.

    Args:
        tier: One of '8gb', '16gb', 'gpu'. KeyError if not in TIER_MODELS.

    Returns:
        The Path where the config was written.

    Raises:
        KeyError: If tier is not one of the valid keys in TIER_MODELS.
    """
    gen_model, emb_model, emb_ver = TIER_MODELS[tier]  # KeyError on bad tier
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tier": tier,
        "generation_model": gen_model,
        "embedding_model": emb_model,
        "embedding_model_version": emb_ver,
    }
    # Atomic write: serialise to a temp file in the same directory, fsync, then
    # os.replace() into place. A crash mid-write leaves the temp file (cleaned up
    # below) rather than a partial config.json — so `l44 ask` never reads a
    # truncated config. os.replace is atomic on the same filesystem (POSIX + NTFS).
    fd, tmp_name = tempfile.mkstemp(
        dir=p.parent, prefix=f".{p.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, p)
    except BaseException:
        # Leave the real config untouched; remove the orphaned temp file.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return p
