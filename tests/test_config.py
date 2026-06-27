# RED state until 06-02 implementation (leopard44_kb.config does not exist yet).
# ModuleNotFoundError at collection is the expected state for this plan (Wave 0).
"""Tests for leopard44_kb.config: load_config/write_config, L44_CONFIG env-override,
call-time path resolution, atomic write, and schema validation.

Requirements covered: INSTALL-01 (config JSON round-trip, env override, atomic write,
schema validation that rejects inconsistent tier/model combinations).
"""
from __future__ import annotations

import json

import pytest

from leopard44_kb.config import TIER_MODELS, load_config, write_config


# ---------------------------------------------------------------------------
# Canonical tier table (mirrors 06-CONTEXT D-01/D-02/D-04)
# ---------------------------------------------------------------------------
# 8gb  -> qwen2.5:3b-instruct-q4_K_M  + all-minilm:latest      (version "latest")
# 16gb -> qwen2.5:7b-instruct-q4_K_M  + nomic-embed-text:v1.5  (version "v1.5")
# gpu  -> qwen2.5:14b-instruct-q4_K_M + nomic-embed-text:v1.5  (version "v1.5")


def test_load_config_returns_none_when_absent(monkeypatch, tmp_path):
    """load_config() returns None when no file exists at the L44_CONFIG path."""
    monkeypatch.setenv("L44_CONFIG", str(tmp_path / "nonexistent.json"))
    result = load_config()
    assert result is None, f"Expected None for absent config file, got {result!r}"


def test_write_config_round_trip_16gb(monkeypatch, tmp_path):
    """write_config('16gb') then load_config() round-trips the full 16gb config dict."""
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("L44_CONFIG", str(cfg_path))

    write_config("16gb")
    result = load_config()

    assert result is not None, "load_config() returned None after write_config('16gb')"
    # Must have exactly these four keys
    assert set(result.keys()) == {
        "tier", "generation_model", "embedding_model", "embedding_model_version"
    }, f"Unexpected keys in config: {set(result.keys())}"
    # Values must match the canonical 16gb row
    assert result["tier"] == "16gb"
    assert result["generation_model"] == "qwen2.5:7b-instruct-q4_K_M"
    assert result["embedding_model"] == "nomic-embed-text:v1.5"
    assert result["embedding_model_version"] == "v1.5"


def test_write_config_gpu_tier(monkeypatch, tmp_path):
    """write_config('gpu') produces generation_model=='qwen2.5:14b-instruct-q4_K_M'."""
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("L44_CONFIG", str(cfg_path))

    write_config("gpu")
    result = load_config()

    assert result is not None
    assert result["generation_model"] == "qwen2.5:14b-instruct-q4_K_M"
    assert result["embedding_model"] == "nomic-embed-text:v1.5"


def test_write_config_8gb_tier(monkeypatch, tmp_path):
    """write_config('8gb') produces all-minilm:latest embedding."""
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("L44_CONFIG", str(cfg_path))

    write_config("8gb")
    result = load_config()

    assert result is not None
    assert result["generation_model"] == "qwen2.5:3b-instruct-q4_K_M"
    assert result["embedding_model"] == "all-minilm:latest"
    assert result["embedding_model_version"] == "latest"


def test_load_config_calltime_path_resolution(monkeypatch, tmp_path):
    """L44_CONFIG is read at call time (not import time) — Pitfall 2.

    Set the env var AFTER import and confirm write/load use the injected path.
    """
    cfg_path = tmp_path / "calltime_config.json"
    # The env var is set HERE — after the module was already imported at the top
    monkeypatch.setenv("L44_CONFIG", str(cfg_path))

    write_config("16gb")
    assert cfg_path.exists(), (
        f"write_config must have written to the L44_CONFIG path set at call time: {cfg_path}"
    )

    result = load_config()
    assert result is not None, "load_config must read from the L44_CONFIG path set at call time"
    assert result["tier"] == "16gb"


def test_write_config_atomic(monkeypatch, tmp_path):
    """write_config does NOT leave a partial/temp file on disk (atomic os.replace contract).

    After write_config returns: (a) the config path exists and is valid JSON;
    (b) no sibling *.tmp files remain in the directory.
    """
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("L44_CONFIG", str(cfg_path))

    write_config("16gb")

    # (a) The resolved path must exist and be valid JSON
    assert cfg_path.exists(), f"Config file must exist at {cfg_path}"
    parsed = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "tier" in parsed, "Config must be valid JSON with a 'tier' key"

    # (b) No sibling temp files remain
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], (
        f"write_config left temp files on disk: {tmp_files}"
    )


def test_write_config_atomic_preserves_existing_on_failure(monkeypatch, tmp_path):
    """A crash mid-write must NOT corrupt a pre-existing config (true atomicity).

    Writes a valid config, then forces os.replace to raise during a second write
    and asserts: the original file is byte-for-byte intact (not truncated/partial)
    and no temp file is left behind. A non-atomic open(path,"w") implementation
    fails this — it truncates the real file before the write can fail.
    """
    import leopard44_kb.config as cfg_mod

    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("L44_CONFIG", str(cfg_path))

    write_config("8gb")
    original_bytes = cfg_path.read_bytes()
    assert json.loads(original_bytes)["tier"] == "8gb"

    def _boom(*_a, **_k):
        raise OSError("simulated crash during os.replace")

    monkeypatch.setattr(cfg_mod.os, "replace", _boom)
    with pytest.raises(OSError):
        write_config("16gb")

    # Original config survives untouched, and no orphaned temp remains.
    assert cfg_path.read_bytes() == original_bytes, "non-atomic write corrupted the existing config"
    assert list(tmp_path.glob("*.tmp")) == [], "failed write left a temp file behind"


def test_load_config_schema_validation(monkeypatch, tmp_path):
    """load_config raises on a tier/model mismatch a hand-edit could produce.

    A config.json with tier='16gb' but embedding_model='all-minilm:latest'
    (which belongs to 8gb, not 16gb) must raise ValueError (or similar)
    rather than silently returning the inconsistent dict.
    """
    cfg_path = tmp_path / "bad_config.json"
    monkeypatch.setenv("L44_CONFIG", str(cfg_path))

    # Write a deliberately inconsistent config (16gb tier but 8gb embedding model)
    bad_cfg = {
        "tier": "16gb",
        "generation_model": "qwen2.5:7b-instruct-q4_K_M",
        "embedding_model": "all-minilm:latest",   # WRONG for 16gb
        "embedding_model_version": "latest",
    }
    cfg_path.write_text(json.dumps(bad_cfg), encoding="utf-8")

    with pytest.raises((ValueError, KeyError, RuntimeError)):
        load_config()


def test_tier_models_constant_covers_all_tiers():
    """TIER_MODELS contains all three tiers with the expected tuple shape."""
    for tier in ("8gb", "16gb", "gpu"):
        assert tier in TIER_MODELS, f"TIER_MODELS missing tier '{tier}'"
        gen_model, embed_model, embed_version = TIER_MODELS[tier]
        assert gen_model, f"Empty generation_model for tier '{tier}'"
        assert embed_model, f"Empty embedding_model for tier '{tier}'"
        assert embed_version, f"Empty embedding_model_version for tier '{tier}'"
