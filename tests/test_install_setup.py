# RED state until 06-03 implementation (scripts.setup_core does not exist yet).
# ModuleNotFoundError at collection is the expected state for this plan (Wave 0).
"""Tests for scripts.setup_core: tier recommendation, GPU probe, daemon check,
OLLAMA_HOST honouring, --tier allowlist, validate_seed_layout, Python pull guard,
D-04 embedding-mismatch guard, smoke-test parse, and main() short-circuit ordering.

Requirements covered: INSTALL-01 (tier selection), INSTALL-02 (Ollama daemon check),
INSTALL-04 (validate_seed_layout fs-check), INSTALL-05 (--tier allowlist),
INSTALL-06 (smoke-test parse), T-06-01 (input validation), T-06-09 (D-04 guard).

NOTE: validate_seed_layout is a pure filesystem check — NO git operations, NO git init.
There is no test_migrate_idempotent and NO reference to migrate_seed_files in this file.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import setup_core


# ---------------------------------------------------------------------------
# GPU_VRAM_THRESHOLD_MIB is the conservative threshold constant from setup_core.
# We assert the sentinel value rather than hardcode 9000 everywhere.
# ---------------------------------------------------------------------------

_SEED_FILES = [
    "bluewater-capability.md",
    "competitive-comparison.md",
    "ex-charter-buying-guide.md",
    "interior-layout.md",
    "known-issues.md",
    "market-value.md",
    "onboard-systems.md",
    "sailing-performance.md",
    "technical-specifications.md",
    "upgrades-modifications.md",
]


# ---------------------------------------------------------------------------
# INSTALL-01: Tier recommendation
# ---------------------------------------------------------------------------


def test_tier_recommendation_gpu(monkeypatch):
    """detect_gpu_vram_mib returns >=threshold -> recommend_tier() returns 'gpu'."""
    threshold = setup_core.GPU_VRAM_THRESHOLD_MIB
    monkeypatch.setattr(setup_core, "detect_gpu_vram_mib", lambda: threshold)
    result = setup_core.recommend_tier()
    assert result == "gpu", (
        f"Expected 'gpu' when VRAM >= threshold ({threshold} MiB), got {result!r}"
    )


def test_tier_recommendation_16gb(monkeypatch):
    """No GPU + 16.0 GB RAM -> recommend_tier() returns '16gb'."""
    monkeypatch.setattr(setup_core, "detect_gpu_vram_mib", lambda: None)
    monkeypatch.setattr(setup_core, "detect_ram_gb", lambda: 16.0)
    result = setup_core.recommend_tier()
    assert result == "16gb", f"Expected '16gb' for 16GB RAM, no GPU; got {result!r}"


def test_tier_recommendation_8gb(monkeypatch):
    """No GPU + 8.0 GB RAM -> recommend_tier() returns '8gb'."""
    monkeypatch.setattr(setup_core, "detect_gpu_vram_mib", lambda: None)
    monkeypatch.setattr(setup_core, "detect_ram_gb", lambda: 8.0)
    result = setup_core.recommend_tier()
    assert result == "8gb", f"Expected '8gb' for 8GB RAM, no GPU; got {result!r}"


# ---------------------------------------------------------------------------
# INSTALL-01: GPU max-across-GPUs and conservative threshold
# ---------------------------------------------------------------------------


def test_gpu_max_across_gpus(monkeypatch):
    """detect_gpu_vram_mib returns the MAX VRAM across multiple GPU lines.

    nvidia-smi returns two lines: '8192 MiB\\n16376 MiB\\n' -> should return 16376.
    """
    def _fake_check_output(cmd, *args, **kwargs):
        return "8192 MiB\n16376 MiB\n"

    monkeypatch.setattr(subprocess, "check_output", _fake_check_output)
    result = setup_core.detect_gpu_vram_mib()
    assert result == 16376, (
        f"Expected detect_gpu_vram_mib() to return 16376 (max across GPUs), got {result!r}"
    )


def test_gpu_single_below_threshold_not_gpu_tier(monkeypatch):
    """A single 8192 MiB GPU does NOT trigger the 'gpu' tier recommendation.

    The conservative threshold must be > 8192 MiB so an 8GB VRAM card is NOT
    automatically recommended for the heavy 14B model.
    """
    assert setup_core.GPU_VRAM_THRESHOLD_MIB > 8192, (
        f"GPU_VRAM_THRESHOLD_MIB must be > 8192 to avoid recommending 14B on 8GB VRAM cards; "
        f"got {setup_core.GPU_VRAM_THRESHOLD_MIB}"
    )


def test_gpu_detection_returns_none_when_absent(monkeypatch):
    """detect_gpu_vram_mib returns None when nvidia-smi is not present."""
    def _raise(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "check_output", _raise)
    result = setup_core.detect_gpu_vram_mib()
    assert result is None, f"Expected None when nvidia-smi absent, got {result!r}"


# ---------------------------------------------------------------------------
# INSTALL-02: Ollama daemon check
# ---------------------------------------------------------------------------


def test_daemon_check_running(monkeypatch):
    """is_ollama_daemon_running() returns True when httpx.get returns 200."""
    import httpx

    class _FakeResp:
        status_code = 200

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResp())
    assert setup_core.is_ollama_daemon_running() is True


def test_daemon_check_not_running(monkeypatch):
    """is_ollama_daemon_running() returns False when httpx.get raises ConnectError."""
    import httpx

    def _raise(*a, **kw):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "get", _raise)
    assert setup_core.is_ollama_daemon_running() is False


def test_daemon_check_non_200(monkeypatch):
    """is_ollama_daemon_running() returns False on a non-200 response."""
    import httpx

    class _FakeResp:
        status_code = 503

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResp())
    assert setup_core.is_ollama_daemon_running() is False


# ---------------------------------------------------------------------------
# INSTALL-02: OLLAMA_HOST honouring (Codex MEDIUM finding)
# ---------------------------------------------------------------------------


def test_daemon_check_honours_ollama_host(monkeypatch):
    """is_ollama_daemon_running() targets OLLAMA_HOST, not a hardcoded localhost:11434.

    When OLLAMA_HOST is set to 'http://127.0.0.1:9999', the daemon check must
    call httpx.get with a URL that starts with that host value.
    """
    import httpx

    captured_urls = []

    class _FakeResp:
        status_code = 200

    def _fake_get(url, *a, **kw):
        captured_urls.append(url)
        return _FakeResp()

    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
    monkeypatch.setattr(httpx, "get", _fake_get)

    setup_core.is_ollama_daemon_running()

    assert len(captured_urls) == 1, f"Expected exactly one httpx.get call, got {captured_urls!r}"
    assert captured_urls[0].startswith("http://127.0.0.1:9999"), (
        f"Expected URL to target OLLAMA_HOST 'http://127.0.0.1:9999', "
        f"but got {captured_urls[0]!r} — daemon check is NOT honouring OLLAMA_HOST"
    )


# ---------------------------------------------------------------------------
# INSTALL-05: --tier allowlist (T-06-01 security test)
# ---------------------------------------------------------------------------


def test_tier_override_8gb(monkeypatch):
    """resolve_tier('8gb') returns '8gb' unchanged."""
    monkeypatch.setattr(setup_core, "recommend_tier", lambda: "16gb")
    result = setup_core.resolve_tier("8gb")
    assert result == "8gb"


def test_tier_override_16gb(monkeypatch):
    """resolve_tier('16gb') returns '16gb' unchanged."""
    monkeypatch.setattr(setup_core, "recommend_tier", lambda: "8gb")
    result = setup_core.resolve_tier("16gb")
    assert result == "16gb"


def test_tier_override_gpu(monkeypatch):
    """resolve_tier('gpu') returns 'gpu' unchanged."""
    monkeypatch.setattr(setup_core, "recommend_tier", lambda: "8gb")
    result = setup_core.resolve_tier("gpu")
    assert result == "gpu"


def test_tier_override_none_calls_recommend(monkeypatch):
    """resolve_tier(None) delegates to recommend_tier()."""
    monkeypatch.setattr(setup_core, "recommend_tier", lambda: "16gb")
    result = setup_core.resolve_tier(None)
    assert result == "16gb"


def test_tier_override_uppercase_rejected():
    """resolve_tier('16GB') is rejected — allowlist is case-sensitive {8gb,16gb,gpu}."""
    with pytest.raises((SystemExit, ValueError)):
        setup_core.resolve_tier("16GB")


def test_tier_override_nonsense_rejected():
    """resolve_tier('nonsense') is rejected — not in allowlist."""
    with pytest.raises((SystemExit, ValueError)):
        setup_core.resolve_tier("nonsense")


def test_tier_override_bad_value_not_passed_downstream(monkeypatch):
    """A rejected bad tier value must not reach write_config or any downstream step."""
    write_config_called = []
    monkeypatch.setattr(setup_core, "write_config", lambda *a, **kw: write_config_called.append(a))

    with pytest.raises((SystemExit, ValueError)):
        setup_core.resolve_tier("badtier")

    assert write_config_called == [], (
        "write_config must NOT be called when resolve_tier rejects a bad tier value"
    )


# ---------------------------------------------------------------------------
# INSTALL-04: validate_seed_layout (pure filesystem check — NO git)
# ---------------------------------------------------------------------------


def test_validate_seed_layout_passes_with_all_files(tmp_path):
    """validate_seed_layout(repo_root) returns None when all 10 .md files exist."""
    leopard44 = tmp_path / "shared" / "leopard44"
    leopard44.mkdir(parents=True)
    for fname in _SEED_FILES:
        (leopard44 / fname).write_text(f"# {fname}\nContent.\n")

    # Must not raise
    result = setup_core.validate_seed_layout(tmp_path)
    assert result is None, f"Expected None (pass), got {result!r}"


def test_validate_seed_layout_fails_missing_file(tmp_path):
    """validate_seed_layout raises (SystemExit or RuntimeError) when a file is missing."""
    leopard44 = tmp_path / "shared" / "leopard44"
    leopard44.mkdir(parents=True)
    # Write all but one
    for fname in _SEED_FILES[:-1]:
        (leopard44 / fname).write_text(f"# {fname}\nContent.\n")
    missing = _SEED_FILES[-1]

    with pytest.raises((SystemExit, RuntimeError)) as exc_info:
        setup_core.validate_seed_layout(tmp_path)

    # The error/exit message should mention the missing file
    err_str = str(exc_info.value)
    assert missing in err_str, (
        f"Expected error to mention missing file '{missing}', got: {err_str!r}"
    )


def test_validate_seed_layout_no_git_operations(tmp_path):
    """validate_seed_layout must NOT call any git subprocess.

    This is enforced architecturally: it is a filesystem check only.
    We verify no subprocess call targeting 'git' is made.
    """
    leopard44 = tmp_path / "shared" / "leopard44"
    leopard44.mkdir(parents=True)
    for fname in _SEED_FILES:
        (leopard44 / fname).write_text(f"# {fname}\nContent.\n")

    git_calls = []

    original_run = subprocess.run
    original_check_call = subprocess.check_call
    original_check_output = subprocess.check_output

    def _spy_run(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and "git" in str(cmd[0]):
            git_calls.append(cmd)
        return original_run(cmd, *a, **kw)

    def _spy_check_call(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and "git" in str(cmd[0]):
            git_calls.append(cmd)
        return original_check_call(cmd, *a, **kw)

    def _spy_check_output(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and "git" in str(cmd[0]):
            git_calls.append(cmd)
        return original_check_output(cmd, *a, **kw)

    import subprocess as sp
    old_run = sp.run
    old_check_call = sp.check_call
    old_check_output = sp.check_output

    sp.run = _spy_run
    sp.check_call = _spy_check_call
    sp.check_output = _spy_check_output
    try:
        setup_core.validate_seed_layout(tmp_path)
    finally:
        sp.run = old_run
        sp.check_call = old_check_call
        sp.check_output = old_check_output

    assert git_calls == [], (
        f"validate_seed_layout must NOT call any git subprocess; got calls: {git_calls}"
    )


# ---------------------------------------------------------------------------
# INSTALL-01/INSTALL-02: Python pull guard (parses ollama list in Python)
# ---------------------------------------------------------------------------


def test_pull_guard_already_present_no_pull(monkeypatch):
    """ensure_model_pulled does NOT pull when the exact full tag is already listed."""
    def _fake_check_output(cmd, *a, **kw):
        # Simulate `ollama list` output with the model present
        return (
            "NAME                                    ID              SIZE  MODIFIED\n"
            "qwen2.5:7b-instruct-q4_K_M             abc123def456    4.7G  3 days ago\n"
            "nomic-embed-text:v1.5                   def789ghi012    270M  3 days ago\n"
        )

    pull_calls = []

    def _fake_check_call(cmd, *a, **kw):
        if "pull" in cmd:
            pull_calls.append(cmd)

    monkeypatch.setattr(subprocess, "check_output", _fake_check_output)
    monkeypatch.setattr(subprocess, "check_call", _fake_check_call)

    setup_core.ensure_model_pulled("qwen2.5:7b-instruct-q4_K_M")

    assert pull_calls == [], (
        f"ensure_model_pulled must NOT pull when the exact tag is already present; "
        f"got calls: {pull_calls}"
    )


def test_pull_guard_absent_triggers_pull(monkeypatch):
    """ensure_model_pulled triggers pull when tag is absent from ollama list."""
    def _fake_check_output(cmd, *a, **kw):
        # Tag NOT in the listing
        return (
            "NAME                                    ID              SIZE  MODIFIED\n"
            "nomic-embed-text:v1.5                   def789ghi012    270M  3 days ago\n"
        )

    pull_calls = []

    def _fake_check_call(cmd, *a, **kw):
        pull_calls.append(cmd)

    monkeypatch.setattr(subprocess, "check_output", _fake_check_output)
    monkeypatch.setattr(subprocess, "check_call", _fake_check_call)

    setup_core.ensure_model_pulled("qwen2.5:7b-instruct-q4_K_M")

    assert any("pull" in str(c) for c in pull_calls), (
        f"ensure_model_pulled must trigger 'ollama pull' when tag is absent; "
        f"got calls: {pull_calls}"
    )


def test_pull_guard_exact_full_tag_match(monkeypatch):
    """ensure_model_pulled uses exact full-tag matching — a listing of
    'qwen2.5:3b-instruct-q4_K_M' does NOT satisfy a request for 'qwen2.5:3b'.
    """
    def _fake_check_output(cmd, *a, **kw):
        # Only the 3b full-tag variant is present
        return (
            "NAME                                    ID              SIZE  MODIFIED\n"
            "qwen2.5:3b-instruct-q4_K_M             xyz111aaa222    2.0G  1 day ago\n"
        )

    pull_calls = []

    def _fake_check_call(cmd, *a, **kw):
        pull_calls.append(cmd)

    monkeypatch.setattr(subprocess, "check_output", _fake_check_output)
    monkeypatch.setattr(subprocess, "check_call", _fake_check_call)

    # Asking for the short tag "qwen2.5:3b" which is NOT in the listing
    setup_core.ensure_model_pulled("qwen2.5:3b")

    assert any("pull" in str(c) for c in pull_calls), (
        "ensure_model_pulled must pull 'qwen2.5:3b' even though "
        "'qwen2.5:3b-instruct-q4_K_M' is present — no partial/prefix matching"
    )


# ---------------------------------------------------------------------------
# T-06-09 / D-04: Embedding-model mismatch guard (assert_embedding_compatible)
# ---------------------------------------------------------------------------


def test_assert_embedding_compatible_raises_on_mismatch(monkeypatch, tmp_path):
    """assert_embedding_compatible raises when existing store's embedding != tier's embedding.

    A store seeded with all-minilm:latest (8gb embedding) must refuse 16gb tier
    (which expects nomic-embed-text:v1.5). D-04 enforcement.
    """
    from leopard44_kb.retrieve import most_common_embedding_model

    monkeypatch.setattr(
        "leopard44_kb.retrieve.most_common_embedding_model",
        lambda conn: "all-minilm:latest",
    )
    # Set L44_DB to a temp path so open_db doesn't use a real store
    monkeypatch.setenv("L44_DB", str(tmp_path / "test_store.db"))

    with pytest.raises((SystemExit, ValueError, RuntimeError)):
        setup_core.assert_embedding_compatible("16gb")  # expects nomic-embed-text:v1.5


def test_assert_embedding_compatible_passes_matching_tier(monkeypatch, tmp_path):
    """assert_embedding_compatible does NOT raise when embedding model matches tier."""
    monkeypatch.setattr(
        "leopard44_kb.retrieve.most_common_embedding_model",
        lambda conn: "all-minilm:latest",
    )
    monkeypatch.setenv("L44_DB", str(tmp_path / "test_store.db"))

    # 8gb tier expects all-minilm:latest — should not raise
    setup_core.assert_embedding_compatible("8gb")


def test_assert_embedding_compatible_passes_empty_store(monkeypatch, tmp_path):
    """assert_embedding_compatible does NOT raise when the store is empty (clean install).

    most_common_embedding_model returns None for an empty store — the guard must not
    block a fresh install (no existing chunks to mismatch against).
    """
    monkeypatch.setattr(
        "leopard44_kb.retrieve.most_common_embedding_model",
        lambda conn: None,
    )
    monkeypatch.setenv("L44_DB", str(tmp_path / "empty_store.db"))

    # Should not raise for any tier when the store is empty
    setup_core.assert_embedding_compatible("16gb")
    setup_core.assert_embedding_compatible("8gb")
    setup_core.assert_embedding_compatible("gpu")


# ---------------------------------------------------------------------------
# INSTALL-06: Smoke-test parse (tightened parse — Codex HIGH fix)
# ---------------------------------------------------------------------------


def test_smoke_test_pass_with_shared_citation(monkeypatch):
    """run_smoke_test returns True when output has non-empty answer + Sources block
    with a '^[N] shared:' line.
    """
    fake_stdout = (
        "Leopard 44 catamarans have a known issue with forestay chainplates.\n"
        "\n---\nSources:\n"
        "[1] shared: Known Issues — Leopard 44\n"
    )

    class _FakeResult:
        stdout = fake_stdout
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeResult())
    assert setup_core.run_smoke_test("What are known issues?") is True


def test_smoke_test_fail_refusal_message(monkeypatch):
    """run_smoke_test returns False when stdout contains the REFUSAL_MESSAGE."""
    from leopard44_kb.answer import REFUSAL_MESSAGE

    class _FakeResult:
        stdout = REFUSAL_MESSAGE
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeResult())
    assert setup_core.run_smoke_test("What are known issues?") is False


def test_smoke_test_fail_vessel_only_citation(monkeypatch):
    """run_smoke_test returns False when Sources block has only 'vessel:' lines (no shared)."""
    fake_stdout = (
        "Some answer text here about the vessel.\n"
        "\n---\nSources:\n"
        "[1] vessel: My maintenance log\n"
    )

    class _FakeResult:
        stdout = fake_stdout
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeResult())
    assert setup_core.run_smoke_test("What are known issues?") is False


def test_smoke_test_fail_empty_stdout(monkeypatch):
    """run_smoke_test returns False when stdout is empty."""
    class _FakeResult:
        stdout = ""
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeResult())
    assert setup_core.run_smoke_test("What are known issues?") is False


def test_smoke_test_fail_no_sources_block(monkeypatch):
    """run_smoke_test returns False when output has answer text but no 'Sources:' block."""
    fake_stdout = "Some answer text but no citation block.\n"

    class _FakeResult:
        stdout = fake_stdout
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeResult())
    assert setup_core.run_smoke_test("What are known issues?") is False


def test_smoke_test_bare_shared_substring_not_sufficient(monkeypatch):
    """run_smoke_test requires '^[N] shared:' WITHIN a Sources block.

    A bare 'shared' substring in the answer text (not in a Sources block as [N] shared:)
    must NOT be treated as a passing citation.
    """
    # 'shared' appears in the answer text but NOT as a [N] shared: citation
    fake_stdout = (
        "The shared knowledge base contains relevant information about the vessel.\n"
        "\n---\nSources:\n"
        "[1] vessel: My maintenance log\n"
    )

    class _FakeResult:
        stdout = fake_stdout
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeResult())
    assert setup_core.run_smoke_test("What are known issues?") is False, (
        "run_smoke_test must require a '^[N] shared:' citation line in the Sources block, "
        "not just a 'shared' substring anywhere in stdout"
    )


# ---------------------------------------------------------------------------
# main() short-circuit ordering (Codex MEDIUM)
# ---------------------------------------------------------------------------


def test_main_short_circuits_daemon_down(monkeypatch):
    """main() returns non-zero and does NOT call pull/write_config/ingest/smoke when daemon down."""
    called = []

    monkeypatch.setattr(setup_core, "is_ollama_daemon_running", lambda: False)
    monkeypatch.setattr(setup_core, "ensure_model_pulled", lambda *a, **kw: called.append("pull"))
    monkeypatch.setattr(setup_core, "write_config", lambda *a, **kw: called.append("write_config"))
    monkeypatch.setattr(setup_core, "validate_seed_layout", lambda *a, **kw: called.append("validate"))
    monkeypatch.setattr(setup_core, "run_smoke_test", lambda *a, **kw: called.append("smoke") or True)

    # Provide argv to indicate daemon is already up (don't test install path here)
    result = setup_core.main(["--tier", "16gb", "--skip-install"])

    assert result != 0, f"main() must return non-zero when daemon is down; got {result!r}"
    assert "pull" not in called, "ensure_model_pulled must NOT be called when daemon down"
    assert "write_config" not in called, "write_config must NOT be called when daemon down"
    assert "smoke" not in called, "run_smoke_test must NOT be called when daemon down"


def test_main_short_circuits_pull_fail(monkeypatch):
    """main() returns non-zero and does NOT call write_config when pull fails."""
    called = []

    monkeypatch.setattr(setup_core, "is_ollama_daemon_running", lambda: True)
    monkeypatch.setattr(setup_core, "resolve_tier", lambda *a, **kw: "16gb")
    monkeypatch.setattr(setup_core, "assert_embedding_compatible", lambda *a, **kw: None)

    def _pull_fail(tag):
        called.append(f"pull:{tag}")
        raise subprocess.CalledProcessError(1, ["ollama", "pull", tag])

    monkeypatch.setattr(setup_core, "ensure_model_pulled", _pull_fail)
    monkeypatch.setattr(setup_core, "write_config", lambda *a, **kw: called.append("write_config"))
    monkeypatch.setattr(setup_core, "run_smoke_test", lambda *a, **kw: called.append("smoke") or True)

    result = setup_core.main(["--tier", "16gb", "--skip-install"])

    assert result != 0, f"main() must return non-zero when pull fails; got {result!r}"
    assert "write_config" not in called, (
        "write_config must NOT be called after a failed pull — config must not persist "
        "a tier pointing at unavailable models"
    )
    assert "smoke" not in called, "run_smoke_test must NOT be called after pull failure"


def test_main_short_circuits_ingest_fail(monkeypatch):
    """main() returns non-zero and does NOT call run_smoke_test when seed ingest fails."""
    called = []

    monkeypatch.setattr(setup_core, "is_ollama_daemon_running", lambda: True)
    monkeypatch.setattr(setup_core, "resolve_tier", lambda *a, **kw: "16gb")
    monkeypatch.setattr(setup_core, "assert_embedding_compatible", lambda *a, **kw: None)
    monkeypatch.setattr(setup_core, "ensure_model_pulled", lambda *a, **kw: called.append("pull"))
    monkeypatch.setattr(setup_core, "write_config", lambda *a, **kw: called.append("write_config"))
    monkeypatch.setattr(setup_core, "validate_seed_layout", lambda *a, **kw: None)

    def _ingest_fail(*a, **kw):
        called.append("ingest")
        raise subprocess.CalledProcessError(1, ["l44", "ingest"])

    monkeypatch.setattr(setup_core, "_run_seed_ingest", _ingest_fail)
    monkeypatch.setattr(setup_core, "run_smoke_test", lambda *a, **kw: called.append("smoke") or True)

    result = setup_core.main(["--tier", "16gb", "--skip-install"])

    assert result != 0, f"main() must return non-zero when ingest fails; got {result!r}"
    assert "smoke" not in called, (
        "run_smoke_test must NOT be called after seed ingest failure"
    )


def test_main_short_circuits_smoke_fail(monkeypatch):
    """main() returns non-zero when run_smoke_test returns False."""
    called = []

    monkeypatch.setattr(setup_core, "is_ollama_daemon_running", lambda: True)
    monkeypatch.setattr(setup_core, "resolve_tier", lambda *a, **kw: "16gb")
    monkeypatch.setattr(setup_core, "assert_embedding_compatible", lambda *a, **kw: None)
    monkeypatch.setattr(setup_core, "ensure_model_pulled", lambda *a, **kw: called.append("pull"))
    monkeypatch.setattr(setup_core, "write_config", lambda *a, **kw: called.append("write_config"))
    monkeypatch.setattr(setup_core, "validate_seed_layout", lambda *a, **kw: None)
    monkeypatch.setattr(setup_core, "_run_seed_ingest", lambda *a, **kw: called.append("ingest"))
    monkeypatch.setattr(setup_core, "run_smoke_test", lambda *a, **kw: False)

    result = setup_core.main(["--tier", "16gb", "--skip-install"])

    assert result != 0, f"main() must return non-zero when smoke-test fails; got {result!r}"


# ===========================================================================
# Plan 13-04 RED tests: PKG-03, PKG-04, PKG-08, uv-bootstrap, behavioral
# ===========================================================================

# Repo root derived from the test file location (tests/ is one level below root)
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# PKG-03: start.command and start.bat launcher content (T-13-42)
# ---------------------------------------------------------------------------


def test_start_command_launches_l44_serve():
    """start.command must invoke 'l44 serve', NOT re-run setup.sh (PKG-03, T-13-42)."""
    content = (_REPO_ROOT / "start.command").read_text()
    assert "l44 serve" in content, (
        "start.command must contain 'l44 serve' to launch the server (not re-run the installer)"
    )


def test_start_command_does_not_reference_setup_sh():
    """start.command must NOT call setup.sh — that was the v1.0 defect (T-13-42)."""
    content = (_REPO_ROOT / "start.command").read_text()
    assert "setup.sh" not in content, (
        "start.command must NOT reference setup.sh "
        "(T-13-42: launcher must not re-run the installer)"
    )


def test_start_command_prepends_local_bin_to_path():
    """start.command must export PATH with $HOME/.local/bin so Finder double-click finds uv."""
    content = (_REPO_ROOT / "start.command").read_text()
    assert ".local/bin" in content, (
        "start.command must prepend $HOME/.local/bin to PATH so a macOS Finder "
        "double-click can find uv (which setup installs to that location)"
    )


def test_start_bat_exists_and_launches_l44_serve():
    """start.bat must exist and invoke 'l44 serve' (PKG-03, new Windows run launcher)."""
    bat = _REPO_ROOT / "start.bat"
    assert bat.exists(), "start.bat must be created (new Windows run launcher, PKG-03)"
    content = bat.read_text()
    assert "l44 serve" in content, (
        "start.bat must contain 'l44 serve' to launch the server on Windows"
    )


def test_start_bat_cd_to_script_dir():
    """start.bat must use %~dp0 to cd to the directory of the .bat file."""
    bat = _REPO_ROOT / "start.bat"
    assert bat.exists(), "start.bat must exist"
    content = bat.read_text()
    assert "%~dp0" in content, (
        "start.bat must use %~dp0 so it works from any working directory "
        "(cd /d \"%~dp0\" is the correct idiom)"
    )


def test_start_bat_prepends_local_bin_to_path():
    """start.bat must prepend %USERPROFILE%\\.local\\bin to PATH so uv is found after setup."""
    bat = _REPO_ROOT / "start.bat"
    assert bat.exists(), "start.bat must exist"
    content = bat.read_text()
    assert r"\local\bin" in content or ".local\\bin" in content, (
        r"start.bat must prepend %USERPROFILE%\.local\bin to PATH "
        r"so uv installed by setup.bat is found on PATH (T-13-45)"
    )


# ---------------------------------------------------------------------------
# PKG-04: requirements.txt pip/venv fallback (D-06)
# ---------------------------------------------------------------------------


def test_requirements_txt_exists():
    """requirements.txt must exist at the repo root (PKG-04, D-06 pip/venv fallback)."""
    req = _REPO_ROOT / "requirements.txt"
    assert req.exists(), (
        "requirements.txt must exist (generated via 'uv export --no-hashes --format requirements.txt')"
    )


def test_requirements_txt_first_line_is_editable():
    """requirements.txt first non-empty, non-comment line must be '-e .' (PKG-04, D-06)."""
    req = _REPO_ROOT / "requirements.txt"
    assert req.exists(), "requirements.txt must exist"
    lines = [
        line.strip()
        for line in req.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines, "requirements.txt must have at least one non-comment line"
    assert lines[0] == "-e .", (
        f"requirements.txt first non-comment line must be '-e .' "
        f"(local package install); got {lines[0]!r}"
    )


def test_requirements_txt_has_pinned_deps():
    """requirements.txt must contain pinned version constraints (PKG-04, D-06)."""
    import re

    req = _REPO_ROOT / "requirements.txt"
    assert req.exists(), "requirements.txt must exist"
    content = req.read_text()
    # uv export produces lines like: httpx==0.27.2
    pinned = re.findall(r"^[\w][\w.\-]+==[0-9]", content, re.MULTILINE)
    assert pinned, (
        "requirements.txt must contain at least one pinned dependency "
        "(e.g., 'httpx==0.27.2'); got an empty or marker-only file"
    )


# ---------------------------------------------------------------------------
# PKG-08: setup_core.py v1.1 provisioning steps
# ---------------------------------------------------------------------------


def test_voice_setup_subprocess_in_setup_core():
    """setup_core.py must reference 'l44 voice setup' subprocess invocation (PKG-08)."""
    content = (_REPO_ROOT / "scripts" / "setup_core.py").read_text()
    assert "voice" in content, (
        "setup_core.py must contain a 'voice' reference for the voice-setup step (PKG-08)"
    )
    # The actual invocation must reference l44 and setup together
    assert "l44" in content and "voice" in content, (
        "setup_core.py must invoke 'l44 voice setup' via subprocess (PKG-08)"
    )


def test_voice_setup_is_non_fatal_check():
    """setup_core.py voice setup must catch CalledProcessError so failure is non-fatal (PKG-08)."""
    content = (_REPO_ROOT / "scripts" / "setup_core.py").read_text()
    # The voice setup section must be wrapped in try/except CalledProcessError
    assert "CalledProcessError" in content, (
        "setup_core.py must catch CalledProcessError so voice setup failure is non-fatal — "
        "a missing whisper model must not abort the install (PKG-08, D-04)"
    )
    assert "voice" in content, "setup_core.py must contain the voice-setup step"


def test_ffmpeg_check_in_setup_core():
    """setup_core.py must check for system ffmpeg (PKG-08, T-13-41 advisory check)."""
    content = (_REPO_ROOT / "scripts" / "setup_core.py").read_text()
    assert "ffmpeg" in content, (
        "setup_core.py must check for system ffmpeg (T-13-41 advisory step)"
    )


def test_ffmpeg_advisory_not_blocking():
    """setup_core.py ffmpeg check must note that system ffmpeg is optional (PyAV bundles it)."""
    content = (_REPO_ROOT / "scripts" / "setup_core.py").read_text()
    assert "ffmpeg" in content, "setup_core.py must reference ffmpeg"
    # The message must indicate ffmpeg is optional / PyAV bundles it
    lower = content.lower()
    assert (
        "optional" in lower
        or "bundled" in lower
        or "pyav" in lower
        or "bundles" in lower
    ), (
        "setup_core.py ffmpeg check must state that system ffmpeg is optional "
        "(voice uses PyAV's bundled FFmpeg, T-13-41) — must not suggest it is required"
    )


def test_schematic_guidance_in_setup_core():
    """setup_core.py must print schematic-render guidance to users (PKG-08)."""
    content = (_REPO_ROOT / "scripts" / "setup_core.py").read_text()
    assert "schematic" in content, (
        "setup_core.py must emit schematic-render guidance "
        "('l44 schematic render <pdf> --pages 61-89') so users know how to load schematics"
    )


def test_voice_setup_uv_resolved_via_shutil_which():
    """setup_core.py must resolve uv via shutil.which before voice subprocess (PKG-08)."""
    content = (_REPO_ROOT / "scripts" / "setup_core.py").read_text()
    assert 'shutil.which("uv")' in content or "shutil.which('uv')" in content, (
        "setup_core.py must call shutil.which(\"uv\") to resolve the uv executable "
        "before the voice-setup subprocess — uv may not be on PATH in all shell environments"
    )


# ---------------------------------------------------------------------------
# UV BOOTSTRAP ordering (Codex HIGH 13-04): setup.sh + setup.bat (T-13-45)
# ---------------------------------------------------------------------------


def test_uv_bootstrap_setup_sh_has_command_v_uv():
    """setup.sh must check 'command -v uv' before calling uv (Codex HIGH 13-04, T-13-45)."""
    content = (_REPO_ROOT / "setup.sh").read_text()
    assert "command -v uv" in content, (
        "setup.sh must check 'command -v uv' so it can detect whether uv is installed "
        "before calling it (Codex HIGH 13-04 — fresh clone has no uv)"
    )


def test_uv_bootstrap_setup_sh_has_installer_url():
    """setup.sh must install uv via the official astral.sh installer (T-13-45, PKG-07)."""
    content = (_REPO_ROOT / "setup.sh").read_text()
    assert "astral.sh/uv/install.sh" in content, (
        "setup.sh must install uv via 'https://astral.sh/uv/install.sh' "
        "when uv is absent (T-13-45, PKG-07 — no pre-installed prerequisites)"
    )


def test_uv_bootstrap_setup_sh_fails_clearly_without_uv():
    """setup.sh must FAIL with an actionable remedy when uv cannot be installed (WR-02).

    The old behaviour fell back to `python3 scripts/setup_core.py`, but setup_core.py
    imports third-party packages (httpx, leopard44_kb) that only exist after `uv sync`,
    so on a bare fresh clone that fallback died with a confusing ModuleNotFoundError —
    exactly the "no prerequisites" case it claimed to handle. The corrected setup.sh
    emits a clear 'uv is required' remedy and exits non-zero instead.
    """
    content = (_REPO_ROOT / "setup.sh").read_text()
    # The broken fallback must be gone.
    assert "python3 scripts/setup_core.py" not in content, (
        "setup.sh must NOT fall back to 'python3 scripts/setup_core.py' — that path "
        "ImportErrors on a fresh clone (WR-02)"
    )
    # The uv-absent branch must emit an actionable remedy and exit non-zero.
    assert "uv is required" in content and "exit 1" in content, (
        "setup.sh must emit a clear 'uv is required' remedy and exit 1 when uv is absent"
    )


def test_uv_bootstrap_ordering_in_setup_sh():
    """setup.sh: 'command -v uv' bootstrap must appear BEFORE the first 'uv run' or 'uv sync'."""
    content = (_REPO_ROOT / "setup.sh").read_text()

    uv_check_pos = content.find("command -v uv")
    uv_run_pos = content.find("uv run")
    uv_sync_pos = content.find("uv sync")

    assert uv_check_pos != -1, "setup.sh must contain 'command -v uv'"

    first_uv_call = min(
        pos for pos in [uv_run_pos, uv_sync_pos] if pos != -1
    )

    assert uv_check_pos < first_uv_call, (
        f"'command -v uv' (pos {uv_check_pos}) must appear BEFORE the first "
        f"'uv run'/'uv sync' call (pos {first_uv_call}) in setup.sh — "
        "calling uv before checking it exists breaks fresh clones (Codex HIGH 13-04)"
    )


def test_uv_bootstrap_setup_bat_has_where_uv():
    """setup.bat must check 'where uv' before delegating to 'uv run' (Codex HIGH 13-04)."""
    content = (_REPO_ROOT / "setup.bat").read_text()
    assert "where uv" in content.lower(), (
        "setup.bat must check 'where uv' to detect whether uv is installed on Windows "
        "(Codex HIGH 13-04 — cannot call 'uv run' before uv exists)"
    )


def test_uv_bootstrap_ordering_in_setup_bat():
    """setup.bat: 'where uv' check must appear BEFORE the 'uv run' delegate line."""
    content = (_REPO_ROOT / "setup.bat").read_text()
    lower = content.lower()
    uv_check_pos = lower.find("where uv")
    uv_run_pos = lower.find("uv run")

    assert uv_check_pos != -1, "setup.bat must contain 'where uv'"
    assert uv_run_pos != -1, "setup.bat must contain 'uv run' delegate"
    assert uv_check_pos < uv_run_pos, (
        f"'where uv' check (pos {uv_check_pos}) must appear BEFORE "
        f"'uv run' (pos {uv_run_pos}) in setup.bat (Codex HIGH 13-04)"
    )


# ---------------------------------------------------------------------------
# Behavioral launcher tests: bash -n syntax validity
# ---------------------------------------------------------------------------


def test_launcher_syntax_start_command_valid():
    """start.command must pass 'bash -n' (syntax valid, no parse errors)."""
    result = subprocess.run(
        ["bash", "-n", str(_REPO_ROOT / "start.command")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"bash -n start.command reported syntax errors:\n{result.stderr}"
    )


def test_launcher_syntax_setup_sh_valid():
    """setup.sh must pass 'bash -n' (syntax valid, no parse errors)."""
    result = subprocess.run(
        ["bash", "-n", str(_REPO_ROOT / "setup.sh")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"bash -n setup.sh reported syntax errors:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Behavioral: mocked-uv start.command invocation
# ---------------------------------------------------------------------------


def test_start_command_with_fake_uv_invokes_run_l44_serve(tmp_path):
    """start.command with a fake uv stub must invoke 'uv run l44 serve' (PKG-03, behavioral).

    Creates a stub uv that logs its args, prepends it to PATH, runs start.command,
    and asserts the stub was called with 'run l44 serve'.
    """
    import os

    # Stub uv: log args to file and exit 0
    stub_uv = tmp_path / "uv"
    log_file = tmp_path / "uv_calls.txt"
    stub_uv.write_text(
        f'#!/usr/bin/env bash\necho "$@" >> "{log_file}"\nexit 0\n'
    )
    stub_uv.chmod(0o755)

    # Env with stub uv on PATH; HOME points to tmp_path so $HOME/.local/bin expansion works
    env = dict(os.environ)
    env["PATH"] = str(tmp_path) + ":" + env.get("PATH", "")
    env["HOME"] = str(tmp_path)

    result = subprocess.run(
        ["bash", str(_REPO_ROOT / "start.command")],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )

    assert log_file.exists(), (
        f"start.command did not invoke the uv stub (log file not created).\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}\nreturncode={result.returncode}"
    )
    uv_args = log_file.read_text().strip()
    assert "run" in uv_args and "l44" in uv_args and "serve" in uv_args, (
        f"start.command must invoke 'uv run l44 serve'; stub recorded: {uv_args!r}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Behavioral: setup_core.main() voice-setup ordering + non-fatal (PKG-08)
# ---------------------------------------------------------------------------


def test_main_ordering_voice_setup_after_uv_sync(monkeypatch):
    """setup_core.main() must call voice setup AFTER the uv-sync step (PKG-08, ordering)."""
    import shutil as _shutil

    call_order: list[str] = []

    monkeypatch.setattr(setup_core, "is_ollama_daemon_running", lambda: True)
    monkeypatch.setattr(setup_core, "resolve_tier", lambda *a, **kw: "16gb")
    monkeypatch.setattr(setup_core, "assert_embedding_compatible", lambda *a, **kw: None)
    monkeypatch.setattr(setup_core, "ensure_model_pulled", lambda *a, **kw: call_order.append("pull"))
    monkeypatch.setattr(setup_core, "write_config", lambda *a, **kw: call_order.append("write_config"))
    monkeypatch.setattr(setup_core, "validate_seed_layout", lambda *a, **kw: None)
    monkeypatch.setattr(setup_core, "_run_seed_ingest", lambda *a, **kw: call_order.append("ingest"))
    monkeypatch.setattr(setup_core, "run_smoke_test", lambda *a, **kw: True)

    # Make ollama and uv appear installed so main() doesn't exit early
    monkeypatch.setattr(
        _shutil,
        "which",
        lambda name: f"/fake/{name}" if name in ("ollama", "uv") else None,
    )

    def _fake_check_call(cmd, *a, **kw):
        if not isinstance(cmd, (list, tuple)):
            return
        cmd_str = " ".join(str(c) for c in cmd)
        if "sync" in cmd_str and "uv" in cmd_str:
            call_order.append("uv_sync")
        elif "pre-commit" in cmd_str:
            call_order.append("pre_commit")
        elif "voice" in cmd_str and "setup" in cmd_str:
            call_order.append("voice_setup")

    monkeypatch.setattr(subprocess, "check_call", _fake_check_call)

    result = setup_core.main(["--tier", "16gb", "--skip-install"])

    assert result == 0, (
        f"main() must return 0 on success; got {result!r}\ncall_order={call_order}"
    )
    assert "uv_sync" in call_order, f"uv sync must be called; call_order={call_order}"
    assert "voice_setup" in call_order, (
        f"voice setup must be called (PKG-08); call_order={call_order}"
    )

    uv_sync_idx = call_order.index("uv_sync")
    voice_setup_idx = call_order.index("voice_setup")
    assert uv_sync_idx < voice_setup_idx, (
        f"voice setup (step 7b) must come AFTER uv sync (step 7); "
        f"call order was: {call_order}"
    )


def test_voice_setup_failure_non_fatal_in_main(monkeypatch):
    """main() must NOT abort when voice setup raises CalledProcessError (PKG-08, non-fatal)."""
    import shutil as _shutil

    call_order: list[str] = []

    monkeypatch.setattr(setup_core, "is_ollama_daemon_running", lambda: True)
    monkeypatch.setattr(setup_core, "resolve_tier", lambda *a, **kw: "16gb")
    monkeypatch.setattr(setup_core, "assert_embedding_compatible", lambda *a, **kw: None)
    monkeypatch.setattr(setup_core, "ensure_model_pulled", lambda *a, **kw: call_order.append("pull"))
    monkeypatch.setattr(setup_core, "write_config", lambda *a, **kw: call_order.append("write_config"))
    monkeypatch.setattr(setup_core, "validate_seed_layout", lambda *a, **kw: None)
    monkeypatch.setattr(setup_core, "_run_seed_ingest", lambda *a, **kw: call_order.append("ingest"))
    monkeypatch.setattr(setup_core, "run_smoke_test", lambda *a, **kw: True)

    monkeypatch.setattr(
        _shutil,
        "which",
        lambda name: f"/fake/{name}" if name in ("ollama", "uv") else None,
    )

    def _fake_check_call(cmd, *a, **kw):
        if not isinstance(cmd, (list, tuple)):
            return
        cmd_str = " ".join(str(c) for c in cmd)
        if "voice" in cmd_str and "setup" in cmd_str:
            call_order.append("voice_setup_attempted")
            raise subprocess.CalledProcessError(1, cmd)
        if "sync" in cmd_str and "uv" in cmd_str:
            call_order.append("uv_sync")
        elif "pre-commit" in cmd_str:
            call_order.append("pre_commit")

    monkeypatch.setattr(subprocess, "check_call", _fake_check_call)

    # Must NOT abort on voice setup failure
    result = setup_core.main(["--tier", "16gb", "--skip-install"])

    assert result == 0, (
        f"main() must return 0 even when voice setup fails (non-fatal, PKG-08); "
        f"got {result!r}\ncall_order={call_order}"
    )
    assert "voice_setup_attempted" in call_order, (
        f"voice setup must have been attempted; call_order={call_order}"
    )
    assert "ingest" in call_order, (
        f"seed ingest must still run after voice setup failure; call_order={call_order}"
    )
