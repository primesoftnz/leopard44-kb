"""RED tests for stt_worker.py subprocess contract (Phase 10).

All unit tests are RED until Plan 03 ships src/leopard44_kb/stt_worker.py.

Key contracts tested:
  - argv[1] = audio path; stdin = marine prompt; stdout = JSON {"text":...} or {"error":...}
  - HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 set in os.environ BEFORE model construction
  - WhisperModel constructed with the RECORDED local path (from data/voice-model-path.txt),
    NEVER with bare "small" as repo-id fallback
  - Missing/unreadable marker → {"error": "not_installed"}, no model load with "small"

The @pytest.mark.slow integration test requires .venv-stt + whisper weights and is
excluded from the default run (see pyproject.toml addopts: -m "not ... slow").
"""
from __future__ import annotations

import io
import json
import sys
import types

import pytest


def _make_fake_faster_whisper(transcript_text: str = " where is the impeller"):
    """Build a fake faster_whisper module that returns a fixed transcript."""
    fake_fw = types.ModuleType("faster_whisper")

    class _FakeModel:
        _constructed_with: list = []

        def __init__(self, model_path_or_id, *args, **kwargs):
            _FakeModel._constructed_with.append(model_path_or_id)

        def transcribe(self, path, **kwargs):
            seg = types.SimpleNamespace(text=transcript_text)
            return [seg], types.SimpleNamespace()

    fake_fw.WhisperModel = _FakeModel
    return fake_fw


# ---------------------------------------------------------------------------
# test_worker_returns_text — happy path
# ---------------------------------------------------------------------------


def test_worker_returns_text(monkeypatch, tmp_path):
    """stt_worker.main() prints {"text": "..."} JSON when faster_whisper faked + marker present.

    Monkeypatches faster_whisper before importing stt_worker so no model loads.
    RED: stt_worker.py does not exist yet.
    """
    audio = tmp_path / "test.webm"
    audio.write_bytes(b"fake")

    # Write a recorded model-path marker (data/voice-model-path.txt)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    marker = tmp_path / "voice-model-path.txt"
    marker.write_text(str(model_dir))

    # Inject fake faster_whisper BEFORE import
    fake_fw = _make_fake_faster_whisper(" where is the impeller")
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    # Set offline env + marker env so worker finds the marker
    monkeypatch.setenv("L44_VOICE_MODEL_MARKER", str(marker))

    import leopard44_kb.stt_worker as worker  # type: ignore[import-not-found]

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("Marine vocabulary: impeller."))
    monkeypatch.setattr(sys, "argv", ["stt_worker.py", str(audio)])
    monkeypatch.setattr(sys, "stdout", captured)

    worker.main()

    result = json.loads(captured.getvalue())
    assert result.get("text"), f"Expected non-empty text; got: {result}"
    assert "error" not in result, f"Unexpected error; got: {result}"


# ---------------------------------------------------------------------------
# test_worker_no_speech — no-alpha transcript → error: no_speech
# ---------------------------------------------------------------------------


def test_worker_no_speech(monkeypatch, tmp_path):
    """stt_worker.main() prints {"error": "no_speech"} when transcript has no alphabetic chars.

    RED: stt_worker.py does not exist yet.
    """
    audio = tmp_path / "silence.webm"
    audio.write_bytes(b"fake")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    marker = tmp_path / "voice-model-path.txt"
    marker.write_text(str(model_dir))

    # transcript with no alphabetic chars
    fake_fw = _make_fake_faster_whisper("... ... ...")
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.setenv("L44_VOICE_MODEL_MARKER", str(marker))

    import leopard44_kb.stt_worker as worker

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(sys, "argv", ["stt_worker.py", str(audio)])
    monkeypatch.setattr(sys, "stdout", captured)

    worker.main()

    result = json.loads(captured.getvalue())
    assert result.get("error") == "no_speech", (
        f"Expected error=no_speech for non-alpha transcript; got: {result}"
    )


# ---------------------------------------------------------------------------
# test_worker_sets_offline_env — HF offline flags (review concern #4)
# ---------------------------------------------------------------------------


def test_worker_sets_offline_env(monkeypatch, tmp_path):
    """stt_worker sets HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before WhisperModel.

    This ensures the worker never attempts a network fetch even if run in a connected
    environment. The env vars must be set BEFORE model construction.
    RED: stt_worker.py does not exist yet.
    """
    audio = tmp_path / "test.webm"
    audio.write_bytes(b"fake")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    marker = tmp_path / "voice-model-path.txt"
    marker.write_text(str(model_dir))

    import os

    env_at_construction: dict = {}

    fake_fw = types.ModuleType("faster_whisper")

    class _EnvCapturingModel:
        _constructed_with: list = []

        def __init__(self, model_path_or_id, *args, **kwargs):
            # Capture env at the moment of construction
            env_at_construction.update(os.environ.copy())
            _EnvCapturingModel._constructed_with.append(model_path_or_id)

        def transcribe(self, path, **kwargs):
            seg = types.SimpleNamespace(text=" impeller")
            return [seg], types.SimpleNamespace()

    fake_fw.WhisperModel = _EnvCapturingModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.setenv("L44_VOICE_MODEL_MARKER", str(marker))

    # Ensure they are NOT already set (so our assertion is meaningful)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    import leopard44_kb.stt_worker as worker

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(sys, "argv", ["stt_worker.py", str(audio)])
    monkeypatch.setattr(sys, "stdout", captured)

    worker.main()

    assert env_at_construction.get("HF_HUB_OFFLINE") == "1", (
        f"HF_HUB_OFFLINE must be '1' at WhisperModel construction; "
        f"env: {env_at_construction.get('HF_HUB_OFFLINE')!r}"
    )
    assert env_at_construction.get("TRANSFORMERS_OFFLINE") == "1", (
        f"TRANSFORMERS_OFFLINE must be '1' at WhisperModel construction; "
        f"env: {env_at_construction.get('TRANSFORMERS_OFFLINE')!r}"
    )


# ---------------------------------------------------------------------------
# test_worker_loads_recorded_path_only — STRICT offline (review concern #1)
# ---------------------------------------------------------------------------


def test_worker_loads_recorded_path_only(monkeypatch, tmp_path):
    """stt_worker constructs WhisperModel with the recorded local path, NOT bare 'small'.

    The marker file (data/voice-model-path.txt) contains the path written by
    `l44 voice setup` via snapshot_download. The worker MUST read and use
    that path — it must never fall back to the bare 'small' repo-id which would
    trigger a network fetch.
    RED: stt_worker.py does not exist yet.
    """
    audio = tmp_path / "test.webm"
    audio.write_bytes(b"fake")

    # Write a realistic recorded path
    model_dir = tmp_path / "hf_cache" / "snapshots" / "abc123"
    model_dir.mkdir(parents=True)
    marker = tmp_path / "voice-model-path.txt"
    marker.write_text(str(model_dir))

    constructed_with: list = []

    fake_fw = types.ModuleType("faster_whisper")

    class _TrackingModel:
        def __init__(self, model_path_or_id, *args, **kwargs):
            constructed_with.append(model_path_or_id)

        def transcribe(self, path, **kwargs):
            seg = types.SimpleNamespace(text=" impeller test")
            return [seg], types.SimpleNamespace()

    fake_fw.WhisperModel = _TrackingModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.setenv("L44_VOICE_MODEL_MARKER", str(marker))

    import leopard44_kb.stt_worker as worker

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(sys, "argv", ["stt_worker.py", str(audio)])
    monkeypatch.setattr(sys, "stdout", captured)

    worker.main()

    assert len(constructed_with) == 1, (
        f"WhisperModel must be constructed exactly once; calls: {constructed_with}"
    )
    assert constructed_with[0] == str(model_dir), (
        f"WhisperModel must be constructed with the recorded local path {str(model_dir)!r}; "
        f"got: {constructed_with[0]!r}"
    )
    assert constructed_with[0] != "small", (
        "WhisperModel must NEVER be constructed with bare 'small' repo-id (network fetch risk)"
    )


# ---------------------------------------------------------------------------
# test_worker_missing_marker_not_installed — no fallback (review concern #1)
# ---------------------------------------------------------------------------


def test_worker_missing_marker_not_installed(monkeypatch, tmp_path):
    """stt_worker outputs {"error": "not_installed"} when marker is missing/unreadable.

    If data/voice-model-path.txt does not exist or points to a nonexistent dir,
    the worker must output {"error": "not_installed"} and NEVER construct WhisperModel
    with the bare "small" repo-id.
    RED: stt_worker.py does not exist yet.
    """
    audio = tmp_path / "test.webm"
    audio.write_bytes(b"fake")

    # Point marker to a nonexistent file
    marker = tmp_path / "voice-model-path.txt"
    # Do NOT create the marker file — it is absent

    constructed_with: list = []

    fake_fw = types.ModuleType("faster_whisper")

    class _TrackingModel:
        def __init__(self, model_path_or_id, *args, **kwargs):
            constructed_with.append(model_path_or_id)

        def transcribe(self, path, **kwargs):
            seg = types.SimpleNamespace(text=" impeller")
            return [seg], types.SimpleNamespace()

    fake_fw.WhisperModel = _TrackingModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.setenv("L44_VOICE_MODEL_MARKER", str(marker))

    import leopard44_kb.stt_worker as worker

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(sys, "argv", ["stt_worker.py", str(audio)])
    monkeypatch.setattr(sys, "stdout", captured)

    worker.main()

    result = json.loads(captured.getvalue())
    assert result.get("error") == "not_installed", (
        f"Expected error=not_installed when marker is absent; got: {result}"
    )
    assert "small" not in [str(c) for c in constructed_with], (
        f"WhisperModel must NEVER be constructed with 'small' on missing marker; "
        f"constructed_with: {constructed_with}"
    )
    assert constructed_with == [], (
        f"WhisperModel must NOT be constructed at all when marker is absent; "
        f"constructed_with: {constructed_with}"
    )


# ---------------------------------------------------------------------------
# @pytest.mark.slow — real .venv-stt integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_worker_real_speech(voice_speech_fixture):
    """Integration: invoke the real .venv-stt worker on short_marine_query.webm.

    Requires .venv-stt/bin/python and whisper small weights to be present.
    Excluded from the default run (addopts: -m "not ... slow").
    Pass criteria: output is {"text": "..."} with non-empty text.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    venv_python = repo / ".venv-stt" / "bin" / "python"
    stt_script = repo / "src" / "leopard44_kb" / "stt_worker.py"

    if not venv_python.exists():
        pytest.skip("'.venv-stt' not found — run 'l44 voice setup' first")

    result = subprocess.run(
        [str(venv_python), str(stt_script), str(voice_speech_fixture)],
        input="Marine vocabulary: impeller, anchor, bilge, reef.",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"stt_worker exited {result.returncode}: stderr={result.stderr!r}"
    )
    data = json.loads(result.stdout)
    assert "text" in data and data["text"], (
        f"Expected non-empty text from real worker; got: {data!r}"
    )
