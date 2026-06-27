"""Tests for the `l44 voice setup` command (Phase 10).

Contracts tested:
  - subprocess.run called with venv-create args (python -m venv .venv-stt)
  - subprocess.run called with pip install faster-whisper==1.2.1 (in order, after venv create)
  - snapshot_download is run INSIDE .venv-stt/bin/python (subprocess), NOT imported in the
    app process; the captured stdout path is written to data/voice-model-path.txt
  - The app venv needs NO huggingface_hub installed for `voice setup` to succeed

GAP-1 fix history: The original test_voice_setup_records_model_path injected a fake
huggingface_hub via sys.modules, masking the fact that the app process directly imported
huggingface_hub which is NOT in the app venv (only in .venv-stt). This masked the crash
exactly as Phase 8 CR-01 masked a real-DB bug with mocks. The tests below exercise the
real subprocess contract and would have caught GAP-1.

The @pytest.mark.slow integration tests require network access and are excluded from
the default run (addopts: -m "not ... slow").
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from leopard44_kb.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Content-aware fake subprocess.run
# ---------------------------------------------------------------------------


def _make_fake_run(snapshot_path: Path):
    """Return a subprocess.run stub that is content-aware.

    - For the snapshot_download call (argv contains 'snapshot_download'):
      returns CompletedProcess with stdout=str(snapshot_path), returncode=0.
    - For all other calls (venv create, pip install, warm-load verify):
      returns CompletedProcess with stdout="", returncode=0.

    This mirrors the real subprocess contract so tests don't need to inject
    huggingface_hub into sys.modules.
    """
    calls: list = []

    def _fake_run(cmd, **kw):
        cmd_list = list(cmd) if not isinstance(cmd, str) else [cmd]
        calls.append(cmd_list)
        # Detect the snapshot_download subprocess by its -c program string
        joined = " ".join(str(x) for x in cmd_list)
        if "snapshot_download" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=str(snapshot_path), stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _fake_run, calls


# ---------------------------------------------------------------------------
# test_voice_setup_invokes_venv_create
# ---------------------------------------------------------------------------


def test_voice_setup_invokes_venv_create(monkeypatch, tmp_path):
    """l44 voice setup calls subprocess.run with venv creation args.

    The first subprocess.run call must include '-m' and 'venv' in its argv.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    snapshot_path = tmp_path / "hf_cache" / "snap0"
    snapshot_path.mkdir(parents=True)
    _fake_run, calls = _make_fake_run(snapshot_path)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = runner.invoke(app, ["voice", "setup"])
    assert result.exit_code == 0, (
        f"Expected exit 0; got {result.exit_code}: {result.output!r}"
    )
    # First subprocess call must be venv creation
    venv_calls = [c for c in calls if isinstance(c, list) and "venv" in c]
    assert venv_calls, (
        f"Expected a venv-creation subprocess call; all calls: {calls}"
    )
    first_venv = venv_calls[0]
    assert "-m" in first_venv, (
        f"venv creation must use '-m venv'; got: {first_venv}"
    )


# ---------------------------------------------------------------------------
# test_voice_setup_installs_faster_whisper
# ---------------------------------------------------------------------------


def test_voice_setup_installs_faster_whisper(monkeypatch, tmp_path):
    """l44 voice setup calls pip install faster-whisper==1.2.1.

    The pip install must happen AFTER the venv is created (ordering enforced by
    checking the relative position of the calls list).
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    snapshot_path = tmp_path / "hf_cache" / "snap0"
    snapshot_path.mkdir(parents=True)
    _fake_run, calls = _make_fake_run(snapshot_path)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    runner.invoke(app, ["voice", "setup"])

    pip_calls = [c for c in calls if isinstance(c, list) and "pip" in str(c)]
    assert pip_calls, f"Expected a pip subprocess call; all calls: {calls}"
    assert any("faster-whisper==1.2.1" in str(c) for c in pip_calls), (
        f"Expected 'faster-whisper==1.2.1' in pip call; pip calls: {pip_calls}"
    )

    # Ordering: venv create must come BEFORE pip install
    venv_idx = next(
        (i for i, c in enumerate(calls) if isinstance(c, list) and "venv" in c),
        None,
    )
    pip_idx = next(
        (
            i
            for i, c in enumerate(calls)
            if isinstance(c, list) and any("faster-whisper" in str(x) for x in c)
        ),
        None,
    )
    assert venv_idx is not None, "venv creation call not found"
    assert pip_idx is not None, "pip install call not found"
    assert venv_idx < pip_idx, (
        f"venv create (call #{venv_idx}) must come before pip install (call #{pip_idx})"
    )


# ---------------------------------------------------------------------------
# test_voice_setup_records_model_path — subprocess snapshot path + marker write
# ---------------------------------------------------------------------------


def test_voice_setup_records_model_path(monkeypatch, tmp_path):
    """l44 voice setup delegates snapshot_download to .venv-stt python and writes the
    captured stdout path to the marker file.

    GAP-1 fix: The old test injected huggingface_hub via sys.modules, masking a real crash.
    This test exercises the real subprocess contract:
      1. .venv-stt/bin/python is invoked with a -c program containing 'snapshot_download'
      2. The captured stdout path (not a return value from an in-process import) is written
         to data/voice-model-path.txt.

    No huggingface_hub is injected into sys.modules — the app venv does not need it.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.chdir(tmp_path)

    # Create data/ dir so the marker can be written (mirrors the real repo structure)
    (tmp_path / "data").mkdir(exist_ok=True)

    snapshot_path = tmp_path / "hf_cache" / "Systran" / "faster-whisper-small" / "snap0"
    snapshot_path.mkdir(parents=True)

    _fake_run, calls = _make_fake_run(snapshot_path)
    monkeypatch.setattr(subprocess, "run", _fake_run)

    # NO fake_hf / sys.modules injection — the app process must never need it
    result = runner.invoke(app, ["voice", "setup"])
    assert result.exit_code == 0, (
        f"Expected exit 0; got {result.exit_code}: {result.output!r}"
    )

    # Assert: .venv-stt python is invoked to run a program containing snapshot_download
    snapshot_calls = [
        c for c in calls
        if isinstance(c, list)
        and "snapshot_download" in " ".join(str(x) for x in c)
    ]
    assert snapshot_calls, (
        f"Expected a subprocess call containing 'snapshot_download'; all calls: {calls}"
    )
    snap_call = snapshot_calls[0]
    # The venv python must be the interpreter (contains '.venv-stt')
    assert any(".venv-stt" in str(x) for x in snap_call), (
        f"snapshot_download must be invoked via .venv-stt python; got: {snap_call}"
    )

    # Marker file must be written with the captured stdout snapshot path
    marker = tmp_path / "data" / "voice-model-path.txt"
    assert marker.exists(), (
        f"data/voice-model-path.txt marker must be written after setup; "
        f"searched at {marker}; output: {result.output!r}"
    )
    written_path = marker.read_text().strip()
    assert written_path == str(snapshot_path), (
        f"Marker must contain the captured stdout snapshot path '{snapshot_path}'; "
        f"got: {written_path!r}"
    )


# ---------------------------------------------------------------------------
# test_voice_setup_app_venv_needs_no_huggingface_hub — RED guard (GAP-1 regression)
# ---------------------------------------------------------------------------


def test_voice_setup_app_venv_needs_no_huggingface_hub(monkeypatch, tmp_path):
    """RED guard: voice setup succeeds and writes the marker even when huggingface_hub
    is NOT importable in the app process.

    This is the test that WOULD HAVE CAUGHT GAP-1: the old code did
    `import huggingface_hub` in the app process, which raises ModuleNotFoundError on
    any real machine where huggingface_hub is absent from the app venv.

    This test proves the fix: snapshot_download runs inside .venv-stt/bin/python (the
    only subprocess call containing 'snapshot_download' in its argv), so the app process
    never needs huggingface_hub installed.

    This test should FAIL against the pre-GAP-1-fix code (which imported huggingface_hub
    directly) and PASS against the fixed code.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    snapshot_path = tmp_path / "hf_cache" / "snap0"
    snapshot_path.mkdir(parents=True)
    _fake_run, calls = _make_fake_run(snapshot_path)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    # Explicitly do NOT inject huggingface_hub into sys.modules

    result = runner.invoke(app, ["voice", "setup"])

    # Command must succeed without huggingface_hub in the app venv
    assert result.exit_code == 0, (
        f"voice setup must succeed without huggingface_hub in app venv; "
        f"got exit {result.exit_code}: {result.output!r}"
    )

    # The marker must be written
    marker = tmp_path / "data" / "voice-model-path.txt"
    assert marker.exists(), (
        "data/voice-model-path.txt must be written; voice setup appears to have "
        "exited before the marker write (possible in-process import of huggingface_hub)"
    )

    # Every snapshot_download invocation must go through .venv-stt python (not app process)
    snapshot_calls = [
        c for c in calls
        if isinstance(c, list)
        and "snapshot_download" in " ".join(str(x) for x in c)
    ]
    assert snapshot_calls, (
        "Expected at least one subprocess call delegating snapshot_download to .venv-stt"
    )
    for call in snapshot_calls:
        assert any(".venv-stt" in str(x) for x in call), (
            f"snapshot_download must always be delegated to .venv-stt python, "
            f"not invoked in-process; call: {call}"
        )


# ---------------------------------------------------------------------------
# @pytest.mark.slow — real setup test (requires network + disk space)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_voice_setup_weights_exist(tmp_path):
    """Integration: after real voice setup, data/voice-model-path.txt points to model.bin.

    Requires network access to download whisper-small weights (~462MB).
    Excluded from the default run (addopts: -m "not ... slow").
    """
    import subprocess
    from pathlib import Path

    # Invoke the real CLI
    result = subprocess.run(
        ["uv", "run", "l44", "voice", "setup"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"l44 voice setup failed: {result.stderr!r}"
    )

    # The marker file must point to a dir containing model.bin
    marker = Path("data") / "voice-model-path.txt"
    assert marker.exists(), f"Marker file not written: {marker}"
    model_path = Path(marker.read_text().strip())
    assert model_path.exists(), f"Recorded path does not exist: {model_path}"

    # faster-whisper stores model.bin inside the snapshot dir
    model_files = list(model_path.glob("model.bin"))
    assert model_files, (
        f"model.bin not found under {model_path}; files: {list(model_path.iterdir())}"
    )
