"""RED tests for the /transcribe and /api/voice-status endpoints (Phase 10).

All tests are RED until Plan 03 ships the endpoints in app.py.

Import of create_app is INSIDE each test function body — never at module top.
This ensures pytest collection succeeds even before the symbols exist.

Note: the existing test_web_offline.py::test_static_tree_has_no_external_origins
scanner already covers voice.js for the zero-outbound guarantee — no separate
external-origins test is added here.

Endpoint contracts tested here (Plan 03 implements):
  POST /transcribe  (multipart `file`)
    → 200 {"text": str}
    → 200 {"error": "no_speech"}
    → 503 {"detail": "...l44 voice setup..."} when not installed OR not_installed
    → 413 when blob > 5MB
    → 429 {"detail": "transcription busy"} when lock held — IMMEDIATELY, no queue

  GET /api/voice-status → 200 {"installed": bool}

  Module symbols monkeypatched: _voice_installed, _stt_subprocess, _stt_lock
"""
from __future__ import annotations

import io
import json

import pytest


# ---------------------------------------------------------------------------
# /transcribe — STT not installed (venv missing)
# ---------------------------------------------------------------------------


def test_transcribe_stt_not_installed(monkeypatch, tmp_path):
    """POST /transcribe returns 503 when .venv-stt/bin/python does not exist.

    _voice_installed patched to False. Detail must contain 'l44 voice setup'.
    RED: _voice_installed and /transcribe do not exist in app.py yet.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app  # type: ignore[attr-defined]
    from fastapi.testclient import TestClient

    monkeypatch.setattr("leopard44_kb.web.app._voice_installed", lambda: False)

    client = TestClient(create_app())
    blob = io.BytesIO(b"fake audio bytes")
    resp = client.post("/transcribe", files={"file": ("audio.webm", blob, "audio/webm")})
    assert resp.status_code == 503, (
        f"Expected 503 when STT not installed; got {resp.status_code}: {resp.text}"
    )
    assert "l44 voice setup" in resp.json().get("detail", ""), (
        f"Expected 'l44 voice setup' hint in detail; got: {resp.json()!r}"
    )


# ---------------------------------------------------------------------------
# /transcribe — happy path (text returned)
# ---------------------------------------------------------------------------


def test_transcribe_returns_text(monkeypatch, tmp_path):
    """POST /transcribe returns {"text": "..."} when STT worker succeeds.

    _voice_installed=True; _stt_subprocess stubbed to return {"text": ...}.
    RED: /transcribe does not exist yet.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr("leopard44_kb.web.app._voice_installed", lambda: True)
    monkeypatch.setattr(
        "leopard44_kb.web.app._stt_subprocess",
        lambda path: {"text": "where is the raw water impeller"},
    )
    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    blob = io.BytesIO(b"fake audio bytes")
    resp = client.post("/transcribe", files={"file": ("audio.webm", blob, "audio/webm")})
    assert resp.status_code == 200, f"Expected 200; got {resp.status_code}: {resp.text}"
    assert resp.json()["text"] == "where is the raw water impeller", (
        f"Unexpected text: {resp.json()!r}"
    )


# ---------------------------------------------------------------------------
# /transcribe — no speech detected
# ---------------------------------------------------------------------------


def test_transcribe_no_speech(monkeypatch, tmp_path):
    """POST /transcribe returns {"error": "no_speech"} when STT worker detects no speech.

    RED: _stt_subprocess and /transcribe do not exist yet.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr("leopard44_kb.web.app._voice_installed", lambda: True)
    monkeypatch.setattr(
        "leopard44_kb.web.app._stt_subprocess",
        lambda path: {"error": "no_speech"},
    )
    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    blob = io.BytesIO(b"fake audio bytes")
    resp = client.post("/transcribe", files={"file": ("audio.webm", blob, "audio/webm")})
    assert resp.status_code == 200, f"Expected 200 for no_speech; got {resp.status_code}"
    assert resp.json().get("error") == "no_speech", (
        f"Expected error=no_speech; got: {resp.json()!r}"
    )


# ---------------------------------------------------------------------------
# /transcribe — worker returns not_installed (review concern #1/#4)
# ---------------------------------------------------------------------------


def test_transcribe_worker_not_installed(monkeypatch, tmp_path):
    """POST /transcribe returns 503 when worker returns {"error": "not_installed"}.

    worker not_installed means the recorded model path is missing/unreadable.
    The endpoint must map this to HTTP 503 with the l44 voice setup hint,
    NOT a 200 garbled response.
    RED: /transcribe + _stt_subprocess + not_installed→503 mapping do not exist.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr("leopard44_kb.web.app._voice_installed", lambda: True)
    monkeypatch.setattr(
        "leopard44_kb.web.app._stt_subprocess",
        lambda path: {"error": "not_installed"},
    )
    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    blob = io.BytesIO(b"fake audio bytes")
    resp = client.post("/transcribe", files={"file": ("audio.webm", blob, "audio/webm")})
    assert resp.status_code == 503, (
        f"Expected 503 when worker returns not_installed; got {resp.status_code}: {resp.text}"
    )
    assert "l44 voice setup" in resp.json().get("detail", ""), (
        f"Expected 'l44 voice setup' in detail; got: {resp.json()!r}"
    )


# ---------------------------------------------------------------------------
# /transcribe — oversize 413 (blob > 5MB)
# ---------------------------------------------------------------------------


def test_transcribe_oversize_413(monkeypatch, tmp_path):
    """POST /transcribe with a >5MB body returns 413 without calling _stt_subprocess.

    The spy asserts _stt_subprocess is NEVER called for an oversized blob.
    RED: /transcribe, size limit, and _stt_subprocess do not exist yet.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr("leopard44_kb.web.app._voice_installed", lambda: True)

    spy_calls: list = []

    def _spy_stt(path):
        spy_calls.append(path)
        return {"text": "should not be reached"}

    monkeypatch.setattr("leopard44_kb.web.app._stt_subprocess", _spy_stt)

    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    # 5MB + 1 byte — over the limit
    oversized = io.BytesIO(b"X" * (5 * 1024 * 1024 + 1))
    resp = client.post(
        "/transcribe", files={"file": ("big.webm", oversized, "audio/webm")}
    )
    assert resp.status_code == 413, (
        f"Expected 413 for oversized blob; got {resp.status_code}: {resp.text}"
    )
    assert spy_calls == [], (
        f"_stt_subprocess must NOT be called for oversized blobs; calls: {spy_calls}"
    )


# ---------------------------------------------------------------------------
# /transcribe — non-blocking 429 (busy lock, no queue) (review concern #2/#3)
# ---------------------------------------------------------------------------


def test_transcribe_busy_429(monkeypatch, tmp_path):
    """POST /transcribe returns 429 immediately when _stt_lock is already held.

    The 429 must be returned IMMEDIATELY — no blocking, no queuing.
    _stt_subprocess must NOT be called.
    RED: _stt_lock (asyncio.Lock), /transcribe, and the non-blocking check do not exist.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr("leopard44_kb.web.app._voice_installed", lambda: True)

    spy_calls: list = []

    def _spy_stt(path):
        spy_calls.append(path)
        return {"text": "should not be reached"}

    monkeypatch.setattr("leopard44_kb.web.app._stt_subprocess", _spy_stt)

    # Monkeypatch _stt_lock to a stub whose .locked() returns True
    import asyncio

    class _HeldLock:
        """Stub that appears to be a held asyncio.Lock."""

        def locked(self) -> bool:
            return True

        async def acquire(self):
            raise AssertionError("acquire must not be called — lock is held")

        def release(self):
            pass

        async def __aenter__(self):
            raise AssertionError("__aenter__ must not be called — lock is held")

        async def __aexit__(self, *a):
            pass

    monkeypatch.setattr("leopard44_kb.web.app._stt_lock", _HeldLock())

    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    blob = io.BytesIO(b"fake audio bytes")
    resp = client.post("/transcribe", files={"file": ("audio.webm", blob, "audio/webm")})
    assert resp.status_code == 429, (
        f"Expected 429 when lock is held; got {resp.status_code}: {resp.text}"
    )
    assert spy_calls == [], (
        f"_stt_subprocess must NOT be called when lock is held; calls: {spy_calls}"
    )
    # "busy" must appear in the detail
    detail = resp.json().get("detail", "")
    assert "busy" in detail.lower() or "transcription" in detail.lower(), (
        f"Expected 'busy'/'transcription' in 429 detail; got: {resp.json()!r}"
    )


# ---------------------------------------------------------------------------
# GET / — mic button in HTML
# ---------------------------------------------------------------------------


def test_mic_button_in_index(monkeypatch, tmp_path):
    """GET / HTML contains the mic button element (id='mic-btn').

    RED: mic button in index.html does not exist yet (Plan 04).
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200, f"GET / returned {resp.status_code}"
    assert 'id="mic-btn"' in resp.text, (
        f"GET / HTML must contain id=\"mic-btn\"; not found in HTML snippet: "
        f"{resp.text[:500]!r}"
    )


# ---------------------------------------------------------------------------
# GET /api/voice-status — mandatory gating (D-05)
# ---------------------------------------------------------------------------


def test_voice_status_endpoint(monkeypatch, tmp_path):
    """GET /api/voice-status returns {"installed": bool}.

    This endpoint is mandatory (D-05): the mic button in the UI gates on it
    to show the 'install voice' hint until setup has run.
    RED: /api/voice-status does not exist in app.py yet.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr("leopard44_kb.web.app._voice_installed", lambda: False)

    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/api/voice-status")
    assert resp.status_code == 200, (
        f"GET /api/voice-status must return 200; got {resp.status_code}"
    )
    data = resp.json()
    assert "installed" in data, (
        f"Response must have 'installed' key; got: {data!r}"
    )
    assert isinstance(data["installed"], bool), (
        f"'installed' must be a bool; got: {type(data['installed'])!r}"
    )


# ---------------------------------------------------------------------------
# GAP-2: off-loop transcription — concurrent /transcribe + /api/voice-status
# must not hang when a blocking transcription is in-flight (T-10-20)
# ---------------------------------------------------------------------------


def test_transcribe_offloads_blocking_call(monkeypatch, tmp_path):
    """transcribe_endpoint must offload _stt_subprocess via asyncio.to_thread so
    the blocking subprocess.run+SQLite does not freeze the uvicorn event loop.

    Strategy: spy on asyncio.to_thread via monkeypatch so the test FAILS if
    _stt_subprocess is called synchronously on the event loop (i.e. the old code
    path `result = _stt_subprocess(tmp_path)` still exists).

    RED: fails until transcribe_endpoint replaces the direct call with
    `result = await asyncio.to_thread(_stt_subprocess, tmp_path)`.
    """
    import asyncio as _asyncio

    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr("leopard44_kb.web.app._voice_installed", lambda: True)
    monkeypatch.setattr(
        "leopard44_kb.web.app._stt_subprocess",
        lambda path: {"text": "ok"},
    )

    to_thread_calls: list = []
    real_to_thread = _asyncio.to_thread

    async def _spy_to_thread(func, *args, **kwargs):
        to_thread_calls.append((func, args))
        return await real_to_thread(func, *args, **kwargs)

    # asyncio is imported at module top in app.py — patch the module attribute
    import leopard44_kb.web.app as _app_mod
    monkeypatch.setattr(_app_mod.asyncio, "to_thread", _spy_to_thread)

    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    blob = io.BytesIO(b"fake audio")
    resp = client.post("/transcribe", files={"file": ("audio.webm", blob, "audio/webm")})

    assert resp.status_code == 200, (
        f"Expected 200 from /transcribe; got {resp.status_code}: {resp.text}"
    )
    assert resp.json() == {"text": "ok"}, f"Unexpected body: {resp.json()!r}"
    assert len(to_thread_calls) == 1, (
        f"asyncio.to_thread must be called exactly once for _stt_subprocess offload; "
        f"calls: {to_thread_calls!r} — did you forget `await asyncio.to_thread(...)`?"
    )
    import leopard44_kb.web.app as _app
    assert to_thread_calls[0][0] is _app._stt_subprocess, (
        f"asyncio.to_thread must be called with _stt_subprocess as the first arg; "
        f"got: {to_thread_calls[0][0]!r}"
    )


# ---------------------------------------------------------------------------
# GAP-3: _voice_installed() must require BOTH venv python AND model marker
# (D-05 honest gate — venv present but marker absent → installed:false)
# ---------------------------------------------------------------------------


def test_voice_status_venv_without_marker_not_installed(monkeypatch, tmp_path):
    """GET /api/voice-status returns installed:false when venv python exists but
    the model marker is absent or points to a non-existent directory.

    Also asserts the positive case: venv python + marker pointing to existing dir
    → installed:true (with the venv-python check patched to True).

    RED: fails until _voice_installed() checks the marker in addition to the venv.
    """
    import os

    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))

    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    # --- Negative case: marker missing (env points to non-existent file) ---
    missing_marker = str(tmp_path / "no-such-marker.txt")
    monkeypatch.setenv("L44_VOICE_MODEL_MARKER", missing_marker)

    # Patch the venv-python existence check so only the marker path matters
    import pathlib
    real_exists = pathlib.Path.exists

    def _patched_exists(self):
        if str(self).endswith(".venv-stt/bin/python"):
            return True  # venv python appears to exist
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", _patched_exists)

    client = TestClient(create_app())
    resp = client.get("/api/voice-status")
    assert resp.status_code == 200, f"Expected 200; got {resp.status_code}"
    data = resp.json()
    assert data.get("installed") is False, (
        f"installed must be False when marker is absent; got: {data!r}"
    )

    # --- Positive case: marker present AND points to an existing dir ---
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    marker_file = tmp_path / "voice-model-path.txt"
    marker_file.write_text(str(model_dir))
    monkeypatch.setenv("L44_VOICE_MODEL_MARKER", str(marker_file))

    client2 = TestClient(create_app())
    resp2 = client2.get("/api/voice-status")
    assert resp2.status_code == 200, f"Expected 200; got {resp2.status_code}"
    data2 = resp2.json()
    assert data2.get("installed") is True, (
        f"installed must be True when venv + valid marker both present; got: {data2!r}"
    )


# ---------------------------------------------------------------------------
# GAP-4: _stt_subprocess JSONDecodeError must return {"error":"garbled"} without
# leaking raw worker stdout to the client (T-10-12 / V7 — mirrors WR-02 stderr fix)
# ---------------------------------------------------------------------------


def test_stt_subprocess_jsondecode_returns_garbled_no_leak(monkeypatch, tmp_path):
    """_stt_subprocess returns {"error":"garbled"} (NOT raw stdout) when the
    STT worker prints non-JSON output.

    Asserts the raw stdout string is NOT present anywhere in the returned dict.

    RED: fails until the JSONDecodeError branch logs + returns garbled, not stdout.
    """
    import subprocess as _subprocess
    import sys

    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))

    # The leaked string that must NOT appear in the return value
    leaked_string = "TRACEBACK leak /home/user/secret/path not json"

    # subprocess is imported locally inside _stt_subprocess, so patch via sys.modules
    # to intercept the local `import subprocess` call.
    class _FakeSubprocess:
        CompletedProcess = _subprocess.CompletedProcess

        def run(self, cmd, **kwargs):
            return _subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=leaked_string,
                stderr="",
            )

    fake_subprocess = _FakeSubprocess()
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

    # open_db / apply_migrations / build_stt_prompt are also locally imported —
    # patch them via their canonical module paths so the local import resolves correctly.
    import leopard44_kb.store as _store
    import leopard44_kb.schema as _schema

    monkeypatch.setattr(_store, "open_db", lambda: _FakeConn())
    monkeypatch.setattr(_schema, "apply_migrations", lambda conn: None)
    monkeypatch.setattr("leopard44_kb.web.app.build_stt_prompt", lambda conn: "Marine vocabulary.")

    from leopard44_kb.web.app import _stt_subprocess

    result = _stt_subprocess(str(tmp_path / "audio.webm"))

    assert result == {"error": "garbled"}, (
        f"JSONDecodeError path must return exactly {{\"error\":\"garbled\"}}; got: {result!r}"
    )
    result_str = str(result)
    assert "secret" not in result_str and leaked_string not in result_str, (
        f"Raw stdout must NOT appear in the return value; got: {result!r}"
    )


class _FakeConn:
    """Minimal DB connection stub for test_stt_subprocess_jsondecode_returns_garbled_no_leak."""

    def execute(self, sql, *args):
        return []

    def close(self):
        pass
