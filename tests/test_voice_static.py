"""Static RED/guard tests for voice.js (Phase 10).

These tests read voice.js from disk statically — no server startup required.

test_voice_js_has_no_auto_submit:
  Locks the D-02 review-before-Ask requirement: voice.js must NEVER call
  requestSubmit(), .submit(), or runQuery() automatically. The transcript
  must populate the #question textarea and let the owner press Ask.

  Behaviour during Wave 0 (before Plan 04 ships voice.js):
    - SKIPs cleanly (not fails) while voice.js is absent.
    - Turns into a GREEN guard the moment Plan 04 writes voice.js without auto-submit.
    - Fails LOUDLY (assertion error) if a later edit introduces auto-submit.

test_voice_js_guards_recorder_start:
  RED guard for GAP-5 (codex cross-AI review): MediaRecorder construction and
  .start(250) inside startRecording() must be wrapped in a try/catch whose catch
  block stops all acquired stream tracks, resets _micState to 'idle', clears timers,
  and shows an error — so a recorder-init failure never leaks the microphone stream.

  Behaviour:
    - SKIPs cleanly while voice.js is absent (Wave-0 convention).
    - Is RED against the unguarded voice.js (before Plan 07 adds the try/catch).
    - Turns GREEN once the try/catch with full cleanup is present.
    - Fails LOUDLY if the guard is later removed.

Note: the zero-outbound guard for voice.js is covered by the existing
test_web_offline.py::test_static_tree_has_no_external_origins scanner.
No duplicate test is added here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_VOICE_JS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "leopard44_kb"
    / "web"
    / "static"
    / "voice.js"
)


def test_voice_js_has_no_auto_submit():
    """voice.js must not auto-submit the query form (D-02 review-before-Ask).

    Asserts that voice.js contains NONE of:
      - requestSubmit(          (HTMLFormElement.requestSubmit API)
      - .submit()               (HTMLFormElement.submit() shorthand)
      - runQuery(               (Leopard 44 KB app.js query runner — must not be called from voice.js)

    SKIPs cleanly while voice.js does not yet exist (Wave 0).
    Fails loudly if any auto-submit path is introduced in a future edit.
    """
    if not _VOICE_JS_PATH.exists():
        pytest.skip(
            f"voice.js not yet present at {_VOICE_JS_PATH} — "
            "this test becomes a GREEN guard once Plan 04 ships voice.js"
        )

    content = _VOICE_JS_PATH.read_text(encoding="utf-8")

    forbidden = [
        "requestSubmit(",
        ".submit()",
        "runQuery(",
    ]
    violations = [pat for pat in forbidden if pat in content]

    assert not violations, (
        f"D-02 violation: voice.js contains auto-submit pattern(s): {violations!r}. "
        "The transcript must populate #question and let the owner press Ask — "
        "never auto-submit the query form."
    )


def test_voice_js_guards_recorder_start():
    """GAP-5 guard: MediaRecorder construction/start must be inside a try/catch that
    stops stream tracks on failure (no mic-stream leak on recorder-init errors).

    Asserts that within startRecording():
      - 'new MediaRecorder' and '.start(' both appear (recorder is constructed + started).
      - A try/catch wraps that region: 'try {' and 'catch' are present in startRecording.
      - The catch block stops the stream tracks: 'getTracks' AND '.stop(' appear.
      - The catch block resets state: '_micState = \\'idle\\'' appears.
      - The catch block clears timers: 'clearInterval' AND 'clearTimeout' appear.
      - The catch block shows a voice error: '_showVoiceError(' appears.

    Strategy: extract the startRecording() function body by slicing from its declaration
    to the next top-level function declaration, then assert on required substrings.
    This is robust to line-number drift.

    RED guard for GAP-5 — fails against the unguarded voice.js; turns GREEN once
    Plan 07 adds the try/catch with full cleanup.

    SKIPs cleanly if voice.js is absent (Wave-0 convention).
    """
    if not _VOICE_JS_PATH.exists():
        pytest.skip(
            f"voice.js not yet present at {_VOICE_JS_PATH} — "
            "this test becomes a RED guard once voice.js exists without the try/catch"
        )

    content = _VOICE_JS_PATH.read_text(encoding="utf-8")

    # Extract the startRecording body: from declaration to next async/function declaration
    start_idx = content.find("async function startRecording(")
    assert start_idx != -1, "startRecording() not found in voice.js"

    # Find the next top-level function after startRecording
    # (stopRecording follows in the current layout)
    next_fn_idx = content.find("\nfunction ", start_idx + 1)
    async_fn_idx = content.find("\nasync function ", start_idx + 1)
    # Take the earlier of the two as the end of startRecording's body
    candidates = [i for i in (next_fn_idx, async_fn_idx) if i != -1]
    end_idx = min(candidates) if candidates else len(content)

    fn_body = content[start_idx:end_idx]

    # 1. Recorder must be constructed and started inside startRecording
    assert "new MediaRecorder" in fn_body, (
        "startRecording() does not contain 'new MediaRecorder' — cannot guard what is not there"
    )
    assert ".start(" in fn_body, (
        "startRecording() does not contain '.start(' — cannot guard what is not there"
    )

    # 2. A try/catch must wrap the recorder construction region.
    # The try block must appear BEFORE _micState = 'recording' (the timers+state are set
    # after the recorder starts cleanly, so try must precede the success path).
    mic_recording_idx = fn_body.find("_micState = 'recording'")
    assert mic_recording_idx != -1, (
        "startRecording() does not set _micState = 'recording' — unexpected shape"
    )
    pre_success_region = fn_body[:mic_recording_idx]
    assert "try {" in pre_success_region or "try{" in pre_success_region, (
        "GAP-5: no 'try {' found before '_micState = \\'recording\\'' in startRecording(). "
        "MediaRecorder construction/start must be wrapped in a try/catch to prevent "
        "stream leaks when the recorder throws on init."
    )
    assert "catch" in fn_body, (
        "GAP-5: no 'catch' found in startRecording() — try/catch is incomplete."
    )

    # 3. The catch block must stop stream tracks (prevent mic-stream leak)
    assert "getTracks" in fn_body, (
        "GAP-5: 'getTracks' not found in startRecording(). "
        "The catch block must call stream.getTracks().forEach(t => t.stop()) to release the mic."
    )
    # .stop() appears on tracks; it also appears in stopRecording but we just need presence in fn_body
    assert ".stop(" in fn_body, (
        "GAP-5: '.stop(' not found in startRecording(). "
        "The catch block must stop the stream tracks."
    )

    # 4. The catch block must reset _micState to 'idle'
    assert "_micState = 'idle'" in fn_body, (
        "GAP-5: \"_micState = 'idle'\" not found in startRecording(). "
        "The catch block must reset _micState to 'idle' so the UI returns to a usable state."
    )

    # 5. The catch block must clear both timers
    assert "clearInterval" in fn_body, (
        "GAP-5: 'clearInterval' not found in startRecording(). "
        "The catch block must clearInterval(_timerInterval) to avoid timer leaks."
    )
    assert "clearTimeout" in fn_body, (
        "GAP-5: 'clearTimeout' not found in startRecording(). "
        "The catch block must clearTimeout(_maxTimer) to avoid timer leaks."
    )

    # 6. The catch block must show a voice error
    assert "_showVoiceError(" in fn_body, (
        "GAP-5: '_showVoiceError(' not found in startRecording(). "
        "The catch block must call _showVoiceError(...) to inform the user of the failure."
    )
