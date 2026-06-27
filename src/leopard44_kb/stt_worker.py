#!/usr/bin/env python3
"""STT worker — invoked as a subprocess from .venv-stt. NEVER imported into the app venv.

Contract:
  argv[1]  = path to audio file (temp file, extension matches Content-Type)
  stdin    = marine initial_prompt string (may be empty)
  stdout   = JSON: {"text": "..."} or {"error": "no_speech"|"garbled"|"not_installed"}
  exit 0   = success (check JSON "error" key for soft errors)
  exit 1   = fatal (missing argv[1] — usage error)

Security invariants:
  - HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 set in os.environ BEFORE WhisperModel is
    constructed (both at module load AND re-applied in main() for test isolation)
  - Model path loaded ONLY from the recorded marker file (never the bare repo-id shorthand)
  - Marker absent/unreadable/recorded-dir-absent → {"error":"not_installed"} — no network fetch
  - Model-load failures (scope A) map to not_installed; decode failures (scope B) map to garbled
  - Neither error escapes as an uncaught traceback

Marker location: data/voice-model-path.txt (relative to repo root, parents[2] of this file),
OR overridden by L44_VOICE_MODEL_MARKER env var (used by tests to redirect without
monkeypatching internals — see 10-01-SUMMARY.md deviation #1).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 0: Set HF offline env flags at module load.
# VOICE-03: guarantees no network fetch even if run in a connected environment.
# os.environ.setdefault preserves caller-set values (e.g. CI override).
# The flags are also re-applied at the top of main() so that test monkeypatching
# (which may delete them after module load) does not bypass the offline guarantee.
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Marker path resolution
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repo root directory (src/leopard44_kb/stt_worker.py → parents[2])."""
    return Path(__file__).resolve().parents[2]


def _read_marker_path() -> str | None:
    """Read and validate the recorded model path from the marker file.

    Returns the resolved path string when the marker file exists and points to
    an existing directory. Returns None on any failure (missing, unreadable,
    empty, or the recorded directory does not exist).

    L44_VOICE_MODEL_MARKER env var overrides the default marker location
    (used by tests to redirect without monkeypatching internals).
    """
    marker_override = os.environ.get("L44_VOICE_MODEL_MARKER")
    if marker_override:
        marker_path = Path(marker_override)
    else:
        marker_path = _repo_root() / "data" / "voice-model-path.txt"

    try:
        recorded = marker_path.read_text().strip()
    except (OSError, IOError):
        return None

    if not recorded:
        return None

    model_dir = Path(recorded)
    if not model_dir.exists():
        return None

    return str(model_dir)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the STT worker: audio path from argv[1], marine prompt from stdin."""
    # --- Usage guard ---
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: stt_worker.py <audio_path>"}))
        sys.exit(1)

    audio_path = sys.argv[1]
    prompt = sys.stdin.read().strip()

    # -----------------------------------------------------------------------
    # Scope A: model RESOLUTION and LOAD.
    #
    # Re-apply the HF offline env flags here so that test isolation (tests may
    # delete these vars via monkeypatch.delenv after module load) does not bypass
    # the offline guarantee.  setdefault is idempotent when the var is already set.
    #
    # The from-import of WhisperModel is placed inside this scope so that the
    # active sys.modules["faster_whisper"] at call time is used — this allows
    # tests to monkeypatch faster_whisper per-call without a stale module-level
    # name binding.
    #
    # Any failure here (missing marker, non-existent dir, WhisperModel
    # construction error) → {"error": "not_installed"} and return.
    # This is a SETUP problem, not a bad-audio problem.
    # -----------------------------------------------------------------------
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from faster_whisper import WhisperModel  # noqa: PLC0415

        model_path = _read_marker_path()
        if model_path is None:
            print(json.dumps({"error": "not_installed"}))
            return

        model = WhisperModel(model_path, device="cpu", compute_type="int8")
    except Exception:
        print(json.dumps({"error": "not_installed"}))
        return

    # -----------------------------------------------------------------------
    # Scope B: audio DECODE and TRANSCRIBE.
    # PyAV/container errors, corrupt blobs, unsupported formats → {"error": "garbled"}.
    # This is a bad-audio problem, not a setup problem.
    # -----------------------------------------------------------------------
    try:
        segments, _info = model.transcribe(
            audio_path,
            beam_size=5,
            language="en",
            initial_prompt=prompt or None,
        )
        text = " ".join(s.text for s in segments).strip()
    except Exception:
        print(json.dumps({"error": "garbled"}))
        return

    # --- No-speech heuristic: transcript with no alphabetic characters ---
    if not any(c.isalpha() for c in text):
        print(json.dumps({"error": "no_speech"}))
    else:
        print(json.dumps({"text": text}))


if __name__ == "__main__":
    main()
