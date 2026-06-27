// voice.js — phone-facing mic UI for offline voice query (Phase 10 / Plan 04)
// No external imports, no CDN, no browser speech API (D-08 / VIS-02)
// Uses MediaRecorder + local /transcribe only (T-10-15 / D-08)
// All server-derived text rendered via .textContent only — never innerHTML (T-10-17)

/* ==========================================================================
   MODULE STATE
   ========================================================================== */

let _micState = 'idle';          // 'idle' | 'recording' | 'transcribing' | 'disabled'
let _mediaRecorder = null;
let _chunks = [];
let _timerInterval = null;
let _maxTimer = null;
let _timerSeconds = 0;
let _voiceInstalled = false;     // set by _initVoice after /api/voice-status

/* ==========================================================================
   MIME TYPE NEGOTIATION (D-08 / Q3)
   ========================================================================== */

function getBestMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/ogg;codecs=opus',
  ];
  for (const t of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(t)) return t;
  }
  return '';
}

function _blobExtension(blob) {
  const mime = (blob.type || '').toLowerCase();
  if (mime.includes('mp4'))  return '.mp4';
  if (mime.includes('ogg'))  return '.ogg';
  if (mime.includes('webm')) return '.webm';
  return '.webm';
}

/* ==========================================================================
   UI UPDATE
   ========================================================================== */

function _updateMicUI() {
  const btn = document.getElementById('mic-btn');
  const label = btn ? btn.querySelector('.mic-btn-label') : null;
  const timer = document.getElementById('mic-timer');

  if (!btn) return;

  btn.classList.remove('recording', 'transcribing', 'disabled');

  switch (_micState) {
    case 'recording':
      btn.classList.add('recording');
      btn.setAttribute('aria-label', 'Stop recording');
      btn.setAttribute('title', 'Stop recording');
      if (label) label.textContent = '';
      if (timer) timer.hidden = false;
      break;

    case 'transcribing':
      btn.classList.add('transcribing');
      btn.setAttribute('aria-label', 'Transcribing…');
      btn.setAttribute('title', 'Transcribing…');
      if (label) label.textContent = '';
      if (timer) timer.hidden = true;
      break;

    case 'disabled':
      btn.classList.add('disabled');
      btn.setAttribute('aria-label', 'Voice input unavailable');
      btn.setAttribute('title', 'Voice input unavailable');
      if (label) label.textContent = '';
      if (timer) timer.hidden = true;
      break;

    case 'idle':
    default:
      btn.setAttribute('aria-label', 'Start voice input');
      btn.setAttribute('title', 'Voice input');
      if (label) label.textContent = '';
      if (timer) timer.hidden = true;
      break;
  }
}

function _updateTimer() {
  const timer = document.getElementById('mic-timer');
  if (!timer) return;
  _timerSeconds += 1;
  const m = Math.floor(_timerSeconds / 60);
  const s = _timerSeconds % 60;
  timer.textContent = m + ':' + (s < 10 ? '0' : '') + s;
}

/* ==========================================================================
   ERROR DISPLAY (D-04 — textContent only, never innerHTML of server data)
   ========================================================================== */

const _ERROR_MESSAGES = {
  permission_denied:  'Microphone permission denied.',
  no_speech:          'No speech detected — try again.',
  garbled:            'Couldn\'t make that out — try again.',
  not_installed:      'Voice not installed — run `l44 voice setup`.',
  unsupported:        'Voice input requires a secure HTTPS connection or a browser that supports MediaRecorder.',
  busy:               'Transcription busy — please wait and try again.',
  network:            'Network error reaching the server.',
};

function _showVoiceError(kind, detail) {
  const el = document.getElementById('voice-error');
  if (!el) return;
  const msg = _ERROR_MESSAGES[kind] || detail || 'Voice error — please try again.';
  el.textContent = msg;    // textContent only — never innerHTML (T-10-17)
  el.hidden = false;
}

function _clearVoiceError() {
  const el = document.getElementById('voice-error');
  if (!el) return;
  el.textContent = '';
  el.hidden = true;
}

/* ==========================================================================
   RECORDING LIFECYCLE
   ========================================================================== */

async function startRecording() {
  _clearVoiceError();
  _chunks = [];
  _timerSeconds = 0;

  const mimeType = getBestMimeType();

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      _showVoiceError('permission_denied');
    } else {
      _showVoiceError('garbled', err.message);
    }
    _micState = 'idle';
    _updateMicUI();
    return;
  }

  // GAP-5: wrap recorder construction + start in try/catch so a thrown error
  // (unsupported options, InvalidStateError, platform failure) never leaks the
  // getUserMedia stream or leaves UI state/timers half-started.
  try {
    const options = mimeType ? { mimeType } : {};
    _mediaRecorder = new MediaRecorder(stream, options);

    _mediaRecorder.addEventListener('dataavailable', function(e) {
      if (e.data && e.data.size > 0) _chunks.push(e.data);
    });

    _mediaRecorder.addEventListener('stop', async function() {
      // Stop all tracks to release mic indicator
      stream.getTracks().forEach(function(t) { t.stop(); });

      const blob = new Blob(_chunks, { type: mimeType || 'audio/webm' });
      _chunks = [];
      await _postTranscribe(blob);
    });

    _mediaRecorder.start(250); // collect data every 250ms
  } catch (err) {
    // Recorder construction or start threw — release the mic stream immediately.
    stream.getTracks().forEach(function(t) { t.stop(); });
    // Clear timers defensively (start may have partially run before the throw).
    if (_maxTimer) { clearTimeout(_maxTimer); _maxTimer = null; }
    if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
    // Reset UI to idle so the owner can retry.
    _micState = 'idle';
    _updateMicUI();
    // Hide any stale timer display.
    const timerEl = document.getElementById('mic-timer');
    if (timerEl) timerEl.hidden = true;
    // Show an inline error — reuse 'garbled' (matches getUserMedia non-permission branch).
    _showVoiceError('garbled', err && err.message);
    return;
  }

  _micState = 'recording';
  _updateMicUI();

  // Elapsed timer (D-03)
  const timer = document.getElementById('mic-timer');
  if (timer) {
    timer.textContent = '0:00';
    timer.hidden = false;
  }
  _timerInterval = setInterval(_updateTimer, 1000);

  // 30s safety cap (D-01)
  _maxTimer = setTimeout(function() { stopRecording(); }, 30000);
}

function stopRecording() {
  if (_maxTimer) { clearTimeout(_maxTimer); _maxTimer = null; }
  if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }

  if (_mediaRecorder && _mediaRecorder.state !== 'inactive') {
    _mediaRecorder.stop();
  }
  _micState = 'transcribing';
  _updateMicUI();
}

function toggleMic() {
  if (_micState === 'disabled') {
    // Click on disabled mic — show the relevant hint (already showing)
    return;
  }
  if (_micState === 'recording') {
    stopRecording();
  } else if (_micState === 'idle') {
    if (!_voiceInstalled) {
      // D-05: not installed — show install hint, do NOT attempt recording
      _showVoiceError('not_installed');
      return;
    }
    startRecording();
  }
  // transcribing state: ignore taps
}

/* ==========================================================================
   POST /transcribe (D-03 / D-04 mapping)
   ========================================================================== */

async function _postTranscribe(blob) {
  _micState = 'transcribing';
  _updateMicUI();

  const form = new FormData();
  form.append('file', blob, 'audio' + _blobExtension(blob));

  let resp;
  try {
    resp = await fetch('/transcribe', { method: 'POST', body: form });
  } catch (netErr) {
    _showVoiceError('network');
    _micState = 'idle';
    _updateMicUI();
    return;
  }

  // 429 — busy, non-blocking (D-05 / T-10-20)
  if (resp.status === 429) {
    _showVoiceError('busy');
    _micState = 'idle';
    _updateMicUI();
    return;
  }

  // 503 — not installed (D-04 error 4)
  if (resp.status === 503) {
    _showVoiceError('not_installed');
    _micState = 'idle';
    _updateMicUI();
    return;
  }

  let data;
  try {
    data = await resp.json();
  } catch (_) {
    _showVoiceError('garbled');
    _micState = 'idle';
    _updateMicUI();
    return;
  }

  _micState = 'idle';
  _updateMicUI();

  // D-04 soft error mapping
  if (data.error === 'no_speech') {
    _showVoiceError('no_speech');                                // D-04 error 2
    return;
  }
  if (data.error === 'garbled' || data.error === 'timeout') {
    _showVoiceError('garbled');                                  // D-04 error 3
    return;
  }
  if (data.error) {
    // any other server error string
    _showVoiceError('garbled', data.error);
    return;
  }

  // WR-04: trust the server's no_speech mapping — do NOT re-gate on an
  // English-only [^a-z] length<3 filter, which silently dropped legitimately
  // short or non-Latin transcripts ("go", part number "22-41016", etc.).
  // Reject only genuinely empty/whitespace-only transcripts. No outbound calls.
  const text = (data.text || '').trim();
  if (!text) {
    _showVoiceError('garbled');
    return;
  }

  // D-02: populate #question for review-before-Ask — never auto-submit
  const textarea = document.getElementById('question');
  if (textarea) {
    textarea.value = text;                                      // textContent would be wrong here (textarea.value)
    textarea.focus();
  }
  _clearVoiceError();
}

/* ==========================================================================
   INITIALISATION — called at DOMContentLoaded
   Sequence: capability guard → secure-context guard → /api/voice-status gate
   ========================================================================== */

async function _initVoice() {
  const btn = document.getElementById('mic-btn');
  if (!btn) return;

  // Capability + secure-context guard (review concern #8 / D-12)
  const supported = (
    window.isSecureContext &&
    typeof navigator !== 'undefined' &&
    navigator.mediaDevices &&
    typeof navigator.mediaDevices.getUserMedia === 'function' &&
    typeof MediaRecorder !== 'undefined'
  );

  if (!supported) {
    _micState = 'disabled';
    _updateMicUI();
    _showVoiceError('unsupported');
    // Do NOT call getUserMedia — typed box stays usable
    return;
  }

  // MANDATORY D-05 gating: fetch /api/voice-status
  try {
    const resp = await fetch('/api/voice-status');
    const data = await resp.json();
    _voiceInstalled = data.installed === true;
  } catch (_) {
    // Fetch failed — fail safe to not-installed state (D-05)
    _voiceInstalled = false;
  }

  if (!_voiceInstalled) {
    // Show persistent install hint on the mic (not a hard disabled state)
    _showVoiceError('not_installed');
    // mic state stays idle so click → toggleMic → not_installed branch (no getUserMedia)
  }

  // Wire click handler
  btn.addEventListener('click', function() { toggleMic(); });
}

/* ==========================================================================
   ENTRY POINT
   ========================================================================== */

document.addEventListener('DOMContentLoaded', function() {
  _initVoice();
});
