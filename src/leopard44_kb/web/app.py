"""FastAPI application factory for the Leopard 44 KB local web UI.

Exposes create_app() which returns the configured FastAPI instance.
The query endpoint re-expresses ask_cmd verbatim as SSE:
  - D-05: below_floor check BEFORE source-emission loop (refusal = zero sources)
  - D-04: source events emitted before token events on a normal answer
  - WR-02: DB connection closed immediately after retrieve(), before LLM streaming
  - T-05-04: layer validated against LAYERS / "all" before reaching retrieve()
  - T-05-05: top_k >= 1 enforced by Pydantic Field(ge=1)
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Iterable
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, field_validator

from leopard44_kb import LAYERS
from leopard44_kb.sources import list_sources_for_layer

# ---------------------------------------------------------------------------
# Package-relative paths (no runtime mkdir — Plan 04 commits the tree)
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent

# Module-relative repo root and schematics directory (WR-02: immune to cwd drift).
# app.py lives at src/leopard44_kb/web/app.py, so:
#   parents[0] = src/leopard44_kb/web/
#   parents[1] = src/leopard44_kb/
#   parents[2] = src/
#   parents[3] = repo root
# Matches the verified idiom in schema.py (parents[2] from src/leopard44_kb/schema.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMATICS_DIR = _REPO_ROOT / "data" / "schematics"

# ---------------------------------------------------------------------------
# D-04: Application-level LLM generation concurrency cap.
# BoundedSemaphore(2) limits in-flight LLM generations to 2.
# threading.BoundedSemaphore is correct here — query_endpoint is a sync def
# running in anyio's threadpool, NOT in an async context.
# Cross-plan invariant: uvicorn MUST run with --workers 1 (07-02 unit + 07-03
# bootstrap) so this module-level semaphore is shared across all requests.
# Do NOT attach this to app.state — one symbol, pinned contract.
# ---------------------------------------------------------------------------

_llm_semaphore = threading.BoundedSemaphore(2)  # D-04: max 2 in-flight LLM generations

# ---------------------------------------------------------------------------
# Voice STT helpers (D-06: faster-whisper NEVER imported here — subprocess only)
# ---------------------------------------------------------------------------

# Path to the STT worker script (run ONLY via .venv-stt/bin/python, never imported)
STT_SCRIPT = _REPO_ROOT / "src" / "leopard44_kb" / "stt_worker.py"

# Single-slot STT concurrency guard (T-10-20 / D-10):
# asyncio.Lock so .locked() provides the exact non-blocking check needed.
_stt_lock = asyncio.Lock()


def _voice_installed() -> bool:
    """Return True only when BOTH conditions hold (D-05 honest gate — GAP-3):

    (a) The .venv-stt virtualenv python exists, AND
    (b) The model marker (data/voice-model-path.txt, or L44_VOICE_MODEL_MARKER
        if set) is readable, non-empty, and points to an existing directory.

    Mirrors the marker-resolution predicate in stt_worker.py without importing it
    (the worker is .venv-stt-bound and must not be imported in the app process).
    Returns False on any OSError / missing path — never raises.
    """
    import os

    if not (_REPO_ROOT / ".venv-stt" / "bin" / "python").exists():
        return False

    try:
        marker_override = os.environ.get("L44_VOICE_MODEL_MARKER")
        marker_path = (
            Path(marker_override) if marker_override else _REPO_ROOT / "data" / "voice-model-path.txt"
        )
        recorded = marker_path.read_text().strip()
        if not recorded:
            return False
        return Path(recorded).exists()
    except OSError:
        return False


def build_stt_prompt(conn) -> str:  # type: ignore[type-arg]
    """Build a marine vocabulary initial_prompt from the corpus DB.

    Sources:
      - items.brand, items.model_number (distinct, non-empty)
      - maintenance_log.parts JSON arrays (distinct tokens)

    Schema-tolerant: each query is wrapped in a narrow try/except so a missing
    table or column (fresh/older DB) yields an empty list rather than raising.
    Deduped preserving first-seen order, capped at 60 terms to avoid whisper
    prompt truncation (D-07).

    Returns "Marine vocabulary: <terms>." or "Marine vocabulary." on empty corpus.
    """
    vocab: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        t = term.strip()
        if t and t not in seen:
            seen.add(t)
            vocab.append(t)

    # items.brand
    try:
        for row in conn.execute("SELECT DISTINCT brand FROM items WHERE brand IS NOT NULL AND brand != ''"):
            _add(row[0])
    except Exception:
        pass

    # items.model_number
    try:
        for row in conn.execute("SELECT DISTINCT model_number FROM items WHERE model_number IS NOT NULL AND model_number != ''"):
            _add(row[0])
    except Exception:
        pass

    # maintenance_log.parts (JSON array per row)
    try:
        for row in conn.execute("SELECT parts FROM maintenance_log WHERE parts IS NOT NULL AND parts != ''"):
            try:
                parts = json.loads(row[0])
                if isinstance(parts, list):
                    for p in parts:
                        if isinstance(p, str):
                            _add(p)
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception:
        pass

    # Cap at 60 terms (D-07: avoid whisper prompt truncation)
    vocab = vocab[:60]

    if not vocab:
        return "Marine vocabulary."
    return "Marine vocabulary: " + ", ".join(vocab) + "."


def _stt_subprocess(audio_path: str) -> dict:  # type: ignore[type-arg]
    """Shell out to .venv-stt/bin/python stt_worker.py with the audio path.

    Builds the marine initial_prompt from the DB, passes it on stdin, and
    parses the JSON stdout from stt_worker.

    Returns {"text": str} on success, {"error": str} on all failure modes.
    NEVER imports faster_whisper in this process (D-06 isolation).

    Catches subprocess.TimeoutExpired (60s) → {"error": "timeout"} (T-10-14).
    """
    import os
    import subprocess

    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    # Build marine prompt (schema-tolerant — empty prompt on a fresh DB is fine)
    conn = open_db()
    try:
        apply_migrations(conn)
        prompt = build_stt_prompt(conn)
    finally:
        conn.close()

    # WR-05: pin the strict-offline flags at the call site (defense-in-depth).
    # The worker's module-level os.environ.setdefault() does NOT override an
    # inherited value, so if the SERVER process is ever started with
    # HF_HUB_OFFLINE=0 / TRANSFORMERS_OFFLINE=0, that would propagate and defeat
    # VOICE-03's "never fetch at sea". Forcing them on here makes the worker
    # offline regardless of the inherited environment.
    child_env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    venv_python = str(_REPO_ROOT / ".venv-stt" / "bin" / "python")
    try:
        result = subprocess.run(
            [venv_python, str(STT_SCRIPT), audio_path],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    if result.returncode != 0:
        # WR-02: a non-zero exit means an UNEXPECTED worker crash (the worker is
        # designed to always exit 0 and emit JSON even on soft/hard errors), so
        # stderr here may carry a full Python traceback with absolute filesystem
        # paths. Do NOT surface raw stderr to the LAN-reachable client (V7).
        # Log it server-side only and return a generic soft error.
        _logger.error(
            "STT worker exited %s; stderr:\n%s",
            result.returncode,
            result.stderr.strip(),
        )
        return {"error": "garbled"}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        # GAP-4 / T-10-12 / V7: log the raw stdout server-side only (mirrors WR-02
        # stderr fix). Never surface raw worker output to the LAN client.
        _logger.warning("STT worker returned non-JSON stdout:\n%s", result.stdout.strip())
        return {"error": "garbled"}


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """POST /query request body."""

    question: str = Field(min_length=1)
    layer: str = "all"
    top_k: int = Field(default=5, ge=1)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


# ---------------------------------------------------------------------------
# POST /annotate/{zone_id} request body
# ---------------------------------------------------------------------------


class AnnotationBody(BaseModel):
    """Body for POST /annotate/{zone_id}."""

    schematic_image: str = Field(min_length=1)
    geometry: list[list[float]]  # [[x_norm, y_norm], ...] normalized 0-1

    @field_validator("geometry")
    @classmethod
    def validate_geometry(cls, v: list[list[float]]) -> list[list[float]]:
        """V5 input validation: >= 3 pairs, each [float, float], each coord 0.0-1.0."""
        if len(v) < 3:
            raise ValueError("geometry must have at least 3 vertex pairs")
        for pair in v:
            if len(pair) != 2:
                raise ValueError(
                    f"each geometry entry must be a [float, float] pair; got length {len(pair)}"
                )
            for coord in pair:
                if not (0.0 <= coord <= 1.0):
                    raise ValueError(
                        f"geometry coordinates must be in 0.0-1.0 range; got {coord}"
                    )
        return v


# ---------------------------------------------------------------------------
# Shared reject-not-sanitize helper for schematic image filenames.
#
# Security contract (D-13, review concern 2):
#   Returns None (REJECT) when the filename is unsafe — NEVER normalizes a
#   traversal into a servable basename.  Callers map None -> 404 (GET) or
#   422/400 (POST).
#
# Rejection conditions (all return None):
#   - empty string
#   - filename != Path(filename).name  (contains /, \\, or .. segments)
#   - does not end in .png (case-insensitive)
#   - resolved path escapes data/schematics/ (defense-in-depth)
#   - file does not exist in data/schematics/
#
# Uses repo_root() from leopard44_kb.paths.  For web routes the cwd is the repo
# root (uvicorn is invoked from there); for tests monkeypatch.chdir() sets it.
# ---------------------------------------------------------------------------


def _resolve_schematic_image(filename: str) -> Path | None:
    """Reject-not-sanitize guard for schematic image filenames.

    Returns the resolved Path inside data/schematics/ when *filename* is a
    safe, existing bare .png name.  Returns None for all unsafe inputs.

    Uses the module-relative _SCHEMATICS_DIR (cwd-independent, WR-02).
    """
    if not filename:
        return None
    # REJECT any name with directory components (/, \\, ..) — do NOT sanitize
    if filename != Path(filename).name:
        return None
    # REJECT non-.png names
    if not filename.lower().endswith(".png"):
        return None
    schematics_dir = _SCHEMATICS_DIR.resolve()
    img_path = (schematics_dir / filename).resolve()
    # Defense-in-depth: verify the resolved path is still inside schematics_dir
    try:
        img_path.relative_to(schematics_dir)
    except ValueError:
        return None
    if not img_path.exists():
        return None
    return img_path


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Return the configured FastAPI application.

    Routes registered:
      GET  /               — index.html (Ask view)
      GET  /explore        — explore.html (Browse view)
      GET  /api/sources    — JSON source list (?layer=all|shared|vessel|community)
      POST /query          — SSE query endpoint (ask_cmd pipeline as SSE)
      /static              — StaticFiles mount (WEB_DIR/static)

    Swagger/ReDoc/openapi.json are disabled (docs_url=None etc.) to minimise
    the local HTTP surface (review LOW).
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    # Static assets — Plan 04 commits the real files; the directory exists as a
    # committed tree artifact so StaticFiles(check_dir=True) is satisfied at startup.
    app.mount(
        "/static",
        StaticFiles(directory=WEB_DIR / "static"),
        name="static",
    )

    templates = Jinja2Templates(directory=WEB_DIR / "templates")

    # ------------------------------------------------------------------
    # GET /
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={},
        )

    # ------------------------------------------------------------------
    # GET /explore
    # ------------------------------------------------------------------

    @app.get("/explore", response_class=HTMLResponse)
    def explore(request: Request) -> HTMLResponse:
        # Build grouped source data in display order: vessel, shared, community.
        # list() materialises each generator immediately — the connection inside
        # list_sources_for_layer must not stay open after we yield from it
        # (RESEARCH Pitfall 4 / T-05-06).
        display_order = ("vessel", "shared", "community")
        groups = [
            (layer, list(list_sources_for_layer(layer)))
            for layer in display_order
        ]
        # Keep only non-empty groups (UI-SPEC § 4.6)
        groups = [(layer, rows) for layer, rows in groups if rows]
        return templates.TemplateResponse(
            request=request,
            name="explore.html",
            context={"groups": groups},
        )

    # ------------------------------------------------------------------
    # GET /annotate — zone annotation list
    # ------------------------------------------------------------------

    @app.get("/annotate", response_class=HTMLResponse)
    def annotate_list(request: Request) -> HTMLResponse:
        """List all zones, unannotated-first, with annotation progress count."""
        from leopard44_kb.store import open_db
        from leopard44_kb.schema import apply_migrations

        conn = open_db()
        try:
            apply_migrations(conn)  # CR-01: migrate before any DB op
            zones = conn.execute(
                "SELECT id, name, label, vertical_desc, schematic_image, geometry "
                "FROM zones "
                "ORDER BY CASE WHEN schematic_image IS NOT NULL AND geometry IS NOT NULL "
                "THEN 1 ELSE 0 END, name"
            ).fetchall()
            # annotated = schematic_image IS NOT NULL AND geometry IS NOT NULL
            annotated = sum(
                1 for z in zones
                if z["schematic_image"] is not None and z["geometry"] is not None
            )
        finally:
            conn.close()
        return templates.TemplateResponse(
            request=request,
            name="annotate_list.html",
            context={"zones": [dict(z) for z in zones], "annotated": annotated, "total": len(zones)},
        )

    # ------------------------------------------------------------------
    # GET /annotate/{zone_id} — polygon editor for a single zone
    # ------------------------------------------------------------------

    @app.get("/annotate/{zone_id}", response_class=HTMLResponse)
    def annotate_editor(zone_id: int, request: Request) -> HTMLResponse:
        """Render the polygon editor for an existing zone (404 if not found)."""
        from leopard44_kb.store import open_db
        from leopard44_kb.schema import apply_migrations

        conn = open_db()
        try:
            apply_migrations(conn)  # CR-01: migrate before any DB op
            row = conn.execute(
                "SELECT id, name, label, vertical_desc, schematic_image, geometry "
                "FROM zones WHERE id = ?",
                (zone_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            raise HTTPException(status_code=404, detail="Zone not found")

        # Deserialize the stored geometry (TEXT/JSON) into a real list BEFORE the
        # template's `| tojson` runs — otherwise it double-encodes to a JS string
        # and loadExisting() throws on `.map`, so a re-opened zone never reloads
        # its saved polygon (Phase 9 visual-UAT round-trip gap).
        zone = dict(row)
        if zone.get("geometry"):
            try:
                zone["geometry"] = json.loads(zone["geometry"])
            except (ValueError, TypeError):
                zone["geometry"] = None

        # List available schematic PNGs from data/schematics/ (D-12: empty list is OK)
        # Use module-relative _SCHEMATICS_DIR — cwd-independent (WR-02).
        if _SCHEMATICS_DIR.exists():
            schematic_files = sorted(
                f.name for f in _SCHEMATICS_DIR.iterdir() if f.suffix.lower() == ".png"
            )
        else:
            schematic_files = []

        return templates.TemplateResponse(
            request=request,
            name="annotate_editor.html",
            context={"zone": zone, "schematic_files": schematic_files},
        )

    # ------------------------------------------------------------------
    # GET /schematic-image/{filename:path} — serve vessel-layer schematic PNGs
    # D-13: NEVER from static/; REJECT (404) any unsafe name via shared helper.
    # {filename:path} declared so slash-containing input reaches the helper.
    # ------------------------------------------------------------------

    @app.get("/schematic-image/{filename:path}")
    def schematic_image(filename: str) -> FileResponse:
        """Serve a PNG from data/schematics/ with reject-not-sanitize path guard."""
        img = _resolve_schematic_image(filename)
        if img is None:
            raise HTTPException(status_code=404, detail="Schematic image not found")
        return FileResponse(str(img), media_type="image/png")

    # ------------------------------------------------------------------
    # POST /annotate/{zone_id} — save normalized geometry + validated schematic_image
    # ------------------------------------------------------------------

    @app.post("/annotate/{zone_id}")
    def save_polygon(zone_id: int, body: AnnotationBody) -> JSONResponse:
        """Persist normalized polygon geometry and a validated schematic_image to the zone row.

        Security (review concern 5, D-13):
          - schematic_image validated via _resolve_schematic_image (reject-not-sanitize);
            rejects URLs, traversal names, non-.png, and non-existent files.
          - geometry validated by AnnotationBody.validate_geometry (>= 3 pairs, 0.0-1.0).
          - CR-01: apply_migrations before any DB access.
        """
        from leopard44_kb.store import open_db
        from leopard44_kb.schema import apply_migrations

        # Validate schematic_image via the shared reject-not-sanitize helper
        if _resolve_schematic_image(body.schematic_image) is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "schematic_image must be a bare .png filename that exists in "
                    "data/schematics/. URLs, traversal names, non-.png, and "
                    "non-existent files are rejected."
                ),
            )

        conn = open_db()
        try:
            apply_migrations(conn)  # CR-01: migrate before any DB op
            row = conn.execute(
                "SELECT id FROM zones WHERE id = ?", (zone_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Zone not found")
            with conn:
                conn.execute(
                    "UPDATE zones SET geometry = ?, schematic_image = ? WHERE id = ?",
                    (json.dumps(body.geometry), body.schematic_image, zone_id),
                )
        finally:
            conn.close()

        return JSONResponse({"ok": True})

    # ------------------------------------------------------------------
    # GET /api/sources
    # ------------------------------------------------------------------

    @app.get("/api/sources")
    def api_sources(layer: str = "all") -> list[dict]:  # type: ignore[type-arg]
        """Return JSON list of sources for the given layer.

        layer=all → concatenate all LAYERS (materialized); layer in LAYERS →
        that layer only; unknown → []. Always materialises with list comprehension
        (RESEARCH Pitfall 4).
        """
        if layer == "all":
            return [dict(r) for l in LAYERS for r in list_sources_for_layer(l)]
        if layer in LAYERS:
            return [dict(r) for r in list_sources_for_layer(layer)]
        return []

    # ------------------------------------------------------------------
    # GET /api/voice-status — D-05: mic button gates on this (D-05)
    # ------------------------------------------------------------------

    @app.get("/api/voice-status")
    def api_voice_status() -> JSONResponse:
        """Return {"installed": bool} — lightweight path-existence check (D-05).

        The mic button in the UI gates on this to show the 'install voice' hint
        until 'l44 voice setup' has run.
        """
        return JSONResponse({"installed": _voice_installed()})

    # ------------------------------------------------------------------
    # POST /transcribe — audio blob → offline transcript via isolated STT venv
    #
    # Security (STRIDE T-10-09/10/11/12/14/20):
    #   V5: size cap (>5MB) rejected before writing temp file or touching lock
    #   V5: Content-Type → fixed extension from allowlist (no user filename)
    #   T-10-20: non-blocking single-slot lock — 429 immediately when held
    #   T-10-14: subprocess timeout → {"error":"timeout"} (not a 500)
    #   V7: no raw stderr/traceback leaked to client
    # ------------------------------------------------------------------

    @app.post("/transcribe")
    async def transcribe_endpoint(file: UploadFile = File(...)) -> JSONResponse:
        """Accept a MediaRecorder audio blob (webm or mp4), transcribe via .venv-stt.

        Returns {"text": str} on success.
        Returns {"error": "no_speech"|"garbled"|"timeout"} for soft STT errors.
        Returns HTTP 503 when STT not installed or worker reports not_installed.
        Returns HTTP 413 when blob exceeds 5MB.
        Returns HTTP 429 immediately when another transcription is in progress.
        """
        import pathlib
        import tempfile

        # --- D-04 error 4: STT not installed (venv absent) ---
        if not _voice_installed():
            raise HTTPException(
                status_code=503,
                detail="STT not installed — run `l44 voice setup`",
            )

        # --- V5: blob size guard — read at most 5MB+1 bytes, reject if exceeded ---
        # Must happen BEFORE touching the lock (prevents size-based DoS bypassing lock).
        content = await file.read(5 * 1024 * 1024 + 1)
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio blob too large (max 5MB)")

        # --- T-10-20: non-blocking concurrency guard (WR-01) ---
        # Atomic acquire-or-429 with the "no queue, no await-while-held" contract:
        #   1. .locked() check — returns 429 IMMEDIATELY when held. This also keeps
        #      the _HeldLock stub path in test_transcribe_busy_429 from ever
        #      calling acquire (the stub asserts acquire() is never reached).
        #   2. `await _stt_lock.acquire()` directly. On a FREE asyncio.Lock,
        #      acquire() completes WITHOUT suspending, and there is NO await
        #      suspension point between the .locked() check and this acquire — so
        #      no other coroutine can interleave and the acquire is guaranteed to
        #      take the slot rather than queue. (asyncio.wait_for(acquire(),
        #      timeout=0) is NOT usable here: on this Python it cancels the
        #      acquire before it resolves and spuriously TimeoutErrors even on a
        #      free lock.) The .locked() race-loser case cannot arise within a
        #      single event loop because acquire-after-free-check is atomic.
        # Only AFTER we own the slot do we write the temp file, so a 429'd request
        # leaves nothing on disk. The slot is released in the finally below.
        if _stt_lock.locked():
            raise HTTPException(status_code=429, detail="transcription busy")
        await _stt_lock.acquire()

        try:
            # --- Extension from Content-Type allowlist (no user-controlled filename) ---
            ct = (file.content_type or "audio/webm").lower()
            ext_map = {
                "audio/webm": ".webm",
                "audio/mp4": ".mp4",
                "audio/ogg": ".ogg",
                "audio/mpeg": ".mp3",
            }
            ext = next((v for k, v in ext_map.items() if ct.startswith(k)), ".webm")

            # --- Write to temp file (server-controlled path — T-10-10) ---
            # Only reached once the single slot is owned, so a 429'd request never
            # writes a temp file (WR-01).
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                    f.write(content)
                    tmp_path = f.name
                # GAP-2 / T-10-20: offload the blocking subprocess.run+SQLite to a
                # thread pool so the uvicorn event loop stays responsive during the
                # ~60s transcription (immediate-429 and /query+/api/voice-status
                # must not hang while a transcription is in progress).
                result = await asyncio.to_thread(_stt_subprocess, tmp_path)
            finally:
                if tmp_path is not None:
                    pathlib.Path(tmp_path).unlink(missing_ok=True)
        finally:
            _stt_lock.release()

        # --- D-04 error mapping ---
        if "error" in result:
            err = result["error"]
            if err == "not_installed":
                # Worker-reported not_installed = missing model marker (VOICE-03)
                # This is a setup problem, not a bad-audio problem → 503 (T-10-21)
                raise HTTPException(
                    status_code=503,
                    detail="STT not installed — run `l44 voice setup`",
                )
            # Soft errors (no_speech, garbled, timeout) → 200 with error key
            return JSONResponse({"error": err})

        return JSONResponse({"text": result.get("text", "")})

    # ------------------------------------------------------------------
    # POST /query — SSE endpoint mirroring ask_cmd verbatim
    # ------------------------------------------------------------------

    @app.post("/query", response_class=EventSourceResponse)
    def query_endpoint(request: QueryRequest) -> Iterable[ServerSentEvent]:
        """Stream the ask_cmd pipeline as Server-Sent Events.

        Event ordering (STRICTLY preserved):
          1. layer validation error → error + done + return
          2. retrieve() — conn opened + closed in try/finally (WR-02)
          3. below_floor check → refusal + done + return  [BEFORE source loop — D-05]
          4. source events × len(chunks)                 [BEFORE any token — D-04]
          5. token events from stream_generate
          6. RuntimeError from Ollama → error + done + return
          7. done (with bad_citations list)

        Uses a SYNC def (not async) so httpx blocking stream runs in anyio's
        threadpool (RESEARCH Pitfall 1).
        """
        # ------------------------------------------------------------------
        # D-04: Non-blocking capacity guard — FIRST statement in the generator
        # body, before any lazy imports or side-effects.
        #
        # CRITICAL (verified HIGH finding): the acquire/release MUST live
        # inside the generator frame. The try/finally below wraps the entire
        # yielding body so the semaphore is held until EventSourceResponse
        # fully consumes the generator — NOT just until generator construction.
        # threading.BoundedSemaphore is correct here (sync def in anyio's
        # threadpool); do NOT use asyncio.Semaphore.
        # ------------------------------------------------------------------
        if not _llm_semaphore.acquire(blocking=False):
            yield ServerSentEvent(
                event="error",
                raw_data="Server is busy processing another query — please try again in a moment.",
            )
            yield ServerSentEvent(event="done", raw_data=json.dumps({"bad_citations": []}))
            return

        try:
            # Lazy imports (mirrors ask_cmd pattern for Phase-N modules).
            from leopard44_kb.retrieve import retrieve
            from leopard44_kb.answer import (
                select_generation_model,
                select_num_predict,
                build_user_message,
                SYSTEM_PROMPT,
                REFUSAL_MESSAGE,
                stream_generate,
                validate_citations,
            )
            from leopard44_kb.store import open_db
            from leopard44_kb.schema import apply_migrations

            # ------------------------------------------------------------------
            # Step 1: Layer resolution (T-03-10 / T-05-04)
            # ------------------------------------------------------------------
            if request.layer == "all":
                layers: list[str] = []
            elif request.layer in LAYERS:
                layers = [request.layer]
            else:
                yield ServerSentEvent(
                    event="error",
                    raw_data="layer must be shared|vessel|community|all",
                )
                yield ServerSentEvent(event="done", raw_data=json.dumps({"bad_citations": []}))
                return

            # ------------------------------------------------------------------
            # Step 2: Retrieve, then CLOSE the connection immediately (WR-02 / T-05-06).
            # The DB handle is NOT needed during token streaming.
            #
            # Exceptions during open_db / apply_migrations / retrieve (e.g. Ollama
            # unreachable during embedding, or SQLite lock under load) are caught here
            # and surfaced as an explicit SSE error+done pair instead of silently
            # closing the stream (CR-01 / Rule-1 fix: open_db() included in the
            # try so a lock/file error also yields error+done).
            # The finally block guarantees conn.close() on every path.
            # ------------------------------------------------------------------
            retrieve_error: Exception | None = None
            conn = None
            try:
                conn = open_db()
                apply_migrations(conn)
                chunks, below_floor = retrieve(conn, request.question, layers, n=request.top_k)
            except RuntimeError as exc:
                retrieve_error = exc
            except Exception as exc:
                retrieve_error = exc
            finally:
                if conn is not None:
                    conn.close()

            if retrieve_error is not None:
                exc = retrieve_error
                if isinstance(exc, RuntimeError):
                    msg = str(exc)
                else:
                    msg = f"Internal error: {exc}"
                yield ServerSentEvent(event="error", raw_data=msg)
                yield ServerSentEvent(event="done", raw_data=json.dumps({"bad_citations": []}))
                return

            # ------------------------------------------------------------------
            # Step 3: REFUSAL CHECK — BEFORE source-emission loop (D-05 / T-05-17)
            # below_floor → refusal + done with ZERO source events.
            # ------------------------------------------------------------------
            if below_floor:
                yield ServerSentEvent(event="refusal", raw_data=REFUSAL_MESSAGE)
                yield ServerSentEvent(event="done", raw_data=json.dumps({"bad_citations": []}))
                return

            # ------------------------------------------------------------------
            # Step 4: Source events (two-stage reveal — D-04)
            # All source cards paint BEFORE the first token.
            # page_start/page_end: use .get() directly so 0 serialises as 0 (WR-04).
            # ------------------------------------------------------------------
            for i, chunk in enumerate(chunks, 1):
                yield ServerSentEvent(
                    event="source",
                    raw_data=json.dumps({
                        "n": i,
                        "layer": chunk["layer"],
                        "title": chunk.get("title") or chunk.get("path") or "unknown source",
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                        # Option A: the exact retrieved passage that grounds the answer,
                        # so the UI can reveal "show the passage used" per source card.
                        # content already carries its heading hierarchy (0d1f378).
                        "section_path": chunk.get("section_path") or "",
                        "content": chunk.get("content") or "",
                    }),
                )

            # ------------------------------------------------------------------
            # Step 5: Generate — mirrors ask_cmd exactly.
            # ------------------------------------------------------------------
            gen_model, tier_label = select_generation_model()
            system = SYSTEM_PROMPT.format(n_chunks=len(chunks))
            user_msg = build_user_message(request.question, chunks)
            num_predict = select_num_predict(tier_label, gen_model)

            full_parts: list[str] = []
            try:
                for token in stream_generate(gen_model, system, user_msg, num_predict=num_predict):
                    yield ServerSentEvent(event="token", raw_data=token)
                    full_parts.append(token)
            except RuntimeError as exc:
                yield ServerSentEvent(event="error", raw_data=str(exc))
                yield ServerSentEvent(event="done", raw_data=json.dumps({"bad_citations": []}))
                return

            # ------------------------------------------------------------------
            # Step 6: Post-stream citation validation + done
            # ------------------------------------------------------------------
            bad = validate_citations("".join(full_parts), len(chunks))

            # ------------------------------------------------------------------
            # Step 7: zone_highlight events — AFTER last source, BEFORE done
            # (review concern 1 / WR-02 / VIS-01 / D-12)
            #
            # The original query conn was closed in the WR-02 finally block above.
            # Open a NEW short-lived connection for the zone JOIN so we never
            # reuse a closed handle (WR-02).  The retrieved `chunks` list is still
            # in scope in this generator closure.
            # ------------------------------------------------------------------
            zone_conn = None
            try:
                zone_conn = open_db()
                apply_migrations(zone_conn)

                # ------------------------------------------------------------------
                # Two-pass zone_highlight resolution with deviation-wins tie-break.
                #
                # PASS 1 — RESOLVE: walk chunks and accumulate a per-zone kind map.
                #   Inventory item chunks contribute kind="inventory" via setdefault
                #   so they are recorded only if the zone has not been seen yet.
                #   Deviation chunks contribute kind="deviation" unconditionally —
                #   this is the tie-break: if an inventory item and a deviation both
                #   resolve to the same zone, the deviation WINS (criterion 3: a
                #   deviation-bearing result must show BLUE even when an item shares
                #   the zone; a silently-amber deviation highlight would mis-attribute
                #   an owner modification as an inventory result).
                #
                # PASS 2 — EMIT: for each zone in first-seen insertion order, fetch
                #   the zones row and yield a zone_highlight SSE event with the
                #   resolved kind field ("deviation" | "inventory").
                # ------------------------------------------------------------------

                # zone_kind: zone_id -> "deviation" | "inventory"
                # Preserves insertion order (Python 3.7+ dict).
                zone_kind: dict[int, str] = {}

                for chunk in chunks:
                    raw_meta = chunk.get("metadata")
                    if raw_meta:
                        try:
                            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
                        except (json.JSONDecodeError, TypeError):
                            meta = {}
                    else:
                        meta = {}
                    if not isinstance(meta, dict):
                        continue

                    # --- inventory item path ---
                    item_id = meta.get("item_id")
                    if item_id is not None:
                        item_row = zone_conn.execute(
                            "SELECT current_zone_id FROM items WHERE id = ?", (item_id,)
                        ).fetchone()
                        if item_row and item_row["current_zone_id"]:
                            zone_id_val = item_row["current_zone_id"]
                            # setdefault: only record "inventory" if zone not already seen
                            zone_kind.setdefault(zone_id_val, "inventory")

                    # --- deviation path ---
                    deviation_id = meta.get("deviation_id")
                    if deviation_id is not None:
                        dev_row = zone_conn.execute(
                            "SELECT zone_id FROM deviations WHERE id = ?", (deviation_id,)
                        ).fetchone()
                        if dev_row and dev_row["zone_id"]:
                            zone_id_val = dev_row["zone_id"]
                            # Unconditional: deviation always wins, upgrading any
                            # prior "inventory" entry for the same zone.
                            zone_kind[zone_id_val] = "deviation"

                # PASS 2 — EMIT one event per resolved zone in first-seen order
                for zone_id_val, kind in zone_kind.items():
                    zone_row = zone_conn.execute(
                        "SELECT id, label, vertical_desc, schematic_image, geometry "
                        "FROM zones WHERE id = ?",
                        (zone_id_val,),
                    ).fetchone()
                    if not zone_row:
                        continue
                    zone_data = {
                        "zone_id": zone_row["id"],
                        "name": zone_row["label"],
                        "cue": zone_row["vertical_desc"],
                        "schematic_image": zone_row["schematic_image"],
                        "geometry": json.loads(zone_row["geometry"]) if zone_row["geometry"] else None,
                        "kind": kind,
                    }
                    yield ServerSentEvent(
                        event="zone_highlight",
                        raw_data=json.dumps(zone_data),
                    )
            finally:
                if zone_conn is not None:
                    zone_conn.close()

            yield ServerSentEvent(event="done", raw_data=json.dumps({"bad_citations": bad}))

        finally:
            _llm_semaphore.release()

    return app
