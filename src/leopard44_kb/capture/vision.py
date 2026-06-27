"""Vision identification module for Leopard 44 KB capture package.

CONNECTED surface — never imported by leopard44_kb.web (offline guarantee
enforced by tests/test_capture_import_boundary.py).

Local-first item identification using qwen2.5vl:7b via Ollama, with a
consent-gated cloud fallback to the Anthropic messages API.

OFFLINE BOUNDARY: This module imports only stdlib + httpx + the zone-list
reader. It must never import leopard44_kb.web or leopard44_kb.answer (enforced by
tests and the capture/__init__.py boundary docstring).

Contracts:
  - identify_item(image_path, zones, cloud=False) → dict
  - Local path: POST /api/generate with options.num_ctx=8192 (required for
    the 32-zone taxonomy; default 4096 overflows per Spike 003 finding #3)
  - Cloud path: fires ONLY when cloud=True (H3 consent gate); never
    auto-fires on low confidence
  - CLOUD_VISION_MODEL is a module constant pinned to a sonnet/opus-tier
    Anthropic vision model (NOT claude-3-haiku, per Spike 003 finding #2)
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.7  # below → low_confidence True (flagged for owner review)

# Pinned to a strong sonnet-tier Anthropic vision model (NOT haiku).
# Spike 003 finding #2: a cheap haiku fallback adds cost with no advantage
# over local qwen2.5vl on hard cases — use sonnet/opus only.
CLOUD_VISION_MODEL = "claude-sonnet-4-6"

# Guard at module load: never accidentally use the cheap haiku model.
assert CLOUD_VISION_MODEL != "claude-3-haiku-20240307", (
    "CLOUD_VISION_MODEL must not be haiku — use a sonnet/opus-tier model "
    "(spike 003: haiku adds cost with no advantage over local qwen2.5vl)"
)

LOCAL_MODEL = "qwen2.5vl:7b"

# Regex to strip markdown JSON fences: ```json ... ``` or ``` ... ```
# Tolerant (M2): handles single-line fenced JSON and an optional `json` tag, with
# or without surrounding newlines. Content between the fences is captured lazily.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """Return the first balanced top-level ``{...}`` JSON object in *text* (M2).

    Real model output on the cloud path (no ``format=json``) may wrap the object in
    markdown fences, prose, or both. Strategy:
      1. Strip markdown fences if present (tolerant — single-line or multi-line).
      2. Scan for the first ``{`` and return through its matching ``}``, tracking
         brace depth while ignoring braces inside JSON string literals (and their
         escapes) so a ``}`` inside a string value does not close the object early.
    If no balanced object is found, return the (fence-stripped) text unchanged so
    json.loads raises a clean error that _normalize_result converts to RuntimeError.
    """
    stripped = text.strip()

    # 1) Unwrap a markdown code fence if one is present anywhere in the text.
    m = _FENCE_RE.search(stripped)
    if m:
        stripped = m.group(1).strip()

    # 2) Walk to the first '{' and return through its matching '}'.
    start = stripped.find("{")
    if start == -1:
        return stripped

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]

    # Unbalanced — return what we have; json.loads will raise → RuntimeError.
    return stripped


# ---------------------------------------------------------------------------
# Internal HTTP wrapper — monkeypatched in tests
# ---------------------------------------------------------------------------

def _httpx_post(url: str, json: dict, timeout: float | None = None, **kw: Any) -> httpx.Response:
    """Thin wrapper around httpx.post so tests can monkeypatch this single symbol."""
    return httpx.post(url, json=json, timeout=timeout, **kw)


# Loopback host names/addresses for the B1 advisory. A host outside this set means
# the (sanitized) photo bytes are leaving this machine.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"})


def _warn_if_non_loopback_ollama(ollama_host: str) -> None:
    """Print a one-line, NON-BLOCKING stderr advisory if OLLAMA_HOST is off-box (B1).

    The local vision path always sends SANITIZED bytes (GPS already stripped), so
    this is a privacy/transparency notice, not a hard error: a LAN Ollama on another
    box is a legitimate setup. Parsing failures are swallowed — the advisory must
    never break a capture.
    """
    from urllib.parse import urlparse

    try:
        # urlparse needs a scheme to populate hostname; OLLAMA_HOST may be a bare
        # host:port (e.g. "192.0.2.5:11434"), so prepend a scheme when absent.
        candidate = ollama_host if "://" in ollama_host else f"http://{ollama_host}"
        host = urlparse(candidate).hostname
    except Exception:  # noqa: BLE001 — advisory must never raise
        return

    if not host:
        return

    if host.lower() in _LOOPBACK_HOSTS:
        return

    print(
        f"note: OLLAMA_HOST is not localhost; the photo will be sent to {host} "
        "(EXIF/GPS already stripped). Omit OLLAMA_HOST or point it at localhost to keep it on-box.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Prompt builder — mirrors spike 003 run_local.py build_prompt()
# ---------------------------------------------------------------------------

def build_prompt(zones: list[str | tuple[str, str]]) -> str:
    """Build the vessel-cataloguing prompt embedding the zone taxonomy.

    Args:
        zones: Either a list of zone name strings or a list of (name, description)
               tuples. When only names are provided, the prompt includes names only.

    Returns:
        The full prompt string to send to the vision model.
    """
    # Normalise to (name, description) tuples
    zone_tuples: list[tuple[str, str]] = []
    for z in zones:
        if isinstance(z, tuple):
            zone_tuples.append(z)
        else:
            zone_tuples.append((str(z), ""))

    zone_block = "\n".join(
        f"  - {name}: {desc}" if desc else f"  - {name}"
        for name, desc in zone_tuples
    )

    return (
        "You are cataloguing the spares, consumables, and equipment aboard a "
        "Leopard 44 sailing catamaran (your Leopard 44 sailing catamaran) from a single photo. "
        "Identify the item as specifically as the photo allows and propose where it "
        "should be stowed, choosing from this vessel's actual storage zones.\n\n"
        "Be honest: if the item is unlabelled or generic, say so and describe what "
        "it physically is (e.g. 'bronze raw-water pump impeller') rather than "
        "guessing a brand. If a field is not legible or not determinable, return "
        "null — do NOT hallucinate. If the frame contains multiple distinct items, "
        "identify the most prominent one in the main fields and list the rest in "
        "'other_items'.\n\n"
        "Set 'confidence' to your genuine certainty in the identification (0.0–1.0). "
        "Set 'legible' to true only if a printed brand/model label is readable in "
        "the photo.\n\n"
        "Vessel storage zones (choose suggested_zone from these names exactly):\n"
        f"{zone_block}\n\n"
        "Respond ONLY with a JSON object of this exact shape:\n"
        "{\n"
        '  "item": string,                     // best specific identification\n'
        '  "brand": string or null,\n'
        '  "model": string or null,\n'
        '  "category": string,                 // e.g. "raw-water pump", "adhesive", "safety"\n'
        '  "marine": boolean,\n'
        '  "legible": boolean,                 // was a printed label readable?\n'
        '  "key_properties": [string],         // size/specs/notable features actually visible\n'
        '  "other_items": [string],            // other distinct items in a multi-item frame\n'
        '  "suggested_zone": string,           // EXACTLY one of the zone names above\n'
        '  "zone_reasoning": string,           // one short sentence\n'
        '  "confidence": number                // 0.0–1.0\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Result normalisation / validation (M3)
# ---------------------------------------------------------------------------

def _coerce_scalar(value: Any) -> str | None:
    """Coerce a model field to ``str`` or ``None`` (M3) — never a dict/list.

    - None / empty / whitespace-only → None
    - str → trimmed str (or None if it trims to empty)
    - dict / list → None (the model returned a structured value where a scalar was
      expected; dropping it is safer than stringifying an object into a DB column)
    - other scalars (int/float/bool) → str()
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (dict, list, tuple, set)):
        return None
    return str(value)


def _coerce_str_list(value: Any) -> list[str]:
    """Coerce a model field to a ``list[str]`` (M3).

    - None → []
    - list/tuple → [str(x) for each non-empty element]
    - str → [str] (NOT iterated char-by-char) when non-empty, else []
    - dict / other → [] (structured/unsupported shape dropped rather than mangled)
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for x in value:
            if x is None:
                continue
            s = x.strip() if isinstance(x, str) else str(x)
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return []


def _normalize_result(
    raw_text: str,
    known_zone_names: set[str],
    source: str,
) -> dict:
    """Parse and validate a raw model response string into a normalized result dict.

    Validation rules (M3):
    - Strips markdown JSON fences (``` json ... ```) before json.loads
    - confidence missing / non-numeric / out of [0,1] → clamp to bound or 0.0 (→ low_confidence True)
    - suggested_zone not in known_zone_names → zone_id=None AND low_confidence True
    - empty/null item → item=None
    - Sets low_confidence = (confidence < CONFIDENCE_THRESHOLD OR validation downgraded it)
    - Sets source field to the provided source string ("local" or "cloud")
    - Adds zone_id: the matched zone name if valid, else None

    Args:
        raw_text: The raw string response from the model.
        known_zone_names: Set of valid zone names to validate suggested_zone against.
        source: Either "local" or "cloud".

    Returns:
        A normalized result dict suitable for the CLI to consume.

    Raises:
        RuntimeError: If the response cannot be parsed as JSON after fence stripping.
    """
    # M2: tolerant extraction — unwrap markdown fences (single- or multi-line, with
    # an optional `json` tag) and pull the first balanced top-level {...} object even
    # when the model wrapped it in prose. Matters on the cloud path (no format=json).
    text = _extract_json_object(raw_text)

    try:
        raw: dict = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Vision model returned non-JSON response: {exc}\nRaw: {raw_text[:200]}"
        ) from exc

    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Vision model returned non-dict JSON: {type(raw).__name__}"
        )

    # --- confidence validation ---
    confidence_downgraded = False
    raw_conf = raw.get("confidence")
    # bool is a subclass of int — exclude it so a hostile `"confidence": true`
    # cannot normalize to 1.0 and bypass the low-confidence gate.
    if raw_conf is None or isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
        # Non-numeric or missing → treat as 0.0 (triggers low_confidence)
        confidence = 0.0
        confidence_downgraded = True
    else:
        # Clamp to [0.0, 1.0]
        confidence = float(max(0.0, min(1.0, raw_conf)))
        if raw_conf != confidence:
            confidence_downgraded = True

    # --- zone validation ---
    # Coerce to str|None FIRST (mirrors the item field). known_zone_names is a set,
    # so a hostile list/dict suggested_zone would raise TypeError on the `in` test —
    # _coerce_scalar nulls non-scalars so the membership check is always safe.
    suggested_zone = _coerce_scalar(raw.get("suggested_zone"))
    if suggested_zone and suggested_zone in known_zone_names:
        zone_id: str | None = suggested_zone
    else:
        # Unknown, missing, or non-scalar zone → None + force low_confidence
        zone_id = None
        confidence_downgraded = True

    # --- item normalisation ---
    item = _coerce_scalar(raw.get("item"))

    # --- low_confidence flag ---
    low_confidence = (confidence < CONFIDENCE_THRESHOLD) or confidence_downgraded

    # M3: type-normalise every field that reaches create_item()/SQLite. A model that
    # returns an object/array for brand/model/category/zone_reasoning would otherwise
    # crash the SQLite bind AFTER the user Accepts; a string for key_properties/other_items
    # would be iterated char-by-char into notes. Coerce scalars→str|None, lists→list[str].
    return {
        "item": item,
        "brand": _coerce_scalar(raw.get("brand")),
        "model": _coerce_scalar(raw.get("model")),
        "category": _coerce_scalar(raw.get("category")),
        "marine": raw.get("marine"),
        "legible": raw.get("legible"),
        "key_properties": _coerce_str_list(raw.get("key_properties")),
        "other_items": _coerce_str_list(raw.get("other_items")),
        "suggested_zone": suggested_zone if zone_id is not None else None,
        "zone_id": zone_id,
        "zone_reasoning": _coerce_scalar(raw.get("zone_reasoning")),
        "confidence": confidence,
        "low_confidence": low_confidence,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Cloud identification (consent-gated — fires ONLY when cloud=True)
# ---------------------------------------------------------------------------

def _identify_cloud(
    image_path: str,
    prompt: str,
    known_zone_names: set[str],
    *,
    api_key: str,
) -> dict:
    """Call Anthropic /v1/messages with the image and prompt.

    Mirrors schematic.suggest_pages' Anthropic path: POST to
    https://api.anthropic.com/v1/messages with x-api-key header (read at call
    time), anthropic-version 2023-06-01, and the pinned CLOUD_VISION_MODEL.

    Gemini/OpenAI: NotImplementedError branches (documented placeholders, kept
    per the locked primary/secondary/tertiary decision).

    Args:
        image_path: Path to the image file.
        prompt: The taxonomy prompt built from build_prompt().
        known_zone_names: Set of valid zone names for result validation.
        api_key: The Anthropic API key (read at call time, never logged).

    Returns:
        A normalized result dict from _normalize_result(..., source="cloud").

    Raises:
        RuntimeError: If the API call fails (network, timeout, HTTP error).
    """
    # SANITIZE BEFORE EGRESS (CR-01/CR-02): never send the raw source bytes to a
    # third party. sanitize_image_bytes re-encodes to a clean JPEG in memory —
    # EXIF-stripped (GPS removed, so the vessel location does NOT leak), EXIF-
    # oriented, and resized ≤1920px. Because the output is always JPEG, the media
    # type is unconditionally image/jpeg — which also makes HEIC/HEIF (the primary
    # iPhone format) and any other Pillow-decodable input work on the cloud path
    # instead of being uploaded as raw bytes with a wrong/defaulted media type.
    # Lazy import keeps the offline-boundary import graph clean (photo is a sibling
    # capture module, never leopard44_kb.web).
    from leopard44_kb.capture.photo import sanitize_image_bytes

    img_bytes = sanitize_image_bytes(image_path)
    b64_data = base64.standard_b64encode(img_bytes).decode("ascii")
    media_type = "image/jpeg"  # always — we re-encode to JPEG above

    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            },
        },
        {
            "type": "text",
            "text": prompt,
        },
    ]

    try:
        response = _httpx_post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": CLOUD_VISION_MODEL,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=120.0,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        response.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            "Cloud-vision API not reachable — check network. "
            "Use local-only (omit --cloud) if offline."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            "Cloud-vision API timed out. "
            "Check network or use local-only (omit --cloud)."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Cloud-vision API returned HTTP {exc.response.status_code}. "
            "Check your ANTHROPIC_API_KEY or use local-only (omit --cloud)."
        ) from exc

    # M1: a malformed/non-JSON cloud body must yield a clean RuntimeError (the CLI
    # catches RuntimeError), never an uncaught JSONDecodeError/KeyError traceback.
    try:
        body = response.json()
        raw_text = ""
        for block in body.get("content", []):
            if block.get("type") == "text":
                raw_text += block.get("text", "")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            f"Cloud-vision API returned an unreadable response body: {exc}"
        ) from exc

    return _normalize_result(raw_text, known_zone_names, source="cloud")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_item(
    image_path: str,
    zones: list,
    *,
    cloud: bool = False,
) -> dict:
    """Identify an item from a photo, using local qwen2.5vl or cloud Anthropic vision.

    Local path (default, cloud=False):
      - Posts to {OLLAMA_HOST}/api/generate with options.num_ctx=8192 and format=json
      - The 32-zone taxonomy is embedded in the prompt
      - Returns a normalized VisionResult dict
      - Low local confidence sets low_confidence=True for the CLI to surface
        a "rerun with --cloud" suggestion — does NOT auto-send photo to cloud (H3)

    Cloud path (cloud=True, consent-gated):
      - Fires ONLY when cloud=True (H3 — photo bytes never leave machine without consent)
      - Reads ANTHROPIC_API_KEY from os.environ at call time (never logged)
      - cloud=True with no key → clear RuntimeError (not a silent local fallback)
      - Uses CLOUD_VISION_MODEL (pinned sonnet/opus-tier, NOT haiku)

    Args:
        image_path: Path to the image file (JPG/PNG/WEBP).
        zones: List of zone name strings OR (name, description) tuples.
               These define the valid zone taxonomy for both the prompt and
               suggested_zone validation.
        cloud: If True, use the cloud Anthropic API. Default False (local only).

    Returns:
        Normalized result dict with keys:
          item, brand, model, category, marine, legible, key_properties,
          other_items, suggested_zone, zone_id, zone_reasoning,
          confidence (float in [0.0, 1.0]), low_confidence (bool), source.

    Raises:
        RuntimeError: Ollama unreachable / model not pulled (local path), or
                      Anthropic API error (cloud path), or cloud=True with no key.
    """
    # Build the prompt with the zone taxonomy
    prompt = build_prompt(zones)

    # Extract the set of known zone names for validation
    known_zone_names: set[str] = set()
    for z in zones:
        if isinstance(z, tuple):
            known_zone_names.add(z[0])
        else:
            known_zone_names.add(str(z))

    # --- Cloud path (consent-gated: ONLY fires when cloud=True) ---
    if cloud:
        # Read API key at call time — never logged
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "cloud=True requires ANTHROPIC_API_KEY to be set in the environment. "
                "Set the key and retry, or omit --cloud to use local-only identification."
            )
        # NOTE: Gemini and OpenAI branches are documented placeholders per the
        # locked primary/secondary/tertiary decision (Anthropic primary).
        # google_key = os.environ.get("GOOGLE_API_KEY")   # → NotImplementedError
        # openai_key = os.environ.get("OPENAI_API_KEY")   # → NotImplementedError
        return _identify_cloud(image_path, prompt, known_zone_names, api_key=api_key)

    # --- Local path (default) ---
    # Read OLLAMA_HOST at call time so monkeypatch.setenv works in tests.
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{ollama_host}/api/generate"

    # B1: NEVER send the raw source bytes to the local model either. OLLAMA_HOST is
    # fully env-controlled and can legitimately point at a LAN Ollama box, so the
    # GPS-bearing original must NOT egress regardless of where it points. Sanitize
    # the same way the cloud path does (EXIF-strip → GPS removed → resize → JPEG
    # re-encode) so the photo location never leaves the box. Lazy import keeps the
    # offline-boundary import graph clean (photo is a sibling capture module).
    from leopard44_kb.capture.photo import sanitize_image_bytes

    # B1 advisory: if OLLAMA_HOST resolves to a NON-loopback host, the (sanitized)
    # photo is leaving this machine — surface a one-line, NON-BLOCKING stderr notice.
    # LAN Ollama on another box is a legitimate setup, so this is a notice, not an error.
    _warn_if_non_loopback_ollama(ollama_host)

    img_bytes = sanitize_image_bytes(image_path)
    b64_data = base64.standard_b64encode(img_bytes).decode("ascii")

    try:
        response = _httpx_post(
            url,
            json={
                "model": LOCAL_MODEL,
                "prompt": prompt,
                "images": [b64_data],
                "format": "json",
                "stream": False,
                # HARD CONSTRAINT (Spike 003 finding #3): the 32-zone taxonomy +
                # image overflows the default num_ctx=4096 → 400 error. Must be ≥ 8192.
                "options": {"num_ctx": 8192},
            },
            timeout=300.0,
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{LOCAL_MODEL}' not found — "
                f"run `ollama pull {LOCAL_MODEL}` then retry."
            )
        response.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            "Ollama not reachable — run `ollama serve` and "
            f"`ollama pull {LOCAL_MODEL}`, then retry."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Ollama identification timed out after 300s. "
            f"Check `ollama ps` for model status, then retry."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Ollama /api/generate returned HTTP {exc.response.status_code}"
        ) from exc

    # M1: a malformed/non-JSON Ollama body, or one missing the "response" key, must
    # yield a clean RuntimeError (the CLI catches RuntimeError), never an uncaught
    # JSONDecodeError/KeyError traceback.
    try:
        raw_response_text = response.json()["response"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Ollama returned an unreadable /api/generate response body: {exc}"
        ) from exc
    return _normalize_result(raw_response_text, known_zone_names, source="local")
