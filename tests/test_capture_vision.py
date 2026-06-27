"""RED tests for CAP-01 / CAP-03: vision module contracts.

Phase 12 Wave-0 RED scaffold — tests FAIL at assertion until Wave-2/3 ships
leopard44_kb/capture/vision.py. Imports of leopard44_kb.capture.* are INSIDE each test
body (RED-at-assertion, not collection error) following the Phase 11/10/9/8
precedent.

Coverage:
  (a) local identify_item posts to /api/generate with options.num_ctx == 8192
  (b) confidence < 0.7  → low_confidence True; >= 0.7 → False
  (c) CLOUD_VISION_MODEL is not haiku, is non-empty, is sonnet/opus-tier
  (d) cloud=False + low confidence + ANTHROPIC_API_KEY set → ZERO api.anthropic.com calls
  (e) markdown-fenced JSON ( ```json {...} ``` ) is tolerated
  (f) unknown suggested_zone → zone_id=None + confidence reset to low
  (g) out-of-range / non-numeric confidence is clamped or rejected to valid bounded value
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ollama_response(payload: dict) -> MagicMock:
    """Fake httpx.Response wrapping an Ollama /api/generate JSON reply."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": json.dumps(payload)}
    return mock_resp


def _write_minimal_jpeg(path) -> None:
    """Write a tiny REAL JPEG that Pillow can decode.

    B1: the local vision path now sanitizes the input via sanitize_image_bytes()
    (EXIF-strip → resize → JPEG re-encode), which calls Image.open(). A bare
    JPEG-magic-byte stub is NOT decodable, so local-path tests must hand a genuine
    image. Uses a 16×16 RGB square, written as JPEG.
    """
    from PIL import Image

    Image.new("RGB", (16, 16), color=(120, 120, 120)).save(str(path), format="JPEG")


def _valid_vision_result(
    *,
    confidence: float = 0.85,
    suggested_zone: str = "Saloon",
    item: str = "raw-water impeller",
) -> dict:
    """Build a minimally valid vision result matching the spike 003 schema."""
    return {
        "item": item,
        "brand": None,
        "model": None,
        "category": "engine part",
        "marine": True,
        "legible": False,
        "key_properties": ["rubber", "6-vane"],
        "other_items": [],
        "suggested_zone": suggested_zone,
        "zone_reasoning": "Stored near engine bay",
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# (a) num_ctx == 8192 in the Ollama payload
# ---------------------------------------------------------------------------

def test_identify_item_sends_num_ctx_8192(monkeypatch, tmp_path):
    """identify_item POSTs to /api/generate with options.num_ctx == 8192.

    RED: ModuleNotFoundError until leopard44_kb/capture/vision.py is created.
    The captured httpx.post payload is asserted directly.
    """
    from leopard44_kb.capture import vision  # RED until Wave 2

    captured_payload: list[dict] = []

    def _fake_post(url: str, json: dict, timeout: float | None = None, **kw) -> MagicMock:
        captured_payload.append(json)
        return _make_ollama_response(_valid_vision_result())

    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", _fake_post)

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)  # B1: real decodable JPEG (local path sanitizes)

    vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    assert len(captured_payload) == 1, "Expected exactly one POST call"
    payload = captured_payload[0]
    assert "options" in payload, f"Expected 'options' key in payload, got: {payload}"
    assert payload["options"]["num_ctx"] == 8192, (
        f"Expected num_ctx=8192, got: {payload['options'].get('num_ctx')}"
    )


# ---------------------------------------------------------------------------
# (b) confidence gate: below 0.7 → low_confidence True, >= 0.7 → False
# ---------------------------------------------------------------------------

def test_identify_item_low_confidence_flag(monkeypatch, tmp_path):
    """confidence < 0.7 → result.low_confidence is True."""
    from leopard44_kb.capture import vision  # RED until Wave 2

    monkeypatch.setattr(
        "leopard44_kb.capture.vision._httpx_post",
        lambda *a, **kw: _make_ollama_response(_valid_vision_result(confidence=0.55)),
    )

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)
    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    assert result["low_confidence"] is True, (
        f"Expected low_confidence=True for conf=0.55, got: {result.get('low_confidence')}"
    )


def test_identify_item_high_confidence_not_flagged(monkeypatch, tmp_path):
    """confidence >= 0.7 → result.low_confidence is False."""
    from leopard44_kb.capture import vision  # RED until Wave 2

    monkeypatch.setattr(
        "leopard44_kb.capture.vision._httpx_post",
        lambda *a, **kw: _make_ollama_response(_valid_vision_result(confidence=0.85)),
    )

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)
    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    assert result["low_confidence"] is False, (
        f"Expected low_confidence=False for conf=0.85, got: {result.get('low_confidence')}"
    )


def test_confidence_threshold_exactly_0_7(monkeypatch, tmp_path):
    """confidence == 0.7 (boundary) → low_confidence is False (not flagged)."""
    from leopard44_kb.capture import vision  # RED until Wave 2

    monkeypatch.setattr(
        "leopard44_kb.capture.vision._httpx_post",
        lambda *a, **kw: _make_ollama_response(_valid_vision_result(confidence=0.7)),
    )

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)
    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    assert result["low_confidence"] is False, (
        f"confidence==0.7 should NOT be low_confidence, got: {result.get('low_confidence')}"
    )


def test_non_scalar_suggested_zone_does_not_crash(monkeypatch, tmp_path):
    """A hostile non-string `suggested_zone` (list/dict) must NOT raise TypeError on
    the `in known_zone_names` (set) membership test — it coerces to None → zone_id
    None + low_confidence, never a crash (malformed-output invariant)."""
    from leopard44_kb.capture import vision  # RED until Wave 2

    payload = _valid_vision_result(suggested_zone="Saloon")
    payload["suggested_zone"] = ["Saloon", "Galley"]  # hostile non-scalar

    monkeypatch.setattr(
        "leopard44_kb.capture.vision._httpx_post",
        lambda *a, **kw: _make_ollama_response(payload),
    )

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)

    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    assert result["zone_id"] is None, "non-scalar zone must resolve to None"
    assert result["suggested_zone"] is None
    assert result["low_confidence"] is True


def test_boolean_confidence_does_not_bypass_low_confidence(monkeypatch, tmp_path):
    """A hostile `"confidence": true` must NOT normalize to 1.0 (bool is a subclass
    of int). It is treated as non-numeric → confidence 0.0 → low_confidence True."""
    from leopard44_kb.capture import vision  # RED until Wave 2

    monkeypatch.setattr(
        "leopard44_kb.capture.vision._httpx_post",
        lambda *a, **kw: _make_ollama_response(_valid_vision_result(confidence=True)),
    )

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)

    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    assert result["confidence"] == 0.0, (
        f"boolean confidence must be rejected to 0.0, got: {result.get('confidence')}"
    )
    assert result["low_confidence"] is True, (
        "boolean confidence must trigger low_confidence, not bypass it"
    )


# ---------------------------------------------------------------------------
# (c) CLOUD_VISION_MODEL is pinned, non-empty, sonnet/opus-tier (not haiku)
# ---------------------------------------------------------------------------

def test_cloud_vision_model_constant():
    """CLOUD_VISION_MODEL is not haiku and is a plausible sonnet/opus-tier identifier.

    RED: ModuleNotFoundError until leopard44_kb/capture/vision.py defines the constant.
    Asserts the constant directly (M1) — no regex on prose.
    """
    from leopard44_kb.capture import vision  # RED until Wave 2

    model = vision.CLOUD_VISION_MODEL
    assert model, "CLOUD_VISION_MODEL must be non-empty"
    assert model != "claude-3-haiku-20240307", (
        "CLOUD_VISION_MODEL must NOT be haiku (spike 003: haiku adds cost with no advantage "
        "over local qwen2.5vl on hard cases — use sonnet/opus)"
    )
    # Must be a plausible sonnet or opus tier Anthropic model
    model_lower = model.lower()
    assert "sonnet" in model_lower or "opus" in model_lower, (
        f"CLOUD_VISION_MODEL should be sonnet or opus tier, got: {model!r}"
    )


# ---------------------------------------------------------------------------
# (d) H3: cloud=False + low confidence + key set → ZERO api.anthropic.com calls
# ---------------------------------------------------------------------------

def test_no_cloud_egress_without_cloud_flag(monkeypatch, tmp_path):
    """cloud=False makes ZERO calls to api.anthropic.com even when confidence is low
    and ANTHROPIC_API_KEY is set in the environment (H3 consent gate).

    RED: ModuleNotFoundError until leopard44_kb/capture/vision.py exists.
    """
    from leopard44_kb.capture import vision  # RED until Wave 2

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-not-real-00000000000000")

    anthropic_calls: list[str] = []

    def _fake_post(url: str, json: dict, timeout: float | None = None, **kw) -> MagicMock:
        if "anthropic.com" in url or "api.anthropic.com" in url:
            anthropic_calls.append(url)
        return _make_ollama_response(_valid_vision_result(confidence=0.45))

    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", _fake_post)

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)
    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    assert anthropic_calls == [], (
        f"cloud=False must make ZERO api.anthropic.com calls, but called: {anthropic_calls}"
    )
    assert result["low_confidence"] is True


# ---------------------------------------------------------------------------
# (e) M3: markdown-fenced JSON is tolerated and parsed
# ---------------------------------------------------------------------------

def test_markdown_fenced_json_is_tolerated(monkeypatch, tmp_path):
    """Model output wrapped in ```json ... ``` fences is still parsed correctly (M3).

    Some models emit markdown-fenced JSON even when asked for plain JSON.
    identify_item must strip the fence and parse the inner object.
    RED: ModuleNotFoundError until leopard44_kb/capture/vision.py exists.
    """
    from leopard44_kb.capture import vision  # RED until Wave 2

    payload = _valid_vision_result(confidence=0.9, item="winch handle")
    fenced_response = f"```json\n{json.dumps(payload)}\n```"

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": fenced_response}

    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", lambda *a, **kw: mock_resp)

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)
    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    assert result["item"] == "winch handle", (
        f"markdown-fenced JSON should be parsed; got item={result.get('item')!r}"
    )


# ---------------------------------------------------------------------------
# (f) M3: unknown suggested_zone → zone_id=None, confidence treated as low
# ---------------------------------------------------------------------------

def test_unknown_suggested_zone_yields_no_zone(monkeypatch, tmp_path):
    """suggested_zone not in the provided zones list → zone_id=None + low_confidence (M3).

    RED: ModuleNotFoundError until leopard44_kb/capture/vision.py exists.
    """
    from leopard44_kb.capture import vision  # RED until Wave 2

    monkeypatch.setattr(
        "leopard44_kb.capture.vision._httpx_post",
        lambda *a, **kw: _make_ollama_response(
            _valid_vision_result(confidence=0.9, suggested_zone="Nowhere In Particular")
        ),
    )

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)
    result = vision.identify_item(
        str(image_path),
        zones=["Saloon", "Port engine room", "Anchor locker"],
        cloud=False,
    )

    # Unknown zone → zone_id must be None and result flagged low
    assert result.get("zone_id") is None, (
        f"Unknown zone should yield zone_id=None, got: {result.get('zone_id')}"
    )
    assert result["low_confidence"] is True, (
        "Unknown zone should set low_confidence=True regardless of raw confidence"
    )


# ---------------------------------------------------------------------------
# (g) M3: out-of-range confidence is clamped/rejected to a valid bounded value
# ---------------------------------------------------------------------------

def test_out_of_range_confidence_above_1_is_clamped(monkeypatch, tmp_path):
    """confidence = 1.7 (above valid range) is clamped to 1.0 or rejected."""
    from leopard44_kb.capture import vision  # RED until Wave 2

    monkeypatch.setattr(
        "leopard44_kb.capture.vision._httpx_post",
        lambda *a, **kw: _make_ollama_response(_valid_vision_result(confidence=1.7)),
    )

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)
    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    conf = result.get("confidence", -999)
    assert 0.0 <= conf <= 1.0, f"confidence=1.7 must be clamped to [0, 1], got: {conf}"


def test_out_of_range_confidence_below_0_is_clamped(monkeypatch, tmp_path):
    """confidence = -0.2 (below valid range) is clamped to 0.0 or rejected."""
    from leopard44_kb.capture import vision  # RED until Wave 2

    monkeypatch.setattr(
        "leopard44_kb.capture.vision._httpx_post",
        lambda *a, **kw: _make_ollama_response(_valid_vision_result(confidence=-0.2)),
    )

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)
    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    conf = result.get("confidence", -999)
    assert 0.0 <= conf <= 1.0, f"confidence=-0.2 must be clamped to [0, 1], got: {conf}"


def test_non_numeric_confidence_is_rejected(monkeypatch, tmp_path):
    """confidence = 'high' (non-numeric string) is replaced with a valid bounded value."""
    from leopard44_kb.capture import vision  # RED until Wave 2

    payload = _valid_vision_result(confidence=0.9)
    payload["confidence"] = "high"  # type: ignore[assignment]

    monkeypatch.setattr(
        "leopard44_kb.capture.vision._httpx_post",
        lambda *a, **kw: _make_ollama_response(payload),
    )

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)
    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    conf = result.get("confidence", -999)
    assert isinstance(conf, float), f"non-numeric confidence should be normalized to float, got: {type(conf)}"
    assert 0.0 <= conf <= 1.0, f"non-numeric confidence should be clamped to [0, 1], got: {conf}"


def test_null_confidence_is_rejected(monkeypatch, tmp_path):
    """confidence = null (JSON None) is treated as 0.0 (minimum, triggers low_confidence)."""
    from leopard44_kb.capture import vision  # RED until Wave 2

    payload = _valid_vision_result(confidence=0.9)
    payload["confidence"] = None  # type: ignore[assignment]

    monkeypatch.setattr(
        "leopard44_kb.capture.vision._httpx_post",
        lambda *a, **kw: _make_ollama_response(payload),
    )

    image_path = tmp_path / "test.jpg"
    _write_minimal_jpeg(image_path)
    result = vision.identify_item(str(image_path), zones=["Saloon"], cloud=False)

    conf = result.get("confidence", -999)
    assert isinstance(conf, (int, float)), f"null confidence should become numeric, got: {type(conf)}"
    assert 0.0 <= conf <= 1.0, f"null confidence should be clamped to [0, 1], got: {conf}"
    # null confidence must trigger low_confidence flag
    assert result["low_confidence"] is True, (
        "null confidence should set low_confidence=True"
    )


# ---------------------------------------------------------------------------
# CR-01 / CR-02: the cloud path sanitizes the image BEFORE egress.
#   - GPS EXIF is stripped (no location leak to api.anthropic.com).
#   - The upload is always a valid JPEG with media_type image/jpeg, so HEIC
#     (the primary iPhone format) works and is never sent as raw mislabelled bytes.
# ---------------------------------------------------------------------------

def _make_anthropic_response(payload: dict) -> MagicMock:
    """Fake httpx.Response wrapping an Anthropic /v1/messages JSON reply."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": json.dumps(payload)}]
    }
    return mock_resp


def _capture_cloud_image_source(monkeypatch) -> dict:
    """Monkeypatch _httpx_post to capture the image 'source' block sent to Anthropic.

    Returns the dict that the caller fills in by reference once identify_item runs:
    {"source": <image source dict>, "url": <posted url>}.
    """
    captured: dict = {}

    def _fake_post(url, json, timeout=None, **kw):  # noqa: A002 — match wrapper sig
        captured["url"] = url
        for block in json["messages"][0]["content"]:
            if block.get("type") == "image":
                captured["source"] = block["source"]
        return _make_anthropic_response(_valid_vision_result(confidence=0.9))

    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", _fake_post)
    return captured


def test_cloud_upload_strips_gps_exif(monkeypatch, gps_exif_jpeg):
    """The bytes the cloud path posts decode to a JPEG with NO GPS IFD (CR-01).

    Uses the gps_exif_jpeg conftest fixture (a JPEG carrying a non-empty GPS
    block). Asserts the base64 image actually sent to api.anthropic.com decodes
    to a JPEG whose piexif.load(...)["GPS"] == {} — i.e. the vessel location is
    stripped before egress, never the raw source bytes.
    """
    import base64 as _b64
    import io as _io

    import piexif
    from PIL import Image as _Image

    from leopard44_kb.capture import vision

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real-0000000000000000")
    captured = _capture_cloud_image_source(monkeypatch)

    # Sanity: the SOURCE file genuinely has a GPS IFD to begin with.
    src_gps = piexif.load(str(gps_exif_jpeg)).get("GPS", {})
    assert src_gps, "fixture precondition: source JPEG must carry a GPS IFD"

    vision.identify_item(str(gps_exif_jpeg), zones=["Saloon"], cloud=True)

    source = captured.get("source")
    assert source is not None, "expected an image block in the Anthropic request"
    assert source["media_type"] == "image/jpeg", (
        f"re-encoded upload must be image/jpeg, got {source['media_type']!r}"
    )

    sent_bytes = _b64.standard_b64decode(source["data"])

    # The uploaded bytes must be a valid JPEG with an EMPTY GPS IFD.
    assert _Image.open(_io.BytesIO(sent_bytes)).format == "JPEG"
    sent_gps = piexif.load(sent_bytes).get("GPS", {})
    assert sent_gps == {}, (
        f"GPS EXIF must be stripped before cloud egress (CR-01), got: {sent_gps!r}"
    )


def test_cloud_upload_heic_becomes_jpeg(monkeypatch, heic_photo):
    """A HEIC input is re-encoded to a valid image/jpeg upload (CR-02).

    HEIC/HEIF is the primary iPhone capture format. The cloud path must decode it
    and upload a JPEG (media_type image/jpeg), never the raw HEIC bytes labelled
    as JPEG. Uses the heic_photo conftest fixture.
    """
    import base64 as _b64
    import io as _io

    from PIL import Image as _Image

    from leopard44_kb.capture import vision

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real-0000000000000000")
    captured = _capture_cloud_image_source(monkeypatch)

    vision.identify_item(str(heic_photo), zones=["Saloon"], cloud=True)

    source = captured.get("source")
    assert source is not None, "expected an image block in the Anthropic request"
    assert source["media_type"] == "image/jpeg", (
        f"HEIC upload must be re-encoded to image/jpeg, got {source['media_type']!r}"
    )

    sent_bytes = _b64.standard_b64decode(source["data"])
    assert _Image.open(_io.BytesIO(sent_bytes)).format == "JPEG", (
        "HEIC input must produce valid JPEG upload bytes (not raw HEIC)"
    )


# ---------------------------------------------------------------------------
# B1: the LOCAL (cloud=False) path also sanitizes — GPS never reaches Ollama,
# regardless of where OLLAMA_HOST points.
# ---------------------------------------------------------------------------

def _capture_local_images(monkeypatch) -> dict:
    """Monkeypatch _httpx_post to capture the base64 images list the local path POSTs."""
    captured: dict = {}

    def _fake_post(url, json, timeout=None, **kw):  # noqa: A002 — match wrapper sig
        captured["url"] = url
        captured["images"] = json.get("images")
        return _make_ollama_response(_valid_vision_result(confidence=0.9))

    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", _fake_post)
    return captured


def test_local_path_strips_gps_before_ollama(monkeypatch, gps_exif_jpeg):
    """B1: bytes handed to _httpx_post in the local `images` list decode to a
    JPEG with NO GPS IFD — the original GPS-bearing bytes never reach Ollama.
    """
    import base64 as _b64
    import io as _io

    import piexif
    from PIL import Image as _Image

    from leopard44_kb.capture import vision

    # Sanity: the source genuinely carries GPS.
    assert piexif.load(str(gps_exif_jpeg)).get("GPS", {}), (
        "fixture precondition: source JPEG must carry a GPS IFD"
    )

    captured = _capture_local_images(monkeypatch)
    vision.identify_item(str(gps_exif_jpeg), zones=["Saloon"], cloud=False)

    images = captured.get("images")
    assert images and len(images) == 1, "local path must POST exactly one image"

    sent_bytes = _b64.standard_b64decode(images[0])
    assert _Image.open(_io.BytesIO(sent_bytes)).format == "JPEG", (
        "local path must send re-encoded JPEG bytes, not raw original"
    )
    sent_gps = piexif.load(sent_bytes).get("GPS", {})
    assert sent_gps == {}, (
        f"B1: GPS EXIF must be stripped before the LOCAL Ollama call, got: {sent_gps!r}"
    )


def test_local_path_non_loopback_ollama_prints_advisory(monkeypatch, gps_exif_jpeg, capsys):
    """B1: a non-loopback OLLAMA_HOST prints a NON-BLOCKING stderr advisory and still works."""
    from leopard44_kb.capture import vision

    monkeypatch.setenv("OLLAMA_HOST", "http://192.0.2.5:11434")
    _capture_local_images(monkeypatch)

    # Must NOT raise — LAN Ollama is a legitimate setup.
    result = vision.identify_item(str(gps_exif_jpeg), zones=["Saloon"], cloud=False)
    assert result["item"], "capture must still succeed against a LAN Ollama host"

    err = capsys.readouterr().err
    assert "192.0.2.5" in err and "OLLAMA_HOST" in err, (
        f"expected a non-loopback advisory mentioning the host, got: {err!r}"
    )


def test_local_path_localhost_ollama_no_advisory(monkeypatch, gps_exif_jpeg, capsys):
    """B1: a loopback OLLAMA_HOST prints NO advisory (default on-box setup)."""
    from leopard44_kb.capture import vision

    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    _capture_local_images(monkeypatch)

    vision.identify_item(str(gps_exif_jpeg), zones=["Saloon"], cloud=False)

    err = capsys.readouterr().err
    assert "OLLAMA_HOST is not localhost" not in err, (
        f"loopback host must not trigger the advisory, got: {err!r}"
    )


# ---------------------------------------------------------------------------
# M1: malformed model response bodies raise a bounded RuntimeError (clean), not
# an uncaught JSONDecodeError / KeyError traceback.
# ---------------------------------------------------------------------------

def test_local_malformed_body_raises_runtimeerror(monkeypatch, gps_exif_jpeg):
    """M1: a local response body whose .json() blows up → RuntimeError (clean)."""
    import pytest

    from leopard44_kb.capture import vision

    def _fake_post(url, json, timeout=None, **kw):  # noqa: A002
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
        return resp

    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", _fake_post)

    with pytest.raises(RuntimeError):
        vision.identify_item(str(gps_exif_jpeg), zones=["Saloon"], cloud=False)


def test_local_missing_response_key_raises_runtimeerror(monkeypatch, gps_exif_jpeg):
    """M1: a local response body missing the 'response' key → RuntimeError, not KeyError."""
    import pytest

    from leopard44_kb.capture import vision

    def _fake_post(url, json, timeout=None, **kw):  # noqa: A002
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"unexpected": "shape"}  # no "response" key
        return resp

    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", _fake_post)

    with pytest.raises(RuntimeError):
        vision.identify_item(str(gps_exif_jpeg), zones=["Saloon"], cloud=False)


def test_cloud_malformed_body_raises_runtimeerror(monkeypatch, gps_exif_jpeg):
    """M1: a cloud response body whose .json() blows up → RuntimeError (clean)."""
    import pytest

    from leopard44_kb.capture import vision

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real-0000000000000000")

    def _fake_post(url, json, timeout=None, **kw):  # noqa: A002
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = ValueError("not json")
        return resp

    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", _fake_post)

    with pytest.raises(RuntimeError):
        vision.identify_item(str(gps_exif_jpeg), zones=["Saloon"], cloud=True)


# ---------------------------------------------------------------------------
# M2: tolerant fence stripping — single-line fenced JSON and prose-wrapped JSON.
# ---------------------------------------------------------------------------

def test_single_line_fenced_json_is_tolerated(monkeypatch, gps_exif_jpeg):
    """M2: ```json {...} ``` on a single line is parsed (no internal newlines)."""
    from leopard44_kb.capture import vision

    payload = _valid_vision_result(confidence=0.9, item="winch handle")
    single_line = f"```json {json.dumps(payload)} ```"

    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"response": single_line}
    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", lambda *a, **kw: resp)

    result = vision.identify_item(str(gps_exif_jpeg), zones=["Saloon"], cloud=False)
    assert result["item"] == "winch handle"


def test_prose_wrapped_json_is_extracted(monkeypatch, gps_exif_jpeg):
    """M2: JSON object surrounded by prose is extracted (first balanced {...})."""
    from leopard44_kb.capture import vision

    payload = _valid_vision_result(confidence=0.9, item="bilge pump")
    prose = f"Here is the result you asked for:\n{json.dumps(payload)}\nHope that helps!"

    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"response": prose}
    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", lambda *a, **kw: resp)

    result = vision.identify_item(str(gps_exif_jpeg), zones=["Saloon"], cloud=False)
    assert result["item"] == "bilge pump"


# ---------------------------------------------------------------------------
# M3: vision fields are type-normalized before they reach SQLite.
# ---------------------------------------------------------------------------

def test_object_brand_is_coerced_to_str_or_none(monkeypatch, gps_exif_jpeg):
    """M3: a dict/list brand from the model is coerced (never dict/list)."""
    from leopard44_kb.capture import vision

    payload = _valid_vision_result(confidence=0.9)
    payload["brand"] = {"name": "Jabsco"}  # model returned an object
    payload["model"] = ["X", "Y"]          # model returned an array
    payload["key_properties"] = "single-string-not-list"
    payload["other_items"] = {"weird": "object"}

    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"response": json.dumps(payload)}
    monkeypatch.setattr("leopard44_kb.capture.vision._httpx_post", lambda *a, **kw: resp)

    result = vision.identify_item(str(gps_exif_jpeg), zones=["Saloon"], cloud=False)

    assert result["brand"] is None or isinstance(result["brand"], str), (
        f"brand must be str|None, got {type(result['brand']).__name__}"
    )
    assert result["model"] is None or isinstance(result["model"], str), (
        f"model must be str|None, got {type(result['model']).__name__}"
    )
    assert isinstance(result["key_properties"], list), "key_properties must be a list"
    assert all(isinstance(p, str) for p in result["key_properties"]), (
        "key_properties must be list[str]"
    )
    assert isinstance(result["other_items"], list), "other_items must be a list"
    assert all(isinstance(p, str) for p in result["other_items"]), (
        "other_items must be list[str]"
    )
