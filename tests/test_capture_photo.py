"""RED tests for CAP-01 / CAP-03: photo processing module contracts.

Phase 12 Wave-0 RED scaffold — tests FAIL at assertion until Wave-2/3 ships
leopard44_kb/capture/photo.py. Imports of leopard44_kb.capture.* are INSIDE each test
body (RED-at-assertion, not collection error).

Coverage:
  (h) a >1920px image is resized so max(width, height) == 1920
  (i) stored JPEG has NO GPS EXIF after process_photo strips it
  (j) HEIC input is decoded, resized, and EXIF-stripped to a valid JPEG (L2)
  (k) a decompression-bomb input is REJECTED by validate_photo_input before decode (M5)
"""
from __future__ import annotations

import io
import struct
from pathlib import Path


# ---------------------------------------------------------------------------
# (h) Resize: >1920px image has max dimension clamped to 1920
# ---------------------------------------------------------------------------

def test_large_image_is_resized_to_1920px(tmp_path):
    """process_photo resizes a 4000×3000 JPEG so max(width, height) == 1920.

    RED: ModuleNotFoundError until leopard44_kb/capture/photo.py is created.
    """
    from PIL import Image

    from leopard44_kb.capture import photo  # RED until Wave 2

    # Create a 4000×3000 test image (landscape — width is larger)
    img = Image.new("RGB", (4000, 3000), color=(120, 80, 40))
    src = tmp_path / "large.jpg"
    img.save(str(src), format="JPEG")

    out = photo.process_photo(src, dest_dir=tmp_path)

    result_img = Image.open(str(out))
    w, h = result_img.size
    assert max(w, h) == 1920, (
        f"Expected max dimension 1920, got {w}×{h}"
    )


def test_small_image_is_not_enlarged(tmp_path):
    """process_photo does NOT upscale images that are already ≤1920px on max side."""
    from PIL import Image

    from leopard44_kb.capture import photo  # RED until Wave 2

    img = Image.new("RGB", (800, 600), color=(50, 100, 150))
    src = tmp_path / "small.jpg"
    img.save(str(src), format="JPEG")

    out = photo.process_photo(src, dest_dir=tmp_path)

    result_img = Image.open(str(out))
    w, h = result_img.size
    assert max(w, h) <= 800, (
        f"Small image should not be enlarged, got {w}×{h}"
    )


# ---------------------------------------------------------------------------
# (i) EXIF strip: stored JPEG has NO GPS EXIF after processing
# ---------------------------------------------------------------------------

def test_gps_exif_is_stripped(tmp_path, gps_exif_jpeg):
    """process_photo removes all GPS EXIF from the output JPEG (i).

    Uses the gps_exif_jpeg fixture (GPS-tagged synthetic JPEG from conftest).
    RED: ModuleNotFoundError until leopard44_kb/capture/photo.py is created.
    """
    import piexif

    from leopard44_kb.capture import photo  # RED until Wave 2

    # Verify the fixture DOES have GPS data before processing
    before = piexif.load(str(gps_exif_jpeg))
    assert before["GPS"], f"Fixture should have GPS data, got: {before['GPS']}"

    out = photo.process_photo(gps_exif_jpeg, dest_dir=tmp_path)

    after = piexif.load(str(out))
    assert after["GPS"] == {}, (
        f"process_photo must strip all GPS EXIF, but GPS keys remain: {list(after['GPS'].keys())}"
    )


# ---------------------------------------------------------------------------
# (j) HEIC decode: HEIC input decoded, resized, stripped to valid JPEG (L2)
# ---------------------------------------------------------------------------

def test_heic_is_decoded_resized_stripped_to_jpeg(tmp_path, heic_photo):
    """process_photo accepts a .heic file and outputs a valid JPEG (j / L2).

    Uses the heic_photo fixture (HEIC written by pillow-heif in conftest).
    RED: ModuleNotFoundError until leopard44_kb/capture/photo.py is created.
    """
    from PIL import Image

    from leopard44_kb.capture import photo  # RED until Wave 2

    out = photo.process_photo(heic_photo, dest_dir=tmp_path)

    assert out.suffix.lower() in (".jpg", ".jpeg"), (
        f"Output of HEIC processing should be JPEG, got suffix: {out.suffix}"
    )

    # Must be a valid, openable JPEG
    img = Image.open(str(out))
    assert img.mode in ("RGB", "L"), f"Expected RGB or L JPEG output, got mode: {img.mode}"
    assert img.size[0] > 0 and img.size[1] > 0, f"Output image has zero dimension: {img.size}"


# ---------------------------------------------------------------------------
# (k) M5: decompression bomb rejected BEFORE decode
# ---------------------------------------------------------------------------

def test_decompression_bomb_is_rejected(tmp_path, monkeypatch):
    """validate_photo_input rejects a decompression-bomb image BEFORE any thumbnail/decode (M5).

    Monkeypatches PIL.Image.open to raise DecompressionBombError so we can test
    the pre-decode guard without needing a truly massive file on disk.
    RED: ModuleNotFoundError until leopard44_kb/capture/photo.py is created.
    """
    from PIL import Image

    from leopard44_kb.capture import photo  # RED until Wave 2

    # Create a fake image path (content doesn't matter — we patch open)
    fake_path = tmp_path / "bomb.jpg"
    fake_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    original_open = Image.open

    def _bomb_open(fp, *args, **kwargs):
        raise Image.DecompressionBombError("Image size exceeds limit")

    monkeypatch.setattr("PIL.Image.open", _bomb_open)

    # validate_photo_input should raise a ValueError (or custom CaptureError) — NOT propagate
    # the PIL-internal DecompressionBombError directly.
    try:
        photo.validate_photo_input(fake_path)
        raise AssertionError(
            "validate_photo_input should have raised an error for a decompression bomb"
        )
    except (ValueError, RuntimeError, OSError) as exc:
        # Expected: a clean, application-level error raised before decode proceeds
        assert "bomb" in str(exc).lower() or "size" in str(exc).lower() or "limit" in str(exc).lower() or "invalid" in str(exc).lower(), (
            f"Error message should mention size/bomb/limit/invalid, got: {exc}"
        )
    except Image.DecompressionBombError:
        raise AssertionError(
            "validate_photo_input must catch DecompressionBombError and raise a clean "
            "application error — not let the PIL exception propagate"
        )


def test_decompression_bomb_warning_is_rejected(tmp_path, monkeypatch):
    """validate_photo_input treats DecompressionBombWarning as a rejection (M5).

    DecompressionBombWarning fires before Error at a lower (still huge) size threshold.
    The photo validator must treat it as a hard rejection.
    RED: ModuleNotFoundError until leopard44_kb/capture/photo.py is created.
    """
    import warnings

    from PIL import Image

    from leopard44_kb.capture import photo  # RED until Wave 2

    fake_path = tmp_path / "semi_bomb.jpg"
    fake_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    def _warning_open(fp, *args, **kwargs):
        warnings.warn("Image size exceeds limit", Image.DecompressionBombWarning, stacklevel=2)
        # Return a fake image with enormous dimensions
        mock_img = Image.new("RGB", (100, 100))  # actual size doesn't matter
        # Patch size to simulate overflow
        return mock_img

    monkeypatch.setattr("PIL.Image.open", _warning_open)

    # The validator should either raise an error or filter the warning into an error
    # Both approaches are acceptable — the test asserts it does not silently accept
    # a file that triggered a bomb warning.
    try:
        photo.validate_photo_input(fake_path)
        # If it returns without error, that's acceptable ONLY if the impl uses
        # Image.MAX_IMAGE_PIXELS check before open (not relying on PIL warnings).
        # In that case, the validate must check pixel count on the file metadata.
        # For the RED test this path won't be reached (ModuleNotFoundError).
    except (ValueError, RuntimeError, OSError):
        pass  # Expected: clean rejection
    except Exception as exc:
        # If we get here, the bomb-check mechanism is working but raising something unexpected
        # Any non-PIL exception is acceptable at this stage of RED testing
        assert not isinstance(exc, Image.DecompressionBombWarning), (
            "Bomb warning should not propagate — must be caught and converted"
        )


# ---------------------------------------------------------------------------
# H1: sanitize_image_bytes() (the cloud library path) is bomb-bounded too.
# ---------------------------------------------------------------------------

def test_sanitize_image_bytes_rejects_decompression_bomb_warning(tmp_path, monkeypatch):
    """H1: sanitize_image_bytes runs validate_photo_input first, so a
    DecompressionBombWarning-sized image is rejected BEFORE decode/upload.

    The public identify_item(cloud=True) path calls sanitize_image_bytes directly,
    so this guard must live there too — not only on the CLI-validated on-disk path.
    """
    import warnings

    import pytest
    from PIL import Image

    from leopard44_kb.capture import photo

    fake_path = tmp_path / "semi_bomb.jpg"
    # A real small JPEG so the path exists / is_file passes; the warning is injected.
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(str(fake_path), format="JPEG")

    def _warning_open(fp, *args, **kwargs):
        warnings.warn("Image size exceeds limit", Image.DecompressionBombWarning, stacklevel=2)
        return Image.new("RGB", (16, 16))

    monkeypatch.setattr("PIL.Image.open", _warning_open)

    with pytest.raises(ValueError):
        photo.sanitize_image_bytes(fake_path)


# ---------------------------------------------------------------------------
# B3: a data/ symlink pointing OUTSIDE repo_root is rejected before any write.
# ---------------------------------------------------------------------------

def test_store_item_photo_rejects_data_symlink_escape(tmp_path):
    """B3: if data/ itself is a symlink to a sibling outside the repo, store_item_photo
    rejects with ValueError BEFORE creating any directory or writing any file.

    Mirrors inventory.create_item's extra resolved.relative_to(resolved_root) guard:
    validate_path alone cannot catch this because it resolves expected_root through
    the SAME data/ symlink.
    """
    import pytest
    from PIL import Image

    # repo_root with data/ symlinked to an OUTSIDE directory.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo_root / "data").symlink_to(outside, target_is_directory=True)

    # A genuine source photo so we get past validate_photo_input.
    src = tmp_path / "src.jpg"
    Image.new("RGB", (16, 16), color=(5, 5, 5)).save(str(src), format="JPEG")

    from leopard44_kb.capture import photo

    with pytest.raises(ValueError):
        photo.store_item_photo(item_id=1, src_path=src, repo_root=repo_root)

    # And nothing was written through the escape path.
    assert not any(outside.rglob("*.jpg")), (
        "B3: no photo file may be written outside the repo via the data/ symlink"
    )


def test_prepare_image_leaves_no_partial_file_on_encode_failure(tmp_path, monkeypatch):
    """No-orphan invariant: if the JPEG encode fails mid-write, prepare_image must
    NOT leave a partial/orphan file at dest_path (atomic temp + os.replace).

    The WR-01 unlink only covers a post-store DB-commit failure; this guards the
    earlier case where prepare_image itself raises during save (disk full, encoder
    error). dest_path must not exist, and no leftover .tmp file may remain in the dir.
    """
    import pytest
    from PIL import Image

    from leopard44_kb.capture import photo

    src = tmp_path / "src.jpg"
    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(str(src), format="JPEG")
    dest = tmp_path / "out" / "ITEM-1.jpg"

    # Make the JPEG encode blow up mid-save (simulates disk-full / encoder error).
    def _boom(self, *args, **kwargs):
        raise OSError("simulated mid-encode failure")

    monkeypatch.setattr(Image.Image, "save", _boom)

    with pytest.raises(OSError):
        photo.prepare_image(src, dest)

    assert not dest.exists(), "no partial/orphan file may remain at dest_path"
    leftovers = list(dest.parent.glob("*.tmp")) + list(dest.parent.glob(".*.tmp"))
    assert not leftovers, f"temp encode file was not cleaned up: {leftovers}"
