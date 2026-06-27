"""Photo processing for the Leopard 44 KB capture pipeline (CAP-03).

CONNECTED surface — never imported by leopard44_kb.web (offline guarantee
enforced by tests/test_capture_import_boundary.py).

Responsibilities:
- Validate photo input: reject missing, directory, non-image, and decompression-bomb
  paths with a clear ValueError (M5).
- HEIC decode support via pillow-heif (L2) — plain Pillow cannot decode HEIC/HEIF
  without the registered opener (confirmed in Spike 003).
- Resize to a maximum longest-edge of 1920 px while preserving aspect ratio.
- Strip ALL GPS EXIF before writing (T-12-07, CAP-03).
- Write the processed image to the vessel layer under data/photos/ (never static/
  or shared/).

Architecture rule: this module imports ONLY Pillow, pillow_heif, piexif, pathlib,
and leopard44_kb.paths. It must NOT import leopard44_kb.web or leopard44_kb.answer — the capture
surface is separate from the offline query surface.
"""
from __future__ import annotations

import io
import os
import tempfile
import warnings
from pathlib import Path
from typing import Union

import piexif
import pillow_heif
from PIL import Image, ImageOps

from leopard44_kb.paths import validate_path

# ---------------------------------------------------------------------------
# L2: Register HEIC/HEIF opener so Pillow can decode iPhone capture photos.
# This runs at import time so every call to Image.open() in this module can
# transparently open .heic / .heif files.
# ---------------------------------------------------------------------------
pillow_heif.register_heif_opener()

# ---------------------------------------------------------------------------
# M5: Decompression-bomb guard. Set a conservative MAX_IMAGE_PIXELS value
# (50 MP) — comfortably above any real 1920px capture (≤3.7 MP) but well
# below a malicious bomb. Pillow raises DecompressionBombError above this
# limit and DecompressionBombWarning slightly below it. Both are treated as
# validation failures in validate_photo_input.
# ---------------------------------------------------------------------------
Image.MAX_IMAGE_PIXELS = 50_000_000  # 50 MP ceiling


def validate_photo_input(path: Union[str, Path]) -> Path:
    """Validate that *path* points to a readable image file.

    Checks (in order):
    1. Path exists and is a regular file (not a directory).
    2. The file is a valid image Pillow can open — uses Image.open() + verify()
       to confirm the image header is genuine.
    3. The pixel count does not exceed Image.MAX_IMAGE_PIXELS and the open does
       not raise DecompressionBombError or DecompressionBombWarning (M5 guard).

    Args:
        path: Path to the candidate image file (str or Path).

    Returns:
        The resolved Path on success.

    Raises:
        ValueError: For any of: missing path, directory, non-image/unreadable
                    image, or decompression-bomb input (M5).
    """
    resolved = Path(path).resolve()

    if not resolved.exists():
        raise ValueError(f"Photo path does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Photo path is not a file (is a directory?): {resolved}")

    # M5: catch DecompressionBombError AND DecompressionBombWarning.
    # Turn warnings into errors inside this block so a warning-level bomb
    # (below the hard limit but still huge) is also a hard rejection.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
            try:
                img = Image.open(resolved)
                # verify() checks the image header fully; it closes the file handle
                # so the image object is no longer usable after this call.
                img.verify()
            except Image.DecompressionBombError as exc:
                raise ValueError(
                    f"Decompression bomb rejected (image size exceeds limit): {resolved}"
                ) from exc
            except Image.DecompressionBombWarning as exc:
                raise ValueError(
                    f"Decompression bomb warning treated as rejection (image too large): {resolved}"
                ) from exc
    except ValueError:
        raise  # re-raise our own ValueErrors unchanged
    except Exception as exc:
        # Pillow raises various exceptions for unreadable / non-image files
        # (UnidentifiedImageError, SyntaxError, OSError, etc.)
        raise ValueError(
            f"Not a valid image file (Pillow could not open/verify): {resolved} — {exc}"
        ) from exc

    return resolved


def _orient_resize_rgb(img: "Image.Image") -> "Image.Image":
    """Apply the shared sanitize pipeline to an open Pillow image.

    Steps (no I/O — pure in-memory transform):
    1. Apply EXIF orientation transpose (upright; handles rotated phone photos).
    2. Convert to RGB/L so the result is JPEG-encodable (HEIC/PNG may be RGBA/P).
    3. Resize so the longest edge is at most 1920 px (preserve aspect, never upscale).

    The returned image carries NO EXIF (orientation has been baked into the pixels
    and ``exif_transpose`` drops the orientation tag); callers must then save as
    JPEG WITHOUT an ``exif=`` keyword to keep the GPS strip (T-12-07, CAP-03).

    This is the single source of truth for both the on-disk path (prepare_image)
    and the in-memory cloud-upload path (sanitize_image_bytes) so the two cannot
    drift (CR-01/CR-02).
    """
    # Honour EXIF orientation so a portrait photo from a phone is upright.
    img = ImageOps.exif_transpose(img)

    # Ensure RGB for JPEG encoding (HEIC/PNG may come in as RGBA or P mode).
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Resize: thumbnail modifies in-place, preserves aspect, never upscales.
    img.thumbnail((1920, 1920), Image.LANCZOS)

    return img


def sanitize_image_bytes(src_path: Union[str, Path]) -> bytes:
    """Return a sanitized in-memory JPEG of *src_path* (EXIF-stripped, ≤1920px).

    Produces the SAME bytes the on-disk store would write, but in memory and
    without touching the filesystem. Used by the cloud-vision path so the photo
    that egresses to a third party is EXIF-stripped (GPS removed), EXIF-oriented,
    resized, and unconditionally re-encoded as JPEG — never the raw source bytes
    (CR-01) and never a mislabelled HEIC/unknown format (CR-02).

    Post-condition:
        piexif.load(returned_bytes)["GPS"] == {} (no GPS IFD in the output) and
        the bytes always decode as a JPEG regardless of the input format.

    Args:
        src_path: Source image (JPEG, PNG, HEIC, WEBP, or any Pillow-supported
                  format). HEIC decodes because register_heif_opener() ran at import.

    Returns:
        The sanitized JPEG bytes.

    Raises:
        ValueError: If src_path is missing, a directory, not a valid image, or a
                    decompression bomb (H1 — same guard as the on-disk path).
    """
    # H1: run the SAME input validation the on-disk path uses BEFORE decoding, so the
    # public identify_item(cloud=True) library entry point is bomb-bounded even when
    # the CLI did not pre-validate. validate_photo_input converts DecompressionBombWarning
    # → ValueError and enforces MAX_IMAGE_PIXELS. We thread its resolved Path through to
    # Image.open so the bytes decoded here are the bytes that were validated.
    resolved_src = validate_photo_input(src_path)

    img = Image.open(resolved_src)
    img = _orient_resize_rgb(img)

    buf = io.BytesIO()
    # No exif= keyword → ALL EXIF (including the GPS IFD) is dropped.
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


def prepare_image(src_path: Union[str, Path], dest_path: Union[str, Path]) -> Path:
    """Process and write a photo: EXIF orientation → resize to 1920px → GPS-strip.

    Steps:
    1. Open via Pillow (HEIC works because register_heif_opener() ran at import).
    2. Apply EXIF orientation transpose (upright storage; handles rotated phone photos).
    3. Resize so the longest edge is at most 1920 px using Image.thumbnail, which
       preserves aspect ratio and never upscales.
    4. Save as JPEG WITHOUT copying any EXIF data — this is the GPS strip (T-12-07,
       CAP-03). The saved file has only the bare pixel data re-encoded as JPEG.

    Post-condition:
        piexif.load(str(dest_path))["GPS"] == {} (no GPS IFD in the output)

    Args:
        src_path: Source image (JPEG, PNG, HEIC, WEBP, or any Pillow-supported format).
                  The caller should pass an already-resolved/validated Path
                  (WR-02): store_item_photo / process_photo thread the single
                  validate_photo_input() result through so the bytes decoded here
                  are the bytes that were validated.
        dest_path: Destination JPEG path. Parent directory must already exist.

    Returns:
        The resolved destination Path.
    """
    src_path = Path(src_path).resolve()
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(src_path)
    img = _orient_resize_rgb(img)

    # Save atomically: encode to a temp file in the SAME directory, then os.replace
    # it onto the final path. A mid-encode failure (disk full, encoder error) leaves
    # only the temp file, which we unlink — never a partial/orphan JPEG at dest_path.
    # This upholds the no-orphan invariant for every caller (store_item_photo's
    # fail-soft contract relies on it; commit_capture keeps photo_path NULL upstream).
    # os.replace is atomic on the same filesystem, so a reader never sees a half-written
    # dest either. Save WITHOUT an exif= keyword — that drops ALL EXIF incl. GPS (T-12-07).
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest_path.stem}.", suffix=".jpg.tmp", dir=str(dest_path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        img.save(str(tmp_path), format="JPEG", quality=88, optimize=True)
        os.replace(str(tmp_path), str(dest_path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return dest_path.resolve()


def store_item_photo(
    item_id: int,
    src_path: Union[str, Path],
    repo_root: Union[str, Path],
) -> str:
    """Store a processed photo for an inventory item in the vessel layer.

    Destination: data/photos/items/ITEM-{item_id}.jpg (vessel layer, gitignored).

    Security: validate_path("vessel", dest_dir, repo_root) is called BEFORE the
    directory is created or any file is written — mirrors deviation.create_deviation
    step 5 path-escape guard (T-12-08).

    Args:
        item_id: The inventory item row ID (used as filename stem).
        src_path: Source image path (passed to validate_photo_input then prepare_image).
        repo_root: Repo root directory (used for validate_path guard + dest resolution).

    Returns:
        Repo-relative path string suitable for items.photo_path:
        e.g. "data/photos/items/ITEM-1.jpg"

    Raises:
        ValueError: If src_path fails validate_photo_input, or if dest_dir is
                    outside the vessel layer (path-escape attempt).
    """
    repo_root = Path(repo_root).resolve()
    src_path = Path(src_path)

    # Validate the input before touching the destination, and thread the SINGLE
    # resolved Path through to prepare_image so the bytes that get decoded/stored
    # are the bytes that were validated (WR-02 — closes the validate→open TOCTOU
    # where a symlinked src_path could be swapped between validation and decode).
    resolved_src = validate_photo_input(src_path)

    dest_dir = repo_root / "data" / "photos" / "items"
    dest_filename = f"ITEM-{item_id}.jpg"
    dest_path = dest_dir / dest_filename

    # Two-layer path-escape guard (B3) — mirrors inventory.create_item's pair:
    # (a) validate_path("vessel", …) catches '..' traversal and symlinks INSIDE
    #     data/ whose target resolves outside repo_root/data/.
    # (b) resolved_dest.relative_to(resolved_root) catches data/ ITSELF being a
    #     symlink pointing outside the repo — validate_path cannot detect that
    #     because it resolves expected_root through the SAME symlink. Resolve BEFORE
    #     mkdir so a symlink escape never creates the off-repo target directory.
    resolved_root = repo_root.resolve()
    resolved_dest = (repo_root / "data").resolve() / "photos" / "items"
    try:
        resolved_dest.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"Photo path {resolved_dest} escapes repo_root {resolved_root} — "
            "suspected symlink attack on data/ directory."
        ) from None
    validate_path("vessel", dest_dir, repo_root)

    dest_dir.mkdir(parents=True, exist_ok=True)

    prepare_image(resolved_src, dest_path)

    # Return the repo-relative path for storage in items.photo_path.
    return str(dest_path.relative_to(repo_root))


def process_photo(
    src_path: Union[str, Path],
    dest_dir: Union[str, Path],
) -> Path:
    """Convenience wrapper: validate, process, and write a photo to dest_dir.

    This is the primary entry point for the capture CLI and tests. It validates
    the input, applies the full processing pipeline (orientation → resize → GPS
    strip), and writes the result as a JPEG to dest_dir.

    The output filename is the source stem + ".jpg". If the source already has
    a .jpg extension the output filename is unchanged. HEIC inputs produce
    source_stem.jpg output.

    Args:
        src_path: Source image path (any format Pillow supports, including HEIC).
        dest_dir: Directory to write the processed JPEG into.

    Returns:
        Path to the written output JPEG.

    Raises:
        ValueError: If src_path is invalid (missing / not-image / decompression bomb).
    """
    src_path = Path(src_path)
    dest_dir = Path(dest_dir)

    # Validate before any decode.
    resolved_src = validate_photo_input(src_path)

    # Build destination: same stem, always .jpg.
    dest_path = dest_dir / (resolved_src.stem + ".jpg")

    return prepare_image(resolved_src, dest_path)
