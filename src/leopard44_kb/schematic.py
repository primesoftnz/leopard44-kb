"""Schematic rendering service for Leopard 44 KB (Phase 9 / ZONE-02 / D-01..D-03).

SETUP-TIME ONLY — import boundary:
  This module uses fitz + httpx + stdlib only. It must NEVER import leopard44_kb.web,
  leopard44_kb.app, or leopard44_kb.answer so it stays off the offline serve/query path.
  The cloud-vision suggester (suggest_pages) is opt-in, online-only, and runs
  only from the CLI; the --pages render path needs no API key.

Public API:
  render_pages(pdf_path, page_numbers, output_dir) -> list[Path]
  parse_page_spec(spec, page_count=None) -> list[int]
  suggest_pages(pdf_path) -> list[int]
"""
from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — already a dependency, same as ingest/pdf.py

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# render_pages: core rasterisation (D-02, ZONE-02)
# ---------------------------------------------------------------------------


def render_pages(
    pdf_path: Path,
    page_numbers: list[int],
    output_dir: Path,
) -> list[Path]:
    """Render selected 1-based page numbers from pdf_path to PNG files in output_dir.

    Page numbers are 1-based (matching the manual's pp. references).  fitz uses
    0-based indices internally, so doc[page_num_1 - 1] is the correct lookup.

    Returns the list of written PNG paths in the same order as page_numbers.

    Raises:
        ValueError: If the PDF cannot be opened.
        IndexError: If a page number is out of range for the document.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF {pdf_path}: {exc}") from exc

    written: list[Path] = []
    try:
        for page_num_1 in page_numbers:
            page = doc[page_num_1 - 1]  # fitz is 0-based; pdf.py uses page.number + 1 for inverse
            pix = page.get_pixmap(dpi=150)  # dpi kwarg verified on fitz 1.27.2.3
            png_name = f"page_{page_num_1:03d}.png"  # zero-padded 3 digits (Claude's Discretion)
            out_path = output_dir / png_name
            pix.save(str(out_path))  # pix.save() detects PNG from .png extension
            written.append(out_path)
    finally:
        doc.close()
    return written


# ---------------------------------------------------------------------------
# parse_page_spec: validating page range / list parser (review fix)
# ---------------------------------------------------------------------------


def parse_page_spec(spec: str, page_count: Optional[int] = None) -> list[int]:
    """Parse a page specification string into a sorted unique list of 1-based page ints.

    Accepted formats:
      "61-89"       — contiguous range (inclusive)
      "61,62,65"    — explicit list
      "61-63,65"    — mix of range and list

    Validation (review fix — T-09-16):
      Raises ValueError for:
        - Non-integer tokens (e.g. "abc", "1.5")
        - Page 0 or negative page numbers
        - Reversed ranges (e.g. "63-61")
        - Any page > page_count (when page_count is provided)

    Returns:
        Sorted, deduplicated list of 1-based page integers.
    """
    pages: set[int] = set()

    # Split on commas to get individual tokens or range expressions
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if not tokens:
        raise ValueError(f"Empty page spec: {spec!r}")

    for token in tokens:
        if "-" in token:
            # Potential range "start-end"; but could be negative like "-3" or "1--3"
            # Split at the LAST hyphen that follows at least one digit on the left
            parts = token.split("-")
            # Reconstruct: handle cases where "-" is used as negation
            # A range is "N-M" where both N and M are positive integers.
            # Any negation sign means a negative number which is invalid.
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                start_str, end_str = parts[0].strip(), parts[1].strip()
                # Validate integer-ness
                try:
                    start = int(start_str)
                    end = int(end_str)
                except ValueError:
                    raise ValueError(
                        f"Non-integer in page range {token!r} — pages must be whole numbers."
                    )
                # Check for negatives or zero
                if start <= 0:
                    raise ValueError(
                        f"Page number {start} is invalid — pages must be >= 1 (1-based)."
                    )
                if end <= 0:
                    raise ValueError(
                        f"Page number {end} is invalid — pages must be >= 1 (1-based)."
                    )
                # Check for reversed range
                if end < start:
                    raise ValueError(
                        f"Reversed page range {token!r}: end ({end}) < start ({start}). "
                        f"Use '{end}-{start}' to reverse."
                    )
                for p in range(start, end + 1):
                    pages.add(p)
            elif len(parts) == 3 and parts[0] == "" and parts[1].strip():
                # Looks like "-N" — a negative page number
                raise ValueError(
                    f"Negative page number in spec {token!r} — pages must be >= 1."
                )
            else:
                # Multi-hyphen or ambiguous — try as integer first
                try:
                    p = int(token)
                except ValueError:
                    raise ValueError(
                        f"Cannot parse page spec token {token!r} — expected integer or 'N-M' range."
                    )
                if p <= 0:
                    raise ValueError(
                        f"Page number {p} is invalid — pages must be >= 1 (1-based)."
                    )
                pages.add(p)
        else:
            # Simple integer token
            # Check for float (e.g. "1.5")
            if "." in token:
                raise ValueError(
                    f"Non-integer page number {token!r} — pages must be whole numbers."
                )
            try:
                p = int(token)
            except ValueError:
                raise ValueError(
                    f"Non-integer page spec token {token!r} — expected a whole number."
                )
            if p <= 0:
                raise ValueError(
                    f"Page number {p} is invalid — pages must be >= 1 (1-based)."
                )
            pages.add(p)

    # Validate against document page count if provided
    if page_count is not None:
        out_of_range = [p for p in pages if p > page_count]
        if out_of_range:
            raise ValueError(
                f"Page(s) {sorted(out_of_range)!r} exceed the document page count "
                f"({page_count}). The document has pages 1–{page_count}."
            )

    return sorted(pages)


# ---------------------------------------------------------------------------
# suggest_pages: cloud-vision page-suggester (D-03, opt-in, online-only)
# ---------------------------------------------------------------------------


def suggest_pages(pdf_path: Path) -> list[int]:
    """Use a cloud-vision API to suggest which pages contain schematics/GA drawings.

    Provider selection (Anthropic primary; Gemini/OpenAI behind a documented branch):
      1. ANTHROPIC_API_KEY — fully implemented (Anthropic messages + vision).
      2. GOOGLE_API_KEY — raises NotImplementedError (not yet wired).
      3. OPENAI_API_KEY — raises NotImplementedError (not yet wired).
      4. No key set — raises RuntimeError with instructions to set a key or use --pages.

    API keys are read at CALL TIME only (never at import time, never logged).

    Sends downscaled thumbnails (dpi=72) to the vision API to reduce token cost.
    Parses the model's response with a regex over raw text — models emit conversational
    chatter even when asked for numbers only (robust parsing per review concern).

    Returns:
        Sorted list of 1-based page numbers suggested as schematics.

    Raises:
        RuntimeError: If no API key is set, or if the API call fails (network,
                      timeout, HTTP error).
        NotImplementedError: If GOOGLE_API_KEY or OPENAI_API_KEY is set but the
                             corresponding vision provider is not yet wired.
    """
    # ---- Provider selection (keys read at call time) ----
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not anthropic_key and not google_key and not openai_key:
        raise RuntimeError(
            "No cloud-vision API key found. Set ANTHROPIC_API_KEY (primary) "
            "or pass --pages to specify schematic pages without a network call."
        )

    if google_key and not anthropic_key:
        raise NotImplementedError(
            "Gemini vision suggester not yet wired — set ANTHROPIC_API_KEY "
            "or pass --pages to skip the cloud suggester."
        )

    if openai_key and not anthropic_key and not google_key:
        raise NotImplementedError(
            "OpenAI vision suggester not yet wired — set ANTHROPIC_API_KEY "
            "or pass --pages to skip the cloud suggester."
        )

    # ---- Anthropic primary path ----
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF {pdf_path}: {exc}") from exc

    page_count = len(doc)

    # Build vision content: one base64-encoded thumbnail per page (dpi=72 for cost)
    content: list[dict] = []
    try:
        for i in range(page_count):
            pix = doc[i].get_pixmap(dpi=72)
            img_bytes = pix.tobytes("png")
            b64 = base64.standard_b64encode(img_bytes).decode("ascii")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            })
    finally:
        doc.close()

    content.append({
        "type": "text",
        "text": (
            f"This PDF has {page_count} pages. I have sent you all pages as images "
            f"(in order, starting from page 1). "
            "Which page numbers contain schematic diagrams, GA (general arrangement) drawings, "
            "or boat layout plans — not just text pages? "
            "List ONLY the page numbers separated by commas, e.g. '61, 62, 65'. "
            "If you are unsure, include the page."
        ),
    })

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            "Cloud-vision API not reachable — check network. "
            "Use --pages to specify pages without a network call."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            "Cloud-vision API timed out. "
            "Use --pages to specify pages without a network call."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Cloud-vision API returned HTTP {exc.response.status_code}. "
            "Use --pages to specify pages without a network call."
        ) from exc

    # ---- Robust response parsing (review + Gemini concern) ----
    # Models emit conversational text even when asked for numbers — extract
    # all integer runs from the raw response and filter to valid page range.
    body = response.json()
    raw_text = ""
    for block in body.get("content", []):
        if block.get("type") == "text":
            raw_text += block.get("text", "")

    # Extract all integer runs; filter to 1-based pages within doc range
    found = [int(m) for m in re.findall(r"\b\d+\b", raw_text)]
    suggested = sorted(set(p for p in found if 1 <= p <= page_count))

    logger.info("suggest_pages: API response=%r → suggested pages=%r", raw_text[:200], suggested)
    return suggested
