"""PDF parsing: TOC-first structure detection, font-size heuristics, page fallback,
OCR opt-in (INGEST-03/04, D-01/D-02)."""
from __future__ import annotations

import logging
import shutil
import statistics
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting (cl100k_base, lazy-initialised)
# ---------------------------------------------------------------------------

_enc = None  # type: ignore[assignment]


def count_tokens(text: str) -> int:
    """Return the number of cl100k_base tokens in *text*.

    The encoder is lazily initialised on first call and cached at module scope.
    The conftest offline fallback in tests/conftest.py replaces count_tokens on
    leopard44_kb.ingest.text_md, but pdf.py defines its own here per the plan contract.
    If tiktoken is unavailable (no cache, no network), a word-count heuristic is
    used so the test suite stays deterministic offline.
    """
    global _enc
    if _enc is None:
        try:
            import tiktoken

            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Offline / no cache — use a word-count heuristic as an approximation.
            import re

            _enc = None  # keep None to re-try next time if cache appears
            # Inline fallback: words + half of punctuation tokens (matches conftest estimate)
            words = text.split()
            punct = len(re.findall(r"[^\w\s]", text))
            return len(words) + punct // 2
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# OCR / image-page helpers
# ---------------------------------------------------------------------------


def is_image_only(page: fitz.Page) -> bool:
    """Return True if *page* has virtually no text but contains at least one image.

    Threshold: fewer than 10 characters of stripped text AND at least one image.
    The 10-char threshold avoids false-positives from pages that contain only a
    page number or watermark (RESEARCH.md Pattern 1, Anti-Pattern notes).
    """
    text = page.get_text("text").strip()
    return len(text) < 10 and len(page.get_images()) > 0


def check_tesseract() -> None:
    """Raise RuntimeError if the ``tesseract`` binary is not on PATH.

    Called only when ``--ocr`` is passed AND an image-only page is encountered
    (D-01: OCR is opt-in; check at ingest time, not at import time — Pitfall 2).
    """
    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "tesseract not found on PATH — install with:\n"
            "  Linux: sudo apt install tesseract-ocr\n"
            "  macOS: brew install tesseract\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        )


# ---------------------------------------------------------------------------
# Section-splitting helper
# ---------------------------------------------------------------------------


def _split_at_sentence(text: str, token_cap: int) -> list[str]:
    """Split *text* into parts each with at most *token_cap* tokens.

    Prefers sentence boundaries ('. ', '! ', '? ', '\\n') to keep semantic
    coherence. Falls back to hard-splitting at word boundaries if no sentence
    boundary exists within the cap.
    """
    import re

    if count_tokens(text) <= token_cap:
        return [text]

    parts: list[str] = []
    # Split into candidate sentences — keep delimiters attached to the preceding sentence.
    sentences = re.split(r"(?<=[.!?])\s+|(?<=\n)\n+", text)
    # Filter out empty strings from splitting
    sentences = [s for s in sentences if s]

    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if count_tokens(candidate) <= token_cap:
            current = candidate
        else:
            if current:
                parts.append(current)
            # If a single sentence is itself oversized, hard-split by words.
            if count_tokens(sentence) > token_cap:
                words = sentence.split()
                chunk_words: list[str] = []
                for word in words:
                    trial = " ".join(chunk_words + [word])
                    if count_tokens(trial) <= token_cap:
                        chunk_words.append(word)
                    else:
                        if chunk_words:
                            parts.append(" ".join(chunk_words))
                        chunk_words = [word]
                current = " ".join(chunk_words) if chunk_words else ""
            else:
                current = sentence
    if current.strip():
        parts.append(current.strip())
    return parts if parts else [text]


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------


def parse_pdf(
    path: Path,
    source_path_str: str,
    ocr: bool = False,
    token_cap: int = 200,
) -> list[dict]:
    """Parse a PDF into a list of chunk dicts (pure transform — no DB access).

    Structure detection preference order (D-02):
    1. Embedded TOC/bookmarks via ``doc.get_toc()``.
    2. Font-size heading heuristic (per-page median body size, 1.15x threshold).
    3. Page-level fallback when neither TOC nor headings are detected.

    Each returned chunk dict has exactly:
        section_path    : str   — hierarchy joined with ' > '
        content         : str   — extracted text
        page_start      : int   — 1-based first page of this chunk
        page_end        : int   — 1-based last page of this chunk
        section_ordinal : int   — 0-based, resets to 0 on each new section_path
        token_count     : int   — cl100k_base token count

    NOTE: parsers do NOT set a global 'ordinal' key — the writer assigns it.

    Raises:
        ValueError: if the file cannot be opened (fitz.FileDataError re-raised).
        RuntimeError: if ocr=True and tesseract is not on PATH.
    """
    try:
        doc = fitz.open(str(path))
    except fitz.FileDataError as e:
        raise ValueError(f"Cannot open PDF {path}: {e}") from e

    toc = doc.get_toc(simple=True)  # [[level, title, page_1based], ...]

    if toc:
        raw_chunks = _parse_with_toc(doc, toc, ocr)
    else:
        raw_chunks = _parse_without_toc(doc, ocr, token_cap)

    # Apply token-cap splitting and assign section_ordinal.
    return _finalise_chunks(raw_chunks, token_cap)


# ---------------------------------------------------------------------------
# TOC-based parsing
# ---------------------------------------------------------------------------


def _parse_with_toc(
    doc: fitz.Document,
    toc: list[list],
    ocr: bool,
) -> list[dict]:
    """Build raw chunks using the bookmark hierarchy.

    NOTE (v1.0 scope): This mapping is coarse — one chunk per page, labelled
    with the deepest bookmark whose 1-based page number <= current page. A single
    page can hold multiple sections; intra-page finer heading splitting is a later
    refinement deferred beyond v1.0 synthetic-fixture scope.

    section_path is built by joining the title hierarchy levels with ' > ', using
    the same separator as WhatsApp and Markdown parsers.
    """
    raw: list[dict] = []

    for page in doc:
        page_num_1 = page.number + 1  # fitz pages are 0-based; TOC entries are 1-based

        # Handle image-only pages.
        if is_image_only(page):
            if not ocr:
                logger.warning(
                    "Page %d of %s is image-only; skipping (pass --ocr to extract text via OCR)",
                    page_num_1,
                    doc.name,
                )
                continue
            else:
                check_tesseract()
                tp = page.get_textpage_ocr()
                text = page.get_text(textpage=tp)
        else:
            text = page.get_text("text")

        if not text.strip():
            continue

        # Find the deepest bookmark whose page <= current page.
        active_entries: list[list] = [e for e in toc if e[2] <= page_num_1]
        section_path = _build_section_path(active_entries) if active_entries else f"Page {page_num_1}"

        raw.append(
            {
                "section_path": section_path,
                "content": text.strip(),
                "page_start": page_num_1,
                "page_end": page_num_1,
            }
        )

    return raw


def _build_section_path(entries: list[list]) -> str:
    """Return a ' > '-joined section path from the deepest bookmark hierarchy.

    Given entries [[1,'Chapter 1',1],[2,'Impeller',3]], build the path that
    represents the current heading nesting at the deepest matching entry.
    """
    if not entries:
        return ""
    # Take the last entry (deepest by page number); then reconstruct hierarchy
    # by walking backwards to collect ancestors at each level.
    last = entries[-1]
    current_level = last[0]
    path_titles: list[str] = [last[1]]

    # Walk backwards to collect parent levels.
    for entry in reversed(entries[:-1]):
        if entry[0] < current_level:
            path_titles.insert(0, entry[1])
            current_level = entry[0]
            if current_level == 1:
                break

    return " > ".join(path_titles)


# ---------------------------------------------------------------------------
# Font-size heuristic parsing
# ---------------------------------------------------------------------------


def _parse_without_toc(
    doc: fitz.Document,
    ocr: bool,
    token_cap: int,
) -> list[dict]:
    """Build raw chunks using font-size heading heuristics or page fallback.

    Per-page: compute median body span size.  A span is a heading candidate if:
      - font_size > body_size * 1.15  (15% larger), OR
      - font name contains 'Bold' AND span text length < 80 chars.

    If fewer than 3 heading candidates are found across the entire document,
    fall back to page-granularity chunking (Pitfall 7).
    """
    # Collect all span data across pages for heading detection.
    page_data: list[dict] = []  # [{page_num_1, text, spans=[{size,font,text}]}]

    for page in doc:
        page_num_1 = page.number + 1

        if is_image_only(page):
            if not ocr:
                logger.warning(
                    "Page %d of %s is image-only; skipping (pass --ocr to extract text via OCR)",
                    page_num_1,
                    doc.name,
                )
                page_data.append({"page_num_1": page_num_1, "text": "", "spans": [], "is_image": True})
                continue
            else:
                check_tesseract()
                tp = page.get_textpage_ocr()
                text = page.get_text(textpage=tp)
                page_data.append({"page_num_1": page_num_1, "text": text, "spans": [], "is_image": False})
                continue

        spans: list[dict] = []
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    spans.append(
                        {
                            "size": span["size"],
                            "font": span["font"],
                            "text": span["text"],
                        }
                    )

        full_text = page.get_text("text")
        page_data.append({"page_num_1": page_num_1, "text": full_text, "spans": spans, "is_image": False})

    # Count heading candidates across all pages to decide on heuristic vs. page fallback.
    total_heading_candidates = 0
    for pd in page_data:
        if not pd["spans"]:
            continue
        sizes = [s["size"] for s in pd["spans"]]
        if not sizes:
            continue
        body_size = statistics.median(sizes)
        for s in pd["spans"]:
            if _is_heading_span(s, body_size):
                total_heading_candidates += 1

    if total_heading_candidates < 3:
        # Page fallback: each page is one chunk.
        return _page_fallback(page_data)

    # Heuristic: use detected heading spans as section boundaries.
    return _heading_heuristic(page_data)


def _is_heading_span(span: dict, body_size: float) -> bool:
    """Return True if this span looks like a heading."""
    size = span["size"]
    font = span["font"]
    text = span["text"].strip()
    if not text:
        return False
    if size > body_size * 1.15:
        return True
    if "Bold" in font and len(text) < 80:
        return True
    return False


def _heading_heuristic(page_data: list[dict]) -> list[dict]:
    """Build raw chunks keyed on detected heading spans."""
    raw: list[dict] = []
    current_section: Optional[str] = None
    current_content: list[str] = []
    current_page_start: Optional[int] = None
    current_page_end: Optional[int] = None

    for pd in page_data:
        page_num_1 = pd["page_num_1"]
        if pd.get("is_image") or not pd["text"].strip():
            continue

        spans = pd["spans"]
        if not spans:
            # No span data — treat whole page text as continuation of current section.
            if current_section is None:
                current_section = f"Page {page_num_1}"
                current_page_start = page_num_1
            current_content.append(pd["text"].strip())
            current_page_end = page_num_1
            continue

        sizes = [s["size"] for s in spans]
        body_size = statistics.median(sizes)

        # Walk the spans to detect headings and collect body text.
        # We gather the page's full text by logical sections detected on this page.
        # Simple approach: treat the first heading span on the page as the section boundary.
        page_headings = [s for s in spans if _is_heading_span(s, body_size)]
        page_body_spans = [s for s in spans if not _is_heading_span(s, body_size)]
        body_text = " ".join(s["text"].strip() for s in page_body_spans if s["text"].strip())

        if page_headings:
            # Flush current section before starting a new one.
            if current_section is not None and current_content:
                content = "\n".join(current_content).strip()
                if content:
                    raw.append(
                        {
                            "section_path": current_section,
                            "content": content,
                            "page_start": current_page_start,
                            "page_end": current_page_end,
                        }
                    )

            # New section: use the first heading text on this page.
            heading_text = page_headings[0]["text"].strip()
            current_section = heading_text
            current_content = [body_text] if body_text else []
            current_page_start = page_num_1
            current_page_end = page_num_1
        else:
            # No heading on this page — continuation of current section.
            if current_section is None:
                current_section = f"Page {page_num_1}"
                current_page_start = page_num_1
            if body_text:
                current_content.append(body_text)
            current_page_end = page_num_1

    # Flush the last section.
    if current_section is not None and current_content:
        content = "\n".join(current_content).strip()
        if content:
            raw.append(
                {
                    "section_path": current_section,
                    "content": content,
                    "page_start": current_page_start,
                    "page_end": current_page_end,
                }
            )

    return raw


def _page_fallback(page_data: list[dict]) -> list[dict]:
    """Build raw chunks at page granularity (fallback when no structure detected)."""
    raw: list[dict] = []
    for pd in page_data:
        page_num_1 = pd["page_num_1"]
        text = pd["text"].strip()
        if not text:
            continue
        raw.append(
            {
                "section_path": f"Page {page_num_1}",
                "content": text,
                "page_start": page_num_1,
                "page_end": page_num_1,
            }
        )
    return raw


# ---------------------------------------------------------------------------
# Finalise: token-cap splitting and section_ordinal assignment
# ---------------------------------------------------------------------------


def _finalise_chunks(raw: list[dict], token_cap: int) -> list[dict]:
    """Apply token-cap splitting and assign section_ordinal to each chunk.

    section_ordinal is section-relative: resets to 0 whenever section_path changes.
    Does NOT emit a global 'ordinal' key — the writer assigns that.
    """
    result: list[dict] = []
    section_ordinal_counter: dict[str, int] = {}

    for raw_chunk in raw:
        section_path = raw_chunk["section_path"]
        content = raw_chunk["content"]
        page_start = raw_chunk["page_start"]
        page_end = raw_chunk["page_end"]

        parts = _split_at_sentence(content, token_cap)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            ordinal = section_ordinal_counter.get(section_path, 0)
            section_ordinal_counter[section_path] = ordinal + 1
            result.append(
                {
                    "section_path": section_path,
                    "content": part,
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_ordinal": ordinal,
                    "token_count": count_tokens(part),
                }
            )

    return result
