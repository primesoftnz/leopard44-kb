# RED state until Phase 2 implementation (see 02-VALIDATION.md). Imports from leopard44_kb.ingest.* fail until production code lands.
"""Tests for INGEST-03/04: PDF structure-aware chunking, image-page detection, page metadata, and layer storage."""
from __future__ import annotations

from pathlib import Path

import pytest

from leopard44_kb.ingest.pdf import is_image_only, parse_pdf

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC = FIXTURES / "synthetic"


# ---------------------------------------------------------------------------
# Unit tests — pure parsing (no DB)
# ---------------------------------------------------------------------------


def test_toc_chunking():
    """PDF with bookmarks produces section_path values derived from TOC headings."""
    toc_pdf = SYNTHETIC / "sample_toc.pdf"
    chunks = parse_pdf(toc_pdf, source_path_str="shared/yanmar/toc.pdf")
    assert len(chunks) > 0, "No chunks returned from TOC PDF"
    section_paths = {c["section_path"] for c in chunks}
    # TOC has 'Chapter 1' and 'Chapter 2' — at least one must appear in section_paths
    assert any(
        "Chapter" in sp for sp in section_paths
    ), f"Expected 'Chapter ...' in section_paths; got {section_paths}"


def test_font_heuristic_chunking():
    """PDF without TOC falls back to font-size heading heuristic for section_path."""
    notoc_pdf = SYNTHETIC / "sample_notoc.pdf"
    chunks = parse_pdf(notoc_pdf, source_path_str="shared/yanmar/notoc.pdf")
    assert len(chunks) > 0, "No chunks returned from no-TOC PDF"
    section_paths = {c["section_path"] for c in chunks}
    # Font heuristic should detect "Cooling System" / "Electrical System" or page-level fallback
    assert len(section_paths) > 0, "section_paths is empty"


def test_image_page_skipped(tmp_path, caplog):
    """Image-only PDF page is detected and skipped with a warning when --ocr is not passed."""
    import logging

    scanned_pdf = SYNTHETIC / "sample_scanned.pdf"
    with caplog.at_level(logging.WARNING, logger="leopard44_kb.ingest.pdf"):
        chunks = parse_pdf(scanned_pdf, source_path_str="data/docs/scanned.pdf", ocr=False)
    # Image-only page should produce no text chunks (or empty content) AND log a warning
    image_skip_warned = any(
        "image" in record.message.lower() or "skip" in record.message.lower() or "ocr" in record.message.lower()
        for record in caplog.records
    )
    # Either a warning was logged or no text chunks were produced
    assert image_skip_warned or all(
        not c["content"].strip() for c in chunks
    ), (
        f"Expected image-skip warning or empty chunks; got {len(chunks)} chunks and log: "
        + "; ".join(r.message for r in caplog.records)
    )


def test_is_image_only_function():
    """is_image_only() returns True for the scanned PDF page and False for a text page."""
    import fitz

    scanned = fitz.open(str(SYNTHETIC / "sample_scanned.pdf"))
    assert is_image_only(scanned[0]), "Expected is_image_only=True for scanned PDF page"

    toc_doc = fitz.open(str(SYNTHETIC / "sample_toc.pdf"))
    assert not is_image_only(toc_doc[0]), "Expected is_image_only=False for text PDF page"


def test_page_metadata_stored(ingest_db, fake_embedder, tmp_path):
    """Integration: every chunk stored after PDF ingest has page_start and page_end set."""
    from leopard44_kb.ingest import ingest_file

    # Place the PDF under shared/ so validate_path accepts it
    shared_dir = tmp_path / "shared" / "yanmar"
    shared_dir.mkdir(parents=True)
    import shutil
    pdf_src = SYNTHETIC / "sample_toc.pdf"
    pdf_dst = shared_dir / "sample_toc.pdf"
    shutil.copy(pdf_src, pdf_dst)

    result = ingest_file(pdf_dst, layer="shared", conn=ingest_db)
    assert result == "ok"

    rows = ingest_db.execute(
        "SELECT page_start, page_end, content FROM chunks"
    ).fetchall()
    assert len(rows) > 0, "No chunks stored after PDF ingest"
    for row in rows:
        assert row["page_start"] is not None, f"page_start is NULL for chunk: {row['content'][:40]!r}"
        assert row["page_end"] is not None, f"page_end is NULL for chunk: {row['content'][:40]!r}"


def test_pdf_layer_shared_stored(ingest_db, fake_embedder, tmp_path):
    """Integration: --layer shared stores source_type='pdf', layer='shared' on source and all chunks, page meta survives."""
    from leopard44_kb.ingest import ingest_file

    shared_dir = tmp_path / "shared" / "yanmar"
    shared_dir.mkdir(parents=True)
    import shutil
    pdf_src = SYNTHETIC / "sample_toc.pdf"
    pdf_dst = shared_dir / "toc_shared.pdf"
    shutil.copy(pdf_src, pdf_dst)

    result = ingest_file(pdf_dst, layer="shared", conn=ingest_db)
    assert result == "ok"

    source = ingest_db.execute(
        "SELECT source_type, layer FROM sources"
    ).fetchone()
    assert source is not None, "No source row stored"
    assert source["source_type"] == "pdf", f"Expected source_type='pdf'; got {source['source_type']!r}"
    assert source["layer"] == "shared", f"Expected layer='shared'; got {source['layer']!r}"

    chunk_layers = ingest_db.execute(
        "SELECT DISTINCT layer FROM chunks"
    ).fetchall()
    assert len(chunk_layers) == 1 and chunk_layers[0]["layer"] == "shared", (
        f"Expected all chunks layer='shared'; got {[r['layer'] for r in chunk_layers]}"
    )

    # page_start/page_end survive into DB
    rows = ingest_db.execute(
        "SELECT page_start, page_end FROM chunks WHERE page_start IS NULL"
    ).fetchall()
    assert len(rows) == 0, f"{len(rows)} chunks have NULL page_start after shared PDF ingest"
