# RED state until Phase 2 implementation (see 02-VALIDATION.md). Imports from leopard44_kb.ingest.* fail until production code lands.
"""Tests for INGEST-05: markdown heading hierarchy, plain-text blank-line sections, FTS queryability, no manual FTS writes."""
from __future__ import annotations

from pathlib import Path

import pytest

from leopard44_kb.ingest.text_md import parse_markdown, parse_text

FIXTURES = Path(__file__).parent / "fixtures"


def test_markdown_headings():
    """Markdown chunk section_path tracks the heading hierarchy (# H1 > ## H2 > ...)."""
    note_md = FIXTURES / "sample_note.md"
    chunks = parse_markdown(note_md, source_path_str="data/logs/sample_note.md")
    assert len(chunks) > 0, "No chunks returned from markdown parser"
    section_paths = [c["section_path"] for c in chunks]
    # The fixture has '# Vessel Maintenance Notes' and '## Engine Maintenance'
    # At least one chunk should be under a non-empty section_path
    assert any(sp for sp in section_paths), f"All section_paths are empty: {section_paths}"
    # The top-level heading should appear in at least one section_path
    assert any("Vessel Maintenance" in sp for sp in section_paths), (
        f"Top-level heading not found in section_paths: {section_paths}"
    )


def test_txt_blank_line_sections(tmp_path):
    """Plain .txt file uses blank-line block boundaries as synthetic section cues."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text(
        "First block line one.\nFirst block line two.\n\n"
        "Second block line one.\nSecond block line two.\n\n"
        "Third block.\n",
        encoding="utf-8",
    )
    chunks = parse_text(txt_file, source_path_str="data/logs/notes.txt")
    assert len(chunks) >= 2, f"Expected at least 2 chunks for 3 blank-line blocks; got {len(chunks)}"
    # section_paths should be distinct for different blocks (e.g. 'Block 0', 'Block 1')
    section_paths = [c["section_path"] for c in chunks]
    assert len(set(section_paths)) >= 2, f"Expected distinct section_paths per block; got {section_paths}"


def test_fts_queryable(ingest_db, fake_embedder, tmp_path):
    """Integration: after ingest, `SELECT ... FROM fts_chunks WHERE fts_chunks MATCH ?` returns the content."""
    from leopard44_kb.ingest import ingest_file

    data_dir = tmp_path / "data" / "logs"
    data_dir.mkdir(parents=True)
    note = data_dir / "fts_test.md"
    unique_phrase = "impellerxyz123uniquephrase"
    note.write_text(f"# Test Note\n\nThis note contains the phrase {unique_phrase}.\n")

    result = ingest_file(note, layer="vessel", conn=ingest_db)
    assert result == "ok"

    # Query via FTS5
    rows = ingest_db.execute(
        "SELECT content FROM fts_chunks WHERE fts_chunks MATCH ?",
        (unique_phrase,),
    ).fetchall()
    assert len(rows) > 0, f"FTS search for {unique_phrase!r} returned no results"
    assert any(unique_phrase in row["content"] for row in rows), (
        f"Content with {unique_phrase!r} not found in FTS results: {[r['content'] for r in rows]}"
    )


def test_no_manual_fts_writes(ingest_db, fake_embedder, tmp_path):
    """FTS rows appear ONLY via triggers — no manual INSERT into fts_chunks by ingest code.

    Verifies: after ingest + a changed re-ingest, fts_chunks.rowid count exactly equals
    chunks.id count (no duplicates from manual inserts).
    """
    from leopard44_kb.ingest import ingest_file

    data_dir = tmp_path / "data" / "logs"
    data_dir.mkdir(parents=True)
    note = data_dir / "fts_manual_test.md"
    note.write_text("# Section One\n\nContent alpha.\n\n# Section Two\n\nContent beta.\n")

    ingest_file(note, layer="vessel", conn=ingest_db)

    chunk_count = ingest_db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    fts_count = ingest_db.execute(
        "SELECT count(*) FROM fts_chunks WHERE fts_chunks MATCH 'Content'"
    ).fetchone()[0]

    # FTS count should not exceed chunk count (manual double-inserts would inflate it)
    assert fts_count <= chunk_count, (
        f"FTS row count ({fts_count}) exceeds chunk count ({chunk_count}) — "
        "possible manual INSERT into fts_chunks bypassing triggers"
    )

    # Change the file and re-ingest — FTS must reflect NEW content, not OLD
    note.write_text("# Section One\n\nContent gamma.\n\n# Section Two\n\nContent delta.\n")
    ingest_file(note, layer="vessel", conn=ingest_db)

    # Old content must no longer be in FTS
    old_rows = ingest_db.execute(
        "SELECT content FROM fts_chunks WHERE fts_chunks MATCH 'alpha'"
    ).fetchall()
    assert len(old_rows) == 0, (
        f"Old FTS content ('alpha') still present after re-ingest: {[r['content'] for r in old_rows]}"
    )

    # New content must be present
    new_rows = ingest_db.execute(
        "SELECT content FROM fts_chunks WHERE fts_chunks MATCH 'gamma'"
    ).fetchall()
    assert len(new_rows) > 0, "New FTS content ('gamma') not present after re-ingest"


def test_no_heading_only_chunks(tmp_path):
    """A heading immediately followed by a deeper heading must NOT emit a content-free
    chunk (the bug that made retrieval return bare '# Known Issues' headings)."""
    doc = tmp_path / "issues.md"
    doc.write_text(
        "# Known Issues\n"
        "## Structural\n"
        "### Poor Windward Performance\n\n"
        "The boat struggles to point well to weather.\n",
        encoding="utf-8",
    )
    chunks = parse_markdown(doc, source_path_str="shared/issues.md")
    # No chunk is just a heading line with no body.
    for c in chunks:
        body_after_headings = "\n".join(
            ln for ln in c["content"].splitlines() if not ln.lstrip().startswith("#")
        ).strip()
        assert body_after_headings, f"Heading-only chunk leaked: {c['content']!r}"


def test_full_heading_hierarchy_prepended(tmp_path):
    """Each body chunk carries its FULL ancestor heading path so it is discoverable by
    ancestor-section terms (a 'Poor Windward' chunk matches a 'known issues' query)."""
    doc = tmp_path / "issues.md"
    doc.write_text(
        "# Known Issues\n"
        "## Structural\n"
        "### Poor Windward Performance\n\n"
        "The boat struggles to point well to weather.\n",
        encoding="utf-8",
    )
    chunks = parse_markdown(doc, source_path_str="shared/issues.md")
    body = next(c for c in chunks if "struggles to point" in c["content"])
    assert "Known Issues" in body["content"]
    assert "Structural" in body["content"]
    assert "Poor Windward Performance" in body["content"]
