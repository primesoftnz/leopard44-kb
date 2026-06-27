# RED state until Phase 3 Wave 2/3 implementation. The ask command is currently a stub.
# Collection may succeed but test execution fails until 03-02 (retrieve.py) and
# 03-03 (answer.py) land. ModuleNotFoundError on leopard44_kb.retrieve/answer is expected.
"""Tests for QUERY-01..03, D-09: ask CLI command — CliRunner integration tests.

Per-requirement verification map source: .planning/phases/03-query-engine/03-VALIDATION.md
Review fixes covered: #4 (out-of-range citation stripped), #7 (layer-leak guards).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlite_vec
from typer.testing import CliRunner

from leopard44_kb.cli import app
from leopard44_kb.schema import apply_migrations
from tests._corpus import seed_corpus

runner = CliRunner()


def _seed_db(db_path: Path, *, include_inactive: bool = False) -> None:
    """Bootstrap a file-backed test DB with the canonical retrieval corpus.

    Writes to a real file so the CLI (which calls open_db() with the L44_DB
    env var) can access it. The corpus itself lives in tests/_corpus.py and is
    shared with the in-memory retrieval_db fixture (IN-05), so the CLI and unit
    paths cannot drift. ``include_inactive`` seeds an additional is_active=0
    chunk for the CR-01 filter regression at the CLI level (IN-06).
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)

    seed_corpus(conn, include_inactive=include_inactive)

    conn.close()


# ---------------------------------------------------------------------------
# QUERY-01: Basic ask command
# ---------------------------------------------------------------------------


def test_ask_returns_answer(monkeypatch, fake_embedder, fake_generator, tmp_path):
    """ask command exits 0 and returns non-empty output."""
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    result = runner.invoke(app, ["ask", "what is the impeller interval?"])
    assert result.exit_code == 0, (
        f"Expected exit 0; got {result.exit_code}: {result.output}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert len(combined.strip()) > 0, "Expected non-empty output from ask command"


def test_ask_no_longer_a_stub(monkeypatch, fake_embedder, fake_generator, tmp_path):
    """ask command does not print 'Not yet implemented' and does not exit with code 2.

    Mirrors test_ingest_no_longer_a_stub from test_sources_cli.py (Phase 3 replaces stub).
    """
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    result = runner.invoke(app, ["ask", "what is the impeller interval?"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Not yet implemented" not in combined, (
        f"Stub message still present: {combined!r}"
    )
    assert result.exit_code != 2, (
        f"Stub exit code 2 still returned; ask is not implemented yet"
    )


# ---------------------------------------------------------------------------
# D-09: Stream order — tokens precede citation block
# ---------------------------------------------------------------------------


def test_stream_then_citations(monkeypatch, fake_embedder, fake_generator, tmp_path):
    """Streamed answer tokens appear before the 'Sources:' citation block in output."""
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    result = runner.invoke(app, ["ask", "what is the impeller interval?"])
    combined = (result.stdout or "") + (result.stderr or "")

    assert "Sources:" in combined, f"Expected 'Sources:' block in output: {combined!r}"
    # The fake generator yields "The impeller interval is 200 hours [1]. ..."
    # Answer tokens should appear before the Sources block.
    sources_pos = combined.find("Sources:")
    answer_pos = combined.find("impeller")  # first answer token
    assert answer_pos != -1, f"Expected answer token 'impeller' in output: {combined!r}"
    assert answer_pos < sources_pos, (
        f"Answer tokens (pos {answer_pos}) should precede Sources block (pos {sources_pos})"
    )


# ---------------------------------------------------------------------------
# QUERY-03 / review fix #7: Layer-leak guards
# ---------------------------------------------------------------------------


def test_layer_shared_no_vessel_citations(monkeypatch, fake_embedder, fake_generator, tmp_path):
    """Review fix #7: --layer shared produces no 'vessel:' lines in the Sources block."""
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    result = runner.invoke(app, ["ask", "what is the impeller interval?", "--layer", "shared"])
    combined = (result.stdout or "") + (result.stderr or "")

    # IN-01: assert the precondition so the guarded check is always exercised —
    # a vacuous pass (retrieval returned nothing) would not prove the layer guard.
    assert "Sources:" in combined, (
        f"Expected a Sources block (retrieval must return shared chunks); got: {combined!r}"
    )
    # Extract Sources block and check no vessel: lines appear
    sources_section = combined[combined.find("Sources:"):]
    lines_after = sources_section.splitlines()
    vessel_lines = [l for l in lines_after if l.strip().startswith("vessel:")]
    assert vessel_lines == [], (
        f"No vessel: citations should appear with --layer shared; got: {vessel_lines}"
    )


def test_layer_vessel_no_shared_citations(monkeypatch, fake_embedder, fake_generator, tmp_path):
    """Review fix #7: --layer vessel produces no 'shared:' lines in the Sources block."""
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    result = runner.invoke(app, ["ask", "what is the impeller interval?", "--layer", "vessel"])
    combined = (result.stdout or "") + (result.stderr or "")

    # IN-01: the layer-isolation guarantee must hold unconditionally — whether
    # retrieval returns a vessel-only Sources block OR refuses below the D-07
    # floor (the fixture's vessel chunk has a deliberately orthogonal vector for
    # the pull-in test, so vessel-only retrieval legitimately refuses). In NO
    # case may a `shared:` citation appear. This assertion runs every time, so
    # the test is no longer vacuous.
    assert result.exit_code == 0, f"ask --layer vessel exited {result.exit_code}: {combined!r}"
    shared_lines = [l for l in combined.splitlines() if l.strip().startswith("shared:")]
    assert shared_lines == [], (
        f"No shared: citations should appear with --layer vessel; got: {shared_lines}"
    )


# ---------------------------------------------------------------------------
# QUERY-02 / review fix #4: Out-of-range citation stripped from Sources block
# ---------------------------------------------------------------------------


def test_out_of_range_citation_stripped(monkeypatch, fake_embedder, out_of_range_generator, tmp_path):
    """Review fix #4: out-of-range [9] emitted by the model is handled before rendering.

    D-09 streaming is preserved — the live answer text may contain [9] which cannot
    be un-sent. Instead we assert:
    (a) a soft warning naming the out-of-range citation appears (stderr or stdout), and
    (b) the code-rendered 'Sources:' block contains no [9] entry while valid [1] IS listed.
    """
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    result = runner.invoke(app, ["ask", "what is the impeller interval?"])
    combined = (result.stdout or "") + (result.stderr or "")

    # (a) Some warning about the out-of-range citation should appear
    assert "[9]" in combined or "out-of-range" in combined.lower() or "invalid" in combined.lower(), (
        f"Expected a warning about out-of-range [9] citation; got: {combined!r}"
    )

    # (b) The Sources: block should not list a [9] entry — only valid citations.
    # IN-01: assert the precondition so the guarded checks are always exercised.
    assert "Sources:" in combined, (
        f"Expected a Sources block (retrieval must return at least [1]); got: {combined!r}"
    )
    sources_section = combined[combined.find("Sources:"):]
    lines_after = sources_section.splitlines()
    source_entries = [l.strip() for l in lines_after if l.strip().startswith("[")]
    nine_entries = [l for l in source_entries if l.startswith("[9]")]
    assert nine_entries == [], (
        f"[9] should not appear as a rendered Sources entry; got: {nine_entries}"
    )
    # The valid [1] source should still be listed
    one_entries = [l for l in source_entries if l.startswith("[1]")]
    assert one_entries, (
        f"[1] (valid citation) should still appear in Sources block; got entries: {source_entries}"
    )


# ---------------------------------------------------------------------------
# CR-01 / IN-06: inactive chunks never reach the rendered Sources block via CLI
# ---------------------------------------------------------------------------


def test_inactive_chunk_excluded_from_cli_sources(monkeypatch, fake_embedder, fake_generator, tmp_path):
    """IN-06: an is_active=0 chunk that WOULD match the query never appears in CLI output.

    The CR-01 is_active filter is verified at the retrieve() level in test_retrieve.py,
    but the file-backed CLI seed previously seeded only active chunks, so the full
    ask -> retrieve -> render path never exercised the filter. Here we seed an inactive
    chunk (source 'Stale Impeller Note', content 'replace the impeller every 999 hours')
    whose vector points the same direction as the top active chunk — it would rank highly
    if the filter were bypassed — and assert its source/content never reaches the user.
    """
    db_path = tmp_path / "test.db"
    _seed_db(db_path, include_inactive=True)
    monkeypatch.setenv("L44_DB", str(db_path))

    result = runner.invoke(app, ["ask", "what is the impeller interval?"])
    combined = (result.stdout or "") + (result.stderr or "")

    assert result.exit_code == 0, f"ask exited {result.exit_code}: {combined!r}"
    # Precondition: retrieval must return the active corpus (otherwise the filter
    # assertion below would pass vacuously).
    assert "Sources:" in combined, (
        f"Expected a Sources block from the active corpus; got: {combined!r}"
    )
    # The inactive chunk's distinctive source title, path, and content must never appear.
    assert "Stale Impeller Note" not in combined, (
        f"Inactive source title leaked into CLI output: {combined!r}"
    )
    assert "stale_impeller" not in combined, (
        f"Inactive source path leaked into CLI output: {combined!r}"
    )
    assert "999 hours" not in combined, (
        f"Inactive chunk content leaked into CLI output: {combined!r}"
    )
