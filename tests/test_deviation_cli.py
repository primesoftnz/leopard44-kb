"""RED tests for DEV-01: `l44 deviation add` review-before-commit CLI.

Per-requirement verification map source:
  .planning/phases/11-factory-deviation-log/11-VALIDATION.md

Nyquist discipline: leopard44_kb.deviation is NEVER imported at module top.
Every behavioral test imports leopard44_kb.deviation INSIDE the test body so:
  - Collection always succeeds (zero ERROR collecting lines)
  - Tests FAIL (ModuleNotFoundError from body-import) until 11-02 creates the
    `deviation` sub-app — the correct RED→GREEN signal
These tests assert the `l44 deviation add` CLI Accept/Edit/Abort review flow
and the clear-an-optional-field sentinel ("-").
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from leopard44_kb.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helper: bootstrap a temp DB for CLI tests
# ---------------------------------------------------------------------------


def _bootstrap_db(db_path):
    """Bootstrap a file-backed DB with sqlite-vec and migrations applied."""
    import sqlite3

    import sqlite_vec

    from leopard44_kb.schema import apply_migrations

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    conn.close()


def _open_db(db_path):
    """Open a bootstrapped DB for direct query verification."""
    import sqlite3

    import sqlite_vec

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


# ---------------------------------------------------------------------------
# (1) `deviation add` shows extracted-fields table then shows the review prompt
# ---------------------------------------------------------------------------


def test_deviation_add_shows_review_prompt(monkeypatch, fake_embedder, tmp_path):
    """deviation add shows the extracted-fields table and the Accept/Edit/Abort prompt.

    Goes GREEN when 11-02 creates the `deviation` sub-app with review flow.
    """
    import leopard44_kb.deviation as dev  # body-import: FAILS with ModuleNotFoundError until 11-02
    from leopard44_kb.deviation import DeviationExtraction

    fixed = DeviationExtraction(
        component="windlass",
        factory_spec="12V Muir 1200W",
        as_built="12V Maxwell 1000W",
        reason="replacement after failure",
        date_noted="2024-01-10",
    )
    monkeypatch.setattr(dev, "extract_fields", lambda text: fixed)

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    result = runner.invoke(
        app, ["deviation", "add", "windlass swapped"], input="q\n"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Accept" in combined or "accept" in combined.lower(), (
        f"Expected 'Accept' in output; got: {combined!r}"
    )
    assert "Edit" in combined or "edit" in combined.lower(), (
        f"Expected 'Edit' in output; got: {combined!r}"
    )
    assert "Abort" in combined or "abort" in combined.lower(), (
        f"Expected 'Abort' in output; got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# (2) Input "q" aborts with exit 0 and writes NOTHING to the DB
# ---------------------------------------------------------------------------


def test_deviation_add_abort_writes_nothing(monkeypatch, fake_embedder, tmp_path):
    """deviation add with input 'q' aborts; exits 0; no deviation row written.

    Goes GREEN when 11-02 creates the `deviation` sub-app with Abort path.
    """
    import leopard44_kb.deviation as dev  # body-import: FAILS with ModuleNotFoundError until 11-02
    from leopard44_kb.deviation import DeviationExtraction

    fixed = DeviationExtraction(component="windlass")
    monkeypatch.setattr(dev, "extract_fields", lambda text: fixed)

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    result = runner.invoke(
        app, ["deviation", "add", "windlass swapped"], input="q\n"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 on Abort; got {result.exit_code}: {combined!r}"
    )

    # Verify no deviations row was written
    conn = _open_db(db_path)
    count = conn.execute("SELECT COUNT(*) FROM deviations").fetchone()[0]
    conn.close()
    assert count == 0, f"Expected 0 deviations rows after Abort; got {count}"


# ---------------------------------------------------------------------------
# (3) Input "a" (Accept) writes the deviation
# ---------------------------------------------------------------------------


def test_deviation_add_accept_writes_deviation(monkeypatch, fake_embedder, tmp_path):
    """deviation add with input 'a' commits the deviation and exits 0.

    Goes GREEN when 11-02 creates the `deviation` sub-app with Accept path.
    """
    import leopard44_kb.deviation as dev  # body-import: FAILS with ModuleNotFoundError until 11-02
    from leopard44_kb.deviation import DeviationExtraction

    fixed = DeviationExtraction(
        component="windlass",
        factory_spec="12V Muir 1200W",
        as_built="12V Maxwell 1000W",
    )
    monkeypatch.setattr(dev, "extract_fields", lambda text: fixed)

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    result = runner.invoke(
        app, ["deviation", "add", "windlass swapped"], input="a\n"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 on Accept; got {result.exit_code}: {combined!r}"
    )

    # Verify one deviations row was written
    conn = _open_db(db_path)
    count = conn.execute("SELECT COUNT(*) FROM deviations").fetchone()[0]
    conn.close()
    assert count == 1, f"Expected 1 deviations row after Accept; got {count}"


# ---------------------------------------------------------------------------
# (4) --yes skips the prompt and commits as-is
# ---------------------------------------------------------------------------


def test_deviation_add_yes_skips_review(monkeypatch, fake_embedder, tmp_path):
    """`deviation add --yes` commits without prompting and exits 0.

    Goes GREEN when 11-02 creates the `deviation` sub-app with --yes path.
    """
    import leopard44_kb.deviation as dev  # body-import: FAILS with ModuleNotFoundError until 11-02
    from leopard44_kb.deviation import DeviationExtraction

    fixed = DeviationExtraction(component="windlass")
    monkeypatch.setattr(dev, "extract_fields", lambda text: fixed)

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app, ["deviation", "add", "windlass swapped", "--yes"]
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 with --yes; got {result.exit_code}: {combined!r}"
    )
    for review_word in ("Accept", "Edit", "Abort"):
        assert review_word not in combined, (
            f"Review prompt word '{review_word}' appeared despite --yes: {combined!r}"
        )


# ---------------------------------------------------------------------------
# (5) Non-TTY without --yes exits non-zero with "no terminal for review"
# ---------------------------------------------------------------------------


def test_deviation_add_no_tty_no_yes_fails(monkeypatch, tmp_path):
    """Non-TTY stdin without --yes exits non-zero with 'no terminal for review'.

    CliRunner stdin is not a TTY; _stdin_isatty is NOT patched here.
    Goes GREEN when 11-02 implements the strict D-10 gate for deviation add.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["deviation", "add", "some text"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code != 0, (
        f"Expected non-zero exit for non-TTY no-yes; got {result.exit_code}: {combined!r}"
    )
    assert "no terminal for review" in combined.lower(), (
        f"Expected 'no terminal for review' message; got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# (6) CLEAR-AN-OPTIONAL-FIELD: "-" sentinel clears an optional field to NULL
# ---------------------------------------------------------------------------


def test_deviation_add_edit_clear_optional_field(monkeypatch, fake_embedder, tmp_path):
    """Edit path: typing "-" for an optional field (reason) clears it to NULL/empty.

    Review finding 6: typing "-" CLEARS the field (sets NULL/empty), whereas Enter
    keeps the existing extracted value. This is distinct from Edit-and-override.

    Goes GREEN when 11-02 implements the clear-field "-" sentinel in the edit loop.
    """
    import leopard44_kb.deviation as dev  # body-import: FAILS with ModuleNotFoundError until 11-02
    from leopard44_kb.deviation import DeviationExtraction

    fixed = DeviationExtraction(
        component="windlass",
        factory_spec="12V Muir 1200W",
        as_built="12V Maxwell 1000W",
        reason="replacement after failure",
        date_noted="2024-01-10",
    )
    monkeypatch.setattr(dev, "extract_fields", lambda text: fixed)

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    # fake_deviation_extractor returns reason="replacement after failure"
    # Edit: choose 'e', then when prompted for 'reason' enter '-' to clear it,
    # then Enter for all remaining fields to keep them, then 'a' to accept.
    # Input sequence: e → (component: keep) → (factory_spec: keep) → (as_built: keep)
    #                 → (reason: -) → (date_noted: keep) → (notes: keep) → a
    user_input = "e\n\n\n\n-\n\n\na\n"

    result = runner.invoke(
        app, ["deviation", "add", "windlass swapped"], input=user_input
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 after edit+clear+accept; got {result.exit_code}: {combined!r}"
    )

    # Verify the reason field is NULL/empty in the committed deviation
    conn = _open_db(db_path)
    row = conn.execute("SELECT reason FROM deviations ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row is not None, "Expected a deviations row after accept"
    reason_val = row[0]
    assert not reason_val, (
        f"Expected reason cleared to NULL/empty after '-' sentinel; got {reason_val!r}"
    )


# (7) notes is shown in the review table AND editable (code-review WR-01)
#     The LLM extracts `notes` and it is persisted to the row + embedded into the
#     retrievable vessel chunk, so the review-before-commit gate MUST surface it
#     (otherwise an LLM-fabricated note ships invisibly into the authoritative layer).
def test_deviation_add_notes_visible_and_editable(monkeypatch, fake_embedder, tmp_path):
    """The review table shows `notes`, and the edit loop can clear it via '-' (WR-01)."""
    import leopard44_kb.deviation as dev
    from leopard44_kb.deviation import DeviationExtraction

    fixed = DeviationExtraction(
        component="windlass",
        factory_spec="12V Muir 1200W",
        as_built="12V Maxwell 1000W",
        reason="replacement after failure",
        date_noted="2024-01-10",
        notes="auto-extracted hull note",
    )
    monkeypatch.setattr(dev, "extract_fields", lambda text: fixed)

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    # Edit: keep component/factory_spec/as_built/reason/date_noted, clear notes via '-', accept.
    # Sequence: e → 5×(keep) → notes:'-' → a
    user_input = "e\n\n\n\n\n\n-\na\n"
    result = runner.invoke(
        app, ["deviation", "add", "windlass swapped"], input=user_input
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, f"Expected exit 0; got {result.exit_code}: {combined!r}"

    # (a) Visibility: the extracted notes value appeared in the review table.
    assert "auto-extracted hull note" in combined, (
        f"Expected the extracted notes to be shown in the review table; got {combined!r}"
    )

    # (b) Editability: clearing notes via '-' persisted NULL/empty.
    conn = _open_db(db_path)
    row = conn.execute("SELECT notes FROM deviations ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row is not None, "Expected a deviations row after accept"
    assert not row[0], f"Expected notes cleared to NULL/empty after '-'; got {row[0]!r}"
