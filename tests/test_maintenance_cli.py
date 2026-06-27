# RED state until Plan 04 lands. add_cmd is still the _not_yet("4") stub; log_cmd,
# leopard44_kb.log, leopard44_kb.maintenance, the _stdin_isatty seam, and the ingest-path guard
# do not exist yet. Collection-time errors are expected for tests that import from these
# modules. Exception: test_maintenance_maint_path_allowed only uses leopard44_kb.paths which
# already exists and may go GREEN immediately.
"""Tests for MAINT-01/03/04/05 and the three Codex review contradictions:
  #1 idempotency-vs-collision, #2 TTY seam, #3 ingest-path layer leak guard.

Per-requirement verification map source: .planning/phases/04-maintenance-log/04-VALIDATION.md
Review-added nodes marked (rev): test_add_rejects_empty_entry, test_log_renders_parts,
test_add_same_day_different_content_collides, test_ingest_maint_path_shared_rejected.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlite_vec
from typer.testing import CliRunner

from leopard44_kb.cli import app
from leopard44_kb.schema import apply_migrations

runner = CliRunner()


def _bootstrap_db(db_path: Path) -> None:
    """Bootstrap a file-backed DB at db_path with sqlite-vec and migrations applied."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    conn.close()


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Open the bootstrapped DB for direct queries in tests."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _make_maint_dir(tmp_path: Path) -> Path:
    """Create the data/logs/maint/ directory under tmp_path, matching the vessel layout."""
    maint_dir = tmp_path / "data" / "logs" / "maint"
    maint_dir.mkdir(parents=True, exist_ok=True)
    return maint_dir


# ---------------------------------------------------------------------------
# MAINT-01: add_cmd writes an entry file
# ---------------------------------------------------------------------------


def test_add_writes_entry_file(monkeypatch, fake_extractor, fake_embedder, tmp_path):
    """add --yes exits 0 and writes at least one .md file under data/logs/maint/."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    result = runner.invoke(
        app, ["add", "replaced port impeller, $45 from Burnsco, 2024-03-15", "--yes"]
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, f"Expected exit 0; got {result.exit_code}: {combined!r}"

    maint_files = list((tmp_path / "data" / "logs" / "maint").glob("*.md"))
    assert len(maint_files) >= 1, (
        f"Expected at least one .md file under data/logs/maint/, found: {maint_files}"
    )


# ---------------------------------------------------------------------------
# MAINT-01: add is no longer a stub
# ---------------------------------------------------------------------------


def test_add_no_longer_a_stub(monkeypatch, fake_extractor, fake_embedder, tmp_path):
    """add does not print 'Not yet implemented' and does not exit with code 2."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    result = runner.invoke(app, ["add", "x", "--yes"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Not yet implemented" not in combined, (
        f"Stub message still present: {combined!r}"
    )
    assert result.exit_code != 2, "Stub exit code 2 still returned; add is not implemented yet"


# ---------------------------------------------------------------------------
# MAINT-03: --yes skips review
# ---------------------------------------------------------------------------


def test_add_yes_skips_review(monkeypatch, fake_extractor, fake_embedder, tmp_path):
    """add --yes exits 0 and produces no review-prompt text in output."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    result = runner.invoke(app, ["add", "replaced impeller", "--yes"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, f"Expected exit 0; got {result.exit_code}: {combined!r}"

    # Review prompt words must not appear when --yes is passed
    for review_word in ("Accept", "Edit", "Abort"):
        assert review_word not in combined, (
            f"Review prompt word '{review_word}' appeared despite --yes: {combined!r}"
        )


# ---------------------------------------------------------------------------
# MAINT-03 / D-10: non-TTY without --yes must fail fast, never hang
# ---------------------------------------------------------------------------


def test_add_no_tty_no_yes_fails(monkeypatch, tmp_path):
    """Non-TTY stdin without --yes exits 1 with 'no terminal for review'; never hangs.

    CliRunner stdin is not a TTY and _stdin_isatty is NOT patched here, so it reports
    False. The command must fail fast (D-10 STRICT gate).
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    result = runner.invoke(app, ["add", "x"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 1, (
        f"Expected exit 1 for non-TTY no-yes; got {result.exit_code}: {combined!r}"
    )
    assert "no terminal for review" in combined.lower(), (
        f"Expected 'no terminal for review' message; got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# MAINT-03 / D-09: Edit overrides a field (Codex HIGH contradiction #2 resolved)
# ---------------------------------------------------------------------------


def test_add_edit_overrides_field(monkeypatch, fake_extractor, fake_embedder, tmp_path):
    """Edit walk with monkeypatched _stdin_isatty→True overrides vendor field.

    CRITICAL: monkeypatches leopard44_kb.cli._stdin_isatty → True so the edit loop is
    reachable under CliRunner (whose stdin is non-TTY). Production keeps the STRICT
    D-10 gate; only the test simulates a terminal via this named seam.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    # Patch the named TTY seam production code defines (Plan 04)
    monkeypatch.setattr("leopard44_kb.cli._stdin_isatty", lambda: True)

    # Input stream: choose Edit ("e"), override vendor ("Baobab"), accept rest with Enter
    # Sequence: review choice → "e", vendor field → "Baobab", remaining fields → Enter each
    user_input = "e\nBaobab\n\n\n\n\n\na\n"

    result = runner.invoke(app, ["add", "replaced impeller"], input=user_input)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, (
        f"Expected exit 0 after edit+accept; got {result.exit_code}: {combined!r}"
    )
    assert "Baobab" in combined, (
        f"Expected overridden vendor 'Baobab' in output; got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# MAINT-04: log filters by system
# ---------------------------------------------------------------------------


def test_log_filters_by_system(monkeypatch, fake_extractor, fake_embedder, tmp_path):
    """Entry appears in log --system engine; absent from log --system electrical."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    # Add entry (fake_extractor → system=engine)
    add_result = runner.invoke(app, ["add", "replaced impeller", "--yes"])
    assert add_result.exit_code == 0, f"add failed: {add_result.output!r}"

    # Log filtered to engine → entry present
    engine_result = runner.invoke(app, ["log", "--system", "engine"])
    engine_combined = (engine_result.stdout or "") + (engine_result.stderr or "")
    assert engine_result.exit_code == 0, f"log --system engine failed: {engine_combined!r}"
    assert "impeller" in engine_combined.lower() or "2024-03-15" in engine_combined, (
        f"Engine entry should appear in log --system engine: {engine_combined!r}"
    )

    # Log filtered to electrical → entry absent
    elec_result = runner.invoke(app, ["log", "--system", "electrical"])
    elec_combined = (elec_result.stdout or "") + (elec_result.stderr or "")
    assert "2024-03-15" not in elec_combined or "engine" not in elec_combined.lower(), (
        f"Engine entry should NOT appear in log --system electrical: {elec_combined!r}"
    )


# ---------------------------------------------------------------------------
# MAINT-04: log date range
# ---------------------------------------------------------------------------


def test_log_date_range(monkeypatch, fake_extractor, fake_embedder, tmp_path):
    """log --since/--until includes or excludes the 2024-03-15 entry accordingly."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    runner.invoke(app, ["add", "replaced impeller", "--yes"])

    # Within range → entry present
    in_range = runner.invoke(app, ["log", "--since", "2024-01-01", "--until", "2024-12-31"])
    in_combined = (in_range.stdout or "") + (in_range.stderr or "")
    assert in_range.exit_code == 0, f"log in-range failed: {in_combined!r}"
    assert "2024-03-15" in in_combined or "impeller" in in_combined.lower(), (
        f"Entry should appear in 2024 range: {in_combined!r}"
    )

    # After range → entry absent
    after_range = runner.invoke(app, ["log", "--since", "2025-01-01"])
    after_combined = (after_range.stdout or "") + (after_range.stderr or "")
    assert "2024-03-15" not in after_combined, (
        f"Entry should NOT appear when --since 2025-01-01: {after_combined!r}"
    )


# ---------------------------------------------------------------------------
# MAINT-04 / D-05 (rev): log renders parts list
# ---------------------------------------------------------------------------


def test_log_renders_parts(monkeypatch, fake_extractor, fake_embedder, tmp_path):
    """log renders the parts value in output (json_extract $.parts + CLI render).

    fake_extractor → parts=["impeller p/n 22-41016"]. Pins both the SELECT
    projecting $.parts AND the CLI json.loads-then-render of the parts array.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    runner.invoke(app, ["add", "replaced impeller", "--yes"])

    log_result = runner.invoke(app, ["log"])
    combined = (log_result.stdout or "") + (log_result.stderr or "")
    assert log_result.exit_code == 0, f"log failed: {combined!r}"
    assert "impeller p/n 22-41016" in combined, (
        f"Parts value should appear in log output: {combined!r}"
    )


# ---------------------------------------------------------------------------
# MAINT-04 / SC3: add entry retrievable via ask
# ---------------------------------------------------------------------------


def test_add_entry_retrievable_via_ask(
    monkeypatch, fake_extractor, fake_embedder, fake_generator, tmp_path
):
    """After add, entry is retrieved by ask and appears in a vessel: Sources line."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    # Add entry (fake_extractor + fake_embedder so no live Ollama)
    add_result = runner.invoke(app, ["add", "replaced port impeller", "--yes"])
    assert add_result.exit_code == 0, f"add failed: {add_result.output!r}"

    # Ask query (fake_generator so no live generation)
    ask_result = runner.invoke(app, ["ask", "impeller interval"])
    ask_combined = (ask_result.stdout or "") + (ask_result.stderr or "")
    assert ask_result.exit_code == 0, f"ask failed: {ask_combined!r}"

    # Sources block must contain a vessel: line referencing the maintenance entry
    assert "Sources:" in ask_combined, (
        f"Expected Sources: block in ask output: {ask_combined!r}"
    )
    sources_section = ask_combined[ask_combined.find("Sources:"):]
    vessel_lines = [l for l in sources_section.splitlines() if "vessel:" in l]
    assert vessel_lines, (
        f"Expected a vessel: Sources line; got sources section: {sources_section!r}"
    )


# ---------------------------------------------------------------------------
# MAINT-04 / INGEST-06: re-add identical content is idempotent (Codex HIGH #1)
# ---------------------------------------------------------------------------


def test_add_reingest_idempotent(monkeypatch, fake_extractor, fake_embedder, tmp_path):
    """Re-add of identical text → ONE source row AND one .md file.

    Identical content must reuse the same filename; content-hash makes the second
    ingest a no-op. Without the reuse fix this would create two files + two sources.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    text = "replaced port impeller $45 from Burnsco 2024-03-15"
    runner.invoke(app, ["add", text, "--yes"])
    runner.invoke(app, ["add", text, "--yes"])

    # Exactly one .md file
    maint_files = list((tmp_path / "data" / "logs" / "maint").glob("*.md"))
    assert len(maint_files) == 1, (
        f"Expected 1 .md file (idempotent re-add), found: {[f.name for f in maint_files]}"
    )

    # Exactly one maintenance_entry source row
    conn = _open_db(db_path)
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM sources WHERE source_type='maintenance_entry'"
    ).fetchone()
    conn.close()
    assert row["cnt"] == 1, (
        f"Expected 1 source row after idempotent re-add, got {row['cnt']}"
    )


# ---------------------------------------------------------------------------
# MAINT-04 (rev): different same-day content produces -2 collision (Codex HIGH #1)
# ---------------------------------------------------------------------------


def test_add_same_day_different_content_collides(monkeypatch, tmp_path):
    """Different same-day content → two .md files; second carries a -2 suffix.

    Tests write_entry directly to isolate the filename-allocation contract.
    Proves different same-day content still gets -2 (collision preserved).
    """
    from leopard44_kb.maintenance import CostModel, MaintenanceExtraction, write_entry

    extraction = MaintenanceExtraction(
        date="2024-03-15",
        system="engine",
        parts=["impeller"],
        vendor="Burnsco",
        cost=CostModel(amount=45.0, currency="NZD"),
    )

    maint_dir = tmp_path / "data" / "logs" / "maint"
    maint_dir.mkdir(parents=True, exist_ok=True)

    # First entry
    path1 = write_entry(extraction, original_text="replaced port impeller worn out", repo_root=tmp_path)
    # Second entry same date, different body text
    path2 = write_entry(extraction, original_text="replaced starboard impeller now", repo_root=tmp_path)

    maint_files = list(maint_dir.glob("*.md"))
    assert len(maint_files) == 2, (
        f"Expected 2 .md files for different same-day entries, found: {[f.name for f in maint_files]}"
    )
    assert path1 != path2, "Different content must produce different filenames"
    # One of them must carry -2 suffix
    names = [f.name for f in maint_files]
    assert any("-2.md" in n for n in names), (
        f"Expected one filename with -2 suffix for same-day collision: {names}"
    )


# ---------------------------------------------------------------------------
# MAINT-01 (rev): empty / whitespace-only entry rejected before LLM call
# ---------------------------------------------------------------------------


def test_add_rejects_empty_entry(monkeypatch, fake_extractor, fake_embedder, tmp_path):
    """add '' and add '   ' are rejected before the LLM call; no file written."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)
    _make_maint_dir(tmp_path)

    for bad_entry in ("", "   "):
        result = runner.invoke(app, ["add", bad_entry, "--yes"])
        combined = (result.stdout or "") + (result.stderr or "")
        assert result.exit_code != 0, (
            f"Expected non-zero exit for entry={bad_entry!r}; got {result.exit_code}: {combined!r}"
        )
        # Clear rejection message
        assert any(word in combined.lower() for word in ("empty", "no text", "blank")), (
            f"Expected rejection message for entry={bad_entry!r}; got: {combined!r}"
        )

    # No .md files written
    maint_files = list((tmp_path / "data" / "logs" / "maint").glob("*.md"))
    assert len(maint_files) == 0, (
        f"Expected no .md files for empty entries; found: {[f.name for f in maint_files]}"
    )


# ---------------------------------------------------------------------------
# MAINT-05 / D-14: add exposes no --layer flag
# ---------------------------------------------------------------------------


def test_add_no_layer_flag(monkeypatch, tmp_path):
    """add --layer shared is rejected; --help omits --layer."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    monkeypatch.chdir(tmp_path)

    # --layer shared must produce a non-zero exit (typer rejects the unknown option)
    result = runner.invoke(app, ["add", "x", "--layer", "shared", "--yes"])
    assert result.exit_code != 0, (
        f"Expected non-zero exit for --layer shared; got {result.exit_code}: {result.output!r}"
    )

    # --help must not list --layer
    help_result = runner.invoke(app, ["add", "--help"])
    help_combined = (help_result.stdout or "") + (help_result.stderr or "")
    assert "--layer" not in help_combined, (
        f"--layer should not appear in add --help: {help_combined!r}"
    )


# ---------------------------------------------------------------------------
# MAINT-05 / D-14 (rev): ingest maint path with --layer shared is rejected (Codex HIGH #3)
# ---------------------------------------------------------------------------


def test_ingest_maint_path_shared_rejected(monkeypatch, fake_embedder, tmp_path):
    """ingest <maint path> --layer shared → non-zero exit; zero non-vessel maintenance_entry rows.

    Closes the ingest-path door: the source_type=='maintenance_entry' and layer!='vessel'
    guard in Plan 03 must reject this. RED until Plan 03 adds the guard.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))

    # Read fixture content before chdir so the relative path resolves against the
    # real repo root (Rule 1 fix: relative path after chdir would FileNotFoundError).
    _here = Path(__file__).parent
    fixture_content = (_here / "fixtures" / "sample_maintenance_entry.md").read_text(encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    # Create a valid maintenance entry file under data/logs/maint/
    maint_dir = tmp_path / "data" / "logs" / "maint"
    maint_dir.mkdir(parents=True, exist_ok=True)
    maint_file = maint_dir / "2024-03-15-test.md"
    maint_file.write_text(fixture_content, encoding="utf-8")

    result = runner.invoke(app, ["ingest", str(maint_file), "--layer", "shared"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code != 0, (
        f"Expected non-zero exit for ingest maint path --layer shared; "
        f"got {result.exit_code}: {combined!r}"
    )

    # No non-vessel maintenance_entry rows
    if db_path.exists():
        conn = _open_db(db_path)
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM sources "
            "WHERE source_type='maintenance_entry' AND layer != 'vessel'"
        ).fetchone()
        conn.close()
        assert row["cnt"] == 0, (
            f"Expected 0 non-vessel maintenance_entry rows; got {row['cnt']}"
        )


# ---------------------------------------------------------------------------
# MAINT-05: validate_path allows vessel maint path; rejects shared maint path
# ---------------------------------------------------------------------------


def test_maintenance_maint_path_allowed(repo_root):
    """validate_path accepts data/logs/maint/... for vessel; rejects for shared.

    This test does not depend on the new modules and may go GREEN immediately —
    it pins the enforcement seam at the paths layer (D-14).
    """
    from leopard44_kb.paths import validate_path

    maint_dir = repo_root / "data" / "logs" / "maint"
    maint_dir.mkdir(parents=True, exist_ok=True)
    maint_file = maint_dir / "2024-03-15-x.md"
    maint_file.touch()

    # Vessel layer accepts data/logs/maint/ path
    result = validate_path("vessel", maint_file, repo_root)
    assert result is not None, "validate_path should return a Path for vessel layer"

    # Shared layer rejects a maintenance path (data/logs/maint/ is not under shared/)
    with pytest.raises(ValueError):
        validate_path("shared", maint_file, repo_root)
