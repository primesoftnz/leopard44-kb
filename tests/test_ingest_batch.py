# RED state until Phase 2 implementation (see 02-VALIDATION.md). Imports from leopard44_kb.ingest.* fail until production code lands.
"""Tests for INGEST-07: per-file progress output, partial failure tolerance, directory filtering."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from leopard44_kb.cli import app

runner = CliRunner()


def _seed_db_env(monkeypatch, tmp_path):
    """Set L44_DB to an isolated test database path."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    return db_path


def _make_valid_note(directory: Path, name: str = "note.md") -> Path:
    """Write a minimal valid markdown note inside the given data/ directory."""
    f = directory / name
    f.write_text("# Test\n\nSome content.\n", encoding="utf-8")
    return f


def test_partial_failure(monkeypatch, fake_embedder, tmp_path):
    """Batch ingest: one good file + one non-existent path → exit code 1, good file still ingested.

    INGEST-07: single file failure is logged, batch continues.
    """
    _seed_db_env(monkeypatch, tmp_path)

    data_dir = tmp_path / "data" / "logs"
    data_dir.mkdir(parents=True)
    good_file = _make_valid_note(data_dir, "good.md")
    bad_path = str(tmp_path / "data" / "logs" / "nonexistent_file_xyz.md")

    result = runner.invoke(app, ["ingest", str(good_file), bad_path])
    # At least one file failed → exit code must be 1
    assert result.exit_code == 1, (
        f"Expected exit code 1 for partial failure; got {result.exit_code}. "
        f"Output: {result.output}"
    )
    # The good file must still appear as ingested (OK in output or in DB)
    combined = (result.output or "") + (result.stderr or "")
    assert "OK" in combined or "ok" in combined.lower(), (
        f"Expected success indication for good file; output: {combined!r}"
    )


def test_progress_output(monkeypatch, fake_embedder, tmp_path):
    """Per-file progress is echoed to stdout: each file gets a line with its result."""
    _seed_db_env(monkeypatch, tmp_path)

    data_dir = tmp_path / "data" / "logs"
    data_dir.mkdir(parents=True)
    f1 = _make_valid_note(data_dir, "progress1.md")
    f2 = _make_valid_note(data_dir, "progress2.md")

    result = runner.invoke(app, ["ingest", str(f1), str(f2)])
    combined = (result.output or "") + (result.stderr or "")
    # Each file should produce at least one output line
    assert "progress1.md" in combined or "OK" in combined, (
        f"Expected per-file progress for progress1.md; output: {combined!r}"
    )
    assert "progress2.md" in combined or combined.count("OK") >= 2, (
        f"Expected per-file progress for progress2.md; output: {combined!r}"
    )


def test_directory_filters_unsupported(monkeypatch, fake_embedder, tmp_path):
    """Directory expansion yields only supported extensions; skipped files are reported.

    A directory containing a .jpg and a .py alongside a .md should ingest only the .md.
    """
    _seed_db_env(monkeypatch, tmp_path)

    data_dir = tmp_path / "data" / "mixed"
    data_dir.mkdir(parents=True)

    supported = data_dir / "document.md"
    supported.write_text("# Doc\n\nSupported content.\n", encoding="utf-8")

    unsupported_jpg = data_dir / "photo.jpg"
    unsupported_jpg.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # fake JPEG header

    unsupported_py = data_dir / "script.py"
    unsupported_py.write_text("print('hello')\n", encoding="utf-8")

    result = runner.invoke(app, ["ingest", str(data_dir)])
    combined = (result.output or "") + (result.stderr or "")

    # Unsupported files should appear as skipped — not as errors in the failure count
    # The supported .md should be ingested successfully
    assert "document.md" in combined or "OK" in combined, (
        f"Supported file not mentioned in output: {combined!r}"
    )
    # The script and photo should be reported as skipped — not cause a global exit 1
    # (skipped ≠ failed; only real ingest errors count toward exit code)
    assert result.exit_code == 0, (
        f"Expected exit 0 when only unsupported files are skipped; got {result.exit_code}. "
        f"Output: {combined!r}"
    )
