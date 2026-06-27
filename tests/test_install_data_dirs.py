# RED state until Plan 03 (see VALIDATION.md). Imports from leopard44_kb.* will fail until production code lands.
"""Tests for INSTALL-03: ensure_data_dirs() bootstraps data/ subdirectories."""
from __future__ import annotations

import pytest

from leopard44_kb.paths import ensure_data_dirs


def test_ensure_data_dirs_creates_subdirs(tmp_path):
    """ensure_data_dirs creates data/whatsapp, data/docs, data/logs, data/photos."""
    ensure_data_dirs(tmp_path)

    for subdir in ("whatsapp", "docs", "logs", "photos"):
        assert (tmp_path / "data" / subdir).is_dir(), (
            f"Expected data/{subdir} to be created by ensure_data_dirs"
        )


def test_ensure_data_dirs_is_idempotent(tmp_path):
    """Calling ensure_data_dirs twice does not raise."""
    ensure_data_dirs(tmp_path)
    ensure_data_dirs(tmp_path)  # Should not raise


def test_ensure_data_dirs_preserves_existing_subdirs(tmp_path):
    """ensure_data_dirs leaves pre-existing data/ content intact.

    Coexists with data/sails/ and data/sources/ from prior ad-hoc work
    (per CONTEXT.md code_context section).
    """
    # Pre-create a directory with content that should be preserved
    (tmp_path / "data" / "sails").mkdir(parents=True, exist_ok=True)
    preserved_file = tmp_path / "data" / "sails" / "x.txt"
    preserved_file.write_text("pre-existing sails data")

    ensure_data_dirs(tmp_path)

    assert preserved_file.exists(), (
        "ensure_data_dirs must not delete pre-existing data/sails/x.txt"
    )
    assert preserved_file.read_text() == "pre-existing sails data"
