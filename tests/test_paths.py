# RED state until Plan 03 (see VALIDATION.md). Imports from leopard44_kb.* will fail until production code lands.
"""Tests for SCHEMA-03: leopard44_kb.paths.validate_path() path enforcement."""
from __future__ import annotations

import pytest
from pathlib import Path

from leopard44_kb.paths import validate_path


def test_vessel_data_path_ok(repo_root):
    """Vessel-layer path under data/ is accepted; returns resolved Path."""
    target = repo_root / "data" / "logs" / "engine.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    result = validate_path("vessel", target, repo_root)
    # Result must be resolvable relative to data/
    result.relative_to((repo_root / "data").resolve())


def test_vessel_outside_data_rejected(repo_root):
    """Vessel-layer path outside data/ raises ValueError mentioning 'data'."""
    target = repo_root / "shared" / "thing.md"
    with pytest.raises(ValueError, match="data"):
        validate_path("vessel", target, repo_root)


def test_shared_path_ok(repo_root):
    """Shared-layer path under shared/ is accepted; returns resolved Path."""
    target = repo_root / "shared" / "leopard44" / "specs.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    result = validate_path("shared", target, repo_root)
    assert isinstance(result, Path)


def test_shared_outside_shared_rejected(repo_root):
    """Shared-layer path outside shared/ raises ValueError."""
    target = repo_root / "data" / "logs" / "private.md"
    with pytest.raises(ValueError):
        validate_path("shared", target, repo_root)


def test_unknown_layer_rejected(repo_root):
    """Unknown layer name raises ValueError mentioning the layer or 'layer'."""
    target = repo_root / "data" / "whatever.md"
    with pytest.raises(ValueError, match=r"garbage|layer"):
        validate_path("garbage", target, repo_root)


def test_path_dotdot_collapses(repo_root):
    """Path with .. segments resolves cleanly under data/ and is accepted."""
    target = repo_root / "data" / ".." / "data" / "logs" / "x.md"
    target_resolved = (repo_root / "data" / "logs" / "x.md")
    target_resolved.parent.mkdir(parents=True, exist_ok=True)
    target_resolved.touch()
    result = validate_path("vessel", target, repo_root)
    result.relative_to((repo_root / "data").resolve())


def test_absolute_path_traversal_rejected(repo_root):
    """Absolute path outside repo root raises ValueError for vessel layer."""
    target = Path("/tmp/escape.md")
    with pytest.raises(ValueError):
        validate_path("vessel", target, repo_root)


def test_symlink_outside_repo_rejected(repo_root):
    """Symlink inside data/ pointing to /etc/passwd resolves outside data/ and is rejected."""
    import os
    log_dir = repo_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    symlink = log_dir / "escape"
    try:
        os.symlink("/etc/passwd", symlink)
    except OSError:
        pytest.skip("Cannot create symlink on this platform")
    with pytest.raises(ValueError):
        validate_path("vessel", symlink, repo_root)
