# No leopard44_kb.* imports needed — subprocess integration test.
"""Tests for CONTRIB-03: pre-commit hook rejects commits under data/."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Skip the whole module if pre-commit is not available in the venv
if shutil.which("pre-commit") is None:
    pytest.skip("pre-commit not in venv — install via uv sync --extra dev", allow_module_level=True)


def _setup_test_repo(tmp_path: Path) -> Path:
    """Create a throwaway git repo with the project's .pre-commit-config.yaml installed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.test"],
        cwd=repo, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, check=True,
    )
    # Copy the real .pre-commit-config.yaml from the project root
    config_src = Path(__file__).resolve().parent.parent / ".pre-commit-config.yaml"
    (repo / ".pre-commit-config.yaml").write_text(config_src.read_text(encoding="utf-8"))
    subprocess.run(["pre-commit", "install"], cwd=repo, check=True)
    return repo


def test_data_commit_is_rejected(tmp_path):
    """Pre-commit hook rejects a commit with a file staged under data/."""
    repo = _setup_test_repo(tmp_path)

    # Stage a file under data/
    (repo / "data").mkdir()
    (repo / "data" / "secret.txt").write_text("vessel data")
    subprocess.run(["git", "add", "data/secret.txt"], cwd=repo, check=True)

    r = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "x"],
        cwd=repo,
        capture_output=True,
    )
    assert r.returncode != 0, (
        "Expected commit to fail when data/ file is staged, but it succeeded"
    )
    combined = r.stdout + r.stderr
    assert b"no-data-commits" in combined, (
        f"Expected 'no-data-commits' in hook output, got: {combined!r}"
    )


def test_commit_outside_data_is_allowed(tmp_path):
    """Pre-commit hook allows a commit with a file staged outside data/."""
    repo = _setup_test_repo(tmp_path)

    # Stage a file outside data/
    (repo / "shared").mkdir()
    (repo / "shared" / "x.md").write_text("# Shared content")
    subprocess.run(["git", "add", "shared/x.md"], cwd=repo, check=True)

    r = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "add shared doc"],
        cwd=repo,
        capture_output=True,
    )
    assert r.returncode == 0, (
        f"Expected commit outside data/ to succeed, got returncode={r.returncode}: "
        f"{(r.stdout + r.stderr).decode()!r}"
    )
