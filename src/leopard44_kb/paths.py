"""Path validation and data-directory bootstrap for Leopard 44 KB.

SCHEMA-03 is enforced here, NOT in SQL — see Pitfall 1 in 01-RESEARCH.md.
"""
from __future__ import annotations

import os
from pathlib import Path

ALLOWED_ROOTS: dict[str, str] = {
    "shared": "shared",
    "vessel": "data",
    "community": "shared",  # v1.0 treats community as shared-equivalent (Open Question 1)
}

DATA_SUBDIRS: tuple[str, ...] = ("whatsapp", "docs", "logs", "photos", "inventory", "schematics")


def validate_path(layer: str, path: Path, repo_root: Path) -> Path:
    """Return the resolved path or raise ValueError. Symlink-aware via Path.resolve().

    Args:
        layer: One of the keys in ALLOWED_ROOTS ('shared', 'vessel', 'community').
        path: The path to validate. May be relative or absolute.
        repo_root: The root of the repository.

    Returns:
        The resolved (canonical, symlink-followed) path.

    Raises:
        ValueError: If layer is unknown, or if the resolved path does not fall
                    under the expected root directory for the given layer.
    """
    if layer not in ALLOWED_ROOTS:
        raise ValueError(
            f"Unknown layer: {layer!r}. Must be one of {sorted(ALLOWED_ROOTS)}."
        )
    # Path.resolve() canonicalises .. segments AND follows symlinks (Pitfall 6).
    # A symlink whose target is outside expected_root will correctly fail relative_to().
    resolved = path.resolve()
    expected_root = (repo_root / ALLOWED_ROOTS[layer]).resolve()
    try:
        resolved.relative_to(expected_root)
    except ValueError:
        raise ValueError(
            f"Layer {layer!r} content must live under {expected_root}; got {resolved}"
        ) from None
    return resolved


def repo_root() -> Path:
    """Return the project root directory.

    Uses the current working directory, which callers (CLI commands run from the
    project root, test fixtures that monkeypatch.chdir to a temp dir) control at
    runtime.  This matches the existing convention in item_add_cmd and zone_add_cmd
    which resolve the repo root via os.getcwd().

    Web app routes that need a module-path-relative root (immune to cwd drift)
    use Path(__file__).resolve().parents[3] directly (app.py is 4 levels deep:
    src/leopard44_kb/web/app.py -> parents[3] = repo root).

    NOTE on the choice of os.getcwd() vs Path(__file__).resolve().parents[N]:
    The CLI is always invoked from the repo root or a fixture-controlled cwd.
    Using cwd keeps test isolation: each test tmpdir becomes its own "repo root"
    via monkeypatch.chdir, matching the file expectations of test_render_cli.
    """
    return Path(os.getcwd())


def ensure_data_dirs(repo_root: Path) -> None:
    """Create data/{whatsapp,docs,logs,photos} subdirs idempotently.

    Preserves pre-existing siblings (data/sails/, data/sources/) — those are
    simply not enumerated here.

    Args:
        repo_root: The root of the repository.
    """
    for subdir in DATA_SUBDIRS:
        (repo_root / "data" / subdir).mkdir(parents=True, exist_ok=True)
