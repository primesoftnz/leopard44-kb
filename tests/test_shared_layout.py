# No leopard44_kb.* imports needed — pure filesystem assertions against committed repo tree.
"""Tests for CONTRIB-01: shared/ directory layout committed in the repo."""
from __future__ import annotations

from pathlib import Path


# Resolve the repo root from this test file's location
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_shared_topic_dirs_exist():
    """CONTRIB-01: shared/ contains expected topic subdirectories."""
    shared = REPO_ROOT / "shared"
    for topic in ("leopard44", "yanmar", "systems", "upgrades"):
        assert (shared / topic).is_dir(), (
            f"Expected shared/{topic} to exist as a directory"
        )


def test_shared_topic_dirs_have_gitkeep():
    """Each topic subdir (except leopard44) contains a .gitkeep placeholder file.

    leopard44/ is excluded because it receives real .md content after the 06-03
    maintainer migration — the .gitkeep is git-rm'd when the 10 seed files land
    (INSTALL-04 Pitfall 1). The other three dirs remain empty until contributions
    or PDF ingests populate them.
    """
    shared = REPO_ROOT / "shared"
    for topic in ("yanmar", "systems", "upgrades"):
        gitkeep = shared / topic / ".gitkeep"
        assert gitkeep.exists(), (
            f"Expected shared/{topic}/.gitkeep to exist"
        )


def test_shared_readme_exists_and_nonempty():
    """CONTRIB-01: shared/README.md exists and has substantial content."""
    readme = REPO_ROOT / "shared" / "README.md"
    assert readme.exists(), "shared/README.md must exist"
    content = readme.read_text(encoding="utf-8")
    assert len(content) > 100, (
        f"shared/README.md is too short ({len(content)} chars); expected > 100"
    )


def test_leopard44_seed_files_exist():
    """INSTALL-04: shared/leopard44/ contains the 10 reference .md files post-migration.

    RED until 06-03 Task 1 (the maintainer git mv) migrates the files.
    """
    leopard44 = REPO_ROOT / "shared" / "leopard44"
    seed_files = [
        "bluewater-capability.md",
        "competitive-comparison.md",
        "ex-charter-buying-guide.md",
        "interior-layout.md",
        "known-issues.md",
        "market-value.md",
        "onboard-systems.md",
        "sailing-performance.md",
        "technical-specifications.md",
        "upgrades-modifications.md",
    ]
    for fname in seed_files:
        fpath = leopard44 / fname
        assert fpath.exists(), (
            f"Expected shared/leopard44/{fname} to exist after migration (INSTALL-04). "
            f"Run 06-03 Task 1 to perform the maintainer git mv."
        )


def test_contributing_md():
    """CONTRIB-02: CONTRIBUTING.md exists at repo root with required content.

    RED until 06-04 lands CONTRIBUTING.md.
    """
    contrib = REPO_ROOT / "CONTRIBUTING.md"
    assert contrib.exists(), (
        "CONTRIBUTING.md must exist at the repo root (CONTRIB-02)"
    )
    content = contrib.read_text(encoding="utf-8")
    assert len(content) > 100, (
        f"CONTRIBUTING.md is too short ({len(content)} chars); expected > 100"
    )
    lower = content.lower()
    assert "shared" in lower, "CONTRIBUTING.md must mention 'shared'"
    assert "fork" in lower, "CONTRIBUTING.md must mention 'fork'"
    assert ("pull request" in lower or " pr " in lower or "\npr\n" in lower or lower.endswith(" pr")), (
        "CONTRIBUTING.md must mention 'pull request' or 'pr'"
    )
    assert "data/" in lower, "CONTRIBUTING.md must mention 'data/' guard"


def test_shared_readme_mentions_attribution_and_layers():
    """shared/README.md mentions key architectural concepts."""
    readme = REPO_ROOT / "shared" / "README.md"
    assert readme.exists(), "shared/README.md must exist"
    content = readme.read_text(encoding="utf-8").lower()
    for keyword in ("shared", "vessel", "attribution"):
        assert keyword in content, (
            f"Expected '{keyword}' in shared/README.md content"
        )


# ---------------------------------------------------------------------------
# PKG-01: MIT LICENSE hygiene
# ---------------------------------------------------------------------------


def test_pkg01_license_exists_and_is_mit():
    """PKG-01: LICENSE file exists at repo root and contains 'MIT License'."""
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.exists(), (
        "LICENSE must exist at the repo root (PKG-01). "
        "Create it with the standard MIT text."
    )
    content = license_path.read_text(encoding="utf-8")
    assert "MIT License" in content, (
        "LICENSE must contain 'MIT License' (PKG-01). "
        f"Found: {content[:80]!r}"
    )
    assert "Greg Stevenson" in content, (
        "LICENSE must contain 'Greg Stevenson' as the copyright holder (PKG-01)."
    )


# ---------------------------------------------------------------------------
# PKG-02: README.md link-resolution hygiene (Codex MEDIUM 13-05)
# ---------------------------------------------------------------------------

import re as _re  # noqa: E402  (module-level would move import above tests above)


def _extract_relative_md_links(text: str) -> list[str]:
    """Return all relative link targets from Markdown link syntax [text](target).

    Skips:
    - http(s):// absolute URLs
    - #anchor-only fragments
    - mailto: links
    """
    pattern = _re.compile(r'\[(?:[^\]]*)\]\(([^)]+)\)')
    targets = []
    for match in pattern.finditer(text):
        target = match.group(1).strip()
        # Strip trailing #anchor from relative links (e.g. "file.md#section")
        if "#" in target:
            target = target.split("#")[0]
        if not target:
            continue  # was a pure anchor "#fragment"
        if target.startswith(("http://", "https://", "mailto:")):
            continue  # absolute URL — not our concern
        targets.append(target)
    return targets


def test_pkg02_readme_exists_and_not_empty():
    """PKG-02: README.md exists at repo root and has substantial content."""
    readme = REPO_ROOT / "README.md"
    assert readme.exists(), "README.md must exist at the repo root (PKG-02)."
    content = readme.read_text(encoding="utf-8")
    assert len(content) > 500, (
        f"README.md is too short ({len(content)} chars); expected > 500 (PKG-02)."
    )
    lower = content.lower()
    assert "prerequisite" in lower or "requirements" in lower, (
        "README.md must mention prerequisites or requirements (PKG-02)."
    )
    assert "setup.sh" in content or "setup" in lower, (
        "README.md must mention setup (PKG-02)."
    )


def test_pkg02_readme_all_relative_links_resolve():
    """PKG-02: Every relative markdown link in README.md resolves to an existing file.

    Repoints Phase-6 git-mv dead links.  Fails if ANY relative link target does
    not exist on disk relative to the repo root.
    """
    readme = REPO_ROOT / "README.md"
    assert readme.exists(), "README.md must exist (PKG-02)."
    content = readme.read_text(encoding="utf-8")
    targets = _extract_relative_md_links(content)

    dead = []
    for target in targets:
        resolved = (REPO_ROOT / target).resolve()
        if not resolved.exists():
            dead.append(target)

    assert not dead, (
        f"README.md contains {len(dead)} dead relative link(s) (PKG-02 / Codex MEDIUM 13-05):\n"
        + "\n".join(f"  - {t}" for t in dead)
        + "\nFix by repointing to existing paths or removing the links."
    )
