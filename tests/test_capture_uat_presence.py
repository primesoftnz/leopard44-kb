"""UAT presence guard for Phase 12 cloud hard-case UAT (M7).

Ensures that once 12-HUMAN-UAT.md exists AND has been filled in by the owner,
it contains per-photo result rows AND an explicit final PASS marker. This is a
PRESENCE check only — it does not validate truth — so the phase cannot be closed
on an empty or stale UAT file.

Skips gracefully when:
  - 12-HUMAN-UAT.md does not yet exist (pre-Task 3 state), OR
  - The file exists but the Overall result is still PENDING (the owner has not
    yet completed the UAT run — the blocking-human checkpoint is not cleared).

Fires (and may FAIL) only after the owner fills in per-photo rows and changes
the Overall marker from PENDING to PASS or FAIL.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _uat_file() -> Path:
    """Return the expected path to 12-HUMAN-UAT.md."""
    # Repo root: tests/ is at the same level as .planning/
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / ".planning" / "phases" / "12-photo-vision-capture" / "12-HUMAN-UAT.md"


def _uat_is_pending() -> bool:
    """Return True if the UAT file exists but is still in PENDING state."""
    p = _uat_file()
    if not p.exists():
        return False  # doesn't exist — the other skipif handles this
    content = p.read_text(encoding="utf-8")
    # The file is pending if the Overall line still says PENDING (not PASS or FAIL)
    return "Overall: PENDING" in content or "status: pending" in content


@pytest.mark.skipif(
    not _uat_file().exists(),
    reason="12-HUMAN-UAT.md not yet authored — awaiting blocking cloud UAT",
)
@pytest.mark.skipif(
    _uat_is_pending(),
    reason="12-HUMAN-UAT.md is still PENDING — owner has not yet completed the cloud UAT",
)
def test_uat_file_has_per_photo_rows_and_pass_marker():
    """When 12-HUMAN-UAT.md is filled in, assert per-photo rows + final PASS marker (M7).

    Per-photo rows: the UAT table must have at least one data row (a line that
    begins with '|' and contains either 'PASS' or 'FAIL').
    Final PASS marker: the file must contain the word 'PASS' (as a word boundary
    match) somewhere in the file — e.g. "Overall: PASS", "**PASS**", "## PASS".

    This guard fires only once the owner fills in the results table AND changes
    the Overall status from PENDING to PASS or FAIL. Until then both skipif
    conditions above cause the test to be skipped gracefully.
    """
    uat_path = _uat_file()
    content = uat_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # --- Check 1: at least one per-photo result row in the table ---
    # A result row is a Markdown table row (starts with |) that contains
    # a non-empty value in at least one cell (not just pipes and spaces).
    filled_rows = [
        line for line in lines
        if line.strip().startswith("|")
        and not set(line.replace("|", "").replace("-", "").strip()) <= {" ", ""}
        and not line.strip().startswith("| Photo")   # skip header
        and "|---" not in line                         # skip separator
    ]
    assert filled_rows, (
        f"12-HUMAN-UAT.md has no filled per-photo result rows. "
        f"The UAT table must be filled in before closing Phase 12.\n"
        f"File path: {uat_path}"
    )

    # --- Check 2: a final PASS marker anywhere in the file ---
    # Accepts: line containing 'PASS' as a whole word (e.g. "Overall: PASS",
    # "**PASS**", "## PASS", "RESULT: PASS").
    import re
    has_pass = any(
        re.search(r"\bPASS\b", line) for line in lines
    )
    assert has_pass, (
        f"12-HUMAN-UAT.md does not contain a final 'PASS' marker. "
        f"The owner must record PASS (or FAIL) after completing the cloud UAT. "
        f"The phase cannot be closed without this marker (M7 presence guard).\n"
        f"File path: {uat_path}"
    )
