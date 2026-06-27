"""Unit tests for leopard44_kb.capture.confirm.normalize_category (WR-05).

The free-form category the vision model returns must map to a valid inventory
enum value WITHOUT the old fragile substring heuristic that misfired on ordinary
boat vocabulary (e.g. "toolbox of spares" → tool, "watermaker spares" → provision).
Resolution is exact-enum-or-synonym, otherwise the safe "spare" default.
"""
from __future__ import annotations

import pytest

from leopard44_kb.capture.confirm import VALID_CATEGORIES, normalize_category


@pytest.mark.parametrize("enum_value", VALID_CATEGORIES)
def test_exact_enum_values_pass_through(enum_value):
    """Every valid enum value maps to itself (case-insensitively)."""
    assert normalize_category(enum_value) == enum_value
    assert normalize_category(enum_value.upper()) == enum_value
    assert normalize_category(f"  {enum_value}  ") == enum_value


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Synonyms → mapped enum value
        ("flares", "safety"),
        ("life jacket", "safety"),
        ("EPIRB", "safety"),
        ("first aid kit", "safety"),
        ("fire extinguisher", "safety"),
        ("toolbox", "tool"),
        ("power tool", "tool"),
        ("food", "provision"),
        ("water", "provision"),
        ("galley", "provision"),
        ("games", "toy"),
    ],
)
def test_curated_synonyms_map_correctly(raw, expected):
    assert normalize_category(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        # These previously MISFIRED under the loose `in` substring matching:
        "toolbox of spares",     # old: "tool" in raw → tool   (should be spare)
        "watermaker spares",     # old: "water" in raw → provision (should be spare)
        "wildlife camera",       # old: "life" in raw → safety (should be spare)
        "lifestyle magazine",    # old: "life" in raw → safety (should be spare)
        # Genuinely unknown free-form boat hardware → safe default.
        "raw-water pump impeller",
        "deck hardware",
        "adhesive",
        "navigation light",
        "bronze through-hull",
        "",                       # empty
    ],
)
def test_unmatched_strings_default_to_spare(raw):
    """Anything not an exact enum/synonym match defaults to spare (no false hits)."""
    assert normalize_category(raw) == "spare"


def test_none_defaults_to_spare():
    assert normalize_category(None) == "spare"


def test_result_is_always_a_valid_enum_value():
    """Whatever the input, the output is always one of the strict enum values."""
    for raw in [
        "flares", "toolbox of spares", "watermaker spares", "adhesive",
        "EPIRB", "", "completely made up nonsense", "TOY", "Provision",
    ]:
        assert normalize_category(raw) in VALID_CATEGORIES
