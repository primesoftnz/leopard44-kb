# RED state until Plans 02 and 03 land. Imports from leopard44_kb.maintenance fail until
# Plan 02 production code exists. Collection-time ModuleNotFoundError on leopard44_kb.maintenance
# is the expected Wave 0 state, matching the Phase 1-3 convention.
"""Tests for MAINT-02 extraction + boundary hardening, date normalisation, pydantic,
parser, slug, and punctuation round-trip.

Per-requirement verification map source: .planning/phases/04-maintenance-log/04-VALIDATION.md
Review-added nodes marked (rev): test_extract_fallback_non_dict_response,
test_extract_fallback_malformed_nested_fields, test_extract_strips_code_fences,
test_front_matter_round_trip_punctuation.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from leopard44_kb.maintenance import (
    CostModel,
    MaintenanceExtraction,
    call_extract_json,
    extract_fields,
    make_entry_filename,
    normalise_date,
    write_entry,
)
from leopard44_kb.ingest.text_md import parse_maintenance_entry


# ---------------------------------------------------------------------------
# MAINT-02: extract_fields round-trip
# ---------------------------------------------------------------------------


def test_extract_round_trip(monkeypatch):
    """extract_fields returns correct structured object when call_extract_json returns valid dict."""
    import leopard44_kb.maintenance as maint

    good_raw = {
        "date": "2024-03-15",
        "system": "engine",
        "system_detail": "raw-water cooling",
        "parts": ["impeller"],
        "cost": {"amount": 45.0, "currency": "NZD"},
        "vendor": "Burnsco",
    }
    monkeypatch.setattr(maint, "call_extract_json", lambda prompt, sys_prompt: good_raw)

    result = extract_fields("replaced port impeller, $45 from Burnsco, 2024-03-15")

    assert result.date == "2024-03-15", f"Expected date=2024-03-15, got {result.date!r}"
    assert result.system == "engine", f"Expected system=engine, got {result.system!r}"
    assert result.cost is not None, "Expected cost to be set"
    assert result.cost.amount == 45.0, f"Expected cost.amount=45.0, got {result.cost.amount!r}"
    assert result.cost.currency == "NZD", f"Expected cost.currency=NZD, got {result.cost.currency!r}"
    assert result.vendor == "Burnsco", f"Expected vendor=Burnsco, got {result.vendor!r}"


# ---------------------------------------------------------------------------
# MAINT-02 / D-12: re-prompt on first ValidationError
# ---------------------------------------------------------------------------


def test_extract_reprompt_on_validation_error(monkeypatch):
    """Re-prompt fires when first call returns bad data; second valid response returned.

    Asserts call_extract_json is invoked exactly twice.
    """
    import leopard44_kb.maintenance as maint

    call_count = [0]
    bad_raw = {"date": "2024-03-15", "cost": "not-a-dict"}  # cost as string breaks CostModel
    good_raw = {
        "date": "2024-03-15",
        "system": "engine",
        "parts": ["impeller"],
        "cost": {"amount": 45.0, "currency": "NZD"},
        "vendor": "Burnsco",
    }

    def _fake_call(prompt, sys_prompt):
        call_count[0] += 1
        if call_count[0] == 1:
            return bad_raw
        return good_raw

    monkeypatch.setattr(maint, "call_extract_json", _fake_call)

    result = extract_fields("replaced impeller")

    assert result is not None, "Expected a result, not None"
    assert result.system == "engine", f"Expected system=engine, got {result.system!r}"
    assert call_count[0] == 2, f"Expected exactly 2 calls, got {call_count[0]}"


# ---------------------------------------------------------------------------
# MAINT-02 / D-13: fallback when both calls fail validation
# ---------------------------------------------------------------------------


def test_extract_reprompt_then_fallback(monkeypatch):
    """Two invalid responses → fallback object with system='other', no raise.

    Proves a poor extraction never loses the entry (D-13 / Pitfall 5).
    """
    import leopard44_kb.maintenance as maint

    # Both calls return a dict that will fail pydantic (missing required system)
    bad_raw = {"date": "2024-03-15"}

    monkeypatch.setattr(maint, "call_extract_json", lambda prompt, sys_prompt: bad_raw)

    result = extract_fields("some ambiguous entry")

    assert result is not None, "Expected a fallback object, not None/raise"
    assert result.system == "other", f"Expected system='other' fallback, got {result.system!r}"


# ---------------------------------------------------------------------------
# MAINT-02 / D-13 (rev): strict fallback when both calls return non-dict
# ---------------------------------------------------------------------------


def test_extract_fallback_non_dict_response(monkeypatch):
    """Non-dict second payload → treated as {} → valid object (system='other'), no raise.

    Tests the strict sanitizer that must handle non-dict payloads (e.g. string or list).
    The previous raw2.get(...) fallback would have crashed on a non-dict second response.
    """
    import leopard44_kb.maintenance as maint

    call_count = [0]

    def _fake_call(prompt, sys_prompt):
        call_count[0] += 1
        if call_count[0] == 1:
            return "not a dict at all"  # first call: non-dict
        return []  # second call: also non-dict (list)

    monkeypatch.setattr(maint, "call_extract_json", _fake_call)

    result = extract_fields("some text")

    assert result is not None, "Expected a fallback object, not None/raise"
    assert isinstance(result, MaintenanceExtraction), "Expected a MaintenanceExtraction instance"
    assert result.system == "other", f"Expected system='other' fallback, got {result.system!r}"


# ---------------------------------------------------------------------------
# MAINT-02 / D-13 (rev): fallback drops malformed nested fields
# ---------------------------------------------------------------------------


def test_extract_fallback_malformed_nested_fields(monkeypatch):
    """Malformed nested cost dropped (None), non-list parts coerced to [], no raise.

    Tests that the sanitizer drops invalid nested structures rather than re-raising.
    """
    import leopard44_kb.maintenance as maint

    bad_raw = {
        "system": "engine",
        "cost": "forty-five",    # string instead of dict — invalid
        "parts": "impeller",     # string instead of list — invalid
    }
    # Both calls return the same malformed dict
    monkeypatch.setattr(maint, "call_extract_json", lambda prompt, sys_prompt: bad_raw)

    result = extract_fields("impeller service")

    assert result is not None, "Expected a result, not None/raise"
    # system may be 'engine' or 'other' depending on whether pydantic accepts it
    assert result.system in ("engine", "other"), f"Unexpected system: {result.system!r}"
    assert result.cost is None, f"Expected cost=None (malformed cost dropped), got {result.cost!r}"
    assert result.parts == [], f"Expected parts=[] (non-list coerced to empty), got {result.parts!r}"


# ---------------------------------------------------------------------------
# MAINT-02 (rev): code-fence strip in call_extract_json
# ---------------------------------------------------------------------------


def test_extract_strips_code_fences(monkeypatch):
    """A ```json-fenced LLM response parses without JSONDecodeError (fences stripped).

    Drives call_extract_json directly by patching the underlying httpx.post seam.
    Asserts that a markdown-fenced JSON payload does not cause a hard failure.
    """
    import leopard44_kb.maintenance as maint
    import leopard44_kb.maintenance

    fenced_content = '```json\n{"system": "engine"}\n```'

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"message": {"content": fenced_content}}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(leopard44_kb.maintenance.httpx, "post", lambda *a, **kw: FakeResponse())

    result = call_extract_json("some prompt text", "system prompt")

    assert isinstance(result, dict), f"Expected a dict, got {type(result)}"
    assert result.get("system") == "engine", f"Expected system=engine, got {result!r}"


# ---------------------------------------------------------------------------
# MAINT-02 / D-02: date normalisation
# ---------------------------------------------------------------------------


def test_normalise_date():
    """normalise_date handles ISO passthrough, NZ DD/MM, relative, missing, unparseable."""
    today = date(2026, 5, 30)

    # ISO passthrough
    assert normalise_date("2024-03-15", today) == "2024-03-15", "ISO passthrough failed"

    # NZ DD/MM locale — 03/04/2024 → 2024-04-03 (D-02 headline rule: day first, not month first)
    assert normalise_date("03/04/2024", today) == "2024-04-03", (
        "NZ DD/MM rule failed: 03/04/2024 should be April 3 (DD/MM), not March 4 (MM/DD)"
    )

    # Missing → today
    assert normalise_date(None, today) == today.isoformat(), "None should default to today"

    # Relative phrase: "yesterday"
    from datetime import timedelta
    expected_yesterday = (today - timedelta(days=1)).isoformat()
    assert normalise_date("yesterday", today) == expected_yesterday, (
        f"'yesterday' should resolve to {expected_yesterday}"
    )

    # Unparseable string → today
    assert normalise_date("sometime in the winter", today) == today.isoformat(), (
        "Unparseable string should default to today"
    )


# ---------------------------------------------------------------------------
# MAINT-04 / D-06 / Pitfall 6: make_entry_filename
# ---------------------------------------------------------------------------


def test_make_entry_filename():
    """make_entry_filename produces YYYY-MM-DD-slug.md; collision appends -2."""
    # Basic filename
    name = make_entry_filename("replaced port impeller worn out", "2024-03-15", set())
    assert name.startswith("2024-03-15-"), f"Expected date prefix: {name!r}"
    assert name.endswith(".md"), f"Expected .md suffix: {name!r}"
    slug_part = name[len("2024-03-15-"):-len(".md")]
    assert slug_part == slug_part.lower(), f"Slug must be lowercase: {slug_part!r}"
    assert "-" in slug_part or slug_part.isalnum(), f"Slug should be hyphenated: {slug_part!r}"

    # Same-day collision → -2 suffix
    existing = {name}
    name2 = make_entry_filename("replaced port impeller worn out", "2024-03-15", existing)
    assert name2 != name, "Collision must produce a different filename"
    assert name2.endswith("-2.md"), f"Collision should append -2: {name2!r}"


# ---------------------------------------------------------------------------
# MAINT-04 / D-07: parse_maintenance_entry
# ---------------------------------------------------------------------------


def test_parse_maintenance_entry():
    """parse_maintenance_entry returns single chunk with body + metadata dict."""
    fixture = Path("tests/fixtures/sample_maintenance_entry.md")
    source_path = "data/logs/maint/2024-03-15-x.md"

    chunks = parse_maintenance_entry(fixture, source_path)

    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    chunk = chunks[0]

    # Body content present, front-matter stripped
    assert "Replaced port impeller" in chunk["content"], (
        f"Body text missing from content: {chunk['content']!r}"
    )
    assert "source_type: maintenance_entry" not in chunk["content"], (
        "Front-matter key must not appear in content"
    )

    # Metadata keys
    meta = chunk["metadata"]
    assert meta["system"] == "engine", f"Expected system=engine, got {meta['system']!r}"
    assert meta["date"] == "2024-03-15", f"Expected date=2024-03-15, got {meta['date']!r}"
    assert meta["cost_currency"] == "NZD", f"Expected cost_currency=NZD, got {meta['cost_currency']!r}"
    assert isinstance(meta["parts"], list), f"Expected parts list, got {type(meta['parts'])}"
    assert "impeller p/n 22-41016" in meta["parts"], (
        f"Expected impeller part in parts: {meta['parts']!r}"
    )


# ---------------------------------------------------------------------------
# MAINT-04 / D-06/D-07 (rev): write_entry → parse_maintenance_entry punctuation round-trip
# ---------------------------------------------------------------------------


def test_front_matter_round_trip_punctuation(tmp_path):
    """write_entry + parse_maintenance_entry round-trip preserves & : / in field values.

    Pins the shared constrained format so Smith & Sons, filter: WIX 51515,
    and raw-water / cooling survive write→parse without corruption.
    RED until both Plan 02 (write_entry) and Plan 03 (parse_maintenance_entry) land.
    """
    extraction = MaintenanceExtraction(
        date="2024-03-15",
        system="engine",
        system_detail="raw-water / cooling",
        parts=["filter: WIX 51515"],
        cost=CostModel(amount=90.0, currency="NZD"),
        vendor="Smith & Sons",
    )

    written_path = write_entry(extraction, original_text="manifold service", repo_root=tmp_path)

    chunks = parse_maintenance_entry(written_path, str(written_path))
    assert len(chunks) == 1
    meta = chunks[0]["metadata"]

    assert meta["vendor"] == "Smith & Sons", (
        f"Ampersand in vendor not preserved: {meta['vendor']!r}"
    )
    assert meta["parts"] == ["filter: WIX 51515"], (
        f"Colon in part not preserved: {meta['parts']!r}"
    )
    assert meta["system_detail"] == "raw-water / cooling", (
        f"Slash in system_detail not preserved: {meta['system_detail']!r}"
    )


# ---------------------------------------------------------------------------
# MAINT-02 / D-01: system taxonomy normalisation
# ---------------------------------------------------------------------------


def test_system_taxonomy_normalisation():
    """MaintenanceExtraction normalises 'Engine' to 'engine'; unknown value falls back to 'other'."""
    # Mixed case normalised
    e1 = MaintenanceExtraction(system="Engine")
    assert e1.system == "engine", f"Expected 'engine', got {e1.system!r}"

    # Out-of-taxonomy value → 'other'
    e2 = MaintenanceExtraction(system="genset")
    assert e2.system == "other", f"Expected 'other' for unknown taxonomy, got {e2.system!r}"


# ---------------------------------------------------------------------------
# CR-01 (code review): never-raise guarantee holds for wrong-type scalar fields
# ---------------------------------------------------------------------------


def test_extract_never_raises_on_wrong_type_scalar(monkeypatch):
    """Wrong-type date/vendor in LLM output must be sanitised, never raise (D-13).

    Regression for CR-01: _sanitize_payload previously copied date/system_detail/
    vendor through verbatim and the final model_validate was unguarded, so a payload
    like {"date": 123, "vendor": {...}} raised ValidationError out of extract_fields
    and crashed add_cmd (which only catches RuntimeError).
    """
    import leopard44_kb.maintenance as maint

    # Both calls return non-string scalars for typed str fields.
    bad_raw = {"date": 123, "system": "engine", "vendor": {"name": "x"}}
    monkeypatch.setattr(maint, "call_extract_json", lambda prompt, sys_prompt: bad_raw)

    # Must NOT raise.
    result = extract_fields("replaced impeller")

    assert isinstance(result, MaintenanceExtraction), "Expected a MaintenanceExtraction, not a raise"
    assert result.system == "engine", f"Expected system=engine, got {result.system!r}"
    assert result.vendor is None, f"Wrong-type vendor must be dropped to None, got {result.vendor!r}"
    # date dropped to None then normalised to today (never raises).
    assert isinstance(result.date, str) and len(result.date) == 10, (
        f"Expected a normalised ISO date string, got {result.date!r}"
    )


# ---------------------------------------------------------------------------
# WR-01 (code review): omitted fields round-trip as None / [] — not "null" / "[]"
# ---------------------------------------------------------------------------


def test_front_matter_none_fields_round_trip(tmp_path):
    """write_entry + parse of an entry with omitted fields yields None / [] in metadata.

    Regression for WR-01: bareword `null` / `[]` previously parsed back as the literal
    strings "null" / "[]", corrupting chunks.metadata and the `log` command output for
    the common (field-omitted) case. The all-fields-populated fixtures masked this.
    """
    extraction = MaintenanceExtraction(
        date="2024-03-15",
        system="engine",
        system_detail=None,
        parts=[],
        cost=None,
        vendor=None,
    )

    written_path = write_entry(extraction, original_text="generic engine service", repo_root=tmp_path)
    meta = parse_maintenance_entry(written_path, str(written_path))[0]["metadata"]

    assert meta["vendor"] is None, f"omitted vendor must be None, got {meta['vendor']!r}"
    assert meta["system_detail"] is None, f"omitted system_detail must be None, got {meta['system_detail']!r}"
    assert meta["cost_amount"] is None, f"omitted cost_amount must be None, got {meta['cost_amount']!r}"
    assert meta["cost_currency"] is None, f"omitted cost_currency must be None, got {meta['cost_currency']!r}"
    assert meta["parts"] == [], f"empty parts must be [], got {meta['parts']!r}"


def test_front_matter_literal_null_value_preserved(tmp_path):
    """A genuine string value equal to 'null' survives the round-trip (quoted on write).

    Defence-in-depth half of WR-01: write-side reserved-bareword quoting + parse-side
    unquoting disambiguate a real "null" from an omitted (None) field.
    """
    extraction = MaintenanceExtraction(
        date="2024-03-15",
        system="engine",
        vendor="null",
    )

    written_path = write_entry(extraction, original_text="x", repo_root=tmp_path)
    meta = parse_maintenance_entry(written_path, str(written_path))[0]["metadata"]

    assert meta["vendor"] == "null", f"a genuine 'null' string vendor must survive, got {meta['vendor']!r}"


# ---------------------------------------------------------------------------
# WR-02 (code review): system normalisation uses whole-word, not substring, matching
# ---------------------------------------------------------------------------


def test_normalise_system_no_false_substring():
    """normalise_system must not misfile via bidirectional substring matching.

    Regression for WR-02: "ac" matched "ground tACkle" and single letters matched
    arbitrary systems, breaking the `log --system` exact-match filter.
    """
    # "ac" (air-con) must fall back to "other", NOT "ground tackle"
    assert MaintenanceExtraction(system="ac").system == "other", "ac must not become ground tackle"
    # Single letters must not misfile
    assert MaintenanceExtraction(system="e").system == "other"
    assert MaintenanceExtraction(system="a").system == "other"
    # Whole-word containment still resolves sensibly
    assert MaintenanceExtraction(system="engine cooling").system == "engine"
    assert MaintenanceExtraction(system="nav").system == "electronics/nav"
