# Wave 0 RED state until Plan 09-02 ships schematic.py and cli.py render command.
# Imports from leopard44_kb.schematic / leopard44_kb.cli are INSIDE each function body —
# this ensures pytest collection succeeds even before the production code exists
# (RED = import/attribute failure or assertion failure, NOT collection crash).
"""Tests for ZONE-02: schematic render service + CLI (Phase 9 Wave 0 RED gate).

Per-requirement map source:
  .planning/phases/09-schematic-rendering-zone-annotation-visual-highlight/09-VALIDATION.md

Contracts pinned:
  - render_pages(pdf_path, page_numbers, output_dir) -> list[Path]
    * 1-based page numbers; output files named page_{NNN:03d}.png
  - parse_page_spec(spec, page_count=None) -> list[int]
    * rejects 0, negatives, reversed ranges, non-integers, out-of-range vs page_count
  - CLI: l44 schematic render <pdf> --pages <range/list>

Synthetic PDF fixture (D-02 / D-04 review fix; Codex HIGH concern 4):
  _make_synthetic_pdf builds a real multi-page PDF IN-TEST via PyMuPDF so the
  ZONE-02 RED gate is real on a clean checkout/CI (not dependent on the private
  copyrighted Owner's Manual).
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF — already a dependency
import pytest


# ---------------------------------------------------------------------------
# Synthetic PDF fixture helper (Codex HIGH + review concern 4)
# Must use real PyMuPDF API so the PDF is valid for fitz rendering.
# ---------------------------------------------------------------------------


def _make_synthetic_pdf(path: Path, n_pages: int) -> Path:
    """Build a real multi-page PDF at *path* using PyMuPDF.

    Creates n_pages blank pages (each valid for rasterisation via get_pixmap()).
    Returns path for chaining convenience.

    No dependency on copyrighted vessel data — safe for clean checkout + CI.
    """
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page()
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# ZONE-02: render_pages — core rendering contract
# ---------------------------------------------------------------------------


def test_render_pages(tmp_path):
    """render_pages() writes exactly one PNG named page_061.png to output_dir.

    Uses a SYNTHETIC PyMuPDF PDF (90 pages) — no dependency on the private manual.
    Asserts: exactly one path returned, name == page_061.png, file exists, size > 0.
    """
    from leopard44_kb.schematic import render_pages  # RED until 09-02

    pdf_path = _make_synthetic_pdf(tmp_path / "synthetic.pdf", n_pages=90)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    written = render_pages(pdf_path, [61], out_dir)

    assert len(written) == 1, f"Expected 1 file written; got {written!r}"
    assert written[0].name == "page_061.png", (
        f"Expected page_061.png; got {written[0].name!r}"
    )
    assert written[0].exists(), f"Output file does not exist: {written[0]}"
    assert written[0].stat().st_size > 0, "Output PNG is empty"


def test_png_naming(tmp_path):
    """render_pages() uses zero-padded page_{NNN:03d}.png naming convention.

    Renders pages [5, 12] from a synthetic PDF; asserts page_005.png and
    page_012.png both exist in the output directory.
    """
    from leopard44_kb.schematic import render_pages  # RED until 09-02

    pdf_path = _make_synthetic_pdf(tmp_path / "synthetic.pdf", n_pages=20)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    written = render_pages(pdf_path, [5, 12], out_dir)

    names = {p.name for p in written}
    assert "page_005.png" in names, f"Expected page_005.png in {names!r}"
    assert "page_012.png" in names, f"Expected page_012.png in {names!r}"
    assert len(written) == 2, f"Expected exactly 2 files; got {written!r}"


def test_render_real_manual_smoke(tmp_path):
    """Optional smoke: render page 61 of the real Owner's Manual if present.

    pytest.skip()s when the private copyrighted PDF is absent (clean checkout/CI).
    This is NOT the required RED gate — test_render_pages uses the synthetic fixture.
    """
    from leopard44_kb.schematic import render_pages  # RED until 09-02

    manual = Path(
        "data/sources/leopard44_factory/L44 A5160 Owner's Manual Sunsail.pdf"
    )
    if not manual.exists():
        pytest.skip("Owner's Manual PDF not present in vessel layer")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    written = render_pages(manual, [61], out_dir)
    assert len(written) == 1
    assert written[0].stat().st_size > 0, "Real-manual render produced empty PNG"


# ---------------------------------------------------------------------------
# ZONE-02: parse_page_spec — validation contract
# ---------------------------------------------------------------------------


def test_parse_page_spec_valid():
    """parse_page_spec returns sorted unique 1-based ints for valid range + list."""
    from leopard44_kb.schematic import parse_page_spec  # RED until 09-02

    result = parse_page_spec("61-63,65")
    assert result == [61, 62, 63, 65], f"Expected [61, 62, 63, 65]; got {result!r}"


@pytest.mark.parametrize(
    "spec,page_count,description",
    [
        ("0", None, "page 0 is invalid (1-based)"),
        ("-3", None, "negative page number"),
        ("63-61", None, "reversed range"),
        ("abc", None, "non-integer spec"),
        ("61.5", None, "float page number"),
        ("999", 90, "page > page_count"),
    ],
)
def test_parse_page_spec_rejects(spec, page_count, description):
    """parse_page_spec raises (ValueError or typer error) for invalid inputs.

    Covers: page 0, negatives, reversed ranges, non-integers, floats,
    and pages exceeding the document page count.
    """
    from leopard44_kb.schematic import parse_page_spec  # RED until 09-02
    import typer

    kwargs = {} if page_count is None else {"page_count": page_count}
    with pytest.raises((ValueError, typer.BadParameter, SystemExit)):
        parse_page_spec(spec, **kwargs)


# ---------------------------------------------------------------------------
# ZONE-02: CLI render command
# ---------------------------------------------------------------------------


def test_render_cli(monkeypatch, tmp_path):
    """l44 schematic render <pdf> --pages 61 exits 0 and writes page_061.png.

    Uses a synthetic PDF so no private data is required.
    Monkeypatches L44_DB and chdir to tmp_path so schematics land in
    tmp_path/data/schematics/page_061.png (relative to cwd).
    """
    from typer.testing import CliRunner
    from leopard44_kb.cli import app  # RED until 09-02

    pdf_path = _make_synthetic_pdf(tmp_path / "synthetic.pdf", n_pages=90)

    # Create the expected data/schematics/ directory under tmp_path
    (tmp_path / "data" / "schematics").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["schematic", "render", str(pdf_path), "--pages", "61"])

    assert result.exit_code == 0, (
        f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}"
    )
    expected_png = tmp_path / "data" / "schematics" / "page_061.png"
    assert expected_png.exists(), (
        f"page_061.png not found at {expected_png}\nCLI output:\n{result.output}"
    )
    assert expected_png.stat().st_size > 0, "Rendered PNG is empty"
