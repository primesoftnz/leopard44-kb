# RED state until Phase 2 implementation (see 02-VALIDATION.md). Imports from leopard44_kb.ingest.* fail until production code lands.
"""Tests for INGEST-02/04: CLI ingest command — layer defaulting, WhatsApp vs plain-text dispatch, PDF layer."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlite_vec
from typer.testing import CliRunner

from leopard44_kb.cli import app
from leopard44_kb.schema import apply_migrations

runner = CliRunner()


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Open a fully-bootstrapped test DB at db_path."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    conn.close()
    return sqlite3.connect(str(db_path))


def test_default_layer_vessel(monkeypatch, fake_embedder, tmp_path):
    """ingest without --layer defaults to layer='vessel' in DB (D-06)."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))

    data_dir = tmp_path / "data" / "logs"
    data_dir.mkdir(parents=True)
    note = data_dir / "vessel_default.md"
    note.write_text("# Default Layer Test\n\nContent.\n", encoding="utf-8")

    result = runner.invoke(app, ["ingest", str(note)])
    assert result.exit_code == 0, (
        f"Expected exit 0; got {result.exit_code}: {result.output}"
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    src = conn.execute("SELECT layer FROM sources").fetchone()
    conn.close()
    assert src is not None, "No source row stored"
    assert src["layer"] == "vessel", f"Expected layer='vessel'; got {src['layer']!r}"


def test_whatsapp_layer_shared(monkeypatch, fake_embedder, tmp_path):
    """ingest --layer shared for a WhatsApp zip stores layer='shared' in DB."""
    import shutil

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))

    shared_dir = tmp_path / "shared" / "leopard44"
    shared_dir.mkdir(parents=True)

    fixtures = Path(__file__).parent / "fixtures"
    zip_dst = shared_dir / "sample_chat.zip"
    shutil.copy(fixtures / "sample_chat.zip", zip_dst)

    result = runner.invoke(app, ["ingest", str(zip_dst), "--layer", "shared"])
    assert result.exit_code == 0, (
        f"Expected exit 0; got {result.exit_code}: {result.output}"
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    src = conn.execute("SELECT layer FROM sources").fetchone()
    conn.close()
    assert src is not None
    assert src["layer"] == "shared", f"Expected layer='shared'; got {src['layer']!r}"


def test_pdf_layer_shared(monkeypatch, fake_embedder, tmp_path):
    """ingest --layer shared for a PDF stores layer='shared' in DB."""
    import shutil

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))

    shared_dir = tmp_path / "shared" / "yanmar"
    shared_dir.mkdir(parents=True)

    fixtures = Path(__file__).parent / "fixtures"
    pdf_dst = shared_dir / "sample_toc.pdf"
    shutil.copy(fixtures / "synthetic" / "sample_toc.pdf", pdf_dst)

    result = runner.invoke(app, ["ingest", str(pdf_dst), "--layer", "shared"])
    assert result.exit_code == 0, (
        f"Expected exit 0; got {result.exit_code}: {result.output}"
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    src = conn.execute("SELECT layer, source_type FROM sources").fetchone()
    conn.close()
    assert src is not None
    assert src["layer"] == "shared"
    assert src["source_type"] == "pdf"


def test_whatsapp_txt_dispatch(monkeypatch, fake_embedder, tmp_path):
    """END-TO-END ROADMAP SC1: a WhatsApp-format .txt in data/whatsapp/ with no --layer
    ingests as source_type='whatsapp' AND layer='vessel'.

    This is the exact success criterion: drop a WhatsApp .txt into data/whatsapp/ and it
    parses as WhatsApp, not generic text (Codex HIGH concern / ROADMAP SC1).
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))

    wa_dir = tmp_path / "data" / "whatsapp"
    wa_dir.mkdir(parents=True)

    fixtures = Path(__file__).parent / "fixtures"
    import shutil

    wa_file = wa_dir / "leopard44_owners_group.txt"
    shutil.copy(fixtures / "sample_chat_android.txt", wa_file)

    # No --layer → must default to vessel AND dispatch to whatsapp parser
    result = runner.invoke(app, ["ingest", str(wa_file)])
    assert result.exit_code == 0, (
        f"Expected exit 0 for WhatsApp .txt dispatch; got {result.exit_code}: {result.output}"
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    src = conn.execute("SELECT source_type, layer FROM sources").fetchone()
    conn.close()

    assert src is not None, "No source row stored after WhatsApp .txt ingest"
    assert src["source_type"] == "whatsapp", (
        f"Expected source_type='whatsapp' for file in data/whatsapp/; "
        f"got {src['source_type']!r}"
    )
    assert src["layer"] == "vessel", (
        f"Expected layer='vessel' (default); got {src['layer']!r}"
    )


def test_plain_txt_dispatch(monkeypatch, fake_embedder, tmp_path):
    """A non-WhatsApp .txt NOT under a whatsapp/ path ingests as source_type='text'."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("L44_DB", str(db_path))

    logs_dir = tmp_path / "data" / "logs"
    logs_dir.mkdir(parents=True)
    plain_txt = logs_dir / "plain_notes.txt"
    plain_txt.write_text(
        "First paragraph of plain text notes.\n\n"
        "Second paragraph.\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["ingest", str(plain_txt)])
    assert result.exit_code == 0, (
        f"Expected exit 0 for plain .txt dispatch; got {result.exit_code}: {result.output}"
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    src = conn.execute("SELECT source_type, layer FROM sources").fetchone()
    conn.close()

    assert src is not None
    assert src["source_type"] == "text", (
        f"Expected source_type='text' for non-WhatsApp .txt; got {src['source_type']!r}"
    )
    assert src["layer"] == "vessel"
