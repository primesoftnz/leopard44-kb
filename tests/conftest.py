"""Shared pytest fixtures for the Leopard 44 KB test suite.

All database fixtures use :memory: or tmp_path — no fixture ever touches
~/.local/share/leopard44-kb/ or the real repo data/ directory.
"""
from __future__ import annotations

import sqlite3

import pytest
import sqlite_vec

from leopard44_kb.schema import apply_migrations


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch, tmp_path):
    """Point L44_CONFIG at a nonexistent path for EVERY test (config isolation).

    Model selection is config-first (answer.select_generation_model →
    config.load_config), so a developer/contributor who has run `setup` — writing
    ~/.local/share/leopard44-kb/config.json, e.g. a GPU-tier 14B config — would
    otherwise leak that installed tier into the RAM-fallback tests and fail them.
    Defaulting L44_CONFIG to a missing file makes load_config() return None, so the
    suite is deterministic regardless of the host's installed config. Tests that need
    a specific config call monkeypatch.setenv('L44_CONFIG', ...) themselves; that runs
    after this autouse fixture (same function-scoped monkeypatch) and overrides it.
    """
    monkeypatch.setenv("L44_CONFIG", str(tmp_path / "no-such-config.json"))


@pytest.fixture
def empty_db() -> sqlite3.Connection:
    """In-memory SQLite connection with sqlite-vec loaded and migrations applied.

    Sequence per RESEARCH.md Pattern 2 + Pitfall 4:
    1. enable_load_extension(True)
    2. sqlite_vec.load(conn)          — uses platform-native helper
    3. enable_load_extension(False)   — security: prevent arbitrary .so loads
    4. PRAGMA foreign_keys = ON       — per-connection, off by default
    5. apply_migrations(conn)         — run schema/001_init.sql
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    yield conn
    conn.close()


@pytest.fixture
def repo_root(tmp_path):
    """Synthetic repo root with data/ and shared/ subdirectories.

    Used by test_paths.py and test_install_data_dirs.py to keep tests
    isolated from the real working tree.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "shared").mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def fake_embedder(monkeypatch):
    """Replace Ollama embed call with a deterministic stub returning 384-dim vectors.

    Monkeypatches leopard44_kb.ingest.embedder.embed_texts and select_model so every
    parsing/chunking/DB test runs without a live Ollama instance.
    embed_texts returns [[0.1] * 384 for _ in texts] per 02-RESEARCH lines 763-776.
    """
    import leopard44_kb.ingest.embedder as emb

    monkeypatch.setattr(
        emb,
        "embed_texts",
        lambda texts, model: [[0.1] * 384 for _ in texts],
    )
    monkeypatch.setattr(
        emb,
        "select_model",
        lambda: ("nomic-embed-text:v1.5", "v1.5"),
    )


@pytest.fixture
def ingest_db(tmp_path) -> sqlite3.Connection:
    """In-memory SQLite connection for ingest tests.

    Same bootstrap sequence as empty_db; kept as a separate fixture so ingest
    tests are labelled distinctly in pytest output.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="session", autouse=True)
def _offline_token_fallback():
    """Offline tiktoken fallback guard (session-scoped, autouse).

    If tiktoken.get_encoding('cl100k_base') fails at session start (no cache
    + no network), monkeypatches leopard44_kb.ingest.text_md.count_tokens (and
    any other production path that calls count_tokens) to a deterministic
    whitespace+punctuation estimator so chunk-boundary tests stay deterministic
    offline. When the real encoder is available it is preferred; this fallback
    fires only when it would otherwise raise.
    """
    try:
        import tiktoken as _tk

        _tk.get_encoding("cl100k_base")
        # Real encoder available — no patching needed.
    except Exception:
        # No cache and no network. Provide a deterministic fallback.
        import importlib
        import re

        def _heuristic_count(text: str) -> int:
            """Rough token estimate: whitespace-split + punctuation tokens."""
            words = text.split()
            punct = len(re.findall(r"[^\w\s]", text))
            return len(words) + punct // 2

        # Patch the production import path used by parsers.
        try:
            import leopard44_kb.ingest.text_md as _text_md

            _text_md.count_tokens = _heuristic_count  # type: ignore[attr-defined]
        except ImportError:
            pass  # module not yet implemented — RED state, nothing to patch


@pytest.fixture
def repo_root_with_shared(tmp_path):
    """Synthetic repo root with shared/ topic subdirectories.

    Used by test_shared_layout.py when testing against an in-memory
    layout rather than the real committed tree.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "data").mkdir(parents=True, exist_ok=True)
    shared = root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    for subdir in ("leopard44", "yanmar", "systems", "upgrades"):
        (shared / subdir).mkdir(parents=True, exist_ok=True)
        (shared / subdir / ".gitkeep").touch()
    return root


@pytest.fixture
def fake_generator(monkeypatch):
    """Replace Ollama generation with a deterministic stub.

    Yields tokens that include [1] and [2] citation markers so citation
    validation and stream-order tests are meaningful without calling Ollama.
    The import of leopard44_kb.answer is inside the fixture body so the fixture
    only resolves once leopard44_kb.answer exists (Wave 2) — RED until then.
    """
    import leopard44_kb.answer as ans

    def _fake_stream(model, system_prompt, user_message, num_predict=75, temperature=0.15):
        tokens = [
            "The ", "impeller ", "interval ", "is ", "200 ", "hours ", "[1]. ",
            "Owner ", "replaced ", "it ", "in ", "2024 ", "[2].",
        ]
        yield from tokens
        return {"model": model, "eval_count": len(tokens), "total_duration_ns": 1_000_000_000}

    monkeypatch.setattr(ans, "stream_generate", _fake_stream)


@pytest.fixture
def out_of_range_generator(monkeypatch):
    """Replace Ollama generation with a stub that emits an out-of-range citation.

    Token stream contains a valid [1] AND an out-of-range [9] so the CLI
    citation-validation test can assert [9] is stripped/warned before the
    citation block renders (review fix #4).
    """
    import leopard44_kb.answer as ans

    def _bad_stream(model, system_prompt, user_message, num_predict=75, temperature=0.15):
        tokens = [
            "The ", "impeller ", "part ", "is ", "[1]. ",
            "See ", "also ", "ref ", "[9].",
        ]
        yield from tokens
        return {"model": model, "eval_count": len(tokens), "total_duration_ns": 1_000_000_000}

    monkeypatch.setattr(ans, "stream_generate", _bad_stream)


@pytest.fixture
def fake_extractor(monkeypatch):
    """Replace Ollama extraction with a deterministic stub returning a fixed MaintenanceExtraction.

    Monkeypatches leopard44_kb.maintenance.extract_fields by module reference so the patch
    survives the lazy-import call style used by add_cmd. The import of leopard44_kb.maintenance
    is inside the fixture body so it only resolves once the module exists (RED until Plan 02).
    Fixed values: system="engine", vendor="Burnsco", cost.amount=45.0, cost.currency="NZD",
    date="2024-03-15", parts=["impeller p/n 22-41016"].
    """
    import leopard44_kb.maintenance as maint
    from leopard44_kb.maintenance import CostModel, MaintenanceExtraction

    fixed = MaintenanceExtraction(
        date="2024-03-15",
        system="engine",
        system_detail="raw-water cooling",
        parts=["impeller p/n 22-41016"],
        cost=CostModel(amount=45.0, currency="NZD"),
        vendor="Burnsco",
    )
    monkeypatch.setattr(maint, "extract_fields", lambda text: fixed)


@pytest.fixture
def retrieval_db(empty_db):
    """In-memory DB with the canonical 3-chunk retrieval corpus.

    The corpus and the embedding ``pack`` helper live in tests/_corpus.py so the
    in-memory unit-test path (here) and the file-backed CLI path (_seed_db in
    test_query_cli.py) cannot drift (IN-05). See _corpus.seed_corpus for the full
    corpus description.
    """
    from tests._corpus import seed_corpus

    seed_corpus(empty_db)
    return empty_db


@pytest.fixture
def fake_zone_ai(monkeypatch):
    """Stub the Ollama zone-description call in inventory.py.

    Monkeypatches leopard44_kb.inventory.call_extract_json by module reference so the
    patch survives the lazy-import call style used by zone_add_cmd. The import of
    leopard44_kb.inventory is INSIDE the fixture body (not at module top) so this fixture
    does not break collection in the pre-inventory.py RED state — it only resolves
    once inventory.py exists in Wave 2. Returns a fixed vertical_desc so zone tests
    that exercise the use_ai=True branch run without a live Ollama.
    """
    import leopard44_kb.inventory as inv  # lazy: only resolves once inventory.py exists (RED until Wave 2)

    monkeypatch.setattr(
        inv,
        "call_extract_json",
        lambda prompt, sys_prompt: {"vertical_desc": "Lower shelf, port side"},
    )


@pytest.fixture
def voice_container_fixture():
    """Return the path to the container-only plumbing fixture for /transcribe tests.

    This fixture is for endpoint shape tests (blob upload, size limit, 503 routing).
    It is NOT a speech clip — do NOT use it for slow STT worker tests.
    """
    from pathlib import Path

    return Path(__file__).parent / "fixtures" / "valid_container.webm"


@pytest.fixture
def voice_speech_fixture():
    """Return the path to the real-speech fixture for @pytest.mark.slow STT tests.

    This is a real ~2-3s speech clip — intended ONLY for slow worker integration
    tests that actually invoke .venv-stt. Never use it in default-run tests.
    """
    from pathlib import Path

    return Path(__file__).parent / "fixtures" / "short_marine_query.webm"


@pytest.fixture
def db_env(monkeypatch, tmp_path):
    """Set L44_DB to an isolated tmp path for voice/serve test isolation.

    Mirrors the monkeypatch.setenv("L44_DB", ...) pattern used in
    test_web_offline.py and test_web_serve.py — shared here for reuse by
    test_web_voice.py, test_voice_setup.py, and test_stt_worker.py.
    """
    db_path = tmp_path / "test_l44.db"
    monkeypatch.setenv("L44_DB", str(db_path))
    return db_path


@pytest.fixture
def fake_deviation_extractor(monkeypatch):
    """Replace Ollama extraction with a deterministic stub returning a fixed DeviationExtraction.

    Monkeypatches leopard44_kb.deviation.extract_fields by module reference so the patch
    survives the lazy-import call style used by deviation add_cmd. The import of
    leopard44_kb.deviation is INSIDE the fixture body so it only resolves once the module
    exists (RED until Plan 11-02) — tests that do NOT request this fixture still
    collect cleanly.
    Fixed values: component="windlass", factory_spec="12V Muir 1200W", as_built="12V Maxwell 1000W",
    reason="replacement after failure", date_noted="2024-01-10".
    """
    import leopard44_kb.deviation as dev  # lazy: only resolves once deviation.py exists (RED until 11-02)
    from leopard44_kb.deviation import DeviationExtraction

    fixed = DeviationExtraction(
        component="windlass",
        factory_spec="12V Muir 1200W",
        as_built="12V Maxwell 1000W",
        reason="replacement after failure",
        date_noted="2024-01-10",
    )
    monkeypatch.setattr(dev, "extract_fields", lambda text: fixed)


@pytest.fixture
def gps_exif_jpeg(tmp_path):
    """Write a small RGB JPEG with a GPS IFD embedded via piexif.

    Returns a Path to the JPEG. The GPS IFD contains GPSLatitudeRef, GPSLatitude,
    GPSLongitudeRef, and GPSLongitude so that the capture EXIF-strip test has a
    non-empty GPS block to verify was removed.

    Coordinates are fabricated constants (not real vessel location).
    """
    import io
    import piexif
    from PIL import Image

    # 32×32 red square — small but valid
    img = Image.new("RGB", (32, 32), color=(200, 50, 50))

    # Build a minimal GPS IFD with fabricated coords
    # GPSLatitudeRef: "S", GPSLatitude: 36°51'0" (Auckland-ish, fabricated)
    # GPSLongitudeRef: "E", GPSLongitude: 174°45'0"
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b"S",
        piexif.GPSIFD.GPSLatitude: ((36, 1), (51, 1), (0, 1)),
        piexif.GPSIFD.GPSLongitudeRef: b"E",
        piexif.GPSIFD.GPSLongitude: ((174, 1), (45, 1), (0, 1)),
    }
    exif_bytes = piexif.dump({"GPS": gps_ifd})

    out_path = tmp_path / "gps_test.jpg"
    img.save(str(out_path), format="JPEG", exif=exif_bytes)
    return out_path


@pytest.fixture
def heic_photo(tmp_path):
    """Write a small HEIC image using pillow-heif and return its Path.

    Registers the HEIF opener on entry. The fixture verifies that plain Pillow
    CANNOT decode HEIC without the opener (the production guard) and that WITH the
    opener it can. Returns the Path to the .heic file — the HEIC decode test input.
    """
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()

    # 32×32 blue square
    img = Image.new("RGB", (32, 32), color=(50, 100, 200))

    out_path = tmp_path / "test_photo.heic"
    img.save(str(out_path), format="HEIF")
    return out_path


@pytest.fixture
def seeded_zone_db(empty_db, tmp_path):
    """empty_db with one zone pre-inserted via direct SQL (no CLI, no AI call).

    Uses a raw INSERT (not zone_add_cmd) so the fixture works during Wave 0 RED
    state before zone_add_cmd is implemented. The INSERT depends on the
    002_inventory.sql migration existing (Wave 2 Plan 02) — until then the INSERT
    fails because the zones table is absent, which is the expected RED state for
    tests that use this fixture.

    Returns the same empty_db connection with the zone row committed.
    """
    with empty_db:
        empty_db.execute(
            "INSERT INTO zones (name, label, side, fore_aft, area, vertical_index) "
            "VALUES ('saloon-port-locker', 'Saloon port locker', 'port', 'mid', 'saloon', 1.0)"
        )
    return empty_db
