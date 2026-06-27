"""Tests for Phase 7 deployment constraints: D-04 concurrency cap, D-05 alpha banner,
offline contract, vessel-layer isolation."""
from __future__ import annotations

import json
import re
import threading

import pytest


# ---------------------------------------------------------------------------
# Module-level SSE parsing helper — NO leopard44_kb.web import (collection-safe).
# Copied verbatim from tests/test_web_query.py (not imported cross-module).
# ---------------------------------------------------------------------------


def _parse_sse_events(client, method: str, url: str, **kwargs) -> list[tuple[str, str]]:
    """Stream a request and return ordered list of (event_name, data) tuples.

    Uses client.stream() + resp.iter_lines() to parse the SSE wire format.
    Handles multi-line event blocks correctly: event: lines set the current
    event name; data: lines accumulate; the blank line (event boundary) emits
    the (name, joined_data) pair and resets state.

    SSE spec: a blank line terminates the event; multiple data: lines within
    a single event are joined with '\\n'. Resetting current_event on every
    data: line (as the old helper did) mis-attributed the second data: line
    of a multi-line token event to "message".
    """
    events: list[tuple[str, str]] = []
    current_event = "message"
    pending_data: list[str] = []
    with client.stream(method, url, **kwargs) as resp:
        for line in resp.iter_lines():
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: "):
                pending_data.append(line[6:])
            elif line == "":
                # blank line = event boundary per SSE spec
                if pending_data:
                    events.append((current_event, "\n".join(pending_data)))
                    pending_data = []
                    current_event = "message"
    return events


# ---------------------------------------------------------------------------
# D-04: at-capacity test — third request gets error+done, no token
# ---------------------------------------------------------------------------


def test_concurrency_cap(monkeypatch, tmp_path):
    """D-04: a third in-flight request gets error+done SSE when cap is 2.

    Saturates web_app._llm_semaphore by acquiring it directly twice, then
    POSTs /query and asserts: error event present, done event present, no
    token events. Releases both acquisitions in a finally.
    """
    import leopard44_kb.web.app as web_app

    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    # Saturate the semaphore directly (holds 2 slots)
    web_app._llm_semaphore.acquire()
    web_app._llm_semaphore.acquire()
    try:
        events = _parse_sse_events(
            client, "POST", "/query",
            json={"question": "ping", "layer": "all"},
        )
    finally:
        web_app._llm_semaphore.release()
        web_app._llm_semaphore.release()

    event_names = [name for name, _ in events]
    assert "error" in event_names, "Expected error SSE when cap is at capacity"
    assert "done" in event_names, "Expected done SSE after error"
    assert "token" not in event_names, "No token events when cap is at capacity"


# ---------------------------------------------------------------------------
# D-04: cap-holds-across-two-concurrent-streams test
# ---------------------------------------------------------------------------


def test_concurrency_cap_holds_across_streams(monkeypatch, tmp_path):
    """D-04 (verified HIGH finding): cap is held for the whole stream duration.

    Starts two concurrent streaming /query requests that block mid-stream
    (stream_generate yields one token then blocks on a threading.Event), fires
    a third request while both are still in-flight, and asserts the third gets
    error+done SSE (not token). This proves the semaphore is held across the
    WHOLE stream, not just generator construction.

    Both retrieve and stream_generate are monkeypatched so the test runs
    without a live DB or Ollama.

    The module-level _llm_semaphore is replaced with a fresh BoundedSemaphore(2)
    for this test to ensure isolation from prior tests in the same process.
    Each streaming thread uses its OWN TestClient instance (separate anyio
    portal) so both streams run concurrently in anyio's threadpool.
    """
    import leopard44_kb.web.app as web_app
    import leopard44_kb.answer as ans
    import leopard44_kb.retrieve as ret

    # Replace the module-level semaphore with a fresh one for isolation.
    # This ensures prior tests (which acquire/release _llm_semaphore directly)
    # cannot leave this test in a depleted state.
    fresh_sem = threading.BoundedSemaphore(2)
    monkeypatch.setattr(web_app, "_llm_semaphore", fresh_sem)

    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    # Event to block the two in-flight generators mid-stream
    unblock_event = threading.Event()
    # Counting semaphore to track when each thread is mid-stream
    mid_stream_count = threading.Semaphore(0)

    # Fake chunk so below_floor is False and the endpoint reaches stream_generate
    fake_chunk = {
        "id": 1, "source_id": 1, "layer": "shared", "ordinal": 0,
        "section_path": "test", "page_start": 0, "page_end": 0,
        "content": "test content", "title": "test",
    }

    def _fake_retrieve(conn, question, layers, n=5):
        return ([fake_chunk], False)

    monkeypatch.setattr(ret, "retrieve", _fake_retrieve)

    # Patch open_db and apply_migrations to avoid SQLite file-locking when two
    # threads open the same DB concurrently. This test is about the semaphore,
    # not DB access.
    import leopard44_kb.store as store_mod
    import leopard44_kb.schema as schema_mod
    from unittest.mock import MagicMock
    monkeypatch.setattr(store_mod, "open_db", lambda: MagicMock())
    monkeypatch.setattr(schema_mod, "apply_migrations", lambda conn: None)

    def _blocking_stream(*args, **kwargs):
        """Yield one token, signal mid-stream, then block until unblock_event."""
        yield "token-1"
        mid_stream_count.release()  # signal: this generator is now mid-flight
        unblock_event.wait(timeout=15)

    monkeypatch.setattr(ans, "stream_generate", _blocking_stream)

    # Shared app instance — module-level _llm_semaphore (now fresh_sem) is
    # shared across all TestClient instances backed by the same app object
    app = create_app()

    thread1_events: list = []
    thread2_events: list = []
    thread1_exc: list = []
    thread2_exc: list = []

    def _run_stream(result_list, exc_list):
        # Each thread creates its own TestClient (= its own anyio portal) so
        # both streams run concurrently rather than serialising on one portal
        client = TestClient(app, raise_server_exceptions=False)
        try:
            events = _parse_sse_events(
                client, "POST", "/query",
                json={"question": "hold", "layer": "all"},
            )
            result_list.extend(events)
        except Exception as e:
            exc_list.append(e)

    t1 = threading.Thread(target=_run_stream, args=(thread1_events, thread1_exc), daemon=True)
    t2 = threading.Thread(target=_run_stream, args=(thread2_events, thread2_exc), daemon=True)

    t1.start()
    t2.start()

    try:
        # Wait until both streaming requests are mid-stream (past the semaphore acquire,
        # each has yielded one token and released mid_stream_count)
        assert mid_stream_count.acquire(timeout=10), "Thread 1 did not reach mid-stream in time"
        assert mid_stream_count.acquire(timeout=10), "Thread 2 did not reach mid-stream in time"

        # Now the LLM semaphore has 0 available slots; fire the third request
        client3 = TestClient(app, raise_server_exceptions=False)
        third_events = _parse_sse_events(
            client3, "POST", "/query",
            json={"question": "should-be-busy", "layer": "all"},
        )

        third_names = [name for name, _ in third_events]
        assert "error" in third_names, (
            f"Third request must get error SSE while two streams are in-flight; got: {third_names!r}"
        )
        assert "done" in third_names, (
            f"Third request must get done SSE; got: {third_names!r}"
        )
        assert "token" not in third_names, (
            f"Third request must not get token events; got: {third_names!r}"
        )
    finally:
        # Unblock both in-flight streams and join threads
        unblock_event.set()
        t1.join(timeout=15)
        t2.join(timeout=15)

    assert not thread1_exc, f"Thread 1 raised: {thread1_exc}"
    assert not thread2_exc, f"Thread 2 raised: {thread2_exc}"


# ---------------------------------------------------------------------------
# D-04: busy path triggers no retrieve() / stream_generate() side-effects
# ---------------------------------------------------------------------------


def test_busy_path_no_side_effects(monkeypatch, tmp_path):
    """D-04: over-capacity path must NOT call retrieve or stream_generate.

    Saturates the semaphore directly, monkeypatches both lazy-import targets
    to raise AssertionError if called, POSTs /query, asserts error+done and
    both mocks have call_count 0.
    """
    import leopard44_kb.web.app as web_app
    from unittest.mock import Mock

    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))

    retrieve_mock = Mock(side_effect=AssertionError("retrieve must not be called when busy"))
    stream_mock = Mock(side_effect=AssertionError("stream_generate must not be called when busy"))

    monkeypatch.setattr("leopard44_kb.retrieve.retrieve", retrieve_mock)
    monkeypatch.setattr("leopard44_kb.answer.stream_generate", stream_mock)

    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    web_app._llm_semaphore.acquire()
    web_app._llm_semaphore.acquire()
    try:
        events = _parse_sse_events(
            client, "POST", "/query",
            json={"question": "ping", "layer": "all"},
        )
    finally:
        web_app._llm_semaphore.release()
        web_app._llm_semaphore.release()

    event_names = [name for name, _ in events]
    assert "error" in event_names, "Expected error SSE on busy path"
    assert "done" in event_names, "Expected done SSE on busy path"
    assert retrieve_mock.call_count == 0, (
        f"retrieve must not be called on busy path; was called {retrieve_mock.call_count} times"
    )
    assert stream_mock.call_count == 0, (
        f"stream_generate must not be called on busy path; was called {stream_mock.call_count} times"
    )


# ---------------------------------------------------------------------------
# D-05: alpha banner present in GET /
# ---------------------------------------------------------------------------


def test_alpha_banner(monkeypatch, tmp_path):
    """D-05: GET / HTML contains the alpha banner element and an ALPHA notice."""
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'class="alpha-banner"' in resp.text, (
        "Alpha banner div (class='alpha-banner') not found in GET / HTML"
    )
    assert "alpha" in resp.text.lower(), "Alpha notice text not found in GET / HTML"


# ---------------------------------------------------------------------------
# D-05: alpha banner introduces no external origins (offline contract preserved)
# ---------------------------------------------------------------------------


def test_no_external_origins(monkeypatch, tmp_path):
    """D-05: GET / HTML contains zero external https:// origins.

    The ?v=alpha cache-bust is a same-origin query string — NOT an external origin.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    external = re.findall(
        r'(?:src|href|@import|url)\s*[=(]\s*["\']?(https?://(?!127\.0\.0\.1|localhost)[^\s"\']+)',
        html,
        re.IGNORECASE,
    )
    assert external == [], f"GET / HTML references external origins: {external}"


# ---------------------------------------------------------------------------
# PKG-05 HYGIENE: rename regression guards (D-12/D-13/D-14)
# ---------------------------------------------------------------------------


def test_no_eurybia_env_residual():
    """PKG-05: zero 'eurybia' occurrences (any casing) in the SHIPPED functional tree.

    GATE SCOPE (Codex HIGH 13-01): covers ONLY src/leopard44_kb/, scripts/,
    schema/*.sql, and the SHIPPED deploy/leopard44-kb.service. Explicitly excludes:
    - deploy.sh, deploy/DEPLOYMENT.md, deploy/ispconfig-vhost.md (non-shipped dev assets)
    - README.md, CONTRIBUTING.md, shared/README.md (prose docs — Plan 13-05 scope)
    - scripts/publish_hygiene.sh (non-shipped dev tool, Plan 13-06 — excluded from the
      publish tree; it necessarily contains "eurybia" both in the dev-tree absolute path
      it is invoked by and in the 'eurybia'/'EURYBIA_' grep patterns it scans for)
    - __pycache__ directories

    Catches: eurybia / Eurybia / EURYBIA and eurybia- / eurybia_ variants.
    Any EURYBIA_* env-var reference is also caught by the case-insensitive scan.

    This test is the standing rename regression guard. It MUST keep passing so
    that Wave-2 changes cannot accidentally reintroduce the author's boat name
    into the public artifact.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"eurybia", re.IGNORECASE)

    failures: list[str] = []

    # Directories and specific file to scan
    scan_dirs = [
        repo_root / "src" / "leopard44_kb",
        repo_root / "scripts",
    ]
    scan_sql_dir = repo_root / "schema"
    shipped_service = repo_root / "deploy" / "leopard44-kb.service"

    # This test file itself is excluded from the scan to avoid a false-fail
    # on the word "eurybia" inside the test assertion strings.
    this_file = Path(__file__).resolve()

    # scripts/publish_hygiene.sh is a NON-SHIPPED dev tool (13-06): it is excluded
    # from the publish tree entirely, yet necessarily contains "eurybia" — both the
    # dev-tree absolute path it is invoked by and the 'eurybia'/'EURYBIA_' grep
    # patterns it scans the publish index for. Exempt it like this_file.
    hygiene_script = (repo_root / "scripts" / "publish_hygiene.sh").resolve()

    for scan_dir in scan_dirs:
        for path in scan_dir.rglob("*"):
            if path == this_file:
                continue
            if path.resolve() == hygiene_script:
                continue
            if "__pycache__" in path.parts:
                continue
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(content.splitlines(), 1):
                if pattern.search(line):
                    failures.append(f"{path.relative_to(repo_root)}:{lineno}: {line.rstrip()}")

    # Schema SQL files
    for sql_file in scan_sql_dir.glob("*.sql"):
        try:
            content = sql_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(content.splitlines(), 1):
            if pattern.search(line):
                failures.append(f"{sql_file.relative_to(repo_root)}:{lineno}: {line.rstrip()}")

    # Shipped systemd unit
    if shipped_service.exists():
        try:
            content = shipped_service.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        for lineno, line in enumerate(content.splitlines(), 1):
            if pattern.search(line):
                failures.append(f"{shipped_service.relative_to(repo_root)}:{lineno}: {line.rstrip()}")

    assert failures == [], (
        f"Found {len(failures)} 'eurybia' occurrence(s) in the shipped functional tree:\n"
        + "\n".join(f"  {f}" for f in failures[:20])
        + (f"\n  ... and {len(failures) - 20} more" if len(failures) > 20 else "")
    )


def test_old_console_script_absent():
    """PKG-05: the old 'eurybia' console script must NOT exist in .venv/bin.

    D-13 clean break — no back-compat fallback. The 'l44' script is the only
    entry point. This test catches any regression where the old entry point
    is accidentally reinstalled (e.g., if pyproject.toml is reverted).
    """
    from pathlib import Path
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    venv_bin = repo_root / ".venv" / "bin"

    # Assert no 'eurybia' script in .venv/bin
    if venv_bin.exists():
        eurybia_scripts = list(venv_bin.glob("eurybia*"))
        assert eurybia_scripts == [], (
            f"Old 'eurybia' entry point found in .venv/bin: {eurybia_scripts}. "
            "Run `uv sync` to reinstall the renamed entry point."
        )

    # Assert 'uv run eurybia --help' exits non-zero
    result = subprocess.run(
        ["uv", "run", "eurybia", "--help"],
        cwd=str(repo_root),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0, (
        f"'uv run eurybia --help' exited 0 — old entry point still exists! "
        f"stdout: {result.stdout!r}, stderr: {result.stderr!r}"
    )


def test_no_copyrighted_pngs():
    """PKG-05: no *.png committed outside data/ and tests/fixtures/synthetic/.

    Copyrighted schematic pages are never committed (D-03). PNGs generated by
    schematic render go to data/schematics/ (gitignored). Capture photos go to
    data/photos/ (gitignored). Only synthetic test fixtures (generated at test
    time in tmp_path) may be PNG files — they live in tests/fixtures/synthetic/.

    The invariant from ROADMAP.md Success Criteria:
      find . -not -path './data/*' -name '*.png'
    should return zero files (modulo gitignored dirs and synthetic fixtures).
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    # Excluded paths: data/ (gitignored store), tests/fixtures/synthetic/ (safe),
    # and the hidden .venv* directories (gitignored)
    excluded_prefixes = [
        repo_root / "data",
        repo_root / "tests" / "fixtures" / "synthetic",
        repo_root / ".venv",
        repo_root / ".venv-stt",
        repo_root / "test-results-e2e",
        repo_root / "test-results",
        repo_root / ".git",
    ]

    def _is_excluded(path: Path) -> bool:
        for prefix in excluded_prefixes:
            try:
                path.relative_to(prefix)
                return True
            except ValueError:
                pass
        return False

    png_files = [
        p for p in repo_root.rglob("*.png")
        if not _is_excluded(p) and p.is_file()
    ]

    assert png_files == [], (
        f"Found {len(png_files)} committed PNG(s) outside allowed dirs — "
        "copyrighted schematics must never be committed:\n"
        + "\n".join(f"  {p.relative_to(repo_root)}" for p in png_files)
    )


# ---------------------------------------------------------------------------
# DEPLOY-02: /api/sources?layer=vessel returns [] on a shared-only store
# ---------------------------------------------------------------------------


def test_vessel_layer_empty(monkeypatch, tmp_path):
    """DEPLOY-02: /api/sources?layer=vessel returns [] when store is shared-only."""
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/api/sources?layer=vessel")
    assert resp.status_code == 200
    assert resp.json() == [], f"Expected [] for vessel layer on empty DB; got {resp.json()!r}"
