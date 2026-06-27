"""E2E (Playwright) fixtures for the Leopard 44 KB local web UI.

Launches the REAL `l44 serve` subprocess against a seeded, file-backed DB and
yields its base URL. Opt in with `-m e2e` (excluded from the default unit run).

Run headed so you can watch it:
    uv run pytest -m e2e --headed --slowmo=400 --video=on --output=test-results-e2e

NOTE (first-cut coupling): the session fixture seeds the corpus with REAL Ollama
embeddings (nomic-embed-text), so the whole e2e suite currently needs Ollama up.
The pure-frontend tests (load / toggle / offline / explore) don't logically need
it — decoupling them onto a fake-embedding server is a documented follow-up.
"""
from __future__ import annotations

import os
import re
import signal
import socket
import sqlite3
import struct
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Known facts seeded into the two layers — one per layer so the scope toggle and
# layer badges have something real to show.
SHARED_FACT = (
    "The Yanmar 4JH45 engine raw-water impeller should be replaced every 200 hours "
    "of operation to avoid cooling failure."
)
VESSEL_FACT = (
    "The vessel's third reef was added in 2024 using Antal low-friction rings and a "
    "dedicated reef line led aft to the cockpit."
)


def _free_port() -> int:
    """Reserve a free localhost port (closed immediately; small TOCTOU window)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _pack(vec: list[float]) -> bytes:
    return struct.pack("384f", *vec)


def _seed_corpus(db_path: Path) -> None:
    """Seed one shared + one vessel chunk with real embeddings (mirrors test_web_live.py)."""
    import sqlite_vec
    from leopard44_kb.ingest.embedder import embed_texts, select_model
    from leopard44_kb.schema import apply_migrations

    model, _ = select_model()
    embeddings = embed_texts([SHARED_FACT, VESSEL_FACT], model)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)

    conn.execute(
        "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
        "VALUES (1,'shared','log','shared/leopard44/engine.md','h-shared','Yanmar 4JH45 Service Notes')"
    )
    conn.execute(
        "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
        "VALUES (2,'vessel','log','vessel/modifications.md','h-vessel','L44 Owner Modifications')"
    )
    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,page_end,"
        "content,content_hash,anchor_key,embedding_model,embedding_model_version) "
        "VALUES (1,1,'shared',0,'Engine > Cooling',1,1,?,'h-shared','ak-shared-1',?,'seed')",
        (SHARED_FACT, model),
    )
    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,page_end,"
        "content,content_hash,anchor_key,embedding_model,embedding_model_version) "
        "VALUES (2,2,'vessel',0,'Rig > Reefing',1,1,?,'h-vessel','ak-vessel-1',?,'seed')",
        (VESSEL_FACT, model),
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
        "VALUES (1,'shared',1,?,1,?)",
        (model, _pack(embeddings[0])),
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
        "VALUES (2,'vessel',2,?,1,?)",
        (model, _pack(embeddings[1])),
    )
    conn.commit()
    conn.close()


@pytest.fixture(scope="session")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Seed a corpus, launch `l44 serve`, yield its base URL, tear it down."""
    db_path = tmp_path_factory.mktemp("e2e-db") / "l44_e2e.db"
    _seed_corpus(db_path)

    port = _free_port()
    env = {**os.environ, "L44_DB": str(db_path), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        ["uv", "run", "l44", "serve", "--port", str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,  # own process group → clean group kill on teardown
    )

    url_re = re.compile(r"http://127\.0\.0\.1:\d+")
    base_url: str | None = None
    captured: list[str] = []
    deadline = time.monotonic() + 90  # generous: first `uv run` may resolve/sync
    while time.monotonic() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            if proc.poll() is not None:
                break
            continue
        captured.append(line)
        m = url_re.search(line)
        if m:
            base_url = m.group(0)
            break

    if base_url is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise RuntimeError(
            "`l44 serve` did not print a URL within 90s.\n--- output ---\n"
            + "".join(captured)
        )

    # Drain remaining stdout so a full pipe never blocks the server.
    def _drain() -> None:
        try:
            assert proc.stdout is not None
            for _ in proc.stdout:
                pass
        except Exception:
            pass

    threading.Thread(target=_drain, daemon=True).start()

    # Readiness poll.
    import httpx

    ready = False
    rdeadline = time.monotonic() + 30
    while time.monotonic() < rdeadline:
        try:
            if httpx.get(base_url + "/", timeout=2.0).status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.3)
    if not ready:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        raise RuntimeError(f"Server at {base_url} did not become ready within 30s.")

    try:
        yield base_url
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
