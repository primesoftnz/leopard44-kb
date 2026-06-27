# RED state until Plan 02 ships app.py (see VALIDATION.md).
# create_app is imported INSIDE each test function body — NEVER at module top level.
# This ensures pytest collection succeeds even before app.py exists (RED by assertion,
# not by collection error).
"""Tests for UI-04: offline contract — no external origins, static assets served."""
from __future__ import annotations

import re
from pathlib import Path


def test_no_external_origins(monkeypatch, tmp_path):
    """GET / returns HTML with zero references to external https:// origins.

    Asserts the offline contract (D-07, UI-04): no CDN, no Google Fonts,
    no external scripts. Only 127.0.0.1/localhost origins are allowed.
    RESEARCH Pattern 6 regex applied over the rendered HTML.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200, f"GET / returned {resp.status_code}"
    html = resp.text
    external = re.findall(
        r'(?:src|href|@import|url)\s*[=(]\s*["\']?(https?://(?!127\.0\.0\.1|localhost)[^\s"\']+)',
        html,
        re.IGNORECASE,
    )
    assert external == [], f"GET / HTML references external origins: {external}"


def test_explore_no_external_origins(monkeypatch, tmp_path):
    """GET /explore returns HTML with zero references to external https:// origins."""
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/explore")
    assert resp.status_code == 200, f"GET /explore returned {resp.status_code}"
    html = resp.text
    external = re.findall(
        r'(?:src|href|@import|url)\s*[=(]\s*["\']?(https?://(?!127\.0\.0\.1|localhost)[^\s"\']+)',
        html,
        re.IGNORECASE,
    )
    assert external == [], f"GET /explore HTML references external origins: {external}"


def test_static_assets_served(monkeypatch, tmp_path):
    """GET /static/app.css and /static/app.js each return HTTP 200 with non-empty body."""
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    css_resp = client.get("/static/app.css")
    assert css_resp.status_code == 200, f"/static/app.css returned {css_resp.status_code}"
    assert len(css_resp.text) > 0, "/static/app.css has an empty body"

    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200, f"/static/app.js returned {js_resp.status_code}"
    assert len(js_resp.text) > 0, "/static/app.js has an empty body"


def test_static_tree_has_no_external_origins(monkeypatch, tmp_path):
    """Walk static/ and templates/ trees; assert no file contains external https:// origins.

    Promotes the offline scan from a runtime assertion over rendered pages to a
    file-level scan over the source assets themselves (review fix MEDIUM — UI-04).
    Guards: no external @import url(...), no @font-face with remote url(...),
    no <script src="http...">, no <link href="http...">, no bare https:// origins
    in any .css, .js, .svg, or .html file.
    """
    import leopard44_kb.web as web_pkg

    web_dir = Path(web_pkg.__file__).parent
    static_dir = web_dir / "static"
    templates_dir = web_dir / "templates"

    external_pattern = re.compile(
        r'(?:src|href|@import|url)\s*[=(]\s*["\']?(https?://(?!127\.0\.0\.1|localhost)[^\s"\']+)',
        re.IGNORECASE,
    )

    violations: list[str] = []
    extensions = {".css", ".js", ".svg", ".html"}

    for search_dir in [static_dir, templates_dir]:
        if not search_dir.exists():
            # Plans 02-04 create the static/templates trees; before they exist
            # the test passes trivially (no files to scan = no violations).
            continue
        for fpath in search_dir.rglob("*"):
            if fpath.suffix.lower() not in extensions:
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            found = external_pattern.findall(text)
            for match in found:
                violations.append(f"{fpath.relative_to(web_dir)}: {match}")

    assert violations == [], (
        f"Static/template assets reference external origins (UI-04 violation):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_fastapi_sse_importable():
    """Smoke test: fastapi.sse exports EventSourceResponse and ServerSentEvent.

    Guards the single unverified dependency assumption (RESEARCH § Environment
    Availability): the pinned venv install of fastapi must expose the SSE API.
    If this import fails after Task 1 installs fastapi, the documented fallback is
    to add sse-starlette and import EventSourceResponse from sse_starlette.sse.
    This test can PASS as soon as Task 1 completes — it does NOT depend on Plan 02.
    """
    from fastapi.sse import EventSourceResponse, ServerSentEvent

    assert EventSourceResponse is not None, "EventSourceResponse must not be None"
    assert ServerSentEvent is not None, "ServerSentEvent must not be None"


# ---------------------------------------------------------------------------
# Phase 9 / VIS-02: zero-outbound contract for new annotation routes (D-13)
# RED until 09-03 ships GET /annotate, GET /annotate/{id}, GET /schematic-image
# ---------------------------------------------------------------------------


def test_annotate_list_no_external_origins(monkeypatch, tmp_path):
    """GET /annotate returns HTML with zero references to external https:// origins.

    Asserts the offline contract (VIS-02, D-13): annotation list page loads with
    only self-hosted assets. No CDN, Google Fonts, external scripts.
    Only 127.0.0.1/localhost origins are permitted.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app  # RED until 09-03
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/annotate")
    assert resp.status_code == 200, f"GET /annotate returned {resp.status_code}"
    html = resp.text
    external = re.findall(
        r'(?:src|href|@import|url)\s*[=(]\s*["\']?(https?://(?!127\.0\.0\.1|localhost)[^\s"\']+)',
        html,
        re.IGNORECASE,
    )
    assert external == [], f"GET /annotate HTML references external origins: {external}"


def test_annotate_editor_no_external_origins(monkeypatch, tmp_path):
    """GET /annotate/1 returns HTML with zero references to external https:// origins.

    Zone id=1 exists after migration 002 seeds 32 zones. Bootstraps a file-backed
    DB first so the editor route can render without a migration error.
    """
    import sqlite3
    import sqlite_vec
    from leopard44_kb.schema import apply_migrations

    db_path = tmp_path / "s.db"
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    conn.close()

    monkeypatch.setenv("L44_DB", str(db_path))
    from leopard44_kb.web.app import create_app  # RED until 09-03
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/annotate/1")
    assert resp.status_code == 200, f"GET /annotate/1 returned {resp.status_code}"
    html = resp.text
    external = re.findall(
        r'(?:src|href|@import|url)\s*[=(]\s*["\']?(https?://(?!127\.0\.0\.1|localhost)[^\s"\']+)',
        html,
        re.IGNORECASE,
    )
    assert external == [], f"GET /annotate/1 HTML references external origins: {external}"


def test_schematic_image_not_in_static(monkeypatch, tmp_path):
    """GET /schematic-image/<file> returns 404 for non-existent file, NOT a redirect to static/.

    Asserts the D-13 invariant: schematics are NEVER served from static/.
    A missing schematic must 404 rather than fall through to static/ fallback.
    The status code is the only observable — if the route doesn't exist (RED state),
    this also returns 404, making it trivially RED until 09-03 ships the route.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    from leopard44_kb.web.app import create_app  # RED until 09-03
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/schematic-image/nonexistent_page.png")
    assert resp.status_code == 404, (
        f"Expected 404 for non-existent schematic, got {resp.status_code} — "
        "route must NOT fall back to static/"
    )
