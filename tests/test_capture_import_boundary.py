"""RED tests for CAP-01: capture module must NOT be importable from the web surface.

Phase 12 Wave-0 RED scaffold. These tests assert that `leopard44_kb.web.app` does not
transitively import any `leopard44_kb.capture.*` module — enforcing the two-surface
architecture split: capture needs internet, web/query must stay fully offline.

Coverage (H2):
  (j) STATIC AST walk: import graph of leopard44_kb.web.app has no leopard44_kb.capture edge
  (k) RUNTIME sys.modules guard: create_app() does not load any leopard44_kb.capture* module
  (l) POST-ROUTE assertion: exercising a web route still has no capture* loaded + no outbound

Design discipline: leopard44_kb.capture is imported INSIDE each test body (not at module top)
so the collection succeeds even in the pre-capture RED state. leopard44_kb.web.app IS imported
inside each test body for the same reason — it exists already, but the test structure is
consistent with the project's RED-at-assertion convention.
"""
from __future__ import annotations

import ast
import sys
from importlib import import_module
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper: locate the leopard44_kb package root in the installed package
# ---------------------------------------------------------------------------

def _leopard44_src_root() -> Path:
    """Return the Path to the leopard44_kb/ package directory."""
    import leopard44_kb
    return Path(leopard44_kb.__file__).parent


# ---------------------------------------------------------------------------
# (j) STATIC AST walk: import graph of leopard44_kb.web.app has no capture edges
# ---------------------------------------------------------------------------

def test_static_ast_no_capture_import_in_web_app(tmp_path):
    """AST-walk the leopard44_kb.web package; assert no import reaches leopard44_kb.capture.

    Recursively parses leopard44_kb.web.app and all modules it imports transitively
    (up to a fixed depth), checking that no import statement references
    'leopard44_kb.capture' or 'capture' from the leopard44_kb package.

    RED: This test will PASS (not RED) as long as capture/ doesn't exist AND
    web/app.py doesn't import it. It becomes a meaningful regression guard once
    capture/ is created in Wave 2.
    """
    leopard44_root = _leopard44_src_root()
    web_app_path = leopard44_root / "web" / "app.py"

    assert web_app_path.exists(), f"leopard44_kb/web/app.py not found at {web_app_path}"

    capture_edges: list[str] = []
    visited: set[Path] = set()

    def _walk_imports(source_path: Path, depth: int = 0) -> None:
        if depth > 5 or source_path in visited:
            return
        visited.add(source_path)

        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "leopard44_kb.capture" in alias.name or alias.name == "capture":
                        capture_edges.append(
                            f"{source_path.name}: import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "leopard44_kb.capture" in module or module == "capture":
                    capture_edges.append(
                        f"{source_path.name}: from {module} import ..."
                    )
                # Recurse into leopard44_kb.* imports (within the package)
                if module.startswith("leopard44_kb.") and not module.startswith("leopard44_kb.capture"):
                    sub_path = (
                        leopard44_root / Path(*module.split(".")[1:]).with_suffix(".py")
                    )
                    if sub_path.exists():
                        _walk_imports(sub_path, depth + 1)

    _walk_imports(web_app_path)

    assert capture_edges == [], (
        f"leopard44_kb.web.app transitively imports leopard44_kb.capture — "
        f"this violates the two-surface architecture split (CAP-01):\n"
        + "\n".join(f"  {e}" for e in capture_edges)
    )


# ---------------------------------------------------------------------------
# (k) RUNTIME sys.modules guard: create_app() loads NO leopard44_kb.capture* module
# ---------------------------------------------------------------------------

def test_runtime_create_app_does_not_load_capture(monkeypatch, tmp_path):
    """create_app() must NOT cause any leopard44_kb.capture* module to appear in sys.modules (H2).

    Snapshots sys.modules before and after create_app() invocation.
    RED: ModuleNotFoundError until create_app exists — but create_app exists already,
    so this test will fail for the right reason only if capture gets imported by web.
    Until capture/ is created, the guard trivially passes (nothing to load).
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "test.db"))

    # Snapshot before
    before_keys = set(sys.modules.keys())

    from leopard44_kb.web.app import create_app  # exists now; RED-safe import inside body
    create_app()

    after_keys = set(sys.modules.keys())
    new_keys = after_keys - before_keys

    capture_loaded = [k for k in new_keys if k.startswith("leopard44_kb.capture")]

    assert capture_loaded == [], (
        f"create_app() loaded leopard44_kb.capture* modules — "
        f"this violates the offline guarantee (H2):\n"
        + "\n".join(f"  {k}" for k in sorted(capture_loaded))
    )


# ---------------------------------------------------------------------------
# (l) POST-ROUTE: exercising a web route loads no capture* and no outbound
# ---------------------------------------------------------------------------

def test_post_route_no_capture_load_no_outbound(monkeypatch, tmp_path):
    """After exercising a representative web route, no leopard44_kb.capture* module is loaded
    and no outbound network call was attempted (H2).

    Uses the GET / route (the index page). Any route that calls core retrieval
    proves the invariant — if capture loaded at any point during normal web
    operation, it would appear in sys.modules after the route is hit.
    RED: ModuleNotFoundError until leopard44_kb.capture exists. Until then, trivially
    passes (capture can't load because the package doesn't exist).
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "test.db"))

    outbound_calls: list[str] = []

    # Intercept any httpx.post or httpx.get calls to external origins
    import httpx

    original_post = httpx.post
    original_get = httpx.get

    def _guarded_post(url: str, *args, **kwargs):
        if "anthropic.com" in url or "openai.com" in url:
            outbound_calls.append(f"POST {url}")
        # Allow localhost calls (Ollama) — the test is about external origins
        return original_post(url, *args, **kwargs)

    def _guarded_get(url: str, *args, **kwargs):
        if "anthropic.com" in url or "openai.com" in url:
            outbound_calls.append(f"GET {url}")
        return original_get(url, *args, **kwargs)

    monkeypatch.setattr(httpx, "post", _guarded_post)
    monkeypatch.setattr(httpx, "get", _guarded_get)

    # Snapshot sys.modules before route
    before_keys = set(sys.modules.keys())

    from leopard44_kb.web.app import create_app  # body import
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200, f"GET / returned {resp.status_code}"

    after_keys = set(sys.modules.keys())
    capture_loaded = [k for k in (after_keys - before_keys) if k.startswith("leopard44_kb.capture")]

    assert capture_loaded == [], (
        f"Exercising GET / loaded leopard44_kb.capture* modules — "
        f"this violates the offline guarantee (H2):\n"
        + "\n".join(f"  {k}" for k in sorted(capture_loaded))
    )

    assert outbound_calls == [], (
        f"Exercising GET / made outbound calls to external origins — "
        f"this violates the offline contract:\n"
        + "\n".join(f"  {c}" for c in outbound_calls)
    )


# ---------------------------------------------------------------------------
# Regression: capture package must not appear as a web.app import once it exists
# ---------------------------------------------------------------------------

def test_web_app_module_file_contains_no_capture_import():
    """Grep-level guard: leopard44_kb/web/app.py source does not contain 'capture' as an import target.

    This is a fast literal scan (no AST) that catches any accidental direct import
    added during Wave 2/3 development of capture/. The AST walk in (j) handles
    transitive imports; this handles direct-in-file carelessness.
    """
    leopard44_root = _leopard44_src_root()
    app_source = (leopard44_root / "web" / "app.py").read_text(encoding="utf-8")

    # Look for any import statement that pulls from leopard44_kb.capture
    import re
    capture_imports = re.findall(
        r'^\s*(?:import|from)\s+.*capture', app_source, re.MULTILINE
    )
    assert capture_imports == [], (
        f"leopard44_kb/web/app.py contains direct capture imports — "
        f"forbidden by two-surface architecture (CAP-01):\n"
        + "\n".join(f"  {line}" for line in capture_imports)
    )
