"""Phase 9 visual-regression guard: the annotation canvas must overlay the schematic image.

A real bug found by the Phase 9 headed HUMAN-UAT (2026-06-14): the `<canvas>` rendered
~2x wider than the `<img>` it overlays (656px vs 340px) because `.canvas-block`
(inline-block) does not shrink-to-fit the max-height-constrained image, so the
CSS `width:100%` sized the canvas to the wrong box. Every unit test passed — none
assert rendered layout. The fix pins the canvas display box to the image's
getBoundingClientRect on load AND on window resize.

This test asserts the canvas box ≈ the image box at two viewport widths. It needs a
real browser + `l44 serve`, so it is opt-in (`-m e2e`) and excluded from the
default unit run. It does NOT need Ollama (pure frontend — no query path).

Run:  uv run pytest tests/e2e/test_phase9_canvas_overlay.py -m e2e
Watch: add --headed --slowmo=400
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMATICS_DIR = REPO_ROOT / "data" / "schematics"
TEST_PNG = SCHEMATICS_DIR / "page_999_e2etest.png"
ZONE_ID = 7  # any seeded zone; saloon-seat-port

pytestmark = pytest.mark.e2e


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _make_test_png(path: Path) -> None:
    """Write a portrait PNG with the same aspect as a 150-DPI A4 schematic page."""
    import fitz  # PyMuPDF — already a project dep

    doc = fitz.open()
    doc.new_page(width=595, height=842)  # A4 points → 150 DPI gives ~1240x1754
    pix = doc[0].get_pixmap(dpi=150)
    path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(path))
    doc.close()


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> str:
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    db_path = tmp_path_factory.mktemp("p9overlay") / "store.db"
    env = dict(os.environ)
    env["L44_DB"] = str(db_path)

    # Migrate (seeds the 32 zones) and drop a test schematic into data/schematics/.
    conn = open_db(db_path)
    apply_migrations(conn)
    conn.close()
    _make_test_png(TEST_PNG)

    port = _free_port()
    proc = subprocess.Popen(
        ["uv", "run", "l44", "serve", "--port", str(port)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        end = time.time() + 30
        while time.time() < end:
            with socket.socket() as t:
                if t.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.3)
        else:
            raise RuntimeError("l44 serve never came up")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        TEST_PNG.unlink(missing_ok=True)


def _boxes(page):
    return page.evaluate(
        """() => {
            const img = document.getElementById('schematic-img');
            const cv = document.getElementById('polygon-canvas');
            const r = e => { const b = e.getBoundingClientRect();
                return {x: b.x, y: b.y, w: b.width, h: b.height}; };
            return {img: r(img), canvas: r(cv)};
        }"""
    )


def _assert_overlay(boxes, tol=2.0):
    img, cv = boxes["img"], boxes["canvas"]
    assert abs(cv["x"] - img["x"]) <= tol, f"x: canvas {cv['x']} vs img {img['x']}"
    assert abs(cv["y"] - img["y"]) <= tol, f"y: canvas {cv['y']} vs img {img['y']}"
    assert abs(cv["w"] - img["w"]) <= tol, f"width: canvas {cv['w']} vs img {img['w']}"
    assert abs(cv["h"] - img["h"]) <= tol, f"height: canvas {cv['h']} vs img {img['h']}"


def test_canvas_overlays_image_on_load_and_resize(served, page):
    page.goto(f"{served}/annotate/{ZONE_ID}")
    page.wait_for_selector("#polygon-canvas")
    page.wait_for_function(
        "() => { const i = document.getElementById('schematic-img'); return i && i.naturalWidth > 0; }"
    )
    page.wait_for_timeout(400)

    # On load at the default viewport, the canvas must overlay the image exactly.
    _assert_overlay(_boxes(page))

    # After a viewport resize, the overlay must re-pin to the (now CSS-scaled) image.
    page.set_viewport_size({"width": 760, "height": 900})
    page.wait_for_timeout(400)
    _assert_overlay(_boxes(page))
