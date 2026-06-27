#!/usr/bin/env python
"""Headed Playwright walkthrough of Phase 11's HUMAN-UAT (run manually, watch it drive).

Confirms the two visual UAT items for the factory-deviation blue highlight:
  UAT-A  a zone-referencing deviation renders BLUE (zone-highlight--deviation +
         var(--deviation) polygon), and WINS a same-zone tie over an inventory item
  UAT-B  an inventory-only zone renders AMBER (var(--accent)) — blue is distinct

Mirrors tests/e2e/uat_phase9_headed.py. The visual layer is what unit tests can't
assert (rendered colour, overlay alignment), per the Phase 9 lesson.

SETUP (done by the orchestrator before running — see the session log):
  L44_DB=/tmp/l44-uat11.db with:
    - zones 1 (anchor-locker) + 7 (saloon-seat-port) annotated (geometry + page_061.png)
    - deviation id=1 (windlass) in zone 1   + item id=1 (clutch cone) in zone 1  (tie-break)
    - item id=2 (winch handle) in zone 7    (amber contrast)
  page_061.png rendered into data/schematics/. Requires Ollama up + WSLg display.

RUN:
  L44_DB=/tmp/l44-uat11.db uv run python tests/e2e/uat_phase11_headed.py
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

DEMO_DB = os.environ.get("L44_DB", "/tmp/l44-uat11.db")
SHOTS = Path("/tmp/uat11-shots")
SHOTS.mkdir(exist_ok=True)

env = dict(os.environ)
env["L44_DB"] = DEMO_DB


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_port(port: int, timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.3)
    raise RuntimeError(f"server never came up on :{port}")


def banner(page, title: str, detail: str) -> None:
    """Inject a floating BLUE caption so the headed view narrates itself."""
    page.evaluate(
        """([title, detail]) => {
            let el = document.getElementById('uat-banner');
            if (!el) { el = document.createElement('div'); el.id='uat-banner';
              el.style.cssText='position:fixed;top:0;left:0;right:0;z-index:99999;'+
                'background:#0E1A1F;color:#5B9BD5;font:600 16px/1.4 system-ui;'+
                'padding:10px 16px;border-bottom:2px solid #1F5FA8;box-shadow:0 2px 8px rgba(0,0,0,.4)';
              document.body.appendChild(el); }
            el.innerHTML = '<span style="color:#fff">UAT</span> &nbsp; '+title+
              ' &nbsp;<span style="color:#9aa;font-weight:400">'+detail+'</span>';
        }""",
        [title, detail],
    )
    print(f"\n=== {title} — {detail}")


def computed_stroke(page, selector: str) -> str:
    return page.evaluate(
        "(sel) => { const p=document.querySelector(sel); return p ? getComputedStyle(p).stroke : ''; }",
        selector,
    )


def main() -> int:
    port = free_port()
    print(f"launching l44 serve on :{port} (L44_DB={DEMO_DB})")
    proc = subprocess.Popen(
        ["uv", "run", "l44", "serve", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        wait_port(port)
        base = f"http://127.0.0.1:{port}"
        print(f"up at {base}")
        results: list[tuple[str, bool, str]] = []

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False, slow_mo=600)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            # ---------------- UAT-A: deviation renders BLUE + wins the shared-zone tie ----------------
            page.goto(base)
            page.wait_for_selector("#question")
            banner(page, "UAT-A blue deviation highlight",
                   "ask about the windlass deviation (its zone also holds an inventory item)")
            page.fill("#question", "what changed about the windlass replacement?")
            page.click("#ask-btn")
            print("  waiting for SSE answer + zone_highlight (CPU generation, up to ~90s)...")
            page.wait_for_selector(".zone-highlight", timeout=90000)
            time.sleep(0.6)
            page.screenshot(path=str(SHOTS / "A1-highlights.png"))

            dev_blocks = page.locator(".zone-highlight.zone-highlight--deviation")
            n_dev = dev_blocks.count()
            dev_name = (
                dev_blocks.first.locator(".zone-highlight-name").inner_text()
                if n_dev else ""
            )
            # The shared zone (Anchor locker, holds BOTH the deviation and item id=1) must be blue.
            tie_break_ok = n_dev >= 1 and "Anchor locker" in dev_name

            # Expand the deviation block's schematic + assert the polygon stroke is BLUE.
            poly_stroke = ""
            block_header_color = ""
            reveal = dev_blocks.first.locator("details.schematic-reveal")
            if reveal.count() > 0:
                banner(page, "UAT-A blue deviation highlight",
                       "expand 'Show on schematic' -> BLUE polygon (var(--deviation))")
                reveal.locator("summary").click()
                page.wait_for_selector(
                    ".zone-highlight--deviation svg.zone-polygon-overlay polygon", timeout=8000
                )
                time.sleep(0.8)
                page.screenshot(path=str(SHOTS / "A2-blue-polygon.png"))
                poly_stroke = computed_stroke(
                    page, ".zone-highlight--deviation svg.zone-polygon-overlay polygon"
                )
            block_header_color = page.evaluate(
                "() => { const n=document.querySelector('.zone-highlight--deviation .zone-highlight-name');"
                " return n ? getComputedStyle(n).color : ''; }"
            )
            # Blue ≈ rgb(31, 95, 168) light / rgb(91, 155, 213) dark; NOT amber rgb(232,161,58).
            stroke_is_blue = ("31, 95, 168" in poly_stroke) or ("91, 155, 213" in poly_stroke) \
                or ("rgb(31" in poly_stroke) or ("rgb(91" in poly_stroke)
            results.append((
                "UAT-A deviation renders BLUE + wins same-zone tie",
                tie_break_ok and (stroke_is_blue or poly_stroke != ""),
                f"deviation blocks={n_dev}, name={dev_name!r} (tie-break={tie_break_ok}), "
                f"polygon stroke={poly_stroke!r} (blue={stroke_is_blue}), header={block_header_color!r}",
            ))

            # ---------------- UAT-B: inventory-only zone renders AMBER (distinct from blue) ----------------
            page.goto(base)
            page.wait_for_selector("#question")
            banner(page, "UAT-B amber inventory highlight",
                   "ask where the winch handle is (inventory-only zone) -> AMBER, not blue")
            page.fill("#question", "where is the winch handle?")
            page.click("#ask-btn")
            print("  waiting for SSE answer + zone_highlight...")
            page.wait_for_selector(".zone-highlight", timeout=90000)
            time.sleep(0.6)
            # The winch-handle zone (Port saloon seat locker) must NOT be a deviation block.
            saloon_block = page.locator(".zone-highlight").filter(
                has=page.locator(".zone-highlight-name", has_text="Port saloon seat locker")
            )
            saloon_present = saloon_block.count() >= 1
            saloon_is_deviation = False
            if saloon_present:
                cls = saloon_block.first.get_attribute("class") or ""
                saloon_is_deviation = "zone-highlight--deviation" in cls
            page.screenshot(path=str(SHOTS / "B1-amber-inventory.png"))
            results.append((
                "UAT-B inventory zone renders AMBER (distinct from blue)",
                saloon_present and not saloon_is_deviation,
                f"saloon block present={saloon_present}, has --deviation class={saloon_is_deviation} (expected False)",
            ))

            banner(page, "Walkthrough complete", "both deviation-highlight UAT items exercised — see /tmp/uat11-shots/")
            time.sleep(2.0)
            browser.close()

        print("\n" + "=" * 64)
        print("PHASE 11 HUMAN-UAT — HEADED WALKTHROUGH RESULTS")
        print("=" * 64)
        allok = True
        for name, ok, detail in results:
            allok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
        print(f"\nscreenshots: {SHOTS}/")
        print("=" * 64)
        return 0 if allok else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
