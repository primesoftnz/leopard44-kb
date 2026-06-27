#!/usr/bin/env python
"""Headed Playwright walkthrough of Phase 9's HUMAN-UAT (run manually, watch it drive).

This is a MANUAL demo harness (not a pytest test) — it opens a real browser and
walks all four Phase 9 visual UAT checks against a live `l44 serve`:
  UAT-1 canvas alignment on resize · UAT-2 polygon round-trip ·
  UAT-3 query highlight appearance · UAT-4 graceful degradation

It found two real bugs on 2026-06-14 (canvas overlay 5f44c81, geometry round-trip
088ce29) that the unit suite missed. Re-run it after any change to the annotation
editor or the zone_highlight SSE path.

SETUP (one-time, before running):
  export L44_DB=/tmp/l44-uat9.db
  rm -f /tmp/l44-uat9.db*
  uv run l44 schematic render "data/sources/leopard44_factory/L44 A5160 Owner's Manual Sunsail.pdf" --pages 61
  uv run python -c "from leopard44_kb.store import open_db; from leopard44_kb.schema import apply_migrations; c=open_db(); apply_migrations(c); c.close()"
  uv run l44 item add "winch handle" --category tool  --zone saloon-seat-port --yes
  uv run l44 item add "flare kit"    --category safety --zone nav-station      --yes
  # (re-running: reset the drawn zone first)
  uv run python -c "from leopard44_kb.store import open_db; from leopard44_kb.schema import apply_migrations; c=open_db(); apply_migrations(c); c.execute('UPDATE zones SET geometry=NULL,schematic_image=NULL WHERE id=7'); c.commit(); c.close()"

RUN:
  L44_DB=/tmp/l44-uat9.db uv run python tests/e2e/uat_phase9_headed.py

Requires Ollama up (nomic-embed-text + a generation model) for UAT-3/4, WSLg display
for the headed browser, and page_061.png rendered into data/schematics/.
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

DEMO_DB = "/tmp/l44-uat9.db"
SHOTS = Path("/tmp/uat9-shots"); SHOTS.mkdir(exist_ok=True)
ZONE_A_ID = 7   # saloon-seat-port  (annotated live in UAT-1/2, holds the winch handle for UAT-3)
SCHEMATIC = "page_061.png"

env = dict(os.environ); env["L44_DB"] = DEMO_DB


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def wait_port(port: int, timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.3)
    raise RuntimeError(f"server never came up on :{port}")


def banner(page, n: int, title: str, detail: str) -> None:
    """Inject a floating caption so the headed view narrates itself."""
    page.evaluate(
        """([n, title, detail]) => {
            let el = document.getElementById('uat-banner');
            if (!el) { el = document.createElement('div'); el.id='uat-banner';
              el.style.cssText='position:fixed;top:0;left:0;right:0;z-index:99999;'+
                'background:#0E1A1F;color:#E8A13A;font:600 16px/1.4 system-ui;'+
                'padding:10px 16px;border-bottom:2px solid #E8A13A;box-shadow:0 2px 8px rgba(0,0,0,.4)';
              document.body.appendChild(el); }
            el.innerHTML = '<span style="color:#fff">UAT-'+n+'</span> &nbsp; '+title+
              ' &nbsp;<span style="color:#9aa;font-weight:400">'+detail+'</span>';
        }""",
        [n, title, detail],
    )
    print(f"\n=== UAT-{n}: {title} — {detail}")


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
            browser = pw.chromium.launch(headless=False, slow_mo=650)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            # ---------------- UAT-1: canvas alignment on resize ----------------
            page.goto(f"{base}/annotate")
            page.wait_for_selector(".zone-list")
            banner(page, 1, "Canvas alignment on resize",
                   "open the saloon-seat-port editor and draw a polygon")
            page.screenshot(path=str(SHOTS / "01a-zone-list.png"))
            page.click(f'a[href="/annotate/{ZONE_A_ID}"]')
            page.wait_for_selector("#polygon-canvas")
            page.wait_for_function("() => { const i=document.getElementById('schematic-img'); return i && i.naturalWidth>0; }")
            time.sleep(0.6)

            canvas = page.locator("#polygon-canvas")
            box = canvas.bounding_box()
            # click 4 corners of a rectangle within the displayed canvas, then dblclick to close
            pts = [(0.30, 0.28), (0.62, 0.28), (0.62, 0.55), (0.30, 0.55)]
            for fx, fy in pts:
                canvas.click(position={"x": box["width"] * fx, "y": box["height"] * fy})
                time.sleep(0.2)
            canvas.dblclick(position={"x": box["width"] * 0.30, "y": box["height"] * 0.28})
            time.sleep(0.5)
            save_enabled_full = page.locator("#save-btn").is_enabled()
            page.screenshot(path=str(SHOTS / "01b-drawn-fullsize.png"))

            # shrink the window — the polygon must stay over the SAME compartment (resolution-independent)
            banner(page, 1, "Canvas alignment on resize",
                   "shrink window -> polygon stays on the same compartment")
            page.set_viewport_size({"width": 760, "height": 900})
            time.sleep(0.8)
            page.screenshot(path=str(SHOTS / "01c-resized-small.png"))

            # draw a SECOND polygon at the smaller size to prove clicks still track after resize
            banner(page, 1, "Canvas alignment on resize",
                   "redraw at the smaller size -> vertices land under the cursor (live getBoundingClientRect)")
            page.locator("#clear-btn").click()
            time.sleep(0.3)
            box2 = canvas.bounding_box()
            for fx, fy in [(0.25, 0.30), (0.70, 0.30), (0.70, 0.60), (0.25, 0.60)]:
                canvas.click(position={"x": box2["width"] * fx, "y": box2["height"] * fy})
                time.sleep(0.2)
            canvas.dblclick(position={"x": box2["width"] * 0.25, "y": box2["height"] * 0.30})
            time.sleep(0.5)
            page.screenshot(path=str(SHOTS / "01d-redrawn-after-resize.png"))
            ok1 = save_enabled_full and page.locator("#save-btn").is_enabled()
            results.append(("UAT-1 canvas alignment / resize", ok1,
                            "polygon drawn full-size + redrawn after resize; save enabled both times"))
            page.set_viewport_size({"width": 1280, "height": 900})
            time.sleep(0.4)

            # ---------------- UAT-2: polygon round-trip ----------------
            banner(page, 2, "Polygon round-trip", "Save the polygon")
            page.locator("#save-btn").click()
            page.wait_for_selector("#save-success:not([hidden])", timeout=8000)
            page.screenshot(path=str(SHOTS / "02a-saved.png"))

            banner(page, 2, "Polygon round-trip", "re-open the zone -> saved polygon reloads on the same compartment")
            page.goto(f"{base}/annotate/{ZONE_A_ID}")
            page.wait_for_selector("#polygon-canvas")
            page.wait_for_function("() => { const i=document.getElementById('schematic-img'); return i && i.naturalWidth>0; }")
            # loadExisting() enables the save button once it maps the persisted geometry
            try:
                page.wait_for_function("() => { const b=document.getElementById('save-btn'); return b && !b.disabled; }", timeout=8000)
                save_enabled_reopen = True
            except Exception:
                save_enabled_reopen = False
            page.screenshot(path=str(SHOTS / "02b-reopened-persisted.png"))
            # confirm the zone now shows "Annotated" in the list
            page.goto(f"{base}/annotate")
            page.wait_for_selector(".zone-list")
            row = page.locator(f'li:has(a[href="/annotate/{ZONE_A_ID}"])')
            annotated_badge = row.get_by_text("Annotated").count() > 0
            page.screenshot(path=str(SHOTS / "02c-list-annotated.png"))
            results.append(("UAT-2 polygon round-trip", save_enabled_reopen and annotated_badge,
                            f"reopened editor reloaded geometry (save enabled={save_enabled_reopen}); list badge 'Annotated'={annotated_badge}"))

            # ---------------- UAT-3: query highlight appearance ----------------
            banner(page, 3, "Query highlight appearance", "ask 'where is the winch handle?' (in the annotated zone)")
            page.goto(base)
            page.wait_for_selector("#question")
            page.fill("#question", "where is the winch handle?")
            page.click("#ask-btn")
            print("  waiting for SSE answer + zone_highlight (CPU generation, up to ~60s)...")
            page.wait_for_selector(".zone-highlight", timeout=90000)
            time.sleep(0.4)
            page.screenshot(path=str(SHOTS / "03a-highlight-block.png"))
            has_name = page.locator(".zone-highlight .zone-highlight-name").count() > 0
            reveal = page.locator(".zone-highlight details.schematic-reveal")
            has_reveal = reveal.count() > 0
            if has_reveal:
                banner(page, 3, "Query highlight appearance", "expand 'Show on schematic' -> thin amber non-scaling outline")
                reveal.locator("summary").click()
                page.wait_for_selector(".zone-highlight svg.zone-polygon-overlay polygon", timeout=8000)
                page.wait_for_function(
                    "() => { const p=document.querySelector('.zone-highlight svg.zone-polygon-overlay polygon');"
                    " return p && p.getAttribute('points'); }", timeout=8000)
                time.sleep(0.8)
                page.screenshot(path=str(SHOTS / "03b-schematic-revealed.png"))
                non_scaling = page.locator(".zone-highlight svg.zone-polygon-overlay polygon").get_attribute("vector-effect")
            else:
                non_scaling = None
            results.append(("UAT-3 query highlight", has_name and has_reveal and non_scaling == "non-scaling-stroke",
                            f"name shown={has_name}, reveal={has_reveal}, vector-effect={non_scaling!r}"))

            # ---------------- UAT-4: graceful degradation ----------------
            banner(page, 4, "Graceful degradation", "ask 'where is the flare kit?' (zone has NO polygon)")
            page.goto(base)
            page.wait_for_selector("#question")
            page.fill("#question", "where is the flare kit?")
            page.click("#ask-btn")
            print("  waiting for SSE answer + zone_highlight (no schematic control expected)...")
            page.wait_for_selector(".zone-highlight", timeout=90000)
            time.sleep(0.6)
            name4 = page.locator(".zone-highlight .zone-highlight-name").count() > 0
            # Graceful degradation is per-zone: the flare kit's zone (Nav station, NO geometry)
            # must show name+cue with NO "Show on schematic" control. (In a 2-item demo corpus
            # the query may also surface the winch handle's annotated zone — that block correctly
            # DOES have a control; we assert specifically on the un-annotated flare-kit zone.)
            nav_block = page.locator(".zone-highlight").filter(has=page.locator(".zone-highlight-name", has_text="Nav station"))
            nav_reveal = nav_block.locator("details.schematic-reveal").count()
            page.screenshot(path=str(SHOTS / "04-graceful-degradation.png"))
            results.append(("UAT-4 graceful degradation", name4 and nav_block.count() >= 1 and nav_reveal == 0,
                            f"Nav-station (un-annotated) block present={nav_block.count()>=1}, its schematic control count={nav_reveal} (expected 0)"))

            banner(page, 0, "Walkthrough complete",
                   "all 4 UAT items exercised — see /tmp/uat9-shots/")
            time.sleep(2.0)
            browser.close()

        print("\n" + "=" * 64)
        print("PHASE 9 HUMAN-UAT — HEADED WALKTHROUGH RESULTS")
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
