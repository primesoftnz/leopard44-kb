"""Headed-friendly browser E2E tests for the Leopard 44 KB local web UI.

Opt in:   uv run pytest -m e2e
Watch it: uv run pytest -m e2e --headed --slowmo=400 --video=on --output=test-results-e2e

The `live_server` fixture (conftest.py) seeds a real 2-layer corpus and launches
the actual `l44 serve` subprocess. The query test additionally hits Ollama for
generation and is marked `live`.
"""
from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

ORIGIN_RE = re.compile(r"^http://127\.0\.0\.1:\d+")


def test_home_loads_with_query_input(page: Page, live_server: str) -> None:
    """SC1: the home page renders the wordmark, a single query input, and example chips."""
    page.goto(live_server)
    expect(page).to_have_title(re.compile("Leopard 44 KB"))
    expect(page.locator(".wordmark")).to_have_text("Leopard 44 KB")
    expect(page.locator("#question")).to_be_visible()
    expect(page.locator("#ask-btn")).to_be_visible()
    expect(page.locator(".example-chip").first).to_be_visible()


def test_scope_toggle_persists_to_localstorage(page: Page, live_server: str) -> None:
    """SC3: scope defaults to All, switches to Vessel, persists across reload via localStorage."""
    page.goto(live_server)
    all_btn = page.locator('.scope-btn[data-scope="all"]')
    vessel_btn = page.locator('.scope-btn[data-scope="vessel"]')

    expect(all_btn).to_have_attribute("aria-pressed", "true")
    vessel_btn.click()
    expect(vessel_btn).to_have_attribute("aria-pressed", "true")
    expect(all_btn).to_have_attribute("aria-pressed", "false")
    assert page.evaluate("() => localStorage.getItem('l44-scope')") == "vessel"

    page.reload()
    expect(page.locator('.scope-btn[data-scope="vessel"]')).to_have_attribute(
        "aria-pressed", "true"
    )


def test_no_external_origin_requests(page: Page, live_server: str) -> None:
    """SC4 (offline): loading / and /explore issues zero requests to non-localhost origins."""
    external: list[str] = []

    def _on_request(req) -> None:  # noqa: ANN001 - playwright Request
        url = req.url
        if not (ORIGIN_RE.match(url) or url.startswith(("data:", "about:", "blob:"))):
            external.append(url)

    page.on("request", _on_request)
    page.goto(live_server)
    page.wait_for_load_state("networkidle")
    page.goto(f"{live_server}/explore")
    page.wait_for_load_state("networkidle")

    assert external == [], f"Unexpected external-origin requests: {external}"


def test_explore_lists_sources_by_layer(page: Page, live_server: str) -> None:
    """SC2/explore: the Explore view lists the seeded sources and their layers."""
    page.goto(f"{live_server}/explore")
    body = page.locator("body")
    expect(body).to_contain_text("Yanmar 4JH45 Service Notes")
    expect(body).to_contain_text("Leopard 44 KB Owner Modifications")
    expect(body).to_contain_text(re.compile("shared", re.IGNORECASE))
    expect(body).to_contain_text(re.compile("vessel", re.IGNORECASE))


@pytest.mark.live
def test_query_streams_answer_with_layer_sources(page: Page, live_server: str) -> None:
    """SC2 (live): a query streams an answer and renders layer-badged source cards.

    Requires Ollama (qwen2.5 generation). Asserts on STRUCTURE, not exact answer text:
    tokens arrive, a source card with a layer badge appears, the Sources section reveals.
    """
    page.goto(live_server)
    page.locator("#question").fill(
        "What is the raw-water impeller replacement interval on the Yanmar 4JH45?"
    )
    page.locator("#ask-btn").click()

    # Source cards paint before/with tokens (two-stage reveal). Retrieve is fast.
    sources = page.locator("#sources-list .source-card")
    expect(sources.first).to_be_visible(timeout=30_000)
    expect(page.locator("#sources")).to_be_visible()
    # The impeller fact lives in the shared layer → expect a SHARED badge.
    expect(page.locator(".source-card .badge").first).to_contain_text(
        re.compile("shared|vessel", re.IGNORECASE)
    )

    # Generation may be cold on the first call — allow a generous budget.
    answer_text = page.locator("#answer-area .answer-text")
    expect(answer_text).not_to_be_empty(timeout=60_000)
