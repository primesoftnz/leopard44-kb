# RED state until Plan 02 ships app.py (see VALIDATION.md).
# create_app is imported INSIDE each test function body — NEVER at module top level.
# This ensures pytest collection succeeds even before app.py exists (RED by assertion,
# not by collection error). The module-level SSE helper uses NO leopard44_kb.web imports.
"""Tests for UI-02/UI-03: SSE query endpoint — two-stage reveal, refusal, error states,
source shape, layer scoping, pipeline fidelity (review fixes D-05 + MEDIUM).

Per-requirement map source: .planning/phases/05-local-web-ui/05-VALIDATION.md
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import sqlite_vec
from tests._corpus import seed_corpus
from leopard44_kb.schema import apply_migrations


# ---------------------------------------------------------------------------
# Module-level SSE parsing helper — NO leopard44_kb.web import (collection-safe).
# RESEARCH Pattern 5: consume SSE stream from TestClient.stream().
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


def _seed_db(db_path: Path) -> None:
    """Bootstrap a file-backed test DB with the canonical retrieval corpus.

    Same pattern as test_query_cli.py::_seed_db — shared corpus from _corpus.py
    ensures the CLI and web paths cannot drift (IN-05).
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)
    seed_corpus(conn)
    conn.close()


# ---------------------------------------------------------------------------
# UI-02 / D-04: Two-stage reveal — source events before token events
# ---------------------------------------------------------------------------


def test_sources_before_tokens(monkeypatch, fake_embedder, fake_generator, tmp_path):
    """D-04: every 'source' event must appear before the first 'token' event.

    Two-stage reveal: chunks are known after retrieve(), so source cards
    must paint before any answer token streams.
    """
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "what is the impeller interval?", "layer": "all"},
    )

    source_indices = [i for i, (name, _) in enumerate(events) if name == "source"]
    token_indices = [i for i, (name, _) in enumerate(events) if name == "token"]

    assert source_indices, "Expected at least one 'source' event"
    assert token_indices, "Expected at least one 'token' event"
    assert max(source_indices) < min(token_indices), (
        f"All 'source' events must precede the first 'token' event (D-04).\n"
        f"Source event indices: {source_indices}\n"
        f"Token event indices: {token_indices}"
    )


# ---------------------------------------------------------------------------
# UI-02 / D-05 (HEADLINE review fix): Refusal emits zero sources AND zero tokens
# ---------------------------------------------------------------------------


def test_refusal_state(monkeypatch, fake_embedder, tmp_path):
    """D-05 (strengthened): a below-floor refusal emits NO source events AND NO token events.

    Forces a refusal by querying an empty DB (no chunks to retrieve).
    Asserts:
      (a) exactly one 'refusal' event carrying REFUSAL_MESSAGE verbatim
      (b) a terminal 'done' event
      (c) ZERO 'token' events
      (d) ZERO 'source' events (strengthened emit-order assertion — review HEADLINE fix)

    The old test only checked token count; this version pins that refusal must not
    paint source cards (D-05: refusal → REFUSAL_MESSAGE, no sources).
    """
    from leopard44_kb.answer import REFUSAL_MESSAGE

    # Empty DB — no sources, no chunks → below_floor guaranteed
    monkeypatch.setenv("L44_DB", str(tmp_path / "empty.db"))

    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "what is the impeller interval?", "layer": "all"},
    )

    event_names = [name for name, _ in events]
    refusal_events = [(name, data) for name, data in events if name == "refusal"]
    done_events = [name for name in event_names if name == "done"]
    token_events = [name for name in event_names if name == "token"]
    source_events = [name for name in event_names if name == "source"]

    # (a) Exactly one refusal event carrying the verbatim message
    assert len(refusal_events) == 1, (
        f"Expected exactly one 'refusal' event; got {refusal_events!r}"
    )
    assert refusal_events[0][1] == REFUSAL_MESSAGE, (
        f"Refusal data must equal REFUSAL_MESSAGE verbatim.\n"
        f"Expected: {REFUSAL_MESSAGE!r}\n"
        f"Got: {refusal_events[0][1]!r}"
    )

    # (b) Terminal 'done' event
    assert done_events, "Expected a terminal 'done' event after refusal"

    # (c) Zero token events
    assert token_events == [], (
        f"Expected ZERO 'token' events on refusal path; got: {token_events!r}"
    )

    # (d) Zero source events (strengthened D-05 check)
    assert source_events == [], (
        f"Expected ZERO 'source' events on refusal path (D-05); got: {source_events!r}"
    )


# ---------------------------------------------------------------------------
# UI-02 / D-05: Ollama-down renders an error event (not a silent hang)
# ---------------------------------------------------------------------------


def test_ollama_down_error(monkeypatch, fake_embedder, tmp_path):
    """D-05: when stream_generate raises RuntimeError, an 'error' event is emitted.

    Monkeypatches leopard44_kb.answer.stream_generate to raise the canonical
    'Ollama not reachable' RuntimeError. Asserts:
      - an 'error' event whose data contains the RuntimeError text
      - a terminal 'done' event
      - no silent hang (TestClient returns normally)
    """
    import leopard44_kb.answer as ans

    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    error_text = "Ollama not reachable at :11434 — run `ollama serve`"

    def _raise_runtimeerror(*args, **kwargs):
        raise RuntimeError(error_text)
        yield  # makes it a generator

    monkeypatch.setattr(ans, "stream_generate", _raise_runtimeerror)

    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "what is the impeller interval?", "layer": "all"},
    )

    error_events = [(name, data) for name, data in events if name == "error"]
    done_events = [name for name, _ in events if name == "done"]

    assert error_events, "Expected at least one 'error' event when Ollama is down"
    assert any(error_text in data for _, data in error_events), (
        f"'error' event data must contain the RuntimeError text.\n"
        f"Expected substring: {error_text!r}\n"
        f"Got error events: {error_events!r}"
    )
    assert done_events, "Expected a terminal 'done' event after error"


# ---------------------------------------------------------------------------
# UI-02: Source event JSON shape
# ---------------------------------------------------------------------------


def test_source_event_shape(monkeypatch, fake_embedder, fake_generator, tmp_path):
    """UI-02: the first 'source' event's JSON has required keys with valid layer."""
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "what is the impeller interval?", "layer": "all"},
    )

    source_events = [(name, data) for name, data in events if name == "source"]
    assert source_events, "Expected at least one 'source' event"

    first_data = json.loads(source_events[0][1])
    # "content" + "section_path" added for Option A (expandable cited passage).
    required_keys = {"n", "layer", "title", "page_start", "page_end", "content", "section_path"}
    missing = required_keys - set(first_data.keys())
    assert not missing, (
        f"Source event JSON missing keys: {missing}. Got: {first_data!r}"
    )

    valid_layers = {"shared", "vessel", "community"}
    assert first_data["layer"] in valid_layers, (
        f"Source event 'layer' must be one of {valid_layers}; got: {first_data['layer']!r}"
    )

    # Option A: the source event carries the exact retrieved passage so the UI can
    # reveal "show the passage used". It must be a non-empty string (the seeded
    # chunk has real content) and a substring of the answer's grounding chunk.
    assert isinstance(first_data["content"], str) and first_data["content"].strip(), (
        f"Source event 'content' must be a non-empty passage; got: {first_data['content']!r}"
    )
    assert isinstance(first_data["section_path"], str), (
        f"Source event 'section_path' must be a string; got: {first_data['section_path']!r}"
    )


# ---------------------------------------------------------------------------
# WR-04: page_start=0 is not falsy — must appear as 0, not null/omitted
# ---------------------------------------------------------------------------


def test_page_zero_not_falsy(monkeypatch, fake_embedder, fake_generator, tmp_path):
    """WR-04: a chunk with page_start=0 must emit page_start=0 in the source event JSON.

    Seeds a custom chunk with page_start=0 (cover page) and asserts the source
    event carries 0, not null (None) or omitted (WR-04 from answer.py pattern:
    'test against None, not truthiness').
    """
    import sqlite3

    db_path = tmp_path / "test.db"
    # Bootstrap a DB with a page_start=0 chunk
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)

    conn.execute(
        "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
        "VALUES (1,'shared','pdf','shared/cover.pdf','h1','Cover Doc')"
    )
    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,"
        "page_end,content,content_hash,anchor_key,embedding_model,embedding_model_version) "
        "VALUES (1,1,'shared',0,'Intro',0,0,'Cover page content.','hc','akc','m','v')"
    )
    import struct
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
        "VALUES (1,'shared',1,'m',1,?)",
        (struct.pack("384f", *([0.1] * 384)),),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "cover page", "layer": "all"},
    )

    source_events = [(name, data) for name, data in events if name == "source"]
    assert source_events, "Expected at least one 'source' event"

    # Find a source event with page_start=0
    page_zero_found = False
    for _, data in source_events:
        parsed = json.loads(data)
        if parsed.get("page_start") == 0:
            page_zero_found = True
            break

    assert page_zero_found, (
        f"Expected a source event with page_start=0 (WR-04: 0 is valid, not falsy).\n"
        f"Source events: {[json.loads(d) for _, d in source_events]!r}"
    )


# ---------------------------------------------------------------------------
# UI-03: Layer scope — shared query produces no vessel source events
# ---------------------------------------------------------------------------


def test_layer_scope(monkeypatch, fake_embedder, fake_generator, tmp_path):
    """UI-03: POST /query with layer='shared' must not return any vessel source events.

    Uses a fake_retrieve fixture to deterministically control which chunks are
    returned, asserting the layer parameter is honored end-to-end.
    """
    import leopard44_kb.retrieve as ret

    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "what is the impeller interval?", "layer": "shared"},
    )

    source_events = [(name, data) for name, data in events if name == "source"]
    vessel_sources = [
        json.loads(data) for _, data in source_events
        if json.loads(data).get("layer") == "vessel"
    ]
    assert vessel_sources == [], (
        f"No vessel source events should appear with layer='shared' (UI-03).\n"
        f"Got vessel sources: {vessel_sources!r}"
    )


# ---------------------------------------------------------------------------
# C5: Bad citations reported in 'done' event
# ---------------------------------------------------------------------------


def test_bad_citations_reported(monkeypatch, fake_embedder, out_of_range_generator, tmp_path):
    """C5: when the model emits an out-of-range [9] citation, the 'done' event's
    bad_citations list must be non-empty.

    Uses out_of_range_generator fixture from conftest.py (yields [1] + [9]).
    """
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "what is the impeller interval?", "layer": "all"},
    )

    done_events = [(name, data) for name, data in events if name == "done"]
    assert done_events, "Expected a terminal 'done' event"

    done_data = json.loads(done_events[0][1])
    assert "bad_citations" in done_data, (
        f"'done' event must have 'bad_citations' key; got: {done_data!r}"
    )
    assert len(done_data["bad_citations"]) > 0, (
        f"Expected non-empty bad_citations with out-of-range [9]; got: {done_data['bad_citations']!r}"
    )


# ---------------------------------------------------------------------------
# Pipeline fidelity (review fix MEDIUM): assert Phase 3 helpers are called
# ---------------------------------------------------------------------------


def test_pipeline_fidelity(monkeypatch, fake_embedder, tmp_path):
    """Review fix MEDIUM: POST /query must call the SAME Phase 3 helpers as ask_cmd.

    Spies on leopard44_kb.answer.build_user_message, stream_generate, validate_citations,
    and asserts:
      - build_user_message called once with (question, chunks)
      - stream_generate called with num_predict == select_num_predict(tier, model)
      - validate_citations called with len(chunks)
      - system prompt passed to stream_generate equals SYSTEM_PROMPT.format(n_chunks=len(chunks))

    This proves /query uses the Phase 3 pipeline, not a near-reimplementation.
    """
    import leopard44_kb.answer as ans
    from leopard44_kb.answer import (
        SYSTEM_PROMPT,
        select_generation_model,
        select_num_predict,
    )

    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    monkeypatch.setenv("L44_DB", str(db_path))

    # Spies — record call arguments
    build_calls: list[tuple] = []
    stream_calls: list[dict] = []
    validate_calls: list[tuple] = []

    original_build = ans.build_user_message
    original_validate = ans.validate_citations

    def _spy_build(question, chunks):
        build_calls.append((question, chunks))
        return original_build(question, chunks)

    tokens_sent: list[str] = []

    def _spy_stream(model, system_prompt, user_message, num_predict=75, temperature=0.15):
        stream_calls.append({
            "model": model,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "num_predict": num_predict,
        })
        # Yield a small deterministic response with valid citations
        for tok in ["Impeller ", "interval ", "[1]."]:
            tokens_sent.append(tok)
            yield tok

    def _spy_validate(text, n_chunks):
        validate_calls.append((text, n_chunks))
        return original_validate(text, n_chunks)

    monkeypatch.setattr(ans, "build_user_message", _spy_build)
    monkeypatch.setattr(ans, "stream_generate", _spy_stream)
    monkeypatch.setattr(ans, "validate_citations", _spy_validate)

    from leopard44_kb.web.app import create_app  # RED until Plan 02
    from fastapi.testclient import TestClient

    question = "what is the impeller interval?"
    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": question, "layer": "all"},
    )

    # Assert build_user_message was called once
    assert len(build_calls) == 1, (
        f"build_user_message must be called exactly once; called {len(build_calls)} times"
    )
    assert build_calls[0][0] == question, (
        f"build_user_message must be called with the original question; got: {build_calls[0][0]!r}"
    )
    chunks_arg = build_calls[0][1]

    # Assert stream_generate called with the tier-resolved num_predict — proving
    # the web path computes the cap via the SAME helper as ask_cmd, not a hard 75.
    gen_model, tier_label = select_generation_model()
    expected_num_predict = select_num_predict(tier_label, gen_model)
    assert len(stream_calls) == 1, (
        f"stream_generate must be called exactly once; called {len(stream_calls)} times"
    )
    assert stream_calls[0]["num_predict"] == expected_num_predict, (
        f"stream_generate must be called with num_predict={expected_num_predict} "
        f"(select_num_predict for tier {tier_label!r}); "
        f"got: {stream_calls[0]['num_predict']!r}"
    )

    # Assert system prompt equals SYSTEM_PROMPT.format(n_chunks=len(chunks))
    expected_system = SYSTEM_PROMPT.format(n_chunks=len(chunks_arg))
    assert stream_calls[0]["system_prompt"] == expected_system, (
        f"system prompt must equal SYSTEM_PROMPT.format(n_chunks={len(chunks_arg)}).\n"
        f"Expected: {expected_system!r}\n"
        f"Got: {stream_calls[0]['system_prompt']!r}"
    )

    # Assert validate_citations was called with len(chunks)
    assert len(validate_calls) == 1, (
        f"validate_citations must be called exactly once; called {len(validate_calls)} times"
    )
    assert validate_calls[0][1] == len(chunks_arg), (
        f"validate_citations must be called with n_chunks={len(chunks_arg)}; "
        f"got: {validate_calls[0][1]!r}"
    )


# ---------------------------------------------------------------------------
# Phase 9 / VIS-01: zone_highlight SSE event — before done, exact contract
# RED until 09-04 wires zone_highlight emission into query_endpoint().
# ---------------------------------------------------------------------------


def test_zone_highlight_event(monkeypatch, fake_embedder, tmp_path):
    """zone_highlight SSE event is emitted BEFORE done when an item chunk has a zone with geometry.

    Seeding:
      - A zone with geometry set + schematic_image set
      - An item placed in that zone
      - An item chunk with metadata.item_id pointing to that item

    Asserts (review concern 1 — load-bearing):
      (a) a 'zone_highlight' event is present in the SSE stream
      (b) its payload has non-None geometry field
      (c) name/cue come from zone label/vertical_desc (D-10)
      (d) CRITICAL: zone_highlight index < done index (emitted BEFORE done, after source events)

    The ordered list of (event_name, data) pairs from _parse_sse_events is used
    to assert ordering exactly — not just presence.
    """
    import struct

    db_path = tmp_path / "test_highlight.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)

    # Seed a zone with geometry + schematic_image
    geometry_json = "[[0.1,0.2],[0.5,0.2],[0.5,0.8]]"
    conn.execute(
        "UPDATE zones SET geometry = ?, schematic_image = ?, vertical_desc = ? WHERE id = 1",
        (geometry_json, "page_061.png", "Port side lower shelf"),
    )

    # Look up the zone name for later assertion
    zone_row = conn.execute(
        "SELECT label, vertical_desc FROM zones WHERE id = 1"
    ).fetchone()
    zone_label = zone_row["label"]
    zone_cue = zone_row["vertical_desc"]

    # Seed a source + item chunk that references item_id in metadata
    conn.execute(
        "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
        "VALUES (99,'vessel','text','vessel/items.txt','hzone','Vessel items')"
    )

    # Seed an item in zone 1 (items table added by 002_inventory.sql)
    # category is NOT NULL in 002_inventory.sql; supply a valid value
    conn.execute(
        "INSERT INTO items(id, name, category, current_zone_id) "
        "VALUES (1, 'Test item', 'spare', 1)"
    )

    # Seed a vec chunk with metadata.item_id = 1 so zone_highlight can be found
    # The chunk content should be retrievable via the fake_embedder (all-0.1 vectors)
    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,"
        "page_end,content,content_hash,anchor_key,embedding_model,embedding_model_version,"
        "metadata) "
        "VALUES (99,99,'vessel',0,'Items',0,0,'Test item is stored here','hvc','akv','m','v',"
        "'{\"item_id\": 1}')"
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
        "VALUES (99,'vessel',99,'m',1,?)",
        (struct.pack("384f", *([0.1] * 384)),),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until 09-04
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "where is the test item?", "layer": "all"},
    )

    # Collect ordered event names for ordering assertion
    event_names = [name for name, _ in events]

    # (a) zone_highlight event must be present
    highlight_events = [(i, name, data) for i, (name, data) in enumerate(events)
                        if name == "zone_highlight"]
    assert highlight_events, (
        f"Expected a 'zone_highlight' event in SSE stream; got events: {event_names!r}"
    )

    # (b) geometry field must be non-None
    first_highlight_idx, _, first_highlight_data = highlight_events[0]
    payload = json.loads(first_highlight_data)
    assert payload.get("geometry") is not None, (
        f"zone_highlight payload must have non-None geometry; got: {payload!r}"
    )

    # (c) name and cue from zone row
    assert "name" in payload, f"zone_highlight payload missing 'name' key: {payload!r}"
    assert "cue" in payload, f"zone_highlight payload missing 'cue' key: {payload!r}"

    # (d) CRITICAL: zone_highlight must appear BEFORE done (review concern 1)
    done_indices = [i for i, (name, _) in enumerate(events) if name == "done"]
    assert done_indices, f"Expected a 'done' event in SSE stream; got: {event_names!r}"
    done_idx = done_indices[0]

    assert first_highlight_idx < done_idx, (
        f"zone_highlight (index {first_highlight_idx}) must appear BEFORE done (index {done_idx}).\n"
        f"Event order: {event_names!r}\n"
        "This is the load-bearing ordering constraint (review concern 1 / VIS-01)."
    )


def test_zone_highlight_graceful_degradation(monkeypatch, fake_embedder, tmp_path):
    """zone_highlight SSE event is emitted with geometry=None when zone has no polygon.

    Pins the EXACT graceful-degradation contract (review MEDIUM concern):
      - event IS emitted (name='zone_highlight' present) — client shows name + cue
      - payload geometry field IS None (not absent; not event-absent)

    Also asserts the event still precedes 'done' (ordering contract must hold even
    when geometry is absent).
    """
    import struct

    db_path = tmp_path / "test_degrade.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn)

    # Zone 2 — leave geometry NULL (no polygon set); set only schematic_image
    conn.execute(
        "UPDATE zones SET geometry = NULL, schematic_image = 'page_061.png',"
        " vertical_desc = 'Starboard shelf' WHERE id = 2"
    )

    # Seed a source + item in zone 2 with an item chunk
    conn.execute(
        "INSERT INTO sources(id,layer,source_type,path,content_hash,title) "
        "VALUES (98,'vessel','text','vessel/items2.txt','hdeg','Degradation test')"
    )
    # category is NOT NULL in 002_inventory.sql; supply a valid value
    conn.execute(
        "INSERT INTO items(id, name, category, current_zone_id) "
        "VALUES (2, 'Degrade item', 'tool', 2)"
    )
    conn.execute(
        "INSERT INTO chunks(id,source_id,layer,ordinal,section_path,page_start,"
        "page_end,content,content_hash,anchor_key,embedding_model,embedding_model_version,"
        "metadata) "
        "VALUES (98,98,'vessel',0,'Items',0,0,'Degrade item lives here','hdegc','akdeg','m','v',"
        "'{\"item_id\": 2}')"
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id,layer,source_id,embedding_model,is_active,embedding) "
        "VALUES (98,'vessel',98,'m',1,?)",
        (struct.pack("384f", *([0.1] * 384)),),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("L44_DB", str(db_path))

    from leopard44_kb.web.app import create_app  # RED until 09-04
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    events = _parse_sse_events(
        client, "POST", "/query",
        json={"question": "where is the degrade item?", "layer": "all"},
    )

    event_names = [name for name, _ in events]

    # Event MUST be present (graceful degradation: emit even without geometry)
    highlight_events = [(i, name, data) for i, (name, data) in enumerate(events)
                        if name == "zone_highlight"]
    assert highlight_events, (
        f"Expected 'zone_highlight' event even when zone geometry is NULL (graceful degradation).\n"
        f"Got events: {event_names!r}"
    )

    # Geometry field must be explicitly None (not absent — exact contract)
    first_highlight_idx, _, first_highlight_data = highlight_events[0]
    payload = json.loads(first_highlight_data)
    assert "geometry" in payload, (
        f"zone_highlight payload must include 'geometry' key even when NULL; got: {payload!r}"
    )
    assert payload["geometry"] is None, (
        f"zone_highlight geometry must be None when zone has no polygon; "
        f"got: {payload['geometry']!r}. "
        "This is the EXACT graceful-degradation contract (review MEDIUM concern)."
    )

    # Ordering: zone_highlight must still precede done
    done_indices = [i for i, (name, _) in enumerate(events) if name == "done"]
    assert done_indices, f"Expected 'done' event; got: {event_names!r}"
    assert first_highlight_idx < done_indices[0], (
        f"zone_highlight (index {first_highlight_idx}) must precede done (index {done_indices[0]}) "
        f"even during graceful degradation.\nEvent order: {event_names!r}"
    )
