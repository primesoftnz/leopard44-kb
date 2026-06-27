"""Leopard 44 KB ingestion pipeline package (Phase 2)."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Supported file suffixes for ingest (used by cli.py for directory filtering).
SUPPORTED_SUFFIXES: tuple[str, ...] = (".pdf", ".md", ".markdown", ".txt", ".zip")

#: Maximum number of texts sent in a single /api/embed request.
#: Bounding sub-batch size prevents timeouts on large documents (Gemini review MEDIUM).
EMBED_SUB_BATCH: int = 64

# parents[3] = repo root for src/leopard44_kb/ingest/__init__.py:
#   parents[0] = src/leopard44_kb/ingest/
#   parents[1] = src/leopard44_kb/
#   parents[2] = src/
#   parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_file(path: Path, source_type: str, ocr: bool = False) -> list[dict]:
    """Dispatch to the appropriate parser based on source_type.

    Returns a list of chunk dicts with keys:
        section_path, content, page_start, page_end, section_ordinal, token_count
    (no 'embedding' key — that is attached by ingest_file after parsing).
    """
    if source_type == "pdf":
        from leopard44_kb.ingest.pdf import parse_pdf

        return parse_pdf(path, str(path), ocr=ocr)
    elif source_type == "whatsapp":
        from leopard44_kb.ingest.whatsapp import extract_chat_txt, parse_whatsapp_file

        if path.suffix.lower() == ".zip":
            # Safe targeted extraction — caller keeps original zip as the source path.
            # The TemporaryDirectory is kept alive until parsing returns (extraction result
            # holds the handle; it is GC'd automatically when the local var goes out of scope).
            extraction = extract_chat_txt(path)
            # Parse the extracted _chat.txt; source_path_str is the ORIGINAL zip path
            # so anchor_key is stable for the zip even if temp dir path varies.
            return parse_whatsapp_file(extraction.path, str(path))
        return parse_whatsapp_file(path, str(path))
    elif source_type == "maintenance_entry":
        from leopard44_kb.ingest.text_md import parse_maintenance_entry

        return parse_maintenance_entry(path, str(path))
    elif source_type == "deviation":
        from leopard44_kb.ingest.text_md import parse_deviation_entry

        return parse_deviation_entry(path, str(path))
    elif source_type == "markdown":
        from leopard44_kb.ingest.text_md import parse_markdown

        return parse_markdown(path, str(path))
    else:
        # plain text
        from leopard44_kb.ingest.text_md import parse_text

        return parse_text(path, str(path))


def _detect_source_type(path: Path) -> str:
    """Return the source_type string for a given file path."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".zip":
        return "whatsapp"
    if suffix in (".md", ".markdown"):
        # Narrowed detection: only files under the canonical data/logs/maint/ tree are
        # maintenance entries.  Check for the CONSECUTIVE sequence ("data", "logs", "maint")
        # in the path parts so an unrelated .../maint/manual.md stays "markdown".
        parts = path.parts
        for i in range(len(parts) - 2):
            if parts[i] == "data" and parts[i + 1] == "logs" and parts[i + 2] == "maint":
                return "maintenance_entry"
        # Generated deviation entries live under the canonical data/deviations/ tree.
        # Detect them by the CONSECUTIVE ("data", "deviations") sequence anywhere in the
        # path — symmetric with the maintenance_entry rule above. This keeps source_type
        # 'deviation' on a broad `l44 ingest data/` re-ingest (preserving the
        # deviation_id/zone_id chunk metadata that drives retrieval + the blue schematic
        # highlight) AND makes the vessel-only layer guard in ingest_file() fire for a
        # deviation file addressed by ANY path.
        #
        # WR-01 fix: the earlier _REPO_ROOT-anchored absolute check returned "markdown"
        # for a deviation file under a non-canonical absolute path (a second clone, a
        # relocated/symlinked data/ dir — exactly what ingest_file's own repo-root walk
        # below supports), which silently bypassed the vessel-only guard and could let
        # PRIVATE deviation content be ingested as --layer community. Matching the
        # maintenance detector fails CLOSED: any data/deviations/ path is vessel-only.
        for i in range(len(parts) - 1):
            if parts[i] == "data" and parts[i + 1] == "deviations":
                return "deviation"
        return "markdown"
    # .txt: use looks_like_whatsapp() to peek at content, then directory-name fallback.
    if suffix == ".txt":
        try:
            from leopard44_kb.ingest.whatsapp import looks_like_whatsapp

            if looks_like_whatsapp(path):
                return "whatsapp"
        except Exception:
            pass
        # Directory-name fallback (e.g. files inside a 'whatsapp' directory).
        parent_parts = [p.lower() for p in path.parts]
        if any("whatsapp" in p for p in parent_parts):
            return "whatsapp"
        return "text"
    return "text"


def ingest_file(
    path: "str | Path",
    layer: str = "vessel",
    ocr: bool = False,
    conn: Optional[sqlite3.Connection] = None,
    title: "str | None" = None,
) -> str:
    """Ingest one file into the store. Returns 'ok' or 'no-op'.

    Two-phase design: first calls store_source_and_chunks(chunks=None) to check
    whether a re-ingest is needed (no-op guard, no parsing/embedding overhead).
    If the file has changed or is new, parses and embeds the file, then calls
    store_source_and_chunks again with the real chunks (the write phase).

    This two-call pattern allows accurate monkeypatching in atomicity tests (the
    second call to store_source_and_chunks is the one that does SQL writes, so a
    failure there tests transaction rollback).

    Args:
        path: File path (absolute or relative to cwd).
        layer: 'vessel' | 'shared' | 'community'. Default 'vessel' per D-06.
        ocr: If True, run OCR on image-only PDF pages (requires tesseract).
        conn: Optional pre-opened connection (for testing). Opens a new one if None.

    Returns:
        'no-op' if file is unchanged (content_hash match); 'ok' otherwise.

    Raises:
        RuntimeError: If Ollama is unreachable (D-09 hard-fail).
        ValueError: If path fails layer validation (paths.validate_path).
    """
    path = Path(path)

    # Layer-leak guard (D-14 / Codex HIGH #3): maintenance entries AND deviation
    # entries are vessel-only (owner-private boat facts, never shared/community).
    # This guard runs as the VERY FIRST check so no store call is ever made and no
    # row is written.  It must precede validate_path so the rejection message is
    # unambiguous ("vessel-only") regardless of where the file lives.
    source_type_early = _detect_source_type(path)
    if source_type_early in ("maintenance_entry", "deviation") and layer != "vessel":
        raise ValueError(
            f"{source_type_early} entries are vessel-only; refusing to ingest {path} as layer={layer!r}"
        )

    import leopard44_kb.ingest.embedder as _emb
    import leopard44_kb.ingest.writer as _writer
    from leopard44_kb.paths import ALLOWED_ROOTS, validate_path
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    # Determine the repo root for path validation. In production, _REPO_ROOT is the real repo.
    # In tests, files live under tmp_path/data/... so we derive a local repo root by walking
    # up from the file until we find a parent whose child matches the expected layer root name.
    resolved = path.resolve()
    layer_root_name = ALLOWED_ROOTS.get(layer, "data")
    repo_root = _REPO_ROOT
    for parent in resolved.parents:
        candidate = parent / layer_root_name
        if candidate.exists() and resolved.is_relative_to(candidate):
            repo_root = parent
            break

    # Validate the path is inside the correct layer root.
    validate_path(layer, path, repo_root)

    # Manage connection lifecycle.
    _own_conn = conn is None
    if _own_conn:
        conn = open_db()
        apply_migrations(conn)

    try:
        # Compute content hash first.
        content_hash = _writer.file_content_hash(path)
        source_type = _detect_source_type(path)

        # Phase 1: no-op check (chunks=None → check mode, no SQL writes).
        # This is the FIRST call to store_source_and_chunks.
        # Returns 'no-op' immediately if file is unchanged (avoids parsing + embedding).
        # Returns 'ok' (proceed signal) if the file is new or has changed.
        # Note: title may be None here for the derived-title case (harmless — check mode
        # writes nothing; the write call below uses the derived title).
        model_name, model_version = _emb.select_model()
        check_result = _writer.store_source_and_chunks(
            conn,
            layer,
            str(path),
            source_type,
            content_hash,
            None,  # check mode: no SQL writes
            model_name,
            model_version,
            title=title,
        )
        if check_result == "no-op":
            return "no-op"

        # Phase 2: parse → embed → write.
        # embed_texts is NOT called for no-op (test_noop_skips_parser_and_ollama).
        chunks = _parse_file(path, source_type, ocr=ocr)

        # Title derivation for maintenance entries when no explicit title is passed.
        # Must run AFTER parsing so we can read metadata from the chunk.
        if source_type == "maintenance_entry" and title is None and chunks:
            title = f"Maintenance log {chunks[0].get('metadata', {}).get('date')}"

        if chunks:
            # Embed in bounded sub-batches (Gemini concern: a 200+-chunk document
            # must not produce a single oversized /api/embed request → cap at EMBED_SUB_BATCH).
            texts = [c["content"] for c in chunks]
            embeddings: list[list[float]] = []
            for start in range(0, len(texts), EMBED_SUB_BATCH):
                batch = texts[start : start + EMBED_SUB_BATCH]
                embeddings.extend(_emb.embed_texts(batch, model_name))
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding

        # Phase 2 write: SECOND call to store_source_and_chunks (does delete+insert).
        # title is now the derived value (for maintenance entries) or the caller-supplied
        # value (or None for non-maintenance files — all existing callers unaffected).
        return _writer.store_source_and_chunks(
            conn,
            layer,
            str(path),
            source_type,
            content_hash,
            chunks,
            model_name,
            model_version,
            title=title,
        )
    finally:
        if _own_conn:
            conn.close()
