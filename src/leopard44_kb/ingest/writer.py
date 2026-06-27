"""Write-side store layer: store_source_and_chunks() for the ingest pipeline (Phase 2)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import unicodedata
from pathlib import Path


def file_content_hash(path: Path) -> str:
    """Return a stable SHA-256 hash of a file's content, normalised for cross-platform stability.

    Normalisation order: BOM strip → NFC unicode → CRLF/CR → LF → sha256(utf-8 bytes).
    Binary files (PDF, zip) that cannot be decoded as utf-8-sig are hashed as raw bytes.

    Note for v1.0: .zip exports are hashed as raw bytes, so a zip whose _chat.txt is identical
    but whose container metadata differs WILL re-ingest — accepted for v1.0; revisit only if
    it becomes a real annoyance.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")  # strips BOM if present
    except UnicodeDecodeError:
        # Binary file (PDF, zip): hash raw bytes directly, no text normalisation.
        return hashlib.sha256(raw).hexdigest()
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_anchor_key(source_path: str, section_path: str, ordinal_within_section: int) -> str:
    """Return a stable anchor key for a chunk (D-11, Pitfall 3 from 01-RESEARCH.md).

    The ordinal is relative to section_path, not to the whole source, so that an insert
    in one section does not shift anchor_keys in sibling sections.

    anchor_key = sha256(source_path + '\\n' + section_path + '\\n' + ordinal_within_section)
    """
    raw = f"{source_path}\n{section_path}\n{ordinal_within_section}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def pack_embedding(vec: list[float]) -> bytes:
    """Pack a float list into sqlite-vec's binary format (little-endian IEEE 754 floats)."""
    return struct.pack(f"{len(vec)}f", *vec)


def store_source_and_chunks(
    conn: sqlite3.Connection,
    layer: str,
    path_str: str,
    source_type: str,
    content_hash: str,
    chunks: "list[dict] | None",
    model_name: str,
    model_version: str,
    *,
    title: "str | None" = None,
) -> str:
    """Upsert source + chunks + vec_chunks for one file. Returns 'ok' or 'no-op'.

    This function operates in two modes based on the `chunks` parameter:

    **Check mode** (chunks=None): Only checks whether a re-ingest is needed.
        Returns 'no-op' if content_hash matches the existing source; returns 'ok'
        (a "proceed" signal) if the file has changed or is new, WITHOUT doing any
        SQL writes. The caller should then parse/embed the file and call this
        function again with the actual chunks.

    **Write mode** (chunks=list): Performs the full delete+insert inside ONE `with conn:`
        transaction. The delete (vec_chunks first, then source via cascade) AND all inserts
        happen atomically — a failure rolls back the entire operation and the old source
        is preserved.

    Re-ingest sequence — CRITICAL (vec_chunks has no FK CASCADE, Pitfall 1):
        In write mode with a changed file:
        a. DELETE FROM vec_chunks WHERE source_id=?  (BEFORE sources — no FK cascade on vec0)
        b. DELETE FROM sources WHERE id=?             (cascades chunks + FTS via triggers)
        c. INSERT INTO sources …                      (fresh source row)
        d. INSERT INTO chunks … for each chunk        (FTS auto-synced by chunks_ai trigger)
        e. INSERT INTO vec_chunks … for each chunk    (explicit — no FK cascade)
        ALL of steps a–e are inside one `with conn:` for atomic rollback on failure.

    Ordinal contract (Codex review HIGH concern):
    - The WRITER assigns chunks.ordinal globally (0-based, across the whole source) for
      UNIQUE(source_id, ordinal). This is INDEPENDENT of the parser's section_ordinal.
    - The writer feeds the parser's chunk["section_ordinal"] into compute_anchor_key() only.
    - The writer NEVER reads a parser-supplied global "ordinal" key.
    """
    # Step 1: check for an existing source.
    existing = conn.execute(
        "SELECT id, content_hash FROM sources WHERE layer=? AND path=?",
        (layer, path_str),
    ).fetchone()

    # Step 2: NO-OP SHORT-CIRCUIT — return before any `with conn:` block.
    if existing is not None and existing["content_hash"] == content_hash:
        return "no-op"

    # Check-only mode: caller will parse/embed then call again with actual chunks.
    if chunks is None:
        return "ok"  # "proceed" signal: file is new or changed, write needed

    # Step 3: Write mode — everything inside ONE `with conn:` transaction.
    # This ensures the delete and all inserts are atomic: a failure at any point
    # (e.g. malformed chunk, packing error) rolls back the whole unit — the old
    # source is NOT lost (Codex review HIGH concern T-02-17).
    with conn:
        # Step 3a/3b: delete old data if a changed source exists.
        if existing is not None:
            old_src_id = existing["id"]
            # vec_chunks FIRST (virtual table — no FK cascade, Pitfall 1).
            conn.execute("DELETE FROM vec_chunks WHERE source_id=?", (old_src_id,))
            # sources DELETE cascades to chunks (ON DELETE CASCADE) + FTS via chunks_ad trigger.
            conn.execute("DELETE FROM sources WHERE id=?", (old_src_id,))

        # Step 3c: insert fresh source row.
        conn.execute(
            "INSERT INTO sources (layer, source_type, path, content_hash, title)"
            " VALUES (?, ?, ?, ?, ?)",
            (layer, source_type, path_str, content_hash, title),
        )
        source_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Step 3d/3e: insert each chunk + its vec row.
        # gidx = global ordinal for chunks.ordinal column (UNIQUE(source_id, ordinal)).
        # chunk["section_ordinal"] = parser's section-relative ordinal (for anchor_key only).
        for gidx, chunk in enumerate(chunks):
            content: str = chunk["content"]
            section_path: str = chunk.get("section_path") or ""
            page_start = chunk.get("page_start")
            page_end = chunk.get("page_end")
            token_count = chunk.get("token_count")
            section_ordinal: int = chunk.get("section_ordinal", gidx)
            embedding: list[float] = chunk["embedding"]

            chunk_content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            anchor_key = compute_anchor_key(path_str, section_path, section_ordinal)
            chunk_metadata = chunk.get("metadata")

            conn.execute(
                "INSERT INTO chunks "
                "(source_id, layer, ordinal, section_path, page_start, page_end, "
                " token_count, content, content_hash, anchor_key, "
                " embedding_model, embedding_model_version, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    layer,
                    gidx,  # global ordinal — writer assigns, not parser
                    section_path,
                    page_start,
                    page_end,
                    token_count,
                    content,
                    chunk_content_hash,
                    anchor_key,
                    model_name,
                    model_version,
                    json.dumps(chunk_metadata) if chunk_metadata is not None else None,
                ),
            )
            chunk_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            conn.execute(
                "INSERT INTO vec_chunks(chunk_id, layer, source_id, embedding_model, is_active, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (chunk_id, layer, source_id, model_name, 1, pack_embedding(embedding)),
            )

    return "ok"
