"""Markdown and plain-text note parser (INGEST-05, D-07).

Markdown: heading hierarchy → section_path.
Plain .txt: blank-line block index or date-header → deterministic synthetic section_path.
"""
from __future__ import annotations

import re
from pathlib import Path

# Attempt to import tiktoken; fall back to a deterministic word-count heuristic
# if the BPE cache is unavailable offline (Pitfall 5 in 02-RESEARCH.md).
try:
    import tiktoken as _tiktoken

    _ENC = _tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Count cl100k_base tokens."""
        return len(_ENC.encode(text))

except Exception:
    # Offline fallback: whitespace split + half the punctuation count.
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        """Rough token estimate used when tiktoken cache is unavailable."""
        words = text.split()
        punct = len(re.findall(r"[^\w\s]", text))
        return len(words) + punct // 2


# Default token cap (safe for all-MiniLM 256 word-piece limit; see RESEARCH.md Pattern 8).
TOKEN_CAP = 200

# Regex to detect Markdown headings (ATX style: # H1, ## H2, etc.)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)

# Regex for date-header detection in plain .txt (YYYY-MM-DD or DD/MM/YYYY at line start).
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})")


def parse_markdown(path: Path, source_path_str: str, token_cap: int = TOKEN_CAP) -> list[dict]:
    """Chunk a Markdown file on heading boundaries (# H1 > ## H2 > ### H3).

    Returns a list of chunk dicts with keys:
        section_path, content, page_start, page_end, section_ordinal, token_count.

    The 'embedding' key is NOT set here — the caller (ingest_file) attaches it.
    """
    text = path.read_text(encoding="utf-8")
    return _chunk_markdown(text, source_path_str, token_cap)


def _chunk_markdown(text: str, source_path_str: str, token_cap: int = TOKEN_CAP) -> list[dict]:
    """Internal: parse markdown text into chunk dicts."""
    lines = text.splitlines(keepends=True)
    chunks: list[dict] = []

    # Heading stack tracks the current hierarchy, e.g. ["# Overview", "## Setup"].
    heading_stack: list[str] = []
    current_lines: list[str] = []

    # Per-section_path ordinal counter (resets when section_path changes).
    section_ordinal_counter: dict[str, int] = {}

    def flush_chunk() -> None:
        """Emit the current buffer as a chunk.

        The FULL heading hierarchy (every level of heading_stack) is prepended to the
        content so each chunk is discoverable by ancestor-section terms in both FTS and
        the vector space — e.g. a "Poor Windward Performance" chunk nested under
        "# Known Issues" matches a "known issues" query — and so the LLM sees which
        document/section the chunk came from.

        Heading-only sections (a heading immediately followed by another heading, with
        no body) are NOT emitted: a content-free chunk carries no retrievable
        information and pollutes top-k with hits that make the LLM refuse. The heading
        still survives in the section_path prepended to the next real chunk.
        """
        body = "".join(current_lines).strip()
        section_path = " > ".join(heading_stack) if heading_stack else "Preamble"
        # Skip heading-only / empty chunks (the heading lives on in section_path and
        # in the next real chunk's prepended hierarchy).
        if not body:
            return
        # Prepend the full heading hierarchy (not just the immediate heading).
        if heading_stack:
            content = f"{chr(10).join(heading_stack)}\n\n{body}"
        else:
            content = body
        # Per-section counter (section-relative ordinal for anchor_key — D-11).
        ordinal = section_ordinal_counter.get(section_path, 0)
        section_ordinal_counter[section_path] = ordinal + 1
        chunks.append(
            {
                "section_path": section_path,
                "content": content,
                "page_start": None,
                "page_end": None,
                "section_ordinal": ordinal,
                "token_count": count_tokens(content),
            }
        )
        current_lines.clear()

    def maybe_split_oversized() -> None:
        """If current buffer exceeds token_cap, split at the last sentence boundary."""
        content = "".join(current_lines)
        if count_tokens(content) <= token_cap:
            return
        # Simple sentence split: split on ". " or ".\n".
        sentences = re.split(r"(?<=\.)\s+", content)
        buf: list[str] = []
        for sentence in sentences:
            buf.append(sentence)
            joined = " ".join(buf)
            if count_tokens(joined) > token_cap and len(buf) > 1:
                # Emit everything except the last sentence.
                current_lines.clear()
                current_lines.append(" ".join(buf[:-1]))
                flush_chunk()
                buf = [sentence]
        # Put remaining sentences back.
        current_lines.clear()
        if buf:
            current_lines.append(" ".join(buf))

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            # Flush current buffer before starting a new section.
            maybe_split_oversized()
            flush_chunk()
            level = len(m.group(1))  # number of # characters
            title = m.group(2).strip()
            # Truncate stack to the appropriate depth.
            heading_stack = [h for h in heading_stack if h.count("#") < level]
            heading_stack.append(f"{'#' * level} {title}")
        else:
            current_lines.append(line)
            maybe_split_oversized()

    # Flush any remaining content.
    maybe_split_oversized()
    flush_chunk()

    return chunks


def parse_text(path: Path, source_path_str: str, token_cap: int = TOKEN_CAP) -> list[dict]:
    """Chunk a plain .txt file on blank-line block boundaries or date headers (D-07).

    Returns a list of chunk dicts with the same keys as parse_markdown.
    """
    text = path.read_text(encoding="utf-8")
    return _chunk_text(text, source_path_str, token_cap)


def _chunk_text(text: str, source_path_str: str, token_cap: int = TOKEN_CAP) -> list[dict]:
    """Internal: split plain text into chunks on blank lines."""
    # Split on blank lines (two or more consecutive newlines).
    blocks = re.split(r"\n{2,}", text.strip())
    chunks: list[dict] = []
    section_ordinal_counter: dict[str, int] = {}

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Detect a date header at the start of the block for a descriptive section_path.
        first_line = block.splitlines()[0] if block.splitlines() else block
        date_m = _DATE_RE.match(first_line)
        if date_m:
            section_path = date_m.group(1)
        else:
            # Use block index as section path (zero-indexed).
            block_idx = len(chunks)
            section_path = f"Block {block_idx}"

        # Split oversized blocks at sentence boundaries.
        sub_blocks = _split_at_token_cap(block, token_cap)
        for sub in sub_blocks:
            ordinal = section_ordinal_counter.get(section_path, 0)
            section_ordinal_counter[section_path] = ordinal + 1
            chunks.append(
                {
                    "section_path": section_path,
                    "content": sub,
                    "page_start": None,
                    "page_end": None,
                    "section_ordinal": ordinal,
                    "token_count": count_tokens(sub),
                }
            )

    return chunks


def _split_at_token_cap(text: str, token_cap: int) -> list[str]:
    """Split text at sentence boundaries to respect the token cap. Returns at least one block."""
    if count_tokens(text) <= token_cap:
        return [text]
    sentences = re.split(r"(?<=\.)\s+", text)
    result: list[str] = []
    buf: list[str] = []
    for sentence in sentences:
        buf.append(sentence)
        joined = " ".join(buf)
        if count_tokens(joined) > token_cap and len(buf) > 1:
            result.append(" ".join(buf[:-1]))
            buf = [sentence]
    if buf:
        result.append(" ".join(buf))
    return result if result else [text]


# ---------------------------------------------------------------------------
# Maintenance entry parser — shared constrained front-matter format
# (inverse of Plan 02's _render_front_matter quoting rule)
# ---------------------------------------------------------------------------

# Matches a leading `---` front-matter block (opening `---`, content, closing `---`).
_FM_BLOCK_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)


def _unquote_value(raw: str) -> str:
    """Strip surrounding double-quotes and unescape \\\" -> \" per the constrained format.

    Plan 02's _render_front_matter quotes a value only when it contains `:`, `#`,
    a leading dash, or a double-quote. Values with only `&` or `/` are left unquoted.
    This inverse rule: if the value is wrapped in double quotes, strip them and
    unescape; otherwise return the raw stripped value unchanged.
    """
    s = raw.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        inner = s[1:-1]
        return inner.replace('\\"', '"')
    return s


def _parse_front_matter(text: str) -> "tuple[dict, str]":
    """Parse a YAML-like front-matter block from the start of *text*.

    Returns ``(fields_dict, body_text)`` where:

    - ``fields_dict`` contains the typed structured fields.
    - ``body_text`` is everything after the closing ``---`` line, with a
      single leading blank line stripped.

    Parsing rules (stdlib ``re`` only — no PyYAML per Anti-Pattern):
    - Simple ``key: value`` lines produce string values (unquoted per the
      inverse of Plan 02's constrained quoting rule).
    - Empty value → ``None``.
    - A ``parts:`` key followed by indented ``  - item`` lines (or unindented
      ``- item`` lines as written by the fixture) produces a Python list.
    - ``cost_amount`` → ``float``; ``cost_currency`` → ``str`` (default "NZD").
    """
    m = _FM_BLOCK_RE.match(text)
    if not m:
        # No front-matter — body is the whole text.
        return {}, text

    fm_text = m.group(1)
    body_raw = text[m.end():]
    # Strip a single leading blank line from the body.
    body = body_raw.lstrip("\r\n").rstrip()

    fields: dict = {}
    lines = fm_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip fully blank lines inside front-matter.
        if not line.strip():
            i += 1
            continue

        # A list key: starts the block, e.g. "parts:".
        list_key_m = re.match(r"^(\w+):\s*$", line)
        if list_key_m:
            key = list_key_m.group(1)
            items: list[str] = []
            i += 1
            # Consume indented or unindented list items (- item).
            while i < len(lines):
                item_m = re.match(r"^[ \t]*-[ \t]+(.*)", lines[i])
                if item_m:
                    items.append(_unquote_value(item_m.group(1)))
                    i += 1
                else:
                    break
            fields[key] = items
            continue

        # Simple key: value line.
        kv_m = re.match(r"^(\w+):\s*(.*)", line)
        if kv_m:
            key = kv_m.group(1)
            raw_val = kv_m.group(2).strip()
            # Bareword YAML null / empty-flow-list map to None / [] so omitted
            # fields do not round-trip as the literal strings "null" / "[]"
            # (WR-01). A genuine value equal to these is quoted by the writer,
            # so it arrives here wrapped in double-quotes and is unquoted below.
            if raw_val == "" or raw_val == "null":
                fields[key] = None
            elif raw_val == "[]":
                fields[key] = []
            else:
                fields[key] = _unquote_value(raw_val)
        i += 1

    # Type coercions.
    if "cost_amount" in fields and fields["cost_amount"] is not None:
        try:
            fields["cost_amount"] = float(fields["cost_amount"])
        except (ValueError, TypeError):
            fields["cost_amount"] = None

    # Default cost_currency to "NZD" when cost_amount present but currency absent.
    if fields.get("cost_amount") is not None and "cost_currency" not in fields:
        fields["cost_currency"] = "NZD"

    return fields, body


def parse_maintenance_entry(path: Path, source_path_str: str) -> list[dict]:
    """Parse a maintenance-entry markdown file into a single chunk dict.

    The chunk carries the same keys as ``parse_markdown`` (section_path, content,
    page_start, page_end, section_ordinal, token_count) PLUS a ``metadata`` key
    containing the seven structured fields extracted from the front-matter:
    {date, system, system_detail, parts, cost_amount, cost_currency, vendor}.

    The chunk content is the body text (front-matter stripped) — the front-matter
    fields live exclusively in ``metadata``.  The caller (``ingest_file``) attaches
    the ``embedding`` key; it is NOT set here.
    """
    text = path.read_text(encoding="utf-8")
    fields, body = _parse_front_matter(text)

    metadata = {
        "date": fields.get("date"),
        "system": fields.get("system"),
        "system_detail": fields.get("system_detail"),
        "parts": fields.get("parts", []),
        "cost_amount": fields.get("cost_amount"),
        "cost_currency": fields.get("cost_currency"),
        "vendor": fields.get("vendor"),
    }

    return [
        {
            "section_path": "Maintenance Log",
            "content": body,
            "page_start": None,
            "page_end": None,
            "section_ordinal": 0,
            "token_count": count_tokens(body),
            "metadata": metadata,
        }
    ]


def _coerce_optional_int(val: object) -> "int | None":
    """Coerce a front-matter scalar to int, treating null/empty/None as None."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "null", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_deviation_entry(path: Path, source_path_str: str) -> list[dict]:
    """Parse a factory-deviation markdown file (data/deviations/DEV-{id}.md) into one chunk.

    Mirrors ``parse_maintenance_entry``: the chunk carries the standard keys PLUS a
    ``metadata`` key with the cross-reference fields the web highlight + retrieval
    rely on: ``{deviation_id, zone_id}`` (parsed from the front-matter). Preserving
    these on re-ingest is what stops a broad ``l44 ingest data/`` from downgrading
    a deviation to plain markdown and dropping the deviation_id (which would break the
    blue zone highlight). The chunk content is the body text (front-matter stripped).
    The caller (``ingest_file``) attaches the ``embedding`` key; it is NOT set here.
    """
    text = path.read_text(encoding="utf-8")
    fields, body = _parse_front_matter(text)

    metadata = {
        "deviation_id": _coerce_optional_int(fields.get("deviation_id")),
        "zone_id": _coerce_optional_int(fields.get("zone_id")),
    }

    return [
        {
            "section_path": "Factory Deviation",
            "content": body,
            "page_start": None,
            "page_end": None,
            "section_ordinal": 0,
            "token_count": count_tokens(body),
            "metadata": metadata,
        }
    ]
