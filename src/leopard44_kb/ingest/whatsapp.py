"""WhatsApp .txt/.zip chat export parser and time-gap chunker (INGEST-01, D-03..D-06).

LAYER GUIDANCE (D-15 / Codex HIGH 13-03 privacy)
-------------------------------------------------
`l44 ingest <export.zip>` defaults to ``--layer vessel`` (private-safe).

**ONLY** the PUBLIC owners'-group WhatsApp export should be ingested with
``--layer community``.  Your private boat WhatsApp exports contain personal
maintenance history and must stay in the ``vessel`` scope so they are NEVER
exposed in the community layer.

Example — public owners' group only:
    l44 ingest owners-group-export.zip --layer community

Private boat WhatsApp (use the default --layer vessel):
    l44 ingest my-boat-chat.zip              # layer=vessel (default, private)

After ingesting the public export as vessel (mistake recovery):
    l44 migrate relayer-whatsapp --source <id>   # explicit owner re-layer
"""
from __future__ import annotations

import logging
import re
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

# Attempt to import tiktoken; fall back to a deterministic word-count heuristic
# if the BPE cache is unavailable offline (Pitfall 5 in 02-RESEARCH.md).
try:
    import tiktoken as _tiktoken

    _ENC = _tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Count cl100k_base tokens."""
        return len(_ENC.encode(text))

except Exception:

    def count_tokens(text: str) -> int:  # type: ignore[misc]
        """Rough token estimate used when tiktoken cache is unavailable."""
        words = text.split()
        punct = len(re.findall(r"[^\w\s]", text))
        return len(words) + punct // 2


_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

GAP_MINUTES = 30
TOKEN_CAP = 200
MEDIA_MARKER = "[media]"

# System message prefixes — lines whose payload starts with any of these are dropped (D-05).
SYSTEM_PREFIXES: tuple[str, ...] = (
    "Messages and calls are end-to-end encrypted",
    "joined using this group",
    "left",
    "added",
    "removed",
    "changed the group",
    "created group",
    "changed their phone number",
    "Your security code with",
    "This message was deleted",
    "You deleted this message",
    "Waiting for this message",
    "null",
)

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
# Android format: DD/MM/YYYY, HH:MM - Sender: message
# Supports:
#   date separators: / . -
#   2-digit or 4-digit year
#   HH:MM or HH:MM:SS
#   optional AM/PM with optional preceding space (ASCII or Unicode)
# U+202F (narrow no-break space) and U+00A0 (no-break space) are normalised
# to regular ASCII space in the pre-pass before these regexes are applied.
ANDROID_RE = re.compile(
    r"^(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})"  # date group
    r",\s*"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)"  # time group (with optional AM/PM)
    r"\s*-\s*"
    r"(.+)",  # payload (sender: body OR system text)
)

# iOS format: [DD/MM/YYYY, HH:MM:SS AM/PM] Sender: message
IOS_RE = re.compile(
    r"^\[(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})"  # date group (inside brackets)
    r",\s*"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)"  # time group
    r"\]\s*"
    r"(.+)",  # payload
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_unicode_spaces(line: str) -> str:
    """Replace U+202F (narrow-no-break-space) and U+00A0 (no-break-space) with ASCII space.

    This pre-pass keeps the timestamp regexes simple — they only need to handle
    a normal space before AM/PM rather than multiple Unicode space variants.
    """
    return line.replace(" ", " ").replace(" ", " ")


def _parse_timestamp(date_str: str, time_str: str) -> datetime | None:
    """Parse a timestamp from the date and time strings extracted by a regex.

    Tries multiple format strings to handle: separators (/ . -), 2/4-digit year,
    HH:MM / HH:MM:SS, and optional AM/PM.  Returns None if no format matches.
    """
    time_str = time_str.strip()
    date_str = date_str.strip()

    # Normalise date separator to /
    date_norm = re.sub(r"[.\-]", "/", date_str)

    # Determine if 2-digit year (two chars after last separator)
    parts = date_norm.split("/")
    if len(parts) == 3 and len(parts[2]) == 2:
        year_fmt = "%y"
    else:
        year_fmt = "%Y"

    # Build candidate time formats
    has_ampm = bool(re.search(r"[APap][Mm]\s*$", time_str))
    if has_ampm:
        # Normalise AM/PM to uppercase and strip extra whitespace
        time_str = re.sub(r"\s+([APap][Mm])\s*$", lambda m: " " + m.group(1).upper(), time_str)
        time_fmts = ["%I:%M:%S %p", "%I:%M %p"]
    else:
        time_fmts = ["%H:%M:%S", "%H:%M"]

    for time_fmt in time_fmts:
        for day_first in [True, False]:
            if day_first:
                date_fmt = f"%d/%m/{year_fmt}"
            else:
                date_fmt = f"%m/%d/{year_fmt}"
            fmt = f"{date_fmt} {time_fmt}"
            try:
                return datetime.strptime(f"{date_norm} {time_str}", fmt)
            except ValueError:
                continue

    return None


def _detect_date_format(date_str: str, time_str: str) -> str:
    """Return a human-readable description of the detected date+time format."""
    time_str = time_str.strip()
    date_str = date_str.strip()

    # Detect date separator
    if "." in date_str:
        sep = "."
    elif "-" in date_str:
        sep = "-"
    else:
        sep = "/"

    # Detect day-first vs month-first (if any field > 12 → that field is the day)
    parts = re.split(r"[/.\-]", date_str)
    if len(parts) == 3:
        try:
            first = int(parts[0])
            second = int(parts[1])
            if first > 12:
                day_order = "DD"
                month_order = "MM"
            elif second > 12:
                day_order = "MM"
                month_order = "DD"
            else:
                day_order = "DD"
                month_order = "MM"
        except ValueError:
            day_order = "DD"
            month_order = "MM"
    else:
        day_order = "DD"
        month_order = "MM"

    year_part = parts[2] if len(parts) == 3 else "YYYY"
    year_str = "YY" if len(year_part) == 2 else "YYYY"

    date_label = f"{day_order}{sep}{month_order}{sep}{year_str}"

    # Detect time format
    has_ampm = bool(re.search(r"[APap][Mm]", time_str))
    has_seconds = bool(re.search(r":\d{2}:\d{2}", time_str))
    if has_ampm:
        time_label = f"h:MM{'SS ' if has_seconds else ' '}AM/PM"
    else:
        time_label = f"HH:MM{'SS' if has_seconds else ''} (24h)"

    return f"{date_label} {time_label}"


class _ExtractionResult(NamedTuple):
    """Result of extract_chat_txt — holds both the file path and the temp dir handle."""

    path: Path
    tmpdir: tempfile.TemporaryDirectory  # type: ignore[type-arg]


def extract_chat_txt(zip_path: Path) -> _ExtractionResult:
    """SAFELY extract _chat.txt from a WhatsApp .zip to a TemporaryDirectory.

    Security contract (T-02-10):
    - Inspects namelist() and extracts only the single target entry (no bulk extract).
    - Selects ONLY the entry whose basename is exactly '_chat.txt'.
    - REJECTS entries whose name is absolute or contains a '..' path component.
    - Extracts via zf.open() → writes to a known-safe path inside the temp dir.

    Caller contract:
    - The returned _ExtractionResult.tmpdir handle must be kept alive until parsing
      completes.  The caller (ingest_file in __init__.py) is responsible for cleanup.
    - The ORIGINAL zip_path is used as the source path for anchor_key and content_hash —
      NOT the extracted _chat.txt path.  The caller must pass zip_path to validate_path
      and file_content_hash, not the extracted path.

    Returns:
        _ExtractionResult(path=Path to extracted _chat.txt, tmpdir=TemporaryDirectory handle)

    Raises:
        ValueError: If no safe _chat.txt entry exists, or if the selected entry has a
                    path-traversal component (absolute path or '..').
    """
    tmpdir = tempfile.TemporaryDirectory(prefix="l44_wa_")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            entries = zf.namelist()
            # Find candidate _chat.txt entries (by basename only — glob approach rejected
            # in favour of explicit basename check to avoid partial-match surprises).
            candidates = [e for e in entries if Path(e).name == "_chat.txt"]
            if not candidates:
                raise ValueError(f"No _chat.txt entry found in {zip_path}")

            # Prefer a top-level entry if available; otherwise take the first candidate.
            top_level = [e for e in candidates if "/" not in e and "\\" not in e]
            chosen = top_level[0] if top_level else candidates[0]

            # --- PATH TRAVERSAL GUARD ---
            # Reject absolute paths (POSIX /... or Windows C:\...).
            if Path(chosen).is_absolute():
                raise ValueError(
                    f"Path traversal rejected: entry '{chosen}' is an absolute path"
                )
            # Reject any component that is '..'.
            normalised = Path(chosen)
            for part in normalised.parts:
                if part == "..":
                    raise ValueError(
                        f"Path traversal rejected: entry '{chosen}' contains '..' component"
                    )

            # Safe extraction: read entry bytes via zf.open(), write to known path.
            safe_dest = Path(tmpdir.name) / "_chat.txt"
            with zf.open(chosen) as src, safe_dest.open("wb") as dst:
                dst.write(src.read())

    except Exception:
        # Clean up temp dir on any error (caller won't get the handle).
        tmpdir.cleanup()
        raise

    return _ExtractionResult(path=safe_dest, tmpdir=tmpdir)


def looks_like_whatsapp(path: Path) -> bool:
    """Peek at the first ~20 non-empty lines; return True if a WhatsApp message line is found.

    Used by the dispatcher to route a WhatsApp-format .txt to this parser rather than
    to the plain-text parser.  Performs the Unicode normalisation pre-pass before matching.
    """
    try:
        lines_checked = 0
        with path.open(encoding="utf-8-sig", errors="replace") as fh:
            for raw_line in fh:
                stripped = raw_line.rstrip("\n")
                if not stripped:
                    continue
                normalised = _normalise_unicode_spaces(stripped)
                if ANDROID_RE.match(normalised) or IOS_RE.match(normalised):
                    return True
                lines_checked += 1
                if lines_checked >= 20:
                    break
    except (OSError, UnicodeDecodeError):
        return False
    return False


def parse_whatsapp(file_path: Path) -> list[dict]:
    """Parse a WhatsApp .txt chat export into a list of message dicts.

    Each returned dict has keys: timestamp (datetime), sender (str), body (str).

    Processing:
    - Applies a Unicode normalisation pre-pass (U+202F and U+00A0 → ASCII space).
    - Tries ANDROID_RE then IOS_RE on each line.
    - Detects and logs the date format once (via the 'leopard44_kb.ingest.whatsapp' logger).
    - Splits the payload at the FIRST ': ' to extract sender and body.
    - Drops system lines (no ': ' separator or body matches SYSTEM_PREFIXES).
    - Collapses '<Media omitted>' to MEDIA_MARKER.
    - Continuation lines (no timestamp match) are appended to the previous message body.
    - Leading continuation lines with no prior message are silently dropped.
    """
    messages: list[dict] = []
    detected_format: str | None = None

    with file_path.open(encoding="utf-8-sig", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            normalised = _normalise_unicode_spaces(line)

            # Try to match a timestamp line.
            m = ANDROID_RE.match(normalised) or IOS_RE.match(normalised)
            if m:
                date_str, time_str, payload = m.group(1), m.group(2), m.group(3)

                # Log detected format once.
                if detected_format is None:
                    detected_format = _detect_date_format(date_str, time_str)
                    _logger.debug("Detected WhatsApp date format: %s", detected_format)

                # Parse timestamp (for time-gap chunking; cosmetic errors don't break grouping).
                ts = _parse_timestamp(date_str, time_str)

                # Split payload into sender and body.
                # We split at the LAST ': ' to handle sender names that themselves
                # contain a colon (e.g. 'Greg: PM: message' → sender='Greg: PM',
                # body='message').  This is the standard robust approach because
                # WhatsApp's body text virtually never starts with a bare 'Word: '
                # sequence that should be attributed to a different sender.
                # Lines with no ': ' separator are system lines — drop them.
                if ": " not in payload:
                    continue
                last_sep = payload.rfind(": ")
                sender = payload[:last_sep]
                body = payload[last_sep + 2:]

                # Strip trailing whitespace from sender; sanity-check it has no newline.
                sender = sender.strip()
                if not sender or "\n" in sender:
                    continue  # malformed — drop

                # Check if payload body starts with a system prefix.
                is_system = any(body.startswith(prefix) for prefix in SYSTEM_PREFIXES)
                if is_system:
                    continue

                # Collapse <Media omitted>.
                if "<Media omitted>" in body:
                    body = body.replace("<Media omitted>", MEDIA_MARKER)

                messages.append({"timestamp": ts, "sender": sender, "body": body})
            else:
                # Continuation line — append to previous message's body.
                if messages and line:
                    messages[-1]["body"] += "\n" + line
                # If no previous message: silently drop the orphan line.

    return messages


# Alias for __init__.py compatibility (ingest_file dispatches to parse_whatsapp_file).
def parse_whatsapp_file(file_path: Path, source_path_str: str) -> list[dict]:
    """Parse a WhatsApp .txt export and return chunk dicts (for ingest_file dispatch).

    This is a thin wrapper around parse_whatsapp + chunk_messages that returns the
    same chunk-dict format expected by writer.store_source_and_chunks.
    """
    stem = file_path.stem
    # Strip the common "WhatsApp Chat with " prefix if present (Open Question 2).
    if stem.startswith("WhatsApp Chat with "):
        chat_name = stem[len("WhatsApp Chat with "):]
    else:
        chat_name = stem

    messages = parse_whatsapp(file_path)
    return chunk_messages(messages, chat_name, source_path_str, token_cap=TOKEN_CAP)


def chunk_messages(
    messages: list[dict],
    chat_name: str,
    source_path_str: str,
    token_cap: int = TOKEN_CAP,
) -> list[dict]:
    """Group parsed WhatsApp messages into time-gap / token-cap bounded chunks.

    Chunk boundary rules (D-03):
    - Start a new chunk when the gap from the previous message exceeds GAP_MINUTES.
    - Start a new chunk when adding the next message would push running token count over token_cap.
    - section_path = f"WhatsApp > {chat_name} > {window_start.isoformat()}" (D-04).
    - section_ordinal is section-relative (resets to 0 for each new window_start).
    - page_start / page_end = None.
    - No global 'ordinal' key is emitted.

    Args:
        messages: Output of parse_whatsapp().
        chat_name: Name of the chat (derived from filename stem by caller).
        source_path_str: String path of the source file (used in anchor_key via writer).
        token_cap: Maximum tokens per chunk (default TOKEN_CAP = 200).

    Returns:
        List of chunk dicts with keys: section_path, content, page_start, page_end,
        section_ordinal, token_count.
    """
    if not messages:
        return []

    chunks: list[dict] = []

    # Filter out messages with None timestamps (unparseable lines).
    # Messages without a timestamp cannot participate in time-gap decisions;
    # they are treated as instant (gap = 0) relative to the previous message.

    current_msgs: list[dict] = []
    window_start: datetime | None = None
    running_tokens: int = 0
    section_ordinal: int = 0  # resets to 0 on each new window/chunk

    def flush_chunk() -> None:
        nonlocal section_ordinal
        if not current_msgs:
            return
        content_lines = [f"{m['sender']}: {m['body']}" for m in current_msgs]
        content = "\n".join(content_lines)
        sp_ts = window_start.isoformat() if window_start else "unknown"
        section_path = f"WhatsApp > {chat_name} > {sp_ts}"
        chunks.append(
            {
                "section_path": section_path,
                "content": content,
                "page_start": None,
                "page_end": None,
                "section_ordinal": section_ordinal,
                "token_count": count_tokens(content),
            }
        )
        section_ordinal += 1

    for msg in messages:
        ts: datetime | None = msg.get("timestamp")
        msg_text = f"{msg['sender']}: {msg['body']}"
        msg_tokens = count_tokens(msg_text)

        if not current_msgs:
            # First message in a new chunk.
            window_start = ts
            current_msgs.append(msg)
            running_tokens = msg_tokens
        else:
            # Check time gap.
            prev_ts: datetime | None = current_msgs[-1].get("timestamp")
            time_gap_exceeded = False
            if ts is not None and prev_ts is not None:
                gap = ts - prev_ts
                time_gap_exceeded = gap > timedelta(minutes=GAP_MINUTES)

            # Check token cap.
            token_cap_exceeded = (running_tokens + msg_tokens) > token_cap

            if time_gap_exceeded or token_cap_exceeded:
                flush_chunk()
                current_msgs = [msg]
                window_start = ts
                running_tokens = msg_tokens
                # section_ordinal increments happen inside flush_chunk
            else:
                current_msgs.append(msg)
                running_tokens += msg_tokens

    # Flush any remaining messages.
    flush_chunk()

    return chunks
