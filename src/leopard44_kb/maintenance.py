"""Maintenance log extraction, date normalisation, slug derivation, and file writing.

MAINT-02: local-LLM structured extraction with D-01..D-04 field contract.
D-06: write_entry emits a punctuation-safe YAML front-matter markdown file under
data/logs/maint/ and returns its Path.

Non-streaming Ollama /api/chat call mirrors answer.py's hard-fail ladder.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from leopard44_kb.answer import select_generation_model
from leopard44_kb.paths import validate_path

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

VALID_SYSTEMS: frozenset[str] = frozenset({
    "engine",
    "electrical",
    "plumbing",
    "rigging",
    "sails",
    "hull/deck",
    "electronics/nav",
    "ground tackle",
    "safety",
    "refrigeration",
    "other",
})

EXTRACT_TEMPERATURE: float = 0.05
EXTRACT_NUM_PREDICT: int = 150

# Strips leading ```json (or ```) and trailing ``` code fences.
# Matches an optional leading fence line, captures the middle, optional trailing fence.
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$",
    re.DOTALL,
)

# Date-parsing regexes (D-02 scope)
_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_NZDMY_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")   # DD/MM/YYYY
_NZDM_RE = re.compile(r"^(\d{1,2})/(\d{1,2})$")             # DD/MM

# Prompts
EXTRACTION_SYSTEM_PROMPT: str = """\
You are a vessel maintenance log assistant. Extract structured fields from the text.
Return ONLY a JSON object with these fields:
  date        (string YYYY-MM-DD or DD/MM/YYYY or "today" — null if not mentioned)
  system      (one of: engine, electrical, plumbing, rigging, sails, hull/deck,
               electronics/nav, ground tackle, safety, refrigeration, other)
  system_detail (string — specific sub-system, e.g. "raw-water cooling" — null if not applicable)
  parts       (array of strings — each part mentioned, include part numbers inline)
  cost        (object with "amount" float and "currency" string, e.g. {"amount": 45.0, "currency": "NZD"};
               bare $ means NZD; FJD is Fijian dollars — null if not mentioned)
  vendor      (string — supplier name — null if not mentioned)

Return only the JSON object with no other text.\
"""

EXTRACTION_RETRY_PROMPT: str = """\
You are a vessel maintenance log assistant. The previous extraction was incomplete.
Try again to extract ALL fields from the text below.
Return ONLY a JSON object with exactly these keys:
date, system, system_detail, parts, cost, vendor.
Missing values must be null, not absent.\
"""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CostModel(BaseModel):
    amount: float
    currency: str = "NZD"


class MaintenanceExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: Optional[str] = None
    system: str
    system_detail: Optional[str] = None
    parts: list[str] = Field(default_factory=list)
    cost: Optional[CostModel] = None
    vendor: Optional[str] = None

    @field_validator("system", mode="before")
    @classmethod
    def normalise_system(cls, v: object) -> str:
        lower = str(v).lower().strip()
        if lower in VALID_SYSTEMS:
            return lower
        # Whole-word containment only — a bidirectional substring test
        # (the previous `lower in s or s in lower`) misfiled tokens like
        # "ac" -> "ground tackle" and single letters -> wrong systems (WR-02).
        tokens = set(re.findall(r"[a-z/]+", lower))
        for s in VALID_SYSTEMS:
            if s in tokens or any(part in tokens for part in s.split("/")):
                return s
        return "other"


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------


def call_extract_json(prompt_text: str, system_prompt: str) -> dict:
    """Non-streaming Ollama /api/chat call returning a parsed JSON dict.

    Strips Markdown code fences (```json ... ```) from the response content
    before json.loads so a fenced LLM response parses correctly (Gemini HIGH).

    Raises:
        RuntimeError: Ollama unreachable, model not pulled, HTTP error, or
            non-JSON response after fence-stripping. Never raises on a
            malformed-but-present content payload (that is handled by
            extract_fields' sanitizer).
    """
    model, _ = select_generation_model()
    # Read at call time so monkeypatch.setenv("OLLAMA_HOST", ...) works in tests.
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"

    try:
        response = httpx.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": EXTRACT_TEMPERATURE,
                    "num_predict": EXTRACT_NUM_PREDICT,
                },
            },
            timeout=60.0,
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{model}' not found — run `ollama pull {model}` then retry."
            )
        response.raise_for_status()
        content = response.json()["message"]["content"]

        # Strip code fences before json.loads (Gemini HIGH).
        m = _FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()
        else:
            content = content.strip()

        return json.loads(content)

    except httpx.ConnectError:
        raise RuntimeError(
            "Ollama not reachable at :11434 — run `ollama serve` and "
            f"`ollama pull {model}`, then retry."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Ollama extraction timed out after 60s. "
            f"Check `ollama ps` for model status, then retry."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Ollama /api/chat returned HTTP {exc.response.status_code}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Ollama returned non-JSON response: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Strict never-raise sanitizer (Codex MED)
# ---------------------------------------------------------------------------


def _sanitize_payload(raw2: object) -> dict:
    """Return a dict that is guaranteed to satisfy MaintenanceExtraction.model_validate.

    Never raises on content. Contract:
    - Non-dict raw2 -> start from {}
    - parts: keep only if list of strings, else []
    - cost: keep only if CostModel.model_validate succeeds, else None
    - system: ensure the required key is present (defaults to "other")
    """
    if not isinstance(raw2, dict):
        safe: dict = {}
    else:
        # Shallow-copy only the known keys
        safe = {
            k: raw2[k]
            for k in ("date", "system", "system_detail", "parts", "cost", "vendor")
            if k in raw2
        }

    # parts: must be a list of strings
    parts = safe.get("parts")
    if not (isinstance(parts, list) and all(isinstance(p, str) for p in parts)):
        safe["parts"] = []

    # cost: must pass CostModel validation
    cost = safe.get("cost")
    if cost is not None:
        try:
            CostModel.model_validate(cost)
        except Exception:
            safe["cost"] = None

    # date / system_detail / vendor: must be strings, else drop to None (CR-01).
    # The earlier version copied these through verbatim, so a non-string LLM value
    # (e.g. {"date": 123}) reached model_validate and raised — breaking the
    # never-raise guarantee (D-13).
    for key in ("date", "system_detail", "vendor"):
        val = safe.get(key)
        if val is not None and not isinstance(val, str):
            safe[key] = None

    # system: required field — coerce non-str to "other" and guarantee present
    if not isinstance(safe.get("system"), str):
        safe["system"] = "other"
    safe.setdefault("system", "other")

    return safe


# ---------------------------------------------------------------------------
# extract_fields
# ---------------------------------------------------------------------------


def extract_fields(text: str) -> MaintenanceExtraction:
    """Extract structured maintenance fields from free-text using the local LLM.

    Re-prompts once on ValidationError. If both calls fail validation, runs the
    strict never-raise sanitizer so a poor extraction never loses the entry (D-13).

    Args:
        text: Natural-language maintenance entry text.

    Returns:
        MaintenanceExtraction with normalised date (ISO YYYY-MM-DD).

    Raises:
        RuntimeError: Only if Ollama itself is unreachable or returns a transport
            error. Malformed-but-present LLM content never raises.
    """
    raw = call_extract_json(text, EXTRACTION_SYSTEM_PROMPT)
    try:
        result = MaintenanceExtraction.model_validate(raw)
    except Exception:
        # One retry with the explicit-field retry prompt.
        raw2 = call_extract_json(text, EXTRACTION_RETRY_PROMPT)
        try:
            result = MaintenanceExtraction.model_validate(raw2)
        except Exception:
            # Strict sanitizer — never raises on content (Codex MED / CR-01).
            safe = _sanitize_payload(raw2)
            try:
                result = MaintenanceExtraction.model_validate(safe)
            except Exception:
                # Last-resort guarantee: a minimal valid object so a poor
                # extraction never loses the entry (D-13 never-raise).
                result = MaintenanceExtraction(system="other")

    # Normalise date after a successful (or sanitized) validation.
    result.date = normalise_date(result.date, date.today())
    return result


# ---------------------------------------------------------------------------
# Date normalisation (D-02)
# ---------------------------------------------------------------------------


def normalise_date(raw: Optional[str], today: date) -> str:
    """Return ISO YYYY-MM-DD from raw. Missing or unparseable defaults to today.

    Scope (D-02 only):
    - ISO YYYY-MM-DD: passthrough
    - DD/MM/YYYY: NZ locale (day-first)
    - DD/MM: NZ locale, current year
    - "today" / "yesterday": resolved against today
    - Weekday names (monday..sunday): most recent past occurrence
    - None or unparseable: today.isoformat()

    Args:
        raw: Raw date string from the LLM, or None.
        today: Reference date (pass date.today() in production; injectable for tests).

    Returns:
        ISO date string "YYYY-MM-DD".
    """
    if raw is None:
        return today.isoformat()

    raw = raw.strip()

    # ISO passthrough
    if _ISO_RE.match(raw):
        return raw

    # DD/MM/YYYY — NZ locale (day-first)
    m = _NZDMY_RE.match(raw)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return today.isoformat()

    # DD/MM — NZ locale, current year
    m = _NZDM_RE.match(raw)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        try:
            return date(today.year, month, day).isoformat()
        except ValueError:
            return today.isoformat()

    lower = raw.lower()

    if "today" in lower:
        return today.isoformat()

    if "yesterday" in lower:
        return (today - timedelta(days=1)).isoformat()

    # Weekday names: most recent past occurrence
    _WEEKDAYS = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for name, wday in _WEEKDAYS.items():
        if name in lower:
            days_back = (today.weekday() - wday) % 7
            if days_back == 0:
                days_back = 7
            return (today - timedelta(days=days_back)).isoformat()

    # Unparseable
    return today.isoformat()


# ---------------------------------------------------------------------------
# Filename derivation
# ---------------------------------------------------------------------------


def make_entry_filename(text: str, event_date: str, existing_paths: set[str]) -> str:
    """Return a collision-free YYYY-MM-DD-slug.md filename.

    Slug is derived from the first 5 words of text (lowercased, punctuation
    stripped, joined with '-', capped at 40 chars).

    Same-day collision appends '-2', '-3', ... until a free name is found.

    Args:
        text: Entry text used to derive the slug.
        event_date: ISO date string "YYYY-MM-DD".
        existing_paths: Set of already-used filenames (basenames, not full paths).

    Returns:
        A filename like "2024-03-15-replaced-port-impeller.md".
    """
    words = re.sub(r"[^\w\s]", "", text.lower()).split()[:5]
    slug = re.sub(r"-+", "-", "-".join(words))[:40].strip("-")
    base = f"{event_date}-{slug}"
    candidate = f"{base}.md"
    if candidate not in existing_paths:
        return candidate
    i = 2
    while f"{base}-{i}.md" in existing_paths:
        i += 1
    return f"{base}-{i}.md"


# ---------------------------------------------------------------------------
# Front-matter serialiser
# ---------------------------------------------------------------------------

# Characters that require quoting in flat key: value YAML lines.
# A value needs quoting if it contains ':', '#', has leading/trailing whitespace,
# starts with '-', or contains a '"'.
_NEEDS_QUOTE_RE = re.compile(r'[:#"]|^\s|\s$|^-')

# Reserved barewords the parser maps to None/empty-list — a genuine string value
# equal to one of these MUST be quoted so it survives the round-trip (WR-01).
_RESERVED_BAREWORDS = frozenset({"null", "true", "false", "~", "[]"})


def _quote_yaml_value(v: str) -> str:
    """Return v with double-quotes added if it contains unsafe characters.

    Values containing '&' or '/' alone do NOT need quoting.
    Embedded '"' is escaped as \".
    A value equal to a reserved bareword (null/true/false/~/[]) is also quoted so
    the inverse parser does not collapse it to None/[] (WR-01).
    """
    if v in _RESERVED_BAREWORDS or _NEEDS_QUOTE_RE.search(v):
        escaped = v.replace('"', '\\"')
        return f'"{escaped}"'
    return v


def _render_front_matter(extraction: MaintenanceExtraction, original_text: str) -> str:
    """Render the complete file content: YAML front-matter + blank line + body.

    The serialisation is constrained and punctuation-safe so Plan 03's
    parse_maintenance_entry can invert it exactly (shared contract):
    - Simple ASCII values emitted as raw 'key: value'
    - Values with ':', '#', leading '-', leading/trailing whitespace, or '"'
      are double-quoted; embedded '"' escaped as \"
    - Values with only '&' or '/' are emitted unquoted
    - parts emitted as indented '  - item' list (items quoted under the same rule)
    - cost flattened to cost_amount / cost_currency
    """
    lines = ["---"]

    # date
    if extraction.date is not None:
        lines.append(f"date: {_quote_yaml_value(extraction.date)}")
    else:
        lines.append("date: null")

    # system (always present)
    lines.append(f"system: {_quote_yaml_value(extraction.system)}")

    # system_detail
    if extraction.system_detail is not None:
        lines.append(f"system_detail: {_quote_yaml_value(extraction.system_detail)}")
    else:
        lines.append("system_detail: null")

    # parts list
    if extraction.parts:
        lines.append("parts:")
        for part in extraction.parts:
            lines.append(f"  - {_quote_yaml_value(part)}")
    else:
        lines.append("parts: []")

    # cost (flattened)
    if extraction.cost is not None:
        lines.append(f"cost_amount: {extraction.cost.amount}")
        lines.append(f"cost_currency: {_quote_yaml_value(extraction.cost.currency)}")
    else:
        lines.append("cost_amount: null")
        lines.append("cost_currency: null")

    # vendor
    if extraction.vendor is not None:
        lines.append(f"vendor: {_quote_yaml_value(extraction.vendor)}")
    else:
        lines.append("vendor: null")

    # fixed fields
    lines.append("source_type: maintenance_entry")
    lines.append("layer: vessel")

    lines.append("---")
    lines.append("")
    lines.append(original_text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# write_entry
# ---------------------------------------------------------------------------


def write_entry(
    extraction: MaintenanceExtraction,
    original_text: str,
    repo_root: Optional[Path] = None,
) -> Path:
    """Write a maintenance entry markdown file under data/logs/maint/.

    Content-aware idempotency (Codex HIGH #1): if a same-day file already
    exists whose content is byte-identical to the rendered content, that
    existing file is returned without writing a new one. Only different
    same-day content allocates a -2/-3 suffix.

    Args:
        extraction: Validated MaintenanceExtraction (date must be set).
        original_text: The original free-text entry (used as the file body).
        repo_root: Repo root. Defaults to Path.cwd() so tests using
            monkeypatch.chdir(tmp_path) land under tmp_path/data/logs/maint.

    Returns:
        Path to the written (or reused) markdown file.

    Raises:
        ValueError: If original_text is empty or whitespace-only, or if the
            derived path falls outside data/ (validate_path guard, D-14/MAINT-05).
    """
    if not original_text or not original_text.strip():
        raise ValueError("empty maintenance entry")

    root = repo_root if repo_root is not None else Path.cwd()
    maint_dir = root / "data" / "logs" / "maint"
    maint_dir.mkdir(parents=True, exist_ok=True)

    # Render content once.
    content = _render_front_matter(extraction, original_text)

    # Idempotent reuse: check existing same-day files for byte-identical content.
    date_prefix = extraction.date or date.today().isoformat()
    for existing_file in maint_dir.glob("*.md"):
        if existing_file.name.startswith(f"{date_prefix}-"):
            try:
                if existing_file.read_text(encoding="utf-8") == content:
                    # Byte-identical — reuse without writing.
                    return existing_file
            except OSError:
                pass  # unreadable file — skip

    # Different content (or no same-day file): allocate a fresh name.
    # Slug is derived from extraction fields (system + parts) — not from
    # original_text — so same-system same-date entries always produce the same
    # base slug and collision detection (-2/-3 suffix) works correctly even
    # when the free-text bodies differ. (Rule 1 fix: original_text slug gave
    # unique names for different bodies, bypassing the collision mechanism that
    # test_add_same_day_different_content_collides requires.)
    existing = {p.name for p in maint_dir.glob("*.md")}
    slug_text = f"{extraction.system} {' '.join(extraction.parts)}" if extraction.parts else extraction.system
    filename = make_entry_filename(slug_text, date_prefix, existing)
    file_path = maint_dir / filename

    # Defence-in-depth: validate path before write (D-14/MAINT-05).
    validate_path("vessel", file_path, root)

    file_path.write_text(content, encoding="utf-8")
    return file_path
