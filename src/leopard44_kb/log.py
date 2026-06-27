"""Read-side query backing the ``l44 log`` CLI subcommand (MAINT-04 / D-05).

Lists maintenance entries stored as vessel-layer chunks, filtered by structured
metadata fields extracted at ingest time. The caller owns the connection lifecycle
and passes an open ``conn`` — this function never calls ``open_db`` itself.

Case-insensitive substring free-text search uses ``LOWER() LIKE ... ESCAPE '\\'``
directly on ``c.content`` and ``c.metadata`` column text (not FTS5 MATCH), per the
Anti-Pattern note: avoids partial-sanitisation FTS injection with free-text input.
The user-supplied free_text value is escaped so literal ``%``, ``_``, and ``\\``
characters do not act as wildcards.
"""
from __future__ import annotations

import sqlite3
from typing import Optional


def list_maintenance_entries(
    conn: sqlite3.Connection,
    system: Optional[str] = None,
    vendor: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    free_text: Optional[str] = None,
) -> list[dict]:
    """Return maintenance entries matching the supplied filters, newest first.

    Always scoped to ``source_type='maintenance_entry'``, ``layer='vessel'``,
    and ``is_active=1`` (the Phase 3 CR-01 lesson: never surface retired chunks).

    Filter semantics:
    - ``system``    — exact match on ``json_extract(metadata, '$.system')``
    - ``vendor``    — exact match on ``json_extract(metadata, '$.vendor')``
    - ``since``     — ISO date lower bound (inclusive); text comparison works because
                      dates are stored as ``YYYY-MM-DD`` strings
    - ``until``     — ISO date upper bound (inclusive)
    - ``free_text`` — case-insensitive substring match (LIKE with ESCAPE ``\\``);
                      ``%``, ``_``, and ``\\`` in the user value are escaped before
                      binding so they are treated as literals, not SQL wildcards.
                      Searched on both ``c.content`` and the raw ``c.metadata`` JSON
                      text.

    The ``parts`` column is projected as the raw JSON array text (e.g.
    ``'["impeller p/n 22-41016"]'``); the CLI caller must ``json.loads`` it before
    rendering.

    Args:
        conn: An open ``sqlite3.Connection`` with migrations applied. NOT closed here.
        system: Optional top-level system filter (e.g. ``"engine"``).
        vendor: Optional vendor filter (e.g. ``"Burnsco"``).
        since: Optional ISO date lower bound (``"YYYY-MM-DD"``).
        until: Optional ISO date upper bound (``"YYYY-MM-DD"``).
        free_text: Optional case-insensitive substring to search in content + metadata.

    Returns:
        A list of dicts (one per matching chunk), ordered newest event-date first.
        Keys: ``title``, ``path``, ``event_date``, ``system``, ``system_detail``,
        ``vendor``, ``cost_amount``, ``cost_currency``, ``parts``.
    """
    where_clauses: list[str] = [
        "s.source_type = 'maintenance_entry'",
        "s.layer = 'vessel'",
        "c.is_active = 1",
    ]
    params: list = []

    if system is not None:
        where_clauses.append("json_extract(c.metadata, '$.system') = ?")
        params.append(system)

    if vendor is not None:
        where_clauses.append("json_extract(c.metadata, '$.vendor') = ?")
        params.append(vendor)

    if since is not None:
        where_clauses.append("json_extract(c.metadata, '$.date') >= ?")
        params.append(since)

    if until is not None:
        where_clauses.append("json_extract(c.metadata, '$.date') <= ?")
        params.append(until)

    if free_text is not None:
        # Escape SQL LIKE wildcards in the user-supplied value so literal
        # characters are not interpreted as pattern metacharacters (Codex LOW).
        # Escape order: backslash first, then %, then _.
        escaped = (
            free_text
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        pattern = f"%{escaped.lower()}%"
        where_clauses.append(
            "(LOWER(c.content) LIKE ? ESCAPE '\\' OR LOWER(c.metadata) LIKE ? ESCAPE '\\')"
        )
        # Bound twice: once for content, once for metadata.
        params.extend([pattern, pattern])

    where = " AND ".join(where_clauses)
    sql = f"""
        SELECT
            s.title,
            s.path,
            json_extract(c.metadata, '$.date')          AS event_date,
            json_extract(c.metadata, '$.system')        AS system,
            json_extract(c.metadata, '$.system_detail') AS system_detail,
            json_extract(c.metadata, '$.vendor')        AS vendor,
            json_extract(c.metadata, '$.cost_amount')   AS cost_amount,
            json_extract(c.metadata, '$.cost_currency') AS cost_currency,
            json_extract(c.metadata, '$.parts')         AS parts
        FROM chunks c
        JOIN sources s ON s.id = c.source_id
        WHERE {where}
        ORDER BY json_extract(c.metadata, '$.date') DESC
    """
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
