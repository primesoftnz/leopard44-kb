"""Read-side query backing the `l44 sources --layer ...` CLI subcommand (SCHEMA-02)."""

from __future__ import annotations

from typing import Iterable, Mapping

from leopard44_kb.schema import apply_migrations
from leopard44_kb.store import open_db


def list_sources_for_layer(layer: str) -> Iterable[Mapping]:
    """Yield sqlite3.Row objects for sources matching the given layer.

    Columns yielded: id, source_type, path, title, ingested_at.

    Caller is responsible for validating layer against LAYERS (from leopard44_kb.__init__).
    This function trusts its caller — the CLI performs typer.BadParameter validation
    before calling here. An unknown layer simply returns zero rows (the parameterised
    WHERE clause binds it safely).

    Connection lifecycle note: This function opens a connection and yields rows.
    The connection remains open until the caller exhausts the iterator — a deliberate
    Phase 1 simplification acceptable for the small dataset sizes expected. If Phase 3+
    introduces large result sets, refactor to a context-managed generator with explicit
    connection teardown.

    Args:
        layer: One of 'shared', 'vessel', 'community' (validated by CLI boundary).

    Yields:
        sqlite3.Row mappings with keys: id, source_type, path, title, ingested_at.
    """
    conn = open_db()
    apply_migrations(conn)
    cur = conn.execute(
        "SELECT id, source_type, path, title, ingested_at FROM sources WHERE layer = ? ORDER BY id",
        (layer,),
    )
    yield from cur
