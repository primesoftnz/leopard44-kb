"""Hand-rolled migration runner for the Leopard 44 KB store.

Migrations live in schema/NNN_*.sql with a 3-digit zero-padded prefix;
they are applied in numeric order and recorded in the schema_version table.
apply_migrations is idempotent — calling it on an up-to-date database is a no-op.

To add a new migration in a future phase, create schema/002_<description>.sql.
The runner picks it up automatically on next open_db() call. Migrations run
inside an implicit transaction (with conn:) so a failure leaves the DB clean.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_VERSION_RE = re.compile(r"^(\d{3})_.*\.sql$")

# parents[2] = repo root for src/ layout: src/leopard44_kb/schema.py
#   parents[0] = src/leopard44_kb/
#   parents[1] = src/
#   parents[2] = repo root
# Then we append 'schema/' to reach the SQL migration files.
_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"


def _current_version(conn: sqlite3.Connection) -> int:
    """Return the highest applied migration version, or 0 if none applied yet."""
    try:
        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else 0
    except sqlite3.OperationalError:
        # First-ever open — schema_version table doesn't exist yet.
        return 0


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply unrun migrations from schema/NNN_*.sql files; return final version.

    The function is idempotent: calling it on an already-current database
    skips every file (version <= current) and returns the existing version
    without inserting duplicate rows into schema_version.

    The 001_init.sql migration already contains:
        INSERT INTO schema_version(version) VALUES (1);
    so the runner does NOT insert a version row for migration 001.
    For future migrations (002+) that omit the INSERT, the runner adds one
    after executing the file.
    """
    files = sorted(
        (p for p in _SCHEMA_DIR.iterdir() if _VERSION_RE.match(p.name)),
        key=lambda p: int(_VERSION_RE.match(p.name).group(1)),
    )
    current = _current_version(conn)
    for f in files:
        version = int(_VERSION_RE.match(f.name).group(1))
        if version <= current:
            continue
        with conn:  # implicit transaction — failure leaves DB unchanged
            conn.executescript(f.read_text(encoding="utf-8"))
            # Robustness refinement: if the .sql file did NOT insert the
            # version row (future migrations may omit it), insert it here.
            # If the .sql already inserted it (001_init.sql does), this check
            # prevents a duplicate PRIMARY KEY error.
            post_version = _current_version(conn)
            if post_version < version:
                conn.execute(
                    "INSERT INTO schema_version(version) VALUES (?)", (version,)
                )
    return _current_version(conn)
