"""Connection factory for the Leopard 44 KB sqlite-vec store.

All connections in the codebase MUST go through open_db() — see Pitfall 4
in 01-RESEARCH.md. Centralising connection creation ensures that PRAGMA
foreign_keys = ON is set on every connection (it defaults to OFF in SQLite).
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import sqlite_vec

# Default store path follows XDG conventions on Linux/macOS.
# Override with the L44_DB environment variable, e.g.:
#   L44_DB=/tmp/test.db l44 sources --layer shared
#
# NOTE: This constant reflects the env at import time. Always call open_db()
# with no arguments (or pass a path explicitly) — do NOT use this constant
# directly in production code, as the env may have been patched after import
# (e.g., in tests via monkeypatch.setenv).
DEFAULT_DB_PATH = Path(
    os.environ.get("L44_DB")
    or (Path.home() / ".local" / "share" / "leopard44-kb" / "store.db")
)


def open_db(path: Path | None = None) -> sqlite3.Connection:
    """Return a connection with sqlite-vec loaded, FK enforcement on, WAL mode.

    Args:
        path: Path to the SQLite database file. Pass None (default) to resolve
              from L44_DB env var at call time, falling back to the XDG
              default (~/.local/share/leopard44-kb/store.db). Reading the env at
              call time (not import time) ensures monkeypatch.setenv works in
              tests without module reload.
              Pass a :memory: path (as a Path object or str) for in-memory DBs;
              however, the conftest fixture uses sqlite3.connect(':memory:')
              directly and applies migrations itself to avoid mkdir side effects.

    Returns:
        A sqlite3.Connection with row_factory=sqlite3.Row, sqlite-vec loaded,
        PRAGMA foreign_keys = ON, and PRAGMA journal_mode = WAL.

    Raises:
        RuntimeError: If the sqlite3 module was compiled without extension
            support (macOS system Python). See Pitfall 2 in 01-RESEARCH.md.
    """
    if path is None:
        path = Path(
            os.environ.get("L44_DB")
            or (Path.home() / ".local" / "share" / "leopard44-kb" / "store.db")
        )
    # Create parent directory on first run (handles fresh laptop installs).
    # For :memory: paths this is a no-op because the parent is an empty string
    # resolved to the cwd, but we guard with a str check.
    path_str = str(path)
    if path_str != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path_str)
    conn.row_factory = sqlite3.Row

    # Extension-load sandwich per RESEARCH.md Pattern 2 + Pitfall 2.
    # enable_load_extension(False) immediately after loading prevents
    # downstream SQL injection from loading arbitrary .so files (T-02-02).
    try:
        conn.enable_load_extension(True)
    except AttributeError as e:
        raise RuntimeError(
            "Your Python's sqlite3 module was compiled without extension support. "
            "On macOS, install Python via Homebrew (`brew install python@3.12`) or python.org; "
            "alternatively `pip install pysqlite3-binary` and we will use that instead."
        ) from e

    # Use sqlite_vec.load(conn) — not conn.load_extension("sqlite_vec") — so the
    # package's platform-detection logic finds the correct binary. See Anti-Patterns
    # in 01-RESEARCH.md.
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # PRAGMA foreign_keys is per-connection and defaults to OFF in SQLite.
    # Without this, every REFERENCES clause is silently advisory (Pitfall 4).
    conn.execute("PRAGMA foreign_keys = ON")

    # WAL mode allows concurrent readers alongside a single writer, which is
    # the common access pattern for a local laptop store. For :memory: DBs this
    # returns "memory" harmlessly.
    conn.execute("PRAGMA journal_mode = WAL")

    return conn
