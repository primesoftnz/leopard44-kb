"""Leopard 44 KB — vessel knowledge base (offline, two-layer)."""

from __future__ import annotations

__version__ = "0.1.0"

# The three valid layer values per D-01. Re-exported here so cli.py, sources.py,
# paths.py.ALLOWED_ROOTS, and the schema CHECK constraint all reference one source
# of truth. Adding a fourth layer requires updating: this constant + paths.ALLOWED_ROOTS
# + schema/001_init.sql CHECK clause + the test fixtures.
LAYERS: tuple[str, ...] = ("shared", "vessel", "community")
