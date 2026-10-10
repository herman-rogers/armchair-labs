"""Request-scoped reads from the published DuckDB table catalog.

The HTTP boundary pins the catalog once per request so every table read inside
that request sees one consistent catalog, and a catalog switch mid-request fails
closed instead of mixing versions.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from engine.tables.query import connect as _connect
from engine.tables.storage import current

_scope: ContextVar[dict | None] = ContextVar("application_query_scope", default=None)


def active() -> bool:
    return _scope.get() is not None


@contextmanager
def query_scope(data_dir: Path):
    """Pin lazily so health/configuration requests need no analytical database."""
    state: dict = {"data": data_dir.resolve()}
    token = _scope.set(state)
    try:
        yield
        if "catalog" in state and current(state["data"]) != state["catalog"]:
            raise ValueError("Table catalog changed during query")
    finally:
        _scope.reset(token)


def catalog_state() -> dict:
    state = _scope.get()
    if state is None:
        raise RuntimeError("Application query scope is not active")
    if "catalog" not in state:
        catalog = current(state["data"])
        if not catalog.get("tables"):
            raise ValueError("No query tables are published; run engine tables refresh")
        state["catalog"] = catalog
    return state


def connect():
    """A DuckDB connection to the catalog pinned by the active request."""
    state = catalog_state()
    return _connect(state["data"], catalog=state["catalog"])
