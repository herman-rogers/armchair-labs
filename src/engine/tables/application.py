"""Request-scoped application reads from the published DuckDB table catalog.

The HTTP boundary enables this for analysis/league routes. Research and builders
keep their existing artifact readers. Missing/stale application tables fail closed;
an application request never falls back to reading product Parquet directly.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from engine.data.releases import digest
from engine.data.verification import observe
from engine.tables.products import PAYLOAD, ROW, selection
from engine.tables.query import connect
from engine.tables.storage import current, load_table

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
        if "catalog" in state and selection(state["data"]) != state["catalog"]["application"]:
            raise ValueError("Application selection changed during query")
    finally:
        _scope.reset(token)


def catalog_state() -> dict:
    state = _scope.get()
    if state is None:
        raise RuntimeError("Application query scope is not active")
    if "catalog" not in state:
        catalog = current(state["data"])
        if not catalog.get("application") or catalog["application"] != selection(state["data"]):
            raise ValueError("Application tables are missing or stale; run engine tables refresh")
        sources = {}
        for name, ref in catalog["tables"].items():
            _, table = load_table(state["data"], name, ref)
            parameters = table["recipe"]["parameters"]
            if "artifact" in parameters:
                sources[parameters["artifact"]] = (name, table, parameters["sha256"])
            elif "source_table" in table["inputs"]:
                sources[table["inputs"]["source_table"]["sha256"]] = (
                    name,
                    table,
                    table["inputs"]["source_table"]["sha256"],
                )
        state.update(catalog=catalog, sources=sources)
    return state


def binding(path: Path):
    state = catalog_state()
    observe(path)
    path = path.resolve()
    sha = digest(path)
    key = path.relative_to(state["data"]).as_posix()
    bound = state["sources"].get(key) or state["sources"].get(sha)
    if not bound or bound[2] != sha:
        raise ValueError(f"Artifact is not in the selected application tables: {key}")
    return state, bound[0], bound[1]


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_frame(path: Path, *, columns: list[str] | None = None, player_id: str | None = None):
    if not active():
        from engine.data.frames import read_frame as research_frame

        return research_frame(path, columns=columns, player_id=player_id)
    state, name, table = binding(path)
    available = [c for c in table["columns"] if c != ROW]
    selected = available if columns is None else columns
    if set(selected) - set(available):
        raise ValueError(f"Unknown columns in {name}: {set(selected) - set(available)}")
    sql = f"SELECT {', '.join(map(quote, selected))} FROM analytics.{quote(name)}"
    parameters = []
    if player_id is not None:
        sql += " WHERE player_id = ?"
        parameters.append(player_id)
    if ROW in table["columns"]:
        sql += f" ORDER BY {quote(ROW)}"
    with connect(state["data"], catalog=state["catalog"]) as db:
        return db.execute(sql, parameters).pl()


def read_json(path: Path):
    if not active():
        from engine.data.verification import read_json as research_json

        return research_json(path)
    state, name, _ = binding(path)
    with connect(state["data"], catalog=state["catalog"]) as db:
        rows = db.execute(
            f"SELECT {quote(PAYLOAD)} FROM analytics.{quote(name)} ORDER BY {quote(ROW)}"
        ).fetchall()
    values = [json.loads(row[0]) for row in rows]
    # Shape is part of the bound recipe, not inferred from row count (one-item
    # arrays and singleton objects must remain distinguishable).
    _, table = load_table(state["data"], name, state["catalog"]["tables"][name])
    return values if table["recipe"]["parameters"]["json_array"] else values[0]
