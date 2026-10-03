"""Pinned local SQL sessions and optional disposable native DuckDB snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import duckdb

from engine.data.releases import digest
from engine.data.shared import inside, json_bytes, lock, write_new
from engine.tables.storage import content_hash, current, file_spec, load_table


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def attach_tables(connection, paths: dict[str, Path]) -> None:
    connection.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    for name, path in paths.items():
        # Names are validated by registry/catalog loading; paths are SQL literals.
        connection.execute(
            f'CREATE VIEW analytics."{name}" AS SELECT * FROM read_parquet({sql_string(str(path))})'
        )


@contextmanager
def connect(
    data_dir: Path | None = None,
    *,
    catalog: dict | None = None,
    native: bool = True,
    memory_limit: str = "1GB",
    threads: int = 4,
):
    """One connection per caller/thread, pinned until close. Never fetches from GCS.

    SQL is trusted local analyst code, not a sandbox for public/user-supplied SQL.
    Check inputs before and after the session to detect mutation of pinned artifacts.
    """
    if data_dir is None:
        from engine.config.settings import get_settings

        data_dir = get_settings().data_dir
    data_dir = data_dir.resolve()
    catalog = current(data_dir) if catalog is None else catalog
    paths = {}
    for name, ref in catalog["tables"].items():
        root, _ = load_table(data_dir, name, ref)
        paths[name] = root / "data.parquet"
    config: dict[str, str | bool | int | float | list[str]] = {
        "memory_limit": memory_limit,
        "threads": threads,
        "TimeZone": "UTC",
    }
    if native:
        path = materialize(data_dir, catalog, config=config)
        connection = duckdb.connect(str(path), read_only=True, config=config)
    else:
        connection = duckdb.connect(config=config)
        attach_tables(connection, paths)
    try:
        yield connection
        for name, ref in catalog["tables"].items():
            load_table(data_dir, name, ref)
    finally:
        connection.close()


def materialize(data_dir: Path, catalog: dict | None = None, *, config=None) -> Path:
    """Build a closed native DB under a unique catalog+engine-version cache key."""
    catalog = current(data_dir) if catalog is None else catalog
    key = content_hash({"catalog": catalog, "duckdb": duckdb.__version__})
    cache = data_dir / "cache/tables"
    target = cache / f"{key}.duckdb"
    sidecar = target.with_suffix(".json")
    with lock(cache):
        sources = {
            name: load_table(data_dir, name, ref)[0] / "data.parquet"
            for name, ref in catalog["tables"].items()
        }
        if target.exists() and sidecar.exists():
            spec = json.loads(sidecar.read_text())
            if target.stat().st_size == spec["size"] and digest(target) == spec["sha256"]:
                return target
        fd, filename = tempfile.mkstemp(suffix=".duckdb", dir=cache)
        os.close(fd)
        temporary = Path(filename)
        temporary.unlink()  # DuckDB requires a nonexistent file or a valid database.
        try:
            build_config = {
                "memory_limit": "1GB",
                **(config or {}),
                # Wide feature tables allocate a column buffer per writer thread.
                # Serialize cache ingestion; query sessions retain their own setting.
                "threads": 1,
                "preserve_insertion_order": False,
            }
            with duckdb.connect(str(temporary), config=build_config) as connection:
                connection.execute("CREATE SCHEMA analytics")
                for name, path in sources.items():
                    connection.execute(
                        f'CREATE TABLE analytics."{name}" AS '
                        f"SELECT * FROM read_parquet({sql_string(str(path))})"
                    )
                    # Flush each table instead of retaining an entire wide source
                    # catalog's uncheckpointed segments within the memory budget.
                    connection.execute("CHECKPOINT")
            for name, ref in catalog["tables"].items():
                load_table(data_dir, name, ref)
            spec = file_spec(temporary)
            os.replace(temporary, target)
            # Cache sidecars are disposable, unlike authoritative manifests.
            sidecar.unlink(missing_ok=True)
            write_new(sidecar, json_bytes(spec))
        finally:
            temporary.unlink(missing_ok=True)
            inside(cache, temporary.name + ".wal").unlink(missing_ok=True)
    return target
