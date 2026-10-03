"""Immutable table files and catalogs; the current pointer is the only mutable data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from engine.data.releases import atomic_json, digest
from engine.data.shared import inside, json_bytes, sha256, write_new
from engine.data.verification import observe, verified
from engine.tables.registry import TableSpec, table_name


def content_hash(value: dict) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_spec(path: Path) -> dict:
    return {"sha256": sha256(path), "size": path.stat().st_size}


def current(data_dir: Path) -> dict:
    path = data_dir / "tables/current.json"
    observe(path)
    if not path.exists():
        return {"schema_version": 1, "tables": {}}
    return load_catalog(data_dir, json.loads(path.read_text()))


def load_catalog(data_dir: Path, ref: dict) -> dict:
    return verified(
        ("table-catalog", str(data_dir.resolve()), ref["path"], ref["sha256"]),
        lambda: _load_catalog(data_dir, ref),
    )


def _load_catalog(data_dir: Path, ref: dict) -> dict:
    path = inside(data_dir, ref["path"])
    if digest(path) != ref["sha256"]:
        raise ValueError("Table catalog checksum mismatch")
    result = json.loads(path.read_text())
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported table catalog")
    for name in result["tables"]:
        table_name(name)
    validate_dependencies(data_dir, result)
    return result


def load_table(data_dir: Path, name: str, ref: dict) -> tuple[Path, dict]:
    return verified(
        ("table", str(data_dir.resolve()), name, ref["path"], ref["sha256"]),
        lambda: _load_table(data_dir, name, ref),
    )


def _load_table(data_dir: Path, name: str, ref: dict) -> tuple[Path, dict]:
    table_name(name)
    path = inside(data_dir, ref["path"])
    if digest(path) != ref["sha256"]:
        raise ValueError(f"Table manifest changed: {name}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("name") != name:
        raise ValueError(f"Invalid table manifest: {name}")
    for filename, spec in manifest["files"].items():
        artifact = inside(path.parent, filename)
        observe(artifact)
        if (
            not artifact.is_file()
            or artifact.stat().st_size != spec["size"]
            or digest(artifact) != spec["sha256"]
        ):
            raise ValueError(f"Missing/corrupt table artifact: {name}/{filename}")
    if "data.parquet" not in manifest["files"]:
        raise ValueError(f"Missing table data: {name}")
    return path.parent, manifest


def publish_catalog(data_dir: Path, catalog: dict) -> dict:
    validate_dependencies(data_dir, catalog)
    digest = content_hash(catalog)
    path = f"tables/catalogs/{digest}.json"
    write_new(inside(data_dir, path), json_bytes(catalog))
    ref = {"path": path, "sha256": digest}
    atomic_json(data_dir / "tables/current.json", ref)
    return ref


def validate_dependencies(data_dir: Path, catalog: dict) -> None:
    for name, ref in catalog["tables"].items():
        _, manifest = load_table(data_dir, name, ref)
        for dep, expected in manifest["dependencies"].items():
            if catalog["tables"].get(dep) != expected:
                raise ValueError(f"Mixed table dependencies: {name} requires another {dep} version")


def validate_frame(frame: pl.DataFrame, spec: TableSpec) -> None:
    missing = set(spec.primary_key) | set(spec.required_columns)
    if missing - set(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(missing - set(frame.columns))}")
    if frame.height < spec.min_rows:
        raise ValueError(f"Expected at least {spec.min_rows} rows; got {frame.height}")
    keys = frame.select(spec.primary_key)
    if any(keys.null_count().row(0)) or keys.n_unique() != frame.height:
        raise ValueError(f"Null or duplicate primary key: {spec.primary_key}")
    for column, expected_type in spec.required_columns.items():
        if str(frame.schema[column]) != expected_type:
            raise ValueError(
                f"Unexpected type for {column}: {frame.schema[column]} != {expected_type}"
            )
    for column, dtype in frame.schema.items():
        if dtype.is_float() and not frame[column].drop_nulls().is_finite().all():
            raise ValueError(f"Nonfinite values in {column}; use null for undefined metrics")
