"""Immutable local data releases; the latest accepted batch is the builder input.

Raw objects preserve captured bytes; cleaned table batches use Parquet. A reader
pins one manifest and verifies its dependency closure before exposing any table.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

from engine.data.frames import read_frame
from engine.data.verification import observe, read_json, verified


def now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
        raise ValueError(f"Invalid release identifier: {value}")
    return value


def protected_artifacts(root: Path, names) -> dict[str, str | None]:
    """Snapshot legacy output guards, including absence and renamed source paths."""
    paths = set(names)
    paths.update(
        name.replace("src/patron/", "src/engine/", 1)
        for name in names
        if name.startswith("src/patron/")
    )
    return {name: digest(root / name) if (root / name).exists() else None for name in sorted(paths)}


def inside(root: Path, name: str) -> Path:
    observe(root / name)
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes release: {name}")
    return path


@lru_cache(maxsize=16384)
def _digest(path: str, size: int, mtime: int, ctime: int) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(path: Path) -> str:
    observe(path)
    stat = path.stat()
    return _digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def preserve_object(data_dir: Path, source: Path, expected: str) -> str:
    """Copy exact bytes once. Never hard-link a mutable source into the raw store."""
    import shutil

    if digest(source) != expected:
        raise ValueError(f"Input changed before capture: {source}")
    destination = data_dir / "raw/objects" / expected[:2] / expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            if digest(temporary) != expected:
                raise ValueError(f"Input changed during capture: {source}")
            with suppress(FileExistsError):
                os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(destination) != expected:
        raise ValueError(f"Raw object was altered: {expected}")
    return str(destination.relative_to(data_dir))


def reference(root: Path) -> dict:
    return {"version": root.name, "manifest_sha256": digest(root / "manifest.json")}


def load_manifest(data_dir: Path, layer: str, ref: dict) -> tuple[Path, dict]:
    return verified(
        ("manifest", str(data_dir.absolute()), layer, ref["version"], ref["manifest_sha256"]),
        lambda: _load_manifest(data_dir, layer, ref),
    )


def _load_manifest(data_dir: Path, layer: str, ref: dict) -> tuple[Path, dict]:
    if layer not in {"raw", "enriched", "tables", "gold"}:
        raise ValueError(f"Invalid data layer: {layer}")
    parent = data_dir / layer / ("snapshots" if layer == "raw" else "releases")
    if layer == "tables":
        parent = data_dir / "tables/batches"
    root = parent / identifier(ref["version"])
    if digest(root / "manifest.json") != ref["manifest_sha256"]:
        raise ValueError(f"{layer} manifest changed")
    manifest = read_json(root / "manifest.json")
    if (
        manifest["version"] != ref["version"]
        or manifest["layer"] != layer
        or manifest["schema_version"] != 1
        or manifest["status"] != "accepted"
    ):
        raise ValueError(f"Incomplete {layer} release")
    files = manifest["files"]
    if not files:
        raise ValueError(f"Empty {layer} release")
    for name, expected in files.items():
        if digest(inside(data_dir if layer == "raw" else root, name)) != expected:
            raise ValueError(f"{layer} artifact changed: {name}")
    if layer != "raw":
        load_manifest(data_dir, "raw" if layer == "enriched" else "enriched", manifest["input"])
        if not manifest.get("quality_passed"):
            raise ValueError(f"{layer} quality checks failed")
        if "quality.json" not in files:
            raise ValueError(f"{layer} quality report is not hash-bound")
        checks = read_json(root / "quality.json").get("checks")
        if (
            not isinstance(checks, dict)
            or not checks
            or not all(v is True for v in checks.values())
        ):
            raise ValueError(f"{layer} quality report did not pass")
    return root, manifest


@dataclass(frozen=True)
class TableRelease:
    root: Path
    manifest: dict

    @property
    def ref(self) -> dict:
        return reference(self.root)

    def path(self, table: str) -> Path:
        spec = self.manifest["tables"][table]
        path = inside(self.root, spec["path"])
        if self.manifest["files"].get(spec["path"]) != spec["sha256"]:
            raise ValueError(f"Unbound source table: {table}")
        if digest(path) != spec["sha256"]:
            raise ValueError(f"Source table changed: {table}")
        return path

    def read(self, table: str, *, columns: list[str] | None = None) -> pl.DataFrame:
        return read_frame(self.path(table), columns=columns)


def latest_batch(data_dir: Path) -> Path:
    """The most recently accepted table batch; immutable pre-table-system releases count too."""
    candidates = []
    for parent in (data_dir / "tables/batches", data_dir / "gold/releases"):
        if not parent.is_dir():
            continue
        observe(parent)
        for root in parent.iterdir():
            manifest = root / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                document = read_json(manifest)
            except (OSError, ValueError):
                continue
            if document.get("status") == "accepted" and document.get("generated_at"):
                candidates.append((document["generated_at"], root))
    if not candidates:
        raise ValueError("No accepted source table batch exists")
    return max(candidates)[1]


def load_tables(data_dir: Path, version: str | None = None) -> TableRelease:
    """Load a verified input batch; without a version, the latest accepted one.

    Batch files are builder inputs. SQL callers use engine.tables.connect and the
    published named-table catalog, which does not require these source archives.
    """
    if version is None:
        parent = latest_batch(data_dir)
    else:
        identifier(version)
        parent = data_dir / "tables/batches" / version
        if not parent.exists():
            parent = data_dir / "gold/releases" / version
    ref = reference(parent)
    layer = "tables" if parent.parent.name == "batches" else "gold"
    root, manifest = load_manifest(data_dir, layer, ref)
    return TableRelease(root, manifest)
