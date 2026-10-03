"""Preserve a completed study and convert its CSV tables without rerunning it."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from engine.data.shared import excluded, identifier, inside, json_bytes, valid, write_new
from engine.tables.storage import file_spec


def archive(data_dir: Path, source: Path, run: str) -> dict:
    """Add a sealed research run. Leave existing paths and bytes untouched."""
    identifier(run)
    source = source.resolve()
    destination = data_dir.resolve() / "research" / run
    if destination.exists():
        raise ValueError(f"Research run already exists: {run}")
    if not source.is_dir() or destination.is_relative_to(source):
        raise ValueError("Source must be a separate existing study directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    (data_dir / ".runtime").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".archive-", dir=data_dir / ".runtime") as temporary:
        root = Path(temporary)
        originals = {}
        tables = {}
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Research archives cannot contain symlinks: {path}")
            if not path.is_file() or excluded(path.relative_to(source)):
                continue
            relative = path.relative_to(source).as_posix()
            spec = file_spec(path)
            target = inside(root / "originals", relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if not valid(target, spec) or not valid(path, spec):
                raise ValueError(f"Research input changed while archiving: {path}")
            originals[relative] = spec
            if path.suffix == ".csv":
                frame = pl.read_csv(target, infer_schema_length=None)
                table_path = Path("tables") / Path(relative).with_suffix(".parquet")
                output = inside(root, table_path.as_posix())
                output.parent.mkdir(parents=True, exist_ok=True)
                frame.write_parquet(output, compression="zstd")
                tables[relative] = {
                    "path": table_path.as_posix(),
                    "rows": frame.height,
                    "columns": {k: str(v) for k, v in frame.schema.items()},
                }
        if not originals:
            raise ValueError("No research artifacts to archive")
        manifest = {
            "schema_version": 1,
            "kind": "research_archive",
            "run": run,
            "archived_at": datetime.now(UTC).isoformat(),
            "source_directory": str(source),
            "originals": originals,
            "tables": tables,
            "files": {
                p.relative_to(root).as_posix(): file_spec(p)
                for p in sorted(root.rglob("*"))
                if p.is_file()
            },
            "limitations": [
                "Archiving is not statistical validation or promotion.",
                "CSV conversion preserves parsed values; original text bytes remain in originals/.",
                "Archive time is not source capture time or observation cutoff.",
            ],
        }
        for name, spec in originals.items():
            if not valid(inside(source, name), spec):
                raise ValueError(f"Research input changed before archive completed: {name}")
        write_new(root / "manifest.json", json_bytes(manifest))
        root.rename(destination)
    return {"run": run, "path": str(destination), "tables": tables}


def verify_archive(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "research_archive":
        raise ValueError("Not a supported research archive")
    for name, spec in manifest["files"].items():
        if not valid(inside(root, name), spec):
            raise ValueError(f"Missing/corrupt research artifact: {name}")
    return manifest
