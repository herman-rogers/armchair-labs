"""Single-writer refreshes with dependency fingerprints and atomic publication."""

from __future__ import annotations

import importlib
import platform
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from engine.data.releases import load_tables
from engine.data.shared import inside, json_bytes, lock, sha256, write_new
from engine.tables.query import attach_tables
from engine.tables.registry import DEFAULT_REGISTRY, TableSpec, order, read_registry
from engine.tables.sources import source_path
from engine.tables.storage import (
    content_hash,
    current,
    file_spec,
    load_table,
    publish_catalog,
    validate_frame,
)


@dataclass
class BuildContext:
    data_dir: Path
    dependencies: dict[str, Path]
    parameters: dict
    observation_cutoff: dict | None = None

    def read(self, name: str) -> pl.DataFrame:
        return pl.read_parquet(self.dependencies[name])


def recipe_inputs(spec: TableSpec, registry_path: Path, data_dir: Path) -> tuple[dict, dict]:
    """Return input identities and exact recipe bytes to retain with the output."""
    files = list(spec.inputs)
    if spec.kind == "parquet" and spec.source:
        files.append(spec.source)
    inputs = {name: file_spec(inside(data_dir, name)) for name in sorted(set(files))}
    code = {}
    modules = [__name__, "engine.tables.storage", *spec.code_modules]
    if spec.kind == "source":
        modules.extend(["engine.tables.sources", "engine.data.releases"])
    if spec.builder:
        modules.append(spec.builder.split(":")[0])
    for name in sorted(set(modules)):
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError(f"Recipe module has no source file: {name}")
        code[f"{name}.py"] = Path(module.__file__).read_bytes()
    if spec.sql_file:
        code["recipe.sql"] = inside(registry_path.parent, spec.sql_file).read_bytes()
    return inputs, code


def refresh(
    data_dir: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    names: list[str] | None = None,
    due: bool = False,
    force: bool = False,
    now: datetime | None = None,
) -> dict:
    registry = read_registry(registry_path)
    if names and due:
        raise ValueError("Choose table names or --due")
    now = now or datetime.now(UTC)
    data_dir = data_dir.resolve()
    with lock(data_dir / ".runtime/table-refresh"):
        from engine.tables.products import recipes, selection

        application = selection(data_dir) if registry_path == DEFAULT_REGISTRY else {}
        product_recipes = recipes(data_dir) if application else {}
        registry.update(product_recipes)
        catalog = current(data_dir)
        manifests = {
            name: load_table(data_dir, name, ref)[1] for name, ref in catalog["tables"].items()
        }
        if due:
            targets = [
                name
                for name, spec in registry.items()
                if spec.refresh_hours is not None
                and (
                    name not in manifests
                    or now - datetime.fromisoformat(manifests[name]["built_at"])
                    >= timedelta(hours=spec.refresh_hours)
                )
            ]
        else:
            targets = list(registry) if names is None else names.copy()
        order(registry, targets)
        # Rebuilding a dependency must never leave its published dependents stale.
        selected = set(order(registry, targets))
        # Application outputs must advance with their selected source/products,
        # even when an individual analytical table was explicitly requested.
        selected.update(product_recipes)
        # A validated input batch is imported as a unit, so querying two base
        # tables cannot accidentally combine different source vintages.
        if any(registry[n].kind == "source" for n in selected) or (
            application and application != catalog.get("application")
        ):
            selected.update(n for n, spec in registry.items() if spec.kind == "source")
        while True:
            dependents = {
                name for name, spec in registry.items() if selected.intersection(spec.dependencies)
            }
            expanded = selected | dependents
            if expanded == selected:
                break
            selected = expanded
        build_order = order(registry, sorted(selected))
        refs = {
            name: ref
            for name, ref in catalog["tables"].items()
            if not (application and name.startswith("app_") and name not in product_recipes)
        }
        results = {}
        source = (
            load_tables(data_dir)
            if any(registry[n].kind == "source" for n in build_order)
            else None
        )
        source_ref = source.ref if source else None
        for name in build_order:
            spec = registry[name]
            dependencies = {dep: refs[dep] for dep in spec.dependencies}
            dep_tables = {dep: load_table(data_dir, dep, ref) for dep, ref in dependencies.items()}
            input_files, code = recipe_inputs(spec, registry_path, data_dir)
            source_file = None
            if spec.kind == "source":
                assert source is not None and spec.source is not None
                source_file = source_path(data_dir, source, spec.source)
                input_files["source_table"] = file_spec(source_file)
            provenance = [
                m.get("source_release") for _, m in dep_tables.values() if m.get("source_release")
            ]
            if provenance and any(ref != provenance[0] for ref in provenance):
                raise ValueError(f"Mixed source releases: {name}")
            table_source = (
                source_ref if spec.kind == "source" else (provenance[0] if provenance else None)
            )
            if spec.builder == "engine.tables.products:build_product":
                table_source = spec.parameters["source_release"]
            cutoffs = [m["observation_cutoff"] for _, m in dep_tables.values()]
            inherited = cutoffs[0] if cutoffs and all(c == cutoffs[0] for c in cutoffs) else {}
            cutoff = spec.observation_cutoff or (
                source.manifest["current_observations"]
                if spec.kind == "source" and source
                else inherited
            )
            import hashlib

            identity = {
                "recipe": spec.model_dump(),
                "inputs": input_files,
                "dependencies": dependencies,
                "source_release": table_source,
                "code": {n: hashlib.sha256(b).hexdigest() for n, b in code.items()},
                "environment": {
                    "python": platform.python_version(),
                    "polars": pl.__version__,
                    "duckdb": duckdb.__version__,
                },
            }
            fingerprint = content_hash(identity)
            if not force and manifests.get(name, {}).get("fingerprint") == fingerprint:
                results[name] = "unchanged"
                continue
            context = BuildContext(
                data_dir,
                {dep: root / "data.parquet" for dep, (root, _) in dep_tables.items()},
                spec.parameters,
                cutoff,
            )
            if spec.kind == "source":
                assert source_file is not None
                frame = pl.read_parquet(source_file)
            elif spec.kind == "python":
                assert spec.builder is not None
                module_name, function = spec.builder.split(":")
                frame = getattr(importlib.import_module(module_name), function)(context)
            elif spec.kind == "parquet":
                assert spec.source is not None
                frame = pl.read_parquet(inside(data_dir, spec.source))
            else:
                with duckdb.connect(config={"threads": 4, "memory_limit": "1GB"}) as connection:
                    attach_tables(connection, context.dependencies)
                    statements = connection.extract_statements(code["recipe.sql"].decode())
                    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                        raise ValueError("SQL recipes must contain exactly one SELECT query")
                    frame = connection.execute(code["recipe.sql"].decode()).pl()
            validate_frame(frame, spec)
            frame = frame.sort(spec.primary_key)
            version = now.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12]
            root = data_dir / "tables/releases" / name / version
            root.mkdir(parents=True, exist_ok=False)
            frame.write_parquet(root / "data.parquet", compression="zstd", statistics=True)
            for filename, payload in code.items():
                write_new(root / "code" / filename, payload)
            manifest = {
                "schema_version": 1,
                "name": name,
                "version": version,
                "built_at": now.isoformat(),
                "observation_cutoff": cutoff,
                "fingerprint": fingerprint,
                **identity,
                "rows": frame.height,
                "columns": {k: str(v) for k, v in frame.schema.items()},
                "null_counts": frame.null_count().row(0, named=True),
                "files": {
                    p.relative_to(root).as_posix(): file_spec(p)
                    for p in sorted(root.rglob("*"))
                    if p.is_file()
                },
            }
            checked_inputs, checked_code = recipe_inputs(spec, registry_path, data_dir)
            if source_file is not None:
                checked_inputs["source_table"] = file_spec(source_file)
            if (checked_inputs, checked_code) != (input_files, code):
                raise ValueError(f"Inputs changed during build: {name}")
            write_new(root / "manifest.json", json_bytes(manifest))
            refs[name] = {
                "path": (root / "manifest.json").relative_to(data_dir).as_posix(),
                "sha256": sha256(root / "manifest.json"),
            }
            results[name] = f"built ({frame.height} rows)"
        if source and load_tables(data_dir).ref != source_ref:
            raise ValueError("Source batch changed during refresh; rerun against the new release")
        checked_registry = read_registry(registry_path)
        if application:
            if selection(data_dir) != application:
                raise ValueError("Application selection changed during table refresh")
            checked_registry.update(recipes(data_dir))
        if checked_registry != registry:
            raise ValueError("Registry changed during refresh")
        if refs != catalog["tables"] or application != catalog.get("application", {}):
            catalog = {"schema_version": 1, "published_at": now.isoformat(), "tables": refs}
            if application:
                catalog["application"] = application
            publish_catalog(data_dir, catalog)
        return {"tables": results, "catalog": content_hash(catalog)}
