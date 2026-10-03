"""Build, inspect, query, cache, and share named analytical datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from engine.config.settings import get_settings
from engine.tables.registry import DEFAULT_REGISTRY, read_registry

app = typer.Typer(no_args_is_help=True, help=__doc__)


def emit(value):
    typer.echo(json.dumps(value, indent=2, default=str, allow_nan=False))


def directory(value: Path | None) -> Path:
    return (value or get_settings().data_dir).resolve()


@app.command("list")
def list_tables(
    data_dir: Path | None = None,
    registry: Path = DEFAULT_REGISTRY,
):
    """Show registered tables, published row counts, cutoffs, and refresh cadence."""
    from engine.tables.storage import current, load_table

    root = directory(data_dir)
    catalog = current(root)
    specs = read_registry(registry)
    result = []
    for name in sorted(specs.keys() | catalog["tables"].keys()):
        spec = specs.get(name)
        manifest = (
            load_table(root, name, catalog["tables"][name])[1] if name in catalog["tables"] else {}
        )
        result.append(
            {
                "name": f"analytics.{name}",
                "description": spec.description if spec else manifest["recipe"]["description"],
                "refresh_hours": spec.refresh_hours
                if spec
                else manifest["recipe"]["refresh_hours"],
                "rows": manifest.get("rows"),
                "built_at": manifest.get("built_at"),
                "observation_cutoff": manifest.get("observation_cutoff"),
            }
        )
    emit(result)


@app.command()
def describe(name: str, data_dir: Path | None = None):
    """Show schema, keys, recipe, input versions, quality, and limitations."""
    from engine.tables.storage import current, load_table

    root = directory(data_dir)
    emit(load_table(root, name, current(root)["tables"][name])[1])


@app.command()
def refresh(
    names: Annotated[list[str] | None, typer.Argument()] = None,
    due: bool = False,
    force: bool = False,
    data_dir: Path | None = None,
    registry: Path = DEFAULT_REGISTRY,
    upload: bool = False,
    store: str = "gs://armchair-labs-data/armchair-labs/tables",
):
    """Build tables and affected dependents; publish only after every build validates.

    --due selects daily/weekly recipes that are due. Unchanged inputs are a no-op.
    Omitting names and --due explicitly refreshes all recipes, including frozen imports.
    """
    from engine.tables.workflow import refresh_and_publish

    root = directory(data_dir)
    emit(
        refresh_and_publish(
            root,
            registry_path=registry,
            names=names,
            due=due,
            force=force,
            upload=upload,
            store=store,
        )
    )


@app.command()
def query(
    sql: str = typer.Argument(..., help="Trusted local SQL, or @path/to/query.sql"),
    output: Path | None = None,
    native: Annotated[bool, typer.Option("--native/--parquet")] = True,
    data_dir: Path | None = None,
):
    """Query a pinned local catalog. Export CSV/Parquet with --output."""
    from engine.tables.query import connect

    statement = Path(sql[1:]).read_text() if sql.startswith("@") else sql
    with connect(directory(data_dir), native=native) as connection:
        frame = connection.execute(statement).pl()
    if output:
        if output.exists():
            raise typer.BadParameter("Output already exists; choose a new path")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".parquet":
            frame.write_parquet(output, compression="zstd")
        elif output.suffix == ".csv":
            frame.write_csv(output)
        else:
            raise typer.BadParameter("Output must end in .parquet or .csv")
        emit({"rows": frame.height, "output": str(output)})
    else:
        emit(frame.to_dicts())


@app.command()
def cache(data_dir: Path | None = None):
    """Build/reuse the disposable native DuckDB snapshot of the selected catalog."""
    from engine.tables.query import materialize

    emit({"database": str(materialize(directory(data_dir)))})


@app.command()
def verify(data_dir: Path | None = None):
    """Verify every selected manifest, Parquet file, recipe, and dependency link."""
    from engine.tables.storage import current, load_table

    root = directory(data_dir)
    catalog = current(root)
    for name, ref in catalog["tables"].items():
        load_table(root, name, ref)
    emit({"verified_tables": len(catalog["tables"])})


@app.command("archive-research")
def archive_research(source: Path, run: str, data_dir: Path | None = None):
    """Copy a completed study into research/<run>; convert CSVs to Parquet."""
    from engine.tables.research import archive

    root = directory(data_dir)
    (root / ".runtime").mkdir(parents=True, exist_ok=True)
    emit(archive(root, source, run))


@app.command()
def publish(
    release: str,
    store: str = "gs://armchair-labs-data/armchair-labs/tables",
    research: Annotated[list[str] | None, typer.Option("--research")] = None,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
):
    """Upload a sealed table bundle; optionally include named research archives."""
    from engine.tables.transport import publish as upload

    emit(
        upload(
            directory(data_dir), release, store_uri=store, cache=cache_dir, research_runs=research
        )
    )


@app.command()
def fetch(
    reference: Annotated[Path, typer.Argument()] = Path("data/releases/tables/current.json"),
    offline: bool = False,
    research: bool = False,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
):
    """Fetch/verify a pinned bundle and atomically select its tables on this machine."""
    from engine.tables.transport import fetch as download

    emit(
        download(
            directory(data_dir),
            reference,
            cache=cache_dir,
            offline=offline,
            include_research=research,
        )
    )
