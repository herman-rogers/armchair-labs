"""Inspect the research model library without training or changing publication."""

import json
from pathlib import Path

import typer

from engine.config.settings import REPO_ROOT, get_settings
from engine.models.catalog import build_catalog, load_families
from engine.models.render import write_catalog

app = typer.Typer(no_args_is_help=True, help=__doc__)


@app.command("list")
def list_models(layer: str | None = None):
    """List curated method families; works without local generated datasets."""
    rows = load_families()
    if layer:
        rows = [r for r in rows if r["layer"] == layer]
    typer.echo(json.dumps(rows, indent=2))


@app.command("show")
def show_model(model_id: str):
    """Describe a method family without inferring serving approval."""
    row = next((r for r in load_families() if r["id"] == model_id), None)
    if row is None:
        raise typer.BadParameter(f"Unknown family: {model_id}")
    typer.echo(json.dumps(row, indent=2))


@app.command("build")
def build_models(
    output: Path | None = None,
    data_dir: Path | None = None,
    open_browser: bool = typer.Option(False, "--open"),
):
    """Inventory local artifacts and generate JSON plus a searchable HTML library."""
    data = (data_dir or get_settings().data_dir).resolve()
    destination = (output or data / "outputs/model_catalog").resolve()
    catalog = build_catalog(REPO_ROOT, data)
    path = write_catalog(catalog, destination)
    typer.echo(
        json.dumps(
            {"html": str(path), "summary": catalog["summary"], "issues": len(catalog["issues"])},
            indent=2,
        )
    )
    if open_browser:
        import webbrowser

        if not webbrowser.open(path.as_uri()):
            typer.echo(f"Browser did not open; use {path}", err=True)
    if any(i["severity"] == "error" for i in catalog["issues"]):
        raise typer.Exit(1)


@app.command("check")
def check_models(data_dir: Path | None = None):
    """Report unresolved source links and current-release metadata problems."""
    catalog = build_catalog(REPO_ROOT, data_dir or get_settings().data_dir)
    typer.echo(json.dumps(catalog["issues"], indent=2))
    if any(i["severity"] in {"error", "warning"} for i in catalog["issues"]):
        raise typer.Exit(1)
