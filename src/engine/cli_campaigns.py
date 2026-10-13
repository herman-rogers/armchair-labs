"""Inspect and verify saved campaigns without training or promoting them."""

import json
from pathlib import Path
from typing import Annotated

import typer

from engine.chairstats import CampaignStore
from engine.chairstats.application import OpeningDriveStore
from engine.config.settings import get_settings

app = typer.Typer(no_args_is_help=True, help=__doc__)


def _store(data_dir: Path | None) -> CampaignStore:
    return OpeningDriveStore((data_dir or get_settings().data_dir) / "campaigns")


@app.command("consolidate")
def consolidate_campaigns(data_dir: Path | None = None):
    """Merge TE into Opening Drive v2 and archive retired application campaigns."""
    from engine.chairstats.consolidate import consolidate_opening_drive

    data = data_dir or get_settings().data_dir
    typer.echo(
        json.dumps(
            consolidate_opening_drive(data / "campaigns", data / "campaign_revisions"), indent=2
        )
    )


@app.command("list")
def list_campaigns(data_dir: Path | None = None):
    """List available immutable campaign bundles."""
    typer.echo(json.dumps(_store(data_dir).campaigns(), indent=2))


@app.command("show")
def show_campaign(campaign_id: str, data_dir: Path | None = None):
    """Show model contracts, forecast cutoffs, and artifact checksums."""
    typer.echo(json.dumps(_store(data_dir).campaign(campaign_id), indent=2))


@app.command("check")
def check_campaign(campaign_id: str, data_dir: Path | None = None):
    """Verify every saved output and weight without deserializing model code."""
    typer.echo(json.dumps(_store(data_dir).check(campaign_id), indent=2))


@app.command("benchmark")
def benchmark_campaign(
    campaign_id: str,
    source: Annotated[list[Path], typer.Option(help="Sealed source.json; repeat per comparator")],
    output: Annotated[Path, typer.Option(help="New directory outside campaign storage")],
    calendar: Annotated[Path | None, typer.Option(help="Cutoff calendar for older bundles")] = None,
    max_age_days: Annotated[int, typer.Option(min=0)] = 14,
    data_dir: Path | None = None,
):
    """Evaluate frozen campaign predictions against external sources; never select models."""
    from engine.evaluation.benchmarks import evaluate

    result = evaluate(_store(data_dir).root, campaign_id, source, output, calendar, max_age_days)
    typer.echo(
        json.dumps(
            {
                "status": result["status"],
                "campaign_id": campaign_id,
                "comparisons": len(result["comparisons"]),
                "report": str(output / "index.html"),
            },
            indent=2,
        )
    )


@app.command("benchmark-calendar")
def benchmark_calendar(
    campaign_id: str,
    output: Annotated[Path, typer.Option()],
    data_dir: Path | None = None,
):
    """Export recorded availability cutoffs for importing external snapshots."""
    from engine.evaluation.sources import export_calendar

    result = export_calendar(_store(data_dir).root, campaign_id, output)
    typer.echo(f"Saved {len(result['rows'])} cutoffs to {output}")


@app.command("benchmark-all")
def benchmark_all_campaigns(
    source: Annotated[list[Path], typer.Option(help="Sealed source.json; repeat per comparator")],
    output: Annotated[Path, typer.Option(help="New evaluation directory")],
    calendar: Annotated[Path | None, typer.Option()] = None,
    max_age_days: Annotated[int, typer.Option(min=0)] = 14,
    data_dir: Path | None = None,
):
    """Create linked external evaluation reports for every local campaign."""
    from engine.evaluation.benchmarks import evaluate_all

    summary = evaluate_all(_store(data_dir).root, source, output, calendar, max_age_days)
    typer.echo(f"Evaluated {len(summary['campaigns'])} campaigns: {output / 'index.html'}")


@app.command("benchmark-import")
def benchmark_import(
    source: Path,
    output: Annotated[Path, typer.Option()],
):
    """Validate and seal a normalized external projection/ranking JSON file."""
    from engine.evaluation.benchmarks import publish_source, read_json

    typer.echo(
        str(publish_source(output, read_json(source), {"raw/import.json": source.read_bytes()}))
    )


@app.command("benchmark-ecr")
def benchmark_ecr(
    archive: Annotated[Path, typer.Option(help="Preserved DynastyProcess db_fpecr parquet")],
    crosswalk: Annotated[Path, typer.Option(help="Exact fantasypros_id to gsis_id parquet")],
    calendar: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    season: Annotated[list[int], typer.Option(help="Repeat for each season")],
):
    """Import dated preseason PPR positional ECR for ranking-only comparisons."""
    from engine.evaluation.sources import ecr_source

    typer.echo(str(ecr_source(archive, crosswalk, calendar, output, season)))


@app.command("benchmark-espn")
def benchmark_espn(
    crosswalk: Annotated[Path, typer.Option(help="Exact espn_id to gsis_id parquet")],
    output: Annotated[Path, typer.Option()],
    season: Annotated[list[int], typer.Option(help="Repeat for each completed season")],
):
    """Fetch historical ESPN season projections; original issue times remain unknown."""
    from engine.evaluation.sources import capture_espn

    typer.echo(str(capture_espn(crosswalk, output, season)))
