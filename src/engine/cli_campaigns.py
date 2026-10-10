"""Inspect and verify saved campaigns without training or promoting them."""

import json
from pathlib import Path

import typer

from engine.chairstats import CampaignStore
from engine.config.settings import get_settings

app = typer.Typer(no_args_is_help=True, help=__doc__)


def _store(data_dir: Path | None) -> CampaignStore:
    return CampaignStore((data_dir or get_settings().data_dir) / "campaigns")


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
