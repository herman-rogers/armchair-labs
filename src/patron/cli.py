"""Command-line entry points."""

from __future__ import annotations

import logging

import polars as pl
import typer

from patron import pipeline
from patron.cli_auth import app as auth_app
from patron.config.league import get_league
from patron.config.settings import get_settings
from patron.data.derived import clear as clear_derived

app = typer.Typer(
    add_completion=False,
    help="Patron Saints analytics engine — league-exact valuation for Sweaty Plays.",
)
app.add_typer(auth_app, name="auth")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )


@app.command()
def board(
    force: bool = typer.Option(False, "--force", help="Rebuild derived caches from scratch."),
    limit: int = typer.Option(160, "--limit", help="Rows in the Markdown board."),
    show: int = typer.Option(25, "--show", help="Rows to print to the terminal."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build the draft board and write it to the outputs directory."""
    _configure_logging(verbose)
    settings = get_settings()
    config = get_league()

    result = pipeline.build(config=config, settings=settings, force=force)

    outputs = settings.outputs_dir
    pipeline.export_board(result.board, outputs / "board.json")
    pipeline.export_markdown(result.board, outputs / "board.md", limit=limit)
    result.player_seasons.write_parquet(outputs / "player_seasons.parquet")
    result.kickers.write_parquet(outputs / "kickers.parquet")
    result.defenses.write_parquet(outputs / "defenses.parquet")

    levels = ", ".join(
        f"{position} {ppg:.1f}" for position, ppg in sorted(result.replacement_levels.items())
    )
    typer.echo(f"\nReplacement levels: {levels}")
    if result.bonus_audit:
        typer.echo(result.bonus_audit.summary())

    display = result.board.head(show).select(
        "rank", "player_display_name", "position", "team", "adj_vor", "ppg", "flags"
    )
    with pl.Config(tbl_rows=show, tbl_hide_dataframe_shape=True, fmt_str_lengths=24):
        typer.echo(f"\n{display}")

    typer.echo(f"\nWrote {result.board.height} players to {outputs}/")


@app.command()
def refresh(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Drop cached data so the next build pulls fresh from nflverse."""
    _configure_logging(verbose)
    settings = get_settings()

    from patron.data.nflverse import clear_cache

    removed = clear_derived(settings)
    clear_cache()

    typer.echo(
        f"Cleared {removed} derived artifact(s) and the nflverse download cache. "
        "The next build will pull fresh."
    )


@app.command()
def validate(
    tolerance_ppg: float = typer.Option(0.3, help="Allowed PPG delta per player."),
    tolerance_vor: float = typer.Option(0.5, help="Allowed VOR delta per player."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compare a freshly built board against the published 2026 draft board."""
    _configure_logging(verbose)
    from patron.validation import compare_to_fixture, load_fixture_board

    settings = get_settings()
    result = pipeline.build(config=get_league(), settings=settings)
    fixture = load_fixture_board(settings.static_dir / "2026_draft_list.md")

    report = compare_to_fixture(
        result.board,
        fixture,
        replacement_levels=result.replacement_levels,
        tolerance_ppg=tolerance_ppg,
        tolerance_vor=tolerance_vor,
    )
    typer.echo(report.render())
    raise typer.Exit(0 if report.passed else 1)


if __name__ == "__main__":
    app()
