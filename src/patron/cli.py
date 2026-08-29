"""Command-line entry points — system operations only.

The CLI does the things a person or a scheduler does *to* the system: sign in, build
the board, run the server, clear caches, check correctness. It deliberately does not
read league data. Pulling ESPN state and rendering it belongs to the server and the
frontend, which is where it can be cached, shared between viewers, and refreshed
without anyone remembering to run a command.
"""

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
    # Keep board.json as the v1 compatibility alias. New consumers select an explicit
    # version; the dashboard/API can switch without rebuilding.
    pipeline.export_board(result.board, outputs / "board.json")
    pipeline.export_board(result.board, outputs / "board_v1.json")
    pipeline.export_board(result.board_v2, outputs / "board_v2.json")
    pipeline.export_markdown(result.board, outputs / "board_v1.md", limit=limit)
    pipeline.export_markdown(result.board_v2, outputs / "board_v2.md", limit=limit)
    result.player_seasons.write_parquet(outputs / "player_seasons.parquet")
    result.kickers.write_parquet(outputs / "kickers.parquet")
    result.defenses.write_parquet(outputs / "defenses.parquet")

    levels = ", ".join(
        f"{position} {ppg:.1f}" for position, ppg in sorted(result.replacement_levels.items())
    )
    typer.echo(f"\nReplacement levels: {levels}")
    if result.bonus_audit:
        typer.echo(result.bonus_audit.summary())

    display = result.board_v2.head(show).select(
        "rank",
        "player_display_name",
        "position",
        "projected_team",
        "v2_overall_vor",
        "adj_proj_vor",
        "proj_ppg",
        "projection_confidence",
        "flags",
    )
    with pl.Config(tbl_rows=show, tbl_hide_dataframe_shape=True, fmt_str_lengths=24):
        typer.echo(f"\n{display}")

    typer.echo(
        f"\nWrote v1 ({result.board.height} players) and "
        f"v2 ({result.board_v2.height} players) to {outputs}/"
    )


@app.command("metric-report")
def metric_report(
    force: bool = typer.Option(False, "--force", help="Rebuild derived caches from scratch."),
    reanalyze: bool = typer.Option(
        False,
        "--reanalyze",
        help="Refit and re-score from the retained fold predictions without rebuilding them.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run rolling v2 backtests and write the frontend metric report."""
    _configure_logging(verbose)
    settings = get_settings()
    if reanalyze:
        result = pipeline.reanalyze_metric_report(settings=settings)
    else:
        result = pipeline.build_metric_report(
            config=get_league(),
            settings=settings,
            force=force,
        )
    paths = pipeline.export_metric_report(result, settings.outputs_dir)
    summary = result.report["data_summary"]
    typer.echo(
        f"Built {summary['forecast_rows']} forecast rows; completed seasons "
        f"{summary['completed_forecasts']}, pending {summary['pending_forecasts']}."
    )
    typer.echo("Wrote " + ", ".join(str(path) for path in paths))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Restart on code changes."),
) -> None:
    """Run the API server, which owns ESPN syncing and serves the frontend's data."""
    import uvicorn

    typer.echo(f"Serving on http://{host}:{port}  (frontend dev server: just web)")
    uvicorn.run("patron.api.app:app", host=host, port=port, reload=reload)


@app.command()
def dev(
    restart: bool = typer.Option(False, "--restart", help="Stop existing services first."),
    api_port: int = typer.Option(8000, "--api-port"),
    web_port: int = typer.Option(5173, "--web-port"),
    reload: bool = typer.Option(True, "--reload/--no-reload"),
) -> None:
    """Run API and frontend under one signal-safe development supervisor."""
    from patron.dev import DevAlreadyRunningError, run

    try:
        exit_code = run(
            restart=restart,
            api_port=api_port,
            web_port=web_port,
            reload=reload,
        )
    except DevAlreadyRunningError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    raise typer.Exit(exit_code)


@app.command("dev-stop")
def dev_stop() -> None:
    """Stop recorded services and reclaim project-owned legacy port orphans."""
    from patron.dev import DevAlreadyRunningError, stop

    try:
        stopped = stop(include_legacy=True)
    except DevAlreadyRunningError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo("Patron dev services stopped." if stopped else "No Patron dev services running.")


@app.command("dev-status")
def dev_status() -> None:
    """Show whether the supervised API and frontend are running."""
    from patron.dev import status

    raise typer.Exit(0 if status() else 1)


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
