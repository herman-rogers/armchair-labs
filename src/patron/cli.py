"""Command-line entry points — system operations only.

The CLI does the things a person or a scheduler does *to* the system: sign in, build
the board, run the server, clear caches, check correctness. It deliberately does not
read league data. Pulling ESPN state and rendering it belongs to the server and the
frontend, which is where it can be cached, shared between viewers, and refreshed
without anyone remembering to run a command.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import yaml

from patron import pipeline
from patron.cli_auth import app as auth_app
from patron.config.league import get_league
from patron.config.settings import CONFIG_DIR, get_settings
from patron.data.derived import clear as clear_derived
from patron.metrics.backtest import MetricReportConfig
from patron.metrics.discovery import (
    DiscoveryConfig,
    build_discovery_report,
    build_weekly_discovery_features,
    fit_discovery_walk_forward,
    render_discovery_markdown,
    report_json,
)
from patron.metrics.prospective import (
    ProspectiveGradePending,
    file_sha256,
    grade_frozen_forecast,
)
from patron.metrics.rich_weekly import (
    build_rich_weekly_features,
    build_rich_weekly_panel,
    build_weekly_route_panel,
)

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
    pipeline.export_board(result.board_adaptive, outputs / "board_adaptive.json")
    pipeline.export_markdown(result.board, outputs / "board_v1.md", limit=limit)
    pipeline.export_markdown(result.board_v2, outputs / "board_v2.md", limit=limit)
    pipeline.export_markdown(
        result.board_adaptive, outputs / "board_adaptive.md", limit=limit
    )
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
        f"v2 ({result.board_v2.height} players), and adaptive "
        f"({result.board_adaptive.height} players) to {outputs}/"
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


@app.command("feature-discovery")
def feature_discovery(
    force_temporal: bool = typer.Option(
        False,
        "--force-temporal",
        help="Rebuild weekly temporal and random-convolution features.",
    ),
    force_rich: bool = typer.Option(
        False,
        "--force-rich",
        help="Rebuild the richer weekly role, availability, and context panel.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run report-only automated feature discovery with nested walk-forward tests."""
    _configure_logging(verbose)
    settings = get_settings()
    settings.ensure_dirs()
    predictions_path = settings.outputs_dir / "metric_backtest_predictions.parquet"
    if not predictions_path.exists():
        raise typer.BadParameter(
            f"no retained folds at {predictions_path}; run `patron metric-report` first"
        )
    report_config = MetricReportConfig.from_config()
    discovery_config = DiscoveryConfig()
    temporal_path = settings.derived_cache_dir / "feature_discovery_temporal_v1.parquet"
    if force_temporal or not temporal_path.exists():
        weeks = pipeline.nflverse.load_player_weeks(list(report_config.input_seasons))
        temporal = build_weekly_discovery_features(
            weeks,
            report_config.forecast_seasons,
            config=discovery_config,
        )
        temporal.write_parquet(temporal_path)
    else:
        temporal = pl.read_parquet(temporal_path)
    retained = pl.read_parquet(predictions_path)
    rich_path = settings.derived_cache_dir / "feature_discovery_rich_weekly_v1.parquet"
    if force_rich or not rich_path.exists():
        panel_path = settings.derived_cache_dir / "feature_discovery_rich_panel_v1.parquet"
        if force_rich or not panel_path.exists():
            input_seasons = list(report_config.input_seasons)
            final_input = max(input_seasons)
            source_2013 = list(range(max(2013, min(input_seasons)), final_input + 1))
            route_seasons = list(range(max(2016, min(input_seasons)), final_input + 1))
            xfp_seasons = list(range(max(2006, min(input_seasons)), final_input + 1))
            logging.getLogger(__name__).info("building rich weekly source panel")
            weeks = pipeline.nflverse.load_player_weeks(input_seasons)
            route_panel = build_weekly_route_panel(
                weeks,
                pipeline.nflverse.load_projection_plays(route_seasons),
                pipeline.nflverse.load_participation_flexible(route_seasons),
            )
            rich_panel = build_rich_weekly_panel(
                weeks,
                pipeline.nflverse.load_team_weeks(input_seasons),
                pipeline.nflverse.load_snap_counts(source_2013),
                pipeline.nflverse.load_players(),
                pipeline.nflverse.load_injuries(source_2013),
                pipeline.nflverse.load_weekly_rosters(source_2013),
                pipeline.nflverse.load_expected_opportunity(xfp_seasons),
                route_panel,
            )
            rich_panel.write_parquet(panel_path)
        else:
            rich_panel = pl.read_parquet(panel_path)
        rich = build_rich_weekly_features(
            rich_panel,
            retained.select("forecast_season", "player_id"),
            history_seasons=report_config.history_seasons,
        )
        rich.write_parquet(rich_path)
    else:
        rich = pl.read_parquet(rich_path)
    predictions, fits = fit_discovery_walk_forward(
        retained,
        temporal_features=temporal,
        rich_features=rich,
        config=discovery_config,
        top_k=report_config.ranking.top_k,
    )
    report = build_discovery_report(
        predictions,
        report_config,
        fits,
        config=discovery_config,
    )
    json_path = settings.outputs_dir / "feature_discovery_report.json"
    markdown_path = settings.outputs_dir / "feature_discovery_report.md"
    output_predictions = settings.outputs_dir / "feature_discovery_predictions.parquet"
    json_path.write_text(report_json(report))
    markdown_path.write_text(render_discovery_markdown(report))
    predictions.write_parquet(output_predictions)
    promoted = [
        row["candidate"] for row in report["promotion_gates"] if row["promotion_ready"]
    ]
    typer.echo(
        f"Scored {predictions.height} rows across {len(fits)} walk-forward refits; "
        f"promotion-ready: {', '.join(promoted) if promoted else 'none'}."
    )
    typer.echo(f"Wrote {json_path}, {markdown_path}, {output_predictions}")


@app.command("grade-prospective")
def grade_prospective(
    season: int = typer.Option(2026, "--season", help="Frozen forecast season to grade."),
    snapshot: Annotated[
        Path | None,
        typer.Option("--snapshot", help="Frozen CSV; defaults to data/static for the season."),
    ] = None,
    outcomes: Annotated[
        Path | None,
        typer.Option("--outcomes", help="Completed metric predictions containing outcomes."),
    ] = None,
) -> None:
    """Grade an immutable prospective forecast, refusing partial-season outcomes."""
    settings = get_settings()
    manifest_path = CONFIG_DIR / f"experimental_freeze_{season}.yaml"
    if not manifest_path.exists():
        raise typer.BadParameter(f"no prospective manifest at {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text())
    snapshot_path = snapshot or settings.static_dir / f"experimental_{season}_predictions.csv"
    outcomes_path = outcomes or settings.outputs_dir / "metric_backtest_predictions.parquet"
    if not snapshot_path.exists():
        raise typer.BadParameter(f"no frozen snapshot at {snapshot_path}")
    if not outcomes_path.exists():
        raise typer.BadParameter(f"no outcome file at {outcomes_path}")
    expected = str(manifest["pending_prediction_sha256"])
    if file_sha256(snapshot_path) != expected:
        raise typer.BadParameter("frozen snapshot file does not match its manifest digest")
    try:
        report = grade_frozen_forecast(
            pl.read_csv(snapshot_path),
            pl.read_parquet(outcomes_path),
            MetricReportConfig.from_config(),
            season,
            expected_sha256=expected,
            grade_not_before=date.fromisoformat(str(manifest["grade_not_before"])),
        )
    except ProspectiveGradePending as error:
        typer.echo(f"PENDING: {error}")
        raise typer.Exit(2) from error
    output_path = settings.outputs_dir / f"prospective_{season}_grade.json"
    output_path.write_text(json.dumps(report, indent=2, default=str))
    overall = report["overall"]
    typer.echo(
        f"Graded untouched {season}: adaptive hit {overall['adaptive_hit_rate']:.3f}, "
        f"incumbent {overall['incumbent_hit_rate']:.3f}, ECR {overall['ecr_hit_rate']:.3f}; "
        f"primary gates {'PASS' if report['passes_primary_gates'] else 'FAIL'}."
    )
    typer.echo(f"Wrote {output_path}")


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
