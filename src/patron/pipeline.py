"""Pipeline orchestration: the one place that loads, computes, and writes.

Everything below `patron.board` and `patron.metrics` is pure. This module is where the
impure edges meet — nflverse downloads on one side, parquet and JSON on the other —
so the pure core stays testable offline and the I/O stays in one auditable place.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from patron.board.builder import build_board, build_player_seasons
from patron.board.rank import apply_rank_key
from patron.config.league import LeagueConfig, get_league
from patron.config.settings import Settings, get_settings
from patron.data import nflverse
from patron.data.derived import cached_frame
from patron.metrics.backtest import (
    MetricReportConfig,
    build_backtest_predictions,
    render_metric_report_markdown,
)
from patron.metrics.backtest import build_metric_report as analyze_metric_report
from patron.metrics.enrichment import (
    build_expected_opportunity,
    build_injury_history,
    build_market_rankings,
    build_nextgen_features,
    build_player_efficiency,
    build_player_usage,
    build_team_tendencies,
    build_team_volume,
    depth_chart_as_of,
    enrich_player_seasons,
    normalize_ppg_for_active_games,
    validate_participation_usage,
)
from patron.metrics.fit import (
    FittedArtifactError,
    apply_fitted_models,
    build_fit_artifact,
    check_fit_artifact,
    fit_walk_forward,
    models_for_season,
)
from patron.metrics.projection import METRIC_VERSION, build_projection_board
from patron.scoring.bonuses import BonusAudit, extract_touchdown_bonuses
from patron.scoring.dst import build_dst_proxy
from patron.scoring.kickers import aggregate_kicker_seasons, score_kicker_weeks

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Everything one board build produced, plus how the build went."""

    board: pl.DataFrame
    board_v2: pl.DataFrame
    player_seasons: pl.DataFrame
    kickers: pl.DataFrame
    defenses: pl.DataFrame
    bonus_audit: BonusAudit | None
    replacement_levels: dict[str, float]


@dataclass
class MetricReportBuildResult:
    """Persistable rolling forecasts and their metric-level analysis."""

    report: dict[str, Any]
    predictions: pl.DataFrame


def load_bonuses(
    config: LeagueConfig,
    force: bool = False,
) -> tuple[pl.DataFrame, BonusAudit | None]:
    """Per-player, per-week big-play bonuses, cached as a derived artifact.

    Reducing multi-season play-by-play is the expensive step in the whole pipeline, and
    it produces a few thousand rows from the largest table nflverse publishes. That is
    exactly the shape worth persisting.

    The audit is only available when the artifact is actually built; a cache hit
    returns None for it, since the reconciliation was done when the file was written.
    """
    audit: BonusAudit | None = None

    def build() -> pl.DataFrame:
        nonlocal audit
        plays = nflverse.load_touchdown_plays(config.seasons)
        bonuses, audit = extract_touchdown_bonuses(plays)
        logger.info(audit.summary())
        if not audit.is_balanced:
            logger.warning(
                "bonus points earned and credited disagree; %s credit(s) were dropped "
                "to null player ids",
                audit.dropped_credits,
            )
        return bonuses

    frame = cached_frame("bonuses", config.seasons, build, force=force)
    return frame, audit


def _season_equivalent(board: pl.DataFrame, season_games: int) -> pl.DataFrame:
    """Fitted season points as a per-scheduled-game rate; the waiver-comparison value."""
    source = (
        pl.col("fitted_season_points") / float(season_games)
        if "fitted_season_points" in board.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    return board.with_columns(source.alias("season_equivalent_ppg"))


def load_fitted_models(
    report_path: Path,
    forecast_season: int,
    report_config: MetricReportConfig,
    live_depth_latest: str | None,
) -> dict[str, Any]:
    """Per-position fitted models for the draft season, from the last metric report.

    The artifact must have been fitted for this season, with this fit configuration and
    depth-chart cutoff, on the same depth-chart snapshot the live board selected.  Any
    mismatch yields no models: the fitted columns stay null and ``apply_rank_key``
    falls back to the configured key with a logged warning, never a silent zero.
    """
    if not report_path.exists():
        logger.warning("no metric report at %s; fitted ranker unavailable", report_path)
        return {}
    report = json.loads(report_path.read_text())
    try:
        check_fit_artifact(
            report.get("fitted_artifact"),
            report_config.fit,
            forecast_season,
            report_config.depth_chart_cutoff,
            live_depth_latest,
        )
    except FittedArtifactError as error:
        logger.warning("fitted ranker rejected: %s; falling back to %s", error, "v2_score")
        return {}
    models = models_for_season(report.get("fitted_models") or [], forecast_season)
    if not models:
        logger.warning("metric report has no fitted models for %s", forecast_season)
    else:
        logger.info(
            "fitted ranker loaded for %s (%s): %s",
            forecast_season,
            report["fitted_artifact"]["fingerprint"],
            ", ".join(sorted(models)),
        )
    return models


def load_draft_depth_chart(
    config: LeagueConfig,
    report_config: MetricReportConfig,
) -> pl.DataFrame | None:
    """Depth chart for the draft season, selected exactly as the backtest selects it.

    The pending backtest fold scores the draft season on the last snapshot published
    on or before the configured cutoff; the live board must use the same snapshot or
    the fitted weights are applied to inputs they were never validated on.
    """
    frames: list[pl.DataFrame] = []
    for season in sorted({config.board_season, config.draft_season}):
        try:
            frames.append(nflverse.load_depth_charts([season]))
        except Exception:  # noqa: BLE001 - one missing season must not erase the board
            logger.warning("depth charts unavailable for %s", season)
    depth = pl.concat(frames, how="diagonal_relaxed") if frames else None
    selected = depth_chart_as_of(depth, config.draft_season, report_config.depth_chart_cutoff)
    if selected is None:
        logger.warning(
            "no depth chart on or before %s-%s",
            config.draft_season,
            report_config.depth_chart_cutoff,
        )
    else:
        latest = selected["depth_chart_date"].cast(pl.String).max()
        logger.info("draft depth chart: %s players, latest snapshot %s", selected.height, latest)
    return selected


def build(
    config: LeagueConfig | None = None,
    settings: Settings | None = None,
    force: bool = False,
    strict_overrides: bool = True,
    report_config: MetricReportConfig | None = None,
) -> BuildResult:
    """Run the full Phase 1 pipeline and return every artifact it produced."""
    config = config or get_league()
    settings = settings or get_settings()
    settings.ensure_dirs()

    weeks = nflverse.load_player_weeks(config.seasons)
    bonuses, audit = load_bonuses(config, force=force)

    player_seasons = build_player_seasons(weeks, bonuses, config)
    team_volume = build_team_volume(nflverse.load_team_weeks(config.seasons))
    participation_seasons = [season for season in config.seasons if season >= 2016]
    usage = cached_frame(
        "projection_usage_v3",
        config.seasons,
        lambda: build_player_usage(
            weeks,
            nflverse.load_projection_plays(config.seasons),
            nflverse.load_participation(participation_seasons),
        ),
        force=force,
        settings=settings,
    )
    injury_history = build_injury_history(nflverse.load_injuries(config.seasons))
    player_seasons = enrich_player_seasons(
        player_seasons,
        usage,
        team_volume,
        injury_history,
    )
    validate_participation_usage(usage, participation_seasons)
    player_seasons = player_seasons.with_columns(
        (pl.col("season") >= 2009).alias("injury_data_available"),
        (pl.col("season") >= 2016).alias("participation_data_available"),
    )
    projection_seasons = normalize_ppg_for_active_games(player_seasons)
    report_config = report_config or MetricReportConfig.from_config()
    depth_chart = load_draft_depth_chart(config, report_config)
    live_depth_latest = (
        depth_chart["depth_chart_date"].cast(pl.String).max() if depth_chart is not None else None
    )
    birth_dates = nflverse.load_birth_dates(config.board_season)
    board = build_board(
        player_seasons,
        birth_dates,
        config=config,
        strict_overrides=strict_overrides,
    ).with_columns(pl.lit("v1").alias(METRIC_VERSION))
    projection_base = build_board(
        projection_seasons,
        birth_dates,
        config=config,
        strict_overrides=strict_overrides,
    ).with_columns(pl.lit("v1").alias(METRIC_VERSION))
    board_v2 = build_projection_board(
        projection_seasons,
        projection_base,
        config,
        current_players=depth_chart,
    )
    board_v2 = apply_rank_key(
        _season_equivalent(
            apply_fitted_models(
                board_v2,
                load_fitted_models(
                    settings.outputs_dir / "metric_report.json",
                    config.draft_season,
                    report_config,
                    live_depth_latest,
                ),
                report_config.fit,
            ),
            config.metrics.projection_season_games,
        ),
        config,
    )

    board_weeks = weeks.filter(pl.col("season") == config.board_season)
    kickers = aggregate_kicker_seasons(
        score_kicker_weeks(board_weeks.filter(pl.col("position") == "K"))
    )
    defenses = build_dst_proxy(board_weeks, nflverse.load_schedules(config.board_season))

    levels = dict(
        board.filter(pl.col("repl_ppg").is_not_null())
        .group_by("position")
        .agg(pl.col("repl_ppg").first())
        .iter_rows()
    )

    return BuildResult(
        board=board,
        board_v2=board_v2,
        player_seasons=projection_seasons,
        kickers=kickers,
        defenses=defenses,
        bonus_audit=audit,
        replacement_levels=levels,
    )


def build_metric_report(
    config: LeagueConfig | None = None,
    settings: Settings | None = None,
    report_config: MetricReportConfig | None = None,
    force: bool = False,
) -> MetricReportBuildResult:
    """Build historical v2 forecast folds and analyze the configured metric catalog."""
    config = config or get_league()
    settings = settings or get_settings()
    report_config = report_config or MetricReportConfig.from_config()
    settings.ensure_dirs()

    history_config = config.model_copy(update={"seasons": list(report_config.input_seasons)})
    weeks = nflverse.load_player_weeks(history_config.seasons)
    bonuses, _ = load_bonuses(history_config, force=force)
    player_seasons = build_player_seasons(weeks, bonuses, history_config)
    team_volume = build_team_volume(nflverse.load_team_weeks(history_config.seasons))
    participation_seasons = [season for season in history_config.seasons if season >= 2016]
    projection_plays = nflverse.load_projection_plays(history_config.seasons)
    usage = cached_frame(
        "metric_report_usage_v4",
        history_config.seasons,
        lambda: build_player_usage(
            weeks,
            projection_plays,
            nflverse.load_participation(participation_seasons),
        ),
        force=force,
        settings=settings,
    )
    injury_seasons = [season for season in history_config.seasons if season >= 2009]
    injuries = build_injury_history(nflverse.load_injuries(injury_seasons))
    expected_seasons = [season for season in history_config.seasons if season >= 2006]
    expected_opportunity = cached_frame(
        "metric_report_expected_opportunity_v1",
        expected_seasons,
        lambda: build_expected_opportunity(
            nflverse.load_expected_opportunity(expected_seasons), weeks
        ),
        force=force,
        settings=settings,
    )
    nextgen_seasons = [season for season in history_config.seasons if season >= 2016]
    nextgen = cached_frame(
        "metric_report_nextgen_v2",
        nextgen_seasons,
        lambda: build_nextgen_features(
            nflverse.load_nextgen_stats(nextgen_seasons, "passing"),
            nflverse.load_nextgen_stats(nextgen_seasons, "receiving"),
            nflverse.load_nextgen_stats(nextgen_seasons, "rushing"),
        ),
        force=force,
        settings=settings,
    )
    team_tendencies = cached_frame(
        "metric_report_team_tendencies_v1",
        history_config.seasons,
        lambda: build_team_tendencies(projection_plays),
        force=force,
        settings=settings,
    )
    player_seasons = enrich_player_seasons(
        player_seasons,
        usage,
        team_volume,
        injuries,
        player_efficiency=build_player_efficiency(weeks),
        expected_opportunity=expected_opportunity,
        nextgen_features=nextgen,
        team_tendencies=team_tendencies,
    )
    validate_participation_usage(usage, participation_seasons)
    player_seasons = player_seasons.with_columns(
        (pl.col("season") >= 2009).alias("injury_data_available"),
        (pl.col("season") >= 2016).alias("participation_data_available"),
    )
    player_seasons = normalize_ppg_for_active_games(player_seasons)

    birth_frames: list[pl.DataFrame] = []
    for season in report_config.input_seasons:
        try:
            birth_frames.append(nflverse.load_birth_dates(season))
        except Exception:  # noqa: BLE001 - one historical roster should not erase the report
            logger.warning("birth dates unavailable for %s", season)
    if not birth_frames:
        raise RuntimeError("no roster birth dates were available for the metric report")
    birth_dates = pl.concat(birth_frames, how="diagonal_relaxed").unique(
        subset=["player_id"], keep="last"
    )

    depth_frames: list[pl.DataFrame] = []
    for season in report_config.forecast_seasons:
        if season < report_config.depth_chart_start_season:
            continue
        try:
            depth_frames.append(nflverse.load_depth_charts([season]))
        except Exception:  # noqa: BLE001 - projection falls back cleanly without depth charts
            logger.warning("depth charts unavailable for %s", season)
    depth_charts = pl.concat(depth_frames, how="diagonal_relaxed") if depth_frames else None

    predictions = build_backtest_predictions(
        player_seasons,
        birth_dates,
        config,
        report_config,
        depth_charts=depth_charts,
        market_rankings=build_market_rankings(
            nflverse.load_fantasy_rankings(),
            nflverse.load_id_crosswalk(),
            report_config.depth_chart_cutoff,
        ),
    )
    report, predictions = _analyze_with_fit(predictions, report_config)
    return MetricReportBuildResult(report=report, predictions=predictions)


def _analyze_with_fit(
    predictions: pl.DataFrame,
    report_config: MetricReportConfig,
) -> tuple[dict[str, Any], pl.DataFrame]:
    predictions = predictions.drop(
        [
            c
            for c in (*report_config.fit.output_columns, "actual_played")
            if c in predictions.columns
        ]
    )
    predictions, fitted_models = fit_walk_forward(predictions, report_config.fit)
    artifact = build_fit_artifact(report_config.fit, predictions, report_config.depth_chart_cutoff)
    report = analyze_metric_report(
        predictions, report_config, fitted_models=fitted_models, fitted_artifact=artifact
    )
    return report, predictions


def reanalyze_metric_report(
    settings: Settings | None = None,
    report_config: MetricReportConfig | None = None,
) -> MetricReportBuildResult:
    """Refit and re-score from the retained fold predictions without rebuilding them.

    Ranking-contract or fitted-model changes need no nflverse rebuild; only changes to
    the projection itself or the historical data do.
    """
    settings = settings or get_settings()
    report_config = report_config or MetricReportConfig.from_config()
    path = settings.outputs_dir / "metric_backtest_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no retained fold predictions at {path}; run `patron metric-report`"
        )
    report, predictions = _analyze_with_fit(pl.read_parquet(path), report_config)
    return MetricReportBuildResult(report=report, predictions=predictions)


def export_metric_report(result: MetricReportBuildResult, outputs: Path) -> tuple[Path, Path, Path]:
    """Write JSON for the API, Markdown for humans, and detailed forecast rows."""
    json_path = outputs / "metric_report.json"
    markdown_path = outputs / "metric_report.md"
    predictions_path = outputs / "metric_backtest_predictions.parquet"
    json_path.write_text(json.dumps(result.report, indent=2, default=str))
    markdown_path.write_text(render_metric_report_markdown(result.report))
    result.predictions.write_parquet(predictions_path)
    return json_path, markdown_path, predictions_path


#: Board columns written to JSON, in display order.
BOARD_EXPORT_COLUMNS: tuple[str, ...] = (
    "metric_version",
    "rank",
    "player_display_name",
    "position",
    "team",
    "adj_vor",
    "vor",
    "ppg",
    "flags",
    "games",
    "season_pts",
    "bonus_pts",
    "floor",
    "volatility",
    "wtd_opp",
    "target_share",
    "air_yards_share",
    "wopr",
    "td_over_exp",
    "age_at_season",
    "repl_ppg",
    "override_delta",
    "override_reason",
    "player_id",
    # v2 forward-projection columns. v1 exports omit these cleanly.
    "projected_team",
    "depth_chart_rank",
    "depth_chart_position",
    "depth_chart_position_group",
    "depth_chart_date",
    "depth_role_factor",
    "v2_overall_vor",
    "v2_position_rank",
    "v2_rank_key",
    "v2_rank_value",
    "v2_rank_vor",
    "fitted_ppg",
    "fitted_games",
    "fitted_season_points",
    "fitted_season_points_direct",
    "fitted_return_prob",
    "fitted_games_if_played",
    "fitted_two_stage",
    "adj_proj_vor",
    "proj_vor",
    "proj_ppg",
    "proj_repl_ppg",
    "historical_ppg_prior",
    "individual_prior_ppg",
    "prior_branch_ppg",
    "component_proj_ppg",
    "projected_floor",
    "projected_ceiling",
    "projected_volatility",
    "projection_confidence",
    "availability_confidence",
    "projected_availability",
    "expected_games",
    "season_equivalent_ppg",
    "historical_injury_report_weeks",
    "injury_missed_equivalents",
    "effective_games",
    "age_factor",
    "qb_context",
    "team_scoring_context",
    "team_pass_volume",
    "team_dropbacks",
    "team_rush_volume",
    "teammate_competition",
    "season_target_share",
    "season_carry_share",
    "projected_target_share",
    "projected_carry_share",
    "projected_wopr",
    "projected_air_yards_share",
    "projected_targets_pg",
    "projected_route_opportunities_pg",
    "projected_route_participation",
    "projected_targets_per_route_opportunity",
    "projected_carries_pg",
    "projected_receptions_pg",
    "projected_receiving_yards_pg",
    "projected_rushing_yards_pg",
    "projected_receiving_tds_pg",
    "projected_rushing_tds_pg",
    "projected_red_zone_targets_pg",
    "projected_end_zone_targets_pg",
    "projected_red_zone_carries_pg",
    "projected_goal_line_carries_pg",
    "projected_pass_attempts_pg",
    "projected_passing_yards_pg",
    "projected_passing_tds_pg",
    "projected_interceptions_pg",
    "projected_bonus_pg",
    "air_yard_factor",
    "opportunity_multiplier",
    "projection_reason",
    "historical_vor",
    "historical_repl_ppg",
)


def export_board(board: pl.DataFrame, path: Path) -> Path:
    """Write the board as JSON records, rounded for display."""
    columns = [column for column in BOARD_EXPORT_COLUMNS if column in board.columns]
    export = board.select(columns).with_columns(
        pl.col(pl.Float64).round(3),
    )
    path.write_text(json.dumps(export.to_dicts(), indent=2, default=str))
    return path


def export_markdown(board: pl.DataFrame, path: Path, limit: int | None = None) -> Path:
    """Write the board in the same shape as `data/static/2026_draft_list.md`.

    Same columns and same order as the artifact this pipeline reproduces, so the two
    can be diffed by eye as well as by the fixture test.
    """
    rows = board.head(limit) if limit else board
    is_v2 = "proj_ppg" in rows.columns
    version = "v2 projected" if is_v2 else "v1 historical"
    vor_column = (
        ("v2_overall_vor" if "v2_overall_vor" in rows.columns else "v2_score")
        if is_v2
        else "adj_vor"
    )
    ppg_column = "proj_ppg" if is_v2 else "ppg"
    team_column = "projected_team" if is_v2 else "team"
    lines = [
        f"# Overall Draft Board ({version}) — {rows.height} players",
        "",
        f"Rank | Player | Pos | Team | {'Overall VOR' if is_v2 else 'VOR'} | PPG | Flags | Notes",
        "---|---|---|---|---|---|---|---",
    ]
    for row in rows.iter_rows(named=True):
        note = row.get("override_reason") or ""
        lines.append(
            f"{row['rank']} | {row['player_display_name']} | {row['position']} | "
            f"{row[team_column]} | {row[vor_column]:.1f} | {row[ppg_column]:.1f} | "
            f"{row['flags']} | {note}"
        )
    lines.append("")
    lines.append(
        "Flags: BUY=TD under-exp w/ volume | TD-luck=over-exp, price down | "
        "age=RB 27.5+ | Xgms=small sample."
    )
    path.write_text("\n".join(lines))
    return path
