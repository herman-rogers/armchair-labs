"""Pipeline orchestration: the one place that loads, computes, and writes.

Everything below `patron.board` and `patron.metrics` is pure. This module is where the
impure edges meet — nflverse downloads on one side, parquet and JSON on the other —
so the pure core stays testable offline and the I/O stays in one auditable place.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from patron.artifacts import atomic_write, verify_draft
from patron.board.builder import build_board, build_player_seasons
from patron.board.rank import apply_rank_key
from patron.config.league import LeagueConfig, get_league
from patron.config.settings import CONFIG_DIR, Settings, get_settings
from patron.data import nfl_transactions, nflverse
from patron.data.derived import cached_frame
from patron.data.historical_evidence import load_transaction_backfill
from patron.metrics.availability import EVIDENCE_FILE, attach_known_absences, load_absences
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
from patron.metrics.experimental import (
    build_combine_features,
    build_contract_features,
    build_forecast_features,
    build_player_background,
    build_rookie_features,
    build_snap_features,
    build_transaction_features,
)
from patron.metrics.fit import (
    FittedArtifactError,
    apply_fitted_models,
    build_fit_artifact,
    check_fit_artifact,
    fit_walk_forward,
    models_for_season,
    production_config,
)
from patron.metrics.forecast import FORECAST_COLUMNS, attach_forecast
from patron.metrics.positions import (
    canonical_positions,
    historical_positions,
    repair_player_week_positions,
)
from patron.metrics.projection import METRIC_VERSION, build_projection_board
from patron.metrics.prospective import load_verified_snapshot
from patron.metrics.rich_weekly import (
    build_rich_weekly_features,
    build_rich_weekly_panel,
    build_weekly_route_panel,
)
from patron.metrics.roster_evidence import attach_roster_evidence, supplement_identity_names
from patron.metrics.transaction_events import normalize_transaction_sources
from patron.scoring.bonuses import BonusAudit, extract_touchdown_bonuses
from patron.scoring.dst import build_dst_proxy
from patron.scoring.kickers import aggregate_kicker_seasons, score_kicker_weeks

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Everything one board build produced, plus how the build went."""

    board: pl.DataFrame
    board_v2: pl.DataFrame
    board_adaptive: pl.DataFrame
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
    settings: Settings | None = None,
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

    frame = cached_frame("bonuses", config.seasons, build, force=force, settings=settings)
    return frame, audit


MARKET_COLUMNS = (
    "market_position",
    "market_ecr",
    "market_ecr_score",
    "market_ecr_sd",
    "market_snapshot",
)


def load_draft_market(
    config: LeagueConfig, report_config: MetricReportConfig
) -> pl.DataFrame | None:
    """FantasyPros consensus for the draft season, resolved to GSIS ids with names."""
    try:
        crosswalk = nflverse.load_id_crosswalk()
        market = build_market_rankings(
            nflverse.load_fantasy_rankings(), crosswalk, report_config.depth_chart_cutoff
        ).filter(pl.col("forecast_season") == config.draft_season)
    except Exception:  # noqa: BLE001 - the market is evidence, never a build dependency
        logger.warning("FantasyPros market rankings unavailable for %s", config.draft_season)
        return None
    if market.height == 0:
        logger.warning("no FantasyPros market snapshot for %s", config.draft_season)
        return None
    identity = crosswalk.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("name").alias("market_name"),
        pl.col("team").alias("market_team"),
    ).drop_nulls(subset=["player_id"])
    market = market.join(identity, on="player_id", how="left")
    logger.info(
        "market snapshot %s: %s ranked players", market["market_snapshot"][0], market.height
    )
    return market


def attach_market(board: pl.DataFrame, market: pl.DataFrame | None) -> pl.DataFrame:
    """Join consensus ranks onto rated rows and add market-only players as rows.

    A player the market ranks but the model cannot (no tape) enters the board with
    identity and market fields only; ``board.rank`` gives him a rank-matched value on
    the model's scale and tags the row ``rank_source = market``.
    """
    if market is None or market.height == 0:
        return board.with_columns(pl.lit(None, dtype=pl.Float64).alias("market_ecr"))
    market = market.filter(pl.col("market_position").is_in(["QB", "RB", "WR", "TE"]))
    joined = board.join(market.select("player_id", *MARKET_COLUMNS), on="player_id", how="left")
    missing = market.filter(~pl.col("player_id").is_in(board["player_id"].to_list()))
    if missing.height == 0:
        return joined
    extra = missing.select(
        "player_id",
        pl.col("market_name").alias("player_display_name"),
        pl.col("market_position").alias("position"),
        pl.col("market_team").alias("team"),
        pl.col("market_team").alias("projected_team"),
        *MARKET_COLUMNS,
    ).with_columns(
        pl.lit(0, dtype=pl.Int64).alias("games"),
        pl.lit(0.0).alias("override_delta"),
        pl.lit("").alias("flags"),
        pl.lit("v2").alias(METRIC_VERSION),
    )
    logger.info("market-only players added to the board: %s", extra.height)
    return pl.concat([joined, extra], how="diagonal_relaxed")


def _season_equivalent(
    board: pl.DataFrame,
    season_games: int,
    source_column: str = "fitted_season_points",
) -> pl.DataFrame:
    """Fitted season points as a per-scheduled-game rate; the waiver-comparison value."""
    source = (
        pl.col(source_column) / float(season_games)
        if source_column in board.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    return board.with_columns(source.alias("season_equivalent_ppg"))


def build_adaptive_board(
    board: pl.DataFrame,
    snapshot: pl.DataFrame,
    config: LeagueConfig,
    selected_sources: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Rank the live player pool with the immutable adaptive forecast for this season.

    Only the frozen adaptive score is joined from the prospective snapshot. Current
    identity, team, ownership, and explicitly labelled market-only fallback rows stay
    on the live board, so this behaves like the other selectable metric systems
    without refitting or mutating the forecast being prospectively graded.
    """
    season = config.draft_season
    frozen = (
        snapshot.filter(pl.col("forecast_season") == season)
        .select(
            "player_id",
            pl.col("fitted_adaptive_ppg_hybrid").alias("adaptive_season_points"),
        )
        .unique(subset=["player_id"])
    )
    if not frozen.height:
        raise ValueError(f"prospective snapshot has no adaptive forecast for {season}")
    sources = selected_sources or {}
    source_expr = pl.col("position").replace_strict(sources, default=None, return_dtype=pl.String)
    frame = board.join(frozen, on="player_id", how="left").with_columns(
        pl.lit("adaptive").alias(METRIC_VERSION),
        pl.lit("frozen_shadow").alias("adaptive_forecast_status"),
        pl.when(pl.col("adaptive_season_points").is_not_null())
        .then(source_expr)
        .otherwise(None)
        .alias("adaptive_selected_source"),
    )
    adaptive_config = config.model_copy(deep=True)
    adaptive_config.metrics.projection_rank_key = {
        position: "adaptive_season_points" for position in config.vor_baseline_rank
    }
    adaptive_config.metrics.projection_overall_key = "adaptive_season_points"
    ranked = apply_rank_key(frame, adaptive_config)
    return _season_equivalent(
        ranked,
        config.metrics.projection_season_games,
        source_column="adaptive_season_points",
    )


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
        logger.warning(
            "fitted ranker rejected: %s; positions fall back to the configured "
            "projection_rank_fallback (adj_proj_vor)",
            error,
        )
        return {}
    models = models_for_season(report.get("fitted_models") or [], forecast_season)
    if not models:
        logger.warning("metric report has no fitted models for %s", forecast_season)
    else:
        logger.info(
            "fitted ranker loaded for %s (%s): %s",
            forecast_season,
            report["fitted_artifact"]["production_fingerprint"],
            ", ".join(sorted(models)),
        )
    return models


def load_draft_depth_chart(
    config: LeagueConfig,
    report_config: MetricReportConfig,
    snapshot_date: str | None = None,
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
    cutoff = str(snapshot_date)[:10][5:] if snapshot_date else report_config.depth_chart_cutoff
    selected = depth_chart_as_of(depth, config.draft_season, cutoff)
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
    include_experiments: bool = False,
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
    # Build from the same dated pending-fold inputs that the approved artifact records.
    # A newer cached depth chart does not silently change the production forecast.
    snapshot_date = None
    model_path = settings.outputs_dir / "production_model.json"
    if model_path.exists():
        artifact = json.loads(model_path.read_text()).get("fitted_artifact")
        try:
            check_fit_artifact(
                artifact,
                report_config.fit,
                config.draft_season,
                report_config.depth_chart_cutoff,
                None,
            )
            snapshot_date = artifact.get("depth_chart_latest")
        except FittedArtifactError:
            pass  # load_fitted_models reports the rejection; the board is marked degraded.
    depth_chart = load_draft_depth_chart(config, report_config, snapshot_date=snapshot_date)
    live_depth_latest = (
        depth_chart["depth_chart_date"].cast(pl.String).max() if depth_chart is not None else None
    )
    live_depth_latest = str(live_depth_latest) if live_depth_latest is not None else None
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
    board_v2 = apply_fitted_models(
        board_v2.with_columns(pl.lit(1.0).alias("returning_indicator")),
        load_fitted_models(
            settings.outputs_dir / "production_model.json",
            config.draft_season,
            report_config,
            live_depth_latest,
        ),
        production_config(report_config.fit),
    )
    # Consensus ranks join every row; players the model cannot rate but the market
    # ranks are added as rows and given a rank-matched value inside apply_rank_key.
    board_v2_base = attach_market(board_v2, load_draft_market(config, report_config))
    board_adaptive = pl.DataFrame()
    if include_experiments:
        snapshot, freeze = load_verified_snapshot(
            settings.static_dir / f"experimental_{config.draft_season}_predictions.csv",
            CONFIG_DIR / f"experimental_freeze_{config.draft_season}.yaml",
        )
        selected_sources = freeze["selectors"]["fitted_adaptive_ppg_hybrid"][
            "selected_source_by_position"
        ]
        board_adaptive = build_adaptive_board(
            board_v2_base,
            snapshot,
            config,
            selected_sources={str(k): str(v) for k, v in selected_sources.items()},
        )
    board_v2 = _season_equivalent(
        apply_rank_key(board_v2_base, config), config.metrics.projection_season_games
    )

    as_of = f"{config.draft_season}-{report_config.depth_chart_cutoff}"
    board_v2 = attach_forecast(board_v2, as_of)
    if board_adaptive.height:
        board_adaptive = attach_forecast(board_adaptive, str(freeze["frozen_on"]))
    archive = settings.static_dir / f"draft_{config.draft_season}"
    if archive.exists():
        board = pl.read_json(verify_draft(archive))

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
        board_adaptive=board_adaptive,
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
    *,
    analyze: bool = True,
    enrichment_dir: Path | None = None,
) -> MetricReportBuildResult:
    """Build historical v2 forecast folds and analyze the configured metric catalog."""
    config = config or get_league()
    settings = settings or get_settings()
    report_config = report_config or MetricReportConfig.from_config()
    settings.ensure_dirs()

    def retain(name: str, frame: pl.DataFrame) -> None:
        if enrichment_dir is not None:
            enrichment_dir.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(enrichment_dir / f"{name}.parquet")

    history_config = config.model_copy(update={"seasons": list(report_config.input_seasons)})
    roster_seasons = [season for season in history_config.seasons if season >= 2004]
    roster_rows = canonical_positions(nflverse.load_weekly_rosters(roster_seasons))
    positions = cached_frame(
        "metric_report_position_history_v1",
        roster_seasons,
        lambda: historical_positions(roster_rows),
        force=force,
        settings=settings,
    )
    weeks = repair_player_week_positions(
        nflverse.load_player_weeks(history_config.seasons),
        positions,
        overrides=history_config.position_overrides,
    )
    bonuses, _ = load_bonuses(history_config, force=force, settings=settings)
    retain(
        "nfl_player_weeks",
        weeks.join(
            bonuses, on=["player_id", "season", "week"], how="left", validate="1:1"
        ).with_columns(
            (pl.col("fantasy_points_ppr") + pl.col("bonus_pts").fill_null(0)).alias("league_points")
        ),
    )
    # Eligibility defines the candidate pool, not which earned points survive.
    # A player switching to LB/DB/LS can still earn offensive/return fantasy points.
    # Aggregate all rows first, then attach the season's observed skill position.
    outcome_seasons = build_player_seasons(
        weeks,
        bonuses,
        history_config.model_copy(
            update={
                "board_positions": weeks["position"].drop_nulls().unique().to_list(),
            }
        ),
    )
    eligible_positions = (
        weeks.filter(pl.col("position").is_in(history_config.board_positions))
        .group_by("player_id", "season")
        .agg(pl.col("position").last())
    )
    player_seasons = outcome_seasons.drop("position").join(
        eligible_positions,
        on=["player_id", "season"],
        how="inner",
        validate="1:1",
    )
    team_weeks = nflverse.load_team_weeks(history_config.seasons)
    team_volume = build_team_volume(team_weeks)
    participation_seasons = [season for season in history_config.seasons if season >= 2016]
    projection_plays = nflverse.load_projection_plays(history_config.seasons)
    participation = nflverse.load_participation_flexible(participation_seasons)
    usage = cached_frame(
        "metric_report_usage_v5",
        history_config.seasons,
        lambda: build_player_usage(
            weeks,
            projection_plays,
            participation,
        ),
        force=force,
        settings=settings,
    )
    injury_seasons = [season for season in history_config.seasons if season >= 2009]
    injury_rows = nflverse.load_injuries(injury_seasons)
    injuries = build_injury_history(injury_rows)
    expected_seasons = [season for season in history_config.seasons if season >= 2006]
    expected_rows = nflverse.load_expected_opportunity(expected_seasons)
    expected_opportunity = cached_frame(
        "metric_report_expected_opportunity_v1",
        expected_seasons,
        lambda: build_expected_opportunity(expected_rows, weeks),
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
    outcome_seasons = normalize_ppg_for_active_games(
        outcome_seasons.join(
            usage.select("player_id", "season", "active_games"),
            on=["player_id", "season"],
            how="left",
        )
    )

    # The expected-opportunity feed has no historical model-publication vintages.
    # In particular, its 2006-2020 training window overlaps early test folds.
    # Keep the raw cache, but quarantine modeled xFP from this rebuilt experiment.
    player_seasons = player_seasons.with_columns(
        *(
            pl.lit(None, dtype=pl.Float64).alias(name)
            for name in (
                "xfp_pg",
                "xfp_total",
                "expected_first_downs_pg",
                "expected_first_downs_total",
            )
            if name in player_seasons.columns
        )
    )

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
        if season < max(2025, report_config.depth_chart_start_season):
            continue
        try:
            depth_frames.append(nflverse.load_depth_charts([season]))
        except Exception:  # noqa: BLE001 - projection falls back cleanly without depth charts
            logger.warning("depth charts unavailable for %s", season)
    depth_charts = pl.concat(depth_frames, how="diagonal_relaxed") if depth_frames else None

    # Report-only additive evidence. These feeds are deliberately assembled outside
    # the production projection so an unsuccessful experiment cannot change the live
    # board. Dated transactions drive the canonical August cutoff roster fields;
    # historical Week 1 rosters remain separate sensitivity-only proxy columns.
    players = nflverse.load_players()
    identity_crosswalk = nflverse.load_id_crosswalk()
    transaction_players = supplement_identity_names(players, identity_crosswalk)
    retain("nfl_players", transaction_players)
    retain("nfl_identity_crosswalk", identity_crosswalk)
    retain("nfl_player_seasons", player_seasons)
    retain("nfl_team_seasons", team_volume)
    retain("nfl_injuries", injury_rows)
    market_rankings = build_market_rankings(
        nflverse.load_fantasy_rankings(),
        identity_crosswalk,
        report_config.depth_chart_cutoff,
    )
    market_cutoffs: dict[int, date] = {}
    if market_rankings.height:
        snapshot_column = (
            "market_overall_snapshot"
            if "market_overall_snapshot" in market_rankings.columns
            else "market_snapshot"
        )
        for row in (
            market_rankings.select("forecast_season", snapshot_column)
            .drop_nulls()
            .unique(subset=["forecast_season"], keep="first")
            .to_dicts()
        ):
            market_cutoffs[int(row["forecast_season"])] = date.fromisoformat(
                str(row[snapshot_column])
            )
    transaction_seasons = list(report_config.forecast_seasons)
    transactions = cached_frame(
        "nfl_transactions_cutoff_v1",
        transaction_seasons,
        lambda: nfl_transactions.load_transactions(
            transaction_seasons, report_config.depth_chart_cutoff
        ),
        force=force,
        settings=settings,
    )
    official_seasons = sorted(season for season in market_cutoffs if season >= 2020)
    if official_seasons:
        official_transactions = cached_frame(
            "nfl_official_transactions_cutoff_v1",
            official_seasons,
            lambda: nfl_transactions.load_official_transactions(
                official_seasons, report_config.depth_chart_cutoff
            ),
            force=force,
            settings=settings,
        )
        if official_transactions.height:
            transactions = pl.concat(
                [
                    normalize_transaction_sources(transactions),
                    normalize_transaction_sources(official_transactions),
                ],
                how="diagonal_relaxed",
            ).sort(["transaction_date", "to_team", "description"])
    backfill = load_transaction_backfill(settings.data_dir)
    if backfill is not None:
        transactions = pl.concat([transactions, backfill], how="diagonal_relaxed").unique(
            subset=["transaction_date", "source_team", "description", "category"]
        )
    combine_features = build_combine_features(nflverse.load_combine(), players)
    retain("nfl_transactions", transactions)
    rookie_features = build_rookie_features(
        players,
        combine_features,
        report_config.forecast_seasons,
        market_rankings=market_rankings,
    )
    candidate_frames = [
        rookie_features.select("forecast_season", "player_id", "player_display_name", "position")
    ]
    if market_rankings.height:
        candidate_frames.append(
            market_rankings.select(
                "forecast_season", "player_id", pl.col("market_position").alias("position")
            ).join(
                transaction_players.select(
                    pl.col("gsis_id").alias("player_id"),
                    pl.col("display_name").alias("player_display_name"),
                ),
                on="player_id",
                how="left",
                validate="m:1",
            )
        )
    forecast_candidates = pl.concat(candidate_frames, how="diagonal_relaxed")
    transaction_features = build_transaction_features(
        transactions,
        player_seasons,
        transaction_players,
        report_config.forecast_seasons,
        report_config.depth_chart_cutoff,
        cutoff_by_season=market_cutoffs,
        trusted_sources_only=True,
        forecast_candidates=forecast_candidates,
    )
    snap_seasons = [season for season in history_config.seasons if season >= 2013]
    snap_rows = canonical_positions(nflverse.load_snap_counts(snap_seasons))
    retain("nfl_snap_counts", snap_rows)
    snap_features = cached_frame(
        "metric_report_snap_features_v2",
        snap_seasons,
        lambda: build_snap_features(snap_rows, players),
        force=force,
        settings=settings,
    )
    experimental_features = build_forecast_features(
        roster_rows,
        player_seasons,
        report_config.forecast_seasons,
        report_config.depth_chart_cutoff,
        depth_charts=depth_charts,
        snap_features=snap_features,
        player_background=build_player_background(players),
        contract_features=build_contract_features(
            nflverse.load_contracts(),
            report_config.forecast_seasons,
            cutoff_by_season=market_cutoffs,
            cutoff=report_config.depth_chart_cutoff,
        ),
        combine_features=combine_features,
        transaction_features=transaction_features,
        cutoff_by_season=market_cutoffs,
    )

    predictions = build_backtest_predictions(
        player_seasons,
        birth_dates,
        config,
        report_config,
        depth_charts=depth_charts,
        market_rankings=market_rankings,
        experimental_features=experimental_features,
        rookie_features=rookie_features,
        cutoff_by_season=market_cutoffs,
        outcome_seasons=outcome_seasons,
    )
    predictions = attach_roster_evidence(predictions, transaction_features, transaction_players)
    rich_source_seasons = [season for season in history_config.seasons if season >= 2013]
    rich_panel = cached_frame(
        "metric_report_rich_weekly_panel_v3",
        history_config.seasons,
        lambda: build_rich_weekly_panel(
            weeks,
            team_weeks,
            snap_rows,
            players,
            injury_rows.filter(pl.col("season").is_in(rich_source_seasons)),
            roster_rows.filter(pl.col("season").is_in(rich_source_seasons)),
            expected_rows,
            build_weekly_route_panel(weeks, projection_plays, participation),
        ),
        force=force,
        settings=settings,
    )
    rich_features = cached_frame(
        f"metric_report_rich_weekly_features_v4_h{report_config.history_seasons}",
        list(report_config.forecast_seasons),
        lambda: build_rich_weekly_features(
            rich_panel.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("xfp"),
                pl.lit(None, dtype=pl.Float64).alias("expected_td"),
                pl.lit(False).alias("xfp_data_available"),
            ),
            predictions.select("forecast_season", "player_id"),
            history_seasons=report_config.history_seasons,
        ),
        force=force,
        settings=settings,
    )
    retain(
        "nfl_weekly_usage",
        rich_panel.drop("xfp", "expected_td", "xfp_data_available"),
    )
    predictions = predictions.join(
        rich_features,
        on=["forecast_season", "player_id"],
        how="left",
        validate="m:1",
    )
    predictions = attach_known_absences(
        predictions,
        load_absences(settings.static_dir / EVIDENCE_FILE),
        cutoff_by_season=market_cutoffs,
        default_cutoff=report_config.depth_chart_cutoff,
    )
    retain("historical_inputs", predictions)
    if not analyze:
        return MetricReportBuildResult(report={}, predictions=predictions)
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
    production_only: bool = False,
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
    predictions = pl.read_parquet(path)
    if production_only:
        # Research columns cannot become incidental inputs to production evaluation.
        columns = {
            "forecast_season",
            "source_season",
            "player_id",
            "player_display_name",
            "position",
            "player_population",
            "returning_indicator",
            "rookie_indicator",
            "outcome_complete",
            "actual_matched",
            "depth_chart_date",
            "draft_pool_version",
            "games",
            "ppg",
            "season_pts",
            "ppg_denominator_games",
            "proj_ppg",
            "adj_proj_vor",
            "market_ecr",
            "market_position",
            "market_ecr_score",
            "market_overall_ecr_score",
            "market_snapshot",
            *(target.key for target in report_config.targets),
            *(feature for spec in report_config.fit.models for feature in spec.features),
        }
        predictions = predictions.select(sorted(columns & set(predictions.columns)))
    report, predictions = _analyze_with_fit(predictions, report_config)
    return MetricReportBuildResult(report=report, predictions=predictions)


def export_metric_report(
    result: MetricReportBuildResult, outputs: Path, stem: str = "metric_report"
) -> tuple[Path, Path, Path]:
    """Write JSON for the API, Markdown for humans, and detailed forecast rows."""
    json_path = outputs / f"{stem}.json"
    markdown_path = outputs / f"{stem}.md"
    predictions_path = outputs / (
        "metric_backtest_predictions.parquet"
        if stem == "metric_report"
        else f"{stem}_predictions.parquet"
    )
    atomic_write(json_path, json.dumps(result.report, indent=2, default=str))
    artifact = result.report.get("fitted_artifact")
    if artifact:
        from patron.metrics.fit import PRODUCTION_OUTPUTS

        production = {
            "fitted_artifact": {
                key: value
                for key, value in artifact.items()
                if key not in {"fingerprint", "fit_config", "outputs", "created_at", "pending_rows"}
            },
            "fitted_models": [
                m
                for m in result.report.get("fitted_models", [])
                if m["model"] in PRODUCTION_OUTPUTS
            ],
        }
        atomic_write(
            outputs / "production_model.json", json.dumps(production, indent=2, default=str)
        )
    markdown_path.write_text(render_metric_report_markdown(result.report))
    result.predictions.write_parquet(predictions_path)
    return json_path, markdown_path, predictions_path


#: Board columns written to JSON, in display order.
BOARD_EXPORT_COLUMNS: tuple[str, ...] = (
    *FORECAST_COLUMNS,
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
    "target_share",
    "air_yards_share",
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
    "rank_source",
    "market_position",
    "market_ecr",
    "market_ecr_sd",
    "market_snapshot",
    "v2_rank_key",
    "v2_rank_value",
    "v2_rank_vor",
    "adaptive_season_points",
    "adaptive_selected_source",
    "adaptive_forecast_status",
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
    "team_scoring_context",
    "team_pass_volume",
    "team_dropbacks",
    "team_rush_volume",
    "season_target_share",
    "season_carry_share",
    "projected_target_share",
    "projected_carry_share",
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
    "air_yard_factor",
    "opportunity_multiplier",
    "projection_reason",
    "historical_vor",
    "historical_repl_ppg",
)


def export_board(board: pl.DataFrame, path: Path) -> Path:
    """Write the board as JSON records, rounded for display."""
    columns = [column for column in BOARD_EXPORT_COLUMNS if column in board.columns]
    preserved = set(FORECAST_COLUMNS)
    if "forecast_source" in board.columns:
        preserved.update(
            feature
            for spec in production_config(MetricReportConfig.from_config().fit).models
            for feature in spec.features
        )
    rounded = [
        name for name in columns if board.schema[name] == pl.Float64 and name not in preserved
    ]
    export = board.select(columns).with_columns(pl.col(rounded).round(3))
    atomic_write(path, json.dumps(export.to_dicts(), indent=2, default=str))
    return path


def export_markdown(board: pl.DataFrame, path: Path, limit: int | None = None) -> Path:
    """Write the board in the same shape as `data/static/2026_draft_list.md`.

    Same columns and same order as the artifact this pipeline reproduces, so the two
    can be diffed by eye as well as by the fixture test.
    """
    rows = board.head(limit) if limit else board
    is_v2 = "proj_ppg" in rows.columns
    version = "v2 projected" if is_v2 else "v1 historical"
    vor_column = "v2_overall_vor" if is_v2 else "adj_vor"
    ppg_column = (
        "forecast_active_ppg"
        if "forecast_active_ppg" in rows.columns
        else ("proj_ppg" if is_v2 else "ppg")
    )
    team_column = "projected_team" if is_v2 else "team"
    lines = [
        f"# Overall Draft Board ({version}) — {rows.height} players",
        "",
        f"Rank | Player | Pos | Team | {'Overall VOR' if is_v2 else 'VOR'} | PPG | Flags | Notes",
        "---|---|---|---|---|---|---|---",
    ]
    for row in rows.iter_rows(named=True):
        note = row.get("override_reason") or ""
        vor_value = row.get(vor_column)
        ppg_value = row.get(ppg_column)
        source = " (market)" if row.get("rank_source") == "market" else ""
        lines.append(
            f"{row['rank']} | {row['player_display_name']}{source} | {row['position']} | "
            f"{row.get(team_column) or ''} | "
            f"{'' if vor_value is None else f'{vor_value:.1f}'} | "
            f"{'' if ppg_value is None else f'{ppg_value:.1f}'} | "
            f"{row.get('flags') or ''} | {note}"
        )
    lines.append("")
    lines.append("Flags: age=RB 27.5+ | Xgms=small sample.")
    path.write_text("\n".join(lines))
    return path
