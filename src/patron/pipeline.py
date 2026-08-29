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

import polars as pl

from patron.board.builder import build_board, build_player_seasons
from patron.config.league import LeagueConfig, get_league
from patron.config.settings import Settings, get_settings
from patron.data import nflverse
from patron.data.derived import cached_frame
from patron.metrics.enrichment import (
    build_injury_history,
    build_player_usage,
    build_team_volume,
    current_depth_chart,
    enrich_player_seasons,
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


def build(
    config: LeagueConfig | None = None,
    settings: Settings | None = None,
    force: bool = False,
    strict_overrides: bool = True,
) -> BuildResult:
    """Run the full Phase 1 pipeline and return every artifact it produced."""
    config = config or get_league()
    settings = settings or get_settings()
    settings.ensure_dirs()

    weeks = nflverse.load_player_weeks(config.seasons)
    bonuses, audit = load_bonuses(config, force=force)

    player_seasons = build_player_seasons(weeks, bonuses, config)
    team_volume = build_team_volume(nflverse.load_team_weeks(config.seasons))
    usage = cached_frame(
        "projection_usage_v2",
        config.seasons,
        lambda: build_player_usage(
            weeks,
            nflverse.load_projection_plays(config.seasons),
            nflverse.load_participation(config.seasons),
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
    depth_chart = current_depth_chart(nflverse.load_depth_charts([config.board_season]))
    birth_dates = nflverse.load_birth_dates(config.board_season)
    board = build_board(
        player_seasons,
        birth_dates,
        config=config,
        strict_overrides=strict_overrides,
    ).with_columns(pl.lit("v1").alias(METRIC_VERSION))
    board_v2 = build_projection_board(
        player_seasons,
        board,
        config,
        current_players=depth_chart,
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
        player_seasons=player_seasons,
        kickers=kickers,
        defenses=defenses,
        bonus_audit=audit,
        replacement_levels=levels,
    )


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
    "v2_score",
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
    "floor_vor",
    "ceiling_vor",
    "projection_confidence",
    "availability_confidence",
    "projected_availability",
    "availability_factor",
    "expected_games",
    "expected_season_points",
    "season_equivalent_ppg",
    "availability_adjusted_vor",
    "historical_injury_report_weeks",
    "injury_missed_equivalents",
    "effective_games",
    "age_factor",
    "td_regression_adjustment",
    "bonus_regression_adjustment",
    "team_context_factor",
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
    vor_column = "v2_score" if is_v2 else "adj_vor"
    ppg_column = "proj_ppg" if is_v2 else "ppg"
    team_column = "projected_team" if is_v2 else "team"
    lines = [
        f"# Overall Draft Board ({version}) — {rows.height} players",
        "",
        f"Rank | Player | Pos | Team | {'V2 Score' if is_v2 else 'VOR'} | PPG | Flags | Notes",
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
