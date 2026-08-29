"""Board assembly: player-weeks in, ranked board out.

Two stages, both pure.

`build_player_seasons` scores weekly rows under league rules and aggregates them into
one row per player per season, carrying the §4 metric catalog. This is the reusable
artifact — in-season work (Phase 4) will want the same table on a rolling window.

`build_board` takes one season of that and turns it into a ranked board: age, VOR
against recomputed replacement levels, flags, manual overrides, rank.

The split matters because replacement level is a property of a season's player pool,
not of a player. It has to be recomputed whenever the pool changes, which in-season is
constantly — a re-rank is honest about what is actually available on the wire, where a
fixed preseason baseline is not.
"""

from __future__ import annotations

import logging

import polars as pl

from patron.board.flags import add_flags
from patron.board.overrides import ADJUSTED_VOR, OverrideSet, apply_overrides
from patron.config.league import LeagueConfig, get_league
from patron.metrics.age import add_age, add_age_flags
from patron.metrics.opportunity import aggregate_opportunity
from patron.metrics.rates import weekly_rates
from patron.metrics.regression import TOTAL_TDS, add_regression_flags, add_td_over_expectation
from patron.metrics.vor import VOR, add_vor, replacement_levels
from patron.scoring.bonuses import BONUS_POINTS
from patron.scoring.engine import (
    BASE_POINTS,
    LEAGUE_POINTS,
    add_league_points,
    base_points_from_nflverse,
)

logger = logging.getLogger(__name__)

RANK_COLUMN = "rank"

_SEASON_KEYS = ("player_id", "season")


def build_player_seasons(
    weeks: pl.DataFrame,
    bonuses: pl.DataFrame,
    config: LeagueConfig | None = None,
) -> pl.DataFrame:
    """Aggregate scored player-weeks into one row per player-season.

    Args:
        weeks: Player-week rows from nflverse, regular season only.
        bonuses: Per-player, per-week big-play bonuses from
            `patron.scoring.bonuses.extract_touchdown_bonuses`.
        config: League config; defaults to the configured league.

    Returns:
        One row per player-season with league points, PPG, floor, volatility,
        opportunity metrics, touchdown-over-expectation, and season bonus totals.
    """
    config = config or get_league()

    skill = weeks.filter(pl.col("position").is_in(config.board_positions))

    # Bonuses join per week, not per season. That is what lets weekly league points —
    # and so the floor and volatility computed from them — include the big-play bonus
    # that this league's edge depends on.
    scored = (
        skill.join(bonuses, on=["season", "week", "player_id"], how="left")
        .pipe(base_points_from_nflverse)
        .pipe(add_league_points)
    )

    rates = weekly_rates(
        scored, floor_quantile=config.metrics.floor_quantile, group_by=_SEASON_KEYS
    )
    opportunity = aggregate_opportunity(
        scored, target_weight=config.metrics.target_weight, group_by=_SEASON_KEYS
    )

    identity = scored.group_by(list(_SEASON_KEYS)).agg(
        pl.col("player_display_name").last().alias("player_display_name"),
        pl.col("position").last().alias("position"),
        # Last team of the season, which is the one that matters for next year.
        pl.col("team").last().alias("team"),
        pl.col("rushing_tds").fill_null(0).sum().alias("rushing_tds"),
        pl.col("receiving_tds").fill_null(0).sum().alias("receiving_tds"),
        pl.col("passing_tds").fill_null(0).sum().alias("passing_tds"),
        pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
        pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
        pl.col("receiving_yards").fill_null(0).sum().alias("receiving_yards"),
        pl.col(BASE_POINTS).sum().alias(BASE_POINTS),
        pl.col(BONUS_POINTS).fill_null(0).sum().alias(BONUS_POINTS),
        pl.col(LEAGUE_POINTS).sum().alias("season_pts"),
    )

    seasons = (
        identity.join(rates, on=list(_SEASON_KEYS), how="left")
        .join(opportunity, on=list(_SEASON_KEYS), how="left")
        # Only rushing and receiving touchdowns regress against a per-touch baseline.
        # Passing touchdowns are a different skill and are excluded on both sides.
        .with_columns((pl.col("rushing_tds") + pl.col("receiving_tds")).alias(TOTAL_TDS))
    )

    return add_td_over_expectation(seasons, window=config.metrics.td_rate_window)


def build_board(
    player_seasons: pl.DataFrame,
    birth_dates: pl.DataFrame,
    config: LeagueConfig | None = None,
    override_set: OverrideSet | None = None,
    strict_overrides: bool = True,
) -> pl.DataFrame:
    """Turn one season of player-seasons into the ranked draft board.

    Args:
        player_seasons: Output of `build_player_seasons`.
        birth_dates: `player_id` to `birth_date`, for the age curve.
        config: League config; defaults to the configured league.
        override_set: Manual overrides; loaded from config by default.
        strict_overrides: Fail when an override matches nobody.

    Returns:
        The board, sorted by override-adjusted VOR, with a 1-based `rank`.
    """
    config = config or get_league()
    metrics = config.metrics

    season = (
        player_seasons.filter(
            (pl.col("season") == config.board_season) & (pl.col("games") >= config.min_games_board)
        )
        .join(birth_dates, on="player_id", how="left")
        .pipe(add_age, season=config.draft_season)
        .pipe(add_age_flags, rb_cliff=metrics.rb_age_cliff)
        .pipe(
            add_regression_flags,
            regress_down_at=metrics.td_regress_down,
            regress_up_at=metrics.td_regress_up,
            regress_up_min_opportunity=metrics.td_regress_up_min_opp,
        )
    )

    # Replacement level is set by the season pool, from players with enough games to
    # count as reliably available. Small-sample players still appear on the board —
    # they just do not get to define the bar.
    levels = replacement_levels(
        season,
        baseline_ranks=config.vor_baseline_rank,
        min_games=config.min_games_baseline,
    )
    logger.info(
        "replacement levels: %s",
        ", ".join(f"{position} {ppg:.1f}" for position, ppg in sorted(levels.items())),
    )

    board = (
        add_vor(season, levels)
        .pipe(add_flags, small_sample_games=config.small_sample_games)
        .pipe(apply_overrides, override_set=override_set, strict=strict_overrides)
        .filter(pl.col(VOR).is_not_null())
        .sort(ADJUSTED_VOR, descending=True, nulls_last=True)
    )

    return board.with_columns(pl.int_range(1, board.height + 1, dtype=pl.Int32).alias(RANK_COLUMN))


def board_summary(board: pl.DataFrame) -> dict[str, object]:
    """Headline counts for logs and the API status endpoint."""
    return {
        "players": board.height,
        "by_position": dict(
            board.group_by("position")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .iter_rows()
        ),
        "flagged": int(board.filter(pl.col("flags") != "").height),
        "overridden": int(board.filter(pl.col("override_delta") != 0).height),
    }
