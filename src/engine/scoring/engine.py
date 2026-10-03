"""The component scorer: league points computed from raw stat components.

Two paths to base scoring exist and they are both here on purpose.

`score_components` builds points up from the stat line — yards, touchdowns,
receptions, turnovers — using only the rules in scoring.yaml. It is slower and it is
the auditable one: if the league changes a setting, this path follows automatically
and you can point at the arithmetic.

`base_points_from_nflverse` takes nflverse's precomputed `fantasy_points_ppr`. The
draft-prep script relied on this shortcut alone, on the observation that this league's
base scoring happens to match nflverse's PPR exactly. That observation is load-bearing
and undertested, so `tests/test_parity.py` asserts the two paths agree on real data.
The day they diverge is the day a league setting drifted, and we want a failing test
rather than a quietly wrong board.

Every function here is pure: DataFrame in, DataFrame out, no network and no disk.
"""

from __future__ import annotations

import polars as pl

from engine.scoring.columns import (
    NFLVERSE_POINTS_COLUMN,
    SCORING_COLUMNS,
    require_columns,
)
from engine.scoring.rules import ScoringRules, get_rules

#: Column written by `score_components`.
COMPONENT_POINTS = "component_pts"

#: Column written by `base_points_from_nflverse`.
BASE_POINTS = "base_pts"

#: Column written by `add_league_points`: base scoring plus big-play bonuses.
LEAGUE_POINTS = "league_pts"

#: Column holding big-play bonus points, produced by `engine.scoring.bonuses`.
BONUS_POINTS = "bonus_pts"


def score_components(
    stats: pl.DataFrame,
    rules: ScoringRules | None = None,
) -> pl.DataFrame:
    """Add `component_pts`: league points built from the stat line.

    Covers everything except the big-play distance bonuses, which need touchdown
    yardage that weekly stats do not carry — see `engine.scoring.bonuses`.

    Args:
        stats: Player-week (or player-season) rows carrying `SCORING_COLUMNS`.
        rules: Scoring rules; defaults to the league's configured rules.

    Returns:
        `stats` with a `component_pts` column appended.
    """
    rules = rules or get_rules()
    require_columns(stats.columns, SCORING_COLUMNS, "score_components input")

    def col(name: str) -> pl.Expr:
        # Weekly rows are null rather than zero for stats a player had no chance to
        # accrue — a receiver has no passing_yards. Nulls would poison the sum.
        return pl.col(name).fill_null(0)

    passing = (
        col("passing_yards") * rules.passing.points_per_yard
        + col("passing_tds") * rules.passing.touchdown
        + col("passing_interceptions") * rules.passing.interception
        + col("passing_2pt_conversions") * rules.passing.two_point_conversion
    )

    rushing = (
        col("rushing_yards") * rules.rushing.points_per_yard
        + col("rushing_tds") * rules.rushing.touchdown
        + col("rushing_2pt_conversions") * rules.rushing.two_point_conversion
    )

    receiving = (
        col("receptions") * rules.receiving.points_per_reception
        + col("receiving_yards") * rules.receiving.points_per_yard
        + col("receiving_tds") * rules.receiving.touchdown
        + col("receiving_2pt_conversions") * rules.receiving.two_point_conversion
    )

    fumbles = (
        col("rushing_fumbles_lost") + col("receiving_fumbles_lost") + col("sack_fumbles_lost")
    ) * rules.misc.fumble_lost

    # Return touchdowns. Not in the design doc's summary of the league sheet, but
    # nflverse pays them and the parity test proved this league does too.
    special_teams = col("special_teams_tds") * rules.misc.special_teams_touchdown

    return stats.with_columns(
        (passing + rushing + receiving + fumbles + special_teams).alias(COMPONENT_POINTS)
    )


def base_points_from_nflverse(stats: pl.DataFrame) -> pl.DataFrame:
    """Add `base_pts` from nflverse's precomputed `fantasy_points_ppr`.

    The fast path. Verified against `score_components` by `tests/test_parity.py`.
    """
    require_columns(stats.columns, (NFLVERSE_POINTS_COLUMN,), "base_points_from_nflverse input")
    return stats.with_columns(pl.col(NFLVERSE_POINTS_COLUMN).fill_null(0.0).alias(BASE_POINTS))


def add_league_points(
    stats: pl.DataFrame,
    base_column: str = BASE_POINTS,
    bonus_column: str = BONUS_POINTS,
) -> pl.DataFrame:
    """Add `league_pts` = base scoring + big-play bonuses.

    Rows with no bonus (the overwhelming majority) carry a null after the bonus join;
    they score zero bonus, not null points.
    """
    require_columns(stats.columns, (base_column,), "add_league_points input")

    bonus = pl.col(bonus_column).fill_null(0.0) if bonus_column in stats.columns else pl.lit(0.0)
    return stats.with_columns((pl.col(base_column).fill_null(0.0) + bonus).alias(LEAGUE_POINTS))
