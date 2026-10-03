"""Per-game rate, floor, and volatility.

PPG is the base currency of the whole board. Season totals mislead the moment games
are missed — a back who was excellent in eight games and a back who was mediocre in
seventeen can land on the same total — so everything downstream sorts on per-game
production and carries the games count beside it.

Floor and volatility come off the weekly distribution. Floor decides who starts;
volatility gets bought deliberately, not accidentally — ceiling when you need to
outscore a favourite, floor when you are the favourite.

All three are computed on `league_pts`, which includes big-play bonuses. The
draft-prep script computed PPG on bonus-inclusive points but floor and volatility on
bonus-exclusive ones, so the three numbers on a row were not describing the same
scoring system. In a league whose edge is the bonus, that understated exactly the
players the edge was meant to find.
"""

from __future__ import annotations

import polars as pl

from engine.scoring.columns import require_columns
from engine.scoring.engine import LEAGUE_POINTS


def weekly_rates(
    weeks: pl.DataFrame,
    floor_quantile: float = 0.25,
    points_column: str = LEAGUE_POINTS,
    group_by: tuple[str, ...] = ("player_id", "season"),
) -> pl.DataFrame:
    """Summarise weekly scoring into games, total, PPG, floor, and volatility.

    Args:
        weeks: Player-week rows carrying `points_column`.
        floor_quantile: Quantile treated as the floor. 0.25 by default — the level a
            player clears in three weeks out of four.
        points_column: Scoring column to summarise. Defaults to `league_pts`; passing
            anything bonus-exclusive here is the defect this module exists to fix.
        group_by: Grouping keys.

    Returns:
        One row per group with `games`, `total_pts`, `ppg`, `floor`, `volatility`.
        `volatility` is null for a single-game sample, where standard deviation is
        undefined — null rather than zero, because "unknown" and "perfectly
        consistent" must not read the same on the board.
    """
    require_columns(weeks.columns, (points_column,), "weekly_rates input")

    points = pl.col(points_column).fill_null(0.0)
    return (
        weeks.group_by(list(group_by))
        .agg(
            pl.len().alias("games"),
            points.sum().alias("total_pts"),
            points.mean().alias("ppg"),
            points.quantile(floor_quantile, interpolation="linear").alias("floor"),
            points.std().alias("volatility"),
        )
        .sort(list(group_by))
    )
