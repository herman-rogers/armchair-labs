"""Volume metrics: the sticky, predictive half of production.

Efficiency regresses and volume persists, so opportunity is what a projection should
lean on. Weighted opportunity — carries plus 2.2x targets — puts a running back's
touches and a receiver's targets on one scale, weighted by what a target is actually
worth in full PPR.

Target share, air-yards share, and WOPR are nflverse's own receiver-opportunity
columns. Air-yards share carries extra weight in this league specifically: deep
targets are what feed the 40+/50+ bonus brackets.

These weekly shares are aggregated **weighted by team pass attempts**, not averaged
flat. A flat mean lets a two-target week spent mostly on the sideline count as heavily
as a twelve-target week, which systematically understates players who missed time —
the same failure mode that makes season totals misleading.
"""

from __future__ import annotations

import polars as pl

from patron.scoring.columns import OPPORTUNITY_COLUMNS, require_columns

WEIGHTED_OPPORTUNITY = "wtd_opp"

#: Share columns aggregated with target-weighting rather than a flat mean.
_SHARE_COLUMNS: tuple[str, ...] = OPPORTUNITY_COLUMNS


def _weighted_share(column: str, weight: pl.Expr) -> pl.Expr:
    """Usage-weighted mean of a weekly share column, ignoring weeks with no usage."""
    contribution = (pl.col(column).fill_null(0.0) * weight).sum()
    total_weight = pl.when(pl.col(column).is_not_null()).then(weight).otherwise(0.0).sum()
    return (contribution / total_weight.replace(0.0, None)).alias(column)


def aggregate_opportunity(
    weeks: pl.DataFrame,
    target_weight: float = 2.2,
    group_by: tuple[str, ...] = ("player_id", "season"),
) -> pl.DataFrame:
    """Aggregate weekly volume into per-season opportunity metrics.

    Args:
        weeks: Player-week rows carrying carries, targets, and the share columns.
        target_weight: What one target is worth relative to one carry. 2.2 in full PPR.
        group_by: Grouping keys.

    Returns:
        One row per group with raw volume columns, `wtd_opp`, and usage-weighted
        `target_share`, `air_yards_share`, and `wopr`.
    """
    required = ("carries", "targets", "receptions", *_SHARE_COLUMNS)
    require_columns(weeks.columns, required, "aggregate_opportunity input")

    # Weight shares by the player's own weekly involvement. Targets are the natural
    # weight for receiving shares: a week with no targets should not vote.
    weight = pl.col("targets").fill_null(0.0)

    return (
        weeks.group_by(list(group_by))
        .agg(
            pl.col("carries").fill_null(0).sum().alias("carries"),
            pl.col("targets").fill_null(0).sum().alias("targets"),
            pl.col("receptions").fill_null(0).sum().alias("receptions"),
            *[_weighted_share(column, weight) for column in _SHARE_COLUMNS],
        )
        .with_columns(
            (pl.col("carries") + target_weight * pl.col("targets")).alias(WEIGHTED_OPPORTUNITY)
        )
        .sort(list(group_by))
    )
