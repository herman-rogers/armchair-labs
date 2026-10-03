"""Kicker scoring under the league's distance brackets.

The distance brackets create a real spread — several points per game — at a position
most of the room drafts as a coin flip. That is only worth exploiting if the scoring
is exact, so the bracket points come from scoring.yaml rather than being hardcoded.

nflverse reports made field goals in fixed 10-yard buckets rather than per-kick
distances. Each bucket is scored by resolving its midpoint against the configured
brackets, which is exact as long as no configured bracket edge falls inside a bucket.
`validate_bracket_alignment` checks that and fails loudly if the league ever adopts,
say, a 45-yard threshold that would split the 40-49 bucket.
"""

from __future__ import annotations

import polars as pl

from engine.scoring.columns import KICKING_COLUMNS, require_columns
from engine.scoring.rules import ScoringRules, get_rules

KICKER_POINTS = "league_pts"

#: nflverse made-FG bucket columns and the yardage range each covers.
_MADE_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("fg_made_0_19", 0, 19),
    ("fg_made_20_29", 20, 29),
    ("fg_made_30_39", 30, 39),
    ("fg_made_40_49", 40, 49),
    ("fg_made_50_59", 50, 59),
    ("fg_made_60_", 60, None),
)


class BracketAlignmentError(ValueError):
    """Raised when a scoring bracket edge falls inside an nflverse distance bucket.

    If that happens, bucket counts can no longer be scored exactly and the kicker
    numbers would be quietly approximate. Play-by-play kick distances would be needed
    instead.
    """


def validate_bracket_alignment(rules: ScoringRules | None = None) -> None:
    """Assert every nflverse bucket maps to exactly one configured bracket."""
    rules = rules or get_rules()

    for column, low, high in _MADE_BUCKETS:
        top = high if high is not None else 99
        low_points = rules.kicking.field_goal_points(low)
        high_points = rules.kicking.field_goal_points(top)
        if low_points != high_points:
            raise BracketAlignmentError(
                f"nflverse bucket {column} ({low}-{top} yds) spans two scoring brackets "
                f"({low_points} pts at {low} yds, {high_points} pts at {top} yds). "
                "Bucket counts can no longer be scored exactly; score kickers from "
                "play-by-play kick distances instead."
            )


def _bucket_points(rules: ScoringRules, low: int, high: int | None) -> float:
    midpoint = low + 2 if high is None else (low + high) // 2
    return rules.kicking.field_goal_points(midpoint)


def score_kicker_weeks(
    stats: pl.DataFrame,
    rules: ScoringRules | None = None,
) -> pl.DataFrame:
    """Add `league_pts` to kicker player-week rows.

    Args:
        stats: Kicker rows carrying `KICKING_COLUMNS`.
        rules: Scoring rules; defaults to the league's configured rules.
    """
    rules = rules or get_rules()
    require_columns(stats.columns, KICKING_COLUMNS, "score_kicker_weeks input")
    validate_bracket_alignment(rules)

    made = pl.lit(0.0)
    for column, low, high in _MADE_BUCKETS:
        made = made + pl.col(column).fill_null(0) * _bucket_points(rules, low, high)

    # The league sheet says "miss -1" without saying whether a blocked kick counts.
    # nflverse tracks blocks separately, so the choice is explicit in config.
    misses = pl.col("fg_missed").fill_null(0)
    if rules.kicking.blocked_counts_as_miss:
        misses = misses + pl.col("fg_blocked").fill_null(0)

    points = (
        made
        + pl.col("pat_made").fill_null(0) * rules.kicking.extra_point
        + misses * rules.kicking.missed_field_goal
    )
    return stats.with_columns(points.alias(KICKER_POINTS))


def aggregate_kicker_seasons(scored: pl.DataFrame) -> pl.DataFrame:
    """Collapse scored kicker weeks into a per-season profile.

    Carries the long-range make counts alongside the points: a kicker's 50+ and 60+
    volume is the part of his line that this league's brackets actually pay for, and
    the part that separates him from the room's assumption that kickers are random.
    """
    return (
        scored.group_by(["player_id", "player_display_name", "position", "season"])
        .agg(
            pl.col("team").last().alias("team"),
            pl.len().alias("games"),
            pl.col(KICKER_POINTS).sum().alias(KICKER_POINTS),
            pl.col("fg_made_50_59").fill_null(0).sum().alias("fg_50s"),
            pl.col("fg_made_60_").fill_null(0).sum().alias("fg_60s"),
        )
        .with_columns((pl.col(KICKER_POINTS) / pl.col("games")).alias("ppg"))
        .sort("ppg", descending=True)
    )
