"""Big-play touchdown bonuses, reduced from play-by-play.

This league pays +2 on a touchdown of 40-49 yards and +3 on 50 or more, and it is the
league's private edge: it is why deep threats and the quarterbacks throwing to them
rank rounds above consensus here. Weekly stats cannot produce it, because they carry
touchdown *counts* and not touchdown *distances*, so this is the one place the
~370-column play-by-play has to be loaded.

Two corrections to the draft-prep script live here.

**Bonuses are keyed by season and week, not just season.** The original aggregated
straight to a season total, which meant weekly league points never included the bonus,
which in turn meant the weekly floor and volatility metrics were computed on a
different scoring system than the PPG they sat next to. In a league whose edge is the
bonus, the floor metric was silently ignoring it.

**Null player IDs are dropped explicitly and counted.** `receiver_player_id` is null on
a rushing touchdown and `rusher_player_id` on a passing one; on the odd play they are
null when they should not be. Grouped without a filter, those rows collect under a null
key and then join to nothing — the points vanish with no error. `BonusAudit` exists so
that loss is measured rather than assumed to be zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from engine.scoring.columns import PBP_TOUCHDOWN_COLUMNS, require_columns
from engine.scoring.rules import ScoringRules, get_rules

BONUS_POINTS = "bonus_pts"

#: (id column, touchdown-type flag) pairs that earn a bonus.
#: A passing touchdown pays BOTH the passer and the receiver; a rushing touchdown
#: pays the rusher alone.
_CREDIT_SOURCES: tuple[tuple[str, str], ...] = (
    ("passer_player_id", "pass_touchdown"),
    ("receiver_player_id", "pass_touchdown"),
    ("rusher_player_id", "rush_touchdown"),
)


@dataclass(frozen=True)
class BonusAudit:
    """Reconciliation of bonus points earned on the field against points credited.

    `expected_points` is what every scoring play should pay out, counting the passer
    and receiver on a passing touchdown as two separate credits. `credited_points` is
    what actually landed on a player. They should be equal; `dropped_credits` counts
    the credits lost to a null player ID when they are not.
    """

    scoring_plays: int
    expected_points: float
    credited_points: float
    dropped_credits: int

    @property
    def is_balanced(self) -> bool:
        return abs(self.expected_points - self.credited_points) < 1e-6

    def summary(self) -> str:
        state = "balanced" if self.is_balanced else "UNBALANCED"
        return (
            f"bonus reconciliation {state}: {self.scoring_plays} scoring plays, "
            f"expected {self.expected_points:.1f} pts, credited {self.credited_points:.1f} pts, "
            f"{self.dropped_credits} credit(s) dropped to null player ids"
        )


def _bonus_expression(rules: ScoringRules) -> pl.Expr:
    """Map `yards_gained` to bonus points, highest matching bracket only.

    Built by walking the configured brackets from the top down, so the first match
    wins and a 55-yard touchdown scores 3 rather than 2 + 3.
    """
    ordered = sorted(rules.bonuses.brackets, key=lambda b: b.min_yards, reverse=True)
    # Polars types a when/then chain differently once it has been extended, which is
    # correct at runtime but not expressible in the annotation.
    expression: Any = pl.when(pl.lit(False)).then(pl.lit(0.0))
    for bracket in ordered:
        condition = pl.col("yards_gained") >= bracket.min_yards
        if bracket.max_yards is not None:
            condition = condition & (pl.col("yards_gained") <= bracket.max_yards)
        expression = expression.when(condition).then(pl.lit(float(bracket.points)))
    return expression.otherwise(pl.lit(0.0))


def extract_touchdown_bonuses(
    pbp: pl.DataFrame,
    rules: ScoringRules | None = None,
    season_types: tuple[str, ...] = ("REG",),
) -> tuple[pl.DataFrame, BonusAudit]:
    """Reduce play-by-play to per-player, per-week big-play bonus points.

    Args:
        pbp: Play-by-play rows carrying `PBP_TOUCHDOWN_COLUMNS`.
        rules: Scoring rules; defaults to the league's configured rules.
        season_types: Season types to keep. Regular season only by default, matching
            the weekly stats filter — postseason bonuses must not leak into a
            regular-season total.

    Returns:
        A `(bonuses, audit)` pair. `bonuses` has columns
        `season, week, player_id, bonus_pts`, one row per player-week that earned
        anything. `audit` reconciles points earned against points credited.
    """
    rules = rules or get_rules()
    require_columns(pbp.columns, PBP_TOUCHDOWN_COLUMNS, "extract_touchdown_bonuses input")

    scoring_plays = (
        pbp.filter((pl.col("touchdown") == 1) & pl.col("season_type").is_in(list(season_types)))
        .select(PBP_TOUCHDOWN_COLUMNS)
        .with_columns(_bonus_expression(rules).alias(BONUS_POINTS))
        .filter(pl.col(BONUS_POINTS) > 0)
    )

    # What the plays *should* pay: a passing touchdown pays two credits (passer and
    # receiver), a rushing touchdown one. Computed from the play flags alone, before
    # any player ID is consulted, so it is independent of the join below.
    expected_points = float(
        scoring_plays.select(
            (
                pl.col(BONUS_POINTS)
                * (
                    2 * (pl.col("pass_touchdown") == 1).cast(pl.Int8)
                    + (pl.col("rush_touchdown") == 1).cast(pl.Int8)
                )
            ).sum()
        ).item()
        or 0.0
    )

    credits: list[pl.DataFrame] = []
    dropped = 0
    for id_column, type_flag in _CREDIT_SOURCES:
        eligible = scoring_plays.filter(pl.col(type_flag) == 1)
        named = eligible.filter(pl.col(id_column).is_not_null())
        dropped += eligible.height - named.height
        credits.append(
            named.select(
                "season",
                "week",
                pl.col(id_column).alias("player_id"),
                BONUS_POINTS,
            )
        )

    if credits:
        bonuses = (
            pl.concat(credits)
            .group_by(["season", "week", "player_id"])
            .agg(pl.col(BONUS_POINTS).sum())
            .sort(["season", "week", "player_id"])
        )
    else:  # pragma: no cover - only reachable if _CREDIT_SOURCES is emptied
        bonuses = pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "player_id": pl.String,
                BONUS_POINTS: pl.Float64,
            }
        )

    credited_points = float(bonuses.select(pl.col(BONUS_POINTS).sum()).item() or 0.0)

    audit = BonusAudit(
        scoring_plays=scoring_plays.height,
        expected_points=expected_points,
        credited_points=credited_points,
        dropped_credits=dropped,
    )
    return bonuses, audit


def season_bonus_totals(bonuses: pl.DataFrame) -> pl.DataFrame:
    """Collapse per-week bonuses to a per-player season total.

    The board shows this as a visible column: it is the league's private edge made
    legible, and the reason a deep-ball quarterback outranks his consensus price.
    """
    return (
        bonuses.group_by(["season", "player_id"])
        .agg(pl.col(BONUS_POINTS).sum())
        .sort(["season", "player_id"])
    )
