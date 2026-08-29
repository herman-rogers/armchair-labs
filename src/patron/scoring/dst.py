"""Defense/special-teams scoring.

Phase 1 keeps the draft-prep proxy: sacks + 2x takeaways + 6x defensive touchdowns,
shown next to average points allowed. It is directional rather than league-exact — it
does not model the per-week points-allowed and yards-allowed brackets — and that was
adequate for a position drafted last.

The proxy's known weakness is worth stating where it will be read: sack rate persists
year over year, takeaways regress hard. A unit that ranks highly here on the strength
of turnovers is a worse bet than one that ranks highly on pressure. `takeaway_share`
is surfaced so that judgement can be made rather than guessed.

A faithful per-week scorer, using the brackets already encoded in scoring.yaml, is
Phase 4 work; `score_dst_week` below is the piece it will build on.
"""

from __future__ import annotations

import polars as pl

from patron.scoring.columns import DEFENSE_COLUMNS, SCHEDULE_COLUMNS, require_columns
from patron.scoring.rules import ScoringRules, get_rules

PROXY_POINTS = "proxy_pts"


def points_allowed_by_team(schedules: pl.DataFrame) -> pl.DataFrame:
    """Average points allowed per team, from game results.

    Each game contributes twice — once per side — so both teams get a row.
    """
    require_columns(schedules.columns, SCHEDULE_COLUMNS, "points_allowed_by_team input")

    regular = schedules.filter(pl.col("game_type") == "REG")
    sides = pl.concat(
        [
            regular.select(
                pl.col("home_team").alias("team"),
                pl.col("away_score").alias("pts_allowed"),
            ),
            regular.select(
                pl.col("away_team").alias("team"),
                pl.col("home_score").alias("pts_allowed"),
            ),
        ]
    )
    return (
        sides.drop_nulls("pts_allowed")
        .group_by("team")
        .agg(
            pl.col("pts_allowed").mean().alias("avg_pts_allowed"),
            pl.len().alias("games"),
        )
    )


def build_dst_proxy(
    stats: pl.DataFrame,
    schedules: pl.DataFrame,
    rules: ScoringRules | None = None,
) -> pl.DataFrame:
    """Rank defenses by the directional proxy score.

    Args:
        stats: Player-week rows carrying `DEFENSE_COLUMNS`; aggregated to team level.
        schedules: Game results, for points allowed.
        rules: Scoring rules; defaults to the league's configured rules.
    """
    rules = rules or get_rules()
    require_columns(stats.columns, DEFENSE_COLUMNS, "build_dst_proxy input")

    def total(column: str) -> pl.Expr:
        return pl.col(column).fill_null(0).sum()

    teams = stats.group_by("team").agg(
        total("def_sacks").alias("sacks"),
        (total("def_interceptions") + total("fumble_recovery_opp")).alias("takeaways"),
        total("def_tds").alias("def_tds"),
        total("def_safeties").alias("safeties"),
    )

    proxy = (
        pl.col("sacks") * rules.defense.sack
        + pl.col("takeaways") * rules.defense.interception
        + pl.col("def_tds") * rules.defense.touchdown
        + pl.col("safeties") * rules.defense.safety
    )

    return (
        teams.join(points_allowed_by_team(schedules), on="team", how="left")
        .with_columns(proxy.alias(PROXY_POINTS))
        .with_columns(
            # How much of the score rests on turnovers, which regress hardest.
            # High share means treat the ranking with suspicion.
            (
                pl.col("takeaways")
                * rules.defense.interception
                / pl.col(PROXY_POINTS).replace(0, None)
            ).alias("takeaway_share")
        )
        .sort(PROXY_POINTS, descending=True)
    )


def score_dst_week(
    points_allowed: float,
    yards_allowed: float,
    sacks: float = 0.0,
    interceptions: float = 0.0,
    fumble_recoveries: float = 0.0,
    touchdowns: float = 0.0,
    safeties: float = 0.0,
    rules: ScoringRules | None = None,
) -> float:
    """League-exact DST points for a single team-week.

    The faithful scorer the proxy stands in for. Not yet wired into the board — it
    needs per-week yards-allowed, which comes from play-by-play aggregation in Phase 4
    — but it is here, and tested, so that work starts from a verified scorer.
    """
    rules = rules or get_rules()
    return (
        sacks * rules.defense.sack
        + interceptions * rules.defense.interception
        + fumble_recoveries * rules.defense.fumble_recovery
        + touchdowns * rules.defense.touchdown
        + safeties * rules.defense.safety
        + rules.defense.points_allowed_points(points_allowed)
        + rules.defense.yards_allowed_points(yards_allowed)
    )
