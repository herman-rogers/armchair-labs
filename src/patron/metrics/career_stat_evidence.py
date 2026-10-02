"""Reviewed zero-stat appearances used only by descriptive career comparisons.

Absent box-score rows alone do not establish zero production. These reviews
identify the player's entire offensive participation and retain their sources.
Published profile artifacts and forecast inputs are not modified.
"""

import polars as pl

ZERO_STAT_REVIEWS = (
    {
        "player_id": "00-0034857",
        "season": 2024,
        "week": 18,
        "team": "BUF",
        "offense_snaps": 1,
        "source_url": "https://www.buffalobills.com/news/"
        "top-3-things-we-learned-from-bills-at-patriots-week-18",
        "source_date": "2025-01-05",
        "evidence": "Josh Allen's only offensive snap was a handoff to James Cook.",
        "zero_metrics": ["attempts", "passing_yards", "carries", "rushing_yards"],
    },
    {
        "player_id": "00-0034857",
        "season": 2025,
        "week": 18,
        "team": "BUF",
        "offense_snaps": 1,
        "source_url": "https://www.buffalobills.com/news/"
        "top-3-things-we-learned-from-bills-vs-jets-week-18-x2872",
        "source_date": "2026-01-04",
        "evidence": "Josh Allen's only offensive snap was a handoff to James Cook III.",
        "zero_metrics": ["attempts", "passing_yards", "carries", "rushing_yards"],
    },
)


def apply_zero_reviews(weeks: pl.DataFrame, metrics: list[str], reviews=ZERO_STAT_REVIEWS):
    """Fill reviewed nulls only when the captured identity and participation agree.

    Recorded values always win. A changed snap count, team or scoring record
    invalidates the match instead of extending an old review to different data.
    The caller must apply its observation cutoff before invoking this function.
    """
    weeks = weeks.with_columns(pl.lit(False).alias("verified_zero"))
    applied = []
    if "team" not in weeks.columns:
        return weeks, applied
    for review in reviews:
        fields = [m for m in metrics if m in review["zero_metrics"]]
        if not fields:
            continue
        match = (
            (pl.col("player_id") == review["player_id"])
            & (pl.col("season") == review["season"])
            & (pl.col("week") == review["week"])
            & (pl.col("team") == review["team"])
            & (pl.col("offense_snaps") == review["offense_snaps"])
            & (pl.col("stat_recorded") == False)  # noqa: E712
            & pl.all_horizontal([pl.col(m).is_null() | (pl.col(m) == 0) for m in fields])
            & pl.any_horizontal([pl.col(m).is_null() for m in fields])
        ).fill_null(False)
        if weeks.filter(match).height != 1:
            continue
        weeks = weeks.with_columns(
            (pl.col("verified_zero") | match).alias("verified_zero"),
            *[
                pl.when(match).then(pl.col(m).fill_null(0)).otherwise(pl.col(m)).alias(m)
                for m in fields
            ],
        )
        applied.append(review)
    return weeks, applied
