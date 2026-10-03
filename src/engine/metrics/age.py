"""Age curves.

Age is a discount, not a disqualification. Running backs fall off a cliff that is
really age times cumulative workload — a 28-year-old with 1,200 career touches is a
different proposition from a 28-year-old with 500 — but touches are not in this
pipeline yet, so age alone stands in, flagged rather than penalised. Receivers age
gently into their early thirties and quarterbacks later still, so the cliff flag is
applied to backs only.

Age is measured as of September 1 of the season being played, not the season the data
came from: the question is how old a player will be while producing, not how old he
was while producing the tape.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from engine.scoring.columns import require_columns

AGE_COLUMN = "age_at_season"
DAYS_PER_YEAR = 365.25


def add_age(
    players: pl.DataFrame,
    season: int,
    birth_date_column: str = "birth_date",
) -> pl.DataFrame:
    """Add `age_at_season`: age in years as of September 1 of `season`.

    Null birth dates yield a null age rather than a wrong one. That happens for
    players nflverse has no roster row for, and the board should show a blank rather
    than silently treat them as young.
    """
    require_columns(players.columns, (birth_date_column,), "add_age input")

    reference = dt.date(season, 9, 1)
    birth = pl.col(birth_date_column).cast(pl.Date)
    age = (pl.lit(reference) - birth).dt.total_days() / DAYS_PER_YEAR

    # Full precision, deliberately. The draft-prep script rounded to one decimal here,
    # which is a display concern leaking into a metric: at 0.1-year granularity a
    # player 18 days short of the 27.5 cliff rounds onto the wrong side of it. The
    # board rounds for presentation; the flag compares the real number.
    return players.with_columns(age.alias(AGE_COLUMN))


def add_age_flags(
    players: pl.DataFrame,
    rb_cliff: float = 27.5,
    age_column: str = AGE_COLUMN,
) -> pl.DataFrame:
    """Add the `rb_age_cliff` flag for running backs at or past the cliff age."""
    require_columns(players.columns, (age_column, "position"), "add_age_flags input")

    return players.with_columns(
        ((pl.col("position") == "RB") & (pl.col(age_column) >= rb_cliff))
        .fill_null(False)
        .alias("rb_age_cliff")
    )
