"""Flag composition.

The board's flag column compresses the model's opinions into a few characters that can
be read at draft speed:

* ``BUY`` — scored well under expectation with real volume behind it; the price is
  wrong in your favour.
* ``TD-luck`` — scored well over expectation; the price is built on luck that will not
  repeat.
* ``age`` — a running back at or past the cliff. A discount, not a disqualification.
* ``Ngms`` — produced in only N games. The per-game number is real; the sample is not
  large. Verify health before acting on it.

Order is fixed — regression, then age, then sample — so the same combination always
renders the same way and the column stays scannable. It matches the published board:
``TD-luck/age``, ``BUY/age``, ``TD-luck/9gms``.
"""

from __future__ import annotations

import polars as pl

FLAGS_COLUMN = "flags"
FLAG_SEPARATOR = "/"


def add_flags(
    board: pl.DataFrame,
    small_sample_games: int = 10,
) -> pl.DataFrame:
    """Add the `flags` string column, composed from the boolean flag columns.

    Args:
        board: Rows carrying `td_regress_up`, `td_regress_down`, `rb_age_cliff`, and
            `games`.
        small_sample_games: Games below which the `Ngms` sample flag is shown.

    Returns:
        `board` with a `flags` column. Players with nothing to say carry an empty
        string rather than a null, so the column renders cleanly.
    """
    # BUY and TD-luck are mutually exclusive by construction (one needs positive
    # over-expectation, the other negative), so a single when/then chain is honest
    # rather than lossy.
    regression = (
        pl.when(pl.col("td_regress_up"))
        .then(pl.lit("BUY"))
        .when(pl.col("td_regress_down"))
        .then(pl.lit("TD-luck"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    age = (
        pl.when(pl.col("rb_age_cliff")).then(pl.lit("age")).otherwise(pl.lit(None, dtype=pl.String))
    )
    sample = (
        pl.when(pl.col("games") < small_sample_games)
        .then(pl.col("games").cast(pl.String) + pl.lit("gms"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )

    return board.with_columns(
        pl.concat_str([regression, age, sample], separator=FLAG_SEPARATOR, ignore_nulls=True)
        .fill_null("")
        .alias(FLAGS_COLUMN)
    )
