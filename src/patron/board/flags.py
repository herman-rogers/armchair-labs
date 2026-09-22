"""Flag composition.

The board's flag column compresses the model's warnings into a few characters that can
be read at draft speed:

* ``age`` — a running back at or past the cliff. A discount, not a disqualification.
* ``Ngms`` — produced in only N games. The per-game number is real; the sample is not
  large. Verify health before acting on it.

The former ``BUY`` and ``TD-luck`` touchdown-regression flags were removed after the
2026 backtest review found TD-over-expectation carried no repeatable next-season
signal (docs/v2_metrics_review.md §4). ``ESPN-only`` is appended separately by the
ESPN crosswalk for players with no usable NFL production.

Order is fixed — age, then sample — so the same combination always renders the same
way and the column stays scannable.
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
        board: Rows carrying `rb_age_cliff` and `games`.
        small_sample_games: Games below which the `Ngms` sample flag is shown.

    Returns:
        `board` with a `flags` column. Players with nothing to say carry an empty
        string rather than a null, so the column renders cleanly.
    """
    age = (
        pl.when(pl.col("rb_age_cliff")).then(pl.lit("age")).otherwise(pl.lit(None, dtype=pl.String))
    )
    sample = (
        pl.when(pl.col("games") < small_sample_games)
        .then(pl.col("games").cast(pl.String) + pl.lit("gms"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )

    return board.with_columns(
        pl.concat_str([age, sample], separator=FLAG_SEPARATOR, ignore_nulls=True)
        .fill_null("")
        .alias(FLAGS_COLUMN)
    )
