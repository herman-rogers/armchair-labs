"""Touchdown-over-expectation: the regression engine.

The most exploitable signal the model produces, because it finds mispricings rather
than confirming them. Touchdowns are the noisiest part of a fantasy line — a back who
scored twelve times on 200 touches did not earn twelve times, he earned about eight
and got lucky four times — and the market prices last year's touchdowns as if they
repeat. They do not; volume does.

So: actual touchdowns minus expected, where expected is the player's touches times
the positional average touchdown rate.

* Well above expectation is a **sell**. The price is built on luck that will not repeat.
* Well below expectation, *and only with real volume behind it*, is a **buy**. Volume
  is what makes it a regression case rather than simply a player who is not good.
  Without the volume gate this flag fires on every deep reserve who caught four passes.

The positional rate window is configurable. The draft-prep script pooled every loaded
season into one blended rate and applied it to each season's rows; that is preserved
as the default (`"all"`) because it is what produced the board being validated against,
and a blended rate is more stable than a single-season one. `"season"` computes it
per season instead.
"""

from __future__ import annotations

import polars as pl

from patron.scoring.columns import require_columns

TD_OVER_EXPECTATION = "td_over_exp"
EXPECTED_TDS = "exp_tds"
TOTAL_TDS = "total_tds"
POSITION_TD_RATE = "pos_td_rate"


def positional_td_rates(
    seasons: pl.DataFrame,
    window: str = "all",
) -> pl.DataFrame:
    """Average touchdowns per touch, by position.

    Args:
        seasons: Player-season rows with `total_tds`, `carries`, and `targets`.
        window: `"all"` pools every season into one rate per position; `"season"`
            computes a separate rate per position per season.

    Returns:
        Position (and season, when windowed) keyed rates in `pos_td_rate`.
    """
    if window not in {"all", "season"}:
        raise ValueError(f"window must be 'all' or 'season', got {window!r}")

    require_columns(
        seasons.columns, (TOTAL_TDS, "carries", "targets", "position"), "positional_td_rates input"
    )

    keys = ["position"] if window == "all" else ["position", "season"]
    touches = pl.col("carries").sum() + pl.col("targets").sum()
    return seasons.group_by(keys).agg(
        (pl.col(TOTAL_TDS).sum() / touches.replace(0, None)).alias(POSITION_TD_RATE)
    )


def add_td_over_expectation(
    seasons: pl.DataFrame,
    window: str = "all",
) -> pl.DataFrame:
    """Add `exp_tds` and `td_over_exp` to player-season rows.

    Expectation is touches times the positional rate. Touches means carries plus
    targets: a quarterback's passing touchdowns are excluded on both sides, since
    throwing a touchdown and running one in are not the same skill and should not
    regress against a shared baseline.
    """
    rates = positional_td_rates(seasons, window=window)
    join_keys = ["position"] if window == "all" else ["position", "season"]

    touches = pl.col("carries").fill_null(0) + pl.col("targets").fill_null(0)
    return (
        seasons.join(rates, on=join_keys, how="left")
        .with_columns((touches * pl.col(POSITION_TD_RATE)).alias(EXPECTED_TDS))
        .with_columns(
            (pl.col(TOTAL_TDS).fill_null(0) - pl.col(EXPECTED_TDS)).alias(TD_OVER_EXPECTATION)
        )
    )


def add_regression_flags(
    seasons: pl.DataFrame,
    regress_down_at: float = 4.0,
    regress_up_at: float = -2.5,
    regress_up_min_opportunity: float = 150.0,
    opportunity_column: str = "wtd_opp",
) -> pl.DataFrame:
    """Add the `td_regress_down` (sell) and `td_regress_up` (buy) boolean flags.

    The buy flag requires **both** conditions. A player far under expectation without
    volume is not a regression candidate, he is a bench player who caught a few passes,
    and firing the flag on him is how a good signal becomes noise.
    """
    require_columns(
        seasons.columns, (TD_OVER_EXPECTATION, opportunity_column), "add_regression_flags input"
    )

    tdoe = pl.col(TD_OVER_EXPECTATION)
    return seasons.with_columns(
        (tdoe >= regress_down_at).fill_null(False).alias("td_regress_down"),
        ((tdoe <= regress_up_at) & (pl.col(opportunity_column) >= regress_up_min_opportunity))
        .fill_null(False)
        .alias("td_regress_up"),
    )
