"""Research/v2 eligibility policy; never applied to the frozen V1 board."""

import polars as pl


def canonical_positions(frame: pl.DataFrame, column: str = "position") -> pl.DataFrame:
    """Use the same FB/HB -> RB policy for candidates, usage and actual outcomes."""
    return frame.with_columns(
        pl.col(column).str.to_uppercase().replace({"FB": "RB", "HB": "RB"}).alias(column)
    )


def historical_positions(rosters: pl.DataFrame) -> pl.DataFrame:
    """Weekly roster position describes that week; current player metadata does not.

    Conflicting same-week classifications remain flagged and unresolved. They are
    never selected according to raw input order.
    """
    return (
        canonical_positions(rosters)
        .filter((pl.col("game_type") == "REG") & pl.col("gsis_id").is_not_null())
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("season", "week").cast(pl.Int32),
            "position",
        )
        .group_by("player_id", "season", "week")
        .agg(
            pl.when(pl.col("position").drop_nulls().n_unique() == 1)
            .then(pl.col("position").drop_nulls().first())
            .otherwise(None)
            .alias("historical_position"),
            (pl.col("position").drop_nulls().n_unique() > 1).alias("position_conflict"),
        )
    )


def repair_player_week_positions(
    weeks: pl.DataFrame,
    positions: pl.DataFrame,
    *,
    overrides: dict[str, str] | None = None,
) -> pl.DataFrame:
    result = (
        canonical_positions(weeks)
        .with_columns(
            pl.col("season", "week").cast(pl.Int32), pl.col("position").alias("raw_position")
        )
        .join(positions, on=["player_id", "season", "week"], how="left", validate="m:1")
        .with_columns(
            pl.coalesce("historical_position", "position").alias("position"),
            pl.when(pl.col("historical_position").is_not_null())
            .then(pl.lit("weekly_roster"))
            .otherwise(pl.lit("stat_feed_fallback"))
            .alias("position_source"),
        )
        .sort("season", "week", "player_id")
    )
    if overrides:
        result = result.with_columns(
            pl.col("player_id")
            .replace_strict(overrides, default=pl.col("position"))
            .alias("position"),
            pl.when(pl.col("player_id").is_in(list(overrides)))
            .then(pl.lit("league_override"))
            .otherwise(pl.col("position_source"))
            .alias("position_source"),
        )
    return result
