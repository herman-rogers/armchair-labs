"""Attach dated roster evidence to every forecast candidate population."""

import polars as pl


def supplement_identity_names(players: pl.DataFrame, crosswalk: pl.DataFrame) -> pl.DataFrame:
    """Add unambiguous stable names only; no current team, position, or biography."""
    names = crosswalk.select("gsis_id", pl.col("name").alias("display_name")).drop_nulls()
    names = (
        names.group_by("gsis_id")
        .agg(
            pl.col("display_name").n_unique().alias("_name_count"),
            pl.col("display_name").first(),
        )
        .filter(pl.col("_name_count") == 1)
        .drop("_name_count")
    )
    extra = names.join(players.select("gsis_id"), on="gsis_id", how="anti")
    return pl.concat([players, extra], how="diagonal_relaxed")


def attach_roster_evidence(
    forecasts: pl.DataFrame, transactions: pl.DataFrame, players: pl.DataFrame
) -> pl.DataFrame:
    names = players.select(
        pl.col("gsis_id").alias("player_id"), pl.col("display_name").alias("_identity_name")
    ).unique("player_id")
    frame = forecasts.join(names, on="player_id", how="left", validate="m:1")
    frame = frame.with_columns(
        pl.coalesce("player_display_name", "_identity_name").alias("player_display_name")
    ).drop("_identity_name")
    if transactions.is_empty():
        return frame
    keys = ["player_id", "forecast_season"]
    columns = [c for c in transactions.columns if c not in keys]
    incoming = transactions.select(*keys, *columns).rename({c: f"_dated_{c}" for c in columns})
    incoming = incoming.with_columns(pl.lit(True).alias("_dated_present"))
    frame = frame.join(incoming, on=keys, how="left", validate="m:1")
    frame = frame.with_columns(
        *(
            pl.when(pl.col("_dated_present").fill_null(False))
            .then(pl.col(f"_dated_{name}"))
            .otherwise(pl.col(name) if name in forecasts.columns else pl.lit(None))
            .alias(name)
            for name in columns
        )
    )
    frame = frame.drop([c for c in frame.columns if c.startswith("_dated_")])
    return frame.with_columns(
        pl.col("cutoff_preseason_team").alias("preseason_team"),
        pl.col("cutoff_preseason_rostered").alias("preseason_rostered"),
        pl.col("cutoff_preseason_reserve").alias("preseason_reserve"),
        pl.col("cutoff_preseason_status_score").alias("preseason_status_score"),
        (
            pl.col("cutoff_preseason_team").is_not_null()
            & ~pl.col("cutoff_state_resolution").is_in(
                [
                    "inferred_prior_team",
                    "no_prior_team",
                    "conflicting_same_day",
                    "unresolved_action",
                ]
            )
        ).alias("cutoff_team_observed"),
    )
