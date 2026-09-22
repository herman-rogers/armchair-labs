"""Canonical preseason forecast; historical/component fields remain compatibility aliases."""

from __future__ import annotations

import polars as pl

FORECAST_COLUMNS = (
    "forecast_active_ppg",
    "forecast_expected_games",
    "forecast_season_points",
    "forecast_source",
    "forecast_as_of",
    "forecast_units",
    "forecast_status",
    "forecast_sample_support",
    "forecast_uncertainty",
    "simulation_source",
    "historical_ppg_definition",
    "forecast_draft_vor",
)


def attach_forecast(frame: pl.DataFrame, as_of: str) -> pl.DataFrame:
    def col(name: str) -> pl.Expr:
        return pl.col(name) if name in frame.columns else pl.lit(None, dtype=pl.Float64)

    adaptive = "adaptive_season_points" in frame.columns
    points = col("adaptive_season_points") if adaptive else col("fitted_season_points")
    market = (
        (pl.col("rank_source") == "market") if "rank_source" in frame.columns else pl.lit(False)
    )
    return frame.with_columns(
        (pl.lit(None, dtype=pl.Float64) if adaptive else col("fitted_ppg")).alias(
            "forecast_active_ppg"
        ),
        (pl.lit(None, dtype=pl.Float64) if adaptive else col("fitted_games")).alias(
            "forecast_expected_games"
        ),
        points.alias("forecast_season_points"),
        pl.when(market)
        .then(pl.lit("market_rank_match"))
        .otherwise(pl.lit("frozen_adaptive" if adaptive else "production_returner_v1"))
        .alias("forecast_source"),
        pl.lit(as_of).alias("forecast_as_of"),
        pl.lit("league_points_per_season").alias("forecast_units"),
        pl.when(points.is_null())
        .then(pl.lit("unavailable"))
        .when(market)
        .then(pl.lit("fallback"))
        .otherwise(pl.lit("frozen_experiment" if adaptive else "preseason"))
        .alias("forecast_status"),
        col("projection_confidence").alias("forecast_sample_support"),
        pl.lit("uncalibrated").alias("forecast_uncertainty"),
        pl.lit("v2_separate_estimate" if adaptive else "v2").alias("simulation_source"),
        pl.lit("participation_observed_active_games").alias("historical_ppg_definition"),
        col("v2_overall_vor").alias("forecast_draft_vor"),
    )
