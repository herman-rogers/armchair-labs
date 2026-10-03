"""Stable display ranks of permitted forecasts, calculated before search or pagination."""

from pathlib import Path

import polars as pl

from engine.data.nextgen import eligible
from engine.tables.application import read_frame, read_json

MARKET_COLUMNS = [
    "player_id",
    "forecast_season",
    "forecast_cutoff_date",
    "position",
    "market_overall_ecr",
    "market_overall_snapshot",
    "market_ecr",
    "market_snapshot",
    "market_position",
]

RANK_COLUMNS = ["overall_rank", "position_rank", "population_rank"]


def with_ranks(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("prediction").rank(method="min", descending=True).alias("overall_rank"),
        pl.col("prediction")
        .rank(method="min", descending=True)
        .over("position")
        .alias("position_rank"),
        pl.col("prediction")
        .rank(method="min", descending=True)
        .over("position", "population")
        .alias("population_rank"),
    )


def baseline_ranks(root: Path) -> pl.DataFrame:
    frame = read_frame(root / "forecasts.parquet").filter(
        (pl.col("model") == "baseline") & (pl.col("target") == "season_points")
    )
    entry = next(
        (
            r
            for r in read_json(root / "registry.json")
            if r["id"] == "forecast:baseline:season_points"
        ),
        {},
    )
    incidents = read_json(root / "incidents.json")
    allowed = [
        (pl.col("position") == r["position"]) & (pl.col("population") == r["population"])
        for r in frame.select("position", "population").unique().to_dicts()
        if eligible(
            entry,
            use="forecast",
            target="season_points",
            horizon="preseason_season",
            position=r["position"],
            population=r["population"],
            incidents=incidents,
        )[0]
    ]
    return with_ranks(frame.filter(pl.any_horizontal(allowed) if allowed else pl.lit(False)))


def market_references(features: pl.DataFrame, season: int, incidents=()) -> pl.DataFrame:
    """Expose dated consensus observations, never substitute missing/late ranks."""
    frame = features.filter(pl.col("forecast_season") == season)
    expressions = [pl.col("player_id")]
    blocked = {d for i in incidents if i.get("status") == "open" for d in i.get("dependencies", [])}
    for source, date, output in (
        ("market_overall_ecr", "market_overall_snapshot", "ecr_overall"),
        ("market_ecr", "market_snapshot", "ecr_position"),
    ):
        valid = (
            pl.col(date).str.to_date(strict=False).is_not_null()
            & (pl.col(date).str.to_date(strict=False) <= pl.col("forecast_cutoff_date"))
            & pl.col(source).is_finite()
            & (pl.col(source) > 0)
        )
        if source == "market_ecr":
            valid = valid & (pl.col("market_position") == pl.col("position"))
        if blocked & {f"preseason_features.{source}", f"preseason_features.{date}"}:
            valid = pl.lit(False)
        expressions.extend(
            [
                pl.when(valid).then(pl.col(source)).otherwise(None).alias(output),
                pl.when(valid).then(pl.col(date)).otherwise(None).alias(output + "_date"),
            ]
        )
    return frame.select(expressions)
