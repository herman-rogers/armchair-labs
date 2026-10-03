"""Repeatable residual screens for cutoff-safe hidden-pattern research.

The metric catalog asks whether a field correlates with next season.  This module asks
the harder question: does it explain what the current fitted model still gets wrong?
Patterns must keep the same direction in three non-overlapping eras before they are
reported as stable.  That is a hypothesis generator, not a promotion rule; surviving
fields still have to win the walk-forward ranking and ECR tests.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import polars as pl

_ERAS = (
    ("discovery", 2007, 2014),
    ("confirmation", 2015, 2018),
    ("modern", 2019, 2025),
)

_RESIDUALS = (
    ("ppg", "actual_ppg", "fitted_ppg"),
    ("games", "actual_games", "fitted_games"),
    ("season_points", "actual_season_points", "fitted_season_points"),
    ("status_games", "actual_games", "fitted_status_games"),
)


def _correlation(frame: pl.DataFrame, metric: str, residual: str) -> float | None:
    pairs = frame.select(metric, residual).drop_nulls()
    if pairs.height < 20 or pairs[metric].n_unique() < 2:
        return None
    value = pairs.select(pl.corr(metric, residual, method="spearman")).item()
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def analyze_residual_patterns(
    predictions: pl.DataFrame,
    metric_keys: Iterable[str],
) -> list[dict[str, Any]]:
    """Screen catalogued metrics against fitted residuals across independent eras.

    Populations include all returners and the cutoff-reconstructed roster subset; when
    present, the old Week-1 proxy remains a sensitivity-only population. Restricting
    the population prevents roster exits from masquerading as a football-performance
    discovery. ``stable`` requires the same non-zero direction in all three eras and
    at least a small (|rho| >= .05) modern residual relationship.
    """
    completed = predictions.filter(pl.col("outcome_complete"))
    populations: list[tuple[str, pl.DataFrame]] = [("all_returners", completed)]
    if "preseason_rostered" in completed.columns:
        populations.append(
            (
                "cutoff_rostered",
                completed.filter(pl.col("preseason_rostered") > 0),
            )
        )
    if "week1_proxy_rostered" in completed.columns:
        populations.append(
            (
                "week1_proxy_rostered",
                completed.filter(pl.col("week1_proxy_rostered") > 0),
            )
        )

    available_metrics = [
        key
        for key in dict.fromkeys(metric_keys)
        if key in completed.columns and completed[key].dtype.is_numeric()
    ]
    results: list[dict[str, Any]] = []
    for residual_name, actual, fitted in _RESIDUALS:
        if actual not in completed.columns or fitted not in completed.columns:
            continue
        residual = f"_research_{residual_name}"
        for population, population_frame in populations:
            frame = population_frame.with_columns(
                (pl.col(actual) - pl.col(fitted)).alias(residual)
            )
            for position in ("QB", "RB", "WR", "TE"):
                position_frame = frame.filter(pl.col("position") == position)
                for metric in available_metrics:
                    eras: dict[str, dict[str, float | int | None]] = {}
                    correlations: list[float] = []
                    for era, start, end in _ERAS:
                        sample = position_frame.filter(
                            pl.col("forecast_season").is_between(start, end)
                        ).select(metric, residual).drop_nulls()
                        correlation = _correlation(sample, metric, residual)
                        eras[era] = {"n": sample.height, "spearman": correlation}
                        if correlation is not None:
                            correlations.append(correlation)
                    same_direction = (
                        len(correlations) == len(_ERAS)
                        and all(value > 0 for value in correlations)
                        or len(correlations) == len(_ERAS)
                        and all(value < 0 for value in correlations)
                    )
                    modern = eras["modern"]["spearman"]
                    stable = bool(
                        same_direction
                        and modern is not None
                        and abs(float(modern)) >= 0.05
                        and min(int(eras[era]["n"] or 0) for era, _, _ in _ERAS) >= 20
                    )
                    stability_floor = (
                        min(abs(value) for value in correlations)
                        if len(correlations) == len(_ERAS)
                        else None
                    )
                    results.append(
                        {
                            "population": population,
                            "position": position,
                            "residual": residual_name,
                            "actual": actual,
                            "baseline": fitted,
                            "metric": metric,
                            "eras": eras,
                            "same_direction": same_direction,
                            "stability_floor": (
                                None if stability_floor is None else round(stability_floor, 4)
                            ),
                            "stable": stable,
                        }
                    )
    return sorted(
        results,
        key=lambda row: (
            not row["stable"],
            -(row["stability_floor"] or 0.0),
            row["population"],
            row["position"],
            row["residual"],
            row["metric"],
        ),
    )
