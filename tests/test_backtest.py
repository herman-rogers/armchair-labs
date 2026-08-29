"""Metric report statistics stay configurable and season-aware."""

from __future__ import annotations

import polars as pl

from patron.metrics.backtest import (
    MetricDefinition,
    MetricReportConfig,
    TargetDefinition,
    analyze_predictions,
    build_metric_report,
)


def report_config() -> MetricReportConfig:
    return MetricReportConfig(
        title="Test report",
        forecast_seasons=(2023, 2024, 2025),
        history_seasons=3,
        depth_chart_cutoff="08-31",
        minimum_sample=4,
        baseline_metric="prior",
        targets=(TargetDefinition("actual_ppg", "Actual PPG", "Outcome"),),
        metrics=(
            MetricDefinition(
                key="signal",
                label="Signal",
                group="Test",
                description="Perfectly ordered signal",
                targets=("actual_ppg",),
                prediction_target="actual_ppg",
            ),
            MetricDefinition(
                key="prior",
                label="Prior",
                group="Baseline",
                description="Baseline",
                targets=("actual_ppg",),
            ),
        ),
    )


def predictions() -> pl.DataFrame:
    rows = []
    for season in (2023, 2024):
        for index in range(1, 6):
            rows.append(
                {
                    "forecast_season": season,
                    "outcome_complete": True,
                    "position": "WR",
                    "signal": float(index),
                    "prior": float(6 - index if season == 2023 else index % 2),
                    "actual_ppg": float(index),
                }
            )
    rows.append(
        {
            "forecast_season": 2025,
            "outcome_complete": False,
            "position": "WR",
            "signal": 10.0,
            "prior": 10.0,
            "actual_ppg": None,
        }
    )
    return pl.DataFrame(rows)


def test_analysis_keeps_fold_results_and_incremental_signal() -> None:
    results, models = analyze_predictions(predictions(), report_config())
    signal = next(
        row
        for row in results
        if row["metric"] == "signal" and row["position"] == "ALL"
    )

    assert signal["n"] == 10
    assert signal["folds"] == 2
    assert signal["spearman"] == 1.0
    assert signal["partial_spearman"] is not None
    assert [fold["forecast_season"] for fold in signal["fold_results"]] == [2023, 2024]
    assert models[0]["mae"] == 0.0


def test_report_distinguishes_completed_and_pending_forecasts() -> None:
    report = build_metric_report(predictions(), report_config())

    assert report["data_summary"]["completed_forecasts"] == [2023, 2024]
    assert report["data_summary"]["pending_forecasts"] == [2025]
    assert report["metrics"][0]["available"] is True
    assert report["metrics"][0]["coverage"] == 1.0
