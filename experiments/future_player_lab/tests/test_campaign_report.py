"""Comparisons require the same forecast rows; ranking diagnostics use saved predictions."""

import polars as pl
import pytest
from experiments.future_player_lab.campaigns.report import (
    errors,
    paired,
    prior_usage_cohorts,
    selection,
)


def observations():
    return pl.DataFrame(
        [
            {
                "year": 2025,
                "player_id": player,
                "target": "points",
                "recipe": recipe,
                "actual": actual,
                "prediction": prediction,
            }
            for recipe, values in [
                ("ensemble", [("a", 10.0, 9.0), ("b", 5.0, 7.0), ("c", 0.0, 1.0)]),
                ("affine", [("a", 10.0, 6.0), ("b", 5.0, 8.0)]),
            ]
            for player, actual, prediction in values
        ]
    )


def test_same_rows_control_errors_and_selection():
    pair = paired(observations(), "points")
    assert pair.height == 2  # c has no comparator, so neither model gets its easy zero.
    result = errors(pair)
    assert result["model_mse"] == 2.5
    assert result["baseline_mse"] == 12.5
    assert result["mse_reduction_pct"] == 80
    rank = selection(pair, 1)
    assert rank["model_selected_actual_points"] == 10
    assert rank["baseline_selected_actual_points"] == 5
    assert rank["model_spearman"] == pytest.approx(1)
    assert rank["baseline_spearman"] == pytest.approx(-1)


def test_different_outcomes_or_duplicate_keys_fail_review():
    frame = observations()
    with pytest.raises(ValueError, match="Duplicate"):
        paired(pl.concat([frame, frame.head(1)]), "points")
    changed = frame.with_columns(
        pl.when(pl.col("recipe") == "affine")
        .then(pl.col("actual") + 1)
        .otherwise(pl.col("actual"))
        .alias("actual")
    )
    with pytest.raises(ValueError, match="outcomes disagree"):
        paired(changed, "points")


def test_usage_threshold_excludes_evaluation_and_later_seasons():
    panel = pl.DataFrame(
        {
            "year": [2024] * 4 + [2025] * 2 + [2026],
            "player_id": ["w", "x", "y", "z", "a", "b", "later"],
            "history_weeks": [1] * 7,
            "history_carries": [10.0, 20.0, 30.0, 40.0, 100.0, 20.0, 999999.0],
            "history_targets": [0.0] * 7,
        }
    )
    result = prior_usage_cohorts(panel, paired(observations(), "points"), "RB")[0]
    assert result["thresholds"] == [{"year": 2025, "upper_quartile_cutoff": 32.5}]
    assert result["n"] == 1
    assert result["model_mse"] == 1
    assert result["baseline_mse"] == 16
