"""Fair-comparison and integrity checks for campaign result comparisons."""

import json

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.campaigns.compare_rb import (
    compare_errors,
    digest,
    paired_rows,
    verify_files,
)


def frame():
    return pl.DataFrame(
        {
            "horizon": ["remaining", "remaining"],
            "year": [2021, 2021],
            "origin": [2, 2],
            "player_id": ["one", "two"],
            "target": ["points", "points"],
            "player_display_name": ["One", "Two"],
            "actual": [4.0, 6.0],
            "prediction": [6.0, 8.0],
        }
    )


def test_same_outcomes_and_population_produce_distinct_mse_rmse_improvements():
    original = frame()
    newer = original.with_columns(pl.Series("prediction", [5.0, 7.0]))
    result = compare_errors(paired_rows(original, newer))
    assert result["n"] == 2
    assert result["mse_improvement_percent"] == 75
    assert result["rmse_improvement_percent"] == 50


def test_missing_candidate_is_not_silently_dropped():
    with pytest.raises(ValueError, match="populations differ"):
        paired_rows(frame(), frame().head(1))


def test_duplicate_forecasts_are_not_arbitrarily_matched():
    with pytest.raises(ValueError, match="Duplicate"):
        paired_rows(frame(), pl.concat([frame(), frame().head(1)]))


def test_different_cutoff_cannot_be_joined_as_same_forecast():
    with pytest.raises(ValueError, match="populations differ"):
        paired_rows(frame(), frame().with_columns(pl.lit(3).alias("origin")))


def test_different_ground_truth_fails_but_float_summation_noise_is_tolerated():
    with pytest.raises(AssertionError):
        paired_rows(frame(), frame().with_columns((pl.col("actual") + 0.1).alias("actual")))
    assert (
        paired_rows(
            frame(), frame().with_columns((pl.col("actual") + 1e-13).alias("actual"))
        ).height
        == 2
    )


def test_unknown_outcome_disagreement_is_not_hidden_by_pairwise_drop():
    with pytest.raises(ValueError, match="availability differs"):
        paired_rows(frame(), frame().with_columns(pl.Series("actual", [4.0, np.nan])))


def test_comparison_uses_same_finite_prediction_rows_for_both_models():
    paired = pl.DataFrame(
        {"actual": [4.0, 6.0], "original": [6.0, np.nan], "opening_drive": [5.0, 100.0]}
    )
    result = compare_errors(paired)
    assert result["n"] == 1
    assert result["opening_drive_rmse"] == 1


def test_modified_artifact_fails_checksum_validation(tmp_path):
    asset = tmp_path / "predictions.json"
    asset.write_text("{}")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "files": {asset.name: digest(asset)},
            }
        )
    )
    verify_files(tmp_path)
    asset.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_files(tmp_path)
