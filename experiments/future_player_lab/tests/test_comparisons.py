import polars as pl
import pytest
from experiments.future_player_lab.tools.compare import compare


def test_comparison_uses_identical_rows_when_baseline_is_missing():
    frame = pl.DataFrame(
        {
            "player_id": ["a", "b", "a", "b"],
            "year": [2025] * 4,
            "origin": [0] * 4,
            "horizon": ["season"] * 4,
            "position": ["WR"] * 4,
            "model": ["candidate", "candidate", "control", "control"],
            "actual_workload": [10.0, 1000.0, 10.0, 1000.0],
            "pred_workload": [12.0, 0.0, 15.0, None],
        }
    )
    result = compare(frame, "candidate", "control", "workload")
    assert result[0]["n"] == 1
    assert result[0]["delta_mse"] == 4 - 25


def test_comparison_rejects_inconsistent_actuals():
    frame = pl.DataFrame(
        {
            "player_id": ["a", "a"],
            "year": [2025] * 2,
            "origin": [0] * 2,
            "horizon": ["season"] * 2,
            "position": ["QB"] * 2,
            "model": ["candidate", "control"],
            "actual_points": [10.0, 11.0],
            "pred_points": [12.0, 12.0],
        }
    )
    with pytest.raises(ValueError, match="disagree"):
        compare(frame, "candidate", "control", "points")


def test_absent_control_does_not_create_fake_comparisons():
    frame = pl.DataFrame(
        {
            "player_id": ["a"],
            "year": [2025],
            "origin": [0],
            "horizon": ["season"],
            "position": ["QB"],
            "model": ["candidate"],
            "actual_points": [10.0],
            "pred_points": [12.0],
        }
    )
    assert compare(frame, "candidate", "missing", "points") == []
