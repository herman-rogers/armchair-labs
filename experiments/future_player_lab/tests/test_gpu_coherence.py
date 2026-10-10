"""Count consistency must respect unknowns and the possibility of negative yardage."""

import math

import polars as pl
import pytest
from experiments.future_player_lab.gpu_lab.coherence import summarize


def test_expected_count_relations_and_unknown_denominators():
    frame = pl.DataFrame(
        {
            "horizon": ["season"] * 3,
            "carries": [2.0, 1.0, 0.0],
            "rushing_tds": [3.0, 0.0, 0.0],
            "targets": [2.0, float("nan"), 0.0],
            "receptions": [3.0, 5.0, 0.0],
            "receiving_tds": [0.0, 6.0, 0.0],
            "rushing_yards": [-2.0, 5.0, 0.0],
            "receiving_yards": [-1.0, 6.0, 0.0],
            "points": [20.7, 42.1, 0.0],
        }
    )
    result = {r["check"]: r for r in summarize(frame, ["horizon"]).to_dicts()}
    targets = result["receptions_exceed_targets"]
    assert targets["n"] == 2
    assert targets["violations"] == 1
    assert targets["violation_pct"] == 50
    assert result["rushing_tds_exceed_carries"]["violations"] == 1
    assert result["receiving_tds_exceed_receptions"]["violations"] == 1
    assert result["direct_points_minus_component_means"][
        "mean_absolute_difference"
    ] == pytest.approx(0)


def test_missing_points_components_are_not_summed_as_zero():
    frame = pl.DataFrame(
        {
            "horizon": ["season"],
            "carries": [0.0],
            "rushing_tds": [0.0],
            "targets": [0.0],
            "receptions": [0.0],
            "receiving_tds": [0.0],
            "rushing_yards": [math.nan],
            "receiving_yards": [0.0],
            "points": [0.0],
        }
    )
    result = (
        summarize(frame, ["horizon"])
        .filter(pl.col("check") == "direct_points_minus_component_means")
        .row(0, named=True)
    )
    assert result["n"] == 0
    assert result["mean_absolute_difference"] is None
