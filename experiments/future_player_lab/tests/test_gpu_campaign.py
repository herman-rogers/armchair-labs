"""Research validity tests with deliberately adversarial chronology/missing labels."""

from types import SimpleNamespace

import numpy as np
import polars as pl
from experiments.future_player_lab.gpu_lab.analyze import (
    choose_members,
    convex_weights,
    diverse_members,
    metric_values,
    stat_points,
)
from experiments.future_player_lab.gpu_lab.refit import predict_pipeline
from experiments.future_player_lab.gpu_lab.run import fold_masks, stat_arrays


def test_future_validation_errors_cannot_change_historical_selection():
    records = []
    for year in range(2018, 2026):
        for recipe, error in [("earlier_winner", 1.0), ("future_winner", 3.0)]:
            records.append({"year": year, "recipe": recipe, "normalized_mse": error})
    annual = pl.DataFrame(records)
    expected = choose_members(annual, 2021)
    altered = annual.with_columns(
        pl.when((pl.col("year") >= 2021) & (pl.col("recipe") == "future_winner"))
        .then(-100000.0)
        .otherwise(pl.col("normalized_mse"))
        .alias("normalized_mse")
    )
    assert choose_members(altered, 2021) == expected
    assert expected[0]["recipe"] == "earlier_winner"


def test_incomplete_candidate_history_is_not_selected():
    annual = pl.DataFrame(
        {
            "year": [2018, 2019, 2020, 2020],
            "recipe": ["complete"] * 3 + ["lucky_single"],
            "normalized_mse": [2.0, 2.0, 2.0, 0.0],
        }
    )
    assert [r["recipe"] for r in choose_members(annual, 2021)] == ["complete"]


def test_folds_keep_all_origins_of_a_season_together():
    years = np.array([2017, 2018, 2019, 2019, 2020, 2020, 2021])
    train, test = fold_masks(years, 2020, history=2)
    assert years[train].tolist() == [2018, 2019, 2019]
    assert years[test].tolist() == [2020, 2020]
    assert not np.any(train & test)


def test_points_propagate_only_relevant_unknown_stats():
    y = np.array([[np.nan, 100, 1, np.nan, 10, 50, 0], [5, np.nan, 0, 2, 2, 12, 0.0]])
    result, baseline = stat_arrays(SimpleNamespace(targets=y, baseline=y))
    assert result[0, -1] == 31
    assert np.isnan(result[1, -1])
    np.testing.assert_array_equal(result, baseline)


def test_metrics_exclude_unknown_labels_but_keep_known_zero():
    result = metric_values(np.array([0.0, 10.0, np.nan]), np.array([2.0, 8.0, 900.0]))
    assert result["n"] == 2
    assert result["mse"] == 4.0


def test_stat_points_use_exact_rule_and_reject_incomplete_components():
    records = []
    for player in ["known", "unknown"]:
        for target, value in [
            ("rushing_yards", 100.0),
            ("rushing_tds", 1.0),
            ("receptions", 10.0),
            ("receiving_yards", 50.0),
            ("receiving_tds", 0.0),
        ]:
            records.append(
                {
                    "player_id": player,
                    "horizon": "season",
                    "year": 2021,
                    "origin": 0,
                    "recipe": "selected",
                    "weeks": 18,
                    "training_max_year": 2020,
                    "target": target,
                    "actual": np.nan if player == "unknown" and target == "receptions" else value,
                    "prediction": value,
                }
            )
    result = stat_points(pl.DataFrame(records))
    assert result.height == 2
    known = result.filter(pl.col("player_id") == "known")
    unknown = result.filter(pl.col("player_id") == "unknown")
    assert known["actual"][0] == 31
    assert known["prediction"][0] == 31
    assert unknown["actual"][0] is None
    assert unknown["prediction"][0] == 31


def test_persisted_pipeline_reproduces_count_support_without_clipping_signed_points():
    count = {"constant_rate": -2.0, "target": "carries"}
    signed = {"constant_rate": -2.0, "target": "points"}
    np.testing.assert_array_equal(predict_pipeline(count, None, [1, 18]), [0, 0])
    np.testing.assert_array_equal(predict_pipeline(signed, None, [1, 18]), [-2, -36])


def test_convex_blend_uses_aligned_player_keys_and_can_recover_complementary_errors():
    records = []
    for year in [2018, 2019, 2020]:
        for player, actual in enumerate([1.0, 4.0, 10.0, 15.0]):
            for recipe, error in [("a", 3.0), ("b", -3.0)]:
                records.append(
                    {
                        "horizon": "season",
                        "target": "points",
                        "year": year,
                        "origin": 0,
                        "player_id": str(player),
                        "recipe": recipe,
                        "actual": actual,
                        "prediction": actual + error,
                    }
                )
    frame = pl.DataFrame(records)
    np.testing.assert_allclose(convex_weights(frame.reverse(), ["a", "b"]), [0.5, 0.5])


def test_diverse_ensemble_does_not_duplicate_the_best_family():
    ranked = [{"recipe": name} for name in ["a", "b", "c", "d"]]
    config = {
        "recipes": [
            {"id": name, "family": family}
            for name, family in [("a", "xgb"), ("b", "xgb"), ("c", "cat"), ("d", "forest")]
        ]
    }
    assert diverse_members(ranked, config) == ["a", "c", "d"]
