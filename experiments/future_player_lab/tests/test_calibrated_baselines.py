"""Simple recalibration must preserve chronological and missing-value contracts."""

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.gpu_lab.calibrated_baselines import (
    AffinePersistence,
    compare_predictions,
    fit_fold,
    select_affine,
)


def test_fold_ignores_future_labels_and_window_excludes_older_rows():
    years = np.array([2018, 2019, 2020, 2021, 2022, 2023])
    x = np.arange(6, dtype=float)
    y = np.array([999, 1, 3, 5, 7, 9], dtype=float)
    model, metadata = fit_fold(years, x, y, 2022, window=3)
    assert metadata["training_min_year"] == 2019
    assert metadata["training_max_year"] == 2021
    assert model.slope_ == pytest.approx(2)
    changed = y.copy()
    changed[years >= 2022] = 1e12
    again, other = fit_fold(years, x, changed, 2022, window=3)
    assert other == metadata
    np.testing.assert_allclose(again.predict(x), model.predict(x))


def test_unknown_predictors_and_labels_are_excluded_without_zero_imputation():
    model = AffinePersistence().fit([0, 1, np.nan, 4, np.inf], [2, 5, 1000, np.nan, 999])
    assert model.training_rows_ == 2
    assert model.dropped_rows_ == 3
    np.testing.assert_allclose(model.predict([0, 2]), [2, 8])
    assert np.isnan(model.predict([np.nan, np.inf])).all()
    with pytest.raises(ValueError, match="No observed"):
        AffinePersistence().fit([np.nan], [1])


def test_nonnegative_slope_and_count_only_output_clipping_preserve_signed_support():
    inverted = AffinePersistence().fit([0, 1, 2], [2, 1, 0])
    assert inverted.slope_ == 0
    assert inverted.intercept_ == 1
    signed = AffinePersistence().fit([0, 1, 2], [-3, -1, 1])
    np.testing.assert_allclose(signed.predict([-1, 0]), [-5, -3])
    np.testing.assert_allclose(signed.predict([-1, 0], count_stat=True), [0, 0])
    constant = AffinePersistence().fit([0, 0], [2, 4])
    assert constant.slope_ == 0 and constant.intercept_ == 3


def test_affine_selection_uses_only_three_strictly_earlier_validation_years():
    records = []
    for year in range(2017, 2023):
        for recipe in ("affine_expanding", "affine_recent3"):
            for player, y in enumerate([0.0, 10.0]):
                # Recent3 wins 2018–20; expanding wins 2017 and the future.
                good = (recipe == "affine_recent3") == (2018 <= year <= 2020)
                records.append(
                    {
                        "horizon": "season",
                        "year": year,
                        "origin": 0,
                        "player_id": str(player),
                        "target": "points",
                        "recipe": recipe,
                        "actual": y,
                        "prediction": y if good else y + 4,
                    }
                )
    frame = pl.DataFrame(records)
    selected, choices, _ = select_affine(frame, [2021])
    choice = choices.row(0, named=True)
    assert choice["member"] == "affine_recent3"
    assert choice["selection_min_year"] == 2018 and choice["selection_max_year"] == 2020
    changed = frame.with_columns(
        pl.when(pl.col("year") >= 2021).then(1e9).otherwise(pl.col("actual")).alias("actual")
    )
    changed_selected, changed_choices, _ = select_affine(changed, [2021])
    assert changed_choices.to_dicts() == choices.to_dicts()
    assert selected["prediction"].to_list() == changed_selected["prediction"].to_list()


def test_matched_comparisons_exclude_unknowns_and_reject_disagreeing_actuals():
    rows = []
    for recipe, predictions in [("persistence", [2.0, np.nan]), ("affine_selected", [1.0, 2.0])]:
        for player, prediction in enumerate(predictions):
            rows.append(
                {
                    "horizon": "season",
                    "year": 2021,
                    "origin": 0,
                    "player_id": str(player),
                    "target": "points",
                    "recipe": recipe,
                    "actual": 0.0,
                    "prediction": prediction,
                }
            )
    frame = pl.DataFrame(rows)
    result = compare_predictions(frame, ["persistence"]).row(0, named=True)
    assert result["n"] == 1 and result["mse_improvement_pct"] == 75
    assert result["delta_mae"] == -1 and result["delta_mse_low95"] is None
    wrong = frame.with_columns(
        pl.when(pl.col("recipe") == "affine_selected")
        .then(100)
        .otherwise(pl.col("actual"))
        .alias("actual")
    )
    with pytest.raises(ValueError, match="outcomes disagree"):
        compare_predictions(wrong, ["persistence"])
