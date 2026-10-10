"""Selection and saved-model behavior independent of expensive GPU fitting."""

import copy

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.campaigns import run


def annual_losses():
    return [
        {"target": "points", "recipe": recipe, "year": year, "mse": float(i + 1)}
        for i, recipe in enumerate(run.RECIPES)
        for year in (2018, 2019, 2020)
    ]


def test_selection_ignores_forecast_year_future_years_and_other_targets():
    annual = annual_losses()
    expected = list(run.RECIPES)[:2]
    for year in (2021, 2022, 2099):
        for i, recipe in enumerate(run.RECIPES):
            annual.append(
                {"target": "points", "recipe": recipe, "year": year, "mse": float(9999 - i * 1000)}
            )
    annual.extend([{**row, "target": "passing_yards", "mse": 0.0} for row in annual.copy()])
    assert run.select_members(annual, "points", 2021) == expected


def test_selection_requires_complete_prior_window_even_for_losing_recipe():
    annual = annual_losses()
    annual.pop()
    with pytest.raises(ValueError, match="Incomplete"):
        run.select_members(annual, "points", 2021)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -1.0])
def test_selection_rejects_invalid_annual_losses(bad):
    annual = annual_losses()
    annual[0]["mse"] = bad
    with pytest.raises(ValueError):
        run.select_members(annual, "points", 2021)


def test_selection_rejects_duplicate_year_losses():
    annual = annual_losses()
    annual.append(dict(annual[0]))
    with pytest.raises(ValueError):
        run.select_members(annual, "points", 2021)


def test_per_calendar_week_baseline_retains_unplayed_weeks():
    panel = pl.DataFrame(
        {
            **{f"history_{s}": [4.0] for s in run.STATS["WR"]},
            "history_weeks": [2],
            "history_games": [1],
            "weeks": [16],
        }
    )
    assert run.baseline_values(panel, "WR", "targets")[0] == 32


class ConstantRate:
    def predict(self, x):
        return np.ones(len(x))


def saved_fixture():
    model = {
        "position": "WR",
        "horizon": "remaining",
        "origin_week": 2,
        "training_through_season": 2025,
        "feature_names": ["age"],
        "models": {
            s: [{"indices": [0], "model": ConstantRate()}] for s in [*run.STATS["WR"], "points"]
        },
    }
    panel = pl.DataFrame(
        {
            "year": [2026],
            "origin": [2],
            "weeks": [16],
            "position": ["WR"],
            "horizon": ["remaining"],
            "age": [24.0],
        }
    )
    return model, panel


@pytest.mark.parametrize(
    "field,value", [("year", 2025), ("origin", 4), ("position", "QB"), ("horizon", "season")]
)
def test_prediction_rejects_wrong_saved_model_contract(field, value):
    model, panel = saved_fixture()
    with pytest.raises(ValueError):
        run.predict(model, panel.with_columns(pl.lit(value).alias(field)))


def test_predictions_return_direct_and_component_points_without_refitting():
    model, panel = saved_fixture()
    original = copy.deepcopy(model)
    predictions = run.predict(model, panel)
    assert predictions["points"][0] == 16.0
    assert predictions["component_points"][0] == pytest.approx(211.2)
    assert model.keys() == original.keys()
