"""Profiles may describe all data; forecasting cohorts must remain chronological."""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.gpu_lab.profiles import (
    chronological_profiles,
    descriptive_profiles,
    fit_role_profiles,
    summarize_conditional_errors,
)
from threadpoolctl import threadpool_limits


def _dataset():
    rng = np.random.default_rng(104)
    x = np.column_stack([rng.normal(size=(90, 4)), np.repeat([-5, 0, 5], 30)])
    x[::7, 1] = np.nan
    return SimpleNamespace(
        x=x,
        names=[f"prior__box__{i}" for i in range(5)],
        targets=rng.poisson(10, (90, 7)).astype(float),
        panel=pl.DataFrame(
            {
                "player_id": [f"p{i}" for i in range(90)],
                "year": np.repeat([2021, 2022, 2023], 30),
                "origin": [2] * 90,
                "horizon": ["remaining"] * 90,
            }
        ),
    )


@pytest.mark.parametrize("method", ["kmeans", "gaussian_mixture"])
def test_future_input_changes_cannot_change_fitted_historical_role_vocabulary(method):
    d = _dataset()
    with threadpool_limits(limits=1):
        a = fit_role_profiles(d.x[:60], d.names, d.x[60:], k=3, seed=5, method=method)
        b = fit_role_profiles(d.x[:60], d.names, d.x[60:] * 1000, k=3, seed=5, method=method)
    np.testing.assert_array_equal(a["train_labels"], b["train_labels"])
    np.testing.assert_array_equal(a["clusterer"].cluster_centers_, b["clusterer"].cluster_centers_)
    np.testing.assert_array_equal(a["representation"].medians_, b["representation"].medians_)


def test_chronological_assignments_have_strict_training_cutoff_and_unique_keys():
    d = _dataset()
    with threadpool_limits(limits=1):
        rows, folds = chronological_profiles(d, [2022, 2023], k=3)
    assert rows.height == 60
    assert (rows["training_max_year"] < rows["year"]).all()
    assert not rows.select("horizon", "player_id", "year", "origin").is_duplicated().any()
    assert [f["training_rows"] for f in folds] == [30, 60]
    assert set(rows["profile_fit_scope"]) == {"earlier_years_only"}


@pytest.mark.parametrize("method", ["kmeans", "gaussian_mixture"])
def test_changing_outcomes_changes_descriptions_but_never_clusters(method):
    d = _dataset()
    with threadpool_limits(limits=1):
        before = descriptive_profiles(d, ks=(3,), seed=4, method=method)
        d.targets[:] = 900
        after = descriptive_profiles(d, ks=(3,), seed=4, method=method)
    assert before["assignments"].equals(after["assignments"])
    first = before["summary"][0]["profiles"][0]["descriptive_future_stat_means"]
    second = after["summary"][0]["profiles"][0]["descriptive_future_stat_means"]
    assert first != second
    assert second == [900] * 7
    assert sum(before["summary"][0]["cluster_sizes"]) == 90
    assert -1 <= before["summary"][0]["cross_seed_mean_adjusted_rand"] <= 1


def test_conditional_skill_compares_identical_players_inside_each_annual_role():
    common = dict(
        horizon=["remaining"] * 4, year=[2023] * 4, origin=[2] * 4, player_id=["a", "b", "c", "d"]
    )
    assignments = pl.DataFrame(
        {
            **common,
            "profile": [0, 0, 1, 1],
            "training_max_year": [2022] * 4,
            "profile_fit_scope": ["earlier_years_only"] * 4,
        }
    )
    rows = pl.concat(
        [
            pl.DataFrame(
                {
                    **common,
                    "target": ["carries"] * 4,
                    "actual": [0.0, 4.0, 10.0, 20.0],
                    "prediction": p,
                    "recipe": [name] * 4,
                }
            )
            for name, p in [
                ("selected", [0.0, 2.0, 8.0, 15.0]),
                ("persistence", [2.0, 2.0, 8.0, 18.0]),
            ]
        ]
    )
    summary = summarize_conditional_errors(rows, assignments)
    selected = summary.filter(pl.col("recipe") == "selected").sort("profile")
    assert selected["n"].to_list() == [2, 2]
    assert selected["mse_improvement_pct"].to_list() == [50.0, -262.5]
    invalid = assignments.with_columns(pl.lit(2023).alias("training_max_year"))
    with pytest.raises(ValueError, match="strictly earlier"):
        summarize_conditional_errors(rows, invalid)
