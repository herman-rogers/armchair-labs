"""Temporal and statistical contracts for the isolated QB capacity experiment."""

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

SPEC = importlib.util.spec_from_file_location(
    "qb_boosting_sweep", Path(__file__).resolve().parents[1] / "research/qb_boosting_sweep.py"
)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def scores(configs, year, winner):
    loss = np.full(len(configs), 10.0)
    loss[winner] = 1.0
    return dict(season=year, mae=loss.tolist(), mse=(loss * 10).tolist())


def test_selection_ignores_test_and_future_outcomes():
    configs = sweep.configurations()
    old = [scores(configs, year, 0) for year in (2007, 2008, 2009)]
    chosen, detail = sweep.select_config(old, configs, 2010, "canonical")
    contaminated = old + [scores(configs, year, len(configs) - 1) for year in (2010, 2025)]
    assert chosen == 0
    assert sweep.select_config(contaminated, configs, 2010, "canonical") == (chosen, detail)
    assert max(detail["years"]) < 2010
    assert sweep.select_config(old, configs, 2009, "canonical")[0] == sweep.fixed_index(configs)


def test_mse_guardrail_rejects_mae_winner():
    configs = sweep.configurations()
    annual = [scores(configs, year, 0) for year in (2007, 2008, 2009)]
    for row in annual:
        row["mse"][0] = 1000.0
    selected, _ = sweep.select_config(annual, configs, 2010, "canonical")
    assert selected != 0


def test_search_is_unique_and_contains_exact_canonical_setup():
    configs = sweep.configurations()
    assert len(configs) == 195
    assert len({c["id"] for c in configs}) == len(configs)
    canonical = configs[sweep.fixed_index(configs)]
    assert canonical["max_iter"] == 120
    assert canonical["min_samples_leaf"] == 30
    assert {c["max_iter"] for c in configs} == set(sweep.STAGES)
    assert {c["max_depth"] for c in configs} == {1, 2, 3, 5, None}


def test_staged_prefix_matches_independent_fit_with_missing_values():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(100, 4))
    x[::9, 1] = np.nan
    y = np.nan_to_num(x[:, 1]) ** 2 + x[:, 0]
    params = dict(
        early_stopping=False,
        random_state=18,
        max_leaf_nodes=7,
        min_samples_leaf=5,
        max_features=0.75,
    )
    with threadpool_limits(limits=1):
        full = HistGradientBoostingRegressor(max_iter=30, **params).fit(x, y)
        shorter = HistGradientBoostingRegressor(max_iter=10, **params).fit(x, y)
        staged = list(full.staged_predict(x))[9]
        np.testing.assert_allclose(staged, shorter.predict(x), rtol=0, atol=0)


def test_equal_season_weighting_and_undefined_rate_coverage():
    rows = [dict(season=2020, actual=0.0, policy=2.0, baseline=3.0)]
    rows += [dict(season=2021, actual=0.0, policy=0.0, baseline=2.0) for _ in range(9)]
    rows += [dict(season=2021, actual=None, policy=100.0, baseline=100.0)]
    summary = sweep.paired_summary(rows, "policy", "baseline", "mae")
    assert summary["mae"] == 1.0
    assert summary["n"] == 10
    assert summary["gain"] == 1.5
    assert summary["reference_mae"] == 2.5


def test_holm_controls_whole_family():
    np.testing.assert_allclose(sweep.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
