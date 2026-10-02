import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from threadpoolctl import threadpool_limits

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from data import STATIC, WEEK_FIELDS, inventory_columns, raw_history  # noqa: E402
from methods import FoldModel, Recipe, choose  # noqa: E402


def test_raw_history_uses_only_earlier_seasons_and_preserves_unknown():
    frame = pl.DataFrame(
        [
            {
                "player_id": "p",
                "forecast_season": 2020,
                "source_season": 2019,
                **{c: None for c in STATIC},
            }
        ]
    )
    records = []
    for year, week, value in [(2019, 1, 0), (2019, 2, 3), (2020, 1, 999), (2010, 1, 7)]:
        records.append(
            {
                "player_id": "p",
                "season": year,
                "week": week,
                "season_type": "REG",
                **{c: value for c in WEEK_FIELDS},
            }
        )
    records[1]["targets"] = None
    weeks = pl.DataFrame(records)
    x, names, _ = raw_history(frame, weeks)
    assert x[0, names.index("week_lag1_w01_targets")] == 0
    assert np.isnan(x[0, names.index("week_lag1_w02_targets")])
    assert np.isnan(x[0, names.index("week_lag1_w03_targets")])
    assert x[0, names.index("annual_lag10_targets")] == 7
    changed = weeks.with_columns(
        pl.when(pl.col("season") == 2020).then(-9876).otherwise(pl.col("targets")).alias("targets")
    )
    other, _, _ = raw_history(frame, changed)
    np.testing.assert_equal(x, other)


def test_registry_selection_ignores_screen_scores_and_aliases():
    base = [
        {
            "stat": "status",
            "status": "admitted_research_only",
            "market_input": False,
            "screened_comparisons": 0,
            "exact_alias_of": "other",
        },
        {"stat": "market_ecr", "status": "admitted_research_only", "market_input": True},
        {"stat": "actual_season_points", "status": "outcome_only", "market_input": False},
    ]
    assert inventory_columns(base) == ["status"]
    assert inventory_columns(base, market=True) == ["status", "market_ecr"]


def test_binary_interaction_survives_automated_selection_but_not_old_gate():
    x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float), (40, 1))
    y = 10 * (x[:, 0] != x[:, 1])
    with threadpool_limits(limits=1):
        with pytest.raises(ValueError, match="no candidates"):
            FoldModel(Recipe("old", "raw", "tree", "legacy", 15), 7).fit(x, y)
        model = FoldModel(Recipe("new", "raw", "svr", "selected", 10), 7).fit(x, y)
        assert set(model.indices) == {0, 1}
        assert np.mean(abs(model.predict(x) - y)) < 0.2


def test_missing_and_zero_are_distinct_and_test_cannot_fit_transform():
    x = np.array([[0, np.nan], [1, np.nan], [2, np.nan], [3, np.nan]])
    with threadpool_limits(limits=1):
        model = FoldModel(Recipe("r", "raw", "ridge", setting=100), 7).fit(x, np.arange(4.0))
    before = model.medians.copy(), model.mean.copy(), model.scale.copy()
    transformed = model._base(np.array([[0, 0], [np.nan, np.nan], [99999, 5]]))
    assert transformed[0, 2] == 0 and transformed[1, 2] == 1
    assert transformed[0, 3] == 0 and transformed[1, 3] == 1
    model.predict(np.array([[99999.0, 77777.0]]))
    for old, current in zip(before, (model.medians, model.mean, model.scale), strict=True):
        np.testing.assert_equal(old, current)


def test_chronological_selection_never_uses_current_or_future_losses():
    history = {
        "a": {2005: 1, 2006: 1, 2007: 1, 2008: 1e12},
        "b": {2005: 2, 2006: 2, 2007: 2, 2008: 0, 2025: 0},
    }
    winner, years, _ = choose(history, ["a", "b"], 2008)
    assert winner == "a"
    assert years == [2005, 2006, 2007]
    with pytest.raises(ValueError, match="Insufficient"):
        choose(history, ["a", "b"], 2007)


def test_pca_fits_only_training_and_predict_is_batch_invariant():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(80, 40))
    y = x[:, 0] ** 2
    with threadpool_limits(limits=1):
        model = FoldModel(Recipe("p", "raw", "svr", "pca", 10), 7).fit(x, y)
        a = model.predict(x[:1])
        b = model.predict(np.vstack((x[:1], np.full((1, 40), 1e9))))
    np.testing.assert_allclose(a, b[:1], atol=1e-9)
    assert model.pca.n_samples_ == 80
