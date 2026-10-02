import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.deep.analyze import chronological, matched
from experiments.future_player_lab.deep.data import inventory_columns, verified_files
from experiments.future_player_lab.deep.diagnostics import paired_scores
from experiments.future_player_lab.deep.ensemble import (
    balanced_weights,
    capture_score,
    combine,
    convex_weights,
    greedy_weights,
    select_policy,
)
from experiments.future_player_lab.deep.models import Discover, fit_path, paths
from experiments.future_player_lab.deep.parallel_folds import fold_config


def test_complementary_models_enter_ensemble():
    y = np.arange(60.0)
    p = np.column_stack([y + 2, y - 4, y + 3])
    w = np.ones(60) / 60
    weights = convex_weights(p, y, w)
    assert weights[1] > 0.2  # Worst standalone candidate cancels the better one's errors.
    assert np.max(abs(p @ weights - y)) < 1e-3
    greedy = greedy_weights(p, y, w, [100])[100]
    assert np.mean((p @ greedy - y) ** 2) < 0.01


def test_seasons_and_origins_weight_equally():
    w = balanced_weights(np.array([2020, 2020, 2021]), np.array([2, 2, 2]))
    np.testing.assert_allclose(w, [0.25, 0.25, 0.5])


@pytest.mark.parametrize("adaptive", [False, True])
def test_parallel_annual_folds_preserve_numerical_predictions(tmp_path, adaptive):
    from experiments.future_player_lab.deep.capacity import task
    from experiments.future_player_lab.deep.run import fit_job

    rng = np.random.default_rng(12)
    years = np.repeat(np.arange(2015, 2020), 30)
    x = rng.normal(size=(len(years), 5)).astype(np.float32)
    y = np.column_stack([x[:, 0] ** 2 + x[:, 1]] * 5)
    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame(
        dict(
            position=["QB"] * len(years),
            horizon=["season"] * len(years),
            year=years,
            exposure=[17] * len(years),
        )
    ).write_parquet(data / "panel.parquet")
    np.save(data / "x.npy", x)
    np.save(data / "targets.npy", y)
    (data / "features.json").write_text(json.dumps({"views": {"all": list(range(5))}}))
    config = dict(
        seed=12, tree_checkpoints=[3, 7], warmup_start=2018, evaluation_years=[2018, 2019]
    )
    recipe = dict(paths(config)[0], view="all", leaf_samples=5)

    def fit(output, settings):
        if adaptive:
            task(str(output), str(data), "QB", "season", "lgb", "all", settings)
        else:
            fit_job(str(data), str(output), "QB", "season", recipe, settings)

    fit(tmp_path / "serial", config)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = []
        for year in [2018, 2019]:
            if adaptive:
                futures.append(
                    pool.submit(
                        task,
                        str(tmp_path / "folds"),
                        str(data),
                        "QB",
                        "season",
                        "lgb",
                        "all",
                        fold_config(config, year),
                    )
                )
            else:
                futures.append(
                    pool.submit(
                        fit_job,
                        str(data),
                        str(tmp_path / "folds"),
                        "QB",
                        "season",
                        recipe,
                        fold_config(config, year),
                    )
                )
        for future in futures:
            future.result()
    for file in (tmp_path / "serial" / "fits").rglob("*.npz"):
        other = tmp_path / "folds" / file.relative_to(tmp_path / "serial")
        with np.load(file) as a, np.load(other) as b:
            assert set(a.files) == set(b.files)
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name])
        first = json.loads(file.with_suffix(".json").read_text())
        second = json.loads(other.with_suffix(".json").read_text())
        assert first["year"] == second["year"]
        assert first["train_last_year"] == second["train_last_year"]
    assert config["warmup_start"] == 2018 and config["evaluation_years"] == [2018, 2019]


def test_capture_uses_position_cutoff_and_preserves_negative_points():
    y = np.array([10.0, 0.0, -5.0])
    p = np.array([3.0, 1.0, 2.0])
    groups = np.zeros(3)
    assert capture_score(y, p, groups, 1) == 1.0
    assert capture_score(y, p, groups, 2) == 0.5
    forecasts = np.column_stack([p, [3.0, 2.0, 1.0]])
    weights = greedy_weights(forecasts, y, np.ones(3) / 3, [5], "capture", groups, top_k=2)[5]
    assert capture_score(y, forecasts @ weights, groups, 2) == 1.0
    history = {year: {"a": {"capture": 0.5}, "b": {"capture": 0.8}} for year in range(2016, 2019)}
    assert select_policy(history, 2019, "capture")[0] == "b"


def test_capacity_comparisons_pair_seasons_and_reject_missing_coverage():
    metrics = pl.DataFrame(
        {
            "model": ["small", "small", "large", "large"],
            "year": [2020, 2021, 2021, 2020],
            "n": [10, 100, 100, 10],
            "mse": [4.0, 16.0, 8.0, 2.0],
            "mae": [2.0, 4.0, 3.0, 1.0],
            "ndcg24": [0.8, 0.8, 0.9, 0.9],
        }
    )
    result = paired_scores(metrics, "large", "small")
    assert result["mse_reduction_pct"] == 50
    assert result["mae_gain"] == 1
    assert result["years_won"] == 2
    with pytest.raises(ValueError, match="identical seasons"):
        paired_scores(metrics.head(3), "large", "small")


def test_selection_cannot_read_current_or_future_losses():
    h = {y: {"a": {"mse": 2}, "b": {"mse": 3}} for y in range(2016, 2026)}
    before = select_policy(h, 2019, "mse")
    h[2019]["a"]["mse"] = 1e99
    h[2025]["b"]["mse"] = -1e99
    assert select_policy(h, 2019, "mse") == before


def test_all_combination_methods_predict_and_preserve_convex_bounds():
    rng = np.random.default_rng(12)
    y = rng.normal(size=120)
    p = y[:, None] + rng.normal(size=(120, 5))
    years = np.repeat([2016, 2017, 2018], 40)
    context = np.column_stack([np.zeros(120), np.zeros(120), np.arange(120)])
    result, detail = combine(
        p, y, years, np.zeros(120), p[:12], np.zeros(12), context, context[:12], list("abcde")
    )
    assert len(result) >= 25
    for name, pred in result.items():
        assert np.isfinite(pred).all()
        if "weights" in detail[name]:
            assert np.all(pred >= p[:12].min(axis=1) - 1e-8)
            assert np.all(pred <= p[:12].max(axis=1) + 1e-8)


@pytest.mark.parametrize("family", ["lgb", "xgb", "cat", "hist", "extra", "forest"])
def test_capacity_prefix_matches_separate_fit(family):
    config = {"seed": 3, "tree_checkpoints": [3, 7]}
    path = next(p for p in paths(config) if p["family"] == family)
    path = dict(path, representation="identity", depth=3)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 6))
    x[::7, 1] = np.nan
    y = np.column_stack([x[:, 0] ** 2] * 5)
    args = (x[:80], y[:80], np.ones(80), np.repeat([2017, 2018], 40), x[80:], np.ones(20))
    multiple, _ = fit_path(path, config, *args)
    single, _ = fit_path(path, {**config, "tree_checkpoints": [3]}, *args)
    np.testing.assert_allclose(
        multiple[path["name"] + "_t3"], single[path["name"] + "_t3"], atol=1e-8
    )


def test_discovery_never_refits_on_prediction_batch():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 8))
    for method in ["selected", "interactions", "leaf"]:
        d = Discover(method, 2).fit(x, x[:, 0] * x[:, 1], np.ones(80))
        one = d.transform(x[:1])
        many = d.transform(np.vstack([x[:1], np.full((3, 8), 1e9)]))
        np.testing.assert_allclose(one[0], many[0])


@pytest.mark.parametrize("family", ["pls", "neural_tree", "hurdle", "multi_extra"])
def test_auxiliary_adapters_keep_five_target_contract(family):
    config = {"seed": 3, "tree_checkpoints": [3]}
    path = next(p for p in paths(config) if p["family"] == family)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 8))
    y = np.abs(rng.normal(size=(80, 5)))
    y[::3, 4] = 0
    y[:10, 1] = np.nan
    out, _ = fit_path(
        path, config, x[:60], y[:60], np.ones(60), np.repeat([2017, 2018], 30), x[60:], np.ones(20)
    )
    assert out[family].shape == (20,)
    assert np.isfinite(out[family]).all()


def test_admissibility_does_not_use_predictive_screen():
    f = pl.DataFrame({"good": [0.0, 1.0], "constant": [1.0, 1.0], "actual_points": [1.0, 3.0]})
    r = [
        {"stat": "good", "status": "admitted_research_only", "screened_comparisons": 0},
        {"stat": "constant", "status": "constant"},
        {"stat": "actual_points", "status": "outcome_only"},
    ]
    assert [x["stat"] for x in inventory_columns(r, f)] == ["good", "constant"]


def test_changed_source_rejected(tmp_path):
    from experiments.future_player_lab.data import digest

    p = tmp_path / "input"
    p.mkdir()
    (p / "x").write_text("correct")
    (p / "manifest.json").write_text(json.dumps({"files": {"x": digest(p / "x")}}))
    pins = {"input/manifest.json": digest(p / "manifest.json")}
    verified_files(tmp_path, "input", ["x"], pins)
    (p / "x").write_text("wrong")
    with pytest.raises(ValueError, match="Source file changed"):
        verified_files(tmp_path, "input", ["x"], pins)


def test_nested_ensemble_forecast_and_interval_ignore_current_labels():
    rng = np.random.default_rng(4)
    years = np.repeat(np.arange(2013, 2021), 12)
    n = len(years)
    y = rng.uniform(0, 100, n)
    p = y[:, None] + rng.normal(0, 10, (n, 5))
    panel = pl.DataFrame(
        dict(
            player_id=[str(i % 12) for i in range(n)],
            position=["QB"] * n,
            year=years,
            origin=[2] * n,
            horizon=["next_four"] * n,
            exposure=[4] * n,
            end=[6] * n,
            name=["test"] * n,
            population=["returner"] * n,
        )
    )
    context = np.column_stack([np.full(n, 2), np.zeros(n), np.full(n, 20)])
    config = dict(
        ensemble_start=2016,
        inner_seasons=3,
        seed=4,
        tree_checkpoints=[40, 120, 360],
        evaluation_years=[2019, 2020],
    )
    a = chronological(panel, y, p, context, list("abcde"), config)
    changed = y.copy()
    changed[years == 2020] += 1e6
    b = chronological(panel, changed, p, context, list("abcde"), config)
    for field in ["prediction", "low80", "high80"]:
        np.testing.assert_allclose(a[0][field].to_numpy(), b[0][field].to_numpy(), equal_nan=True)
    assert a[2] == b[2]


def test_matched_comparison_rejects_different_outcomes_and_horizons():
    frame = pl.DataFrame(
        dict(
            player_id=["a"],
            position=["QB"],
            year=[2025],
            origin=[2],
            horizon=["next_four"],
            end=[6],
            actual=[5.0],
            prediction=[6.0],
        )
    )
    base = frame.select("player_id", "position", "year", "origin", "horizon").with_columns(
        pl.lit(6).alias("published_end"),
        pl.lit(5.0).alias("published_actual"),
        pl.lit(4.0).alias("published"),
    )
    assert matched(frame, base).height == 1
    with pytest.raises(ValueError, match="outcomes disagree"):
        matched(frame, base.with_columns(pl.lit(7.0).alias("published_actual")))
    with pytest.raises(ValueError, match="horizons differ"):
        matched(frame, base.with_columns(pl.lit(7).alias("published_end")))


def test_benchmark_keeps_points_after_position_change():
    from experiments.future_player_lab.deep.benchmark import reconcile_published

    benchmark = pl.DataFrame(
        dict(
            player_id=["a"],
            position=["WR"],
            year=[2025],
            origin=[2],
            horizon=["next_four"],
            published_end=[6],
            published_actual=[2.0],
            published=[12.0],
        )
    )
    weeks = pl.DataFrame(
        dict(
            player_id=["a"] * 3,
            season=[2025] * 3,
            week=[1, 3, 4],
            position=["WR", "WR", "DB"],
            season_type=["REG"] * 3,
            league_points=[50.0, 2.0, 7.0],
        )
    )
    fixed, changes = reconcile_published(benchmark, weeks)
    assert fixed["published_actual"].item() == 9.0
    assert fixed["published_original_actual"].item() == 2.0
    assert fixed["published"].item() == 12.0
    assert len(changes) == 1
    with pytest.raises(ValueError, match="other than"):
        reconcile_published(benchmark.with_columns(pl.lit(99.0).alias("published_actual")), weeks)
    unknown = benchmark.with_columns(pl.lit(None, dtype=pl.Float64).alias("published_actual"))
    pending, changes = reconcile_published(unknown, weeks)
    assert pending["published_actual"].null_count() == 1
    assert changes == []


@pytest.mark.parametrize("engine", ["lgb", "xgb", "cat"])
def test_adaptive_capacity_uses_earlier_season_then_refits(engine):
    from experiments.future_player_lab.deep.capacity import fit_adaptive

    rng = np.random.default_rng(19)
    x = rng.normal(size=(120, 8))
    years = np.repeat([2016, 2017, 2018], 40)
    y = x[:, 0] ** 2 + rng.normal(size=120)
    x[:80, 7] = np.nan
    pred, diag = fit_adaptive(engine, x, y, years, x[:8], ceiling=12, patience=3)
    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert diag["validation_year"] == 2018
    assert diag["inner_train_last_year"] == 2017
    assert 1 <= diag["trees"] <= 12
