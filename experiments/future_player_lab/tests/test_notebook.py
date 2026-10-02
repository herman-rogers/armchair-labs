"""Notebook-specific chronology, original-recipe replay and artifact contracts."""

import json

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.notebooks import workbench as w
from threadpoolctl import threadpool_limits


def synthetic_panel():
    rng = np.random.default_rng(72)
    years = np.repeat(np.arange(2010, 2026), 30)
    x = rng.normal(size=(len(years), 5))
    actual = np.maximum(0, 20 + 5 * x[:, 0] + 3 * x[:, 1] ** 2 + rng.normal(size=len(years)))
    panel = pl.DataFrame(
        dict(
            player_id=[str(i) for i in range(len(years))],
            name=[f"Player {i}" for i in range(len(years))],
            year=years,
            position=["TE"] * len(years),
            horizon=["remaining"] * len(years),
            origin=[2] * len(years),
            population=["returner"] * len(years),
            actual=actual,
            exposure=[10.0] * len(years),
        )
    )
    return panel, x


def test_browser_numeric_controls_are_canonicalized():
    c = w.validate_trial({"feature_fraction": 1, "l2": 10, "max_trees": 120.0})
    assert isinstance(c["feature_fraction"], float)
    assert isinstance(c["max_trees"], int)
    assert c == w.validate_trial(w.DEFAULT_TRIAL)
    with pytest.raises(ValueError, match="integer"):
        w.validate_trial({"max_trees": 12.5})


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_position_filter_preserves_context_and_relevant_stats(position):
    names = [
        "summary:4:mean:passing_yards",
        "inventory:career_pass_attempts_log",
        "college:1:passing_tds",
        "inventory:rich_passing_epa_rate_mean_yoy",
        "sequence:1:rushing_yards",
        "summary:4:mean:receiving_yards",
        "inventory:team_pass_volume",
        "inventory:rich_s0_team_dropbacks_mean",
        "static:age_at_season",
        "summary:4:mean:league_points",
    ]
    mask = w.feature_audit(names, position, "position")["included"].to_list()
    assert mask[:4] == [position == "QB"] * 4
    assert mask[4:] == [True, position != "QB", True, True, True, True]
    assert w.feature_audit(names, position, "all")["included"].all()


def test_excluded_passing_inputs_cannot_change_rb_forecasts():
    panel, x = synthetic_panel()
    panel = panel.with_columns(pl.lit("RB").alias("position"))
    names = ["rushing_yards", "receptions", "passing_yards", "age", "team_pass_volume"]
    altered = x.copy()
    altered[:, 2] = np.arange(len(x)) * 1000
    c = {**w.DEFAULT_TRIAL, "position": "RB", "feature_scope": "position", "max_trees": 12}
    with threadpool_limits(limits=1):
        a = w.compute_trial(panel, x, names, c)
        b = w.compute_trial(panel, altered, names, c)
    np.testing.assert_array_equal(a["forecasts"]["prediction"], b["forecasts"]["prediction"])
    assert a["metadata"]["excluded_features"] == ["passing_yards"]


def test_stat_labels_respect_cutoff_end_and_unknowns():
    panel = pl.DataFrame(
        dict(
            player_id=["a", "b", "c", "d"],
            year=[2025] * 4,
            origin=[2] * 4,
            end=[6] * 4,
            exposure=[4] * 4,
            actual=[1.0, 0.0, 1.0, None],
        )
    )
    weeks = pl.DataFrame(
        dict(
            player_id=["a"] * 4 + ["c"],
            season=[2025] * 5,
            week=[1, 3, 6, 7, 3],
            rushing_yards=[100, 20, 30, 999, None],
        )
    )
    result = w.stat_panel(panel, weeks, "rushing_yards")
    assert result["actual"].to_list() == [50.0, 0.0, None, None]
    assert result["persistence_prediction"][0] == 200
    # A future-label change must not change the cutoff-only persistence forecast.
    changed = weeks.with_columns(
        pl.when(pl.col("week") > 2)
        .then(12345)
        .otherwise(pl.col("rushing_yards"))
        .alias("rushing_yards")
    )
    assert w.stat_panel(panel, changed, "rushing_yards")["persistence_prediction"][0] == 200
    preseason = panel.head(1).with_columns(
        pl.lit(0).alias("origin"), pl.lit(18).alias("end"), pl.lit(18).alias("exposure")
    )
    prior = weeks.filter(pl.col("week") == 1).with_columns(
        pl.lit(2024, dtype=pl.Int64).alias("season")
    )
    assert (
        w.stat_panel(preseason, pl.concat([weeks, prior]), "rushing_yards")[
            "persistence_prediction"
        ][0]
        == 100
    )


def test_stat_targets_are_separate_from_points_and_original_presets():
    assert "passing_yards" not in w.POSITION_TARGETS["RB"]
    assert "targets" in w.POSITION_TARGETS["RB"]
    with pytest.raises(ValueError, match="Original presets"):
        w.validate_trial({"target": "receiving_yards", "view": "original_profile"})
    a, _ = w.trial_key({"target": "targets"})
    b, _ = w.trial_key({"target": "receptions"})
    assert a != b


@pytest.mark.parametrize("engine", ["hist", "lgb", "xgb", "cat"])
@pytest.mark.parametrize("target", ["points", "targets"])
def test_trial_future_labels_cannot_select_capacity_or_change_predictions(engine, target):
    panel, x = synthetic_panel()
    config = {
        **w.DEFAULT_TRIAL,
        "engine": engine,
        "target": target,
        "max_trees": 12,
        "leaf_samples": 5,
        "representation": "selected",
        "depth": 3,
    }
    altered = panel.with_columns(
        pl.when(pl.col("year") == 2025)
        .then(pl.col("actual") + 10000)
        .otherwise(pl.col("actual"))
        .alias("actual")
    )
    with threadpool_limits(limits=1):
        a = w.compute_trial(panel, x, list("abcde"), config)
        b = w.compute_trial(altered, x, list("abcde"), config)
    assert a["metadata"] == b["metadata"]
    np.testing.assert_array_equal(a["forecasts"]["prediction"], b["forecasts"]["prediction"])
    assert (
        a["capacity"]
        .filter(pl.col("split") != "Test")
        .equals(b["capacity"].filter(pl.col("split") != "Test"))
    )
    assert a["learning"]["last_train_year"].max() == 2023
    assert a["metadata"]["refit_last_year"] == 2024


@pytest.mark.parametrize("method", ["equal", "convex", "greedy"])
def test_blend_uses_prior_forecasts_only(method):
    panel, _ = synthetic_panel()
    frame = pl.concat(
        [
            panel.with_columns(
                (pl.col("actual") * 0.8 + 3).alias("prediction"), pl.lit("a").alias("model")
            ),
            panel.with_columns(
                (pl.col("actual") * 1.1 - 1).alias("prediction"), pl.lit("b").alias("model")
            ),
        ]
    )
    altered = frame.with_columns(
        pl.when(pl.col("year") == 2025)
        .then(pl.col("actual") + 10000)
        .otherwise(pl.col("actual"))
        .alias("actual")
    )
    a, wa, years = w.blend_trial(frame, 2025, method)
    b, wb, _ = w.blend_trial(altered, 2025, method)
    assert wa.equals(wb)
    np.testing.assert_array_equal(a["prediction"], b["prediction"])
    assert years == [2020, 2021, 2022, 2023, 2024]
    assert np.isclose(wa["weight"].sum(), 1)
    assert wa["weight"].min() >= 0


def test_artifact_change_is_detected_after_previous_read(tmp_path):
    file = tmp_path / "value.json"
    file.write_text("{}")
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(status="complete", artifacts={"value.json": w.digest(file)}))
    )
    w.checked(tmp_path, "value.json")
    file.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Changed artifact"):
        w.checked(tmp_path, "value.json")


@pytest.mark.skipif(not w.REPORT.exists(), reason="Local research artifacts required")
def test_notebook_comparison_reproduces_completed_report():
    f = w.comparison_frame("TE", "remaining", ["published", "combined:policy_mae"])
    summary = w.summary_scores(w.annual_scores(f), "published")
    actual = summary.filter(pl.col("model") == "combined:policy_mae").row(0, named=True)
    saved = next(
        r
        for r in w.read_json(w.REPORT / "published_library/comparisons.json")
        if r["position"] == "TE" and r["horizon"] == "remaining" and r["model"] == "policy_mae"
    )
    assert actual["rows"] == saved["n"]
    assert np.isclose(actual["mse_reduction_pct"], -saved["delta_mse_pct"])
    assert np.isclose(actual["mae_gain"], -saved["delta_mae"])


@pytest.mark.skipif(not w.REPORT.exists(), reason="Local research artifacts required")
@pytest.mark.parametrize(
    "view,model",
    [
        ("original_profile", "existing:profile_boost"),
        ("original_enriched", "existing:enriched_boost"),
    ],
)
def test_original_booster_replay_matches_saved_predictions(view, model):
    panel, x, _, original = w.sandbox_data("TE", "remaining", view)
    years = panel["year"].to_numpy()
    train, test = years < 2025, years == 2025
    assert years.min() == 2004  # Do not lose early training years in the benchmark join.
    c = {**w.DEFAULT_TRIAL, "view": view, "depth": 0}
    with threadpool_limits(limits=1):
        pred, _ = w.fit_prefixes(
            x[train],
            panel["training_actual"].to_numpy()[train],
            years[train],
            [x[test]],
            c,
            [120],
            original,
        )
    replay = (
        panel.filter(pl.Series(test)).select(w.KEYS).with_columns(pl.Series("replay", pred[120][1]))
    )
    saved = w.one_forecast("TE", "remaining", model).filter(pl.col("year") == 2025)
    matched = saved.join(replay, on=w.KEYS, validate="1:1")
    assert matched.height == saved.height
    np.testing.assert_allclose(matched["prediction"], matched["replay"], rtol=0, atol=1e-8)


@pytest.mark.skipif(not w.REPORT.exists(), reason="Local research artifacts required")
def test_saved_overfitting_curve_uses_same_rate_units():
    curve = w.saved_capacity("TE", "remaining", "lgb_04")
    panel, _ = w.task_data("TE", "remaining")
    forecast = w.one_forecast("TE", "remaining", "lgb_04_t120").filter(pl.col("year") == 2025)
    expected = np.sqrt(
        np.mean(
            (((forecast["prediction"] - forecast["actual"]) / forecast["exposure"]) ** 2).to_numpy()
        )
    )
    row = curve.filter(
        (pl.col("year") == 2025) & (pl.col("trees") == 120) & (pl.col("split") == "Later season")
    )
    assert np.isclose(row["rmse_rate"][0], expected)
    assert panel.height > forecast.height
