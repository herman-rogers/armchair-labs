"""Regression tests for analysis replay, unknown labels and interrupted artifacts."""

import json
from types import SimpleNamespace

import joblib
import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.gpu_lab import analyze as analysis
from experiments.future_player_lab.gpu_lab import diagnostics
from experiments.future_player_lab.gpu_lab import refit as final
from experiments.future_player_lab.gpu_lab.diagnostics import controlled_comparisons
from experiments.future_player_lab.gpu_lab.figures import figures
from experiments.future_player_lab.gpu_lab.run import digest, json_write


class IdentityRepresentation:
    def transform(self, values):
        return values


class FirstColumnModel:
    def predict(self, values):
        return values[:, 0]


def _rows(recipes=("a", "b", "training_mean", "persistence")):
    records = []
    errors = {"a": 1.0, "b": -2.0, "training_mean": 4.0, "persistence": 6.0}
    for year in range(2018, 2022):
        for player in range(12):
            for recipe in recipes:
                actual = float(player + 5)
                records.append(
                    {
                        "horizon": "season",
                        "year": year,
                        "origin": 0,
                        "player_id": str(player),
                        "target": "carries",
                        "actual": actual,
                        "prediction": actual + errors[recipe],
                        "recipe": recipe,
                        "weeks": 17,
                        "training_max_year": year - 1,
                        "fit_seconds": 0.0,
                    }
                )
    return pl.DataFrame(records)


def _recipe(name, family="hist"):
    return {
        "id": name,
        "family": family,
        "device": "cpu",
        "history": 0,
        "view": "all",
        "seed": 17,
        "representation": {"kind": "raw"},
        "params": {"max_iter": 2, "min_samples_leaf": 2},
    }


def _setup_run(tmp_path):
    config = {
        "years": [2018, 2019, 2020, 2021],
        "evaluation_years": [2021],
        "selection_years": 3,
        "horizons": ["season"],
        "targets": ["carries"],
        "recipes": [_recipe("a"), _recipe("b")],
        "seed": 17,
        "threads": 1,
        "origins": [2],
        "input_cache": "fixture",
    }
    json_write(tmp_path / "config.json", config)
    rows = _rows()
    horizon = tmp_path / "season"
    horizon.mkdir()
    rows.filter(pl.col("recipe").is_in(["training_mean", "persistence"])).write_parquet(
        horizon / "baselines.parquet"
    )
    for recipe in ["a", "b"]:
        for year in config["years"]:
            checkpoint = horizon / "fits" / recipe / str(year)
            checkpoint.mkdir(parents=True)
            rows.filter((pl.col("recipe") == recipe) & (pl.col("year") == year)).write_parquet(
                checkpoint / "predictions.parquet"
            )
            json_write(
                checkpoint / "complete.json",
                {"files": {"predictions.parquet": digest(checkpoint / "predictions.parquet")}},
            )
    return config


def test_analysis_replay_is_deterministic_and_final_seal_covers_later_files(tmp_path):
    _setup_run(tmp_path)
    dest = analysis.analyze(tmp_path)
    first = {
        name: digest(dest / name)
        for name in [
            "annual_metrics.parquet",
            "predictions.parquet",
            "policy_selection.parquet",
            "distributions.parquet",
        ]
    }
    analysis.analyze(tmp_path)
    assert first == {name: digest(dest / name) for name in first}
    assert json.loads((dest / "manifest.json").read_text())["status"] == "core_complete"
    (dest / "figures").mkdir()
    (dest / "figures" / "test.svg").write_text("<svg />")
    (dest / "manifest.json.tmp").write_text("interrupted publication")
    analysis.seal_analysis(tmp_path)
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert "manifest.json.tmp" not in manifest["files"]
    assert manifest["files"]["figures/test.svg"] == digest(dest / "figures/test.svg")


def test_null_component_label_preserves_unknown_points_and_forecast():
    records = []
    for target, actual in [
        ("rushing_yards", 100.0),
        ("rushing_tds", 1.0),
        ("receptions", None),
        ("receiving_yards", 50.0),
        ("receiving_tds", 0.0),
    ]:
        records.append(
            {
                "horizon": "season",
                "year": 2021,
                "origin": 0,
                "player_id": "p",
                "recipe": "selected",
                "weeks": 17,
                "training_max_year": 2020,
                "target": target,
                "actual": actual,
                "prediction": 2.0,
            }
        )
    row = analysis.stat_points(pl.DataFrame(records)).row(0, named=True)
    assert row["actual"] is None
    assert row["prediction"] == pytest.approx(26.4)
    assert analysis.stat_points(pl.DataFrame(records[:-1])).is_empty()


def test_point_policies_survive_when_residual_calibration_has_too_few_rows():
    rows = _rows().filter(pl.col("player_id").is_in(["0", "1", "2"]))
    annual = analysis.scores(rows, ["horizon", "target", "recipe", "year"])
    config = {
        "horizons": ["season"],
        "targets": ["carries"],
        "evaluation_years": [2021],
        "selection_years": 3,
        "recipes": [_recipe("a"), _recipe("b")],
    }
    policies, selections, distributions = analysis.policy_predictions(rows, annual, config)
    assert policies.height and selections.height
    assert distributions.is_empty()
    assert {"crps", "calibration_max_year"} <= set(distributions.columns)


def test_missing_ensemble_member_prediction_cannot_be_summed_as_zero():
    rows = _rows().filter((pl.col("year") == 2021) & (pl.col("player_id") == "0"))
    rows = rows.with_columns(
        pl.when(pl.col("recipe") == "a")
        .then(None)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )
    result = analysis.blend(rows, ["a", "b"], "ensemble", 2021)
    assert result["prediction"].item() is None


def _policy_artifacts(tmp_path, bundles, policies):
    for (target, recipe), bundle in bundles.items():
        path = tmp_path / "season" / target / recipe
        path.mkdir(parents=True)
        joblib.dump(bundle, path / "pipeline.joblib")
    json_write(tmp_path / "policies.json", policies)
    json_write(
        tmp_path / "manifest.json",
        {
            "status": "complete",
            "files": {
                str(path.relative_to(tmp_path)): digest(path)
                for path in tmp_path.rglob("*")
                if path.is_file()
            },
        },
    )


def test_ensemble_replay_aligns_member_features_and_detects_corruption(tmp_path):
    model = {
        "model": FirstColumnModel(),
        "representation": IdentityRepresentation(),
        "feature_order": ["b", "a"],
        "target": "points",
    }
    constant = {"constant_rate": -2.0, "target": "points", "feature_order": []}
    policies = [
        {
            "horizon": "season",
            "target": "points",
            "policy": "ensemble_convex",
            "members": ["model", "constant"],
            "weights": [0.25, 0.75],
        }
    ]
    _policy_artifacts(
        tmp_path, {("points", "model"): model, ("points", "constant"): constant}, policies
    )
    loaded = final.load_policy(tmp_path, "season", "points")
    prediction = final.predict_policy(loaded, [[100, -4], [200, 8]], ["a", "b"], [2, 3])
    np.testing.assert_array_equal(prediction, [-5, 1.5])
    with pytest.raises(ValueError, match="Missing forecast"):
        final.predict_policy(loaded, [[100]], ["a"], [2])
    (tmp_path / "season/points/constant/pipeline.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt fitted artifact"):
        final.load_policy(tmp_path, "season", "points")


def test_component_points_replay_preserves_signed_yards_and_clips_counts(tmp_path):
    rates = {
        "rushing_yards": -10,
        "rushing_tds": -2,
        "receptions": 3,
        "receiving_yards": 5,
        "receiving_tds": 1,
    }
    bundles = {
        (stat, "constant"): {"target": stat, "constant_rate": rate, "feature_order": []}
        for stat, rate in rates.items()
    }
    policies = [
        {
            "horizon": "season",
            "target": stat,
            "policy": "ensemble_convex",
            "members": ["constant"],
            "weights": [1.0],
        }
        for stat in rates
    ]
    _policy_artifacts(tmp_path, bundles, policies)
    loaded = final.load_policy(tmp_path, "season", "points_from_stats")
    np.testing.assert_array_equal(
        final.predict_policy(loaded, np.empty((2, 0)), [], [1, 2]), [8.5, 17]
    )


def test_interrupted_refit_resumes_only_verified_members_and_rejects_stale_selection(
    tmp_path, monkeypatch
):
    _setup_run(tmp_path)
    analysis.analyze(tmp_path)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(32, 3)).astype(np.float32)
    y = np.repeat((x[:, 0] + 5)[:, None], 7, axis=1)
    dataset = SimpleNamespace(
        x=x,
        names=["a", "b", "c"],
        targets=y,
        baseline=y,
        exposure=np.ones(32),
        panel=pl.DataFrame({"year": np.repeat(np.arange(2014, 2022), 4)}),
        provenance={"fixture": "v1"},
    )
    monkeypatch.setattr(final, "load_dataset", lambda *args: dataset)
    original = final.build_estimator
    calls = []

    def interrupt_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated interrupt")
        return original(*args, **kwargs)

    monkeypatch.setattr(final, "build_estimator", interrupt_second)
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        final.refit(tmp_path)
    checkpoint = tmp_path / "final_models/season/carries/a/pipeline.joblib"
    checksum, modified = digest(checkpoint), checkpoint.stat().st_mtime_ns
    (tmp_path / "final_models/manifest.json.tmp").write_text("interrupted publication")
    monkeypatch.setattr(final, "build_estimator", original)
    directory = final.refit(tmp_path)
    assert digest(checkpoint) == checksum and checkpoint.stat().st_mtime_ns == modified
    assert final.verify_refit(tmp_path)["status"] == "complete"
    assert "manifest.json.tmp" not in final.verify_refit(tmp_path)["files"]
    assert final.refit(tmp_path) == directory
    policy = final.load_policy(directory, "season", "carries")
    assert np.isfinite(final.predict_policy(policy, x, dataset.names, dataset.exposure)).all()
    saved = tmp_path / "analysis/predictions.parquet"
    table = pl.read_parquet(saved).with_columns((pl.col("prediction") + 1).alias("prediction"))
    table.write_parquet(saved)
    with pytest.raises(ValueError, match="selection"):
        final.refit(tmp_path)


def test_figures_support_single_horizon_and_no_representation_comparisons(tmp_path):
    config = _setup_run(tmp_path)
    analysis.analyze(tmp_path)
    # Identical raw recipes may have comparisons, but no controlled projections.
    config["recipes"] = config["recipes"][:1]
    json_write(tmp_path / "config.json", config)
    controlled_comparisons(tmp_path, config)
    figures(tmp_path)
    for name in ["predictive_skill", "representations", "calibration"]:
        assert (tmp_path / "analysis/figures" / f"{name}.svg").is_file()


def test_permutation_verifies_saved_pipeline_forecasts_before_diagnostics(tmp_path, monkeypatch):
    config = _setup_run(tmp_path)
    analysis.analyze(tmp_path)
    saved = _rows().filter((pl.col("year") == 2021) & (pl.col("recipe") == "a"))
    actual = saved["actual"].to_numpy()
    exposure = np.full(len(actual), 17.0)
    dataset = SimpleNamespace(
        x=((actual + 1) / exposure)[:, None],
        names=["a"],
        exposure=exposure,
        targets=np.repeat(actual[:, None], 7, axis=1),
        baseline=np.repeat(actual[:, None], 7, axis=1),
        panel=saved.select("horizon", "year", "origin", "player_id"),
    )
    monkeypatch.setattr(diagnostics, "load_dataset", lambda *args: dataset)
    checkpoint = tmp_path / "season/fits/a/2021"
    json_write(checkpoint / "feature_order.json", ["a"])
    joblib.dump(IdentityRepresentation(), checkpoint / "preprocessing.joblib")
    joblib.dump(FirstColumnModel(), checkpoint / "carries.joblib")

    def seal_checkpoint():
        json_write(
            checkpoint / "complete.json",
            {
                "files": {
                    path.name: digest(path)
                    for path in checkpoint.iterdir()
                    if path.name != "complete.json"
                },
            },
        )

    seal_checkpoint()
    diagnostics.permutation_diagnostics(tmp_path, config, repeats=1)
    importance = pl.read_parquet(tmp_path / "analysis/native_importance.parquet")
    assert importance.is_empty() and "feature" in importance.columns
    result = pl.read_parquet(tmp_path / "analysis/source_permutation.parquet")
    assert result.height == 1 and result["delta_mse"].item() > 0
    saved.with_columns((pl.col("prediction") + 1).alias("prediction")).write_parquet(
        checkpoint / "predictions.parquet"
    )
    seal_checkpoint()
    with pytest.raises(AssertionError):
        diagnostics.permutation_diagnostics(tmp_path, config, repeats=1)
