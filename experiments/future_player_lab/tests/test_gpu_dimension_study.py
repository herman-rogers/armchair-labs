"""The dimension supplement must compare exactly matched, frozen forecast rows."""

import copy

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.gpu_lab import dimension_study
from experiments.future_player_lab.gpu_lab.dimension_study import (
    KINDS,
    aligned_forecasts,
    checkpoint_rows,
    main_ready,
    score_comparisons,
    study_config,
)
from experiments.future_player_lab.gpu_lab.run import STATS, digest, json_write


def configuration():
    raw = {
        "id": "r000_xgboost_raw_all",
        "family": "xgboost",
        "device": "cuda",
        "representation": {"kind": "raw", "n_components": 48, "missing_indicators": True},
        "params": {"n_estimators": 160, "max_depth": 3},
        "view": "all",
        "history": 0,
        "seed": 20261003,
    }
    recipes = [raw]
    for kind in KINDS:
        recipe = copy.deepcopy(raw)
        recipe["id"] = f"anchor_{kind}"
        recipe["representation"]["kind"] = kind
        recipes.append(recipe)
    return {
        "recipes": recipes,
        "years": list(range(2018, 2026)),
        "evaluation_years": list(range(2021, 2026)),
        "horizons": ["season", "remaining"],
        "targets": STATS,
    }


def forecast():
    return pl.DataFrame(
        {
            "horizon": ["season"] * 3,
            "year": [2021] * 3,
            "origin": [0] * 3,
            "player_id": ["a", "b", "c"],
            "target": ["points"] * 3,
            "weeks": [17] * 3,
            "actual": [2.0, None, -1.0],
            "prediction": [1.0, 2.0, -1.0],
            "training_max_year": [2020] * 3,
        }
    )


def test_eight_recipes_change_only_component_count_and_id():
    main = configuration()
    original = copy.deepcopy(main)
    config, controls = study_config(main)
    assert main == original
    assert len(config["recipes"]) == 8
    for recipe in config["recipes"]:
        anchor = copy.deepcopy(controls[recipe["representation"]["kind"]])
        anchor["id"] = recipe["id"]
        assert recipe["representation"]["n_components"] in {16, 128}
        anchor["representation"]["n_components"] = recipe["representation"]["n_components"]
        assert recipe == anchor
    main["recipes"][1]["params"]["max_depth"] = 4
    with pytest.raises(ValueError, match="matched 48-component"):
        study_config(main)


def test_alignment_sorts_keys_and_preserves_unknown_and_signed_points():
    frame = forecast()
    candidate, reference = aligned_forecasts(frame.reverse(), frame)
    assert candidate.equals(reference)
    assert np.isnan(candidate["actual"].to_numpy()[1])
    assert candidate["prediction"][2] == -1


@pytest.mark.parametrize("kind", ["missing", "duplicate", "actual", "weeks", "chronology", "nan"])
def test_alignment_rejects_nonmatching_or_invalid_forecasts(kind):
    control = forecast()
    candidate = control
    if kind == "missing":
        candidate = candidate.head(2)
    elif kind == "duplicate":
        candidate = pl.concat([candidate, candidate.head(1)])
    elif kind == "actual":
        candidate = candidate.with_columns(pl.col("actual").fill_null(0))
    elif kind == "weeks":
        candidate = candidate.with_columns(pl.lit(16).alias("weeks"))
    elif kind == "chronology":
        candidate = candidate.with_columns(pl.lit(2021).alias("training_max_year"))
    else:
        candidate = candidate.with_columns(pl.lit(float("nan")).alias("prediction"))
    with pytest.raises(ValueError):
        aligned_forecasts(candidate, control)


def test_controlled_metrics_use_same_known_rows_and_reference48():
    config, controls = study_config(configuration())
    config.update(horizons=["season"], targets=["points"], years=[2021], evaluation_years=[2021])
    rows = []
    recipes = list(controls.values()) + config["recipes"]
    for recipe in recipes:
        kind = recipe["representation"]["kind"]
        width = None if kind == "raw" else recipe["representation"]["n_components"]
        # Raw and 48 forecasts are exact. New widths have error 1 on both known rows.
        prediction = [2.0, 99.0, -1.0] if width in {None, 48} else [3.0, -99.0, 0.0]
        rows.append(
            forecast().with_columns(
                pl.Series("prediction", prediction),
                pl.lit(recipe["id"]).alias("recipe"),
                pl.lit(kind).alias("representation"),
                pl.lit(width, dtype=pl.Int64).alias("n_components"),
                pl.lit(width or 200).alias("effective_components"),
            )
        )
    annual, comparisons, annual_comparisons = score_comparisons(pl.concat(rows), config, controls)
    assert annual.height == 13
    assert comparisons.height == 24
    assert annual_comparisons.height == 12
    assert comparisons["n"].unique().to_list() == [2]
    assert comparisons["unknown_labels"].unique().to_list() == [1]
    assert comparisons["years"].unique().to_list() == ["2021"]
    new = comparisons.filter(pl.col("n_components") != 48)
    assert new["mse"].unique().to_list() == [1]
    assert new["raw_mse"].unique().to_list() == [0]
    assert new["reference48_mse"].unique().to_list() == [0]
    assert new["mse_improvement_pct"].null_count() == new.height


def test_gate_requires_main_postprocess_and_refit_complete(tmp_path):
    json_write(tmp_path / "manifest.json", {"status": "complete"})
    assert not main_ready(tmp_path)
    for directory in ["analysis", "final_models"]:
        (tmp_path / directory).mkdir()
        json_write(tmp_path / directory / "manifest.json", {"status": "complete"})
    json_write(tmp_path / "postprocess.json", {"status": "running"})
    assert not main_ready(tmp_path)
    json_write(tmp_path / "postprocess.json", {"status": "complete"})
    assert main_ready(tmp_path)
    json_write(tmp_path / "postprocess.json", {"status": "failed"})
    with pytest.raises(ValueError, match="needs repair"):
        main_ready(tmp_path)


def test_checkpoint_provenance_requires_gpu_verification_and_saved_pipelines(tmp_path, monkeypatch):
    monkeypatch.setattr(dimension_study, "ROOT", tmp_path)
    run = tmp_path / "main"
    recipe = configuration()["recipes"][1]
    checkpoint = run / "season/fits" / recipe["id"] / "2021"
    checkpoint.mkdir(parents=True)
    forecast().with_columns(pl.lit(recipe["id"]).alias("recipe")).write_parquet(
        checkpoint / "predictions.parquet"
    )
    metadata = {
        "recipe": recipe,
        "year": 2021,
        "training_max_year": 2020,
        "representation": {"output_features": 48},
        "fits": [
            {
                "stat": "points",
                "backend": {
                    "device_verified": True,
                    "effective_device": "cuda:0",
                    "requested_params": recipe["params"],
                    "seed": recipe["seed"],
                },
            }
        ],
    }
    json_write(checkpoint / "metadata.json", metadata)
    for name in ["points.joblib", "preprocessing.joblib", "feature_order.json"]:
        (checkpoint / name).write_text("opaque test artifact")

    def seal():
        json_write(
            checkpoint / "complete.json",
            {
                "files": {
                    p.name: digest(p) for p in checkpoint.iterdir() if p.name != "complete.json"
                }
            },
        )

    seal()
    config = {"horizons": ["season"], "targets": ["points"], "years": [2021]}
    rows, receipts = checkpoint_rows(run, [recipe], config, True)
    assert rows.height == 3
    assert len(receipts) == 1
    assert rows["forecast_source"].unique().to_list() == ["reused_control"]
    metadata["fits"][0]["backend"]["device_verified"] = False
    json_write(checkpoint / "metadata.json", metadata)
    seal()
    with pytest.raises(ValueError, match="Unverified CUDA"):
        checkpoint_rows(run, [recipe], config, True)
    (checkpoint / "predictions.parquet").write_text("corrupt")
    with pytest.raises(ValueError, match="Corrupt checkpoint"):
        checkpoint_rows(run, [recipe], config, True)
