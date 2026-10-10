"""The artifact audit detects semantically inconsistent but rehashed research files."""

import hashlib
import json

import polars as pl
import pytest
from experiments.future_player_lab.campaigns.audit import audit_cell


@pytest.fixture
def sealed_cell(tmp_path):
    predictions = {
        "a": [2.0, 8.0],
        "b": [1.0, 9.0],
        "ensemble": [1.5, 8.5],
        "selected": [1.0, 9.0],
        "affine": [3.0, 7.0],
        "persistence": [4.0, 6.0],
    }
    rows = pl.concat(
        [
            pl.DataFrame(
                {
                    "target": ["points"] * 2,
                    "recipe": [recipe] * 2,
                    "year": [2021] * 2,
                    "origin": [2] * 2,
                    "player_id": ["one", "two"],
                    "actual": [0.0, 10.0],
                    "prediction": values,
                    "history_games": [1, 1],
                    "history_weeks": [2, 2],
                }
            )
            for recipe, values in predictions.items()
        ]
    )
    rows.write_parquet(tmp_path / "evaluation_predictions.parquet")
    loss = {name: values[0] ** 2 for name, values in predictions.items()}
    records = [
        {
            "target": "points",
            "recipe": recipe,
            "cohort": "all_candidates",
            "year": None,
            "n": 2,
            "mse": mse,
            "rmse": mse**0.5,
            "mae": mse**0.5,
        }
        for recipe, mse in loss.items()
    ]
    annual = [
        {
            "target": "points",
            "recipe": recipe,
            "year": year,
            "n": 2,
            "mse": loss[recipe],
            "rmse": loss[recipe] ** 0.5,
            "mae": loss[recipe] ** 0.5,
        }
        for recipe in ("a", "b")
        for year in range(2018, 2022)
    ]
    metrics = {
        "development_evaluation_years": [2021],
        "rows": records,
        "candidate_annual": annual,
        "choices": [
            {
                "target": "points",
                "year": 2021,
                "members": ["b", "a"],
                "selection_years": [2018, 2019, 2020],
                "training_max_year": 2020,
            }
        ],
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "models.joblib").write_bytes(b"Audit never deserializes executable weights")
    manifest = {
        "model_id": "wr_remaining",
        "targets": ["points"],
        "recipes": {"a": {}, "b": {}},
        "training_through_season": 2020,
        "final_members": {"points": ["b", "a"]},
        "source_files": {"training.py": "a" * 64},
        "dataset_manifest_sha256": "b" * 64,
    }
    contract = {
        "sources": manifest["source_files"],
        "recipes": manifest["recipes"],
        "dataset_sha256": manifest["dataset_manifest_sha256"],
    }
    (tmp_path / "fit_contract.json").write_text(json.dumps(contract))

    def reseal():
        manifest["files"] = {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in (
                "metrics.json",
                "models.joblib",
                "evaluation_predictions.parquet",
                "fit_contract.json",
            )
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    reseal()
    return tmp_path, reseal


def test_audit_recomputes_metrics_and_selections_without_loading_weights(sealed_cell):
    directory, _ = sealed_cell
    result = audit_cell(directory)
    assert result["status"] == "verified"
    assert result["max_ensemble_difference"] == 0
    assert result["recomputed_metric_groups"] == 8
    assert result["matched_affine_rows"] == 2
    assert result["fit_source_contract_verified"]


@pytest.mark.parametrize("fault", ["ensemble", "outcome", "population", "duplicate"])
def test_rehashed_inconsistent_forecasts_are_rejected(sealed_cell, fault):
    directory, reseal = sealed_cell
    path = directory / "evaluation_predictions.parquet"
    rows = pl.read_parquet(path)
    if fault in {"ensemble", "outcome"}:
        column = "prediction" if fault == "ensemble" else "actual"
        rows = rows.with_columns(
            pl.when(pl.col("recipe") == "ensemble")
            .then(100.0)
            .otherwise(pl.col(column))
            .alias(column)
        )
    elif fault == "population":
        rows = rows.filter(~((pl.col("recipe") == "a") & (pl.col("player_id") == "one")))
    else:
        rows = pl.concat([rows, rows.head(1)])
    rows.write_parquet(path)
    reseal()
    with pytest.raises(ValueError):
        audit_cell(directory)


@pytest.mark.parametrize("fault", ["metric", "selection"])
def test_rehashed_inconsistent_evaluation_metadata_is_rejected(sealed_cell, fault):
    directory, reseal = sealed_cell
    path = directory / "metrics.json"
    report = json.loads(path.read_text())
    if fault == "metric":
        report["rows"][0]["mse"] = 123456.0
    else:
        report["choices"][0]["members"] = ["a", "b"]
    path.write_text(json.dumps(report))
    reseal()
    with pytest.raises(ValueError):
        audit_cell(directory)


def test_mutated_weight_bytes_fail_the_audit(sealed_cell):
    directory, _ = sealed_cell
    (directory / "models.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        audit_cell(directory)


@pytest.mark.parametrize(
    "key,value", [("sources", {}), ("recipes", {}), ("dataset_sha256", "c" * 64)]
)
def test_rehashed_fit_contract_cannot_disagree_with_sealed_manifest(sealed_cell, key, value):
    directory, reseal = sealed_cell
    path = directory / "fit_contract.json"
    contract = json.loads(path.read_text())
    contract[key] = value
    path.write_text(json.dumps(contract))
    reseal()
    with pytest.raises(ValueError, match="Fit-time"):
        audit_cell(directory)
