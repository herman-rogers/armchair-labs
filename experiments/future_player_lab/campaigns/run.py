"""Fit and evaluate immutable Opening Drive campaign cells with chronological selection."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from ..gpu_lab.calibrated_baselines import AffinePersistence
from ..gpu_lab.run import digest, json_write, packages
from .data import ROOT, SCORING, STATS, dependency_paths, load_dataset, matrix, points

CAMPAIGN_ID = "opening_drive_v1"
YEARS = list(range(2018, 2026))
EVALUATION_YEARS = list(range(2021, 2026))
RECIPES = {
    "xgb_raw_draft": {"engine": "xgboost", "depth": 3, "child": 20, "lambda": 15, "view": "all"},
    "xgb_shallow_draft": {
        "engine": "xgboost",
        "depth": 2,
        "child": 40,
        "lambda": 30,
        "view": "all",
    },
    "xgb_no_provider_draft": {
        "engine": "xgboost",
        "depth": 3,
        "child": 20,
        "lambda": 15,
        "view": "no_provider",
    },
    "lightgbm_raw_draft": {"engine": "lightgbm", "view": "all"},
}
COUNT_TARGETS = {
    "carries",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_tds",
    "attempts",
    "completions",
    "passing_tds",
    "passing_interceptions",
}


def columns(names, recipe):
    spec = RECIPES[recipe]
    # Observed box/NGS/play facts remain; remove provider-derived efficiency composites.
    excluded = (
        "wopr",
        "racr",
        "pacr",
        "dakota",
        "fantasy",
        "epa",
        "wpa",
        "expect",
        "xyac",
        "_wp",
        "_ep",
    )
    return [
        i
        for i, name in enumerate(names)
        if spec["view"] == "all"
        or (
            name.split("__")[-1] not in {"cp", "cpoe", "xpass"}
            and not any(k in name.lower() for k in excluded)
        )
    ]


def estimator(recipe, device="cuda"):
    spec = RECIPES[recipe]
    common = dict(
        n_estimators=160,
        learning_rate=0.035,
        random_state=20261009,
        n_jobs=4,
        colsample_bytree=0.8,
        reg_lambda=15,
    )
    if spec["engine"] == "xgboost":
        common.update(reg_lambda=spec["lambda"])
        return XGBRegressor(
            **common,
            tree_method="hist",
            device=device,
            max_depth=spec["depth"],
            min_child_weight=spec["child"],
            subsample=0.85,
            max_bin=128,
            objective="reg:squarederror",
        )
    return LGBMRegressor(**common, num_leaves=15, min_child_samples=25, verbosity=-1)


def clamp(prediction, target):
    return np.maximum(prediction, 0) if target in COUNT_TARGETS else prediction


def select_members(annual, target, forecast_year, count=2):
    """Equal-year MSE rank uses only the three seasons strictly before prediction."""
    window = list(range(forecast_year - 3, forecast_year))
    ranks = []
    for recipe in RECIPES:
        rows = [
            r
            for r in annual
            if r["target"] == target and r["recipe"] == recipe and r["year"] in window
        ]
        if (
            len(rows) != 3
            or {r["year"] for r in rows} != set(window)
            or any(r["mse"] is None or not np.isfinite(r["mse"]) or r["mse"] < 0 for r in rows)
        ):
            raise ValueError(
                f"Incomplete chronological selection history for {target}/{recipe}/{forecast_year}"
            )
        ranks.append((float(np.mean([r["mse"] for r in rows])), recipe))
    return [recipe for _, recipe in sorted(ranks)[:count]]


def metrics(actual, prediction):
    keep = np.isfinite(actual) & np.isfinite(prediction)
    if not keep.any():
        return {"n": 0, "mse": None, "rmse": None, "mae": None}
    residual = actual[keep] - prediction[keep]
    mse = float(np.mean(residual**2))
    return {
        "n": int(keep.sum()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
    }


def baseline_values(panel, position, target):
    rates = {
        s: panel[f"history_{s}"].to_numpy() / panel["history_weeks"].to_numpy()
        for s in STATS[position]
    }
    value = points(position, rates) if target == "points" else rates[target]
    return value * panel["weeks"].to_numpy()


def fit_cell(
    position,
    horizon,
    output,
    cache,
    device="cuda",
    *,
    dataset_loader=None,
    extra_sources=(),
    availability_contract=None,
):
    started = time.monotonic()
    directory = Path(output) / f"{position.lower()}_{horizon}"
    if (directory / "manifest.json").exists():
        saved = json.loads((directory / "manifest.json").read_text())
        if availability_contract is not None:
            if (
                saved.get("availability") != availability_contract
                or saved["campaign_id"] != CAMPAIGN_ID
            ):
                raise ValueError("Sealed availability campaign contract changed")
            for name, sha in saved["source_files"].items():
                if digest(ROOT / name) != sha:
                    raise ValueError(f"Sealed training source changed: {name}")
            (dataset_loader or load_dataset)(position, horizon, cache)
        for name, sha in saved["files"].items():
            if digest(directory / name) != sha:
                raise ValueError(f"Sealed model artifact changed: {name}")
        print(f"verified existing cell {directory}", flush=True)
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    panel, names = (dataset_loader or load_dataset)(position, horizon, cache)
    sources = list(dict.fromkeys([Path(__file__), *dependency_paths(), *extra_sources]))
    contract = {
        "campaign_id": CAMPAIGN_ID,
        "recipes": RECIPES,
        "years": YEARS,
        "evaluation_years": EVALUATION_YEARS,
        "device": device,
        "dataset_sha256": digest(Path(cache) / f"{position.lower()}_{horizon}" / "dataset.json"),
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    if availability_contract is not None:
        contract["availability"] = availability_contract
    contract_path = directory / "fit_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Partial training contract changed: start a fresh run")
    json_write(contract_path, contract)
    x = matrix(panel, names)
    years = panel["year"].to_numpy()
    exposure = panel["weeks"].to_numpy()
    targets = [*STATS[position], "points"]
    identity = panel.select(
        "player_id",
        "player_display_name",
        "year",
        "origin",
        "team",
        "history_games",
        "history_weeks",
    )
    rows, annual, choices = [], [], []
    saved_models = {}
    for target in targets:
        y = panel[f"y_{target}"].to_numpy().astype(float)
        baseline = baseline_values(panel, position, target)
        folds = {}
        for year in YEARS:
            train = (years < year) & np.isfinite(y)
            test = years == year
            if not test.any() or not train.any():
                raise ValueError("Missing chronological training or evaluation fold")
            affine = AffinePersistence().fit(baseline[train], y[train])
            forecasts = {
                "persistence": clamp(baseline[test], target),
                "affine": affine.predict(baseline[test], count_stat=target in COUNT_TARGETS),
            }
            for recipe in RECIPES:
                checkpoint = directory / "checkpoints" / f"{target}_{year}_{recipe}.npy"
                checkpoint.parent.mkdir(exist_ok=True)
                if checkpoint.exists():
                    receipt = json.loads(checkpoint.with_suffix(".json").read_text())
                    if digest(checkpoint) != receipt["sha256"]:
                        raise ValueError("Checkpoint checksum mismatch")
                    pred = np.load(checkpoint, allow_pickle=False)
                    if pred.shape != (int(test.sum()),) or not np.isfinite(pred).all():
                        raise ValueError("Invalid checkpoint predictions")
                else:
                    indices = columns(names, recipe)
                    model = estimator(recipe, device)
                    model.fit(x[train][:, indices], y[train] / exposure[train])
                    pred = clamp(model.predict(x[test][:, indices]) * exposure[test], target)
                    np.save(checkpoint, pred)
                    json_write(checkpoint.with_suffix(".json"), {"sha256": digest(checkpoint)})
                forecasts[recipe] = pred
                annual.append(
                    {"target": target, "recipe": recipe, "year": year, **metrics(y[test], pred)}
                )
            folds[year] = forecasts
            if year in EVALUATION_YEARS:
                members = select_members(annual, target, year)
                forecasts["ensemble"] = np.mean([forecasts[m] for m in members], axis=0)
                forecasts["selected"] = forecasts[members[0]]
                choices.append(
                    {
                        "target": target,
                        "year": year,
                        "members": members,
                        "selection_years": list(range(year - 3, year)),
                        "training_max_year": year - 1,
                    }
                )
                ids = identity.filter(pl.col("year") == year)
                for recipe, pred in forecasts.items():
                    rows.append(
                        ids.with_columns(
                            pl.lit(target).alias("target"),
                            pl.lit(recipe).alias("recipe"),
                            pl.Series("actual", y[test]),
                            pl.Series("prediction", pred),
                        )
                    )
            print(f"fit {position}/{horizon}/{target}/{year}", flush=True)
        members = select_members(annual, target, 2026)
        final = []
        for recipe in members:
            indices = columns(names, recipe)
            known = np.isfinite(y)
            model = estimator(recipe, device)
            model.fit(x[known][:, indices], y[known] / exposure[known])
            final.append({"recipe": recipe, "indices": indices, "model": model})
        saved_models[target] = final
    oof = pl.concat(rows)
    oof.write_parquet(directory / "evaluation_predictions.parquet")
    report = []
    for (target, recipe), block in oof.partition_by(["target", "recipe"], as_dict=True).items():
        for cohort, subgroup in [
            ("all_candidates", block),
            (
                "established_appearances",
                block.filter(pl.col("history_games") / pl.col("history_weeks") >= 0.5),
            ),
        ]:
            for year, group in [
                (None, subgroup),
                *[(int(k[0]), v) for k, v in subgroup.partition_by("year", as_dict=True).items()],
            ]:
                report.append(
                    {
                        "target": target,
                        "recipe": recipe,
                        "cohort": cohort,
                        "year": year,
                        **metrics(group["actual"].to_numpy(), group["prediction"].to_numpy()),
                    }
                )
    report_doc = {
        "schema_version": 1,
        "position": position,
        "horizon": horizon,
        "development_evaluation_years": EVALUATION_YEARS,
        "selection": "Per target, uniform top two trees from previous three years equal-year MSE",
        "cohort": "established_appearances: observed games / elapsed calendar weeks >= 0.5, known at origin",
        "rows": report,
        "choices": choices,
        "candidate_annual": annual,
        "limitations": [
            "Historical development seasons reused from earlier research; no pristine holdout claim.",
            "Historical provider vintages are retrospective, not archived issue-time inputs.",
            "Candidate universe includes inactive backups/zero outcomes; unlinked newcomers can be absent.",
            "QB features include passing box and NGS; play-by-play is inherited rush/receiver summaries, not passer-specific.",
            "Independent stat and direct fantasy outputs need not be arithmetically coherent.",
        ],
    }
    if availability_contract is not None:
        report_doc["availability"] = availability_contract
        report_doc["limitations"].append(
            "Dated availability archive reconstructed retrospectively; missing evidence stays unknown. "
            "Known retired players excluded equally from fitting, selection, and baseline evaluation."
        )
    json_write(directory / "metrics.json", report_doc)
    artifact = {
        "schema_version": 1,
        "position": position,
        "horizon": horizon,
        "origin_week": 0 if horizon == "season" else 2,
        "training_through_season": 2025,
        "feature_names": names,
        "targets": targets,
        "models": saved_models,
        "scoring": SCORING[position],
    }
    if availability_contract is not None:
        artifact["availability"] = availability_contract
    joblib.dump(artifact, directory / "models.joblib", compress=3)
    roundtrip = joblib.load(directory / "models.joblib")
    for target, members in saved_models.items():
        for before, after in zip(members, roundtrip["models"][target], strict=True):
            sample = x[-12:, before["indices"]]
            np.testing.assert_array_equal(
                before["model"].predict(sample), after["model"].predict(sample)
            )
    json_write(directory / "feature_names.json", names)
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "model_id": f"{position.lower()}_{horizon}",
        "position": position,
        "horizon": horizon,
        "origin_week": 0 if horizon == "season" else 2,
        "training_through_season": 2025,
        "training_first_season": int(years.min()),
        "targets": targets,
        "scoring": SCORING[position],
        "feature_count": len(names),
        "training_rows": len(years),
        "runtime_seconds": time.monotonic() - started,
        "recipes": RECIPES,
        "packages": packages(),
        "disposition": "research_preview",
        "final_members": {t: [m["recipe"] for m in ms] for t, ms in saved_models.items()},
        "source_files": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "dataset_manifest_sha256": digest(
            Path(cache) / f"{position.lower()}_{horizon}" / "dataset.json"
        ),
        "files": {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
    }
    if availability_contract is not None:
        manifest["availability"] = availability_contract
    json_write(directory / "manifest.json", manifest)
    print(f"complete {position}/{horizon}: {manifest['runtime_seconds']:.1f}s", flush=True)
    return manifest


def load_model(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if "models.joblib" not in manifest["files"]:
        raise ValueError("Model weights missing from sealed manifest")
    for name, sha in manifest["files"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"Saved model checksum mismatch: {name}")
    return joblib.load(directory / "models.joblib")


def predict(model, panel):
    if set(panel["horizon"].to_list()) != {model["horizon"]}:
        raise ValueError("Horizon does not match trained model contract")
    if set(panel["position"].to_list()) != {model["position"]}:
        raise ValueError("Position does not match trained model contract")
    if panel["year"].min() <= model["training_through_season"]:
        raise ValueError("Prediction season overlaps final training data")
    if set(panel["origin"].to_list()) != {model["origin_week"]}:
        raise ValueError("Origin does not match trained model contract")
    x = matrix(panel, model["feature_names"])
    exposure = panel["weeks"].to_numpy()
    outputs = {}
    for target, members in model["models"].items():
        outputs[target] = np.mean(
            [clamp(m["model"].predict(x[:, m["indices"]]) * exposure, target) for m in members],
            axis=0,
        )
    outputs["component_points"] = points(model["position"], outputs)
    return outputs


def main():
    global CAMPAIGN_ID, RECIPES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="opening_drive_v1_001")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/opening_drive_v1.json"
    )
    parser.add_argument("--positions", nargs="+", default=["QB", "WR", "RB"], choices=list(STATS))
    parser.add_argument(
        "--horizons", nargs="+", default=["remaining", "season"], choices=["season", "remaining"]
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not re.fullmatch(r"[a-z][a-z0-9_]+", args.run_id) or not re.fullmatch(
        r"[a-z][a-z0-9_]+", config["campaign_id"]
    ):
        raise ValueError("Unsafe run or campaign ID")
    CAMPAIGN_ID, RECIPES = config["campaign_id"], config["recipes"]
    if config["validation_years"] != YEARS or config["evaluation_years"] != EVALUATION_YEARS:
        raise ValueError("This version supports 2018–2025 validation and 2021–2025 evaluation")
    if config["targets"] != {p: [*STATS[p], "points"] for p in config["positions"]}:
        raise ValueError("Unsupported target contract")
    output = ROOT / "experiments/future_player_lab/runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "campaign_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run campaign configuration changed")
    json_write(config_path, config)
    with threadpool_limits(limits=4):
        for position in args.positions:
            for horizon in args.horizons:
                fit_cell(position, horizon, output, output / "inputs", args.device)


if __name__ == "__main__":
    main()
