"""Persist next-use models trained through all completed seasons, with no validation claim."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from experiments.future_player_lab.gpu_backends import build_estimator
from threadpoolctl import threadpool_limits

from .analyze import choose_members, convex_weights, diverse_members
from .data import load_dataset
from .factorized import DENOMINATORS, FactorizedTree
from .recipes import view_mask
from .representations import fit_representation
from .run import COUNT_STATS, ROOT, RUNS, STATS, digest, json_write, packages, stat_arrays


def predict_pipeline(bundle, x, exposure):
    """Reproduce support and exposure postprocessing from the evaluated research pipeline."""
    if "constant_rate" in bundle:
        rates = np.full(len(exposure), bundle["constant_rate"])
    else:
        if np.asarray(x).shape[1] != len(bundle["feature_order"]):
            raise ValueError("Inputs must use the saved feature order")
        transformed = bundle["representation"].transform(x)
        rates = np.asarray(bundle["model"].predict(transformed)).reshape(-1)
    if bundle["target"] in COUNT_STATS:
        rates = np.maximum(0, rates)
    return rates * np.asarray(exposure)


def _signature(out):
    """All inputs that can change member selection, weights, or fitted pipelines."""
    sources = [Path(__file__), Path(choose_members.__code__.co_filename)]
    config = json.loads((out / "config.json").read_text())
    original = out / "signature.json"
    if original.exists():
        training = json.loads(original.read_text())
        for name, expected in training["sources"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(
                    f"Frozen training implementation changed before final refit: {name}"
                )
        if training["packages"] != packages():
            raise ValueError("Training package versions changed before final refit")
    inputs = [
        out / "config.json",
        out / "analysis/annual_metrics.parquet",
        out / "analysis/predictions.parquet",
    ]
    for path in [out / "signature.json", *[out / h / "inputs.json" for h in config["horizons"]]]:
        if path.exists():
            inputs.append(path)
    return {
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "inputs": {str(p.relative_to(out)): digest(p) for p in inputs},
        "packages": packages(),
    }


def _verify_files(directory, manifest):
    if not manifest.get("files"):
        raise ValueError(f"Artifact manifest has no files: {directory}")
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Corrupt fitted artifact: {path}")


def verify_refit(out):
    out = Path(out)
    directory = out / "final_models"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("signature") != _signature(out):
        raise ValueError("Final model inputs, selection, source, or packages changed")
    _verify_files(directory, manifest)
    return manifest


def load_policy(directory, horizon, target, policy="ensemble_convex"):
    """Load a checksum-verified local ensemble, including component-stat PPR means.

    Joblib files must come from a trusted producer. Integrity checks do not make
    untrusted pickle data safe. This loader never fits a model or reads outcomes.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Final models are incomplete")
    _verify_files(directory, manifest)
    policies = json.loads((directory / "policies.json").read_text())
    if target == "points_from_stats":
        components = {
            "rushing_yards": 0.1,
            "rushing_tds": 6.0,
            "receptions": 1.0,
            "receiving_yards": 0.1,
            "receiving_tds": 6.0,
        }
    else:
        components = {target: 1.0}
    loaded = []
    for stat, coefficient in components.items():
        match = [
            r
            for r in policies
            if (r["horizon"], r["target"], r["policy"]) == (horizon, stat, policy)
        ]
        if len(match) != 1:
            raise ValueError(f"Expected one saved policy for {horizon}/{stat}/{policy}")
        record = match[0]
        weights = np.asarray(record["weights"], dtype=float)
        if (
            len(weights) != len(record["members"])
            or not len(weights)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1)
        ):
            raise ValueError("Saved policy weights are invalid")
        for name, weight in zip(record["members"], weights, strict=True):
            path = directory / horizon / stat / name / "pipeline.joblib"
            if str(path.relative_to(directory)) not in manifest["files"]:
                raise ValueError(f"Policy member is not in the completed manifest: {path}")
            bundle = joblib.load(path)
            if bundle["target"] != stat:
                raise ValueError("Saved pipeline target does not match its policy")
            loaded.append({"pipeline": bundle, "weight": float(weight * coefficient)})
    return {"horizon": horizon, "target": target, "policy": policy, "components": loaded}


def predict_policy(bundle, x, feature_names, exposure):
    """Align each member's saved source columns by name, then apply its stored weight."""
    x, exposure = np.asarray(x), np.asarray(exposure, dtype=float)
    names = list(feature_names)
    if x.ndim != 2 or x.shape != (len(exposure), len(names)):
        raise ValueError("Feature matrix, names, and exposure must align")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique")
    if not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise ValueError("Forecast exposure must be finite and positive")
    indices = {name: i for i, name in enumerate(names)}
    prediction = np.zeros(len(exposure), dtype=float)
    for component in bundle["components"]:
        pipeline = component["pipeline"]
        required = pipeline.get("feature_order", [])
        missing = set(required) - set(indices)
        if missing:
            raise ValueError(f"Missing forecast input features: {sorted(missing)}")
        values = x[:, [indices[name] for name in required]]
        member = predict_pipeline(pipeline, values, exposure)
        if member.shape != prediction.shape or not np.isfinite(member).all():
            raise ValueError("Saved member produced invalid forecast predictions")
        prediction += component["weight"] * member
    return prediction


def _save_pipeline(bundle, dest, x, exposure):
    """Verify a serialization roundtrip before publishing the pipeline file."""
    expected = predict_pipeline(bundle, x, exposure)
    temporary = dest / "pipeline.joblib.tmp"
    joblib.dump(bundle, temporary, compress=3)
    restored = joblib.load(temporary)
    actual = predict_pipeline(restored, x, exposure)
    if not np.isfinite(actual).all():
        raise ValueError("Final fitted artifact produced nonfinite predictions")
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    temporary.replace(dest / "pipeline.joblib")


def refit(out):
    out = Path(out)
    config = json.loads((out / "config.json").read_text())
    annual = pl.read_parquet(out / "analysis/annual_metrics.parquet")
    historical_predictions = pl.read_parquet(out / "analysis/predictions.parquet")
    year = max(config["years"]) + 1
    directory = out / "final_models"
    signature = _signature(out)
    if (directory / "manifest.json").exists():
        verify_refit(out)
        return directory
    if (directory / "signature.json").exists():
        if json.loads((directory / "signature.json").read_text()) != signature:
            raise ValueError("Partial final refit inputs, selection, source, or packages changed")
    else:
        directory.mkdir(exist_ok=True)
        if any(not path.name.endswith(".tmp") for path in directory.iterdir()):
            raise ValueError("Existing final_models has no signature; cannot resume it safely")
        json_write(directory / "signature.json", signature)
    sources = [Path(__file__), Path(choose_members.__code__.co_filename)]
    for source in sources:
        shutil.copyfile(source, directory / source.name)
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    input_records = {}
    records, policy_records = [], []
    for horizon in config["horizons"]:
        dataset = load_dataset(horizon, tuple(config["origins"]), config["input_cache"])
        inputs = out / horizon / "inputs.json"
        if inputs.exists() and json.loads(inputs.read_text())["provenance"] != dataset.provenance:
            raise ValueError(f"Final refit input provenance differs from training: {horizon}")
        years = dataset.panel["year"].to_numpy()
        targets, _ = stat_arrays(dataset)
        input_records[horizon] = dataset.provenance
        for target in config["targets"]:
            scores = annual.filter(
                (pl.col("horizon") == horizon)
                & (pl.col("target") == target)
                & (pl.col("recipe") != "persistence")
            )
            ranked = choose_members(
                scores, year, config["selection_years"], count=len(config["recipes"]) + 1
            )
            diverse = diverse_members(ranked, config)
            policies = {
                "selected": [ranked[0]["recipe"]] if ranked else [],
                "ensemble_top3": [r["recipe"] for r in ranked[:3]],
                "ensemble_diverse": diverse,
                "ensemble_convex": diverse,
            }
            required = {name for members in policies.values() for name in members}
            choices = [r for r in ranked if r["recipe"] in required]
            if not choices:
                raise ValueError(f"No complete earlier selection history for {horizon}/{target}")
            prior = historical_predictions.filter(
                (pl.col("horizon") == horizon)
                & (pl.col("target") == target)
                & (pl.col("year") < year)
                & (pl.col("year") >= year - config["selection_years"])
            )
            for policy, members in policies.items():
                weights = (
                    convex_weights(prior, members)
                    if policy == "ensemble_convex"
                    else [1 / len(members)] * len(members)
                )
                policy_records.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "policy": policy,
                        "members": members,
                        "weights": weights,
                        "selection_max_year": year - 1,
                    }
                )
            # Persist the union needed to reproduce every prespecified blend policy.
            for rank, choice in enumerate(choices):
                recipe_id = choice["recipe"]
                dest = directory / horizon / target / recipe_id
                dest.mkdir(parents=True, exist_ok=True)
                if (dest / "complete.json").exists():
                    receipt = json.loads((dest / "complete.json").read_text())
                    _verify_files(dest, receipt)
                    records.append(json.loads((dest / "metadata.json").read_text()))
                    continue
                if recipe_id == "training_mean":
                    mask = years < year
                    rates = targets[mask, STATS.index(target)] / dataset.exposure[mask]
                    bundle = {
                        "constant_rate": float(np.nanmean(rates)),
                        "target": target,
                        "feature_order": [],
                        "postprocessing": "predict_pipeline",
                    }
                    _save_pipeline(bundle, dest, None, dataset.exposure[mask][:16])
                    record = {
                        "horizon": horizon,
                        "target": target,
                        "recipe": recipe_id,
                        "rank": rank,
                        "training_max_year": int(years[mask].max()),
                    }
                else:
                    recipe = next(r for r in config["recipes"] if r["id"] == recipe_id)
                    mask = years < year
                    if recipe["history"]:
                        mask &= years >= year - recipe["history"]
                    cols = view_mask(dataset.names, recipe["view"])
                    names = np.array(dataset.names)[cols].tolist()
                    x = dataset.x[mask][:, cols]
                    y = targets[mask, STATS.index(target)] / dataset.exposure[mask]
                    with threadpool_limits(limits=config["threads"]):
                        rep = fit_representation(x, names, recipe["representation"], recipe["seed"])
                        transformed = rep.transform(x)
                        if recipe.get("target_mode") == "factorized" and target in DENOMINATORS:
                            model = FactorizedTree(recipe, target, config["threads"])
                            opportunities = targets[mask, STATS.index(DENOMINATORS[target])]
                            model.fit(transformed, y, opportunities / dataset.exposure[mask])
                            metadata = model.metadata
                        else:
                            model, metadata = build_estimator(
                                recipe["family"],
                                recipe["params"],
                                recipe["seed"],
                                device=recipe["device"],
                                threads=config["threads"],
                            )
                            known = np.isfinite(y)
                            model.fit(transformed[known], y[known])
                    bundle = {
                        "model": model,
                        "representation": rep,
                        "feature_order": names,
                        "target": target,
                        "output": "per calendar week; multiply by forecast exposure",
                        "postprocessing": (
                            "predict_pipeline: zero-clip count rates before exposure"
                        ),
                        "count_stat": target in COUNT_STATS,
                    }
                    _save_pipeline(bundle, dest, x[:16], dataset.exposure[mask][:16])
                    record = {
                        "horizon": horizon,
                        "target": target,
                        "recipe": recipe_id,
                        "rank": rank,
                        "backend": metadata,
                        "training_min_year": int(years[mask].min()),
                        "training_max_year": int(years[mask].max()),
                    }
                    del model, rep, transformed, bundle
                    gc.collect()
                record.update(selection_max_year=year - 1, selection_loss=choice["selection_loss"])
                records.append(record)
                json_write(dest / "metadata.json", record)
                json_write(
                    dest / "complete.json",
                    {
                        "files": {
                            name: digest(dest / name)
                            for name in ["pipeline.joblib", "metadata.json"]
                        },
                        "artifact_roundtrip_verified": True,
                    },
                )
                print(f"Refitted {horizon}/{target}/{recipe_id}", flush=True)
    json_write(directory / "policies.json", policy_records)
    json_write(
        directory / "manifest.json",
        {
            "status": "complete",
            "signature": signature,
            "purpose": "Next-use fitted artifacts; no prospective validation",
            "selection_year": year,
            "training_cutoff_year": year - 1,
            "prediction_entrypoint": "gpu_lab.refit.load_policy + predict_policy",
            "packages": packages(),
            "sources": source_hashes,
            "input_provenance": input_records,
            "selection_metrics_sha256": digest(out / "analysis/annual_metrics.parquet"),
            "models": records,
            "files": {
                str(p.relative_to(directory)): digest(p)
                for p in directory.rglob("*")
                if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".tmp")
            },
        },
    )
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    refit(RUNS / args.run_id)
