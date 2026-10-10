"""Verify completed research artifacts without fitting or selecting another model."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import polars as pl

from .analyze import KEY
from .run import COUNT_STATS, ROOT, RUNS, digest, json_write, verify_checkpoint


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(directory, manifest):
    for relative, expected in manifest["files"].items():
        require(
            digest(directory / relative) == expected, f"Artifact changed: {directory / relative}"
        )


def backend_leaves(metadata):
    if "components" in metadata:
        for component in metadata["components"]:
            yield from backend_leaves(component)
    else:
        yield metadata


def audit(out):
    config = json.loads((out / "config.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    require(manifest["status"] == "complete", "Training must have completed without failures")
    require(manifest["code_unchanged"], "Training code changed during the run")
    require(not json.loads((out / "failures.json").read_text()), "Training failures remain")
    signature = json.loads((out / "signature.json").read_text())
    for relative, expected in signature["sources"].items():
        require(
            digest(out / "implementation" / relative) == expected, f"Snapshot changed: {relative}"
        )
        require(digest(ROOT / relative) == expected, f"Training implementation changed: {relative}")

    expected_folds = len(config["horizons"]) * len(config["recipes"]) * len(config["years"])
    expected_fits = expected_folds * len(config["targets"])
    require(manifest["completed_fits"] == expected_fits, "Recorded target-fit count is incomplete")
    checkpoints = sorted(out.glob("*/fits/*/*/complete.json"))
    require(len(checkpoints) == expected_folds, "Unexpected number of completed checkpoints")
    tree_counts, target_counts, rows_by_horizon = Counter(), Counter(), Counter()
    targets = set(config["targets"])
    for horizon in config["horizons"]:
        for recipe in config["recipes"]:
            for year in config["years"]:
                checkpoint = out / horizon / "fits" / recipe["id"] / str(year)
                verify_checkpoint(checkpoint)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                require(metadata["recipe"] == recipe, f"Recipe mismatch: {checkpoint}")
                require(metadata["year"] == year, f"Year mismatch: {checkpoint}")
                require(metadata["training_max_year"] < year, f"Future training rows: {checkpoint}")
                require(
                    {f["stat"] for f in metadata["fits"]} == targets,
                    f"Targets missing: {checkpoint}",
                )
                for fit in metadata["fits"]:
                    require(
                        fit["known_training_rows"] > 0, f"No known training labels: {checkpoint}"
                    )
                    target_counts[recipe["family"]] += 1
                    for backend in backend_leaves(fit["backend"]):
                        require(
                            backend["family"] == recipe["family"], f"Backend mismatch: {checkpoint}"
                        )
                        if recipe["device"] == "cuda":
                            require(backend["device_verified"], f"Unverified GPU: {checkpoint}")
                            require(
                                backend["effective_device"].startswith("cuda"),
                                f"CPU fallback: {checkpoint}",
                            )
                        tree_counts[recipe["family"]] += 1
                predictions = pl.read_parquet(checkpoint / "predictions.parquet")
                require(
                    predictions.select(KEY).unique().height == predictions.height,
                    f"Duplicate forecast keys: {checkpoint}",
                )
                require(
                    set(predictions["target"]) == targets, f"Missing forecast targets: {checkpoint}"
                )
                require(
                    predictions["prediction"].is_finite().all(),
                    f"Invalid predictions: {checkpoint}",
                )
                require(
                    predictions["year"].unique().to_list() == [year],
                    f"Mixed forecast years: {checkpoint}",
                )
                require(
                    predictions["recipe"].unique().to_list() == [recipe["id"]],
                    f"Mixed forecast recipes: {checkpoint}",
                )
                counts = predictions.filter(pl.col("target").is_in(COUNT_STATS))["prediction"]
                require((counts >= 0).all(), f"Negative count predictions: {checkpoint}")
                rows_by_horizon[horizon] += predictions.height

    for relative in ["analysis", "final_models"]:
        directory = out / relative
        saved = json.loads((directory / "manifest.json").read_text())
        require(saved["status"] == "complete", f"Incomplete {relative}")
        verify_files(directory, saved)
    require(
        json.loads((out / "postprocess.json").read_text())["status"] == "complete",
        "Postprocessing is incomplete",
    )
    report = {
        "status": "passed",
        "configuration_sha256": digest(out / "config.json"),
        "training_manifest_sha256": digest(out / "manifest.json"),
        "expected_checkpoints": expected_folds,
        "verified_checkpoints": len(checkpoints),
        "target_pipeline_fits": sum(target_counts.values()),
        "target_pipeline_fits_by_family": dict(sorted(target_counts.items())),
        "estimator_fits_including_factorized_components": sum(tree_counts.values()),
        "estimator_fits_by_family": dict(sorted(tree_counts.items())),
        "candidate_forecast_rows": dict(rows_by_horizon),
        "checks": [
            "Every configured recipe/horizon/year/target completed",
            "All training and checkpoint artifact SHA256 hashes match",
            "All model training years precede their prediction year",
            "Every CUDA estimator reports verified effective GPU execution",
            "Forecast keys are unique, predictions finite, counts nonnegative",
            "Analysis and final-model artifact SHA256 hashes match",
            "Postprocessing completed",
        ],
        "limits": "Integrity checks do not establish point-in-time provider-vintage validity "
        "or prospective predictive skill.",
    }
    json_write(out / "audit.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    audit(RUNS / args.run_id)
