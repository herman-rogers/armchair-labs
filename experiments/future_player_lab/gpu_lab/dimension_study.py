"""Prespecified projection-dimension sensitivity; descriptive, never policy selection.

The eight new recipes vary only projection width (16/128) from four main XGBoost
anchors. Raw and 48-component forecasts are reused after checkpoint verification.
The root completion manifest covers both training and matched scoring; the frozen
campaign runner writes its separate training manifest under ``training/``.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl

from .run import (
    COUNT_STATS,
    ROOT,
    RUNS,
    STATS,
    digest,
    json_write,
    packages,
    run_campaign,
    source_paths,
    verify_checkpoint,
)

KINDS = ("pca", "group_pca", "svd", "random_projection")
DIMENSIONS = (16, 128)
KEY = ["horizon", "year", "origin", "player_id", "target"]
DEFAULT_MAIN = "rb_gpu_20261003_001"
DEFAULT_RUN = "rb_gpu_dimensions_20261003_001"


def read_json(path):
    return json.loads(Path(path).read_text())


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def study_config(main_config):
    """Clone the fixed anchors, rejecting any unmatched model settings."""
    if (
        main_config["years"] != list(range(2018, 2026))
        or set(main_config["horizons"]) != {"season", "remaining"}
        or main_config["targets"] != STATS
        or main_config["evaluation_years"] != list(range(2021, 2026))
    ):
        raise ValueError("This prespecified study requires both horizons and all 2018–25 folds")
    raw = next(r for r in main_config["recipes"] if r["id"] == "r000_xgboost_raw_all")
    if raw["family"] != "xgboost" or raw["device"] != "cuda":
        raise ValueError("The raw anchor must use CUDA XGBoost")

    def settings(recipe):
        result = copy.deepcopy(recipe)
        result.pop("id")
        result["representation"].pop("kind")
        return result

    controls = {"raw": copy.deepcopy(raw)}
    recipes = []
    for kind in KINDS:
        candidates = [
            r
            for r in main_config["recipes"]
            if r["representation"]["kind"] == kind and settings(r) == settings(raw)
        ]
        if len(candidates) != 1 or candidates[0]["representation"]["n_components"] != 48:
            raise ValueError(f"Expected exactly one matched 48-component {kind} anchor")
        controls[kind] = copy.deepcopy(candidates[0])
        for width in DIMENSIONS:
            recipe = copy.deepcopy(candidates[0])
            recipe["id"] = f"d{len(recipes):03d}_xgboost_{kind}_{width}_all"
            recipe["representation"]["n_components"] = width
            recipes.append(recipe)
    config = copy.deepcopy(main_config)
    config["recipes"] = recipes
    config["note"] = (
        "Prespecified 16/128 projection-dimension sensitivity; descriptive historical "
        "development comparisons only. No retrospective promotion into main policies."
    )
    return config, controls


def timing_estimate(main, config, controls):
    """Read only timing metadata, never outcomes, before freezing/launching the study."""
    records, base_seconds = [], 0.0
    for kind in KINDS:
        rid = controls[kind]["id"]
        paths = []
        for horizon in config["horizons"]:
            for year in config["years"]:
                suffix = Path(horizon) / "fits" / rid / str(year) / "metadata.json"
                for path in [main / suffix, main / "workers/gpu" / suffix]:
                    if path.exists() and (path.parent / "complete.json").exists():
                        paths.append(path)
                        break
        metadata = [read_json(path) for path in paths]
        fit_times = [fit["seconds"] for m in metadata for fit in m["fits"]]
        if not fit_times:
            raise ValueError(f"No timing evidence for {kind}; estimate before launching")
        mean_fit = float(np.mean(fit_times))
        mean_transform = float(np.mean([m["transform_seconds"] for m in metadata]))
        folds = len(config["horizons"]) * len(config["years"]) * len(DIMENSIONS)
        seconds = folds * (len(config["targets"]) * mean_fit + mean_transform)
        base_seconds += seconds
        records.append(
            {
                "representation": kind,
                "observed_checkpoints": len(paths),
                "observed_target_fits": len(fit_times),
                "mean_fit_seconds": mean_fit,
                "mean_transform_seconds": mean_transform,
                "projected_seconds_at_48": seconds,
                "timing_sources": {str(p.relative_to(main)): digest(p) for p in paths},
            }
        )
    return {
        "estimated_at_utc": utc_now(),
        "expected_fits": 1024,
        "projected_seconds_at_48_component_cost": base_seconds,
        "budget_minutes": [15, 25],
        "records": records,
        "limitation": "128-component cost and serialization/GC overhead may be larger; "
        "the estimate does not change any settings or choose recipes.",
    }


def verify_files(directory, receipt):
    if not receipt.get("files"):
        raise ValueError(f"Empty artifact receipt: {directory}")
    for name, expected in receipt["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {directory / name}")


def main_ready(main):
    """GPU work starts only after the entire main postprocess pipeline completes."""
    files = [
        "manifest.json",
        "postprocess.json",
        "analysis/manifest.json",
        "final_models/manifest.json",
    ]
    statuses = {
        name: read_json(main / name).get("status") if (main / name).exists() else "missing"
        for name in files
    }
    if any(
        status in {"failed", "complete_with_failures", "invalid_source_change"}
        for status in statuses.values()
    ):
        raise ValueError(f"Main campaign needs repair before supplement: {statuses}")
    return all(status == "complete" for status in statuses.values())


def aligned_forecasts(candidate, reference):
    """Require identical keys, unknown-label masks and exposures; never silently intersect."""
    ordered = []
    for frame in [candidate, reference]:
        if frame.height != frame.select(KEY).unique().height:
            raise ValueError("Duplicate forecast keys")
        if frame.select(KEY).null_count().to_numpy().sum():
            raise ValueError("Null forecast keys")
        if not np.isfinite(frame["prediction"].to_numpy()).all():
            raise ValueError("Nonfinite predictions")
        if frame.filter(pl.col("training_max_year") >= pl.col("year")).height:
            raise ValueError("Forecast training chronology violation")
        if frame.filter(pl.col("target").is_in(COUNT_STATS) & (pl.col("prediction") < 0)).height:
            raise ValueError("Negative count prediction")
        ordered.append(frame.sort(KEY))
    left, right = ordered
    if not left.select(KEY).equals(right.select(KEY)):
        raise ValueError("Candidate/control forecast key sets differ")
    for column in ["actual", "weeks"]:
        a = left[column].to_numpy().astype(float)
        b = right[column].to_numpy().astype(float)
        if not np.array_equal(a, b, equal_nan=True):
            raise ValueError(f"Candidate/control {column} differs")
    exposure = left["weeks"].to_numpy()
    if not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise ValueError("Forecast exposure must be positive and finite")
    return left, right


def metrics(actual, prediction):
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    known = np.isfinite(actual)
    if not known.any():
        return {"n": 0, "mse": None, "rmse": None, "mae": None, "normalized_mse": None}
    error = actual[known] - prediction[known]
    mse = float(np.mean(error**2))
    return {
        "n": int(known.sum()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "normalized_mse": mse / max(float(np.var(actual[known])), 1e-8),
    }


def improvement(candidate, reference):
    if candidate is None or reference is None or reference == 0:
        return None
    return 100 * (1 - candidate / reference)


def checkpoint_rows(run, recipes, config, reused):
    """Verify receipts, CUDA execution and full target/fold coverage before reading scores."""
    rows, receipts = [], []
    for horizon in config["horizons"]:
        for recipe in recipes:
            for year in config["years"]:
                path = run / horizon / "fits" / recipe["id"] / str(year)
                receipt = verify_checkpoint(path)
                metadata = read_json(path / "metadata.json")
                if (
                    metadata["recipe"] != recipe
                    or metadata["year"] != year
                    or metadata["training_max_year"] >= year
                ):
                    raise ValueError(f"Checkpoint configuration/chronology mismatch: {path}")
                if sorted(fit["stat"] for fit in metadata["fits"]) != sorted(config["targets"]):
                    raise ValueError(f"Incomplete target set: {path}")
                for fit in metadata["fits"]:
                    backend = fit["backend"]
                    if (
                        not backend.get("device_verified")
                        or not backend.get("effective_device", "").startswith("cuda")
                        or backend["requested_params"] != recipe["params"]
                        or backend["seed"] != recipe["seed"]
                    ):
                        raise ValueError(f"Unverified CUDA anchor settings: {path}")
                if year == max(config["years"]):
                    expected = {f"{target}.joblib" for target in config["targets"]}
                    expected |= {"preprocessing.joblib", "feature_order.json"}
                    if not expected.issubset(receipt["files"]):
                        raise ValueError(f"Missing saved 2025 pipelines: {path}")
                frame = pl.read_parquet(path / "predictions.parquet")
                if (
                    set(frame["target"]) != set(config["targets"])
                    or set(frame["year"]) != {year}
                    or set(frame["horizon"]) != {horizon}
                    or set(frame["recipe"]) != {recipe["id"]}
                ):
                    raise ValueError(f"Forecast identity mismatch: {path}")
                aligned_forecasts(frame, frame)
                kind = recipe["representation"]["kind"]
                width = None if kind == "raw" else recipe["representation"]["n_components"]
                rows.append(
                    frame.with_columns(
                        pl.lit(kind).alias("representation"),
                        pl.lit(width, dtype=pl.Int64).alias("n_components"),
                        pl.lit(metadata["representation"]["output_features"]).alias(
                            "effective_components"
                        ),
                        pl.lit(run.parent.name if not reused else run.name).alias("source_run_id"),
                        pl.lit(str(path.relative_to(ROOT))).alias("source_checkpoint"),
                        pl.lit("reused_control" if reused else "new_fit").alias("forecast_source"),
                    )
                )
                receipts.append(
                    {
                        "checkpoint": str(path.relative_to(ROOT)),
                        "complete_sha256": digest(path / "complete.json"),
                        "files": receipt["files"],
                    }
                )
    return pl.concat(rows, how="vertical_relaxed"), receipts


def score_comparisons(rows, config, controls):
    annual, comparisons, annual_comparisons = [], [], []
    for horizon in config["horizons"]:
        for target in config["targets"]:
            subset = rows.filter((pl.col("horizon") == horizon) & (pl.col("target") == target))
            raw = subset.filter(pl.col("recipe") == controls["raw"]["id"])
            for (rid,), candidate in subset.group_by("recipe", maintain_order=True):
                candidate, raw_aligned = aligned_forecasts(candidate, raw)
                kind = candidate["representation"][0]
                width = candidate["n_components"][0]
                identity = {
                    "horizon": horizon,
                    "target": target,
                    "recipe": rid,
                    "representation": kind,
                    "n_components": width,
                }
                for (year,), part in candidate.group_by("year", maintain_order=True):
                    annual.append(
                        {
                            **identity,
                            "year": year,
                            "effective_components": int(part["effective_components"][0]),
                            **metrics(part["actual"], part["prediction"]),
                        }
                    )
                if kind == "raw":
                    continue
                reference = subset.filter(pl.col("recipe") == controls[kind]["id"])
                candidate, reference = aligned_forecasts(candidate, reference)
                aligned = candidate.with_columns(
                    raw_aligned["prediction"].alias("raw_prediction"),
                    reference["prediction"].alias("reference48_prediction"),
                )
                scopes = [
                    ("evaluation", config["evaluation_years"]),
                    ("all_folds", config["years"]),
                ]
                scopes += [("annual", [year]) for year in config["years"]]
                for scope, years in scopes:
                    part = aligned.filter(pl.col("year").is_in(years))
                    if sorted(part["year"].unique().to_list()) != sorted(years):
                        raise ValueError("Missing comparison year")
                    scored = metrics(part["actual"], part["prediction"])
                    raw_score = metrics(part["actual"], part["raw_prediction"])
                    ref_score = metrics(part["actual"], part["reference48_prediction"])
                    record = {
                        **identity,
                        "scope": scope,
                        "years": ",".join(map(str, years)),
                        "n_years": len(years),
                        "forecast_rows": part.height,
                        "unknown_labels": part.height - scored["n"],
                        **scored,
                        "effective_components_min": int(part["effective_components"].min()),
                        "effective_components_max": int(part["effective_components"].max()),
                        "raw_recipe": controls["raw"]["id"],
                        "raw_mse": raw_score["mse"],
                        "raw_rmse": raw_score["rmse"],
                        "raw_mae": raw_score["mae"],
                        "reference48_recipe": controls[kind]["id"],
                        "reference48_mse": ref_score["mse"],
                        "reference48_rmse": ref_score["rmse"],
                        "reference48_mae": ref_score["mae"],
                        "mse_improvement_pct": improvement(scored["mse"], raw_score["mse"]),
                        "vs_48_mse_improvement_pct": improvement(scored["mse"], ref_score["mse"]),
                    }
                    if scope == "annual":
                        annual_comparisons.append({**record, "year": years[0]})
                    else:
                        comparisons.append(record)
    frames = [
        pl.DataFrame(records, infer_schema_length=None)
        for records in [annual, comparisons, annual_comparisons]
    ]
    return (
        frames[0].sort("horizon", "target", "recipe", "year"),
        frames[1].sort("scope", "horizon", "target", "representation", "n_components"),
        frames[2].sort("horizon", "target", "representation", "n_components", "year"),
    )


def readout(out, comparisons):
    lines = [
        "# Projection-dimension sensitivity",
        "",
        "Descriptive historical development comparisons, not prospective validation. "
        "These results do not alter the main campaign's policies or fitted ensembles.",
        "",
        "Eight recipes were fixed before supplement outcomes were read: PCA, source-balanced "
        "PCA, SVD and sparse random projection at 16 and 128 components. They use exactly the "
        "main anchor's XGBoost hyperparameters, seed, data, folds and missingness processing. "
        "Raw and 48-component controls are checksum-verified saved main forecasts, without "
        "retraining. Source-balanced PCA scales columns so each source family has equal "
        "aggregate variance before a single global PCA. Effective widths are exported.",
        "",
        "All 1,024 new target fits cover both RB horizons, eight targets and 2018–25 folds. "
        "Every transform and model uses earlier seasons only. Tables below pool 2021–25 "
        "forecasts on identical player/origin/target rows; unknown labels are excluded "
        "identically. MSE improvement is positive when better than the named control. "
        "Direct PPR uses rush/receive scoring and excludes fumbles, passing, bonuses and "
        "conversions. Count forecasts are clipped to zero; yards and points remain signed.",
        "",
        "| Horizon | Representation | Requested components | PPR RMSE | Raw RMSE | "
        "MSE improvement vs raw | MSE improvement vs 48 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons.filter(
        (pl.col("scope") == "evaluation") & (pl.col("target") == "points")
    ).to_dicts():
        lines.append(
            f"| {row['horizon']} | {row['representation']} | {row['n_components']} | "
            f"{row['rmse']:.3f} | {row['raw_rmse']:.3f} | "
            f"{row['mse_improvement_pct']:+.2f}% | "
            f"{row['vs_48_mse_improvement_pct']:+.2f}% |"
        )
    lines += [
        "",
        "A single seed, one tree configuration and three projection sizes cannot "
        "establish a universal best unsupervised method. Rankings across many targets "
        "and sizes are exploratory and have no multiplicity adjustment. There is no "
        "fresh untouched test set; 2025 was already examined during prior research. "
        "Annual results expose seasonal variation but do not make this prospective.",
        "",
        "- [All-target matched comparisons](analysis/controlled_comparisons.parquet)",
        "- [Annual metrics](analysis/annual_metrics.parquet)",
        "- [Annual matched comparisons](analysis/annual_comparisons.parquet)",
        "- [Forecasts and effective widths](analysis/predictions.parquet)",
        "- [Frozen study configuration](config.json)",
        "- [Control checkpoint provenance](controls.json)",
        "- [Completion manifest](manifest.json)",
        "",
        "Saved 2025 pipelines are under `training/{horizon}/fits/{recipe}/2025/`; "
        "they were trained through 2024 for that retrospective forecast. They are "
        "not additional final refits for 2026.",
        "",
    ]
    (out / "readout.md").write_text("\n".join(lines))


def frozen_signature(main, config, controls):
    return {
        "config": config,
        "controls": controls,
        "main_run": str(main),
        "main_config_sha256": digest(main / "config.json"),
        "main_signature_sha256": digest(main / "signature.json"),
        "sources": {
            str(p.relative_to(ROOT)): digest(p) for p in [*source_paths(), Path(__file__).resolve()]
        },
        "packages": packages(),
    }


def verify_frozen(main, signature):
    if frozen_signature(main, signature["config"], signature["controls"]) != signature:
        raise ValueError("Study/main configuration, frozen sources or packages changed")
    main_signature = read_json(main / "signature.json")
    if (
        main_signature["packages"] != signature["packages"]
        or (
            "config" in main_signature
            and main_signature["config"] != read_json(main / "config.json")
        )
        or any(
            signature["sources"].get(name) != sha for name, sha in main_signature["sources"].items()
        )
    ):
        raise ValueError("Current environment does not match the main campaign signature")


def run_study(main, out, wait=False):
    main, out = Path(main).resolve(), Path(out).resolve()
    config, controls = study_config(read_json(main / "config.json"))
    signature = frozen_signature(main, config, controls)
    verify_frozen(main, signature)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / f".{out.name}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns the dimension study") from exc
        if out.exists():
            if read_json(out / "signature.json") != signature:
                raise ValueError("Existing study has different frozen settings or sources")
            previous = read_json(out / "manifest.json")
            if previous["status"] == "complete":
                verify_files(out, previous)
                return
        else:
            out.mkdir()
            json_write(out / "signature.json", signature)
            json_write(out / "config.json", config)
            json_write(out / "timing_estimate.json", timing_estimate(main, config, controls))
            for name in signature["sources"]:
                destination = out / "implementation" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
        manifest = {
            "status": "waiting",
            "created_utc": utc_now(),
            "expected_fits": 1024,
            "completed_fits": 0,
            "reused_control_fits": 640,
            "scope": "Descriptive development comparison; no main policy changes",
            "main_run": str(main),
            "signature_sha256": digest(out / "signature.json"),
        }
        json_write(out / "manifest.json", manifest)
        print(
            f"Frozen eight recipes; waiting for {main.name} postprocess completion. "
            "Estimated runtime 15–25 minutes after that gate.",
            flush=True,
        )
        try:
            while not main_ready(main):
                if not wait:
                    raise ValueError("Main postprocess is incomplete; use --wait")
                time.sleep(30)
            verify_frozen(main, signature)
            for subdir in ["analysis", "final_models"]:
                verify_files(main / subdir, read_json(main / subdir / "manifest.json"))
            control_rows, receipts = checkpoint_rows(main, list(controls.values()), config, True)
            control_record = {
                "main_config_sha256": digest(main / "config.json"),
                "main_signature_sha256": digest(main / "signature.json"),
                "main_postprocess_sha256": digest(main / "postprocess.json"),
                "main_analysis_manifest_sha256": digest(main / "analysis/manifest.json"),
                "main_refit_manifest_sha256": digest(main / "final_models/manifest.json"),
                "inputs": {h: read_json(main / h / "inputs.json") for h in config["horizons"]},
                "checkpoints": receipts,
                "reused_target_fits": 640,
            }
            if (out / "controls.json").exists() and read_json(
                out / "controls.json"
            ) != control_record:
                raise ValueError("Verified main controls changed during study resume")
            json_write(out / "controls.json", control_record)
            manifest.update(status="running", training_started_utc=utc_now())
            json_write(out / "manifest.json", manifest)
            training = out / "training"
            run_campaign(training, config, resume=training.exists())
            trained = read_json(training / "manifest.json")
            if trained["status"] != "complete" or trained["completed_fits"] != 1024:
                raise ValueError(f"Supplement training incomplete: {trained}")
            verify_frozen(main, signature)
            for horizon in config["horizons"]:
                if (
                    read_json(training / horizon / "inputs.json")
                    != control_record["inputs"][horizon]
                ):
                    raise ValueError(f"Supplement inputs differ from main: {horizon}")
            manifest.update(status="scoring", completed_fits=1024)
            json_write(out / "manifest.json", manifest)
            new_rows, new_receipts = checkpoint_rows(training, config["recipes"], config, False)
            json_write(out / "training_checkpoints.json", new_receipts)
            rows = pl.concat([control_rows, new_rows], how="vertical_relaxed").sort("recipe", *KEY)
            annual, comparisons, annual_comparisons = score_comparisons(rows, config, controls)
            analysis = out / "analysis"
            analysis.mkdir(exist_ok=True)
            for name, frame in [
                ("predictions", rows),
                ("annual_metrics", annual),
                ("controlled_comparisons", comparisons),
                ("annual_comparisons", annual_comparisons),
            ]:
                frame.write_parquet(analysis / f"{name}.parquet")
            readout(out, comparisons)
            verify_frozen(main, signature)
            manifest.update(
                status="complete",
                completed_utc=utc_now(),
                completed_checkpoints=len(new_receipts),
                reused_control_checkpoints=len(receipts),
                training_seconds=trained["seconds"],
                all_devices_verified_cuda=True,
                matched_control_rows_verified=True,
                source_unchanged=True,
                files={
                    str(p.relative_to(out)): digest(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file()
                    and p != out / "manifest.json"
                    and not p.name.endswith(".tmp")
                    and not p.name.endswith(".lock")
                },
            )
            json_write(out / "manifest.json", manifest)
            print(f"Dimension study complete: {out}", flush=True)
        except Exception as exc:
            manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            json_write(out / "manifest.json", manifest)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run-id", default=DEFAULT_MAIN)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    for run_id in [args.run_id, args.main_run_id]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
            raise ValueError("Unsafe run ID")
    if args.run_id == args.main_run_id:
        raise ValueError("Supplement must use an isolated run directory")
    run_study(RUNS / args.main_run_id, RUNS / args.run_id, wait=args.wait)


if __name__ == "__main__":
    main()
