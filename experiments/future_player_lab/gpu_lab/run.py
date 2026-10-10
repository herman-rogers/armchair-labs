"""Resumable, serial-GPU, raw-input RB tree campaign with immutable fit checkpoints."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .recipes import recipe_library, view_mask

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/future_player_lab/runs"
COUNT_STATS = {"carries", "rushing_tds", "targets", "receptions", "receiving_tds"}
STATS = [
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "points",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    tmp.replace(path)


def command(*args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def packages():
    result = {}
    for name in [
        "numpy",
        "polars",
        "scipy",
        "scikit-learn",
        "xgboost",
        "catboost",
        "lightgbm",
        "cuml-cu13",
        "cupy-cuda13x",
        "numba",
    ]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def source_paths():
    return [
        *[
            Path(__file__).parent / name
            for name in [
                "run.py",
                "recipes.py",
                "data.py",
                "representations.py",
                "factorized.py",
                "parallel.py",
            ]
        ],
        ROOT / "experiments/future_player_lab/gpu_backends.py",
        ROOT / "experiments/future_player_lab/notebooks/raw_distribution_data.py",
        ROOT / "experiments/future_player_lab/notebooks/rb_probability.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]


def stat_arrays(dataset):
    targets = np.asarray(dataset.targets, dtype=float)
    baseline = np.asarray(dataset.baseline, dtype=float)
    coefficients = np.array([0, 0.1, 6, 0, 1, 0.1, 6])
    # Propagate unknown stats used by the scoring rule; unused nulls do not erase points.
    scoring = coefficients != 0
    points = targets[:, scoring] @ coefficients[scoring]
    baseline_points = baseline[:, scoring] @ coefficients[scoring]
    return np.column_stack([targets, points]), np.column_stack([baseline, baseline_points])


def fold_masks(years, year, history=0):
    train = years < year
    if history:
        train &= years >= year - history
    test = years == year
    if not train.any() or not test.any():
        raise ValueError(f"Empty fold at {year}")
    return train, test


def row_frame(dataset, mask, stat, actual, prediction, recipe, year, elapsed):
    panel = dataset.panel
    keys = [
        n
        for n in [
            "player_id",
            "player_display_name",
            "team",
            "candidate_basis",
            "year",
            "origin",
            "weeks",
            "horizon",
            "age",
        ]
        if n in panel.columns
    ]
    result = panel.filter(pl.Series(mask)).select(keys)
    return result.with_columns(
        pl.lit(stat).alias("target"),
        pl.Series("actual", actual),
        pl.Series("prediction", prediction),
        pl.lit(recipe).alias("recipe"),
        pl.lit(year - 1).alias("training_max_year"),
        pl.lit(elapsed).alias("fit_seconds"),
    )


def verify_checkpoint(path):
    receipt = json.loads((path / "complete.json").read_text())
    for name, sha in receipt["files"].items():
        if digest(path / name) != sha:
            raise ValueError(f"Corrupt checkpoint: {path / name}")
    return receipt


def write_baselines(dataset, out, years):
    targets, baseline = stat_arrays(dataset)
    season = dataset.panel["year"].to_numpy()
    rows = []
    for year in years:
        train, test = fold_masks(season, year)
        for j, stat in enumerate(STATS):
            y = targets[:, j]
            valid = test & np.isfinite(y)
            pred = baseline[:, j]
            rate = y[train] / dataset.exposure[train]
            mean = float(np.nanmean(rate))
            for name, values in [("persistence", pred), ("training_mean", mean * dataset.exposure)]:
                rows.append(row_frame(dataset, valid, stat, y[valid], values[valid], name, year, 0))
    pl.concat(rows).write_parquet(out / "baselines.parquet")


def run_campaign(out, config, resume=False):
    """Prevent two processes from writing the same run, including simultaneous resumes."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / f".{out.name}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns run {out.name}") from exc
        return _run_campaign(out, config, resume)


def _run_campaign(out, config, resume=False):
    from experiments.future_player_lab.gpu_backends import build_estimator

    from .data import load_dataset
    from .factorized import DENOMINATORS, FactorizedTree
    from .representations import fit_representation

    out = Path(out)
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths()}
    signature = {"config": config, "sources": sources, "packages": packages()}
    if resume:
        original = json.loads((out / "signature.json").read_text())
        if signature != original:
            raise ValueError("Resume requires identical configuration, code, and packages")
    else:
        out.mkdir(parents=True, exist_ok=False)
        json_write(out / "signature.json", signature)
        json_write(out / "config.json", config)
        (out / "workspace.patch").write_text(command("git", "diff", "--binary"))
        for name in sources:
            dest = out / "implementation" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        json_write(
            out / "hardware.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "commit": command("git", "rev-parse", "HEAD"),
                "git_status": command("git", "status", "--short"),
                "gpu": command("nvidia-smi"),
                "cpu": command("lscpu"),
                "pid": os.getpid(),
                "threads": config["threads"],
                "gpu_workers": 1,
            },
        )
    recipes = config["recipes"]
    started = time.monotonic()
    failures, completed = [], 0
    manifest = {
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration_sha256": digest(out / "config.json"),
        "gpu_workers": 1,
    }
    json_write(out / "manifest.json", manifest)
    for horizon in config["horizons"]:
        print(f"Loading raw {horizon} inputs", flush=True)
        dataset = load_dataset(
            horizon, origins=tuple(config["origins"]), cache_dir=Path(config["input_cache"])
        )
        dest = out / horizon
        dest.mkdir(exist_ok=True)
        input_record = {
            "provenance": dataset.provenance,
            "features": dataset.names,
            "rows": dataset.panel.height,
        }
        if (
            resume
            and (dest / "inputs.json").exists()
            and json.loads((dest / "inputs.json").read_text()) != input_record
        ):
            raise ValueError("Resume input provenance changed")
        json_write(dest / "inputs.json", input_record)
        dataset.audit.write_parquet(dest / "target_audit.parquet")
        write_baselines(dataset, dest, config["years"])
        targets, _ = stat_arrays(dataset)
        years = dataset.panel["year"].to_numpy()
        for recipe in recipes:
            cols = view_mask(dataset.names, recipe["view"])
            names = np.asarray(dataset.names)[cols].tolist()
            x = dataset.x[:, cols]
            for year in config["years"]:
                checkpoint = dest / "fits" / recipe["id"] / str(year)
                if (checkpoint / "complete.json").exists():
                    verify_checkpoint(checkpoint)
                    completed += len(config["targets"])
                    continue
                checkpoint.mkdir(parents=True, exist_ok=True)
                train, test = fold_masks(years, year, recipe["history"])
                try:
                    before = time.monotonic()
                    with threadpool_limits(limits=config["threads"]):
                        representation = fit_representation(
                            x[train], names, recipe["representation"], seed=recipe["seed"]
                        )
                        tx = np.ascontiguousarray(
                            representation.transform(x[train]), dtype=np.float32
                        )
                        vx = np.ascontiguousarray(
                            representation.transform(x[test]), dtype=np.float32
                        )
                    transform_seconds = time.monotonic() - before
                    if not np.isfinite(tx).all() or not np.isfinite(vx).all():
                        raise ValueError("Preprocessing produced nonfinite inputs")
                    summaries, predictions = [], []
                    for stat in config["targets"]:
                        j = STATS.index(stat)
                        y = targets[:, j] / dataset.exposure
                        known = np.isfinite(y[train])
                        before = time.monotonic()
                        model, metadata = build_estimator(
                            recipe["family"],
                            recipe["params"],
                            seed=recipe["seed"],
                            device=recipe["device"],
                            threads=config["threads"],
                        )
                        with threadpool_limits(limits=config["threads"]):
                            if recipe.get("target_mode") == "factorized" and stat in DENOMINATORS:
                                model = FactorizedTree(recipe, stat, config["threads"])
                                denominator = STATS.index(DENOMINATORS[stat])
                                opportunities = targets[:, denominator] / dataset.exposure
                                model.fit(tx, y[train], opportunities[train])
                                metadata = model.metadata
                            else:
                                model.fit(tx[known], y[train][known].astype(np.float32))
                            rates = np.asarray(model.predict(vx), dtype=float).reshape(-1)
                        elapsed = time.monotonic() - before
                        if not np.isfinite(rates).all():
                            raise ValueError("Nonfinite model predictions")
                        if stat in COUNT_STATS:
                            rates = np.maximum(0, rates)
                        pred = rates * dataset.exposure[test]
                        predictions.append(
                            row_frame(
                                dataset,
                                test,
                                stat,
                                targets[test, j],
                                pred,
                                recipe["id"],
                                year,
                                elapsed,
                            )
                        )
                        summaries.append(
                            {
                                "stat": stat,
                                "seconds": elapsed,
                                "known_training_rows": int(known.sum()),
                                "backend": metadata,
                            }
                        )
                        if year == max(config["years"]):
                            # Trusted local artifacts; exact environment recorded in signature.json.
                            joblib.dump(model, checkpoint / f"{stat}.joblib", compress=3)
                        del model
                        gc.collect()
                    pl.concat(predictions).write_parquet(checkpoint / "predictions.parquet")
                    rep_summary = representation.summary()
                    json_write(
                        checkpoint / "metadata.json",
                        {
                            "recipe": recipe,
                            "year": year,
                            "training_max_year": int(years[train].max()),
                            "transform_seconds": transform_seconds,
                            "representation": rep_summary,
                            "fits": summaries,
                        },
                    )
                    if year == max(config["years"]):
                        joblib.dump(representation, checkpoint / "preprocessing.joblib", compress=3)
                        json_write(checkpoint / "feature_order.json", names)
                    json_write(
                        checkpoint / "complete.json",
                        {
                            "files": {
                                p.name: digest(p)
                                for p in sorted(checkpoint.iterdir())
                                if p.is_file() and p.name != "complete.json"
                            },
                        },
                    )
                    completed += len(config["targets"])
                    secs = sum(s["seconds"] for s in summaries)
                    print(
                        f"{horizon} {recipe['id']} {year}: {len(summaries)} fits, "
                        f"{tx.shape[1]} inputs, {secs:.1f}s + transform {transform_seconds:.1f}s "
                        f"({completed} fits completed)",
                        flush=True,
                    )
                    del representation, tx, vx
                except Exception as exc:
                    failure = {
                        "horizon": horizon,
                        "recipe": recipe["id"],
                        "year": year,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    json_write(checkpoint / "failure.json", failure)
                    print(f"FAILED {horizon} {recipe['id']} {year}: {exc}", flush=True)
                json_write(
                    out / "progress.json",
                    {
                        "completed_fits": completed,
                        "failed_folds": len(failures),
                        "seconds": time.monotonic() - started,
                        "last": str(checkpoint.relative_to(out)),
                    },
                )
                gc.collect()
        del dataset, targets, x
        gc.collect()
    json_write(out / "failures.json", failures)
    manifest.update(
        status="complete_with_failures" if failures else "complete",
        completed_fits=completed,
        failed_folds=len(failures),
        seconds=time.monotonic() - started,
        code_unchanged=sources == {str(p.relative_to(ROOT)): digest(p) for p in source_paths()},
    )
    if not manifest["code_unchanged"]:
        manifest["status"] = "invalid_source_change"
    json_write(out / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--parallel-cpu", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--input-cache", default=str(RUNS / "gpu_raw_inputs_001"))
    p.add_argument("--years", type=int, nargs="+", default=list(range(2018, 2026)))
    p.add_argument("--horizons", nargs="+", default=["season", "remaining"])
    p.add_argument("--origins", type=int, nargs="+", default=[2])
    p.add_argument("--targets", nargs="+", choices=STATS, default=STATS)
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        raise ValueError("Run ID must contain only letters, digits, underscores, or hyphens")
    config = {
        "seed": 20261003,
        "threads": args.threads,
        "input_cache": args.input_cache,
        "years": args.years,
        "horizons": args.horizons,
        "origins": args.origins,
        "targets": args.targets,
        "recipes": recipe_library(smoke=args.smoke),
        "selection_years": 3,
        "evaluation_years": [y for y in args.years if y >= 2021],
        "scoring": "rush/receive PPR only; excludes fumbles, passing, bonuses, conversions",
        "note": "Historical development evidence: 2025 has previously been examined.",
    }
    if args.parallel_cpu:
        from .parallel import run_parallel

        run_parallel(RUNS / args.run_id, config, args.resume)
    else:
        run_campaign(RUNS / args.run_id, config, args.resume)


if __name__ == "__main__":
    main()
