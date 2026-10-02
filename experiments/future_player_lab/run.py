"""Run with python -m experiments.future_player_lab.run; resumable immutable protocol."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .data import TARGETS, build_panel, digest, load_inputs
from .evaluate import interval, score, select
from .models import MultiTarget, Recipe, recipes, view_indices
from .report import generate

ROOT = Path(__file__).resolve().parents[2]
LAB = Path(__file__).resolve().parent


def write_json(path, value):
    def native(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=native, allow_nan=False))
    tmp.replace(path)


def task(run_path, position, horizon, config, candidate_specs):
    run = Path(run_path)
    folder = run / "tasks" / f"{position}_{horizon}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        completed = json.loads((folder / "complete.json").read_text())
        if any(digest(folder / n) != sha for n, sha in completed["artifacts"].items()):
            raise ValueError(f"Completed task artifacts changed: {folder.name}")
        return f"{position}/{horizon}: cached"
    panel = pl.read_parquet(run / "panel.parquet")
    ix = np.flatnonzero(
        (panel["position"].to_numpy() == position) & (panel["horizon"].to_numpy() == horizon)
    )
    panel = panel[ix]
    x = np.load(run / "x.npy", mmap_mode="r")[ix]
    y = np.load(run / "y.npy", mmap_mode="r")[ix]
    baseline = np.load(run / "baseline.npy", mmap_mode="r")[ix]
    names = json.loads((run / "features.json").read_text())["names"]
    years = panel["year"].to_numpy()
    exposure = panel["exposure"].to_numpy()
    origins = panel["origin"].to_numpy()
    evaluation = config["evaluation_years"]
    validation = list(range(min(evaluation) - config["inner_seasons"], max(evaluation) + 1))
    history, metrics, choices, errors, predictions, diagnostics = {}, [], [], [], [], []
    policy_residuals = {
        name: [[] for _ in TARGETS]
        for name in ("selected_mse", "selected_rank", "selected_ensemble")
    }
    candidates = [Recipe(**r) for r in candidate_specs]
    with threadpool_limits(limits=config["threads_per_worker"]):
        for year in validation:
            test = years == year
            if not test.any():
                continue
            current = {"persistence": baseline[test] * exposure[test, None]}
            for candidate in candidates:
                checkpoint = folder / f"{year}_{candidate.name}.npz"
                dpath = folder / f"{year}_{candidate.name}.json"
                train = years < year
                if candidate.train_years:
                    train &= years >= year - candidate.train_years
                cols = view_indices(names, candidate.view)
                try:
                    if checkpoint.exists() and dpath.exists():
                        diag = json.loads(dpath.read_text())
                        if digest(checkpoint) != diag["prediction_sha256"]:
                            raise ValueError("Forecast checkpoint checksum mismatch")
                        with np.load(checkpoint) as saved:
                            pred = saved["prediction"]
                    else:
                        start = time.monotonic()
                        weight = np.ones(int(train.sum()))
                        if candidate.half_life:
                            weight = 2.0 ** ((years[train] - year + 1) / candidate.half_life)
                        model = MultiTarget(candidate, config["seed"]).fit(
                            x[train][:, cols], y[train] / exposure[train, None], weight
                        )
                        pred = model.predict(x[test][:, cols]) * exposure[test, None]
                        diag = {
                            "year": year,
                            "candidate": candidate.name,
                            "train_rows": int(train.sum()),
                            "train_last_year": int(years[train].max()),
                            "fit_seconds": time.monotonic() - start,
                            "learners": [learner.diagnostics() for _, learner in model.learners],
                        }
                        # Save interpretable input evidence for unprojected trees/ridge.
                        evidence = []
                        interaction_evidence = []
                        for target_cols, learner in model.learners:
                            local_names = [names[cols[j]] for j in learner.transformer.active]
                            local_names += ["missing:" + names[j] for j in cols]
                            if candidate.representation == "interactions":
                                interaction_evidence.extend(
                                    [[local_names[a], local_names[b]] for a, b in learner.pairs]
                                )
                            if candidate.representation != "identity":
                                continue
                            estimator = learner.models[0]
                            weights = getattr(estimator, "feature_importances_", None)
                            if weights is None and hasattr(estimator, "coef_"):
                                weights = abs(np.atleast_2d(estimator.coef_)).mean(axis=0)
                            if weights is not None:
                                evidence += [
                                    {
                                        "feature": local_names[j],
                                        "weight": float(weights[j]),
                                        "targets": [TARGETS[k] for k in target_cols],
                                    }
                                    for j in np.argsort(-weights)[:40]
                                ]
                        diag["input_evidence"] = evidence
                        diag["learned_interaction_candidates"] = interaction_evidence
                        np.savez_compressed(checkpoint, prediction=pred)
                        diag["prediction_sha256"] = digest(checkpoint)
                        write_json(dpath, diag)
                    if pred.shape != (int(test.sum()), len(TARGETS)):
                        raise ValueError("Prediction shape mismatch")
                    if not np.isfinite(pred[:, 0]).all():
                        raise ValueError("Nonfinite points predictions")
                    current[candidate.name] = pred
                    diagnostics.append(diag)
                except Exception as exc:
                    errors.append(
                        {
                            "year": year,
                            "candidate": candidate.name,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    write_json(folder / "errors.json", errors)
            # Selection sees only prior-year losses; current-year metrics are appended below.
            radii = {}
            if year in evaluation:
                for policy, criterion in (
                    ("selected_mse", "mse"),
                    ("selected_rank", "ndcg24"),
                    ("selected_ensemble", "mse"),
                ):
                    ranked, inner, losses = select(
                        history, year, config["inner_seasons"], criterion
                    )
                    eligible = [name for name in ranked if name in current]
                    if not eligible:
                        raise ValueError("No successful selected candidate")
                    chosen = eligible[:3] if policy == "selected_ensemble" else eligible[:1]
                    current[policy] = np.mean([current[name] for name in chosen], axis=0)
                    radii[policy] = np.array(
                        [
                            interval(np.asarray(r), config["interval_alpha"])
                            for r in policy_residuals[policy]
                        ]
                    )
                    choices.append(
                        {
                            "year": year,
                            "policy": policy,
                            "chosen": chosen,
                            "validation_years": inner,
                            "losses": losses,
                        }
                    )
            for name, pred in current.items():
                if name not in policy_residuals:
                    history.setdefault(name, {})[year] = score(
                        y[test, 0], pred[:, 0], origins[test]
                    )
                if year not in evaluation:
                    continue
                for j, target in enumerate(TARGETS):
                    stats = score(y[test, j], pred[:, j], origins[test])
                    radius = radii.get(name, np.full(len(TARGETS), np.nan))[j]
                    stats["coverage80"] = None
                    stats["interval_width"] = None
                    if np.isfinite(radius):
                        valid = np.isfinite(y[test, j]) & np.isfinite(pred[:, j])
                        half = radius * exposure[test]
                        if valid.any():
                            stats["coverage80"] = float(
                                np.mean(abs(y[test, j][valid] - pred[:, j][valid]) <= half[valid])
                            )
                            stats["interval_width"] = float(np.mean(2 * half[valid]))
                    metrics.append(
                        {
                            "position": position,
                            "horizon": horizon,
                            "year": year,
                            "target": target,
                            "model": name,
                            **stats,
                        }
                    )
                    if name in policy_residuals:
                        valid = np.isfinite(y[test, j]) & np.isfinite(pred[:, j])
                        policy_residuals[name][j].extend(
                            ((y[test, j] - pred[:, j]) / exposure[test])[valid]
                        )
                # Every forecast saved; unknown labels serialize as nulls in Parquet.
                out = panel.filter(pl.Series(test)).select(
                    "player_id",
                    "name",
                    "position",
                    "horizon",
                    "year",
                    "origin",
                    "end",
                    "population",
                )
                out = out.with_columns(pl.lit(name).alias("model"))
                for j, target in enumerate(TARGETS):
                    radius = radii.get(name, np.full(len(TARGETS), np.nan))[j]
                    out = out.with_columns(
                        pl.Series(f"actual_{target}", y[test, j]).fill_nan(None),
                        pl.Series(f"pred_{target}", pred[:, j]).fill_nan(None),
                        pl.Series(f"low_{target}", pred[:, j] - radius * exposure[test]).fill_nan(
                            None
                        ),
                        pl.Series(f"high_{target}", pred[:, j] + radius * exposure[test]).fill_nan(
                            None
                        ),
                    )
                predictions.append(out)
            print(
                f"{position}/{horizon}/{year}: {len(current)} forecasts, {len(errors)} failures",
                flush=True,
            )
    pl.DataFrame(metrics, infer_schema_length=None).write_parquet(folder / "metrics.parquet")
    combined = pl.concat(predictions, how="vertical_relaxed")
    combined.write_parquet(folder / "predictions.parquet")
    cohort_metrics = []
    adaptive = combined.filter(pl.col("model").is_in(["persistence", *policy_residuals]))
    for (year, population, model), block in adaptive.partition_by(
        ["year", "population", "model"], as_dict=True
    ).items():
        cohort_metrics.append(
            {
                "year": year,
                "population": population,
                "model": model,
                **score(
                    block["actual_points"].to_numpy(),
                    block["pred_points"].to_numpy(),
                    block["origin"].to_numpy(),
                ),
            }
        )
    pl.DataFrame(cohort_metrics, infer_schema_length=None).write_parquet(folder / "cohorts.parquet")
    write_json(folder / "choices.json", choices)
    write_json(folder / "diagnostics.json", diagnostics)
    write_json(folder / "errors.json", errors)
    write_json(
        folder / "complete.json",
        {
            "status": "complete",
            "errors": len(errors),
            "metric_rows": len(metrics),
            "prediction_rows": combined.height,
            "artifacts": {
                p.name: digest(p)
                for p in folder.iterdir()
                if p.is_file() and p.name != "complete.json"
            },
        },
    )
    return f"{position}/{horizon}: complete ({len(errors)} candidate failures)"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB / "config.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        raise ValueError("Unsafe run identifier")
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(
            positions=["QB"],
            horizons=["season"],
            evaluation_years=[2024, 2025],
            inner_seasons=1,
            trials_per_family=1,
            families=["ridge", "hist"],
            workers=1,
        )
    if (
        not config["evaluation_years"]
        or min(config["evaluation_years"]) - config["inner_seasons"] < 2005
    ):
        raise ValueError("Evaluation needs earlier training and validation seasons")
    unknown = set(config["horizons"]) - {"season", "next_week", "next_four", "remaining"}
    if unknown or not 0 < config["interval_alpha"] < 1 or config["trials_per_family"] < 1:
        raise ValueError("Invalid experiment configuration")
    run = LAB / "runs" / args.run_id
    protocol = {"config": config, "source": {p.name: digest(p) for p in sorted(LAB.glob("*.py"))}}
    if args.resume:
        saved = json.loads((run / "protocol.json").read_text())
        if protocol != saved:
            raise ValueError("Cannot resume after code/config changes; use a new run ID")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "protocol.json", protocol)
        write_json(run / "config.json", config)
        (run / "source").mkdir()
        for p in LAB.glob("*.py"):
            shutil.copy2(p, run / "source" / p.name)
    started = datetime.now(UTC).isoformat()
    tables, hashes = load_inputs(ROOT, config)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "polars", "scikit-learn", "scipy")
    }
    if args.resume and (run / "manifest.json").exists():
        old = json.loads((run / "manifest.json").read_text())
        if old["inputs"] != hashes or old["environment"] != versions:
            raise ValueError("Cannot resume with changed inputs/environment")
    manifest = {
        "status": "running",
        "started": started,
        "smoke": args.smoke,
        "inputs": hashes,
        "environment": versions,
        "python": sys.version,
        "research_only": True,
        "protocol": "chronological_discovery_v1",
    }
    write_json(run / "manifest.json", manifest)
    try:
        if not (run / "panel_audit.json").exists():
            panel, x, y, baseline, names, groups = build_panel(tables, config)
            panel.write_parquet(run / "panel.parquet")
            np.save(run / "x.npy", x)
            np.save(run / "y.npy", y)
            np.save(run / "baseline.npy", baseline)
            write_json(run / "features.json", {"names": names, "groups": groups})
            write_json(
                run / "panel_audit.json",
                {
                    "rows": panel.height,
                    "features": len(names),
                    "targets": list(TARGETS),
                    "players": panel["player_id"].n_unique(),
                    "years": panel["year"].unique().sort().to_list(),
                    "unknown_labels": {
                        t: int(np.isnan(y[:, j]).sum()) for j, t in enumerate(TARGETS)
                    },
                    "zero_points_rows": int((y[:, 0] == 0).sum()),
                    "future_feature_violations": panel.filter(
                        pl.col("history_max_time") > pl.col("cutoff_time")
                    ).height,
                    "cohorts": panel.group_by("position", "horizon", "population").len().to_dicts(),
                    "array_sha256": {
                        n: digest(run / n)
                        for n in (
                            "x.npy",
                            "y.npy",
                            "baseline.npy",
                            "panel.parquet",
                            "features.json",
                        )
                    },
                },
            )
            del x, y, baseline, panel
        else:
            audit = json.loads((run / "panel_audit.json").read_text())
            if any(digest(run / name) != sha for name, sha in audit["array_sha256"].items()):
                raise ValueError("Panel checkpoint changed")
        del tables
        candidates = [dataclasses.asdict(r) for r in recipes(config)]
        write_json(run / "candidates.json", candidates)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "POLARS_MAX_THREADS",
        ):
            os.environ[key] = str(config["threads_per_worker"])
        with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
            futures = [
                executor.submit(task, str(run), p, h, config, candidates)
                for p in config["positions"]
                for h in config["horizons"]
            ]
            for future in as_completed(futures):
                print(future.result(), flush=True)
        generate(run, config)
        unchanged = all(digest(ROOT / p) == sha for p, sha in hashes.items())
        if not unchanged:
            raise ValueError("Input artifacts changed during execution")
        if protocol["source"] != {p.name: digest(p) for p in sorted(LAB.glob("*.py"))}:
            raise ValueError("Experiment implementation changed during execution")
        errors = sum(
            json.loads(p.read_text())["errors"] for p in (run / "tasks").glob("*/complete.json")
        )
        manifest.update(
            status="complete" if not errors else "complete_with_failures",
            candidate_failures=errors,
            input_hashes_unchanged=True,
            finished=datetime.now(UTC).isoformat(),
        )
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_json(run / "manifest.json", manifest)
    print(f"Readout: {run / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
