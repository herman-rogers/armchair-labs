"""Refit a completed experiment's selected pipelines for an unlabeled preseason.

This creates research forecasts, with an explicit issuance time and data-cutoff basis.
It never represents reconstructed preseason predictions as originally issued forecasts.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from ..data import TARGETS, build_panel, digest, load_inputs
from ..evaluate import interval, select
from ..models import MultiTarget, Recipe, view_indices

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args(argv)
    for value in (args.source, args.run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
            raise ValueError("Unsafe run identifier")
    source, output = LAB / "runs" / args.source, LAB / "runs" / args.run_id
    source_manifest = json.loads((source / "manifest.json").read_text())
    if source_manifest["status"] != "complete":
        raise ValueError(
            "Inference requires a completed source experiment without failed candidates"
        )
    output.mkdir(parents=True, exist_ok=False)
    implementation = {
        str(p.relative_to(LAB)): digest(p) for p in [*LAB.glob("*.py"), Path(__file__)]
    }
    for name in implementation:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LAB / name, target)
    issued = datetime.now(UTC).isoformat()
    config = json.loads((source / "config.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    if config != protocol["config"]:
        raise ValueError("Source experiment configuration changed")
    if digest(LAB / "models.py") != protocol["source"]["models.py"]:
        raise ValueError("Estimator implementation differs from the evaluated experiment")
    versions = {name: importlib.metadata.version(name) for name in source_manifest["environment"]}
    if versions != source_manifest["environment"]:
        raise ValueError("Estimator dependency versions have changed")
    audit = json.loads((source / "panel_audit.json").read_text())
    if any(digest(source / name) != sha for name, sha in audit["array_sha256"].items()):
        raise ValueError("Source training arrays changed")
    tables, hashes = load_inputs(ROOT, config)
    if hashes != source_manifest["inputs"]:
        raise ValueError("Source experiment inputs have changed")
    original = pl.read_parquet(source / "panel.parquet")
    if original["year"].max() >= args.year:
        raise ValueError("Forecast year must follow all training examples")
    names = json.loads((source / "features.json").read_text())["names"]
    train_x = np.load(source / "x.npy", mmap_mode="r")
    train_y = np.load(source / "y.npy", mmap_mode="r")
    for key in ("preseason_features", "season_outcomes"):
        tables[key] = tables[key].filter(pl.col("forecast_season") == args.year)
    if not tables["preseason_features"].height:
        raise ValueError("No cutoff-known candidate records for requested year")
    if tables["season_outcomes"]["outcome_complete"].any():
        raise ValueError("This command is for unlabeled forecast seasons")
    forecast_config = {**config, "evaluation_years": [args.year], "horizons": ["season"]}
    panel, x, labels, baseline, forecast_names, _ = build_panel(
        tables, forecast_config, include_unlabeled=True
    )
    if forecast_names != names or not np.isnan(labels).all():
        raise ValueError("Inference feature contract mismatch or unexpected outcome labels")
    candidates = {
        r["name"]: Recipe(**r) for r in json.loads((source / "candidates.json").read_text())
    }
    records, decisions = [], []
    with threadpool_limits(limits=config["threads_per_worker"]):
        for position in config["positions"]:
            folder = source / "tasks" / f"{position}_season"
            completed = json.loads((folder / "complete.json").read_text())
            if any(digest(folder / name) != sha for name, sha in completed["artifacts"].items()):
                raise ValueError("Training experiment artifacts changed")
            metrics = pl.read_parquet(folder / "metrics.parquet").filter(
                pl.col("target") == "points"
            )
            history = {}
            for row in metrics.iter_rows(named=True):
                if not row["model"].startswith("selected_"):
                    history.setdefault(row["model"], {})[row["year"]] = row
            forecast_mask = panel["position"].to_numpy() == position
            future = panel.filter(pl.Series(forecast_mask))
            fx = x[forecast_mask]
            predictions = {
                "persistence": baseline[forecast_mask] * future["exposure"].to_numpy()[:, None]
            }
            old = pl.read_parquet(folder / "predictions.parquet")
            for policy, criterion in (
                ("selected_mse", "mse"),
                ("selected_rank", "ndcg24"),
                ("selected_ensemble", "mse"),
            ):
                order, validation, losses = select(
                    history, args.year, config["inner_seasons"], criterion
                )
                chosen = order[:3] if policy == "selected_ensemble" else order[:1]
                for name in chosen:
                    if name in predictions:
                        continue
                    candidate = candidates[name]
                    train = (
                        (original["position"].to_numpy() == position)
                        & (original["horizon"].to_numpy() == "season")
                        & (original["year"].to_numpy() < args.year)
                    )
                    if candidate.train_years:
                        train &= original["year"].to_numpy() >= args.year - candidate.train_years
                    cols = view_indices(names, candidate.view)
                    years = original["year"].to_numpy()[train]
                    weight = np.ones(int(train.sum()))
                    if candidate.half_life:
                        weight = 2.0 ** ((years - args.year + 1) / candidate.half_life)
                    exposure = original["exposure"].to_numpy()[train]
                    model = MultiTarget(candidate, config["seed"]).fit(
                        train_x[train][:, cols], train_y[train] / exposure[:, None], weight
                    )
                    predictions[name] = (
                        model.predict(fx[:, cols]) * future["exposure"].to_numpy()[:, None]
                    )
                prediction = np.mean([predictions[name] for name in chosen], axis=0)
                block = future.select(
                    "player_id", "name", "position", "year", "forecast_cutoff_date"
                )
                block = block.with_columns(
                    pl.lit(policy).alias("policy"),
                    pl.lit(issued).alias("issued_at"),
                    pl.lit("preseason_reconstruction").alias("forecast_basis"),
                )
                residual_history = old.filter(pl.col("model") == policy)
                for j, target in enumerate(TARGETS):
                    residual = (
                        (residual_history[f"actual_{target}"] - residual_history[f"pred_{target}"])
                        / (residual_history["end"] - residual_history["origin"])
                    ).to_numpy()
                    radius = (
                        interval(residual, config["interval_alpha"]) * future["exposure"].to_numpy()
                    )
                    low, high = prediction[:, j] - radius, prediction[:, j] + radius
                    if j in (1, 3, 4):
                        low = np.maximum(0, low)
                    if j == 4:
                        high = np.minimum(future["exposure"].to_numpy(), high)
                    block = block.with_columns(
                        pl.Series(f"pred_{target}", prediction[:, j]).fill_nan(None),
                        pl.Series(f"low_{target}", low).fill_nan(None),
                        pl.Series(f"high_{target}", high).fill_nan(None),
                    )
                records.append(block)
                decisions.append(
                    {
                        "position": position,
                        "policy": policy,
                        "chosen": chosen,
                        "validation_years": validation,
                        "losses": losses,
                    }
                )
            print(f"{position}: forecast {future.height} players", flush=True)
    forecasts = pl.concat(records)
    forecasts = forecasts.with_columns(
        pl.col("pred_points")
        .rank("ordinal", descending=True)
        .over("position", "policy")
        .alias("position_rank")
    )
    forecasts.sort("policy", "position", "position_rank").write_parquet(
        output / "forecasts.parquet"
    )
    forecasts.sort("policy", "position", "position_rank").write_csv(output / "forecasts.csv")
    if any(digest(ROOT / name) != sha for name, sha in hashes.items()):
        raise ValueError("Input tables changed during inference")
    if any(digest(LAB / name) != sha for name, sha in implementation.items()):
        raise ValueError("Implementation changed during inference")
    (output / "choices.json").write_text(json.dumps(decisions, indent=2))
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_run": args.source,
                "source_manifest_sha256": digest(source / "manifest.json"),
                "inputs": hashes,
                "issued_at": issued,
                "forecast_year": args.year,
                "forecast_basis": "preseason_reconstruction",
                "unknown_labels_verified": True,
                "rows": forecasts.height,
                "code_sha256": {
                    str(p.relative_to(LAB)): digest(p) for p in [*LAB.glob("*.py"), Path(__file__)]
                },
            },
            indent=2,
        )
    )
    shutil.copy2(__file__, output / "forecast.py")
    (output / "README.md").write_text(
        f"# {args.year} preseason reconstruction\n\n"
        f"Issued {issued}. These forecasts were generated now using cutoff-known preseason inputs "
        "and training outcomes from earlier seasons. They were not issued before the NFL season "
        "and are not an untouched prospective validation. No outcome labels from the forecast "
        "year were used. Each policy selects its pipeline using earlier validation seasons. "
        "Intervals are empirical prior-error intervals; see the source experiment protocol.\n\n"
        "Open forecasts.csv or forecasts.parquet for all players and policies. "
        "No application rankings or production registries were changed.\n"
    )
    print(output / "forecasts.csv")


if __name__ == "__main__":
    main()
