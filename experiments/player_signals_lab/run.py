#!/usr/bin/env python3
# ruff: noqa: E402
# Disable bytecode and set the read-only source path before importing dependencies.
"""Run disposable experiments without touching any Patron publication path."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
import platform
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path.insert(0, str(REPO / "src"))
# Prevent joblib's optional hardware probe from launching a subprocess on macOS.
# Models are already deliberately single-threaded inside threadpool_limits.
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import polars as pl
from data import FAMILIES, make_panel
from evaluation import calibrated_intervals, interval_summary, summarize
from methods import add_rates, fit_pool, predict_regression, recipes, role_mixture
from report import html_report, markdown_report
from safety import install_write_guard, new_run, sha256, snapshot, verify_source_pins, write_json

from patron.data.releases import load_gold
from patron.metrics.nextgen import fit_fold


def protected_paths(gold, config):
    paths = [gold.root / "manifest.json", REPO / "pyproject.toml"]
    paths += [
        gold.path(name) for name in ("preseason_features", "season_outcomes", "nfl_player_weeks")
    ]
    paths += [REPO / name for name in config["source_sha256"]]
    paths += [p for p in LAB.iterdir() if p.is_file() and p.suffix in (".py", ".md", ".json")]
    paths += list((LAB / "tests").glob("*.py"))
    return paths


def run(config, destination):
    verify_source_pins(REPO, config["source_sha256"])
    gold = load_gold(REPO / "data", version=config["gold_version"])
    if gold.ref["manifest_sha256"] != config["gold_manifest_sha256"]:
        raise ValueError("Pinned gold manifest hash mismatch")
    paths = protected_paths(gold, config)
    before = snapshot(paths, REPO)
    monitored = [REPO / "data/current.json"] + list((REPO / "src").rglob("*.py"))
    monitored += list((REPO / "data/research").glob("*/registry.json"))
    monitored += list((REPO / "data/research").glob("*/manifest.json"))
    workspace_before = snapshot(monitored, REPO)
    manifest = {
        "run_id": destination.name,
        "status": "running",
        "research_only": True,
        "started_at": datetime.now(UTC).isoformat(),
        "gold": gold.ref,
        "evaluation_start": config["evaluation_start"],
        "evaluation_end": config["evaluation_end"],
        "tasks": config["tasks"],
        "protocol_version": config["protocol_version"],
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "polars", "scipy", "scikit-learn")},
        "protected_before": before,
        "workspace_before": workspace_before,
        "isolation": "create-only run; Python write/network/process guard; post-run hashes",
    }
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "config.json", config)
    for path in paths:
        if path.suffix in (".py", ".md"):
            copy = destination / "source" / path.relative_to(REPO)
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(path.read_bytes())
    write_json(
        destination / "catalog_at_start.json", json.loads((REPO / "data/current.json").read_text())
    )
    try:
        features = gold.read("preseason_features")
        outcomes = gold.read("season_outcomes")
        weeks = gold.read("nfl_player_weeks")
        panel = make_panel(features, outcomes, weeks)
        # No incomplete current-season labels enter training, evaluation, or calibration.
        complete = [
            r
            for r in panel
            if r["outcome_complete"] and r["forecast_season"] >= config["training_start"]
        ]
        if config["evaluation_end"] > max(r["forecast_season"] for r in complete):
            raise ValueError("Evaluation cannot include an incomplete season")
        base = sorted(c for c in panel[0] if c.startswith("x_"))
        predictions, folds = [], []
        for position, target in config["tasks"]:
            candidates = [r for r in complete if r["position"] == position]
            observed = [r for r in candidates if r[f"y_{target}"] is not None]
            history = defaultdict(list)
            specs = recipes(position, base)
            for year in range(config["evaluation_start"], config["evaluation_end"] + 1):
                train = [r for r in observed if r["forecast_season"] < year]
                test = [r for r in observed if r["forecast_season"] == year]
                if not test or len({r["forecast_season"] for r in train}) < 3:
                    continue
                assert max(r["forecast_season"] for r in train) < year
                pools = {f: fit_pool(train, f) for f in FAMILIES[position]}
                training = add_rates(train, pools, position)
                testing = add_rates(test, pools, position)
                original, _ = fit_fold(train, test, target, base)
                forecasts = {"reference": original["baseline"], "nextgen_boost": original["boost"]}
                controls = {"reference": "reference", "nextgen_boost": "reference"}
                probabilities = {}
                for model, (columns, control) in specs.items():
                    if model.endswith("role_mixture"):
                        forecasts[model], probabilities[model] = role_mixture(
                            training, testing, target, columns, config["booster"]
                        )
                    else:
                        forecasts[model] = predict_regression(
                            training, testing, target, columns, config["booster"]
                        )
                    controls[model] = control
                # Same scale rule for each model; anchor comes only from its training labels.
                positive = [r[f"y_{target}"] for r in train if r[f"y_{target}"] > 0]
                anchor = max(1.0, float(np.median(positive)) if positive else 1.0)
                fold_rows = []
                for model, values in forecasts.items():
                    for i, (row, value) in enumerate(zip(test, values, strict=True)):
                        if not np.isfinite(value):
                            raise ValueError("Non-finite forecast in an evaluated fold")
                        prior_family = FAMILIES[position][0]
                        item = {
                            "player_id": row["player_id"],
                            "player_name": row["player_display_name"],
                            "season": year,
                            "cutoff": row["forecast_cutoff_date"],
                            "position": position,
                            "target": target,
                            "model": model,
                            "control": controls[model],
                            "population": row["player_population"],
                            "actual": row[f"y_{target}"],
                            "prediction": float(value),
                            "control_prediction": float(forecasts[controls[model]][i]),
                            "market_observed": bool(row["m_observed"]),
                            "proxy_observed": row["proxy_observed"],
                            "proxy_half_season": row["proxy_half_season"],
                            "prior_exposure": row[f"exposure_{prior_family}"],
                            "prior_role_games": row["r_prior_role_games"],
                            "known_available_games_cap": row["known_available_games_cap"],
                            "actual_role_state": row["y_role_state"] if position == "QB" else None,
                            "role_probability_none": None,
                            "role_probability_short": None,
                            "role_probability_sustained": None,
                            "scale": float(np.sqrt(max(float(value), anchor))),
                        }
                        if model in probabilities:
                            for k, name in enumerate(("none", "short", "sustained")):
                                item[f"role_probability_{name}"] = float(probabilities[model][i, k])
                        item.update(calibrated_intervals(history[model], item, config))
                        fold_rows.append(item)
                # Add the entire fold only after all predictions/intervals are fixed.
                # Same-season player outcomes can never calibrate another player.
                for item in fold_rows:
                    history[item["model"]].append(item)
                predictions.extend(fold_rows)
                folds.append(
                    {
                        "position": position,
                        "target": target,
                        "test_season": year,
                        "train_min": min(r["forecast_season"] for r in train),
                        "train_max": max(r["forecast_season"] for r in train),
                        "train_n": len(train),
                        "test_n": len(test),
                        "candidates": sum(r["forecast_season"] == year for r in candidates),
                        "missing_target": sum(r["forecast_season"] == year for r in candidates)
                        - len(test),
                        "market_covered": sum(bool(r["m_observed"]) for r in test),
                        "proxy_covered": sum(r["proxy_observed"] for r in test),
                        "pools": pools,
                        "scale_anchor": anchor,
                        "features": {"nextgen_boost": base, **{k: v[0] for k, v in specs.items()}},
                        "role_train_counts": {
                            str(s): sum(r["y_role_state"] == s for r in train) for s in range(3)
                        }
                        if position == "QB"
                        else None,
                    }
                )
                print(
                    f"{position} {target} {year}: train={len(train)} test={len(test)} "
                    f"models={len(forecasts)}",
                    flush=True,
                )
        if not predictions:
            raise ValueError("No eligible evaluation folds")
        pl.DataFrame(predictions, infer_schema_length=None).write_parquet(
            destination / "predictions.parquet"
        )
        write_json(destination / "folds.json", folds)
        summaries = summarize(predictions, config)
        intervals = interval_summary(predictions, config)
        write_json(destination / "summaries.json", summaries)
        write_json(destination / "intervals.json", intervals)
        (destination / "report.md").write_text(markdown_report(summaries, intervals, manifest))
        (destination / "index.html").write_text(html_report(summaries, manifest))
        after = snapshot(paths, REPO)
        manifest["protected_inputs_unchanged"] = before == after
        manifest["changed_protected_paths"] = [p for p in before if before[p] != after.get(p)]
        if before != after:
            raise RuntimeError("Protected inputs changed during the run; results are invalid")
        # An explicit gold version does not read the current catalog or product registry.
        # Concurrent app publishing is recorded but cannot select this experiment's data.
        workspace_after = snapshot(monitored, REPO, missing_ok=True)
        manifest["concurrent_workspace_changes"] = [
            p for p in workspace_before if workspace_before[p] != workspace_after.get(p)
        ]
        manifest["workspace_after"] = workspace_after
        write_json(
            destination / "catalog_at_end.json",
            json.loads((REPO / "data/current.json").read_text()),
        )
        manifest.update(
            status="complete",
            completed_at=datetime.now(UTC).isoformat(),
            prediction_rows=len(predictions),
            fold_count=len(folds),
            comparison_count=len(summaries),
            calibrated_comparisons=len(intervals),
            outputs={
                p.name: sha256(p)
                for p in destination.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        )
        write_json(destination / "manifest.json", manifest)
        print(f"Complete: {destination / 'report.md'}", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        after = snapshot(paths, REPO)
        manifest["protected_inputs_unchanged"] = before == after
        write_json(destination / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("trial_%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--config", type=Path, default=LAB / "config.json")
    parser.add_argument(
        "--smoke", action="store_true", help="Two tasks, three recent folds; plumbing only"
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.is_relative_to(LAB):
        raise ValueError("Keep experimental configurations inside the lab")
    config = json.loads(config_path.read_text())
    if args.smoke:
        config.update(
            tasks=[["QB", "attempts"], ["WR", "targets"]],
            evaluation_start=2023,
            evaluation_end=2025,
            bootstrap_draws=200,
        )
    if config["evaluation_start"] < config["training_start"] + 3:
        raise ValueError("Require at least three earlier training seasons")
    destination = new_run(LAB, args.run_id)
    install_write_guard(destination)
    try:
        run(config, destination)
    except BaseException as error:
        if not (destination / "manifest.json").exists():
            write_json(
                destination / "manifest.json",
                {
                    "run_id": destination.name,
                    "status": "failed",
                    "phase": "initialization",
                    "research_only": True,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise


if __name__ == "__main__":
    main()
