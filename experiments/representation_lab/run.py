#!/usr/bin/env python3
"""Create-only, offline representation search. No Armchair Labs publication integration."""

# ruff: noqa: E402
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
import platform
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path.insert(0, str(REPO / "experiments/player_signals_lab"))
from safety import install_write_guard, new_run, sha256, snapshot, write_json

sys.path.pop(0)

import numpy as np
import polars as pl
from data import inventory_columns, numeric, raw_history
from methods import FoldModel, choose, point_predictions, recipes
from report import build_report
from threadpoolctl import threadpool_limits


def load_inputs(config):
    for path, expected in config["input_sha256"].items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"Pinned input mismatch: {path}")
    source = REPO / config["inventory_root"]
    registry = json.loads((source / "registry.json").read_text())
    source_manifest = json.loads((source / "manifest.json").read_text())
    for filename in ("features.parquet", "registry.json", "specs.json"):
        if sha256(source / filename) != source_manifest["files"][filename]:
            raise ValueError(f"Artifact manifest mismatch: {filename}")
    frame = pl.read_parquet(source / "features.parquet").sort(
        "position", "forecast_season", "player_id"
    )
    keys = ["player_id", "forecast_season", "forecast_cutoff_date"]
    if frame.select(keys).is_duplicated().any():
        raise ValueError("Duplicate forecast key")
    if not frame["outcome_complete"].all():
        raise ValueError("Incomplete outcomes in input matrix")
    if not frame["actual_season_points"].is_finite().all():
        raise ValueError("Nonfinite outcomes")
    if frame["forecast_season"].max() >= 2026:
        raise ValueError("2026 is not a completed outcome")
    cutoffs = frame["forecast_cutoff_date"].str.to_date()
    if not (cutoffs.dt.year() == frame["forecast_season"]).all():
        raise ValueError("Forecast cutoff year mismatch")
    for column in ("availability_latest_known_on", "cutoff_last_transaction_date"):
        if column in frame:
            dates = frame[column].cast(pl.String).str.slice(0, 10).str.to_date(strict=False)
            if (dates > cutoffs).fill_null(False).any():
                raise ValueError(f"Post-cutoff evidence in {column}")
    gold = REPO / config["gold_root"]
    gold_manifest = json.loads((gold / "manifest.json").read_text())
    if sha256(gold / "manifest.json") != source_manifest["gold"]["manifest_sha256"]:
        raise ValueError("Gold lineage mismatch")
    spec = gold_manifest["tables"]["nfl_player_weeks"]
    if sha256(gold / spec["path"]) != spec["sha256"]:
        raise ValueError("Weekly table checksum mismatch")
    weeks = pl.read_parquet(gold / spec["path"])
    raw, raw_names, raw_groups = raw_history(frame, weeks)
    inv_names = inventory_columns(registry)
    market_names = inventory_columns(registry, market=True)
    basic_names = json.loads((source / "specs.json").read_text())["basic"]
    groups = {r["stat"]: r["family"] for r in registry}
    recent = [
        i
        for i, c in enumerate(raw_names)
        if not c.startswith(("annual_", "week_lag2_", "week_lag3_"))
    ]
    matrices = {
        "raw": raw,
        "recent": raw[:, recent],
        "inventory": numeric(frame, inv_names),
        "market": numeric(frame, market_names),
        "basic": numeric(frame, basic_names),
    }
    names = {
        "raw": raw_names,
        "recent": [raw_names[i] for i in recent],
        "inventory": inv_names,
        "market": market_names,
        "basic": basic_names,
    }
    families = {"raw": raw_groups, "inventory": [groups[c] for c in inv_names]}
    admitted = [r for r in registry if r["status"] == "admitted_research_only"]
    inventory = {
        "admitted_count": len(admitted),
        "binary_count_descriptive_only": sum(
            frame[r["stat"]].drop_nulls().n_unique() == 2 for r in admitted
        ),
        "columns": names,
        "groups": families,
        "market_count": len(market_names) - len(inv_names),
        "rows": frame.height,
        "input_policy": (
            "Registry admissibility only; no screen, alias, global coverage or score filters"
        ),
    }
    return frame, matrices, names, families, inventory


def run(config, destination):
    protected = [REPO / p for p in config["input_sha256"]]
    protected += [
        p
        for p in LAB.rglob("*")
        if p.is_file()
        and "runs" not in p.relative_to(LAB).parts
        and p.suffix in (".py", ".json", ".md")
    ]
    before = snapshot(protected, REPO)
    publication = [
        REPO / "data/current.json",
        REPO / "data/static/experimental_2026_predictions.csv",
    ]
    publication_before = snapshot(publication, REPO)
    manifest = {
        "status": "running",
        "research_only": True,
        "started_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "polars", "scikit-learn")},
        "protected_before": before,
        "publication_before": publication_before,
        "config": config,
        "recipes": [asdict(r) for r in recipes()],
    }
    write_json(destination / "manifest.json", manifest)
    for p in protected:
        if p.suffix in (".py", ".md", ".json"):
            copy = destination / "source" / p.relative_to(REPO)
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(p.read_bytes())
    try:
        frame, matrices, names, families, inventory = load_inputs(config)
        write_json(destination / "inventory.json", inventory)
        # Cache fixed-recipe chronological predictions once. These are exactly the
        # inner validation fits needed by subsequent outer seasons, not fits on
        # the outer training matrix followed by retrospective validation.
        predictions, folds, permutations = [], [], []
        diagnostics = []
        specs = recipes()
        selectors = {
            "raw_search": [r.name for r in specs if r.inputs in ("raw", "recent")],
            "inventory_search": [
                r.name for r in specs if r.inputs == "inventory" and r.representation != "legacy"
            ],
            "joint_search": [
                r.name
                for r in specs
                if r.inputs in ("raw", "recent", "inventory") and r.representation != "legacy"
            ],
        }
        for position in config["positions"]:
            ids = np.flatnonzero(frame["position"].to_numpy() == position)
            local = frame[ids.tolist()]
            years = local["forecast_season"].to_numpy()
            points = local["actual_season_points"].to_numpy()
            length = np.where(years >= 2021, 17, 16)
            y = points / length
            blocks = {k: v[ids] for k, v in matrices.items()}
            history = defaultdict(dict)
            for year in range(config["warmup_start"], config["evaluation_end"] + 1):
                train, test = np.flatnonzero(years < year), np.flatnonzero(years == year)
                if not len(test) or not len(train):
                    continue
                selected = {}
                if year >= config["evaluation_start"]:
                    for name, eligible in selectors.items():
                        selected[name] = choose(history, eligible, year, config["inner_seasons"])
                forecasts, fold_diagnostics = {}, []
                for recipe in specs:
                    x = blocks[recipe.inputs]
                    model = FoldModel(recipe, config["seed"]).fit(x[train], y[train])
                    pred = point_predictions(model, x[test], length[test])
                    if not np.isfinite(pred).all():
                        raise ValueError(f"Nonfinite prediction: {position} {year} {recipe.name}")
                    forecasts[recipe.name] = pred
                    fold_diagnostics.append(
                        {
                            "position": position,
                            "season": year,
                            "model": recipe.name,
                            **model.diagnostics(names[recipe.inputs]),
                        }
                    )
                    # Descriptive held-out permutation only. It never changes fitting,
                    # chosen recipes or later search eligibility.
                    if year >= config["evaluation_start"] and recipe.name in (
                        "raw_tree15",
                        "inventory_tree15",
                    ):
                        group_names = families[recipe.inputs]
                        base_loss = float(np.mean((points[test] - pred) ** 2))
                        rng = np.random.default_rng(config["seed"] + year)
                        order = rng.permutation(len(test))
                        for group in sorted(set(group_names)):
                            cols = [j for j, g in enumerate(group_names) if g == group]
                            shuffled = x[test].copy()
                            shuffled[:, cols] = shuffled[order][:, cols]
                            loss = np.mean(
                                (points[test] - point_predictions(model, shuffled, length[test]))
                                ** 2
                            )
                            permutations.append(
                                {
                                    "position": position,
                                    "season": year,
                                    "model": recipe.name,
                                    "group": group,
                                    "mse_increase": float(loss - base_loss),
                                }
                            )
                # Selection is fixed BEFORE current labels are scored.
                for name, (winner, inner_years, losses) in selected.items():
                    forecasts[name] = forecasts[winner].copy()
                    folds.append(
                        {
                            "position": position,
                            "season": year,
                            "search": name,
                            "chosen_model": winner,
                            "inner_years": inner_years,
                            "validation_mse": losses,
                            "train_min": int(years[train].min()),
                            "train_max": int(years[train].max()),
                            "train_n": len(train),
                            "test_n": len(test),
                        }
                    )
                for recipe in specs:
                    history[recipe.name][year] = float(
                        np.mean((points[test] - forecasts[recipe.name]) ** 2)
                    )
                if year >= config["evaluation_start"]:
                    for model_name, pred in forecasts.items():
                        for row, value in zip(
                            local[test.tolist()].iter_rows(named=True), pred, strict=True
                        ):
                            predictions.append(
                                {
                                    "player_id": row["player_id"],
                                    "position": position,
                                    "season": year,
                                    "forecast_cutoff_date": row["forecast_cutoff_date"],
                                    "population": row["player_population"],
                                    "model": model_name,
                                    "actual": row["actual_season_points"],
                                    "prediction": float(value),
                                }
                            )
                diagnostics.extend(fold_diagnostics)
                # Checkpoint every fold into this run only, allowing failed trials to be audited.
                write_json(
                    destination / "progress.json",
                    {
                        "position": position,
                        "season": year,
                        "completed_position_seasons": len(diagnostics) // len(specs),
                    },
                )
                print(
                    f"{position} {year}: train={len(train)} test={len(test)} recipes={len(specs)}",
                    flush=True,
                )
            pl.DataFrame([r for r in predictions if r["position"] == position]).write_parquet(
                destination / f"predictions_{position}.parquet"
            )
            write_json(
                destination / f"diagnostics_{position}.json",
                [r for r in diagnostics if r["position"] == position],
            )
            write_json(destination / "folds.json", folds)
            write_json(destination / "permutations.json", permutations)
        pl.DataFrame(predictions).write_parquet(destination / "predictions.parquet")
        build_report(destination, predictions, folds, permutations, diagnostics, config, inventory)
        after = snapshot(protected, REPO)
        manifest["protected_inputs_unchanged"] = before == after
        manifest["publication_unchanged"] = publication_before == snapshot(publication, REPO)
        if before != after or not manifest["publication_unchanged"]:
            raise RuntimeError("Protected source/input/publication changed during run")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest["files"] = {
            str(p.relative_to(destination)): sha256(p)
            for p in destination.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(destination / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=LAB / "config.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--positions", nargs="+", choices=("QB", "RB", "WR", "TE"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.positions:
        config["positions"] = args.positions
    if args.smoke:
        config.update(positions=["QB"], evaluation_start=2008, evaluation_end=2008)
    destination = new_run(LAB, args.run_id)
    # Initialize thread libraries before enabling the process/write guard.
    with threadpool_limits(limits=1):
        install_write_guard(destination)
        run(config, destination)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
