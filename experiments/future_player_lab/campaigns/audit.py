"""Independently verify sealed campaign forecasts, chronological choices, and metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

KEY = ["year", "origin", "player_id"]


def _ranked(annual, recipes, target, forecast_year, count):
    losses = []
    for recipe in recipes:
        rows = [
            r
            for r in annual
            if r["target"] == target
            and r["recipe"] == recipe
            and forecast_year - 3 <= r["year"] < forecast_year
        ]
        if len(rows) != 3 or {r["year"] for r in rows} != set(
            range(forecast_year - 3, forecast_year)
        ):
            raise ValueError("Incomplete or duplicate earlier selection history")
        values = np.asarray([r["mse"] for r in rows], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid earlier selection losses")
        losses.append((float(values.mean()), recipe))
    return [recipe for _, recipe in sorted(losses)[:count]]


def _error(actual, predicted):
    known = np.isfinite(actual) & np.isfinite(predicted)
    delta = predicted[known] - actual[known]
    if not len(delta):
        return dict(n=0, mse=None, rmse=None, mae=None)
    mse = float(np.mean(delta * delta))
    return dict(n=len(delta), mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.mean(abs(delta))))


def _equal_metrics(expected, actual):
    largest_difference = 0.0
    for metric in ("n", "mse", "rmse", "mae"):
        if expected[metric] is None or actual[metric] is None:
            if expected[metric] is not actual[metric]:
                raise ValueError(f"Unknown metric mismatch: {metric}")
        elif not np.isclose(expected[metric], actual[metric], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Recomputed metric differs: {metric}")
        else:
            largest_difference = max(largest_difference, abs(expected[metric] - actual[metric]))
    return float(largest_difference)


def audit_cell(directory: Path) -> dict:
    """Read only: no fitting, no weight deserialization, no rewriting sealed artifacts."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    files = manifest["files"]
    if not {"models.joblib", "metrics.json", "evaluation_predictions.parquet"}.issubset(files):
        raise ValueError("Sealed model manifest omits required files")
    for name, expected in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Sealed artifact escapes model directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    source_contract_verified = False
    dataset_manifest_verified = False
    input_panel = None
    if "fit_contract.json" in files:
        contract = json.loads((directory / "fit_contract.json").read_text())
        if contract["sources"] != manifest["source_files"]:
            raise ValueError("Fit-time sources differ from sealed model sources")
        if contract["recipes"] != manifest["recipes"]:
            raise ValueError("Fit-time recipes differ from sealed model recipes")
        if contract["dataset_sha256"] != manifest["dataset_manifest_sha256"]:
            raise ValueError("Fit-time dataset differs from sealed model dataset")
        source_contract_verified = True
        dataset = directory.parent / "inputs" / manifest["model_id"] / "dataset.json"
        if dataset.exists():
            if hashlib.sha256(dataset.read_bytes()).hexdigest() != contract["dataset_sha256"]:
                raise ValueError("Dataset manifest differs from fit-time contract")
            dataset_manifest_verified = True
            dataset_doc = json.loads(dataset.read_text())
            panel_path = dataset.parent / "panel.parquet"
            if hashlib.sha256(panel_path.read_bytes()).hexdigest() != dataset_doc["panel_sha256"]:
                raise ValueError("Frozen input panel checksum mismatch")
            input_panel = pl.read_parquet(panel_path)
    report = json.loads((directory / "metrics.json").read_text())
    rows = pl.read_parquet(directory / "evaluation_predictions.parquet")
    keys = ["target", "recipe", *KEY]
    if rows.select(keys).is_duplicated().any():
        raise ValueError("Duplicate evaluation forecast identity")
    if set(rows["year"]) != set(report["development_evaluation_years"]):
        raise ValueError("Evaluation years differ from declared assessment")
    recipes = list(manifest["recipes"])
    expected_recipes = {*recipes, "affine", "persistence", "selected", "ensemble"}
    if set(rows["target"]) != set(manifest["targets"]):
        raise ValueError("Evaluation target set differs from model contract")
    max_blend_error = 0.0
    matched_baseline_rows = 0
    for target in manifest["targets"]:
        group = rows.filter(pl.col("target") == target)
        if set(group["recipe"]) != expected_recipes:
            raise ValueError("Evaluation recipe set differs from model contract")
        reference = group.filter(pl.col("recipe") == "ensemble").sort(KEY)
        for recipe in expected_recipes:
            candidate = group.filter(pl.col("recipe") == recipe).sort(KEY)
            if not candidate.select(KEY).equals(reference.select(KEY)):
                raise ValueError("Models were evaluated on different player cohorts")
            if not np.allclose(
                candidate["actual"].to_numpy(), reference["actual"].to_numpy(), equal_nan=True
            ):
                raise ValueError("Models were evaluated against different outcomes")
            if (
                recipe not in {"affine", "persistence"}
                and not candidate["prediction"].is_finite().all()
            ):
                raise ValueError("Tree predictions contain missing or nonfinite values")
            if recipe == "affine":
                matched_baseline_rows += int(
                    (candidate["actual"].is_finite() & candidate["prediction"].is_finite()).sum()
                )
        choices = [r for r in report["choices"] if r["target"] == target]
        if len(choices) != len(report["development_evaluation_years"]) or {
            r["year"] for r in choices
        } != set(report["development_evaluation_years"]):
            raise ValueError("Selection choices omit or duplicate an evaluation year")
        for choice in choices:
            year = choice["year"]
            members = _ranked(
                report["candidate_annual"], recipes, target, year, len(choice["members"])
            )
            if (
                members != choice["members"]
                or choice["selection_years"] != list(range(year - 3, year))
                or choice["training_max_year"] != year - 1
            ):
                raise ValueError("Saved selection disagrees with preceding-year evidence")
            fold = group.filter(pl.col("year") == year)
            components = [
                fold.filter(pl.col("recipe") == recipe).sort(KEY)["prediction"].to_numpy()
                for recipe in members
            ]
            actual_blend = (
                fold.filter(pl.col("recipe") == "ensemble").sort(KEY)["prediction"].to_numpy()
            )
            difference = float(np.max(abs(actual_blend - np.mean(components, axis=0))))
            max_blend_error = max(max_blend_error, difference)
            if difference > 1e-8:
                raise ValueError("Saved ensemble is not the mean of selected earlier-year members")
            selected = (
                fold.filter(pl.col("recipe") == "selected").sort(KEY)["prediction"].to_numpy()
            )
            if not np.allclose(selected, components[0], rtol=0, atol=1e-8):
                raise ValueError("Saved single-model selection disagrees with its member")
        final = manifest["final_members"][target]
        if final != _ranked(
            report["candidate_annual"],
            recipes,
            target,
            manifest["training_through_season"] + 1,
            len(final),
        ):
            raise ValueError("Final refit members disagree with chronological selection")
    metric_count = 0
    checkpoint_count = 0
    max_metric_difference = 0.0
    max_checkpoint_difference = 0.0
    seen = set()
    for record in report["rows"]:
        identity = tuple(record[c] for c in ("target", "recipe", "cohort", "year"))
        if identity in seen:
            raise ValueError("Duplicate reported metric group")
        seen.add(identity)
        group = rows.filter(
            (pl.col("target") == record["target"]) & (pl.col("recipe") == record["recipe"])
        )
        if record["year"] is not None:
            group = group.filter(pl.col("year") == record["year"])
        if record["cohort"] in {"established_usage", "established_appearances"}:
            group = group.filter(pl.col("history_games") / pl.col("history_weeks") >= 0.5)
        elif record["cohort"] != "all_candidates":
            raise ValueError("Unrecognized cutoff-defined cohort")
        max_metric_difference = max(
            max_metric_difference,
            _equal_metrics(
                record, _error(group["actual"].to_numpy(), group["prediction"].to_numpy())
            ),
        )
        metric_count += 1
    for record in report["candidate_annual"]:
        if record["year"] in report["development_evaluation_years"]:
            group = rows.filter(
                (pl.col("target") == record["target"])
                & (pl.col("recipe") == record["recipe"])
                & (pl.col("year") == record["year"])
            )
            max_metric_difference = max(
                max_metric_difference,
                _equal_metrics(
                    record, _error(group["actual"].to_numpy(), group["prediction"].to_numpy())
                ),
            )
            metric_count += 1
        if input_panel is not None:
            checkpoint = (
                directory
                / "checkpoints"
                / (f"{record['target']}_{record['year']}_{record['recipe']}.npy")
            )
            receipt = json.loads(checkpoint.with_suffix(".json").read_text())
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != receipt["sha256"]:
                raise ValueError("Candidate checkpoint checksum mismatch")
            predicted = np.load(checkpoint, allow_pickle=False)
            actual = input_panel.filter(pl.col("year") == record["year"])[
                f"y_{record['target']}"
            ].to_numpy()
            if predicted.shape != actual.shape or not np.isfinite(predicted).all():
                raise ValueError("Candidate checkpoint predictions are invalid")
            max_metric_difference = max(
                max_metric_difference, _equal_metrics(record, _error(actual, predicted))
            )
            checkpoint_count += 1
            if record["year"] in report["development_evaluation_years"]:
                original = (
                    input_panel.filter(pl.col("year") == record["year"])
                    .select(*KEY, pl.col(f"y_{record['target']}").alias("actual"))
                    .with_columns(pl.Series("prediction", predicted))
                    .sort(KEY)
                )
                reported = group.sort(KEY)
                if not original.select(KEY).equals(reported.select(KEY)):
                    raise ValueError("Evaluation cohort differs from frozen input rows")
                if not np.allclose(
                    original["actual"].to_numpy(),
                    reported["actual"].to_numpy(),
                    rtol=0,
                    atol=1e-10,
                    equal_nan=True,
                ):
                    raise ValueError("Reported evaluation labels differ from frozen inputs")
                difference = float(
                    np.max(
                        abs(original["prediction"].to_numpy() - reported["prediction"].to_numpy())
                    )
                )
                if difference > 1e-8:
                    raise ValueError("Reported candidate forecasts differ from checkpoint rows")
                max_checkpoint_difference = max(max_checkpoint_difference, difference)
    return {
        "model_id": manifest["model_id"],
        "status": "verified",
        "verified_files": len(files),
        "evaluation_prediction_rows": rows.height,
        "recomputed_metric_groups": metric_count,
        "matched_affine_rows": matched_baseline_rows,
        "max_ensemble_difference": max_blend_error,
        "fit_source_contract_verified": source_contract_verified,
        "dataset_manifest_verified": dataset_manifest_verified,
        "recomputed_candidate_checkpoints": checkpoint_count,
        "max_metric_difference": max_metric_difference,
        "max_checkpoint_prediction_difference": max_checkpoint_difference,
        "limitations": (
            "No models are independently refitted. Where frozen inputs/checkpoints remain, their "
            "annual errors also reproduce the warmup selection losses. This is an integrity "
            "audit, not new forecast evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"schema_version": 1, "cells": [audit_cell(p) for p in args.directories]}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        if any(args.output.resolve().is_relative_to(p.resolve()) for p in args.directories):
            raise ValueError("Write audit results outside sealed model directories")
        with args.output.open("x") as output:
            output.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
