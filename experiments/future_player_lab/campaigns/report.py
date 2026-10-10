"""Review sealed campaign evaluations on identical rows; never select or fit models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from .data import points


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(frame: pl.DataFrame, target: str, recipe: str = "ensemble") -> pl.DataFrame:
    keys = ["year", "player_id", "target"]
    group = frame.filter(pl.col("target") == target)
    model = group.filter(pl.col("recipe") == recipe)
    baseline = group.filter(pl.col("recipe") == "affine").select(
        *keys,
        pl.col("actual").alias("baseline_actual"),
        pl.col("prediction").alias("baseline_prediction"),
    )
    if model.select(keys).is_duplicated().any() or baseline.select(keys).is_duplicated().any():
        raise ValueError("Duplicate player/year/target evaluation rows")
    result = model.join(baseline, on=keys, how="inner", validate="1:1")
    if result.filter(
        pl.col("actual").is_finite()
        & pl.col("baseline_actual").is_finite()
        & ((pl.col("actual") - pl.col("baseline_actual")).abs() > 1e-8)
    ).height:
        raise ValueError("Model and baseline outcomes disagree")
    return result.filter(
        pl.all_horizontal(
            pl.col(c).is_finite() for c in ["actual", "prediction", "baseline_prediction"]
        )
    )


def errors(frame: pl.DataFrame) -> dict:
    if frame.is_empty():
        return {"n": 0}
    actual = frame["actual"].to_numpy()
    output = {"n": frame.height}
    for label, column in [("model", "prediction"), ("baseline", "baseline_prediction")]:
        residual = frame[column].to_numpy() - actual
        output[f"{label}_mse"] = float(np.mean(residual**2))
        output[f"{label}_rmse"] = float(np.sqrt(np.mean(residual**2)))
        output[f"{label}_mae"] = float(np.mean(np.abs(residual)))
    output["mse_reduction_pct"] = (
        100 * (1 - output["model_mse"] / output["baseline_mse"]) if output["baseline_mse"] else None
    )
    return output


def selection(frame: pl.DataFrame, top_k: int) -> dict:
    """Descriptive fantasy selection: equal slots and the same known candidate pool."""
    result = {"n": frame.height, "top_k": min(top_k, frame.height)}
    actual_rank = frame["actual"].rank(method="average").to_numpy()
    for label, column in [("model", "prediction"), ("baseline", "baseline_prediction")]:
        rank = frame[column].rank(method="average").to_numpy()
        correlation = (
            float(np.corrcoef(actual_rank, rank)[0, 1])
            if len(rank) > 1 and np.std(rank) and np.std(actual_rank)
            else None
        )
        chosen = frame.sort([column, "player_id"], descending=[True, False]).head(top_k)
        result[f"{label}_spearman"] = correlation
        result[f"{label}_selected_actual_points"] = float(chosen["actual"].sum())
    result["selected_points_difference"] = (
        result["model_selected_actual_points"] - result["baseline_selected_actual_points"]
    )
    return result


def prior_usage_cohorts(panel: pl.DataFrame, pairs: pl.DataFrame, position: str) -> list[dict]:
    """Use prior-known usage and thresholds learned from strictly earlier seasons."""
    volume = (
        pl.col("history_attempts") + pl.col("history_carries")
        if position == "QB"
        else pl.col("history_targets") + pl.col("history_carries")
    )
    panel = panel.with_columns((volume / pl.col("history_weeks")).alias("known_usage"))
    pieces, thresholds = [], []
    for (year,), group in pairs.partition_by("year", as_dict=True).items():
        past = panel.filter(
            (pl.col("year") < year)
            & pl.col("known_usage").is_finite()
            & (pl.col("known_usage") > 0)
        )["known_usage"].to_numpy()
        threshold = float(np.quantile(past, 0.75)) if len(past) else None
        current = panel.filter(pl.col("year") == year).select("year", "player_id", "known_usage")
        joined = group.join(current, on=["year", "player_id"], how="left", validate="1:1")
        if threshold is not None:
            pieces.append(joined.filter(pl.col("known_usage") >= threshold))
        thresholds.append({"year": year, "upper_quartile_cutoff": threshold})
    return [
        {
            "cohort": "high_prior_usage",
            "thresholds": thresholds,
            **errors(pl.concat(pieces) if pieces else pairs.head(0)),
        }
    ]


def coherence(frame: pl.DataFrame, position: str) -> dict:
    wide = frame.filter(pl.col("recipe") == "ensemble").pivot(
        on="target", index=["year", "player_id"], values="prediction"
    )
    relationships = (
        [
            ("completions", "attempts"),
            ("passing_tds", "completions"),
            ("passing_interceptions", "attempts"),
        ]
        if position == "QB"
        else [
            ("receptions", "targets"),
            ("receiving_tds", "receptions"),
            ("rushing_tds", "carries"),
        ]
    )
    checks = []
    for smaller, larger in relationships:
        gaps = wide.select((pl.col(smaller) - pl.col(larger)).alias("gap"))["gap"]
        checks.append(
            {
                "constraint": f"{smaller} <= {larger}",
                "violations": int((gaps > 1e-6).sum()),
                "max_excess": max(0.0, float(gaps.max())),
            }
        )
    wide = wide.with_columns(
        points(position, {c: pl.col(c) for c in wide.columns}).alias("component_points")
    )
    gap = (wide["points"] - wide["component_points"]).abs()
    return {
        "rows": wide.height,
        "constraints": checks,
        "direct_component_points_mean_absolute_difference": float(gap.mean()),
        "direct_component_points_max_absolute_difference": float(gap.max()),
    }


def build_report(run: Path) -> dict:
    cells = []
    for path in sorted(run.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if "model_id" not in manifest:
            continue
        for name in ["metrics.json", "evaluation_predictions.parquet"]:
            if digest(path.parent / name) != manifest["files"][name]:
                raise ValueError(f"Changed evaluation artifact: {path.parent / name}")
        frame = pl.read_parquet(path.parent / "evaluation_predictions.parquet")
        dataset_path = run / "inputs" / manifest["model_id"] / "dataset.json"
        if digest(dataset_path) != manifest["dataset_manifest_sha256"]:
            raise ValueError("Changed dataset manifest")
        dataset = json.loads(dataset_path.read_text())
        panel_path = dataset_path.parent / "panel.parquet"
        if digest(panel_path) != dataset["panel_sha256"]:
            raise ValueError("Changed historical feature panel")
        panel = pl.read_parquet(panel_path)
        comparisons, annual, cohorts = [], [], []
        for target in manifest["targets"]:
            pair = paired(frame, target)
            comparisons.append({"target": target, **errors(pair)})
            if target == "points":
                cohorts = prior_usage_cohorts(panel, pair, manifest["position"])
                for (year,), block in pair.partition_by("year", as_dict=True).items():
                    annual.append(
                        {
                            "year": year,
                            **errors(block),
                            **selection(block, 12 if manifest["position"] in {"QB", "TE"} else 24),
                        }
                    )
        cells.append(
            {
                "model_id": manifest["model_id"],
                "position": manifest["position"],
                "horizon": manifest["horizon"],
                "source_manifest": str(path),
                "source_manifest_sha256": digest(path),
                "comparisons": comparisons,
                "annual_points": annual,
                "points_cohorts": cohorts,
                "coherence": coherence(frame, manifest["position"]),
                "final_members": manifest["final_members"],
            }
        )
    if not cells:
        raise ValueError("No completed campaign cells")
    return {
        "schema_version": 1,
        "cells": cells,
        "method": "Earlier-selected ensemble versus affine persistence; identical finite rows.",
        "selection_note": (
            "Top 12 QB or TE / 24 RB or WR by forecast; "
            "realized points on the same candidate universe. "
            "Diagnostic only, not a selection objective."
        ),
        "limits": "Historical development evidence; these seasons are not a pristine holdout.",
    }


def formatted(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown(report: dict) -> str:
    lines = [
        "# Opening Drive evaluation",
        "",
        report["method"],
        "",
        report["limits"],
        "",
        "| Model | Rows | RMSE | Baseline RMSE | MSE reduction | MAE | Baseline MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        point = next(row for row in cell["comparisons"] if row["target"] == "points")
        if not point["n"]:
            continue
        lines.append(
            f"| {cell['model_id']} | {point['n']} | {point['model_rmse']:.2f} | "
            f"{point['baseline_rmse']:.2f} | {formatted(point['mse_reduction_pct'])}% | "
            f"{point['model_mae']:.2f} | {point['baseline_mae']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Positive MSE reduction means lower squared error. Components are assessed separately.",
            "",
            report["selection_note"],
            "",
        ]
    )
    for cell in report["cells"]:
        cohort = cell["points_cohorts"][0]
        if cohort["n"]:
            lines.extend(
                [
                    f"High prior usage, {cell['model_id']}: {cohort['n']} rows; "
                    f"points MSE reduction {formatted(cohort['mse_reduction_pct'])}%.",
                    "",
                ]
            )
    lines.extend(
        [
            "High prior usage means at least the earlier-seasons' upper quartile of positive "
            "attempts plus carries (QB), or targets plus carries (RB/WR/TE), per calendar week. "
            "The forecast year's future outcomes do not define this cohort.",
            "",
        ]
    )
    for cell in report["cells"]:
        lines.extend(
            [
                f"## {cell['model_id']}",
                "",
                "| Target | MSE reduction | Model RMSE | Baseline RMSE |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in cell["comparisons"]:
            if row["n"]:
                lines.append(
                    f"| {row['target']} | {formatted(row['mse_reduction_pct'])}% | "
                    f"{row['model_rmse']:.2f} | {row['baseline_rmse']:.2f} |"
                )
        lines.extend(
            [
                "",
                "| Year | MSE reduction | Rank correlation | Baseline rank | Slot points gain |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in cell["annual_points"]:
            lines.append(
                f"| {row['year']} | {formatted(row['mse_reduction_pct'])}% | "
                f"{formatted(row['model_spearman'], 3)} | "
                f"{formatted(row['baseline_spearman'], 3)} | "
                f"{row['selected_points_difference']:+.1f} |"
            )
        lines.append("")
        consistency = cell["coherence"]
        lines.append(
            f"Independent-stat consistency ({consistency['rows']} rows): "
            + "; ".join(
                f"{check['constraint']}: {check['violations']} violations"
                for check in consistency["constraints"]
            )
            + ". Predictions are retained without post-hoc constraint adjustment."
        )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.run.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "report.md").write_text(markdown(report))
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
