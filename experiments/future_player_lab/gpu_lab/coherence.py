"""Descriptive cross-stat consistency checks; never repairs or selects a forecast."""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import polars as pl

from .run import RUNS, digest, json_write

INDEX = ["horizon", "year", "origin", "player_id"]
REQUIRED = [
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "points",
]
RELATIONS = [
    ("receptions_exceed_targets", "receptions", "targets"),
    ("rushing_tds_exceed_carries", "rushing_tds", "carries"),
    ("receiving_tds_exceed_receptions", "receiving_tds", "receptions"),
]


def summarize(wide, groups):
    records = []
    for key, frame in wide.group_by(groups):
        identity = dict(zip(groups, key, strict=True))
        for check, smaller, larger in RELATIONS:
            low, high = frame[smaller].to_numpy(), frame[larger].to_numpy()
            known = np.isfinite(low) & np.isfinite(high)
            delta = low[known] - high[known]
            violation = delta > 1e-6
            records.append(
                {
                    **identity,
                    "check": check,
                    "n": int(known.sum()),
                    "violations": int(violation.sum()),
                    "violation_pct": float(100 * violation.mean()) if known.any() else None,
                    "mean_positive_excess": float(delta[violation].mean())
                    if violation.any()
                    else 0.0,
                }
            )
        stat_points = (
            0.1 * frame["rushing_yards"].to_numpy()
            + 0.1 * frame["receiving_yards"].to_numpy()
            + frame["receptions"].to_numpy()
            + 6 * frame["rushing_tds"].to_numpy()
            + 6 * frame["receiving_tds"].to_numpy()
        )
        direct = frame["points"].to_numpy()
        known = np.isfinite(stat_points) & np.isfinite(direct)
        delta = direct[known] - stat_points[known]
        records.append(
            {
                **identity,
                "check": "direct_points_minus_component_means",
                "n": int(known.sum()),
                "mean_difference": float(delta.mean()) if known.any() else None,
                "mean_absolute_difference": float(np.abs(delta).mean()) if known.any() else None,
            }
        )
    return pl.DataFrame(records, infer_schema_length=None).sort(*groups, "check")


def coherence(out):
    saved = json.loads((out / "analysis/manifest.json").read_text())
    if saved["status"] not in {"core_complete", "complete"}:
        raise ValueError("Completed analysis is required")
    prediction_path = out / "analysis/predictions.parquet"
    if digest(prediction_path) != saved["files"]["predictions.parquet"]:
        raise ValueError("Prediction artifact does not match the analysis manifest")
    config = json.loads((out / "config.json").read_text())
    rows = pl.read_parquet(prediction_path).filter(
        pl.col("year").is_in(config["evaluation_years"]) & pl.col("target").is_in(REQUIRED)
    )
    if set(rows["target"]) != set(REQUIRED):
        raise ValueError("Consistency diagnostics require all seven stats and direct points")
    wide = rows.select(*INDEX, "recipe", "target", "prediction").pivot(
        on="target", index=[*INDEX, "recipe"], values="prediction"
    )
    predictions = summarize(wide, ["horizon", "recipe"])
    observed = (
        rows.select(*INDEX, "target", "actual")
        .unique()
        .pivot(on="target", index=INDEX, values="actual")
    )
    actual = summarize(observed, ["horizon"])
    destination = out / "stat_consistency"
    destination.mkdir(exist_ok=True)
    predictions.write_parquet(destination / "forecast_consistency.parquet")
    predictions.write_csv(destination / "forecast_consistency.csv")
    actual.write_parquet(destination / "observed_consistency.parquet")
    shutil.copyfile(__file__, destination / "source.py")
    chosen = predictions.filter(
        pl.col("recipe").is_in(
            ["selected", "ensemble_top3", "ensemble_diverse", "ensemble_convex", "persistence"]
        )
    )
    lines = [
        "# Cross-stat consistency",
        "",
        "Descriptive checks on the saved evaluation forecasts; "
        "no projection, repair, or new selection.",
        "Independent target models can violate relationships among expected counts. Receiving TDs "
        "should not exceed receptions, receptions should not exceed targets, and rushing TDs "
        "should not exceed carries under this outcome contract. Provider annotations can also "
        "contain inconsistencies; observed labels are checked separately.",
        "",
        "Negative rushing/receiving yards and negative scoring totals are possible and are not "
        "counted as consistency violations. Direct PPR minus component means measures disagreement "
        "between separately fitted point models, not a comparison of marginal medians or a "
        "constructed joint predictive distribution.",
        "",
        "| Horizon | Policy | Check | Known rows | Violations | Violation % | "
        "Mean absolute points gap |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in chosen.iter_rows(named=True):

        def value(name, row=row):
            v = row.get(name)
            return "—" if v is None else f"{v:.2f}"

        lines.append(
            f"| {row['horizon']} | {row['recipe']} | {row['check']} | {row['n']} | "
            f"{value('violations')} | {value('violation_pct')} | "
            f"{value('mean_absolute_difference')} |"
        )
    lines += [
        "",
        "All recipes: [CSV](forecast_consistency.csv), [Parquet](forecast_consistency.parquet).",
        "Observed-label checks: [Parquet](observed_consistency.parquet).",
    ]
    (destination / "readout.md").write_text("\n".join(lines) + "\n")
    json_write(
        destination / "manifest.json",
        {
            "status": "complete",
            "predictions_sha256": digest(prediction_path),
            "configuration_sha256": digest(out / "config.json"),
            "files": {
                p.name: digest(p)
                for p in sorted(destination.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    print(f"Consistency checks saved: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    coherence(RUNS / args.run_id)
