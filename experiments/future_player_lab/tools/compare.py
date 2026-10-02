"""Matched-row comparisons against both persistence and the fixed tree anchor.

Independent analysis: consumes completed task forecasts and writes a new run directory.
Usage: python -m experiments.future_player_lab.tools.compare --source exploration_001
       --run-id comparison_001
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from ..data import TARGETS, digest
from ..evaluate import matched_scores as compare
from ..evaluate import paired_interval

LAB = Path(__file__).resolve().parents[1]
POLICIES = ("selected_mse", "selected_rank", "selected_ensemble")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    for value in (args.source, args.run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
            raise ValueError("Unsafe run identifier")
    source, output = LAB / "runs" / args.source, LAB / "runs" / args.run_id
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] not in ("complete", "complete_with_failures"):
        raise ValueError("Source run must finish before analysis")
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((source / "config.json").read_text())
    summaries, annual, inputs, choices, convergence = [], [], {}, Counter(), Counter()
    for task in sorted((source / "tasks").iterdir()):
        if not task.is_dir():
            continue
        path = task / "predictions.parquet"
        inputs[str(path.relative_to(LAB))] = digest(path)
        frame = pl.read_parquet(path)
        for choice in json.loads((task / "choices.json").read_text()):
            if choice["policy"] == "selected_mse":
                choices.update(choice["chosen"])
        for diag in json.loads((task / "diagnostics.json").read_text()):
            for learner in diag["learners"]:
                for field in ("optimizer_budget_reached", "encoder_budget_reached"):
                    if learner[field]:
                        convergence.update([diag["candidate"]])
        for target in TARGETS:
            for model in POLICIES:
                for control in ("persistence", "hist_000"):
                    rows = compare(frame, model, control, target)
                    if not rows:
                        continue
                    meta = {
                        "position": frame["position"][0],
                        "horizon": frame["horizon"][0],
                        "target": target,
                        "model": model,
                        "control": control,
                    }
                    annual.extend({**meta, **row} for row in rows)
                    diff = [r["delta_mse"] for r in rows]
                    low, high = paired_interval(diff, config["bootstrap_draws"], config["seed"])
                    control_mse = np.mean([r["control_mse"] for r in rows])
                    summaries.append(
                        {
                            **meta,
                            "years": len(rows),
                            "n": sum(r["n"] for r in rows),
                            "model_rmse": float(np.sqrt(np.mean([r["model_mse"] for r in rows]))),
                            "control_rmse": float(np.sqrt(control_mse)),
                            "delta_mse_pct": float(100 * np.mean(diff) / control_mse)
                            if control_mse
                            else None,
                            "annual_wins": sum(d < 0 for d in diff),
                            "delta_low": low,
                            "delta_high": high,
                        }
                    )
    pl.DataFrame(annual).write_parquet(output / "matched_annual.parquet")
    pl.DataFrame(summaries).write_parquet(output / "matched_summary.parquet")
    (output / "matched_summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    lines = [
        "# Matched forecast comparisons",
        "",
        "Every comparison uses the same players, forecast origins and observed target labels. "
        "Negative Δ MSE is better. These are retrospective, unadjusted comparisons.",
        "",
        "| Position | Horizon | Policy | Control | RMSE | Control RMSE | Δ MSE % | Years won |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in summaries:
        if r["target"] == "points" and r["model"] in ("selected_mse", "selected_ensemble"):
            percent = f"{r['delta_mse_pct']:.1f}%" if r["delta_mse_pct"] is not None else "—"
            lines.append(
                f"| {r['position']} | {r['horizon']} | {r['model']} | {r['control']} "
                f"| {r['model_rmse']:.2f} | {r['control_rmse']:.2f} "
                f"| {percent} | {r['annual_wins']}/{r['years']} |"
            )
    lines += [
        "",
        "## Model selection and training",
        "",
        f"Points-policy choices: {dict(choices)}",
        "",
        "Fits reaching their neural optimizer/encoder budget (not certified converged): "
        f"{dict(convergence)}",
        "",
        "The main dashboard scores each model's available predictions. For outputs with unknown "
        "persistence inputs, its sample sizes can differ. Use these matched comparisons for "
        "relative performance claims. Points predictions have complete labels and predictions.",
        "",
        "A three-year pass is insufficient to establish stable superiority. Inspect annual "
        "results, wider historical profiles and prospective evidence before adopting a recipe.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    shutil.copy2(__file__, output / "compare.py")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_run": args.source,
                "source_manifest_sha256": digest(source / "manifest.json"),
                "inputs": inputs,
                "analysis_sha256": digest(__file__),
                "choice_counts": dict(choices),
                "optimizer_budget_counts": dict(convergence),
            },
            indent=2,
        )
    )
    print(output / "report.md")


if __name__ == "__main__":
    main()
