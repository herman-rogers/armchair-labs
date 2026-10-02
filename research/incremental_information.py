"""Audit saved challengers against strong baselines; never fit or promote a model.

Run with explicit accepted history, outlook, college and a new output version.
All uncertainty uses equal-weight season means, preserving within-season dependence.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from artifact_inputs import require_audited_version
from data_integrity_audit import digest

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260922
RESAMPLES = 10000


def season_summary(values: list[float]) -> dict:
    """Paired differences: positive always means the challenger improves."""
    if not values or not np.isfinite(values).all():
        raise ValueError("Expected finite season differences")
    x = np.asarray(values)
    interval = None
    leave_one_out = None
    if len(x) > 1:
        boot = np.random.default_rng(SEED).choice(x, (RESAMPLES, len(x))).mean(axis=1)
        interval = np.quantile(boot, [0.025, 0.975]).tolist()
        omitted = (x.sum() - x) / (len(x) - 1)
        leave_one_out = [float(omitted.min()), float(omitted.max())]
    return {
        "mean": float(x.mean()),
        "season_bootstrap_95": interval,
        "positive_seasons": int((x > 1e-9).sum()),
        "seasons": len(x),
        "leave_one_season_out_mean_range": leave_one_out,
    }


def require_unique(frame: pl.DataFrame, keys: list[str]) -> None:
    if frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        raise ValueError(f"Null observation key: {keys}")
    if frame.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError(f"Duplicate observation key: {keys}")


def point_comparisons(
    frame: pl.DataFrame, actual: str, challenger: str, baselines: list[str]
) -> list[dict]:
    """Use one common finite sample for the entire baseline ladder.

    delta = challenger - baseline; e = actual - baseline.
    Baseline MSE - challenger MSE = mean(2*e*delta) - mean(delta**2).
    This diagnoses a saved forecast adjustment, not causal feature attribution.
    """
    columns = [actual, challenger, *baselines]
    common = frame.filter(pl.all_horizontal(pl.col(columns).is_finite().fill_null(False)))
    if common.is_empty():
        raise ValueError("No common finite forecasts")
    if common["season"].max() > 2025:
        raise ValueError("2026 outcomes must remain ungraded")
    results = []
    for baseline in baselines:
        folds = []
        for (season,), group in common.group_by("season", maintain_order=True):
            y, b, c = (group[key].to_numpy() for key in (actual, baseline, challenger))
            error, delta = y - b, c - b
            folds.append(
                {
                    "season": season,
                    "n": len(y),
                    "baseline_mae": float(np.abs(error).mean()),
                    "challenger_mae": float(np.abs(y - c).mean()),
                    "mae_improvement": float((np.abs(error) - np.abs(y - c)).mean()),
                    "mse_improvement": float((error**2 - (y - c) ** 2).mean()),
                    "residual_alignment": float((2 * error * delta).mean()),
                    "adjustment_cost": float((delta**2).mean()),
                }
            )
        folds.sort(key=lambda r: r["season"])
        results.append(
            {
                "baseline": baseline,
                "challenger": challenger,
                "input_rows": frame.height,
                "common_rows": common.height,
                "excluded_nonfinite_rows": frame.height - common.height,
                "pooled_mae": {
                    key: float((common[key] - common[actual]).abs().mean())
                    for key in (baseline, challenger)
                },
                "mae_improvement": season_summary([f["mae_improvement"] for f in folds]),
                "mse_improvement": season_summary([f["mse_improvement"] for f in folds]),
                "residual_alignment": float(np.mean([f["residual_alignment"] for f in folds])),
                "adjustment_cost": float(np.mean([f["adjustment_cost"] for f in folds])),
                "folds": folds,
            }
        )
    return results


def role_comparisons(selections: pl.DataFrame) -> list[dict]:
    require_unique(selections, ["cohort", "policy", "season", "week", "player_id"])
    if selections["season"].max() > 2025:
        raise ValueError("2026 outcomes must remain ungraded")
    results = []
    for cohort in sorted(selections["cohort"].unique()):
        pool = selections.filter(pl.col("cohort") == cohort)
        for baseline in ("trailing_points", "recent_snaps", "usage_ridge"):
            policies = {
                policy: {
                    (season, week): {r["player_id"]: r["next4_points"] for r in g.to_dicts()}
                    for (season, week), g in pool.filter(pl.col("policy") == policy).group_by(
                        "season", "week"
                    )
                }
                for policy in (baseline, "role_ridge")
            }
            base, candidate = policies[baseline], policies["role_ridge"]
            if not base or base.keys() != candidate.keys():
                raise ValueError("Role policies must cover identical decision windows")
            folds = {}
            for (season, week), chosen in sorted(candidate.items()):
                reference = base[(season, week)]
                if len(chosen) != 3 or len(reference) != 3:
                    raise ValueError("Role pilot requires three selections per policy/window")
                if not np.isfinite([*chosen.values(), *reference.values()]).all():
                    raise ValueError("Role outcomes must be finite")
                common = chosen.keys() & reference.keys()
                if any(chosen[p] != reference[p] for p in common):
                    raise ValueError("Role policies disagree on an observed outcome")
                f = folds.setdefault(
                    season,
                    {
                        "season": season,
                        "windows": 0,
                        "picks": 0,
                        "changed_picks": 0,
                        "baseline_points": 0.0,
                        "challenger_points": 0.0,
                    },
                )
                f["windows"] += 1
                f["picks"] += 3
                f["changed_picks"] += 3 - len(common)
                f["baseline_points"] += sum(reference.values())
                f["challenger_points"] += sum(chosen.values())
            for f in folds.values():
                f["gain_per_selection"] = (f["challenger_points"] - f["baseline_points"]) / f[
                    "picks"
                ]
            ordered = [folds[s] for s in sorted(folds)]
            picks = sum(f["picks"] for f in ordered)
            changed = sum(f["changed_picks"] for f in ordered)
            results.append(
                {
                    "cohort": cohort,
                    "baseline": baseline,
                    "challenger": "role_ridge",
                    "units": "next-four-calendar-week points per selection",
                    "picks": picks,
                    "changed_picks": changed,
                    "unchanged_selection_fraction": 1 - changed / picks,
                    "gain_per_selection": season_summary(
                        [f["gain_per_selection"] for f in ordered]
                    ),
                    "folds": ordered,
                }
            )
    return results


def verify_files(directory: Path, hashes: dict, recorded: dict) -> None:
    if not hashes:
        raise ValueError(f"Missing artifact hashes: {directory}")
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"Artifact changed or outside its version: {name}")
        recorded[str(path)] = expected


def load_release(path: Path, history: dict, required: set[str], recorded: dict) -> dict:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("protected_artifacts_unchanged") is not True
        or manifest.get("history") != history
        or not required <= set(manifest.get("output_sha256", {}))
    ):
        raise ValueError(f"Incomplete or mismatched release: {path}")
    recorded[str(manifest_path)] = digest(manifest_path)
    verify_files(path, manifest["source_sha256"], recorded)
    verify_files(path, manifest["output_sha256"], recorded)
    return manifest


def build_report(history_dir: Path, outlook_dir: Path, college_dir: Path) -> dict:
    history = require_audited_version(history_dir)
    recorded = {}
    acceptance_path = history_dir / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    required = {
        "outputs/research_rb_role_selections.parquet",
        "outputs/value_research_2026-09-22.json",
    }
    if (
        acceptance.get("status") != "accepted_for_research_not_promoted"
        or acceptance.get("audited_input") != history
        or not required <= set(acceptance.get("research_output_sha256", {}))
    ):
        raise ValueError("History lacks matching accepted research outputs")
    recorded[str(acceptance_path)] = digest(acceptance_path)
    verify_files(history_dir, acceptance["research_output_sha256"], recorded)
    load_release(outlook_dir, history, {"validation_predictions.parquet"}, recorded)
    load_release(college_dir, history, {"nfl_predictions.parquet", "report.json"}, recorded)

    role = role_comparisons(
        pl.read_parquet(history_dir / "outputs/research_rb_role_selections.parquet")
    )
    outlook = pl.read_parquet(outlook_dir / "validation_predictions.parquet")
    require_unique(outlook, ["player_id", "season", "cutoff_week"])
    for row in outlook.to_dicts():
        if any(s >= row["season"] for s in row["train_seasons"] + row["calibration_seasons"]):
            raise ValueError("Outlook training/calibration includes the evaluated season")
    outlook_results = []
    for (position, week), pool in outlook.group_by("position", "cutoff_week"):
        for result in point_comparisons(
            pool, "next4_points", "forecast_next4", ["points_pace_next4", "usage_baseline_next4"]
        ):
            outlook_results.append({"position": position, "cutoff_week": week, **result})
    outlook_results.sort(key=lambda r: (r["cutoff_week"], r["position"], r["baseline"]))

    college = pl.read_parquet(college_dir / "nfl_predictions.parquet")
    require_unique(college, ["player_id", "forecast_year", "target"])
    if college.filter(pl.col("train_latest_outcome_year") >= pl.col("forecast_year")).height:
        raise ValueError("College training outcomes have not matured before the forecast")
    # Match the published 2018+ window. Pending/immature outcomes are reported separately.
    college = college.filter(pl.col("forecast_year") >= 2018)
    pending = college.filter(pl.col("actual").is_null())
    completed = college.filter(pl.col("actual").is_not_null()).with_columns(
        pl.col("forecast_year").alias("season")
    )
    if completed.filter(
        pl.col("forecast_year")
        + pl.when(pl.col("target") == "nfl_first3_points").then(2).otherwise(0)
        > 2025
    ).height:
        raise ValueError("College evaluation contains incomplete outcome horizons")
    college_results = []
    for (position, target), pool in completed.group_by("position", "target"):
        for result in point_comparisons(pool, "actual", "college_plus_draft", ["draft_only"]):
            college_results.append({"position": position, "target": target, **result})
    college_results.sort(key=lambda r: (r["target"], r["position"]))

    market = json.loads((history_dir / "outputs/value_research_2026-09-22.json").read_text())
    # Retain all existing variants, with their original coverage and units; no winner selection.
    report = {
        "schema_version": 1,
        "status": "exploratory_not_promoted",
        "saved_at": datetime.now(UTC).isoformat(),
        "history": history,
        "source_sha256": recorded,
        "implementation_sha256": digest(Path(__file__)),
        "design": {
            "resamples": RESAMPLES,
            "seed": SEED,
            "uncertainty": "paired, equal-weight season means; 95% descriptive intervals",
            "point_error_units": "target units for MAE; squared target units for MSE",
            "mse_identity": "improvement = residual_alignment - adjustment_cost",
            "primary_outlook_readout": "Week 2, all scored candidates by position",
            "college_window": "2018+ entry classes with completed target horizons through 2025",
            "promotions": 0,
        },
        "role_selection": role,
        "outlook": outlook_results,
        "college": college_results,
        "college_pending_rows": pending.height,
        "college_identity_sensitivity": json.loads((college_dir / "report.json").read_text())[
            "strict_identity_sensitivity"
        ],
        "market": {
            "units": market["target_units"],
            "experiments": market["experiments"],
            "limitations": market["limitations"],
        },
        "limitations": [
            "Reanalysis of previously inspected forecasts, not independent confirmation.",
            "No interval is adjusted for the many positions, targets, windows and prior variants.",
            "Season blocks preserve within-season dependence; "
            "cross-season player dependence remains.",
            "College three-year targets overlap calendar seasons across entry classes.",
            "Forecast deltas diagnose predictive contribution, "
            "not information-theoretic or causal redundancy.",
            "ECR ranks are not points; "
            "market comparisons retain their original selection-value units.",
            "No synchronized executable market or waiver prices; no decision-profit claim.",
            "Historical reconstruction and original study coverage/identity limits still apply.",
        ],
    }
    for name, expected in recorded.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Input changed during audit: {name}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "outlook", "college"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not args.version or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.version
    ):
        parser.error("Version must contain only letters, digits, underscores or hyphens")
    output = ROOT / "data/research" / args.version
    if output.exists():
        parser.error("Output version already exists; choose a new version")
    report = build_report(args.history.resolve(), args.outlook.resolve(), args.college.resolve())
    report["version"] = args.version
    output.mkdir()  # create-only, including protection against a concurrent writer
    (output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "report.json")


if __name__ == "__main__":
    main()
