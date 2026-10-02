"""Offline nested walk-forward correction of the accepted RB usage baseline.

The protocol is fixed in rb_residual_protocol.json. All outputs are create-only
research artifacts; no application pointer, ranking or forecast is published.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import role_transition_pilot as pilot
from artifact_inputs import require_audited_version
from data_integrity_audit import digest, read_sources, select_sources
from incremental_information import point_comparisons, require_unique, season_summary, verify_files

from patron.metrics.positions import canonical_positions

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name("rb_residual_protocol.json")
COHORTS = ("all_late_price", "small_prior_sample")
POLICIES = ("usage_ridge", "role_ridge", "recalibration", "residual_raw", "residual_tuned")


def eligible_windows(rows: list[dict], cohort: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if (
            row["adp"] is not None
            and 100 < row["adp"] <= 300
            and row["last3_points"] < 12
            and (cohort != "small_prior_sample" or row["small_prior_sample"])
        ):
            groups[row["season"], row["week"]].append(row)
    return {key: group for key, group in sorted(groups.items()) if len(group) >= 3}


def adjusted(row: dict, correction: str, weight: float, half_life: float) -> float:
    # Return directly so zero correction does not depend on a fitted correction value.
    if weight == 0:
        return row["usage_ridge"]
    n = row["prior_season_games"] + row["observed_games"]
    reliability = n / (n + half_life) if n + half_life else 0.0
    return row["usage_ridge"] + weight * reliability * row[correction]


def configurations(protocol: dict) -> list[tuple[float, float]]:
    return [(0.0, 0.0)] + [
        (weight, half_life)
        for weight in protocol["shrinkage_weights"]
        if weight > 0
        for half_life in protocol["sample_half_lives"]
    ]


def choose_shrinkage(rows: list[dict], outer_season: int, correction: str, protocol: dict) -> dict:
    """Tune only on predictions themselves fitted before each inner validation season."""
    earlier = [r for r in rows if r["season"] < outer_season]
    windows = eligible_windows(earlier, "all_late_price")
    years = sorted({year for year, _ in windows})
    if len(years) < protocol["tuning_minimum_prior_validation_seasons"]:
        raise ValueError("Not enough earlier validation seasons for shrinkage tuning")
    trials = []
    for weight, half_life in configurations(protocol):
        scored = defaultdict(list)
        for (year, _), group in windows.items():
            chosen = sorted(
                group, key=lambda r: (-adjusted(r, correction, weight, half_life), r["player_id"])
            )[:3]
            scored[year].extend(r["next4_points"] for r in chosen)
        folds = [{"season": y, "points_per_pick": float(np.mean(scored[y]))} for y in years]
        trials.append(
            {
                "weight": weight,
                "half_life": half_life,
                "points_per_pick": float(np.mean([f["points_per_pick"] for f in folds])),
                "folds": folds,
            }
        )
    best = max(t["points_per_pick"] for t in trials)
    selected = min(
        (t for t in trials if best - t["points_per_pick"] <= 1e-10),
        key=lambda t: (t["weight"], -t["half_life"]),
    )
    return {
        "season": outer_season,
        "correction": correction,
        "validation_seasons": years,
        "selected": selected,
        "trials": trials,
    }


def nested_predictions(
    rows: list[dict], protocol: dict
) -> tuple[list[dict], list[dict], list[dict]]:
    """Each baseline residual and each correction forecast is chronological out of fold."""
    require_unique(pl.DataFrame(rows), ["player_id", "season", "week"])
    years = sorted({r["season"] for r in rows})
    if max(years) > 2025:
        raise ValueError("2026 outcomes must remain ungraded")
    oof = []
    for year in years:
        train = [r for r in rows if r["season"] < year]
        if len({r["season"] for r in train}) < protocol["baseline_minimum_prior_seasons"]:
            continue
        test = [r.copy() for r in rows if r["season"] == year]
        forecasts = pilot.fit_predict(train, test, pilot.USAGE)
        for row, prediction in zip(test, forecasts, strict=True):
            row.update(
                usage_ridge=float(prediction),
                residual=row["next4_points"] - prediction,
                baseline_train_latest=year - 1,
            )
        oof.extend(test)
    corrected, diagnostics = [], []
    for year in sorted({r["season"] for r in oof}):
        train = [r for r in oof if r["season"] < year]
        if (
            len({r["season"] for r in train})
            < protocol["correction_minimum_prior_residual_seasons"]
        ):
            continue
        test = [r.copy() for r in oof if r["season"] == year]
        for label, features in (
            ("role_delta", protocol["residual_features"]),
            ("recalibration_delta", protocol["recalibration_features"]),
        ):
            predictions = pilot.fit_predict(train, test, features, target="residual")
            for row, prediction in zip(test, predictions, strict=True):
                row[label] = float(prediction)
        diagnostics.append(
            {
                "season": year,
                "residual_training_rows": len(train),
                "residual_training_seasons": sorted({r["season"] for r in train}),
                "baseline_train_latest": year - 1,
                "correction_train_latest": max(r["season"] for r in train),
                "outcome_maturation": "all training windows end by Week 17 of an earlier season",
            }
        )
        corrected.extend(test)
    output, tuning = [], []
    for year in protocol["outer_test_seasons"]:
        test = [r.copy() for r in corrected if r["season"] == year]
        if not test:
            raise ValueError(f"Missing requested outer fold: {year}")
        original = pilot.fit_predict([r for r in rows if r["season"] < year], test, pilot.ROLE)
        for row, prediction in zip(test, original, strict=True):
            row["role_ridge"] = float(prediction)
            row["residual_raw"] = row["usage_ridge"] + row["role_delta"]
        for policy, delta in (
            ("recalibration", "recalibration_delta"),
            ("residual_tuned", "role_delta"),
        ):
            choice = choose_shrinkage(corrected, year, delta, protocol)
            tuning.append({"policy": policy, **choice})
            for row in test:
                row[policy] = adjusted(
                    row, delta, choice["selected"]["weight"], choice["selected"]["half_life"]
                )
        output.extend(test)
    return output, tuning, diagnostics


def evaluate(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    summaries, error_summaries, selections = [], [], []
    for cohort in COHORTS:
        windows = eligible_windows(rows, cohort)
        by_policy = defaultdict(dict)
        for policy in POLICIES:
            groups = defaultdict(list)
            for (year, _week), group in windows.items():
                chosen = sorted(group, key=lambda r: (-r[policy], r["player_id"]))[:3]
                groups[year].extend(chosen)
                selections.extend({"cohort": cohort, "policy": policy, **r} for r in chosen)
            for year, chosen in groups.items():
                by_policy[policy][year] = {
                    "season": year,
                    "picks": len(chosen),
                    "windows": len(chosen) // 3,
                    "points_per_pick": float(np.mean([r["next4_points"] for r in chosen])),
                    "keys": {(r["week"], r["player_id"]) for r in chosen},
                }
        for policy in POLICIES:
            for baseline in ("usage_ridge", "recalibration"):
                if policy == baseline:
                    continue
                folds = []
                for year, fold in sorted(by_policy[policy].items()):
                    reference = by_policy[baseline][year]
                    folds.append(
                        {
                            **{k: v for k, v in fold.items() if k != "keys"},
                            "baseline_points_per_pick": reference["points_per_pick"],
                            "gain": fold["points_per_pick"] - reference["points_per_pick"],
                            "changed_picks": len(fold["keys"] - reference["keys"]),
                        }
                    )
                summaries.append(
                    {
                        "cohort": cohort,
                        "policy": policy,
                        "baseline": baseline,
                        "picks": sum(f["picks"] for f in folds),
                        "changed_picks": sum(f["changed_picks"] for f in folds),
                        "gain_per_selection": season_summary([f["gain"] for f in folds]),
                        "folds": folds,
                    }
                )
        frame = pl.DataFrame([r for group in windows.values() for r in group])
        for policy in POLICIES[1:]:
            baselines = ["usage_ridge"] + (["recalibration"] if policy != "recalibration" else [])
            for summary in point_comparisons(frame, "next4_points", policy, baselines):
                error_summaries.append({"cohort": cohort, **summary})
    return summaries, error_summaries, selections


def reconstruct(history_dir: Path) -> tuple[list[dict], dict, dict]:
    history = require_audited_version(history_dir)
    recorded = {}
    acceptance_path = history_dir / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    if (
        acceptance.get("status") != "accepted_for_research_not_promoted"
        or acceptance["audited_input"] != history
    ):
        raise ValueError("Accepted history does not match audited inputs")
    recorded[str(acceptance_path)] = digest(acceptance_path)
    verify_files(history_dir, acceptance["research_output_sha256"], recorded)
    report = json.loads((history_dir / "outputs/role_transition_pilot_2026-09-22.json").read_text())
    if digest(Path(pilot.__file__)) != report["implementation_sha256"]:
        raise ValueError("Pilot implementation differs from the accepted experiment")
    verify_files(ROOT, report["source_sha256"], recorded)
    paths = [ROOT / name for name in report["source_sha256"]]
    index = {
        p: set(pl.scan_parquet(p).collect_schema().names()) for p in paths if p.suffix == ".parquet"
    }
    identities = read_sources(select_sources(index, ["gsis_id", "display_name", "short_name"]))
    snaps = canonical_positions(
        read_sources(select_sources(index, ["pfr_player_id", "offense_snaps", "offense_pct"]))
    )
    snaps = snaps.filter(
        (pl.col("game_type") == "REG")
        & (pl.col("position") == "RB")
        & pl.col("season").is_between(2013, 2025)
    ).join(
        identities.select(pl.col("gsis_id").alias("player_id"), "pfr_id").drop_nulls(),
        left_on="pfr_player_id",
        right_on="pfr_id",
        how="inner",
    )
    require_unique(snaps, ["player_id", "season", "week"])
    if snaps.filter(
        ~pl.col("offense_pct").is_finite() | ~pl.col("offense_pct").is_between(0, 1)
    ).height:
        raise ValueError("Invalid snap shares")
    points = pl.read_parquet(history_dir / "outputs/research_audited_weekly_points.parquet")
    adp = pl.read_csv(history_dir / "static/market_adp_backfill.csv").filter(
        pl.col("preferred") & pl.col("gsis_id").is_not_null()
    )
    return pilot.make_rows(points, snaps, adp), history, recorded


def verify_reproduction(rows: list[dict], selections: list[dict], history_dir: Path) -> dict:
    keys = ["player_id", "season", "week"]
    original = pl.read_parquet(history_dir / "outputs/research_rb_role_predictions.parquet")
    current = pl.DataFrame(rows)
    a, b = original.sort(keys), current.sort(keys)
    if a.select(keys).to_dicts() != b.select(keys).to_dicts():
        raise ValueError("Candidate universe differs from accepted role pilot")
    errors = {}
    for key in [*pilot.ROLE, "adp", "next4_points", "usage_ridge", "role_ridge"]:
        av, bv = a[key].to_numpy(), b[key].to_numpy()
        if not np.allclose(av, bv, rtol=0, atol=1e-9, equal_nan=True):
            raise ValueError(f"Accepted pilot reproduction failed: {key}")
        errors[key] = float(np.nanmax(np.abs(av - bv)))
    saved = pl.read_parquet(history_dir / "outputs/research_rb_role_selections.parquet")
    selection_keys = ["cohort", "policy", *keys]
    policies = ["usage_ridge", "role_ridge"]
    a = saved.filter(pl.col("policy").is_in(policies)).select(selection_keys).sort(selection_keys)
    b = (
        pl.DataFrame(selections)
        .filter(pl.col("policy").is_in(policies))
        .select(selection_keys)
        .sort(selection_keys)
    )
    if a.to_dicts() != b.to_dicts():
        raise ValueError("Baseline selections differ from accepted pilot")
    return {
        "prediction_rows": current.height,
        "selection_rows": a.height,
        "maximum_absolute_differences": errors,
        "passed": True,
    }


def advancement_checks(selections: list[dict], errors: list[dict], protocol: dict) -> dict:
    def gain(cohort, baseline):
        return next(
            r["gain_per_selection"]
            for r in selections
            if r["cohort"] == cohort
            and r["policy"] == "residual_tuned"
            and r["baseline"] == baseline
        )

    primary = gain("all_late_price", "usage_ridge")
    control = gain("all_late_price", "recalibration")
    error = next(
        r
        for r in errors
        if r["cohort"] == "all_late_price"
        and r["challenger"] == "residual_tuned"
        and r["baseline"] == "usage_ridge"
    )
    return {
        "material_primary_gain": primary["mean"]
        >= protocol["exploratory_advancement_checks"][
            "minimum_primary_points_per_pick_gain_vs_usage"
        ],
        "positive_primary_interval_vs_usage": primary["season_bootstrap_95"][0] > 0,
        "positive_primary_interval_vs_recalibration": control["season_bootstrap_95"][0] > 0,
        "primary_mae_not_worse": error["mae_improvement"]["mean"] >= 0,
        "secondary_selection_gain_not_negative": gain("small_prior_sample", "usage_ridge")["mean"]
        >= 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not args.version or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.version
    ):
        parser.error("Invalid version identifier")
    output = ROOT / "data/research" / args.version
    if output.exists():
        parser.error("Output exists; choose a new version")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["ridge_penalty"] != 10.0:
        raise ValueError("Accepted pilot fitter requires ridge penalty 10")
    print("Verifying accepted inputs and reconstructing the original RB panel...", flush=True)
    history_dir = args.history.resolve()
    rows, history, recorded = reconstruct(history_dir)
    implementation = [
        Path(__file__),
        PROTOCOL_PATH,
        Path(pilot.__file__),
        Path(__file__).with_name("incremental_information.py"),
        Path(__file__).with_name("artifact_inputs.py"),
        Path(__file__).with_name("data_integrity_audit.py"),
        ROOT / "src/patron/metrics/positions.py",
    ]
    recorded.update({str(p.resolve()): digest(p) for p in implementation})
    protected = {
        str(p): digest(p)
        for p in [
            *(ROOT / "data/outputs").glob("board*.json"),
            *(ROOT / "data/static/draft_2026").rglob("*"),
            ROOT / "data/static/experimental_2026_predictions.csv",
            ROOT / "src/patron/config/experimental_freeze_2026.yaml",
        ]
        if p.is_file()
    }
    print("Fitting chronological baseline residuals and nested shrinkage choices...", flush=True)
    predictions, tuning, folds = nested_predictions(rows, protocol)
    summaries, errors, selections = evaluate(predictions)
    reproduction = verify_reproduction(predictions, selections, history_dir)
    checks = advancement_checks(summaries, errors, protocol)
    for name, expected in {**recorded, **protected}.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Input or protected artifact changed during experiment: {name}")
    report = {
        "version": args.version,
        "saved_at": datetime.now(UTC).isoformat(),
        "status": "exploratory_not_promoted",
        "history": history,
        "protocol": protocol,
        "source_sha256": recorded,
        "protected_sha256": protected,
        "protected_artifacts_unchanged": True,
        "baseline_reproduction": reproduction,
        "fold_chronology": folds,
        "tuning": tuning,
        "selection_comparisons": summaries,
        "point_error_comparisons": errors,
        "advancement_checks": checks,
        "advance_to_prospective_test": all(checks.values()),
        "limitations": [
            "Previously inspected historical seasons, not an untouched confirmatory sample.",
            "Ten shrinkage configurations per correction; "
            "selection uses only prior validation seasons.",
            "Earliest evaluation has only two prior validation seasons; "
            "baseline training grows over time.",
            "Season bootstrap preserves overlapping windows within seasons, "
            "not cross-season dependence.",
            "Current-week offensive activity is required; "
            "inactive players are outside the candidate pool.",
            "Preseason ADP is not executable same-week waiver price; no acquisition-profit claim.",
            "Revised historical stats/snaps and original missing-data assumptions "
            "remain limitations.",
            "No 2026 outcomes graded, current forecasts generated or default rankings changed.",
        ],
    }
    output.mkdir()
    (output / "implementation").mkdir()
    for path in implementation:
        (output / "implementation" / path.name).write_bytes(path.read_bytes())
    pl.DataFrame(predictions).write_parquet(output / "predictions.parquet")
    pl.DataFrame(selections).write_parquet(output / "selections.parquet")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "version": args.version,
                "status": "complete",
                "history": history,
                "protocol_sha256": digest(PROTOCOL_PATH),
                "output_sha256": {
                    str(p.relative_to(output)): digest(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(output / "report.json")


if __name__ == "__main__":
    main()
