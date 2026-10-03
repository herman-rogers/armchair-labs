"""Offline preseason workload experiment on an accepted, frozen historical rebuild.

This tests prior-role representations, not knowledge of next season's job. Four
fixed ridge specifications predict workload and points independently. No future
role labels enter features, and no workload multiplier discounts points twice.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from artifact_inputs import require_audited_version
from data_integrity_audit import digest
from value_capture_audit import finite, model_values, rank_ids, score, summarize

from engine.metrics.outlook import fit, team_name, unique
from engine.metrics.positions import canonical_positions

ROOT = Path(__file__).resolve().parents[1]
POSITIONS = ("QB", "RB", "TE")
BASE = ("ppg", "games", "age_at_season", "player_experience", "season_length")
ANNUAL = (
    *BASE,
    "prior_attempts_pg",
    "prior_carries_pg",
    "prior_targets_pg",
    "prior_snaps_pg",
    "prior_snap_share",
    "prior_offensive_games",
)
RECENT = (
    *ANNUAL,
    "late_points_pg",
    "late_attempts_pg",
    "late_carries_pg",
    "late_targets_pg",
    "late_snaps_pg",
    "late_snap_share",
)
ROLE = (*RECENT, "role_games", "role_ppg", "other_ppg", "late_role_fraction")
SPECS = {"production": BASE, "annual_usage": ANNUAL, "recent_usage": RECENT, "role_summary": ROLE}
WORKLOADS = {
    "QB": ("attempts", "carries"),
    "RB": ("carries", "targets"),
    "TE": ("targets", "offense_snaps"),
}


def mean(values):
    return float(np.mean(values)) if values else None


def role_week(position, snap, point):
    """Observable high-usage proxies; TE target earning is not route/block charting."""
    share = float(snap.get("offense_pct") or 0)
    if position == "QB":
        return share >= 0.5
    if position == "RB":
        return share >= 0.5 and (point.get("carries", 0) + point.get("targets", 0)) >= 12
    return (point.get("targets") or 0) >= 4


def build_rows(candidates, points, snaps, schedule, identities, complete_snap_seasons):
    """Features use only forecast_season - 1; full-season labels stay separate.

    The late window is the last four scheduled games of the last *prior-season*
    observed team. Missing appearances in that window remain zero with certified
    feed coverage, rather than moving the window backward to a player's last game.
    """
    keys = ["player_id", "season", "week"]
    unique(points, keys)
    unique(snaps, keys)
    unique(candidates, ["player_id", "forecast_season"])
    pg, sg, team_weeks = defaultdict(dict), defaultdict(dict), defaultdict(set)
    for r in points.to_dicts():
        pg[r["player_id"], r["season"]][r["week"]] = r
    for r in snaps.to_dicts():
        if not finite(r["offense_pct"]) or not 0 <= r["offense_pct"] <= 1:
            raise ValueError("Invalid offensive snap share")
        sg[r["player_id"], r["season"]][r["week"]] = r
    for r in schedule.filter(pl.col("game_type") == "REG").to_dicts():
        for team in (r["home_team"], r["away_team"]):
            team_weeks[r["season"], team_name(team)].add(r["week"])
    rows = []
    for candidate in candidates.to_dicts():
        year, pid = candidate["forecast_season"], candidate["player_id"]
        if candidate["position"] not in POSITIONS or candidate["player_population"] != "returner":
            continue
        if not 2014 <= year <= 2025 or not candidate["outcome_complete"]:
            continue
        prior_points, prior_snaps = pg[pid, year - 1], sg[pid, year - 1]
        eligible = pid in identities and year - 1 in complete_snap_seasons
        # Only past observations resolve the prior team. Unknown late windows stay unknown.
        observations = sorted(
            [(w, r["team"]) for w, r in prior_points.items() if r.get("team")]
            + [(w, r["team"]) for w, r in prior_snaps.items() if r.get("team")]
        )
        team = team_name(observations[-1][1]) if observations else None
        weeks = sorted(team_weeks[year - 1, team]) if team else []
        late = weeks[-4:]
        n = 16 if year - 1 <= 2020 else 17

        def stat(w, field, prior_snaps=prior_snaps, prior_points=prior_points):
            source = prior_snaps if field == "offense_snaps" else prior_points
            return float(source.get(w, {}).get(field) or 0)

        all_weeks = sorted(set(prior_points) | set(prior_snaps))
        active = [w for w in all_weeks if stat(w, "offense_snaps") > 0]
        high = [
            w
            for w in active
            if role_week(candidate["position"], prior_snaps[w], prior_points.get(w, {}))
        ]
        # Box-score appearances match the historical PPG denominator, even in tiny roles.
        other = [w for w in prior_points if w not in high]
        row = {
            **candidate,
            "eligible": eligible,
            "season": year,
            "season_length": 16 if year <= 2020 else 17,
            "prior_team": team,
            "late_weeks": late,
            "prior_offensive_games": len(active) if eligible else None,
            "role_games": len(high) if eligible else None,
            "role_ppg": mean([stat(w, "league_points") for w in high]) if eligible else None,
            "other_ppg": mean([stat(w, "league_points") for w in other]) if eligible else None,
            "late_role_fraction": sum(w in high for w in late) / len(late)
            if eligible and late
            else None,
        }
        for field, label in [
            ("attempts", "attempts"),
            ("carries", "carries"),
            ("targets", "targets"),
            ("offense_snaps", "snaps"),
            ("league_points", "points"),
        ]:
            row[f"prior_{label}_pg"] = sum(stat(w, field) for w in all_weeks) / n
            row[f"late_{label}_pg"] = mean([stat(w, field) for w in late]) if late else None
        row["prior_snap_share"] = sum(r["offense_pct"] for r in prior_snaps.values()) / n
        row["late_snap_share"] = (
            mean([float(prior_snaps.get(w, {}).get("offense_pct") or 0) for w in late])
            if late
            else None
        )
        if not eligible:
            for name in ("prior_snaps_pg", "late_snaps_pg", "prior_snap_share", "late_snap_share"):
                row[name] = None
        row["late_role_growth"] = bool(
            eligible
            and late
            and row["late_snap_share"] - row["prior_snap_share"] >= 0.15
            and row["late_role_fraction"] >= 0.5
        )
        # Outcomes may change without changing any field above, including eligibility.
        future_points, future_snaps = pg[pid, year], sg[pid, year]
        for field in WORKLOADS[candidate["position"]]:
            source = future_snaps if field == "offense_snaps" else future_points
            row[f"actual_{field}"] = sum(float(r.get(field) or 0) for r in source.values())
            if field == "offense_snaps" and year not in complete_snap_seasons:
                row[f"actual_{field}"] = None
        rows.append(row)
    return rows


def predict(rows):
    result, folds = [], []
    for year in range(2019, 2026):
        for position in POSITIONS:
            train = [
                r
                for r in rows
                if r["eligible"] and r["season"] < year and r["position"] == position
            ]
            test = [
                r.copy()
                for r in rows
                if r["eligible"] and r["season"] == year and r["position"] == position
            ]
            if not test:
                continue
            if len(train) < 30:
                raise ValueError("Insufficient earlier-season training rows")
            folds.append(
                {
                    "season": year,
                    "position": position,
                    "train": len(train),
                    "test": len(test),
                    "last_train_season": max(r["season"] for r in train),
                }
            )
            for spec, features in SPECS.items():
                for target in ("season_points", *WORKLOADS[position]):
                    fitted = fit(train, features, f"actual_{target}")
                    if fitted is None:
                        raise ValueError("Insufficient target coverage")
                    for row, value in zip(test, fitted.predict(test), strict=True):
                        row[f"{spec}_{target}"] = max(float(value), 0.0)
            result.extend(test)
    return result, folds


def interval(values):
    a = np.array(values)
    boot = np.random.default_rng(20260922).choice(a, (10000, len(a))).mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def diagnostics(rows):
    results = []
    for position in POSITIONS:
        for cohort in ("all", "prior_under10", "late_role_growth", "career_years_2_3"):
            group = [
                r
                for r in rows
                if r["position"] == position
                and (
                    cohort == "all"
                    or (cohort == "prior_under10" and r["games"] < 10)
                    or (cohort == "late_role_growth" and r["late_role_growth"])
                    or (cohort == "career_years_2_3" and (r.get("player_experience") or 99) <= 2)
                )
            ]
            for target in ("season_points", *WORKLOADS[position]):
                models = list(SPECS) + (["saved_v2"] if target == "season_points" else [])
                for model in models:
                    field = "fitted_season_points" if model == "saved_v2" else f"{model}_{target}"
                    good = [
                        r
                        for r in group
                        if finite(r.get(f"actual_{target}")) and finite(r.get(field))
                    ]
                    if not good:
                        continue
                    folds = []
                    for year in sorted({r["season"] for r in good}):
                        fold = [r for r in good if r["season"] == year]
                        mae = mean([abs(r[field] - r[f"actual_{target}"]) for r in fold])
                        folds.append(
                            {
                                "season": year,
                                "n": len(fold),
                                "mae": mae,
                                **{
                                    f"improvement_vs_{ref}": mean(
                                        [
                                            abs(r[f"{ref}_{target}"] - r[f"actual_{target}"])
                                            for r in fold
                                        ]
                                    )
                                    - mae
                                    for ref in ("production", "annual_usage", "recent_usage")
                                },
                            }
                        )
                    results.append(
                        {
                            "position": position,
                            "cohort": cohort,
                            "target": target,
                            "model": model,
                            "n": len(good),
                            "folds": folds,
                            "season_mean_mae": mean([f["mae"] for f in folds]),
                            **{
                                f"vs_{ref}": {
                                    "mae_improvement": mean(
                                        [f[f"improvement_vs_{ref}"] for f in folds]
                                    ),
                                    "season_bootstrap_95": interval(
                                        [f[f"improvement_vs_{ref}"] for f in folds]
                                    ),
                                }
                                for ref in ("production", "annual_usage", "recent_usage")
                            },
                        }
                    )
    return results


def market_evaluation(predictions, full):
    """Keep the original all-player pool; unchanged forecasts fill unmodeled rows."""
    lookup = {(r["player_id"], r["season"]): r for r in predictions}
    summaries, misses = {}, []
    for model in ("saved_v2", *SPECS):
        folds = []
        for year in range(2019, 2026):
            rows = [
                r.copy()
                for r in full
                if r["forecast_season"] == year
                and r["outcome_complete"]
                and all(
                    finite(r.get(k))
                    for k in (
                        "fitted_season_points",
                        "market_overall_ecr_score",
                        "actual_availability_value",
                    )
                )
            ]
            replaced = 0
            for r in rows:
                new = lookup.get((r["player_id"], year))
                r["challenger"] = (
                    new[f"{model}_season_points"]
                    if new and model != "saved_v2"
                    else r["fitted_season_points"]
                )
                replaced += bool(new and model != "saved_v2")
            market = rank_ids(rows, {r["player_id"]: r["market_overall_ecr_score"] for r in rows})
            order = rank_ids(rows, model_values(rows, "challenger"))
            actual = rank_ids(rows, {r["player_id"]: r["actual_availability_value"] for r in rows})
            folds.append(
                {"season": year, "replaced_forecasts": replaced, **score(rows, order, market, 60)}
            )
            for r in rows:
                pid = r["player_id"]
                if pid in set(actual[:60]) - set(market[:60]):
                    misses.append(
                        {
                            "model": model,
                            "season": year,
                            "player_id": pid,
                            "name": r["player_display_name"],
                            "position": r["position"],
                            "prior_games": r["games"],
                            "recovered": pid in order[:60],
                            "modeled": (pid, year) in lookup,
                        }
                    )
        summaries[model] = {
            **summarize(folds, 60),
            "missed_cohorts": {
                label: {"misses": len(group), "recovered": sum(r["recovered"] for r in group)}
                for label, group in {
                    "prior_under10": [
                        r for r in misses if r["model"] == model and r["prior_games"] < 10
                    ],
                    "RB": [r for r in misses if r["model"] == model and r["position"] == "RB"],
                }.items()
            },
        }
    return summaries, misses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history, output = args.history.resolve(), args.output.resolve()
    provenance = require_audited_version(history)
    if output.exists() or not output.is_relative_to(ROOT / "data/research"):
        raise ValueError("Output must be a new directory under data/research")
    index = {
        p: set(pl.scan_parquet(p).collect_schema())
        for p in (history / "cache/nflverse").glob("*.parquet")
    }
    sources_used = []

    def sources(required):
        paths = sorted(p for p, cols in index.items() if set(required) <= cols)
        if not paths:
            raise ValueError(f"Missing frozen sources: {required}")
        sources_used.extend(paths)
        return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")

    players = sources(["gsis_id", "pfr_id", "rookie_season"])
    ids = players.select(pl.col("gsis_id").alias("player_id"), "pfr_id").drop_nulls()
    unique(ids, ["pfr_id"])
    raw_snaps = canonical_positions(sources(["pfr_player_id", "offense_snaps", "offense_pct"]))
    raw_snaps = raw_snaps.filter(
        (pl.col("game_type") == "REG") & pl.col("season").is_between(2013, 2025)
    )
    mapped = raw_snaps.join(
        ids, left_on="pfr_player_id", right_on="pfr_id", how="inner", validate="m:1"
    )
    snaps = mapped.filter(pl.col("position").is_in([*POSITIONS, "WR"]))
    schedule = sources(["game_id", "home_team", "away_team", "gameday", "game_type"])
    schedule = schedule.filter(
        (pl.col("game_type") == "REG") & pl.col("season").is_between(2013, 2025)
    )
    raw = sources(["player_id", "fantasy_points_ppr", "sack_fumbles_lost"])
    attempts = raw.filter(
        (pl.col("season_type") == "REG") & pl.col("season").is_between(2013, 2025)
    ).select("player_id", "season", "week", "attempts")
    weekly_path = history / "outputs/research_audited_weekly_points.parquet"
    pred_path = history / "outputs/metric_backtest_predictions.parquet"
    sources_used.extend([weekly_path, pred_path])
    points = pl.read_parquet(weekly_path).filter(pl.col("season").is_between(2013, 2025))
    unidentified_points = points["player_id"].null_count()
    points = points.filter(pl.col("player_id").is_not_null())
    points = points.join(attempts, on=["player_id", "season", "week"], how="left", validate="1:1")
    if points["attempts"].null_count():
        raise ValueError("Missing attempts join")
    expected, observed, scored = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in schedule.to_dicts():
        for team in (r["home_team"], r["away_team"]):
            expected[r["season"]].add((r["week"], team_name(team)))
    for r in raw_snaps.select("season", "week", "team").unique().to_dicts():
        observed[r["season"]].add((r["week"], team_name(r["team"])))
    for r in points.select("season", "week", "team").unique().to_dicts():
        scored[r["season"]].add((r["week"], team_name(r["team"])))
    if any(not teams <= scored[year] for year, teams in expected.items()):
        raise ValueError("Incomplete scoring team/week coverage")
    complete = {year for year, teams in expected.items() if teams <= observed[year]}
    cols = [
        "player_id",
        "forecast_season",
        "player_display_name",
        "position",
        "player_population",
        "ppg",
        "games",
        "age_at_season",
        "player_experience",
        "outcome_complete",
        "actual_ppg",
        "actual_games",
        "actual_season_points",
        "actual_availability_value",
        "fitted_ppg",
        "fitted_games",
        "fitted_season_points",
        "market_overall_ecr_score",
        "forecast_cutoff_date",
    ]
    candidates = pl.read_parquet(pred_path, columns=cols).with_columns(
        pl.col("forecast_cutoff_date").cast(pl.String)
    )
    protected_names = json.loads((history / "manifest.json").read_text())[
        "protected_artifact_sha256"
    ]
    protected = {name: digest(ROOT / name) for name in protected_names}
    rows = build_rows(candidates, points, snaps, schedule, set(ids["player_id"]), complete)
    predictions, folds = predict(rows)
    market, misses = market_evaluation(predictions, candidates.to_dicts())
    # Reconcile this run's retained-baseline comparison against the user's repaired audit.
    audit_path = history / "outputs/value_research_2026-09-22.json"
    sources_used.append(audit_path)
    audited = json.loads(audit_path.read_text())["details"]["fitted_season_points:all"]
    if market["saved_v2"]["calls"] != audited["modern_cohorts"]["population:returner"]["calls"]:
        raise ValueError("Baseline pool does not reproduce the repaired audit")
    for label, audit_key in (("RB", "position:RB"), ("prior_under10", "prior_sample:under_10")):
        reference = audited["modern_missed_cohorts"][audit_key]
        if market["saved_v2"]["missed_cohorts"][label] != {
            "misses": reference["market_misses"],
            "recovered": reference["model_recovered"],
        }:
            raise ValueError("Baseline miss cohorts do not reproduce the repaired audit")
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "history": provenance,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(sources_used))},
        "implementation_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                ROOT / "src/engine/metrics/outlook.py",
                ROOT / "src/engine/metrics/positions.py",
                ROOT / "research/value_capture_audit.py",
                ROOT / "research/artifact_inputs.py",
                ROOT / "research/data_integrity_audit.py",
            )
        },
        "design": {
            "test_seasons": [2019, 2025],
            "training_start": 2014,
            "ridge_penalty": 10,
            "specifications": SPECS,
            "workload_targets": WORKLOADS,
            "point_target": "season points, including nonappearances",
            "role_thresholds": {
                "QB": "at least 50% offensive snaps",
                "RB": "at least 50% snaps and 12 carries + targets",
                "TE": "at least 4 targets",
            },
            "late_window": "last four scheduled games of last prior-season observed team",
            "market_pool": "original all-player ECR pool; V2 fallback outside modeled rows",
            "modeling": "separate points/workload regressions; no forecast multiplication",
        },
        "coverage": {
            "feature_rows": len(rows),
            "eligible_rows": sum(r["eligible"] for r in rows),
            "test_rows": len(predictions),
            "complete_snap_seasons": sorted(complete),
            "unmapped_raw_snap_rows": raw_snaps.height - mapped.height,
            "unidentified_weekly_point_rows_excluded": unidentified_points,
        },
        "folds": folds,
        "diagnostics": diagnostics(predictions),
        "market": market,
        "lamar_2019": next(
            r for r in predictions if r["player_id"] == "00-0034796" and r["season"] == 2019
        ),
        "limitations": [
            "Exploratory reused seasons, not an independent confirmation or prospective edge.",
            "Roles are prior usage proxies, not verified future jobs or causal effects.",
            "TE targets do not distinguish routes from blocking; on-field snaps are not routes.",
            "No offseason role evidence: a new job with no prior usage signal can remain missed.",
            "Cached sources may include later revisions; historical vintages are unavailable.",
            "Team/week feed completeness does not certify every individual player row.",
            "Last prior team anchors the late window; trade and absence context is not modeled.",
            "Returners only; rookies and uncovered identities retain V2 in market comparisons.",
            "Seven seasons and small subgroups limit intervals; no multiplicity correction.",
            "Top-60 swaps are not a legal roster simulation or executable acquisition profit.",
        ],
        "protected_unchanged": all(digest(ROOT / name) == sha for name, sha in protected.items()),
        "protected_artifact_sha256": protected,
    }
    if not report["protected_unchanged"]:
        raise ValueError("Protected artifacts changed during the research run")
    json.dumps(report, allow_nan=False)  # Validate before creating any output files.
    output.mkdir(parents=True, exist_ok=False)
    implementation = output / "implementation"
    for name in report["implementation_sha256"]:
        target = implementation / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(output / "features.parquet")
    pl.DataFrame(predictions, infer_schema_length=None).write_parquet(
        output / "predictions.parquet"
    )
    pl.DataFrame(misses).write_parquet(output / "market_misses.parquet")
    report["output_sha256"] = {p.name: digest(p) for p in sorted(output.glob("*.parquet"))}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "coverage": report["coverage"],
                "market": {
                    k: {n: v[n] for n in ("calls", "hits", "net_value_per_fold", "missed_cohorts")}
                    for k, v in market.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
