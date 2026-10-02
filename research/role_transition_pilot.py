"""Exploratory RB watchlist test using audited scoring, raw snaps, and preseason ADP.

Run data_integrity_audit.py first, then this script. Writes research artifacts only.
This is NOT a waiver backtest: historical ownership/FAAB and live data vintages are
unavailable. Preseason ADP is known by every evaluated in-season decision date.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import data_integrity_audit
import numpy as np
import polars as pl
from artifact_inputs import require_audited_version
from data_integrity_audit import ROOT, catalog, digest, read_sources, select_sources

from patron.metrics.positions import canonical_positions

BASE = ["last3_points", "season_points_pg", "observed_games", "log_adp"]
USAGE = [*BASE, "snap_last2", "last3_carries", "last3_targets"]
ROLE = [
    *BASE,
    "snap_last2",
    "snap_change",
    "snap_observations",
    "last3_carries",
    "last3_targets",
    "prior_season_games",
    "small_prior_sample",
]
TARGET = "next4_points"
POLICIES = [
    "trailing_points",
    "preseason_adp",
    "recent_snaps",
    "trailing_opportunity",
    "price_production_ridge",
    "usage_ridge",
    "role_ridge",
]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def fit_predict(train, test, features, target=TARGET):
    """Fixed ridge penalty; all scaling and coefficients learned on earlier seasons."""
    x = np.array([[r[f] for f in features] for r in train], dtype=float)
    xt = np.array([[r[f] for f in features] for r in test], dtype=float)
    y = np.array([r[target] for r in train], dtype=float)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - center) / scale
    xt = (xt - center) / scale
    coef = np.linalg.solve(x.T @ x + 10.0 * np.eye(x.shape[1]), x.T @ (y - y.mean()))
    return xt @ coef + y.mean()


def make_rows(points, snaps, adp):
    point_map = {
        (r["player_id"], r["season"], r["week"]): r
        # The snap feed defines the RB decision pool. Once a player is selected,
        # a subsequent roster/position label must not erase earned outcome points.
        for r in points.to_dicts()
    }
    prior_games = defaultdict(int)
    for player_id, season, _ in point_map:
        prior_games[(player_id, season)] += 1
    price_map = {(r["gsis_id"], r["forecast_season"]): r["adp"] for r in adp.to_dicts()}
    grouped = defaultdict(dict)
    for r in snaps.to_dicts():
        grouped[(r["player_id"], r["season"])][r["week"]] = r
    rows = []
    for (player_id, season), observed in sorted(grouped.items()):
        price = price_map.get((player_id, season))
        for week in range(3, 14):
            now = observed.get(week)
            if not now or now["offense_snaps"] <= 0:
                continue
            past = range(max(1, week - 2), week + 1)
            future = range(week + 1, week + 5)

            def stat(w, field, player_id=player_id, season=season):
                return float(point_map.get((player_id, season, w), {}).get(field) or 0)

            recent = [observed[w]["offense_pct"] for w in (week - 1, week) if w in observed]
            earlier = [
                observed[w]["offense_pct"]
                for w in range(max(1, week - 4), week - 1)
                if w in observed
            ]
            previous_games = prior_games[(player_id, season - 1)]
            next_points = [stat(w, "league_points") for w in future]
            rows.append(
                {
                    "player_id": player_id,
                    "season": season,
                    "week": week,
                    "name": now["player"],
                    "adp": price,
                    "last3_points": mean([stat(w, "league_points") for w in past]),
                    "season_points_pg": mean(
                        [stat(w, "league_points") for w in range(1, week + 1)]
                    ),
                    "observed_games": sum(
                        w <= week and observed[w]["offense_snaps"] > 0 for w in observed
                    ),
                    "log_adp": math.log(max(1.0, price or 300.0)),
                    "snap_last2": mean(recent),
                    "snap_change": mean(recent) - mean(earlier) if earlier else 0.0,
                    "snap_observations": len(recent) + len(earlier),
                    "last3_carries": mean([stat(w, "carries") for w in past]),
                    "last3_targets": mean([stat(w, "targets") for w in past]),
                    "prior_season_games": previous_games,
                    "small_prior_sample": float(1 <= previous_games < 10),
                    "next4_points": sum(next_points),
                    "next4_surplus12": sum(max(v - 12.0, 0.0) for v in next_points),
                    "next4_12point_weeks": sum(v >= 12.0 for v in next_points),
                }
            )
    return rows


def summarize(folds, baseline_folds):
    reference = {r["season"]: r for r in baseline_folds}
    result = {"seasons": len(folds), "fold_results": folds}
    for outcome in ["next4_points", "next4_surplus12", "next4_12point_weeks"]:
        diffs = np.array([r[outcome] - reference[r["season"]][outcome] for r in folds])
        rng = np.random.default_rng(20260922)
        boot = rng.choice(diffs, (10000, len(diffs)), replace=True).mean(axis=1)
        result[outcome] = {
            "mean_per_pick": mean([r[outcome] for r in folds]),
            "lift_vs_trailing_points": float(diffs.mean()),
            "season_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "positive_seasons": int((diffs > 1e-9).sum()),
        }
    return result


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    provenance = require_audited_version(args.data_dir)
    OUT = args.data_dir / "outputs"
    data_integrity_audit.CACHE = args.data_dir / "cache/nflverse"
    index = catalog()
    weekly_path = OUT / "research_audited_weekly_points.parquet"
    points = pl.read_parquet(weekly_path)
    assert points["season"].max() <= 2025
    adp_path = args.data_dir / "static/market_adp_backfill.csv"
    identity_paths = select_sources(index, ["gsis_id", "display_name", "short_name"])
    snap_paths = select_sources(index, ["pfr_player_id", "offense_snaps", "offense_pct"])
    source_files = [weekly_path, adp_path, *identity_paths, *snap_paths]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    players = read_sources(identity_paths)
    snaps = canonical_positions(read_sources(snap_paths))
    snaps = snaps.filter(
        (pl.col("game_type") == "REG")
        & (pl.col("position") == "RB")
        & pl.col("season").is_between(2013, 2025)
    ).join(
        players.select(pl.col("gsis_id").alias("player_id"), "pfr_id").drop_nulls(),
        left_on="pfr_player_id",
        right_on="pfr_id",
        how="inner",
    )
    assert not snaps.select(pl.struct("player_id", "season", "week").is_duplicated().any()).item()
    assert snaps.filter(~pl.col("offense_pct").is_between(0, 1)).height == 0
    adp = pl.read_csv(adp_path).filter(pl.col("preferred") & pl.col("gsis_id").is_not_null())
    rows = make_rows(points, snaps, adp)
    results, model_diagnostics = [], []
    for season in range(2019, 2026):
        train = [r for r in rows if r["season"] < season]
        test = [r.copy() for r in rows if r["season"] == season]
        assert max(r["season"] for r in train) < season
        for label, features in [
            ("price_production_ridge", BASE),
            ("usage_ridge", USAGE),
            ("role_ridge", ROLE),
        ]:
            predicted = fit_predict(train, test, features)
            for row, value in zip(test, predicted, strict=True):
                row[label] = float(value)
        for r in test:
            r["trailing_points"] = r["last3_points"]
            r["preseason_adp"] = -(r["adp"] or 300)
            r["recent_snaps"] = r["snap_last2"]
            r["trailing_opportunity"] = r["last3_carries"] + 2.0 * r["last3_targets"]
        results.extend(test)
        model_diagnostics.append(
            {
                "season": season,
                "train_rows": len(train),
                "test_rows": len(test),
                "latest_train_season": max(r["season"] for r in train),
            }
        )
    summaries, selections, pool_sizes = [], [], []
    for cohort in ["all_late_price", "small_prior_sample"]:
        folds = {policy: [] for policy in POLICIES}
        for season in range(2019, 2026):
            picks = {policy: [] for policy in POLICIES}
            windows = 0
            for week in range(3, 14):
                eligible = [
                    r
                    for r in results
                    if r["season"] == season
                    and r["week"] == week
                    and r["adp"] is not None
                    and 100 < r["adp"] <= 300
                    and r["last3_points"] < 12
                    and (cohort != "small_prior_sample" or r["small_prior_sample"])
                ]
                # Identical candidate pool and number of selections for every policy.
                if len(eligible) < 3:
                    continue
                pool_sizes.append(
                    {
                        "cohort": cohort,
                        "season": season,
                        "week": week,
                        "eligible": len(eligible),
                    }
                )
                windows += 1
                for policy in POLICIES:
                    chosen = sorted(eligible, key=lambda r: (-r[policy], r["player_id"]))[:3]
                    picks[policy].extend(chosen)
                    selections.extend({"cohort": cohort, "policy": policy, **r} for r in chosen)
            for policy, chosen in picks.items():
                if chosen:
                    folds[policy].append(
                        {
                            "season": season,
                            "windows": windows,
                            "picks": len(chosen),
                            **{
                                outcome: mean([r[outcome] for r in chosen])
                                for outcome in [
                                    "next4_points",
                                    "next4_surplus12",
                                    "next4_12point_weeks",
                                ]
                            },
                        }
                    )
        for policy in POLICIES:
            if folds[policy]:
                summaries.append(
                    {
                        "cohort": cohort,
                        "policy": policy,
                        **summarize(folds[policy], folds["trailing_points"]),
                    }
                )
    # Custom-scoring magnitude, not a prediction of next year's bonuses.
    bonus_scale = (
        points.filter(
            pl.col("position").is_in(["QB", "RB", "WR", "TE"])
            & pl.col("season").is_between(2019, 2025)
        )
        .group_by("player_id", "season", "position")
        .agg(
            pl.len().alias("stat_games"),
            pl.col("bonus_pts").fill_null(0).sum().alias("bonuses"),
            pl.col("league_points").sum(),
        )
        .filter(pl.col("stat_games") >= 8)
        .with_columns((pl.col("bonuses") / pl.col("stat_games")).alias("bonus_ppg"))
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "implementation_sha256": digest(Path(__file__)),
        "audited_input": provenance,
        "source_sha256": source_hashes,
        "design": {
            "test_seasons": [2019, 2025],
            "decision_weeks": [3, 13],
            "horizon": "next four calendar weeks; byes and absences count zero",
            "candidate_pool": "RBs with offensive snaps in decision week, "
            "ADP >100 and <=300, trailing PPG <12",
            "small_sample": "1–9 raw box-score appearances at any position "
            "in previous season; not rookies",
            "primary_target": TARGET,
            "picks_per_decision": 3,
            "ridge_penalty": 10.0,
            "baseline_features": BASE,
            "usage_features": USAGE,
            "role_features": ROLE,
        },
        "limitations": [
            "Exploratory historical proxy experiment, not a prospective validation or promotion.",
            "Preseason ADP is a known cost stratum, "
            "not the current waiver/FAAB price or ownership.",
            "Final revised historical stats/snaps lack archived publication-time vintages.",
            "No contract, transaction, route, injury, current-year roster, "
            "or xFP features are used.",
            "Overlapping windows and repeated picks are dependent; uncertainty resamples seasons.",
            "Surplus over 12 is a fixed diagnostic, "
            "not best-free-agent replacement or legal lineup gain.",
            "Candidate eligibility requires observed current-week offensive snaps; "
            "inactive players omitted.",
            "Usage ablation and simple snap/opportunity controls added after the first "
            "role-versus-points comparison to test whether role change adds beyond usage level.",
        ],
        "model_diagnostics": model_diagnostics,
        "summaries": summaries,
        "candidate_pools": pool_sizes,
        "bonus_magnitude": bonus_scale.group_by("position")
        .agg(
            pl.len().alias("player_seasons"),
            pl.col("bonus_ppg").median().alias("median"),
            pl.col("bonus_ppg").quantile(0.9).alias("p90"),
            pl.col("bonus_ppg").max().alias("max"),
        )
        .sort("position")
        .to_dicts(),
    }
    comparisons = []
    for cohort in ["all_late_price", "small_prior_sample"]:
        group = {r["policy"]: r for r in summaries if r["cohort"] == cohort}
        for reference in ["recent_snaps", "usage_ridge"]:
            for outcome in ["next4_points", "next4_surplus12", "next4_12point_weeks"]:
                diffs = np.array(
                    [
                        a[outcome] - b[outcome]
                        for a, b in zip(
                            group["role_ridge"]["fold_results"],
                            group[reference]["fold_results"],
                            strict=True,
                        )
                    ]
                )
                boot = (
                    np.random.default_rng(20260922)
                    .choice(
                        diffs,
                        (10000, len(diffs)),
                        replace=True,
                    )
                    .mean(axis=1)
                )
                comparisons.append(
                    {
                        "cohort": cohort,
                        "reference": reference,
                        "outcome": outcome,
                        "role_lift": float(diffs.mean()),
                        "season_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
                    }
                )
    report["stronger_baseline_comparisons"] = comparisons
    assert source_hashes == {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    (OUT / "role_transition_pilot_2026-09-22.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    pl.DataFrame(results).write_parquet(OUT / "research_rb_role_predictions.parquet")
    pl.DataFrame(selections).write_parquet(OUT / "research_rb_role_selections.parquet")
    for row in summaries:
        print(row["cohort"], row["policy"], row["next4_points"])


if __name__ == "__main__":
    main()
