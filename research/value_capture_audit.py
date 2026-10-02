"""Research-only audit of retained forecasts against consensus and observed ADP.

Run: .venv/bin/python research/value_capture_audit.py
No forecasts are fitted or promoted. The 2026 forecast is excluded throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from artifact_inputs import require_audited_version

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/outputs/metric_backtest_predictions.parquet"
ADP = ROOT / "data/static/market_adp_backfill.csv"
OUTPUT = ROOT / "data/outputs/value_research_2026-09-22.json"
MODELS = (
    "fitted_season_points",
    "fitted_adaptive_ppg_hybrid",
    "fitted_nextgen_season_points",
    "fitted_market_availability_returner_season_points",
)
TARGET = "actual_availability_value"
REPLACEMENT = {"QB": 12, "RB": 25, "WR": 35, "TE": 12}
WINDOWS = {"expanded": (2011, 2025), "modern": (2019, 2025)}


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def rank_ids(rows, scores):
    return [
        row["player_id"]
        for row in sorted(rows, key=lambda r: (-scores[r["player_id"]], r["player_id"]))
    ]


def model_values(rows, model):
    """Same positional replacement convention as the retained overall backtest.

    Every chosen model here is on a season-points scale, including the Adaptive
    selector whose name does not end in season_points. Constant scale does not
    affect overall ranking, but explicit units make interpretation unambiguous.
    """
    values = {}
    for position, rank in REPLACEMENT.items():
        group = [r for r in rows if r["position"] == position]
        eligible = sorted(
            [r[model] / 17 for r in group if (r.get("games") or 0) >= 8], reverse=True
        )
        replacement = eligible[min(rank, len(eligible)) - 1] if eligible else 0.0
        for row in group:
            values[row["player_id"]] = row[model] / 17 - replacement
    return values


def score(rows, order, market_order, k):
    by_id = {r["player_id"]: r for r in rows}
    actual = rank_ids(rows, {i: r[TARGET] for i, r in by_id.items()})
    chosen, market, ideal = set(order[:k]), set(market_order[:k]), set(actual[:k])
    bought, rejected = chosen - market, market - chosen

    def actual_sum(ids):
        return sum(by_id[i][TARGET] for i in ids)

    return {
        "players": len(rows),
        "calls": len(bought),
        "hits": len(bought & ideal),
        "market_alternative_hits": len(rejected & ideal),
        "market_misses": len(ideal - market),
        "top_k_hits": len(chosen & ideal),
        "market_top_k_hits": len(market & ideal),
        "net_value": actual_sum(bought) - actual_sum(rejected),
        "selected_value": actual_sum(chosen),
        "market_selected_value": actual_sum(market),
    }


def summarize(folds, k):
    if not folds:
        return None

    def total(key):
        return sum(f[key] for f in folds)

    differences = np.array([f["net_value"] for f in folds])
    rng = np.random.default_rng(20260922)
    boot = rng.choice(differences, size=(10000, len(folds)), replace=True).mean(axis=1)
    calls = total("calls")
    return {
        "seasons": [f["season"] for f in folds],
        "folds": len(folds),
        "common_players_mean": total("players") / len(folds),
        "calls": calls,
        "hits": total("hits"),
        "precision": total("hits") / calls if calls else None,
        "market_alternative_hits": total("market_alternative_hits"),
        "market_alternative_precision": total("market_alternative_hits") / calls if calls else None,
        "market_misses": total("market_misses"),
        "recovered_market_misses": total("hits") / total("market_misses")
        if total("market_misses")
        else None,
        "hit_rate": total("top_k_hits") / (len(folds) * k),
        "market_hit_rate": total("market_top_k_hits") / (len(folds) * k),
        "net_value_per_fold": float(differences.mean()),
        "net_value_fold_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "net_positive_folds": int((differences > 1e-9).sum()),
        "net_negative_folds": int((differences < -1e-9).sum()),
        "fold_results": folds,
    }


def row_groups(row):
    experience = row.get("player_experience")
    groups = ["position:" + row["position"], "population:" + row["player_population"]]
    if row["player_population"] == "rookie":
        groups.append("career:rookie")
    elif finite(experience):
        groups.append("career:years_2_3" if experience <= 2 else "career:year_4_plus")
    games = row.get("games")
    if finite(games):
        groups.append("prior_sample:under_10" if games < 10 else "prior_sample:10_plus")
    if finite(row.get("team_changed")):
        groups.append("team:moved" if row["team_changed"] else "team:same")
    market_rank = row["market_rank"]
    groups.append(
        "market:61_100"
        if market_rank <= 100
        else "market:101_150"
        if market_rank <= 150
        else "market:151_plus"
    )
    return groups


def cohort_summary(calls):
    grouped = defaultdict(list)
    for row in calls:
        for group in row_groups(row):
            grouped[group].append(row)
    return {
        group: {
            "calls": len(rows),
            "hits": sum(r["hit"] for r in rows),
            "precision": sum(r["hit"] for r in rows) / len(rows),
            "mean_actual_value": sum(r[TARGET] for r in rows) / len(rows),
            "seasons": len({r["forecast_season"] for r in rows}),
        }
        for group, rows in sorted(grouped.items())
    }


def missed_cohort_summary(misses):
    grouped = defaultdict(list)
    for row in misses:
        for group in row_groups(row):
            grouped[group].append(row)
    return {
        group: {
            "market_misses": len(rows),
            "model_recovered": sum(r["recovered"] for r in rows),
            "recovery_rate": sum(r["recovered"] for r in rows) / len(rows),
        }
        for group, rows in sorted(grouped.items())
    }


def main():
    global INPUT, ADP, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    provenance = require_audited_version(args.data_dir)
    INPUT = args.data_dir / "outputs/metric_backtest_predictions.parquet"
    ADP = args.data_dir / "static/market_adp_backfill.csv"
    OUTPUT = args.data_dir / "outputs/value_research_2026-09-22.json"
    input_digest = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    adp_digest = hashlib.sha256(ADP.read_bytes()).hexdigest()
    names = pl.scan_parquet(INPUT).collect_schema().names()
    descriptive = [
        "forecast_season",
        "player_id",
        "player_display_name",
        "position",
        "player_population",
        "player_experience",
        "age_at_season",
        "games",
        "team_changed",
        "preseason_reserve",
        "depth_chart_rank",
        "market_snapshot",
        "market_overall_snapshot",
        "market_overall_ecr_score",
        "actual_ppg",
        "actual_games",
        "actual_season_points",
        TARGET,
        "outcome_complete",
    ]
    frame = pl.read_parquet(INPUT, columns=[c for c in [*descriptive, *MODELS] if c in names])
    frame = frame.filter(
        pl.col("outcome_complete") & pl.col("forecast_season").is_between(2011, 2025)
    )
    adp = pl.read_csv(ADP).filter(pl.col("preferred") & pl.col("gsis_id").is_not_null())
    adp = adp.select(
        "forecast_season", pl.col("gsis_id").alias("player_id"), "adp", "source", "window"
    )
    assert not adp.select(pl.struct("forecast_season", "player_id").is_duplicated().any()).item()
    frame = frame.join(adp, on=["forecast_season", "player_id"], how="left")
    rows = frame.to_dicts()
    assert all(r["forecast_season"] <= 2025 for r in rows)
    assert not frame.select(pl.struct("forecast_season", "player_id").is_duplicated().any()).item()
    experiments, details = [], {}
    for price in ("ecr", "adp"):
        price_key = "market_overall_ecr_score" if price == "ecr" else "adp"
        for model in MODELS:
            for population in ("all", "returner"):
                for k in (24, 60, 120):
                    folds = defaultdict(list)
                    calls, misses, coverage = [], [], []
                    for season in range(2011, 2026):
                        universe = [
                            r
                            for r in rows
                            if r["forecast_season"] == season
                            and (population == "all" or r["player_population"] == population)
                        ]
                        common = [
                            r
                            for r in universe
                            if finite(r.get(model))
                            and finite(r.get(price_key))
                            and finite(r.get(TARGET))
                        ]
                        if len(common) < k:
                            continue
                        market_scores = {
                            r["player_id"]: r[price_key] * (1 if price == "ecr" else -1)
                            for r in common
                        }
                        market_order = rank_ids(common, market_scores)
                        values = model_values(common, model)
                        model_order = rank_ids(common, values)
                        market_rank = {i: n for n, i in enumerate(market_order, 1)}
                        model_rank = {i: n for n, i in enumerate(model_order, 1)}
                        policies = {"model": model_order}
                        for weight in (0.1, 0.2):
                            blend = {
                                i: -((1 - weight) * market_rank[i] + weight * model_rank[i])
                                for i in model_rank
                            }
                            policies[
                                f"market_{round(100 * (1 - weight))}_model_{round(100 * weight)}"
                            ] = rank_ids(common, blend)
                        for policy, order in policies.items():
                            folds[policy].append(
                                {"season": season, **score(common, order, market_order, k)}
                            )
                        if k != 60 or price != "ecr":
                            continue
                        actual_order = rank_ids(common, {r["player_id"]: r[TARGET] for r in common})
                        actual_rank = {i: n for n, i in enumerate(actual_order, 1)}
                        chosen, market, ideal = (
                            set(model_order[:k]),
                            set(market_order[:k]),
                            set(actual_order[:k]),
                        )
                        common_ids = set(model_order)
                        true_top = set(
                            rank_ids(universe, {r["player_id"]: r[TARGET] for r in universe})[:k]
                        )
                        coverage.append(
                            {
                                "season": season,
                                "universe": len(universe),
                                "common": len(common),
                                "actual_top_missing_from_common": len(true_top - common_ids),
                            }
                        )
                        for row in common:
                            i = row["player_id"]
                            augmented = {
                                **row,
                                "model_rank": model_rank[i],
                                "market_rank": market_rank[i],
                                "actual_rank": actual_rank[i],
                                "hit": i in ideal,
                                "rank_gap": market_rank[i] - model_rank[i],
                            }
                            if i in chosen - market:
                                calls.append(augmented)
                            if i in ideal - market:
                                misses.append({**augmented, "recovered": i in chosen})
                    for policy, results in folds.items():
                        for window, (start, end) in WINDOWS.items():
                            summary = summarize(
                                [f for f in results if start <= f["season"] <= end], k
                            )
                            if summary:
                                experiments.append(
                                    {
                                        "price": price,
                                        "model": model,
                                        "population": population,
                                        "k": k,
                                        "policy": policy,
                                        "window": window,
                                        **summary,
                                    }
                                )
                    if calls:
                        modern_calls = [r for r in calls if r["forecast_season"] >= 2019]
                        modern_misses = [r for r in misses if r["forecast_season"] >= 2019]
                        details[f"{model}:{population}"] = {
                            "coverage": coverage,
                            "expanded_cohorts": cohort_summary(calls),
                            "modern_cohorts": cohort_summary(modern_calls),
                            "modern_missed_cohorts": missed_cohort_summary(modern_misses),
                            "gap_precision": [
                                {
                                    "gap": gap,
                                    "calls": len(selected),
                                    "hits": sum(r["hit"] for r in selected),
                                    "precision": sum(r["hit"] for r in selected) / len(selected)
                                    if selected
                                    else None,
                                }
                                for gap in (0, 20, 60, 120)
                                for selected in [[r for r in calls if r["rank_gap"] >= gap]]
                            ],
                            "captured_examples": sorted(
                                [r for r in modern_calls if r["hit"]], key=lambda r: -r[TARGET]
                            )[:12],
                            "missed_examples": sorted(
                                [r for r in modern_misses if not r["recovered"]],
                                key=lambda r: -r[TARGET],
                            )[:16],
                            "false_positive_examples": sorted(
                                [r for r in modern_calls if not r["hit"]],
                                key=lambda r: r["model_rank"],
                            )[:12],
                            "market_misses_by_position": {
                                pos: {
                                    "misses": len(rs),
                                    "recovered": sum(r["recovered"] for r in rs),
                                }
                                for pos in REPLACEMENT
                                for rs in [[r for r in modern_misses if r["position"] == pos]]
                            },
                        }
    consensus_vs_adp = []
    for population in ("all", "returner"):
        for k in (24, 60, 120):
            price_folds = []
            for season in range(2011, 2026):
                common = [
                    r
                    for r in rows
                    if r["forecast_season"] == season
                    and (population == "all" or r["player_population"] == population)
                    and finite(r.get("market_overall_ecr_score"))
                    and finite(r.get("adp"))
                ]
                if len(common) < k:
                    continue
                ecr_order = rank_ids(
                    common, {r["player_id"]: r["market_overall_ecr_score"] for r in common}
                )
                adp_order = rank_ids(common, {r["player_id"]: -r["adp"] for r in common})
                price_folds.append({"season": season, **score(common, ecr_order, adp_order, k)})
            for window, (start, end) in WINDOWS.items():
                summary = summarize([f for f in price_folds if start <= f["season"] <= end], k)
                if summary:
                    consensus_vs_adp.append(
                        {"population": population, "k": k, "window": window, **summary}
                    )
    assert input_digest == hashlib.sha256(INPUT.read_bytes()).hexdigest(), (
        "Input changed during audit"
    )
    assert adp_digest == hashlib.sha256(ADP.read_bytes()).hexdigest(), "ADP changed during audit"
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audited_input": provenance,
        "input_sha256": input_digest,
        "adp_sha256": adp_digest,
        "target": TARGET,
        "target_units": "positive actual per-active-game VOR multiplied by actual games / 17",
        "limitations": [
            "Separately rebuilt historical forecasts are reused after integrity acceptance; "
            "no live-board promotion. Production fits are returner-only.",
            "Every policy is compared with its price baseline on exactly the same players; "
            "different model pairs may have different coverage.",
            "ADP is final preseason MFL PPR behavior, not an exact-date or ESPN league-matched "
            "price; it can contain later news than ECR and model cutoffs.",
            "Rank blend weights 10% and 20% were fixed before this run; this remains "
            "exploratory reuse of previously inspected historical seasons.",
            "Cohorts and examples explain errors; subgroup selection on these outcomes "
            "is not independent validation.",
            "Top-K selection is a diagnostic, not a legal snake-draft, lineup, "
            "or championship simulation.",
            "Bootstrap intervals resample seasons; they do not correct for model search "
            "or all dependence between eras.",
            "Full target population is limited to saved forecast rows; players absent "
            "from that table cannot be measured here.",
        ],
        "experiments": experiments,
        "consensus_vs_adp": consensus_vs_adp,
        "details": details,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
    print(f"Wrote {OUTPUT}: {len(experiments)} comparisons")
    for entry in experiments:
        if (
            entry["price"] == "ecr"
            and entry["population"] == "returner"
            and entry["k"] == 60
            and entry["window"] == "modern"
        ):
            print(
                entry["model"],
                entry["policy"],
                "calls",
                entry["calls"],
                "hits",
                entry["hits"],
                "alternatives",
                entry["market_alternative_hits"],
                "net",
                round(entry["net_value_per_fold"], 3),
                "CI",
                [round(x, 3) for x in entry["net_value_fold_bootstrap_95"]],
            )


if __name__ == "__main__":
    main()
