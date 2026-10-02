"""Matched feature, strength, receiving execution, and QB role ablations."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import polars as pl
from data import FAMILIES, RATE_PAIRS, finite
from followup_common import CONFIG, REPO, columns, predict, record, save_points
from methods import fit_pool
from safety import write_json

NGS = {
    "QB": [("passing", "completion_percentage_above_expectation", "attempts")],
    "RB": [("rushing", "rush_yards_over_expected_per_att", "rush_attempts")],
    "WR": [
        ("receiving", "avg_yac_above_expectation", "receptions"),
        ("receiving", "avg_separation", "targets"),
    ],
    "TE": [
        ("receiving", "avg_yac_above_expectation", "receptions"),
        ("receiving", "avg_separation", "targets"),
    ],
}
EXECUTION = ["e_catch_fraction", "e_yac_per_reception", "e_air_per_target"]
DURATION = ["d_prior_starts", "d_late_starts", "d_designed_carries", "d_scramble_carries"]
NEWS = ["d_news_observed", "d_news_starter", "d_news_competition", "d_news_backup"]


def normal_team(team):
    return {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX"}.get(team, team)


def schedule_labels(schedule):
    first, starts, missing = {}, defaultdict(list), defaultdict(int)
    games = schedule.filter(pl.col("game_type") == "REG").sort("gameday").to_dicts()
    for game in games:
        for side in ("home", "away"):
            key = (game["season"], normal_team(game[f"{side}_team"]))
            pid = game[f"{side}_qb_id"]
            if key not in first:
                first[key] = pid
                if pid is None:
                    missing[game["season"]] += 1
            if pid:
                starts[pid, game["season"]].append(game["week"])
    opening = defaultdict(set)
    for (year, _), pid in first.items():
        if pid:
            opening[year].add(pid)
    return opening, starts, dict(missing)


def dated_news(ledger, pid, year, cutoff):
    known = [
        r
        for r in ledger
        if r["player_id"] == pid
        and int(r["season"]) == year
        and r["published_on"] < cutoff
        and (not r["event_on"] or r["event_on"] < cutoff)
    ]
    if not known:
        return dict(
            d_news_observed=0.0, d_news_starter=None, d_news_competition=None, d_news_backup=None
        )
    latest = max(r["published_on"] for r in known)
    states = {r["state"] for r in known if r["published_on"] == latest}
    if len(states) != 1:
        return dict(
            d_news_observed=0.0, d_news_starter=None, d_news_competition=None, d_news_backup=None
        )
    state = states.pop()
    return {
        "d_news_observed": 1.0,
        **{f"d_news_{s}": float(state == s) for s in ("starter", "competition", "backup")},
    }


def complete_sum(values):
    return sum(values) if values and all(finite(v) is not None for v in values) else None


def ratio(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def prepare(inputs, *, archive=True):
    rows = inputs.panel()
    fields = [
        "player_id",
        "forecast_season",
        "forecast_cutoff_date",
        "source_team",
        "designed_carries",
        "scramble_carries",
        "market_ecr",
        "market_ecr_sd",
        "market_overall_ecr",
        "market_overall_ecr_sd",
    ]
    originals = {
        (r["player_id"], r["forecast_season"]): r
        for r in inputs.gold("preseason_features").select(fields).to_dicts()
    }
    extra_counters = ["receptions", "targets", "receiving_yards_after_catch", "receiving_air_yards"]
    counters = list(
        dict.fromkeys([c for pair in RATE_PAIRS.values() for c in pair] + extra_counters)
    )
    weekly = inputs.gold("nfl_player_weeks").filter(pl.col("season_type") == "REG")
    annual = (
        weekly.group_by("player_id", "season")
        .agg(
            [
                pl.when(pl.col(c).is_not_null().all())
                .then(pl.col(c).sum())
                .otherwise(None)
                .alias(c)
                for c in counters
            ]
        )
        .to_dicts()
    )
    history = defaultdict(dict)
    for r in annual:
        history[r["player_id"]][r["season"]] = r
    segments = (
        weekly.with_columns(
            pl.col("team").replace(
                {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX"}
            )
        )
        .group_by("player_id", "season", "team", "position")
        .agg(
            [
                pl.when(pl.col(c).is_not_null().all())
                .then(pl.col(c).sum())
                .otherwise(None)
                .alias(c)
                for c in counters
            ]
        )
        .to_dicts()
    )
    by_team = defaultdict(list)
    for r in segments:
        by_team[r["season"], r["team"], r["position"]].append(r)
    providers = inputs.raw_providers()
    ngs = {}
    for kind in ("passing", "rushing", "receiving"):
        records = (
            providers[kind]
            .filter((pl.col("season_type") == "REG") & (pl.col("week") == 0))
            .to_dicts()
        )
        ngs[kind] = {(r["player_gsis_id"], r["season"]): r for r in records}
        if len(ngs[kind]) != len(records):
            raise ValueError(f"Ambiguous annual NGS records for {kind}")
    opening, starts, missing = schedule_labels(providers["schedule"])
    ledger_path = inputs.bind(REPO / "research/qb_role_evidence_20260923.csv")
    with ledger_path.open() as stream:
        ledger = list(csv.DictReader(stream))
    for row in rows:
        pid, year, position = row["player_id"], row["forecast_season"], row["position"]
        source = originals[pid, year]
        recent = [r for y, r in history[pid].items() if year - 3 <= y < year]
        last = history[pid].get(year - 1, {})
        row["source_team"] = normal_team(source["source_team"])
        for family in FAMILIES[position]:
            num, den = RATE_PAIRS[family]
            total, n = (
                complete_sum([r[num] for r in recent]),
                complete_sum([r[den] for r in recent]),
            )
            row[f"rate_multi_{family}"] = ratio(total, n)
            row[f"exposure_multi_{family}"] = n if ratio(total, n) is not None else None
            row[f"u_multi_{family}"] = ratio(total, n)
            row[f"u_multi_n_{family}"] = float(np.log1p(n)) if n is not None and n > 0 else None
            teammates = by_team.get((year - 1, row["source_team"], position), [])
            others = [r for r in teammates if r["player_id"] != pid]
            own = [r for r in teammates if r["player_id"] == pid]
            peer = ratio(
                complete_sum([r[num] for r in others]), complete_sum([r[den] for r in others])
            )
            own_rate = ratio(
                complete_sum([r[num] for r in own]), complete_sum([r[den] for r in own])
            )
            row[f"c_relative_{family}"] = (
                own_rate - peer if own_rate is not None and peer is not None else None
            )
            row[f"c_peer_{family}"] = peer
        row["e_catch_fraction"] = ratio(last.get("receptions"), last.get("targets"))
        row["e_yac_per_reception"] = ratio(
            last.get("receiving_yards_after_catch"), last.get("receptions")
        )
        row["e_air_per_target"] = ratio(last.get("receiving_air_yards"), last.get("targets"))
        row["d_prior_starts"] = len(starts.get((pid, year - 1), []))
        row["d_late_starts"] = sum(w >= 14 for w in starts.get((pid, year - 1), []))
        row["d_designed_carries"] = finite(source["designed_carries"])
        row["d_scramble_carries"] = finite(source["scramble_carries"])
        row["y_opening"] = None if missing.get(year, 0) else float(pid in opening[year])
        row.update(dated_news(ledger, pid, year, row["forecast_cutoff_date"]))
        threshold = {"QB": 200, "RB": 100, "WR": 50, "TE": 50}[position]
        exposure = row[f"exposure_{FAMILIES[position][0]}"]
        row["pool_role"] = int((exposure or 0) >= threshold)
        for index, (kind, field, denominator) in enumerate(NGS[position]):
            source_ngs = ngs[kind].get((pid, year - 1), {})
            rate, n = finite(source_ngs.get(field)), finite(source_ngs.get(denominator))
            row[f"rate_ngs{index}"] = rate if n and n > 0 else None
            row[f"exposure_ngs{index}"] = n if rate is not None and n and n > 0 else None
            row[f"g_ngs{index}"] = row[f"rate_ngs{index}"]
            row[f"g_ngs_n{index}"] = float(np.log1p(n)) if n and n > 0 else None
    archive_columns = []
    audit = {
        "opening_label_missing_by_season": missing,
        "dated_news_rows": sum(r["d_news_observed"] for r in rows if r["position"] == "QB"),
    }
    if archive:
        root = REPO / "data/research/individual_stats_20260923_r4"
        manifest = json.loads(inputs.bind(root / "manifest.json").read_text())
        specs = json.loads(
            inputs.bind(root / "specs.json", manifest["files"]["specs.json"]).read_text()
        )
        selected = specs["profile_market"]
        if any(any(word in c for word in ("actual", "outcome", "prediction")) for c in selected):
            raise ValueError("Outcome-like column in archived feature recipe")
        frame = pl.read_parquet(
            inputs.bind(root / "features.parquet", manifest["files"]["features.parquet"]),
            columns=["player_id", "forecast_season", "forecast_cutoff_date", "actual_season_points"]
            + selected,
        )
        archived = {
            (r["player_id"], r["forecast_season"], str(r["forecast_cutoff_date"])): r
            for r in frame.to_dicts()
        }
        matched = 0
        for row in rows:
            key = row["player_id"], row["forecast_season"], row["forecast_cutoff_date"]
            old = archived.get(key)
            if old is None or not np.isclose(
                old["actual_season_points"], row["y_season_points"], atol=1e-6
            ):
                raise ValueError(f"Archived profile cohort/label mismatch: {key}")
            matched += 1
            row.update({f"a_{c}": finite(old[c]) for c in selected})
            current = originals[row["player_id"], row["forecast_season"]]
            for prefix, rank_name in (
                ("position", "market_ecr"),
                ("overall", "market_overall_ecr"),
            ):
                rank = finite(current[rank_name])
                row[f"a_market_{prefix}_log_rank"] = (
                    float(np.log(rank)) if rank and rank > 0 else None
                )
                row[f"a_market_{prefix}_inverse_rank"] = 1 / rank if rank and rank > 0 else None
                row[f"a_market_{prefix}_dispersion"] = finite(current[rank_name + "_sd"])
        archive_columns = [f"a_{c}" for c in selected]
        audit.update(
            archived_matched_rows=matched,
            archived_predictors=len(selected),
            archived_market_repaired=True,
            archived_gold=manifest["gold"],
        )
    return rows, archive_columns, audit


def pool_features(train, test, position):
    specs = {}
    for family in FAMILIES[position]:
        specs[f"multi_{family}"] = fit_pool(train, f"multi_{family}")
        specs[family] = fit_pool(train, family)
        for role in (0, 1):
            subset = [
                r for r in train if r["pool_role"] == role and r.get(f"rate_{family}") is not None
            ]
            specs[f"role{role}_{family}"] = (
                fit_pool(subset, family) if len(subset) >= 40 else specs[family]
            )
    for index in range(len(NGS[position])):
        specs[f"ngs{index}"] = fit_pool(train, f"ngs{index}")

    def apply(rows):
        result = []
        for original in rows:
            row = dict(original)
            for label, pool in specs.items():
                field = label.split("_", 1)[1] if label.startswith("role") else label
                n, rate = row[f"exposure_{field}"], row[f"rate_{field}"]
                value = (
                    ((n * rate + pool["strength"] * pool["mean"]) / (n + pool["strength"]))
                    if n
                    else None
                )
                row[f"s_{label}"] = value
            for family in FAMILIES[position]:
                row[f"s_role_{family}"] = row[f"s_role{row['pool_role']}_{family}"]
            result.append(row)
        return result

    return apply(train), apply(test), specs


def run(inputs, root):
    rows, archival, audit = prepare(inputs)
    write_json(root / "feature_audit.json", audit)
    tasks = CONFIG["tasks"] + [
        ["QB", "passing_efficiency"],
        ["WR", "receiving_efficiency"],
        ["TE", "receiving_efficiency"],
    ]
    outputs, folds = [], []
    for position, target in tasks:
        cohort = [r for r in rows if r["position"] == position and r[f"y_{target}"] is not None]
        base, raw = columns(cohort, raw=False), columns(cohort)
        families = FAMILIES[position]
        multi = [c for f in families for c in (f"u_multi_{f}", f"u_multi_n_{f}")]
        multi_pool = [c for f in families for c in (f"s_multi_{f}", f"u_multi_n_{f}")]
        role_pool = [c for f in families for c in (f"s_role_{f}", f"n_{f}")]
        context = [c for f in families for c in (f"c_relative_{f}", f"c_peer_{f}")]
        ngs_raw = [c for i in range(len(NGS[position])) for c in (f"g_ngs{i}", f"g_ngs_n{i}")]
        ngs_pool = [c for i in range(len(NGS[position])) for c in (f"s_ngs{i}", f"g_ngs_n{i}")]
        specs = {
            "base_market": (base, "base_market"),
            "raw_market": (raw, "base_market"),
            "multi_raw": (raw + multi, "raw_market"),
            "multi_pooled": (raw + multi_pool, "multi_raw"),
            "role_pooled": (base + role_pool, "raw_market"),
            "team_context": (raw + context, "raw_market"),
            "ngs_raw": (raw + ngs_raw, "raw_market"),
            "ngs_pooled": (raw + ngs_pool, "ngs_raw"),
        }
        if position in ("RB", "WR", "TE"):
            specs["receiving_execution"] = (raw + EXECUTION, "raw_market")
        if position == "QB":
            specs.update(
                qb_duration=(raw + DURATION, "raw_market"),
                qb_news=(raw + DURATION + NEWS, "qb_duration"),
            )
            if target in ("attempts", "carries", "season_points"):
                specs["opening_mixture"] = (raw + DURATION + NEWS, "qb_news")
        if target == "season_points":
            specs.update(
                profile_market=(archival, "raw_market"),
                profile_plus_context=(archival + context, "profile_market"),
                profile_plus_ngs=(archival + ngs_raw, "profile_market"),
            )
            if position in ("RB", "WR", "TE"):
                specs["profile_plus_execution"] = (archival + EXECUTION, "profile_market")
        for year in range(2007, 2026):
            train = [r for r in cohort if r["forecast_season"] < year]
            test = [r for r in cohort if r["forecast_season"] == year]
            if not test or len({r["forecast_season"] for r in train}) < 3:
                continue
            training, testing, priors = pool_features(train, test, position)
            forecasts = {}
            for name, (features, control) in specs.items():
                if name == "opening_mixture":
                    eligible = [r for r in training if r["y_opening"] is not None]
                    probability = predict(
                        eligible, testing, "y_opening", features, kind="probability"
                    )
                    experts = [
                        predict(
                            [r for r in eligible if r["y_opening"] == role],
                            testing,
                            f"y_{target}",
                            features,
                        )
                        for role in (0, 1)
                    ]
                    forecast = (1 - probability) * experts[0] + probability * experts[1]
                else:
                    forecast = predict(training, testing, f"y_{target}", features)
                for i, row in enumerate(test):
                    if row["known_available_games_cap"] == 0:
                        forecast[i] = 0.0
                forecasts[name] = forecast
                for row, value, reference in zip(test, forecast, forecasts[control], strict=True):
                    outputs.append(
                        record(row, target, name, control, row[f"y_{target}"], value, reference)
                    )
            folds.append(
                dict(
                    position=position,
                    target=target,
                    season=year,
                    train_n=len(train),
                    test_n=len(test),
                    train_max=max(r["forecast_season"] for r in train),
                    priors=priors,
                    feature_recipes={k: v[0] for k, v in specs.items()},
                )
            )
            print(f"signals {position} {target} {year}: {len(specs)} recipes", flush=True)
    summaries = save_points(root, "signals", outputs)
    write_json(root / "signal_folds.json", folds)
    return dict(predictions=len(outputs), folds=len(folds), comparisons=len(summaries))
