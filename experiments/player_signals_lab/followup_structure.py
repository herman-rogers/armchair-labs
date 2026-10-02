"""Team coherence, rookie development, participation, role transitions, and policy diagnostics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import polars as pl
from data import finite
from followup_common import columns, holm, predict, record, save_points
from followup_signals import normal_team
from safety import write_json


def reconcile(values, budget, residual_fraction, weight=1.0):
    values = np.maximum(np.asarray(values, float), 0)
    if values.sum() <= 0:
        return values.copy()
    allocated = max(0.0, budget) * (1 - np.clip(residual_fraction, 0, 1))
    return (1 - weight) * values + weight * allocated * values / values.sum()


def team_experiments(inputs, rows, old, root):
    teams = inputs.gold("nfl_team_seasons").to_dicts()
    by_team = {(r["season"], normal_team(r["team"])): r for r in teams}
    meta = {(r["player_id"], r["forecast_season"]): r for r in rows}
    team_rows = []
    for (year, team), r in by_team.items():
        if year > 2025:
            continue
        prior = by_team.get((year - 1, team), {})
        row = dict(
            player_id=team,
            position="TEAM",
            player_population="all",
            forecast_season=year,
            t_year=year,
            t_length=17 if year >= 2021 else 16,
        )
        for field in (
            "team_pass_attempts",
            "official_team_carries",
            "team_dropbacks",
            "team_games",
        ):
            row[f"t_prior_{field}"] = prior.get(field)
            row[f"y_{field}"] = r[field]
        team_rows.append(row)
    features = sorted(c for c in team_rows[0] if c.startswith("t_"))
    forecasts, outputs = {}, []
    for target in ("team_pass_attempts", "official_team_carries"):
        for year in range(2007, 2026):
            train = [r for r in team_rows if 2004 <= r["forecast_season"] < year]
            test = [r for r in team_rows if r["forecast_season"] == year]
            values = predict(train, test, f"y_{target}", features)
            for row, value in zip(test, values, strict=True):
                baseline = row[f"t_prior_{target}"] * row["t_length"] / row["t_prior_team_games"]
                outputs.append(
                    record(
                        row,
                        target,
                        "team_boost",
                        "prior_volume",
                        row[f"y_{target}"],
                        value,
                        baseline,
                    )
                )
                forecasts[target, year, row["player_id"]] = value
    weekly = inputs.gold("nfl_player_weeks").filter(pl.col("season_type") == "REG")
    actual = (
        weekly.group_by("player_id", "season", "team")
        .agg(pl.col("attempts").sum(), pl.col("carries").sum())
        .to_dicts()
    )
    actual_map = {(r["player_id"], r["season"], normal_team(r["team"])): r for r in actual}
    team_counts = Counter((r["player_id"], r["season"]) for r in actual)
    allocation, residual_audit = [], []
    for position, target, team_target in (
        ("QB", "attempts", "team_pass_attempts"),
        ("RB", "carries", "official_team_carries"),
    ):
        cohorts = defaultdict(list)
        for r in rows:
            team = normal_team(r["cutoff_preseason_team"])
            if r["position"] == position and team and (r["forecast_season"], team) in by_team:
                cohorts[r["forecast_season"], team].append(r)
        residuals = []
        for (year, team), group in cohorts.items():
            total = by_team[year, team][team_target]
            observed = sum(
                actual_map.get((r["player_id"], year, team), {}).get(target, 0) for r in group
            )
            if total:
                residuals.append((year, max(0.0, 1 - observed / total)))
        by_year_team = defaultdict(list)
        for r in old.filter(
            (pl.col("position") == position) & (pl.col("target") == target)
        ).to_dicts():
            team = normal_team(meta[r["player_id"], r["season"]]["cutoff_preseason_team"])
            if team and (team_target, r["season"], team) in forecasts:
                by_year_team[r["season"], team].append(r)
        for (year, team), group in sorted(by_year_team.items()):
            earlier = [fraction for season, fraction in residuals if 2004 <= season < year]
            residual = float(np.mean(earlier))
            budget = float(forecasts[team_target, year, team])
            original = np.array([r["prediction"] for r in group])
            for name, weight in (("team_full", 1.0), ("team_half", 0.5)):
                predictions = reconcile(original, budget, residual, weight)
                for r, value in zip(group, predictions, strict=True):
                    row = meta[r["player_id"], year]
                    allocation.append(
                        record(
                            row,
                            target,
                            name,
                            "independent_market",
                            r["actual"],
                            value,
                            r["prediction"],
                            cutoff_team=team,
                            multi_team_outcome=team_counts[r["player_id"], year] > 1,
                        )
                    )
            residual_audit.append(
                dict(
                    position=position,
                    season=year,
                    team=team,
                    residual_fraction=residual,
                    residual_train_team_seasons=len(earlier),
                    predicted_budget=budget,
                    independent_sum=float(original.sum()),
                    allocated_sum=float(reconcile(original, budget, residual).sum()),
                    residual_budget=float(budget - reconcile(original, budget, residual).sum()),
                )
            )
    save_points(root, "team_volume", outputs)
    save_points(root, "team_allocation", allocation)
    write_json(root / "team_allocation_audit.json", residual_audit)
    return len(outputs), len(allocation)


def rookie_experiments(inputs, rows, root):
    history = defaultdict(list)
    for r in inputs.gold("college_annual").to_dicts():
        history[r["college_id"]].append(r)
    links = defaultdict(set)
    for r in inputs.gold("college_identity_links").filter(pl.col("status") == "linked").to_dicts():
        links[r["player_id"]].add(r["college_id"])
    measures = [
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "carries",
        "share_receiving_yards",
        "share_carries",
    ]
    candidates, audit = [], []
    for source in rows:
        if source["player_population"] != "rookie":
            continue
        row = dict(source)
        ids = links.get(row["player_id"], set())
        earlier = (
            [
                r
                for cid in ids
                for r in history[cid]
                if r["season"] < row["forecast_season"]
                and r["complete_team_season"]
                and r["season_end_date"]
                and r["season_end_date"] < row["forecast_cutoff_date"]
            ]
            if len(ids) == 1
            else []
        )
        earlier.sort(key=lambda r: r["season"])
        last = earlier[-1] if earlier else {}
        row.update(
            k_age=row["x_age"],
            k_year=row["forecast_season"],
            k_draft=row["x_draft"],
            k_college_years=len(earlier),
            k_observed=float(bool(earlier)),
            k_transfer_count=sum(r["team_count"] > 1 for r in earlier)
            + sum(
                set(a["teams"]) != set(b["teams"])
                for a, b in zip(earlier, earlier[1:], strict=False)
            ),
        )
        for field in measures:
            values = [
                (r["season"], finite(r[field])) for r in earlier if finite(r[field]) is not None
            ]
            row[f"k_last_{field}"] = finite(last.get(field))
            row[f"k_peak_{field}"] = max(v for _, v in values) if values else None
            row[f"k_trend_{field}"] = (
                float(np.polyfit([y for y, _ in values], [v for _, v in values], 1)[0])
                if len(values) >= 2
                else None
            )
        candidates.append(row)
        audit.append(
            dict(
                player_id=row["player_id"],
                season=row["forecast_season"],
                college_ids=len(ids),
                complete_earlier_years=len(earlier),
            )
        )
    outputs = []
    base = ["k_age", "k_year"]
    latest = ["k_observed"] + [f"k_last_{f}" for f in measures]
    development = ["k_college_years", "k_transfer_count"] + [
        f"k_{kind}_{f}" for kind in ("peak", "trend") for f in measures
    ]
    recipes = {
        "age_control": (base, "age_control"),
        "pre_draft_latest": (base + latest, "age_control"),
        "pre_draft_development": (base + latest + development, "pre_draft_latest"),
        "post_draft_latest": (base + latest + ["k_draft"], "pre_draft_latest"),
        "post_draft_development": (base + latest + ["k_draft"] + development, "post_draft_latest"),
    }
    for position in ("QB", "RB", "WR", "TE"):
        for year in range(2007, 2026):
            train = [
                r for r in candidates if r["position"] == position and r["forecast_season"] < year
            ]
            test = [
                r for r in candidates if r["position"] == position and r["forecast_season"] == year
            ]
            forecasts = {}
            for name, (features, control) in recipes.items():
                values = predict(train, test, "y_season_points", features)
                forecasts[name] = values
                for row, value, reference in zip(test, values, forecasts[control], strict=True):
                    outputs.append(
                        record(
                            row,
                            "rookie_season_points",
                            name,
                            control,
                            row["y_season_points"],
                            value,
                            reference,
                        )
                    )
        print(f"college development {position}", flush=True)
    save_points(root, "rookie", outputs)
    write_json(root / "rookie_coverage.json", audit)
    return len(outputs)


def availability_experiments(inputs, rows, root):
    schedule = inputs.gold("nfl_schedule").filter(pl.col("game_type") == "REG")
    snaps = inputs.gold("nfl_snap_counts").filter(pl.col("game_type") == "REG")
    schedule_counts = {
        r["season"]: r["n"]
        for r in schedule.group_by("season").agg(pl.col("game_id").n_unique().alias("n")).to_dicts()
    }
    coverage = {
        r["season"]: r["n"] / schedule_counts[r["season"]]
        for r in snaps.group_by("season").agg(pl.col("game_id").n_unique().alias("n")).to_dicts()
        if r["season"] in schedule_counts
    }
    identities = inputs.gold("players").select("gsis_id", "pfr_id").drop_nulls()
    unique_ids = identities.group_by("pfr_id").len().filter(pl.col("len") == 1).select("pfr_id")
    identities = identities.join(unique_ids, on="pfr_id")
    matched = set(identities["gsis_id"])
    counts = {
        (r["gsis_id"], r["season"]): r["n"]
        for r in snaps.filter(pl.col("offense_snaps") > 0)
        .join(identities, left_on="pfr_player_id", right_on="pfr_id")
        .group_by("gsis_id", "season")
        .agg(pl.col("game_id").n_unique().alias("n"))
        .to_dicts()
    }
    injury = inputs.gold("nfl_injuries").filter(pl.col("game_type") == "REG").to_dicts()
    by_player = defaultdict(list)
    source_years = set()
    for r in injury:
        if r["gsis_id"]:
            by_player[r["gsis_id"], int(r["season"])].append(r)
            source_years.add(int(r["season"]))
    candidates, differences = [], []
    for source in rows:
        row = dict(source)
        pid, year = row["player_id"], row["forecast_season"]
        if coverage.get(year, 0) < 0.98 or pid not in matched:
            continue
        row["y_snap_games"] = counts.get((pid, year), 0)
        row["i_prior_snaps"] = (
            counts.get((pid, year - 1), 0) if coverage.get(year - 1, 0) >= 0.98 else None
        )
        reports = [
            r
            for r in by_player[pid, year - 1]
            if r["date_modified"] is not None
            and str(r["date_modified"])[:10] < row["forecast_cutoff_date"]
        ]
        row["i_report_weeks"] = (
            len({r["week"] for r in reports}) if year - 1 in source_years else None
        )
        row["i_out_weeks"] = (
            len({r["week"] for r in reports if r["report_status"] == "Out"})
            if year - 1 in source_years
            else None
        )
        row["i_questionable_weeks"] = (
            len({r["week"] for r in reports if r["report_status"] == "Questionable"})
            if year - 1 in source_years
            else None
        )
        candidates.append(row)
        differences.append(
            dict(
                position=row["position"],
                season=year,
                player_id=pid,
                snap_games=row["y_snap_games"],
                scoring_games=row["y_scoring_appearances"],
            )
        )
    outputs = []
    for position in ("QB", "RB", "WR", "TE"):
        cohort = [r for r in candidates if r["position"] == position]
        basic = columns(cohort) + ["i_prior_snaps"]
        for year in range(2007, 2026):
            train = [r for r in cohort if r["forecast_season"] < year]
            test = [r for r in cohort if r["forecast_season"] == year]
            if not test or len({r["forecast_season"] for r in train}) < 3:
                continue
            control = predict(train, test, "y_snap_games", basic)
            extra = predict(
                train,
                test,
                "y_snap_games",
                basic + ["i_report_weeks", "i_out_weeks", "i_questionable_weeks"],
            )
            for row, value, reference in zip(test, extra, control, strict=True):
                outputs.append(
                    record(
                        row,
                        "offensive_snap_appearances",
                        "prior_reports",
                        "history_market",
                        row["y_snap_games"],
                        min(value, 18),
                        min(reference, 18),
                    )
                )
    save_points(root, "availability", outputs)
    pl.DataFrame(differences).write_parquet(root / "appearance_ascertainment.parquet")
    write_json(
        root / "appearance_source_coverage.json",
        dict(snap_game_coverage=coverage, injury_source_years=sorted(source_years)),
    )
    return len(outputs)


def weekly_role_experiment(inputs, root):
    raw = inputs.raw_providers()["schedule"].filter(pl.col("game_type") == "REG").sort("week")
    games = defaultdict(list)
    for r in raw.to_dicts():
        for side in ("home", "away"):
            if r[f"{side}_qb_id"]:
                games[r["season"], normal_team(r[f"{side}_team"])].append(
                    (r["week"], r[f"{side}_qb_id"])
                )
    rows = []
    for (year, team), season in games.items():
        if not 2004 <= year <= 2025:
            continue
        for end in (4, 8, 12, 16):
            past = [(w, p) for w, p in season if w <= end]
            future = [(w, p) for w, p in season if end < w <= end + 4]
            if not past or not future:
                continue
            for pid in {p for _, p in past}:
                starts = [w for w, p in past if p == pid]
                recent = [(w, p) for w, p in past if w > end - 4]
                rows.append(
                    dict(
                        player_id=pid,
                        position="QB",
                        player_population="earlier_starter",
                        forecast_season=year,
                        team=team,
                        cutoff_week=end,
                        y_retained=float(sum(p == pid for _, p in future) / len(future) >= 0.5),
                        w_recent_share=sum(p == pid for _, p in recent) / len(recent),
                        w_cumulative_share=len(starts) / len(past),
                        w_last_starter=float(past[-1][1] == pid),
                        w_gap=end - max(starts),
                        w_year=year,
                        w_cutoff=end,
                    )
                )
    outputs = []
    features = sorted(k for k in rows[0] if k.startswith("w_"))
    for year in range(2007, 2026):
        train = [r for r in rows if r["forecast_season"] < year]
        test = [r for r in rows if r["forecast_season"] == year]
        values = predict(train, test, "y_retained", features, kind="probability")
        for row, value in zip(test, values, strict=True):
            similar = [r for r in train if r["w_last_starter"] == row["w_last_starter"]]
            control = (sum(r["y_retained"] for r in similar) + 1) / (len(similar) + 2)
            outputs.append(
                record(
                    row,
                    "next_block_majority_starter",
                    "usage_transition",
                    "calibrated_persistence",
                    row["y_retained"],
                    value,
                    control,
                    cutoff_week=row["cutoff_week"],
                    log_loss=float(
                        -np.log(np.clip(value if row["y_retained"] else 1 - value, 1e-9, 1))
                    ),
                    control_log_loss=float(-np.log(control if row["y_retained"] else 1 - control)),
                )
            )
    save_points(root, "weekly_roles", outputs)
    return len(outputs)


def decision_sensitivity(old, root):
    output = []
    points = old.filter(
        (pl.col("target") == "season_points") & pl.col("scaled_80_lower").is_not_null()
    )
    for (position, year), group in points.group_by("position", "season"):
        rows = group.to_dicts()
        k = min(len(rows), {"QB": 12, "RB": 24, "WR": 30, "TE": 12}[position])
        for policy, field in (
            ("mean", "prediction"),
            ("median", "median_prediction"),
            ("lower_bound", "scaled_80_lower"),
        ):
            chosen = sorted(rows, key=lambda r: (-r[field], r["player_id"]))[:k]
            output.append(
                dict(
                    position=position,
                    season=year,
                    policy=policy,
                    selected=k,
                    realized_points=sum(r["actual"] for r in chosen),
                    zero_outcomes=sum(r["actual"] == 0 for r in chosen),
                    player_ids=[r["player_id"] for r in chosen],
                )
            )
    write_json(root / "decision_policy_sensitivity.json", output)
    return len(output)


def run(inputs, root):
    rows = inputs.panel()
    old = inputs.previous().filter(pl.col("model") == "market_control")
    counts = {}
    counts["team_volume"], counts["team_allocation"] = team_experiments(inputs, rows, old, root)
    print("team coherence finished", flush=True)
    counts["rookie"] = rookie_experiments(inputs, rows, root)
    counts["availability"] = availability_experiments(inputs, rows, root)
    print("availability ascertainment finished", flush=True)
    counts["weekly_roles"] = weekly_role_experiment(inputs, root)
    counts["policy_slices"] = decision_sensitivity(old, root)
    files = sorted(root.glob("*_summaries.json"))
    tables = {path: json.loads(path.read_text()) for path in files}
    holm([r for table in tables.values() for r in table])
    for path, table in tables.items():
        write_json(path, table)
    return counts
