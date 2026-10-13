"""Chronological lineup uncertainty experiments; never a pregame forecast archive.

Offense uses Opening Drive's original scoring contract. K/DST retains ESPN scores.
Joint draws use a Gaussian copula with empirical marginal quantile functions; pair
correlations are estimated on previous-season out-of-time errors, shrunk toward
independence, then projected to a positive semidefinite correlation matrix.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from engine.metrics.weekly_distribution import joint_draws

POSITIONS = {"QB", "RB", "WR", "TE"}
QUANTILES = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
NONSTARTERS = {"BE", "IR", "ER", ""}


def finite(value):
    return isinstance(value, (float, int)) and bool(np.isfinite(value))


def normalize_team(value):
    return {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "WSH": "WAS"}.get(value, value)


def relationship(a, b):
    """No assumed correlation across unrelated games."""
    ta, tb = normalize_team(a.get("team")), normalize_team(b.get("team"))
    if not ta or not tb:
        return None
    if ta == tb:
        kind = "same_team"
    elif ta == normalize_team(b.get("opponent")) or tb == normalize_team(a.get("opponent")):
        kind = "opponents"
    else:
        return None
    return (kind, *sorted([a["position"], b["position"]]))


def estimate_correlations(rows, before_season, shrinkage=200.0):
    """Pool earlier OOT residual pairs, never same-season/in-sample residuals."""
    games = defaultdict(list)
    for row in rows:
        if row["season"] >= before_season or not finite(row.get("target_points")):
            continue
        if row.get("phase") in {"train", "training", "in_sample"}:
            raise ValueError("Covariance requires out-of-time predictions")
        if not finite(row.get("prediction")):
            continue
        team, opp = normalize_team(row.get("team")), normalize_team(row.get("opponent"))
        if not team or not opp:
            continue
        games[(row["season"], row["week"], tuple(sorted((team, opp))))].append(row)
    pairs = defaultdict(list)
    for players in games.values():
        for a, b in combinations(players, 2):
            if a["player_id"] == b["player_id"]:
                continue
            key = relationship(a, b)
            if key:
                # Standardize heteroskedastic residuals when OOT scale is available.
                sa, sb = a.get("std"), b.get("std")
                sa = sa if finite(sa) and sa > 0 else 1.0
                sb = sb if finite(sb) and sb > 0 else 1.0
                x = (a["target_points"] - a["prediction"]) / sa
                y = (b["target_points"] - b["prediction"]) / sb
                if a["position"] > b["position"]:
                    x, y = y, x
                pairs[key].append((x, y))
    result = {}
    for key, values in pairs.items():
        v = np.asarray(values)
        raw = float(np.corrcoef(v.T)[0, 1]) if len(v) >= 8 and np.all(v.std(axis=0) > 0) else 0.0
        result[key] = dict(
            n=len(v),
            raw=raw,
            correlation=float(np.clip(raw * len(v) / (len(v) + shrinkage), -0.5, 0.5)),
        )
    return result


def empirical_crps(samples, actual):
    values = np.sort(samples)
    n = len(values)
    return float(
        np.abs(values - actual).mean() - np.sum((2 * np.arange(n) - n + 1) * values) / n**2
    )


def aggregate(players, estimates, seed=42, actual=None):
    mean = sum(p["prediction"] for p in players)
    independent = joint_draws(players, n=4096, seed=seed)
    joint = joint_draws(players, estimates, n=4096, seed=seed)
    return dict(
        prediction=mean,
        crps=empirical_crps(joint, actual) if finite(actual) else None,
        independent_crps=empirical_crps(independent, actual) if finite(actual) else None,
        independent_std=float(independent.std()),
        std=float(joint.std()),
        independent_q10=float(np.quantile(independent, 0.1)),
        independent_q90=float(np.quantile(independent, 0.9)),
        q10=float(np.quantile(joint, 0.1)),
        q50=float(np.quantile(joint, 0.5)),
        q90=float(np.quantile(joint, 0.9)),
    )


def interval_score(actual, lo, hi, alpha=0.2):
    return hi - lo + 2 / alpha * max(lo - actual, 0) + 2 / alpha * max(actual - hi, 0)


def metrics(rows, mean="prediction", target="actual", prefix=""):
    paired = [r for r in rows if finite(r.get(mean)) and finite(r.get(target))]
    if not paired:
        return dict(n=0)
    errors = np.array([r[mean] - r[target] for r in paired])
    result = dict(
        n=len(paired),
        mae=float(np.abs(errors).mean()),
        rmse=float(np.sqrt(np.square(errors).mean())),
        bias=float(errors.mean()),
    )
    if mean not in {"prediction", prefix + "prediction"}:
        return result
    proper = (
        [r[prefix + "crps"] for r in paired if finite(r.get(prefix + "crps"))]
        if target in {"actual", "offense_actual"}
        else []
    )
    if proper:
        result["crps"] = float(np.mean(proper))
    intervals = [
        r for r in paired if finite(r.get(prefix + "q10")) and finite(r.get(prefix + "q90"))
    ]
    if intervals:
        result.update(
            interval_n=len(intervals),
            coverage80=float(
                np.mean([r[prefix + "q10"] <= r[target] <= r[prefix + "q90"] for r in intervals])
            ),
            width80=float(np.mean([r[prefix + "q90"] - r[prefix + "q10"] for r in intervals])),
            interval_score80=float(
                np.mean(
                    [
                        interval_score(r[target], r[prefix + "q10"], r[prefix + "q90"])
                        for r in intervals
                    ]
                )
            ),
        )
    return result


def season_bootstrap(rows, difference):
    """Five-season cluster interval, not an independence claim for player/lineup rows."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["season"]].append(difference(row))
    values = np.array([np.mean(v) for v in grouped.values()])
    if not len(values):
        return dict(n_seasons=0)
    rng = np.random.default_rng(20261011)
    boot = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return dict(
        n_seasons=len(values),
        season_equal_mean_difference=float(values.mean()),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        caveat=(
            "Season-block bootstrap with few seasons; exploratory uncertainty, "
            "not independent lineup observations"
        ),
    )


def synthetic_experiment(rows):
    """Fixed-seed, predicted-rank draft of 10 artificial offense-only lineups/week."""
    groups = defaultdict(list)
    for row in rows:
        if finite(row.get("target_points")):
            groups[(row["season"], row["week"])].append(row)
    estimates = {
        season: estimate_correlations(rows, season) for season in sorted({s for s, _ in groups})
    }
    output = []
    for (season, week), players in sorted(groups.items()):
        if season < 2021 or season > 2025:
            continue
        for design, rank_key in [("v3_ranked", "prediction"), ("v2_ranked", "v2_weekly_mean")]:
            pools = {
                pos: sorted(
                    [p for p in players if p["position"] == pos],
                    key=lambda p: p[rank_key],
                    reverse=True,
                )
                for pos in POSITIONS
            }
            # Rotating snake drafts produce legal disjoint rosters from forecast rank alone.
            rng = np.random.default_rng(season * 100 + week)
            lineups = [[] for _ in range(10)]
            for pos, count in [("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1)]:
                for _ in range(count):
                    for i in rng.permutation(10):
                        if pools[pos]:
                            lineups[i].append(pools[pos].pop(0))
            flex = sorted(
                pools["RB"] + pools["WR"] + pools["TE"], key=lambda p: p[rank_key], reverse=True
            )
            for i in rng.permutation(10):
                if flex:
                    lineups[i].append(flex.pop(0))
            for i, lineup in enumerate(lineups):
                if len(lineup) != 8:
                    continue
                result = aggregate(
                    lineup,
                    estimates[season],
                    seed=season * 10000 + week * 10 + i,
                    actual=sum(p["target_points"] for p in lineup),
                )
                result.update(
                    design=design,
                    season=season,
                    week=week,
                    team_id=i,
                    actual=sum(p["target_points"] for p in lineup),
                    v2_prediction=sum(p["v2_weekly_mean"] for p in lineup),
                    v2_exact_all=all(p.get("v2_exact_available") == 1 for p in lineup),
                    v2_proxy_player_count=sum(p.get("v2_exact_available") != 1 for p in lineup),
                    player_ids=",".join(str(p["player_id"]) for p in lineup),
                    has_qb_receiver_stack=any(
                        a["team"] == b["team"]
                        and {a["position"], b["position"]} in ({"QB", "WR"}, {"QB", "TE"})
                        for a, b in combinations(lineup, 2)
                    ),
                    same_team_pairs=sum(
                        relationship(a, b) is not None and relationship(a, b)[0] == "same_team"
                        for a, b in combinations(lineup, 2)
                    ),
                    opposing_game_pairs=sum(
                        relationship(a, b) is not None and relationship(a, b)[0] == "opponents"
                        for a, b in combinations(lineup, 2)
                    ),
                )
                output.append(result)
    evaluated = [r for r in output if r["v2_exact_all"]]
    summary = dict(
        v3_joint=metrics(evaluated),
        v3_independent=metrics(evaluated, prefix="independent_"),
        v2_mean=metrics(evaluated, mean="v2_prediction"),
        by_stack={
            str(stack): dict(
                v3_joint=metrics([r for r in evaluated if r["has_qb_receiver_stack"] == stack]),
                v3_independent=metrics(
                    [r for r in evaluated if r["has_qb_receiver_stack"] == stack],
                    prefix="independent_",
                ),
            )
            for stack in (True, False)
        },
        by_design={
            d: dict(
                v3_joint=metrics([r for r in evaluated if r["design"] == d]),
                v3_independent=metrics(
                    [r for r in evaluated if r["design"] == d], prefix="independent_"
                ),
                v2_mean=metrics([r for r in evaluated if r["design"] == d], mean="v2_prediction"),
            )
            for d in sorted({r["design"] for r in evaluated})
        },
        by_season={
            str(s): dict(
                v3=metrics([r for r in evaluated if r["season"] == s]),
                independent=metrics(
                    [r for r in evaluated if r["season"] == s], prefix="independent_"
                ),
                v2=metrics([r for r in evaluated if r["season"] == s], mean="v2_prediction"),
            )
            for s in sorted({r["season"] for r in evaluated})
        },
    )
    summary["paired_season_bootstrap"] = dict(
        v3_minus_v2_absolute_error=season_bootstrap(
            evaluated,
            lambda r: abs(r["prediction"] - r["actual"]) - abs(r["v2_prediction"] - r["actual"]),
        ),
        joint_minus_independent_crps=season_bootstrap(
            evaluated, lambda r: r["crps"] - r["independent_crps"]
        ),
        joint_minus_independent_interval_score=season_bootstrap(
            evaluated,
            lambda r: (
                interval_score(r["actual"], r["q10"], r["q90"])
                - interval_score(r["actual"], r["independent_q10"], r["independent_q90"])
            ),
        ),
    )
    summary["comparison_cohort"] = (
        "Only lineups with exact retained-v2 forecasts for every starter; "
        "identical paired lineups and scoring"
    )
    summary["coverage"] = dict(
        all_lineups=len(output),
        exact_v2_lineups=len(evaluated),
        lineups_with_v2_proxy=len(output) - len(evaluated),
        proxy_player_allocations=sum(r["v2_proxy_player_count"] for r in output),
        total_player_allocations=8 * len(output),
    )
    summary["all_lineups_with_v2_proxy"] = dict(
        label="Sensitivity only: retained v2 plus history proxy for missing v2 player forecasts",
        v3_joint=metrics(output),
        v3_independent=metrics(output, prefix="independent_"),
        v2_plus_history_proxy_mean=metrics(output, mean="v2_prediction"),
    )
    return output, summary, estimates


def special_history(snapshot):
    observed = {}
    for game in snapshot["week_lineups"]:
        if game["week"] >= snapshot["week"]:
            continue
        for p in game["home_lineup"] + game["away_lineup"]:
            if p["position"] in {"K", "DST", "D/ST"} and finite(p.get("points")):
                observed[(p["espn_id"], game["week"])] = dict(
                    p,
                    week=game["week"],
                    position="DST" if p["position"] == "D/ST" else p["position"],
                )
    return list(observed.values())


def special_estimate(player, week, history, use_projection=False):
    """Prior-score fallback uncertainty is not claimed calibrated for ESPN means."""
    position = "DST" if player["position"] == "D/ST" else player["position"]
    previous = sorted(
        [r for r in history if r["espn_id"] == player["espn_id"] and r["week"] < week],
        key=lambda r: r["week"],
    )[-4:]
    if use_projection and finite(player.get("projected_points")):
        mean, source = (
            player["projected_points"],
            "ESPN captured projection; pregame timing unverified",
        )
    elif previous:
        mean, source = (
            np.mean([r["points"] for r in previous]),
            f"Prior {len(previous)} recorded scores",
        )
    else:
        population = [
            r["points"] for r in history if r["position"] == position and r["week"] < week
        ]
        mean, source = (
            (np.mean(population), "Earlier position scores; no player history")
            if population
            else (None, "No prior K/DST scoring data")
        )
    residuals = []
    for row in history:
        if row["position"] != position or row["week"] >= week:
            continue
        earlier = sorted(
            [r for r in history if r["espn_id"] == row["espn_id"] and r["week"] < row["week"]],
            key=lambda r: r["week"],
        )[-4:]
        if earlier:
            residuals.append(row["points"] - np.mean([r["points"] for r in earlier]))
    # RMS includes baseline bias; tiny samples are explicitly withheld.
    std = float(np.sqrt(np.mean(np.square(residuals)))) if len(residuals) >= 8 else None
    return dict(
        prediction=float(mean) if mean is not None else None,
        std=std,
        basis=source,
        uncertainty_basis=(
            "RMS of earlier chronological prior-score errors (includes bias); "
            "ESPN transfer unvalidated"
        ),
        uncertainty_n=len(residuals),
        residual_bias=float(np.mean(residuals)) if residuals else None,
        residual_centered_std=float(np.std(residuals)) if len(residuals) >= 8 else None,
    )


def verified_archive_count(root):
    from engine.espn.forecast_log import timestamp

    valid, total = 0, 0
    for path in (root / "data/outputs/weekly_forecasts").rglob("*.json"):
        record = json.loads(path.read_text())
        if not record.get("source"):
            continue
        total += 1
        ts = [timestamp(record.get(key)) for key in ("generated_at", "captured_at", "published_at")]
        deadline = timestamp(record.get("pregame_deadline"))
        if all(ts) and deadline and max(ts) < deadline:
            valid += 1
    return dict(records=total, verified_pregame_records=valid)


def model_scoring_actual(stats, position, espn_points=None):
    """Unknown observed components stay unknown; independently confirmed absence is zero."""
    if stats is None:
        return 0.0 if espn_points == 0 else None
    required = ["rushing_yards", "rushing_tds"]
    required += (
        ["passing_yards", "passing_tds", "passing_interceptions"]
        if position == "QB"
        else ["receptions", "receiving_yards", "receiving_tds"]
    )
    if any(not finite(stats.get(k)) for k in required):
        return None
    value = 0.1 * stats["rushing_yards"] + 6 * stats["rushing_tds"]
    if position == "QB":
        return (
            value
            + 0.04 * stats["passing_yards"]
            + 4 * stats["passing_tds"]
            - 2 * stats["passing_interceptions"]
        )
    return value + stats["receptions"] + 0.1 * stats["receiving_yards"] + 6 * stats["receiving_tds"]


def league_experiment(rows, root, snapshot=None, inputs=None):
    """Reconstruct observed league lineups; keep scoring contract and timing explicit."""
    from engine.chairstats.store import CampaignStore
    from engine.chairstats.views import rankings
    from engine.tables import observations
    from engine.tables.application import connect, query_scope

    if snapshot is None:
        snapshot = json.loads((root / "data/outputs/league_snapshot.json").read_text())
    season = snapshot["season"]
    if inputs is None:
        with query_scope(root / "data"), connect() as db:
            identities = observations.espn_ids(db)
            actuals = (
                db.execute(
                    "SELECT * FROM analytics.current_weeks WHERE season=? AND season_type='REG'",
                    [season],
                )
                .pl()
                .to_dicts()
            )
    else:
        identities = {int(k): v for k, v in inputs["identities"].items()}
        actuals = inputs["actuals"]
    amap = {(r["player_id"], r["week"]): r for r in actuals}
    assert len(amap) == len(actuals), "Duplicate current player-week actuals"
    pmap = {(r["player_id"], r["week"]): r for r in rows if r["season"] == season}
    tmap = {t["team_id"]: t["team_name"] for t in snapshot["teams"]}
    current_roster = snapshot.get("players", snapshot.get("rostered", []))
    if not current_roster:
        current_roster = snapshot.get("rostered_players", [])
    roster = {r["espn_id"]: r for r in current_roster}
    history = special_history(snapshot)
    store = CampaignStore(root / "data/campaigns") if inputs is None else None
    manifest = store.campaign("opening_drive_v2") if store else None
    captured_rankings = {}
    estimates = estimate_correlations(rows, season)
    teams, players = [], []
    for week in sorted(
        {g["week"] for g in snapshot["week_lineups"] if g["week"] <= snapshot["week"]}
    ):
        if inputs is None:
            horizon = "season" if week <= 2 else "remaining"
            models = [m for m in manifest["models"] if m["horizon"] == horizon]
            origins = set.intersection(
                *[
                    {
                        f["origin_week"]
                        for f in m["forecasts"]
                        if f["season"] == season and f["origin_week"] < week
                    }
                    for m in models
                ]
            )
            ranked = (
                rankings(
                    store,
                    "opening_drive_v2",
                    horizon=horizon,
                    season=season,
                    origin_week=max(origins),
                )["rankings"]
                if origins
                else []
            )
        else:
            ranked = inputs["rankings"][str(week)]
        captured_rankings[str(week)] = ranked
        v2map = {r["player_id"]: r for r in ranked}
        for game in [g for g in snapshot["week_lineups"] if g["week"] == week]:
            for side in ("home", "away"):
                lineup = []
                for player in game[side + "_lineup"]:
                    if player["slot"] in NONSTARTERS:
                        continue
                    pid = identities.get(player["espn_id"])
                    pos = "DST" if player["position"] == "D/ST" else player["position"]
                    record = dict(
                        season=season,
                        week=week,
                        team_id=game[side + "_team_id"],
                        player_id=pid,
                        espn_id=player["espn_id"],
                        player_display_name=player["player_display_name"],
                        position=pos,
                        slot=player["slot"],
                        team=normalize_team(roster.get(player["espn_id"], {}).get("espn_team")),
                        opponent=normalize_team(player.get("pro_opponent")),
                        prediction=None,
                        std=None,
                        v2_prediction=None,
                        actual_espn=player.get("points") if week < snapshot["week"] else None,
                        actual=None,
                    )
                    if pos in POSITIONS:
                        pred = pmap.get((pid, week))
                        baseline = v2map.get(pid)
                        if pred:
                            record.update(
                                {
                                    k: pred.get(k)
                                    for k in [
                                        "prediction",
                                        "std",
                                        "team",
                                        "opponent",
                                        *[f"q{int(q * 100):02d}" for q in QUANTILES],
                                    ]
                                }
                            )
                            record["basis"] = "Opening Drive v3 weekly model"
                        if baseline:
                            record["v2_prediction"] = baseline.get("points_per_game")
                        if week < snapshot["week"]:
                            stats = amap.get((pid, week))
                            record["actual"] = model_scoring_actual(
                                stats, pos, player.get("points")
                            )
                    else:
                        record.update(
                            special_estimate(
                                player, week, history, use_projection=week == snapshot["week"]
                            )
                        )
                        record["v2_prediction"] = record["prediction"]
                        record["actual"] = record["actual_espn"]
                    if player.get("on_bye"):
                        record.update(
                            prediction=0.0, std=0.0, v2_prediction=0.0, basis="Scheduled bye"
                        )
                        for q in QUANTILES:
                            record[f"q{int(q * 100):02d}"] = 0.0
                    lineup.append(record)
                    players.append(record)
                complete = bool(lineup) and all(finite(p.get("prediction")) for p in lineup)
                uncertainty = complete and all(finite(p.get("std")) for p in lineup)
                offense = [p for p in lineup if p["position"] in POSITIONS]
                result = dict(
                    season=season,
                    week=week,
                    team_id=game[side + "_team_id"],
                    team_name=tmap[game[side + "_team_id"]],
                    completed=week < snapshot["week"],
                    starter_count=len(lineup),
                    covered_starters=sum(finite(p.get("prediction")) for p in lineup),
                    prediction=sum(p["prediction"] for p in lineup) if complete else None,
                    v2_prediction=sum(p["v2_prediction"] for p in lineup)
                    if lineup and all(finite(p["v2_prediction"]) for p in lineup)
                    else None,
                    actual=sum(p["actual"] for p in lineup)
                    if lineup and all(finite(p["actual"]) for p in lineup)
                    else None,
                    actual_espn=game.get(side + "_score") if week < snapshot["week"] else None,
                    offense_prediction=sum(p["prediction"] for p in offense)
                    if offense and all(finite(p["prediction"]) for p in offense)
                    else None,
                    offense_v2_prediction=sum(p["v2_prediction"] for p in offense)
                    if offense and all(finite(p["v2_prediction"]) for p in offense)
                    else None,
                    offense_actual=sum(p["actual"] for p in offense)
                    if offense and all(finite(p["actual"]) for p in offense)
                    else None,
                    uncertainty_available=uncertainty,
                )
                if offense and all(finite(p["prediction"]) and finite(p["std"]) for p in offense):
                    result.update(
                        {
                            "offense_" + k: v
                            for k, v in aggregate(
                                offense, estimates, actual=result["offense_actual"]
                            ).items()
                            if k != "prediction"
                        }
                    )
                if uncertainty:
                    result.update(
                        aggregate(
                            lineup,
                            estimates,
                            seed=season * 10000 + week * 100 + result["team_id"],
                            actual=result["actual"],
                        )
                    )
                teams.append(result)
    from engine.scoring.rules import get_rules

    kicking = get_rules().kicking
    kicker_audit = []
    for p in history:
        if p["position"] != "K":
            continue
        r = amap.get((identities.get(p["espn_id"]), p["week"]))
        if r is None:
            continue
        points = sum(
            (r.get("fg_made_" + b) or 0) * kicking.field_goal_points(d)
            for b, d in [
                ("0_19", 10),
                ("20_29", 25),
                ("30_39", 35),
                ("40_49", 45),
                ("50_59", 55),
                ("60_", 65),
            ]
        )
        points += (r.get("pat_made") or 0) * kicking.extra_point + (
            r.get("fg_missed") or 0
        ) * kicking.missed_field_goal
        kicker_audit.append(
            dict(
                espn_id=p["espn_id"],
                week=p["week"],
                actual_espn=p["points"],
                config_score=points,
                blocked_penalty_score=points - (r.get("fg_blocked") or 0),
            )
        )
    calibration = dict(
        capture_season=season,
        capture_week=snapshot["week"],
        captured_at=snapshot["captured_at"],
        covariance_before_season=season,
        special_errors_before_week=snapshot["week"],
        evidence=(
            "Chronological OOT offensive residuals; K/DST prior-score errors from earlier "
            "completed captured weeks; K/DST ESPN transfer provisional"
        ),
        correlations=[
            dict(relationship=k[0], position_a=k[1], position_b=k[2], **v)
            for k, v in estimates.items()
        ],
        special_teams={},
    )
    for position in ("K", "DST"):
        estimate = special_estimate(
            dict(espn_id=-999999, position=position), snapshot["week"], history
        )
        calibration["special_teams"][position] = dict(
            std=estimate["std"],
            n=estimate["uncertainty_n"],
            basis=estimate["uncertainty_basis"],
            residual_bias=estimate["residual_bias"],
            residual_centered_std=estimate["residual_centered_std"],
        )
    notes = [
        (
            "All league results are retrospective development diagnostics, not "
            "verified pregame results; selected historical lineups are observed "
            "after the fact."
        ),
        (
            "Offense uses original position-specific PPR contract excluding "
            "fumbles, bonuses, two-point plays and cross-position production; mixed "
            "actual = that offense plus ESPN K/DST. Official ESPN score is "
            "separately retained."
        ),
        (
            "Completed-week K/DST means use only earlier recorded scores, with "
            "earlier-position fallback. Current K/DST means retain captured ESPN "
            "projections; projection capture may occur after kickoff."
        ),
        (
            "K/DST uncertainty uses only earlier prior-score forecast errors "
            "(minimum eight); it is exploratory and has not been calibrated for "
            "ESPN projection errors. Missing support withholds full-lineup ranges."
        ),
        (
            "Offensive pair correlations use only earlier-season out-of-time "
            "residuals, shrink toward zero with 200 pseudo-pairs, and are capped at "
            "0.5 absolute before PSD projection. K/DST covariance defaults to "
            "independence because no adequate historical residual archive exists."
        ),
        (
            "Raw standardized-error Pearson correlation is used as an approximate latent copula "
            "parameter, not exact outcome covariance. Joint quantiles come from 4,096 "
            "Gaussian-copula draws with "
            "256-point mean-centered marginals, linearly interpolated between quantiles "
            "with bounded flat P01/P99 tails. Dependence and tail assumptions require "
            "external validation."
        ),
        (
            "Current week may already be underway. These are research estimates "
            "based on the latest snapshot, not newly verified pregame forecasts."
        ),
    ]
    paired_hybrid = [
        r for r in teams if all(finite(r.get(k)) for k in ("prediction", "v2_prediction", "actual"))
    ]
    paired_offense = [
        r
        for r in teams
        if all(
            finite(r.get(k))
            for k in ("offense_prediction", "offense_v2_prediction", "offense_actual")
        )
    ]
    paired_official = [
        r
        for r in teams
        if all(finite(r.get(k)) for k in ("prediction", "v2_prediction", "actual_espn"))
    ]
    return (
        teams,
        players,
        dict(
            season=season,
            current_week=snapshot["week"],
            captured_at=snapshot["captured_at"],
            archive=inputs["archive"] if inputs else verified_archive_count(root),
            _inputs=dict(
                identities=identities,
                actuals=actuals,
                rankings=captured_rankings,
                archive=inputs["archive"] if inputs else verified_archive_count(root),
            ),
            calibration=calibration,
            kicker_scoring_reconciliation=dict(
                n=len(kicker_audit),
                config_exact=sum(
                    abs(r["config_score"] - r["actual_espn"]) < 1e-8 for r in kicker_audit
                ),
                blocked_penalty_exact=sum(
                    abs(r["blocked_penalty_score"] - r["actual_espn"]) < 1e-8 for r in kicker_audit
                ),
                rows=kicker_audit,
            ),
            metrics=dict(
                v3_hybrid=metrics(paired_hybrid),
                v2_hybrid=metrics(paired_hybrid, mean="v2_prediction"),
                v3_offense=metrics(
                    paired_offense,
                    mean="offense_prediction",
                    target="offense_actual",
                    prefix="offense_",
                ),
                v2_offense=metrics(
                    paired_offense,
                    mean="offense_v2_prediction",
                    target="offense_actual",
                    prefix="offense_",
                ),
                v3_official_espn=metrics(paired_official, target="actual_espn"),
                v2_official_espn=metrics(
                    paired_official, mean="v2_prediction", target="actual_espn"
                ),
            ),
            coverage=dict(
                completed_team_weeks=sum(r["completed"] for r in teams),
                paired_hybrid_team_weeks=len(paired_hybrid),
                paired_offense_team_weeks=len(paired_offense),
                v3_offense_available=sum(
                    r["completed"] and finite(r["offense_prediction"]) for r in teams
                ),
                v2_offense_available=sum(
                    r["completed"] and finite(r["offense_v2_prediction"]) for r in teams
                ),
                v3_hybrid_available=sum(r["completed"] and finite(r["prediction"]) for r in teams),
                v2_hybrid_available=sum(
                    r["completed"] and finite(r["v2_prediction"]) for r in teams
                ),
            ),
            notes=notes,
        ),
    )


def kicker_experiment(root):
    """Historical kicker-score uncertainty under the repository's explicit rules."""
    from scipy.special import ndtr
    from sklearn.ensemble import HistGradientBoostingRegressor

    from engine.scoring.rules import get_rules
    from engine.tables.application import connect, query_scope

    rules = get_rules().kicking
    with query_scope(root / "data"), connect() as db:
        data = (
            db.execute(
                "SELECT * FROM analytics.nfl_player_weeks WHERE position='K' "
                "AND season>=2003 ORDER BY season,week,player_id"
            )
            .pl()
            .to_dicts()
        )
    history, records = defaultdict(list), []
    unknown_scoring_rows = 0
    bands = [("0_19", 10), ("20_29", 25), ("30_39", 35), ("40_49", 45), ("50_59", 55), ("60_", 65)]
    for row in data:
        required = ["fg_made", "fg_missed", "pat_made", *["fg_made_" + name for name, _ in bands]]
        if rules.blocked_counts_as_miss:
            required.append("fg_blocked")
        if any(not finite(row.get(k)) for k in required):
            unknown_scoring_rows += 1
            continue
        assert sum(row["fg_made_" + name] for name, _ in bands) == row["fg_made"], (
            "Kicker field-goal bands do not reconcile"
        )
        actual = sum(
            row["fg_made_" + name] * rules.field_goal_points(distance) for name, distance in bands
        )
        actual += (row.get("pat_made") or 0) * rules.extra_point
        actual += (row.get("fg_missed") or 0) * rules.missed_field_goal
        if rules.blocked_counts_as_miss:
            actual += (row.get("fg_blocked") or 0) * rules.missed_field_goal
        prior = history[row["player_id"]]
        if len(prior) >= 4:
            recent = prior[-4:]
            prediction = float(np.mean([r["actual"] for r in recent]))
            errors = [
                r["actual"] - r["prediction"]
                for r in prior[-16:]
                if r.get("prediction") is not None
            ]
            records.append(
                dict(
                    season=row["season"],
                    week=row["week"],
                    player_id=row["player_id"],
                    prediction=prediction,
                    actual=actual,
                    error=actual - prediction,
                    player_var=float(np.mean(np.square(errors))) if errors else 0.0,
                    history_n=len(errors),
                    recent_std=float(np.std([r["actual"] for r in recent])),
                    recent_mean=prediction,
                )
            )
        else:
            prediction = None
        prior.append(dict(actual=actual, prediction=prediction))
    output = []
    features = ["recent_mean", "recent_std", "player_var", "history_n", "week"]
    for season in range(2018, 2026):
        train = [r for r in records if r["season"] < season]
        test = [r for r in records if r["season"] == season]
        recent = [r for r in train if r["season"] >= season - 5]
        train_bias = float(np.mean([r["error"] for r in train]))
        train_centered_std = float(np.std([r["error"] for r in train]))
        pooled_var = float(np.mean([r["error"] ** 2 for r in train]))
        recent_var = float(np.mean([r["error"] ** 2 for r in recent]))
        tree = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=80,
            l2_regularization=10,
            random_state=42,
        )
        tree.fit([[r[k] for k in features] for r in train], [r["error"] ** 2 for r in train])
        predicted_variance = tree.predict([[r[k] for k in features] for r in test])
        for row, tree_var in zip(test, predicted_variance, strict=True):
            methods = dict(
                pooled=pooled_var,
                recent5=recent_var,
                player_shrunk=(row["player_var"] * row["history_n"] + pooled_var * 30)
                / (row["history_n"] + 30),
                conditional_tree=max(float(tree_var), 1.0),
            )
            for method, variance in methods.items():
                std = np.sqrt(variance)
                output.append(
                    dict(
                        **row,
                        method=method,
                        training_residual_bias=train_bias,
                        training_residual_centered_std=train_centered_std,
                        std=float(std),
                        crps=float(
                            std
                            * (
                                (row["error"] / std) * (2 * ndtr(row["error"] / std) - 1)
                                + 2 * np.exp(-0.5 * (row["error"] / std) ** 2) / np.sqrt(2 * np.pi)
                                - 1 / np.sqrt(np.pi)
                            )
                        ),
                        q10=row["prediction"] - 1.2815515655 * std,
                        q90=row["prediction"] + 1.2815515655 * std,
                    )
                )
    selection = {
        m: metrics([r for r in output if r["method"] == m and r["season"] <= 2020])
        for m in sorted({r["method"] for r in output})
    }
    winner = min(selection, key=lambda m: selection[m]["interval_score80"])
    report = dict(
        selected_method=winner,
        selection=selection,
        heldout={
            m: metrics([r for r in output if r["method"] == m and r["season"] >= 2021])
            for m in selection
        },
        history_rows=len(records),
        source_rows=len(data),
        unknown_scoring_rows_excluded=unknown_scoring_rows,
        scoring_sha256=sha256((root / "src/engine/config/scoring.yaml").read_bytes()).hexdigest(),
        scale_estimand=(
            "Predictive RMS error, including residual bias; conditional_tree "
            "estimates conditional MSE, not pure conditional variance"
        ),
        limitations=[
            (
                "All scales use squared errors around the retained prior-four mean; "
                "systematic bias is included. Training residual bias and centered dispersion "
                "are saved separately."
            ),
            (
                "Prior-four-game kicker mean with same-player history crossing "
                "offseason; only kickers with four earlier recorded games are "
                "evaluated."
            ),
            (
                "Uses repository kicking rules and historical box-score bands. Does "
                "not establish ESPN scoring agreement or ESPN-projection "
                "calibration; no automatic substitution into current ESPN "
                "uncertainty."
            ),
            (
                "Selection uses 2018–2020 only, evaluation 2021–2025. Each variance "
                "fit uses seasons strictly before target season. Normal intervals "
                "are an approximation."
            ),
        ],
    )
    return output, report


def write_rows(path, rows):
    if rows:
        pl.from_dicts(rows, infer_schema_length=None).write_parquet(path.with_suffix(".parquet"))
        pl.from_dicts(rows, infer_schema_length=None).write_csv(path.with_suffix(".csv"))


def build_lineup_report(run_dir, output_dir=None):
    root = Path(__file__).resolve().parents[4]
    run_dir = Path(run_dir)
    out = Path(output_dir) if output_dir else run_dir / "lineups"
    out.mkdir(parents=True, exist_ok=True)
    snapshot_path = out / "league_snapshot.json"
    if not snapshot_path.exists():
        snapshot_path.write_bytes((root / "data/outputs/league_snapshot.json").read_bytes())
    snapshot_bytes = snapshot_path.read_bytes()
    frozen_snapshot = json.loads(snapshot_bytes)
    sources = out / "source_snapshot"
    sources.mkdir(exist_ok=True)
    for path in (Path(__file__), root / "src/engine/metrics/weekly_distribution.py"):
        (sources / path.name).write_bytes(path.read_bytes())
    rows = pl.read_parquet(run_dir / "selected_predictions.parquet").to_dicts()
    kicker_rows, kicker_summary = kicker_experiment(root)
    write_rows(out / "kicker_benchmark", kicker_rows)
    synthetic, summary, estimates = synthetic_experiment(rows)
    write_rows(out / "synthetic_lineups", synthetic)
    inputs_path = out / "league_inputs.json"
    inputs = json.loads(inputs_path.read_text()) if inputs_path.exists() else None
    teams, players, league = league_experiment(rows, root, snapshot=frozen_snapshot, inputs=inputs)
    captured_inputs = league.pop("_inputs")
    if not inputs_path.exists():
        inputs_path.write_text(json.dumps(captured_inputs, allow_nan=False, default=str))
    league["inputs_sha256"] = sha256(inputs_path.read_bytes()).hexdigest()
    league["snapshot_sha256"] = sha256(snapshot_bytes).hexdigest()
    (out / "calibration.json").write_text(
        json.dumps(league["calibration"], indent=2, allow_nan=False)
    )
    write_rows(out / "league_team_weeks", teams)
    write_rows(out / "league_player_weeks", players)
    write_rows(out / "current_lineups", [r for r in teams if r["week"] == league["current_week"]])
    correlations = [
        dict(
            before_season=season, relationship=key[0], position_a=key[1], position_b=key[2], **value
        )
        for season, values in estimates.items()
        for key, value in values.items()
    ]
    write_rows(out / "correlations", correlations)
    result = dict(
        generated_at=datetime.now(UTC).isoformat(),
        synthetic=summary,
        kickers=kicker_summary,
        league=league,
        synthetic_design=(
            "Ten fixed-seed legal artificial offense-only lineups per week, drafted "
            "by predicted rank without outcomes. 1QB/2RB/3WR/1TE/1FLEX. Not actual "
            "fantasy league backtests; same lineups for v2 and v3."
        ),
        synthetic_dependence_note=(
            "2018–2020 selected OOT residuals supply initial covariance; each later "
            "season only sees prior-season residuals. Same-year errors never fit "
            "covariance."
        ),
    )
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_lineup_report(args.run_dir, args.output_dir), indent=2))
