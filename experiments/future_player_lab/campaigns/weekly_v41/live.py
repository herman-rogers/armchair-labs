"""Live v4.1 inference from one accepted snapshot and the sealed v4.1 research run.

Every number comes from models fitted in the v4.1 run. Data captured by other
collectors (accepted snapshots, official injury report captures) are inputs; no
other campaign's forecast, bundle or fitted model is read.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..weekly_v31.artifacts import digest, verify
from ..weekly_v32 import data as v32_data
from ..weekly_v32 import live as v32_live
from ..weekly_v32 import modeling as v32_modeling
from ..weekly_v32 import uncertainty as v32_uncertainty
from ..weekly_v32.publisher import rows as weekly_rows
from ..weekly_v4 import horizons as v4_horizons
from ..weekly_v4 import uncertainty as v4_uncertainty
from . import data, horizons, weekly

CAMPAIGN = "opening_drive_v41"
PROJECTION_METHOD = "opening_drive_v41_residual_components"
ANCHOR_BASIS = "v41_learned_residual"
LINEAGE = "opening_drive_v41 full retrain: v3 weekly method + market lines -> v3.2 injury/" \
    "workload method -> v4 horizon/component method"


def research_contract(run):
    run = Path(run)
    manifest = verify(run)
    if manifest.get("kind") != "opening_drive_v41_research":
        raise ValueError("Not a sealed v4.1 research run")
    return manifest


def live_panel(snapshot, horizon="early_week"):
    document = json.loads(Path(snapshot).read_text())
    if document["status"] != "accepted":
        raise ValueError("Live inference requires an accepted snapshot")
    season, through = (document["current_observations"][c] for c in ("season", "through_week"))
    frame, features, receipt, _ = data.prepare(snapshot, start_year=season - 1, include_byes=True)
    week = through + 1
    live = frame[frame.season.eq(season) & frame.week.eq(week)]
    byes = live[live.horizon.eq("early_week") & live.is_bye.eq(1)].copy()
    games = live[live.horizon.eq(horizon) & live.is_bye.ne(1)].copy()
    if games.empty or games.target_points.notna().any():
        raise ValueError("Live rows must be nonempty and unobserved")
    if pd.concat([games, byes]).player_id.duplicated().any():
        raise ValueError("Duplicate live identities")
    return games, byes, features, dict(receipt, season=season, through_week=through, week=week)


def attach_live_reports(frame, reports, horizon):
    """Replace archive-based report features with admitted official captures, if any."""
    if reports is None or reports.empty:
        return frame
    cutoff = "early_week" if horizon == "early_week" else "pregame_60m"
    out = v32_data.attach_reports(frame.drop(columns=list(v32_data.INJURY_FEATURES)), reports, cutoff)
    return v32_data.teammate_features(out)


def stage1(run, frame, features):
    choice = json.loads((Path(run) / "stage1_selection.json").read_text())
    models = joblib.load(Path(run) / "stage1_models_2026.joblib")
    out = pd.Series(np.nan, index=frame.index)
    for position, part in frame.groupby("position"):
        method = choice[position]["mean_method"]
        out.loc[part.index] = weekly.predict(models[(position, method)], part)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite stage-1 prediction")
    return out


def stage2(run, frame, horizon):
    """Selected injury/workload method, conserved teammate transfers, distributions."""
    run = Path(run)
    selections = json.loads((run / "selection.json").read_text())
    contracts = json.loads((run / "uncertainty.json").read_text())[horizon]
    frame = frame.reset_index(drop=True)
    values, info, bundles = {}, {}, {}
    for p, idx in frame.groupby("position").groups.items():
        bundles[p] = joblib.load(run / "models" / f"{horizon}_{p}_2026.joblib")
        values[p], info[p] = v32_modeling.predict(bundles[p], frame.loc[idx])
    n = len(frame)
    probability, before, after = np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3))
    for p, idx in frame.groupby("position").groups.items():
        probability[idx] = info[p]["team"]["probability"]
        before[idx] = (
            info[p]["no_injury"]["active_workload"] * info[p]["no_injury"]["probability"][:, None]
        )
        after[idx] = info[p]["team"]["active_workload"] * probability[idx, None]
    for name, fraction in (("redistribution_half", 0.5), ("redistribution_full", 1.0)):
        adjusted, _ = v32_modeling.redistribute(frame, before, after, probability, fraction)
        active = np.divide(
            adjusted, probability[:, None], out=np.zeros_like(adjusted), where=probability[:, None] > 0
        )
        for p, idx in frame.groupby("position").groups.items():
            values[p][name] = v32_modeling.points_from_workload(
                bundles[p], frame.loc[idx], probability[idx], active[idx]
            )
            info[p][name] = dict(probability=probability[idx], active_workload=active[idx])
    result = frame.copy()
    draws = np.zeros((n, 256))
    for p, idx in frame.groupby("position").groups.items():
        method = next(
            r["method"]
            for r in selections
            if r["year"] == 2026 and r["horizon"] == horizon and r["position"] == p
        )
        result.loc[idx, "prediction"] = values[p][method]
        stage = method if method.startswith("redistribution") else "team"
        result.loc[idx, "participation_probability"] = info[p][stage]["probability"]
        result.loc[idx, "chosen_method"] = method
        for j, c in enumerate(data.OPPORTUNITIES):
            result.loc[idx, f"expected_{c}"] = (
                info[p][stage]["active_workload"][:, j] * info[p][stage]["probability"]
            )
    if not np.isfinite(result.prediction).all():
        raise ValueError("Nonfinite v4.1 prediction")
    result["report_status_label"] = np.select(
        [
            result.report_out.eq(1),
            result.report_doubtful.eq(1),
            result.report_questionable.eq(1),
            result.report_known.eq(1),
        ],
        ["Out", "Doubtful", "Questionable", "Reported"],
        default="Unknown",
    )
    deterministic = result.report_out.fillna(0).eq(1) & result.prediction.eq(0)
    result["report_out"] = deterministic.astype(float)
    for p, idx in result.groupby("position").groups.items():
        draws[idx] = v4_uncertainty.samples(result.loc[idx], contracts[p])
    result = v32_uncertainty.frame_with_samples(result, draws)
    result["forecast_horizon"] = horizon
    return result


def bye_rows(byes):
    out = byes.copy()
    out["prediction"] = 0.0
    out["chosen_method"] = "deterministic_bye"
    out["participation_probability"] = np.nan
    for c in data.OPPORTUNITIES:
        out[f"expected_{c}"] = 0.0
    out = v32_uncertainty.frame_with_samples(out, np.zeros((len(out), 256)))
    out["forecast_horizon"] = "bye"
    out["report_status_label"] = "Bye"
    return out


def accepted_totals(document, season, origin):
    box, _ = data.current_asset(document, "weeks", origin)
    box = box[box.season.eq(season) & box.season_type.eq("REG")].copy()
    if box.duplicated(["player_id", "week"]).any():
        raise ValueError("Duplicate accepted actuals")
    box["points"] = data.v3_data.score_points(box)
    return box.groupby("player_id")[["points", *horizons.COMPONENTS]].agg(
        lambda s: s.sum() if s.notna().all() else np.nan
    )


def projection(run, players, snapshot):
    """Coherent weekly/remaining/season rows from one origin (v4 curve contract)."""
    run, snapshot = Path(run), Path(snapshot)
    models = joblib.load(run / "horizon_models_2026.joblib")
    document = json.loads(snapshot.read_text())
    season, origin = (document["current_observations"][c] for c in ("season", "through_week"))
    if models["training_through_season"] >= season:
        raise ValueError("Horizon models must be fitted before the forecast season")
    schedule, _ = data.current_asset(document, "schedule")
    calendar = v4_horizons.schedules(schedule)
    totals = accepted_totals(document, season, origin)
    feature_rows, plans, kept = [], {}, []
    for player in players.to_dict("records"):
        pid = player["player_id"]
        team = data.v3_data.TEAM_ALIASES.get(player["team"], player["team"])
        games = calendar.get((season, team), [])
        future = [w for w in games if w > origin]
        current = origin + 1 in future
        if not games or bool(player.get("is_bye", 0) == 1) == current:
            # One bad team mapping should not block the slate; drop and report the player.
            warnings.warn(f"Skipping {pid}: bye/team disagrees with accepted schedule", stacklevel=2)
            continue
        plans[pid] = (games, future, current)
        kept.append(player)
        for later, count in ((0, int(current)), (1, len(future) - int(current))):
            feature_rows.append(
                dict(player, origin_week=origin, later_games=later, horizon_games=max(count, 1))
            )
    features = pd.DataFrame(feature_rows).reset_index(drop=True)
    estimates = horizons.predictions(models, features)
    output = {h: [] for h in ("weekly", "remaining", "season")}
    weekly_by_id = {r["player_id"]: r for r in weekly_rows(players)}
    targets = horizons.TARGETS
    for i, player in enumerate(kept):
        pid, position = player["player_id"], player["position"]
        games, future, current = plans[pid]
        weekly_components = {
            t: float(estimates.loc[2 * i, t]) if current else 0.0 for t in targets[position]
        }
        later_components = {t: float(estimates.loc[2 * i + 1, t]) for t in targets[position]}
        anchor = float(estimates.loc[2 * i + 1, "points"])
        curve = [
            dict(
                week=w,
                points=float(player["prediction"]) if w == origin + 1 else anchor,
                components=weekly_components if w == origin + 1 else later_components,
                basis="v41_weekly" if w == origin + 1 else "v41_learned_horizon_extension",
            )
            for w in future
        ]
        remaining = {t: sum(g["components"][t] for g in curve) for t in targets[position]}
        remaining["points"] = sum(g["points"] for g in curve)
        earned = {
            t: float(totals.loc[pid, t]) if pid in totals.index else 0.0
            for t in ("points", *targets[position])
        }
        if not all(np.isfinite(v) for v in earned.values()):
            raise ValueError(f"Missing accepted actuals for {pid}")
        plan = dict(
            method=PROJECTION_METHOD,
            future_weeks=curve,
            remaining_points=remaining["points"],
            actual_points_to_date=earned["points"],
            actual_components_to_date={t: earned[t] for t in targets[position]},
            remaining_scheduled_games=len(future),
            season_scheduled_games=len(games),
            anchor_points_per_game=anchor,
            anchor_basis=ANCHOR_BASIS,
            schedule_team=player["team"],
            base_weekly_points=float(player["base_prediction"]),
        )
        for horizon, count in (("weekly", int(current)), ("remaining", len(future)), ("season", len(games))):
            values = (
                dict(weekly_components, points=float(player["prediction"]))
                if horizon == "weekly"
                else remaining
                if horizon == "remaining"
                else {t: remaining[t] + earned[t] for t in remaining}
            )
            row = dict(
                player_id=pid,
                player_name=player.get("player_name"),
                position=position,
                team=player["team"],
                season=season,
                origin_week=origin,
                horizon=horizon,
                predictions=values,
                scheduled_games=count,
                points_per_scheduled_game=values["points"] / count if count else None,
                component_points=v4_horizons.component_points(position, values),
                candidate_basis="v4.1 eligible pool rebuilt from the accepted snapshot",
                model_lineage=LINEAGE,
                base_campaign=CAMPAIGN,
            )
            if horizon == "weekly":
                extra = weekly_by_id[pid]
                for key in ("availability", "participation_probability", "expected_workload", "opponent"):
                    row[key] = extra.get(key)
                row["distribution"] = dict(
                    extra["distribution"], distribution_method="v41_prior_year_empirical_errors"
                )
            else:
                row["projection"] = plan
            output[horizon].append(row)
    _total_distributions(run, players, output)
    for rows in output.values():
        for position in targets:
            cohort = sorted(
                (r for r in rows if r["position"] == position),
                key=lambda r: (-r["predictions"]["points"], r["player_id"]),
            )
            for rank, row in enumerate(cohort, 1):
                row["rank"] = rank
    output["provenance"] = dict(
        research_manifest_sha256=digest(run / "manifest.json"),
        snapshot_sha256=digest(snapshot),
        evaluated_origin=origin in horizons.ORIGINS,
        lineage=LINEAGE,
    )
    return output


def _total_distributions(run, players, output):
    """Fixed-origin total errors (v4 method, v4.1 errors); last-game totals reuse weekly."""
    contract = json.loads((Path(run) / "total_uncertainty.json").read_text())
    # Origins outside the evaluated range (e.g. preseason 0 when backfilling) borrow the
    # nearest evaluated origin's errors.
    evaluated = sorted(int(k) for k in contract["origins"])
    for origin in range(19):
        if str(origin) not in contract["origins"] and evaluated:
            nearest = min(evaluated, key=lambda e: (abs(e - origin), e))
            contract["origins"][str(origin)] = contract["origins"][str(nearest)]
    remaining = output["remaining"]
    frame = pd.DataFrame(
        [
            dict(
                player_id=r["player_id"],
                position=r["position"],
                season=r["season"],
                origin_week=r["origin_week"],
                prediction=r["predictions"]["points"],
            )
            for r in remaining
        ]
    )
    draws = np.zeros((len(frame), 256))
    multi = np.array(
        [
            r["scheduled_games"] > 1
            or (
                r["scheduled_games"] == 1
                and r["projection"]["future_weeks"][0]["week"] != r["origin_week"] + 1
            )
            for r in remaining
        ]
    )
    if multi.any():
        draws[multi] = v4_uncertainty.total_samples(frame.loc[multi], contract)
    by_id = players.set_index("player_id")
    for i, row in enumerate(remaining):
        if not multi[i] and row["scheduled_games"]:
            draws[i] = np.asarray(by_id.loc[row["player_id"], "samples"])
    distributions = v32_uncertainty.frame_with_samples(frame, draws).set_index("player_id")
    for horizon in ("remaining", "season"):
        for row in output[horizon]:
            result = distributions.loc[row["player_id"]]
            shift = row["projection"]["actual_points_to_date"] if horizon == "season" else 0.0
            row["distribution"] = {
                **{c: float(result[c]) + shift for c in v32_uncertainty.QCOLS},
                "std": float(result["std"]),
                "samples": (np.asarray(result["samples"]) + shift).tolist(),
                "distribution_method": "v41_fixed_origin_total_errors",
            }


def forecast(snapshot, run, horizon="early_week", reports=None):
    """Return (players frame, projection dict, receipt) for the week after the snapshot.

    `reports` is a DataFrame of admitted official reports (first element returned by
    `official_reports`). Weekly components come from the horizon component models while
    weekly points come from stage 2; both are exposed, as in v4.
    """
    research_contract(run)
    games, byes, features, receipt = live_panel(snapshot, horizon)
    games = attach_live_reports(games, reports, horizon)
    games["base_prediction"] = stage1(run, games, features)
    # Byes: no weekly model row; horizon anchor uses past-only history (v4 fallback).
    byes["base_prediction"] = (
        byes.recent8_points_mean.fillna(byes.player_history_mean).fillna(0.0)
    )
    players = pd.concat([stage2(run, games, horizon), bye_rows(byes)], ignore_index=True)
    players["is_bye"] = players.is_bye.fillna(0)
    return players, projection(run, players, snapshot), receipt


def official_reports(directories, season, week, captured_at):
    """Admit official weekly report captures (data) using the v3.2 freshness rules.

    Returns `(reports, audit)`; pass only `reports` to `forecast`.
    """
    return v32_live.usable_reports(directories, season, week, captured_at)
