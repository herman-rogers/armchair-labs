"""Construct dated live inputs and replay the locked v3.2 selection policy."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from engine.chairstats import CampaignStore

from ..weekly_v3 import data as v3_data
from ..weekly_v3.rescore import _rows, _saved_models, validate_snapshot
from ..weekly_v31.artifacts import digest, timestamp, verify
from . import data, modeling, uncertainty


def asset(document, name):
    ref = document["assets"][f"current/{name}.parquet"]
    path = data.DATA / ref["object"]
    if digest(path) != ref["sha256"]:
        raise ValueError(f"Current {name} source changed")
    return pd.read_parquet(path)


def schedule_times(schedule):
    schedule = schedule.copy()
    local = pd.to_datetime(schedule.gameday.astype(str) + " " + schedule.gametime.fillna("00:00"))
    schedule["kickoff"] = local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    schedule.loc[schedule.gametime.isna(), "kickoff"] = pd.NaT
    return schedule


def baseline(snapshot, registry, captured_at):
    document = json.loads(Path(snapshot).read_text())
    current = document["current_observations"]
    season, origin = current["season"], current["through_week"]
    validate_snapshot(snapshot, season, origin)
    if any(timestamp(document[k]) > captured_at for k in ("registered_at",)) or (
        timestamp(current["saved_at"]) > captured_at
    ):
        raise ValueError("Snapshot was not available before capture")
    store = CampaignStore(registry)
    store.check("opening_drive_v3")
    manifest = store.campaign("opening_drive_v3")
    bundle = Path(registry) / "opening_drive_v3"
    payloads, matrices = [], []
    for entry in manifest["models"]:
        if entry["horizon"] != "weekly":
            continue
        ref = next(
            (r for r in entry["forecasts"] if r["season"] == season and r["origin_week"] == origin),
            None,
        )
        if ref is None:
            raise ValueError("Refresh v3's four positions before capturing v3.2")
        payload = json.loads((bundle / ref["path"]).read_text())
        if payload.get("snapshot_sha256") != digest(snapshot):
            raise ValueError("V3 forecast does not match the accepted snapshot")
        if timestamp(payload["generated_at"]) > captured_at:
            raise ValueError("Baseline generated after capture")
        matrix = entry["artifacts"][payload["feature_matrix_artifact"]]
        matrices.append(bundle / matrix["path"])
        payloads.append(payload)
    if len(payloads) != 4 or len({digest(p) for p in matrices}) != 1:
        raise ValueError("Complete v3 baseline must share one feature matrix")
    panel = pd.read_parquet(matrices[0])
    if panel.target_points.notna().any() or panel.duplicated("player_id").any():
        raise ValueError("Live baseline has labels or duplicate identities")
    if not (panel.season.eq(season) & panel.week.eq(origin + 1)).all():
        raise ValueError("Baseline matrix crosses the forecast origin")
    rows = [r for p in payloads for r in p["rows"]]
    return panel, rows, manifest


def workload_history(snapshot, registry, directory, research_run):
    """Use the same eligibility/label contract, with new completed weeks only."""
    document = json.loads(Path(snapshot).read_text())
    season, through = (document["current_observations"][c] for c in ("season", "through_week"))
    previous = v3_data.SNAPSHOT, v3_data.CAMPAIGN
    try:
        v3_data.SNAPSHOT = Path(snapshot)
        v3_data.CAMPAIGN = Path(registry) / "opening_drive_v2"
        current, _, _ = v3_data.build_panel(directory, start_year=season, end_year=season)
    finally:
        v3_data.SNAPSHOT, v3_data.CAMPAIGN = previous
    current = current[current.week.le(through) & current.scheduled_game.eq(1)].copy()
    box = asset(document, "weeks")
    box = box[box.season.eq(season) & box.week.le(through) & box.season_type.eq("REG")]
    box = box.dropna(subset=["player_id"])
    snaps = asset(document, "snaps")
    snaps = snaps[snaps.season.eq(season) & snaps.week.le(through)]
    snaps = snaps[~snaps.duplicated(["player_id", "season", "week"], keep=False)]
    keys = ["player_id", "season", "week"]
    current = current.merge(
        box[[*keys, *data.OPPORTUNITIES]], on=keys, how="left", validate="one_to_one"
    )
    current = current.merge(snaps, on=keys, how="left", validate="one_to_one")
    for c in data.OPPORTUNITIES:
        current[f"label_{c}"] = current[c].where(current[c].ge(0))
        current.loc[current.target_points.notna() & ~current.appeared, f"label_{c}"] = 0
    current["label_snaps"], current["label_snap_share"] = current.offense_snaps, current.offense_pct
    played = (
        current.offense_snaps.gt(0)
        | current[list(data.OPPORTUNITIES)].gt(0).any(axis=1)
        | current.target_points.fillna(0).ne(0)
    )
    current["label_played"] = np.where(
        (current.offense_snaps.notna() | played) & current.target_points.notna(), played, np.nan
    )
    historical = pd.read_parquet(Path(research_run) / "panel.parquet")
    historical = historical[historical.horizon.eq("early_week") & historical.season.lt(season)]
    columns = [
        *keys,
        *[f"label_{c}" for c in ("snaps", "snap_share", "played", *data.OPPORTUNITIES)],
    ]
    return pd.concat([historical[columns], current[columns]], ignore_index=True)


def attach_workloads(panel, history):
    # Each player's last four eligible scheduled weeks, including supported zero weeks.
    columns = [f"label_{c}" for c in ("snaps", "snap_share", "played", *data.OPPORTUNITIES)]
    if panel.empty or panel.season.nunique() != 1 or panel.week.nunique() != 1:
        raise ValueError("Live workload construction requires one forecast week")
    season, week = int(panel.season.iloc[0]), int(panel.week.iloc[0])
    if ((history.season > season) | ((history.season == season) & (history.week >= week))).any():
        raise ValueError("Workload history contains this game or future outcomes")
    if history.duplicated(["player_id", "season", "week"]).any():
        raise ValueError("Duplicate workload history")
    prior = history.sort_values(["season", "week"]).groupby("player_id").tail(4)
    means = prior.groupby("player_id")[columns].mean()
    out = panel.copy()
    for c in columns:
        out[c.replace("label_", "prior4_")] = out.player_id.map(means[c])
    return out


def usable_reports(directories, season, week, captured_at, max_age_hours=36):
    frames, audit = [], []
    for directory in directories:
        directory = Path(directory)
        verify(directory)
        receipt = json.loads((directory / "summary.json").read_text())
        observed = timestamp(receipt["observed_at"])
        fresh = 0 <= (captured_at - observed).total_seconds() <= max_age_hours * 3600
        eligible = (
            receipt.get("status") == "usable_weekly_report"
            and fresh
            and (receipt.get("season"), receipt.get("week")) == (season, week)
        )
        audit.append(
            dict(
                path=str(directory),
                manifest_sha256=digest(directory / "manifest.json"),
                admitted=eligible,
                status=receipt.get("status"),
                fresh=fresh,
            )
        )
        if eligible:
            frame = pd.read_parquet(directory / "reports.parquet").dropna(subset=["player_id"])
            frame = frame[frame.evidence_kind.eq("official_weekly_report")]
            if not (pd.to_datetime(frame.usable_at, utc=True) == pd.Timestamp(observed)).all():
                raise ValueError("Official observations were backdated")
            frames.append(frame)
    columns = [
        "player_id",
        "season",
        "week",
        "team",
        "usable_at",
        "report_status",
        "practice_status",
        "source",
    ]
    reports = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    reports["season"], reports["week"] = reports.season.astype(int), reports.week.astype(int)
    reports["usable_at"] = pd.to_datetime(reports.usable_at, utc=True)
    if len(reports):
        keys = ["player_id", "season", "week", "team", "usable_at"]
        for c in ("report_status", "practice_status"):
            conflict = reports.groupby(keys)[c].transform(lambda s: s.dropna().nunique()).gt(1)
            reports.loc[conflict, c] = "Conflict"
        reports = reports.drop_duplicates(keys, keep="last")
    return reports[columns], audit


def predict(integration, frame, horizon):
    """Frozen selected points, participation and workloads for a complete team pool."""
    integration = Path(integration)
    selections = json.loads((integration / "selection.json").read_text())
    frame = frame.reset_index(drop=True)
    values, info, bundles = {}, {}, {}
    result = frame.copy()
    for p, idx in frame.groupby("position").groups.items():
        bundles[p] = joblib.load(integration / "models" / f"{horizon}_{p}_2026.joblib")
        values[p], info[p] = modeling.predict(bundles[p], frame.loc[idx])
    probability, before, after = (
        np.zeros(len(frame)),
        np.zeros((len(frame), 3)),
        np.zeros((len(frame), 3)),
    )
    for p, idx in frame.groupby("position").groups.items():
        probability[idx] = info[p]["team"]["probability"]
        before[idx] = (
            info[p]["no_injury"]["active_workload"] * info[p]["no_injury"]["probability"][:, None]
        )
        after[idx] = info[p]["team"]["active_workload"] * probability[idx, None]
    transfers = []
    for name, fraction in (("redistribution_half", 0.5), ("redistribution_full", 1.0)):
        adjusted, budgets = modeling.redistribute(frame, before, after, probability, fraction)
        active = np.divide(
            adjusted,
            probability[:, None],
            out=np.zeros_like(adjusted),
            where=probability[:, None] > 0,
        )
        for p, idx in frame.groupby("position").groups.items():
            values[p][name] = modeling.points_from_workload(
                bundles[p], frame.loc[idx], probability[idx], active[idx]
            )
            info[p][name] = dict(probability=probability[idx], active_workload=active[idx])
        transfers.extend(dict(method=name, **b) for b in budgets)
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
            result.loc[idx, f"active_{c}"] = info[p][stage]["active_workload"][:, j]
            result.loc[idx, f"expected_{c}"] = (
                info[p][stage]["active_workload"][:, j] * info[p][stage]["probability"]
            )
    if not np.isfinite(result.prediction).all():
        raise ValueError("Nonfinite v3.2 prediction")
    contract = json.loads((integration / "uncertainty.json").read_text())[horizon]
    result = uncertainty.frame_with_samples(result, uncertainty.samples(result, contract))
    result["forecast_horizon"] = horizon
    return result, transfers


def replay_baseline(integration, panel, season, origin):
    bundle = Path(integration) / "baseline"
    entries = json.loads((bundle / "entries.json").read_text())
    rows = []
    for entry in entries:
        mean, spread = _saved_models(bundle, entry)
        rows.extend(_rows(panel, entry, mean, spread, season, origin))
    return rows
