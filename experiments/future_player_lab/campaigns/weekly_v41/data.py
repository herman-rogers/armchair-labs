"""Fresh v4.1 panel: v3 feature method, market lines, v3.2 injury/workload method.

Every column is rebuilt from source data. No earlier version's trained model or
saved prediction enters the panel: the v3 builder receives no v2 baselines, and
its persistence fallback column is excluded from the feature allowlist.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..weekly_v3 import data as v3_data
from ..weekly_v31.artifacts import digest
from ..weekly_v32 import data as v32_data

ROOT = v3_data.ROOT
DATA = ROOT / "data"
DEFAULT_SNAPSHOT = DATA / "raw/snapshots/weekly_20261027_w7_review/manifest.json"
HORIZONS = v32_data.HORIZONS
OPPORTUNITIES = v32_data.OPPORTUNITIES
# Frozen-model outputs (v2) are never features; injury_* is superseded by v3.2 reports.
EXCLUDED = ("v2_weekly_mean", "v2_exact_available")
MARKET_FEATURES = (
    "market_total",
    "market_spread",
    "market_implied",
    "market_opp_implied",
    "market_known",
    "env_dome",
    "env_outdoor",
    "env_wind",
    "env_temp",
    "env_rest",
    "env_rest_diff",
    "env_div_game",
)


def market_frame(schedule):
    """One row per (game, team). Lines are as published in the schedule source.

    nflverse historical lines are closing lines; a live accepted snapshot carries the
    line known when it was captured. `spread_line` is the home team's expected margin.
    """
    schedule = schedule[schedule.game_type.eq("REG")] if "game_type" in schedule else schedule
    rows = []
    for g in schedule.to_dict("records"):
        total, spread = g.get("total_line"), g.get("spread_line")
        known = bool(pd.notna(total) and pd.notna(spread))
        for side, other, sign in (("home", "away", 1.0), ("away", "home", -1.0)):
            team_spread = sign * spread if known else np.nan
            rest, other_rest = g.get(f"{side}_rest"), g.get(f"{other}_rest")
            roof = g.get("roof")
            rows.append(
                dict(
                    game_id=g["game_id"],
                    team=v3_data.TEAM_ALIASES.get(g[f"{side}_team"], g[f"{side}_team"]),
                    market_total=total if known else np.nan,
                    market_spread=team_spread,
                    market_implied=(total + team_spread) / 2 if known else np.nan,
                    market_opp_implied=(total - team_spread) / 2 if known else np.nan,
                    market_known=float(known),
                    env_dome=float(roof in ("dome", "closed")) if pd.notna(roof) else np.nan,
                    env_outdoor=float(roof in ("outdoors", "open")) if pd.notna(roof) else np.nan,
                    env_wind=g.get("wind"),
                    env_temp=g.get("temp"),
                    env_rest=rest,
                    env_rest_diff=(rest - other_rest)
                    if pd.notna(rest) and pd.notna(other_rest)
                    else np.nan,
                    env_div_game=g.get("div_game"),
                )
            )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate game/team market rows")
    return frame


def attach_market(panel, schedule):
    market = market_frame(schedule)
    out = panel.copy()
    out["_team"] = out.team.replace(v3_data.TEAM_ALIASES)
    out = out.merge(
        market.rename(columns={"team": "_team"}),
        on=["game_id", "_team"],
        how="left",
        validate="many_to_one",
    ).drop(columns="_team")
    out["market_known"] = out.market_known.fillna(0.0)
    return out


def _schedule(snapshot):
    receipt = json.loads((v3_data.HISTORY / "manifest.json").read_text())
    schedule = pd.read_parquet(v3_data.HISTORY / "schedule.parquet")
    if digest(v3_data.HISTORY / "schedule.parquet") != receipt["files"]["schedule.parquet"]:
        raise ValueError("Historical schedule changed")
    document = json.loads(Path(snapshot).read_text())
    spec = document["assets"]["current/schedule.parquet"]
    path = DATA / spec["object"]
    if digest(path) != spec["sha256"]:
        raise ValueError("Snapshot schedule changed")
    current = pd.read_parquet(path)
    current = current[current.season.eq(document["current_observations"]["season"])]
    return pd.concat([schedule, current], ignore_index=True).drop_duplicates(
        "game_id", keep="last"
    )


def base_panel(snapshot=DEFAULT_SNAPSHOT, start_year=2003, end_year=2026):
    """The v3 feature construction with no v2 baselines and an explicit snapshot."""
    snapshot = Path(snapshot)
    previous = v3_data.SNAPSHOT, v3_data.load_v2
    try:
        v3_data.SNAPSHOT = snapshot
        v3_data.load_v2 = lambda schedule, hashes: {}
        with tempfile.TemporaryDirectory(prefix=".v41-panel-") as tmp:
            panel, features, receipt = v3_data.build_panel(tmp, start_year, end_year)
    finally:
        v3_data.SNAPSHOT, v3_data.load_v2 = previous
    if (panel.baseline_source != "earlier_history_proxy_not_v2").any():
        raise ValueError("A v2 baseline entered the v4.1 panel")
    features = [f for f in features if f not in EXCLUDED and not f.startswith("injury_")]
    panel = attach_market(panel, _schedule(snapshot))
    return panel, [*features, *MARKET_FEATURES], receipt


def current_asset(document, name, through=None):
    spec = document["assets"][f"current/{name}.parquet"]
    path = DATA / spec["object"]
    if digest(path) != spec["sha256"]:
        raise ValueError(f"Snapshot {name} changed")
    frame = pd.read_parquet(path)
    if through is not None and "week" in frame:
        frame = frame[frame.week.le(through)]
    return frame, {str(path): spec["sha256"]}


def workload_labels(panel, snapshot):
    """Opportunity/snap/participation labels using the v3.2 label contract."""
    document = json.loads(Path(snapshot).read_text())
    through = document["current_observations"]["through_week"]
    history_manifest = json.loads((v3_data.HISTORY / "manifest.json").read_text())
    box = v32_data.checked(v3_data.HISTORY / "box.parquet", history_manifest)
    current, hashes = current_asset(document, "weeks", through)
    box = pd.concat([box, current], ignore_index=True)
    box = box[box.season_type.eq("REG")].drop_duplicates(["player_id", "season", "week"])
    keys = ["player_id", "season", "week"]
    out = panel.merge(
        box[[*keys, *OPPORTUNITIES]].rename(columns={c: f"label_{c}" for c in OPPORTUNITIES}),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    for c in OPPORTUNITIES:
        out.loc[out.target_points.notna() & ~out.appeared, f"label_{c}"] = 0
        out.loc[out[f"label_{c}"].lt(0), f"label_{c}"] = np.nan
    previous = v32_data.SNAPSHOT
    try:
        v32_data.SNAPSHOT = Path(snapshot)
        snaps, more = v32_data.load_snaps()
    finally:
        v32_data.SNAPSHOT = previous
    hashes.update(more)
    out = out.merge(
        snaps[[*keys, "offense_snaps", "offense_pct"]].rename(
            columns={"offense_snaps": "label_snaps", "offense_pct": "label_snap_share"}
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    team_counts = (
        snaps.assign(team=snaps.team.replace(v3_data.TEAM_ALIASES))
        .groupby(["season", "week", "team"])
        .player_id.nunique()
    )
    support = pd.MultiIndex.from_frame(out[["season", "week", "team"]]).map(team_counts)
    known_absence = (
        out.label_snaps.isna()
        & ~out.appeared
        & out.target_points.notna()
        & pd.Series(np.asarray(support, dtype=float), index=out.index).ge(20)
        & out.player_id.isin(snaps.attrs["mapped_player_ids"])
    )
    out.loc[known_absence, ["label_snaps", "label_snap_share"]] = 0
    played = (
        out.label_snaps.gt(0)
        | out[[f"label_{c}" for c in OPPORTUNITIES]].gt(0).any(axis=1)
        | out.target_points.fillna(0).ne(0)
    )
    observed = out.label_snaps.notna() | played
    out["label_played"] = np.where(observed & out.target_points.notna(), played.astype(float), np.nan)
    return v32_data.lag_workloads(out), hashes


def kickoffs(schedule):
    schedule = schedule[schedule.game_type.eq("REG")].copy()
    local = pd.to_datetime(
        schedule.gameday.astype(str) + " " + schedule.gametime.fillna("00:00"), errors="coerce"
    )
    schedule["kickoff"] = local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    schedule.loc[schedule.gametime.isna(), "kickoff"] = pd.NaT
    return schedule[["game_id", "kickoff"]]


def prepare(snapshot=DEFAULT_SNAPSHOT, start_year=2003, include_byes=False):
    """Full research panel: scheduled rows, both timing arms, all labels kept separate.

    `include_byes` keeps bye rows in the early-week arm for live horizon curves; byes
    have no kickoff, so they never enter the pregame arm, and research fits exclude them.
    """
    panel, features, receipt = base_panel(snapshot, start_year=start_year)
    panel["row_id"] = np.arange(len(panel))
    panel, hashes = workload_labels(panel, snapshot)
    panel = panel.merge(
        kickoffs(_schedule(snapshot)), on="game_id", how="left", validate="many_to_one"
    )
    panel["early_week"] = pd.to_datetime(panel.feature_cutoff, utc=True)
    panel["pregame_60m"] = panel.kickoff - pd.Timedelta(hours=1)
    reports, audit, report_hashes = v32_data.load_reports()
    hashes.update(report_hashes)
    hashes.update(receipt.get("source_hashes", {}))
    frames = []
    for horizon in HORIZONS:
        scheduled = panel.scheduled_game.eq(1)
        if include_byes and horizon == "early_week":
            scheduled = scheduled | panel.is_bye.eq(1)
        part = panel[scheduled & panel[horizon].notna()].copy()
        part = v32_data.teammate_features(v32_data.attach_reports(part, reports, horizon))
        part["horizon"] = horizon
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    leaked = [f for f in features if f.startswith("label_") or f == "target_points"]
    if leaked:
        raise ValueError(f"Outcome columns in feature allowlist: {leaked}")
    receipt = dict(receipt, features=features, source_hashes=hashes)
    return frame, features, receipt, audit
