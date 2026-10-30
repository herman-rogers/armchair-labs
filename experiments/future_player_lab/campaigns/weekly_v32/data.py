"""Audit source timing and construct an explicitly pregame injury/workload panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..weekly_v3.data import HISTORY, ROOT, SNAPSHOT, TEAM_ALIASES
from ..weekly_v3.run import DEFAULT_OUTPUT as V3_RUN
from ..weekly_v31.artifacts import digest

DATA = ROOT / "data"
ARCHIVE = DATA / "research/injury_archive_20260925_r1"
SOURCES = DATA / "research/opening_drive_v2_sources"
HORIZONS = ("early_week", "pregame_60m")
INJURY_FEATURES = (
    "report_known",
    "report_age_days",
    "report_out",
    "report_doubtful",
    "report_questionable",
    "report_probable",
    "practice_known",
    "practice_dnp",
    "practice_limited",
    "practice_full",
)
WORK_FEATURES = tuple(
    f"prior4_{c}" for c in ("snaps", "snap_share", "played", "attempts", "carries", "targets")
)
TEAM_FEATURES = tuple(
    f"room_{c}"
    for c in (
        "out_usage",
        "questionable_usage",
        "dnp_usage",
        "other_usage",
        "own_usage_share",
        "out_players",
    )
)
OPPORTUNITIES = ("attempts", "carries", "targets")


def current_asset(name):
    manifest = json.loads(SNAPSHOT.read_text())
    spec = manifest["assets"][f"current/{name}.parquet"]
    path = DATA / spec["object"]
    if digest(path) != spec["sha256"]:
        raise ValueError("Frozen current asset changed")
    f = pd.read_parquet(path)
    if name != "schedule":
        f = f[f.week.le(manifest["current_observations"]["through_week"])]
    return f, {str(path): spec["sha256"], str(SNAPSHOT): digest(SNAPSHOT)}


def checked(path, manifest, key=None):
    path = Path(path)
    expected = manifest.get("files", manifest.get("artifact_hashes", {}))[key or path.name]
    if isinstance(expected, dict):
        expected = expected.get("sha256")
    if digest(path) != expected:
        raise ValueError(f"Source checksum mismatch: {path}")
    return pd.read_parquet(path)


def normalize_reports(frame, source):
    out = frame.rename(columns={"gsis_id": "player_id"}).copy()
    if "game_type" in out:
        regular = out.game_type.eq("REG") | out.get(
            "season_type", pd.Series("", index=out.index)
        ).eq("REG")
        out = out[regular]
    out["usable_at"] = pd.to_datetime(out.date_modified, utc=True, errors="coerce")
    out = out.dropna(subset=["player_id", "usable_at", "season", "week", "team"])
    out["team"] = out.team.replace(TEAM_ALIASES)
    out["season"], out["week"] = out.season.astype(int), out.week.astype(int)
    out["source"] = source
    if "practice_status" not in out:
        out["practice_status"] = None
    keep = [
        "player_id",
        "season",
        "week",
        "team",
        "usable_at",
        "report_status",
        "practice_status",
        "source",
    ]
    for col in ("retrieved_at", "source_url", "source_sha256", "date_basis"):
        if col in out:
            keep.append(col)
    return out[keep]


def load_reports():
    source_manifest = json.loads((SOURCES / "manifest.json").read_text())
    archive_manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    original = checked(SOURCES / "injuries.parquet", source_manifest)
    corroborated = checked(ARCHIVE / "corroborated_reports.parquet", archive_manifest)
    reports = pd.concat(
        [
            normalize_reports(original, "nflverse_dated_reconstruction"),
            normalize_reports(corroborated, "corroborated_article_reconstruction"),
        ],
        ignore_index=True,
    )
    keys = ["player_id", "season", "week", "team", "usable_at"]
    # Contradictions at the same timestamp are unknown, not silently first/last wins.
    grouped = reports.groupby(keys, dropna=False)
    conflict = grouped.report_status.transform(lambda x: x.dropna().nunique()).gt(1)
    practice_conflict = grouped.practice_status.transform(lambda x: x.dropna().nunique()).gt(1)
    reports.loc[conflict, "report_status"] = "Conflict"
    reports.loc[practice_conflict, "practice_status"] = "Conflict"
    reports = reports.sort_values("source").drop_duplicates(keys, keep="first")
    audit = (
        original.groupby("season")
        .agg(rows=("gsis_id", "size"), dated=("date_modified", "count"))
        .reset_index()
    )
    return (
        reports,
        audit,
        {
            str(p): digest(p)
            for p in (
                SOURCES / "injuries.parquet",
                SOURCES / "manifest.json",
                ARCHIVE / "corroborated_reports.parquet",
                ARCHIVE / "manifest.json",
            )
        },
    )


def attach_reports(frame, reports, cutoff_column):
    left = frame[["player_id", "season", "week", "team", cutoff_column]].copy()
    left["_order"] = np.arange(len(left))
    left["cutoff"] = pd.to_datetime(left[cutoff_column], utc=True)
    right = reports.copy()
    right["usable_at"] = pd.to_datetime(right.usable_at, utc=True)
    # Report must refer to THIS player's THIS game week and pregame team.
    joined = pd.merge_asof(
        left.sort_values("cutoff"),
        right.sort_values("usable_at"),
        left_on="cutoff",
        right_on="usable_at",
        by=["player_id", "season", "week", "team"],
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_order")
    out = frame.copy()
    age = (joined.cutoff - joined.usable_at).dt.total_seconds() / 86400
    fresh = age.between(0, 7)
    known = fresh & joined.report_status.notna() & joined.report_status.ne("Conflict")
    practice = fresh & joined.practice_status.notna() & joined.practice_status.ne("Conflict")
    out["report_known"] = known.to_numpy(float)
    out["report_age_days"] = age.where(fresh).to_numpy()
    for status in ("Out", "Doubtful", "Questionable", "Probable"):
        out[f"report_{status.lower()}"] = np.where(known, joined.report_status.eq(status), np.nan)
    out["practice_known"] = practice.to_numpy(float)
    for name, text in (("dnp", "Did Not Participate"), ("limited", "Limited"), ("full", "Full")):
        out[f"practice_{name}"] = np.where(
            practice, joined.practice_status.fillna("").str.contains(text, case=False), np.nan
        )
    out["report_source"] = joined.source.where(fresh).to_numpy()
    out["report_usable_at"] = joined.usable_at.where(fresh).to_numpy()
    return out


def teammate_features(frame):
    """Sum only teammates' pregame expected usage and dated current-week designations."""
    out = frame.copy()
    usage = out[[f"prior4_{c}" for c in OPPORTUNITIES]].fillna(0).clip(lower=0).sum(axis=1)
    # All offensive teammates included; model can learn position-specific responses.
    group = [out.season, out.week, out.team]
    out["room_other_usage"] = usage.groupby(group).transform("sum") - usage
    out["room_own_usage_share"] = usage / usage.groupby(group).transform("sum").clip(lower=1)
    for key, flag in (
        ("out_usage", "report_out"),
        ("questionable_usage", "report_questionable"),
        ("dnp_usage", "practice_dnp"),
    ):
        value = usage * out[flag].fillna(0)
        out[f"room_{key}"] = value.groupby(group).transform("sum") - value
    flags = out.report_out.fillna(0)
    out["room_out_players"] = flags.groupby(group).transform("sum") - flags
    return out


def load_snaps():
    manifest_path = DATA / "raw/snapshots/canonical_20260923_r2/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mapping_manifest = json.loads((HISTORY / "manifest.json").read_text())
    mapping = checked(HISTORY / "mapping.parquet", mapping_manifest)
    counts = mapping.groupby("pfr_id").player_id.nunique()
    mapping = mapping[mapping.pfr_id.isin(counts[counts.eq(1)].index)].drop_duplicates("pfr_id")
    reverse = mapping.groupby("player_id").pfr_id.nunique()
    mapping = mapping[mapping.player_id.isin(reverse[reverse.eq(1)].index)]
    frames, hashes = (
        [],
        {
            str(manifest_path): digest(manifest_path),
            str(HISTORY / "mapping.parquet"): digest(HISTORY / "mapping.parquet"),
        },
    )
    for name, spec in manifest["assets"].items():
        if not name.endswith(".parquet") or not name.startswith("history/"):
            continue
        path = DATA / spec["object"]
        if "offense_snaps" not in pq.read_schema(path).names:
            continue
        if digest(path) != spec["sha256"]:
            raise ValueError("Snap source changed")
        f = pd.read_parquet(path)
        f = f[f.game_type.eq("REG") & f.season.le(2025)].copy()
        f = f.merge(
            mapping, left_on="pfr_player_id", right_on="pfr_id", how="left", validate="many_to_one"
        )
        frames.append(f)
        hashes[str(path)] = spec["sha256"]
    snaps = pd.concat(frames, ignore_index=True).dropna(subset=["player_id"])
    current, more = current_asset("snaps")
    snaps = pd.concat([snaps, current], ignore_index=True)
    hashes.update(more)
    ambiguous = snaps.duplicated(["player_id", "season", "week"], keep=False)
    quarantined = int(ambiguous.sum())
    snaps = snaps[~ambiguous].copy()
    snaps.attrs["quarantined_rows"] = quarantined
    snaps.attrs["mapped_player_ids"] = mapping.player_id.tolist()
    return snaps, hashes


def lag_workloads(frame):
    """Strictly earlier scheduled games; never impute a missing measured workload as zero."""
    out = frame.sort_values(["player_id", "season", "week"]).copy()
    for source, name in (
        ("label_snaps", "snaps"),
        ("label_snap_share", "snap_share"),
        ("label_played", "played"),
        *[(f"label_{c}", c) for c in OPPORTUNITIES],
    ):
        out[f"prior4_{name}"] = out.groupby("player_id")[source].transform(
            lambda s: s.shift().rolling(4, min_periods=1).mean()
        )
    return out.sort_values("row_id")


def prepare():
    manifest = json.loads((V3_RUN / "manifest.json").read_text())
    panel = checked(V3_RUN / "panel.parquet", manifest)
    panel["row_id"] = np.arange(len(panel))
    dataset = json.loads((V3_RUN / "dataset.json").read_text())
    if digest(V3_RUN / "dataset.json") != manifest["artifact_hashes"]["dataset.json"]:
        raise ValueError("Dataset changed")
    raw_features = [c for c in dataset["features"] if not c.startswith("injury_")]
    baseline = checked(V3_RUN / "mean_predictions.parquet", manifest)
    selection = json.loads((V3_RUN / "selection.json").read_text())
    if digest(V3_RUN / "selection.json") != manifest["artifact_hashes"]["selection.json"]:
        raise ValueError("V3 selection changed")
    baseline = pd.concat(
        [
            baseline[baseline.position.eq(p) & baseline.mean_method.eq(s["mean_method"])]
            for p, s in selection.items()
        ]
    )
    panel = panel[panel.scheduled_game.eq(1) & panel.season.ge(2013)].copy()
    panel = panel.merge(
        baseline[["row_id", "prediction"]].rename(columns={"prediction": "base_prediction"}),
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    history_manifest = json.loads((HISTORY / "manifest.json").read_text())
    box = checked(HISTORY / "box.parquet", history_manifest)
    current, current_hashes = current_asset("weeks")
    box = pd.concat([box, current], ignore_index=True)
    box = box[box.season_type.eq("REG")].copy()
    box = box.drop_duplicates(["player_id", "season", "week"])
    panel = panel.merge(
        box[["player_id", "season", "week", *OPPORTUNITIES]].rename(
            columns={c: f"label_{c}" for c in OPPORTUNITIES}
        ),
        on=["player_id", "season", "week"],
        how="left",
        validate="one_to_one",
    )
    for c in OPPORTUNITIES:
        # Known no appearance has zero use; unknown or malformed captured values remain missing.
        panel.loc[panel.target_points.notna() & ~panel.appeared, f"label_{c}"] = 0
        panel.loc[panel[f"label_{c}"].lt(0), f"label_{c}"] = np.nan
    snaps, hashes = load_snaps()
    hashes.update(current_hashes)
    panel = panel.merge(
        snaps[["player_id", "season", "week", "offense_snaps", "offense_pct"]].rename(
            columns={"offense_snaps": "label_snaps", "offense_pct": "label_snap_share"}
        ),
        on=["player_id", "season", "week"],
        how="left",
        validate="one_to_one",
    )
    # Absence inference requires a captured team game and a known identity in this source.
    team_counts = (
        snaps.assign(team=snaps.team.replace(TEAM_ALIASES))
        .groupby(["season", "week", "team"])
        .player_id.nunique()
    )
    support = (
        pd.MultiIndex.from_frame(panel[["season", "week", "team"]]).map(team_counts).to_numpy()
    )
    known_absence = (
        panel.label_snaps.isna()
        & ~panel.appeared
        & panel.target_points.notna()
        & pd.Series(support, index=panel.index).ge(20)
        & panel.player_id.isin(snaps.attrs["mapped_player_ids"])
    )
    panel.loc[known_absence, ["label_snaps", "label_snap_share"]] = 0
    played = (
        panel.label_snaps.gt(0)
        | panel[[f"label_{c}" for c in OPPORTUNITIES]].gt(0).any(axis=1)
        | panel.target_points.fillna(0).ne(0)
    )
    observed = panel.label_snaps.notna() | played
    panel["label_played"] = np.where(
        observed & panel.target_points.notna(), played.astype(float), np.nan
    )
    panel = lag_workloads(panel)
    schedule = checked(HISTORY / "schedule.parquet", history_manifest)
    current, more = current_asset("schedule")
    hashes.update(more)
    schedule = pd.concat([schedule, current], ignore_index=True).drop_duplicates(
        "game_id", keep="last"
    )
    schedule = schedule[schedule.game_type.eq("REG")].copy()
    local = pd.to_datetime(
        schedule.gameday.astype(str) + " " + schedule.gametime.fillna("00:00"), errors="coerce"
    )
    schedule["kickoff"] = local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    schedule.loc[schedule.gametime.isna(), "kickoff"] = pd.NaT
    panel = panel.merge(
        schedule[["game_id", "kickoff"]], on="game_id", how="left", validate="many_to_one"
    )
    panel["early_week"] = pd.to_datetime(panel.feature_cutoff, utc=True)
    panel["pregame_60m"] = panel.kickoff - pd.Timedelta(hours=1)
    reports, audit, report_hashes = load_reports()
    audit.attrs["ambiguous_snap_rows_quarantined"] = snaps.attrs["quarantined_rows"]
    hashes.update(report_hashes)
    for path in (
        V3_RUN / "panel.parquet",
        V3_RUN / "mean_predictions.parquet",
        V3_RUN / "selection.json",
        V3_RUN / "dataset.json",
        HISTORY / "box.parquet",
        HISTORY / "schedule.parquet",
    ):
        hashes[str(path)] = digest(path)
    frames = []
    for horizon in HORIZONS:
        available = panel[panel[horizon].notna() & panel.base_prediction.notna()].copy()
        part = teammate_features(attach_reports(available, reports, horizon))
        part["horizon"] = horizon
        frames.append(part)
    return pd.concat(frames, ignore_index=True), raw_features, reports, audit, hashes
