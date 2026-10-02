"""Dated availability evidence. No inferred health, rewritten labels, or proportional caps."""

import hashlib
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ARCHIVE_VERSION = "injury_archive_20260925_r1"
ARCHIVE_MANIFEST_SHA256 = "fe52e77b885b6a38c4ff4ebb02f5dbfbb80ae5461c154bb0277684f36154ac11"


def selected_events(events, player_id, season, cutoff, include_uncertain=False):
    """Resolve revisions using knowledge dates only; conflicts do not impose rules."""
    incidents = defaultdict(list)
    for event in events:
        if event["player_id"] != player_id or int(event["season"]) != season:
            continue
        published, known = event.get("source_published_on"), event.get("known_on")
        if not published or not known or known < published:
            raise ValueError("Absence evidence needs ordered publication and knowledge dates")
        if known > cutoff:
            continue
        incidents[event["absence_id"]].append(event)
    selected, conflicts = [], []
    for incident, revisions in incidents.items():
        newest = max(e["known_on"] for e in revisions)
        current = [e for e in revisions if e["known_on"] == newest]
        states = {
            (
                e.get("certainty", "confirmed"),
                tuple(sorted(e.get("unavailable_games", []))),
                tuple(sorted(e.get("unavailable_weeks", []))),
            )
            for e in current
        }
        if len(states) != 1:
            conflicts.append(incident)
        elif current[0].get("certainty", "confirmed") in (
            {"confirmed", "uncertain"} if include_uncertain else {"confirmed"}
        ):
            selected.append(sorted(current, key=lambda e: e["evidence_id"])[0])
    return selected, conflicts


def annotate(panel, events, schedules, preseason, injuries=None):
    """Availability at the end of the day before the first predicted NFL week starts.

    Preseason uses its original dated forecast cutoff. A team-game absence can be
    mapped to calendar weeks only with an observed dated team. A complete season
    absence needs no team inference. Uncertain news and undated injury reports
    cannot force zeros. Output rows retain the exact order of the input panel.
    """
    schedule = schedules.filter(pl.col("game_type") == "REG")
    season_weeks = {}
    team_weeks = defaultdict(set)
    for row in schedule.to_dicts():
        year, week = int(row["season"]), int(row["week"])
        day = str(row["gameday"])[:10]
        season_weeks[year, week] = min(season_weeks.get((year, week), day), day)
        for team in [row["home_team"], row["away_team"]]:
            team_weeks[year, team].add(week)
    roster = {(r["player_id"], int(r["forecast_season"])): r for r in preseason.to_dicts()}
    event_index = defaultdict(list)
    for event in events:
        event_index[event["player_id"], int(event["season"])].append(event)
    report_index = defaultdict(list)
    if injuries is not None:
        for r in injuries.to_dicts():
            if r.get("gsis_id") and r.get("date_modified") is not None:
                report_index[r["gsis_id"], int(r["season"])].append(r)
    output = []
    for row in panel.to_dicts():
        pid, year, origin, end = (
            row["player_id"],
            int(row["year"]),
            int(row["origin"]),
            int(row["end"]),
        )
        if origin:
            first = season_weeks.get((year, origin + 1))
            if first is None:
                raise ValueError(f"No dated schedule for {year} week {origin + 1}")
            cutoff = (date.fromisoformat(first) - timedelta(days=1)).isoformat()
        else:
            cutoff = str(row["forecast_cutoff_date"])[:10]
        r = roster.get((pid, year), {})
        observed_team = (
            r.get("cutoff_preseason_team")
            if r.get("cutoff_state_observed")
            and str(r.get("forecast_cutoff_date", "9999"))[:10] <= cutoff
            else None
        )
        # A preseason team is only suitable for mapping preseason game ordinals.
        # Later forecasts require a current dated team; do not carry a stale team forward.
        if origin:
            observed_team = None
        known, conflicts = selected_events(
            event_index[pid, year], pid, year, cutoff, include_uncertain=True
        )
        confirmed = [e for e in known if e.get("certainty", "confirmed") == "confirmed"]
        season_games = 17 if year >= 2021 else 16
        unavailable_weeks, games = set(), set()
        evidence = []
        for event in known:
            evidence.append(event["evidence_id"])
            if event.get("certainty", "confirmed") != "confirmed":
                continue
            ordinal = set(event.get("unavailable_games", []))
            if not ordinal <= set(range(1, season_games + 1)):
                raise ValueError("Absence game ordinal outside season")
            games |= ordinal
            unavailable_weeks.update(event.get("unavailable_weeks", []))
            if observed_team:
                mapping = sorted(team_weeks.get((year, observed_team), []))
                if len(mapping) == season_games:
                    unavailable_weeks.update(mapping[i - 1] for i in ordinal)
        full_season = set(range(1, season_games + 1)) <= games
        forecast_weeks = set(range(origin + 1, end + 1))
        full_period = (full_season or forecast_weeks <= unavailable_weeks) and not conflicts
        # Only publication-dated, pre-cutoff structured observations become inputs.
        reports = [
            r
            for r in report_index[pid, year]
            if str(r["date_modified"])[:10] <= cutoff
            and int(r["week"]) <= end
            and r.get("game_type") == "REG"
        ]
        latest = max((str(r["date_modified"]) for r in reports), default=None)
        current = [r for r in reports if str(r["date_modified"]) == latest]
        recent = [
            r
            for r in reports
            if (date.fromisoformat(cutoff) - date.fromisoformat(str(r["date_modified"])[:10])).days
            <= 28
        ]
        output.append(
            dict(
                availability_row_key=f"{pid}|{year}|{origin}|{row.get('horizon', '')}",
                availability_cutoff=cutoff,
                availability_force_zero=full_period,
                availability_evidence_ids=evidence,
                availability_conflicts=conflicts,
                availability_known_absence_games=float(len(games)) if confirmed else None,
                availability_known_unavailable_weeks=float(len(forecast_weeks & unavailable_weeks))
                if confirmed
                else None,
                availability_confirmed_full_absence=float(full_period) if confirmed else None,
                availability_uncertain_absence=float(len(known) != len(confirmed))
                if known
                else None,
                availability_dated_reports=float(len(reports)),
                availability_recent_reports=float(len(recent)),
                availability_latest_out=float(
                    any(str(r.get("report_status", "")).lower() == "out" for r in current)
                )
                if any(r.get("report_status") for r in current)
                else None,
                availability_latest_dnp=float(
                    any("did not" in str(r.get("practice_status", "")).lower() for r in current)
                )
                if any(r.get("practice_status") for r in current)
                else None,
            )
        )
    return pl.DataFrame(output, infer_schema_length=None).with_columns(
        pl.col("availability_evidence_ids", "availability_conflicts").cast(pl.List(pl.String)),
        pl.col(FEATURES).cast(pl.Float64),
    )


FEATURES = [
    "availability_known_absence_games",
    "availability_known_unavailable_weeks",
    "availability_confirmed_full_absence",
    "availability_uncertain_absence",
    "availability_dated_reports",
    "availability_recent_reports",
    "availability_latest_out",
    "availability_latest_dnp",
]


def adjust_forecasts(forecasts, annotations):
    """Apply the identical zero rule to model and baseline; preserve raw forecasts."""
    if len(forecasts) != len(annotations):
        raise ValueError("Availability rows must align with forecasts")
    if {"player_id", "year", "origin", "horizon"} <= set(forecasts.columns):
        keys = [
            f"{r['player_id']}|{int(r['year'])}|{int(r['origin'])}|{r['horizon']}"
            for r in forecasts.to_dicts()
        ]
        if keys != annotations["availability_row_key"].to_list():
            raise ValueError("Availability identities/order do not match forecasts")
    f = forecasts.hstack(annotations)
    return f.with_columns(
        pl.col("prediction").alias("raw_prediction"),
        pl.col("persistence_prediction").alias("raw_persistence_prediction"),
    ).with_columns(
        pl.when(pl.col("availability_force_zero")).then(0.0).otherwise(pl.col(c)).alias(c)
        for c in ["prediction", "persistence_prediction"]
    )


def append_features(x, names, annotations):
    return np.column_stack(
        [x, annotations.select(FEATURES).to_numpy().astype(float)]
    ), names + FEATURES


def archive_location(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    return root, root / "data/research" / ARCHIVE_VERSION


def verified_manifest(archive, allow_unsealed=False):
    raw = (archive / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if allow_unsealed and not manifest.get("sealed"):
        return manifest
    if not manifest.get("sealed") or hashlib.sha256(raw).hexdigest() != ARCHIVE_MANIFEST_SHA256:
        raise ValueError("Injury archive is unsealed or its pinned manifest changed")
    return manifest


def archive_status():
    _, archive = archive_location()
    manifest = verified_manifest(archive)
    for name in ["audit_summary.json", "coverage_by_season.csv"]:
        if hashlib.sha256((archive / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Injury archive summary changed: {name}")
    return json.loads((archive / "audit_summary.json").read_text()), pl.read_csv(
        archive / "coverage_by_season.csv"
    )


def load_inputs(root=None, *, allow_unsealed=False):
    """Verify the versioned archive and pinned gold sources before training."""
    root, archive = archive_location(root)
    manifest = verified_manifest(archive, allow_unsealed=allow_unsealed)

    def verify(path, expected):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Availability source changed: {path}")

    for name in ["absence_events.json", "injury_reports.parquet"]:
        verify(archive / name, manifest["files"][name])
    gold = root / manifest["gold_root"]
    verify(gold / "manifest.json", manifest["gold_manifest_sha256"])
    tables = json.loads((gold / "manifest.json").read_text())["tables"]
    frames = []
    for name in ["nfl_schedule", "preseason_features"]:
        spec = tables[name]
        verify(gold / spec["path"], spec["sha256"])
        frames.append(pl.read_parquet(gold / spec["path"]))
    injuries = pl.read_parquet(archive / "injury_reports.parquet")
    if "corroborated_reports.parquet" in manifest["files"]:
        path = archive / "corroborated_reports.parquet"
        verify(path, manifest["files"][path.name])
        injuries = pl.concat([injuries, pl.read_parquet(path)], how="diagonal_relaxed")
    events = json.loads((archive / "absence_events.json").read_text())["events"]
    return events, *frames, injuries
