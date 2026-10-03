"""Dated, player-specific absence facts; independent of roster reconstruction.

Game numbers are regular-season team games, never calendar weeks (byes do not
serve suspensions). Each revision replaces one incident's entire game set. An
empty set clears that incident only; it is not evidence of overall availability.
These are date-granularity, end-of-day preseason reconstructions.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

EVIDENCE_FILE = "historical_absences.json"
AVAILABILITY_SCHEMA: dict[str, Any] = {
    "availability_evidence_status": pl.String,
    "known_absence_games": pl.Int64,
    "known_absence_game_numbers": pl.List(pl.Int64),
    "known_available_games_cap": pl.Float64,
    "known_suspension_games": pl.Int64,
    "availability_evidence_ids": pl.List(pl.String),
    "availability_source_urls": pl.List(pl.String),
    "availability_latest_known_on": pl.Date,
}


def season_games(season: int) -> int:
    if season < 2004:
        raise ValueError("Absence reconstruction supports forecast seasons 2004 onward")
    return 17 if season >= 2021 else 16


def validate_absences(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fail closed on malformed identity, chronology, or game scope."""
    seen = set()
    identities: dict[str, tuple[str, int]] = {}
    for event in events:
        for key in (
            "evidence_id",
            "absence_id",
            "player_id",
            "player_name",
            "source_url",
            "summary",
        ):
            if not isinstance(event.get(key), str) or not event[key].strip():
                raise ValueError(f"Absence evidence requires {key}")
        if event["evidence_id"] in seen:
            raise ValueError(f"Duplicate absence evidence: {event['evidence_id']}")
        seen.add(event["evidence_id"])
        if not event["source_url"].startswith("https://"):
            raise ValueError("Absence source requires an HTTPS URL")
        published = date.fromisoformat(event["source_published_on"])
        known = date.fromisoformat(event["known_on"])
        date.fromisoformat(event["verified_on"])
        if known < published:
            raise ValueError("Absence cannot be known before its supporting source")
        season = event["season"]
        if type(season) is not int:
            raise ValueError("Absence season must be an integer")
        identity = (event["player_id"], season)
        if identities.setdefault(event["absence_id"], identity) != identity:
            raise ValueError("An absence incident cannot change player or season")
        if event["kind"] not in {"suspension", "injury", "opt_out", "retirement"}:
            raise ValueError("Unsupported absence kind")
        games = event["unavailable_games"]
        if (
            not isinstance(games, list)
            or any(type(game) is not int or not 1 <= game <= season_games(season) for game in games)
            or len(games) != len(set(games))
        ):
            raise ValueError("Absences require unique regular-season team game numbers")
    return events


def load_absences(path: Path) -> list[dict[str, Any]]:
    # A missing ledger is a missing required input, not an empty verified archive.
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported absence evidence schema")
    events = validate_absences(document["events"])
    for event in events:
        filename = event.get("source_capture_file")
        if filename:
            if Path(filename).name != filename:
                raise ValueError("Invalid absence source capture path")
            raw = gzip.decompress(
                (path.parent.parent / "cache/historical_evidence/v1" / filename).read_bytes()
            )
            if hashlib.sha256(raw).hexdigest() != event.get("source_sha256"):
                raise ValueError("Absence source capture changed")
    return events


def attach_known_absences(
    forecasts: pl.DataFrame,
    events: list[dict[str, Any]],
    *,
    cutoff_by_season: dict[int, date] | None = None,
    default_cutoff: str = "08-31",
) -> pl.DataFrame:
    """Join evidence by stable ID and forecast date, including rookie candidates.

    Conflicting latest same-day revisions remain unresolved. Independent incidents
    with resolved evidence can still impose a cap; overlapping games count once.
    No transaction status or prior team is promoted to verified roster evidence.
    """
    validate_absences(events)
    inferred_cutoffs: dict[int, date] = {}
    if "forecast_cutoff_date" in forecasts.columns:
        for existing in (
            forecasts.select("forecast_season", "forecast_cutoff_date")
            .drop_nulls()
            .unique()
            .to_dicts()
        ):
            season, cutoff = int(existing["forecast_season"]), existing["forecast_cutoff_date"]
            if season in inferred_cutoffs and inferred_cutoffs[season] != cutoff:
                # Per-row cutoffs remain valid, but cannot supply a missing row's date.
                raise ValueError(
                    "Multiple forecast cutoffs per season; attach one window at a time"
                )
            inferred_cutoffs[season] = cutoff
    index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        index[event["player_id"], event["season"]].append(event)
    output = []
    cutoffs = []
    for row in forecasts.to_dicts():
        season = int(row["forecast_season"])
        cutoff = (
            row.get("forecast_cutoff_date")
            or (cutoff_by_season or {}).get(season)
            or inferred_cutoffs.get(season)
        )
        cutoff = date.fromisoformat(str(cutoff or f"{season}-{default_cutoff}"))
        if cutoff.year != season:
            raise ValueError("Forecast cutoff must belong to forecast season")
        cutoffs.append(cutoff)
        incidents: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in index.get((row["player_id"], season), []):
            if date.fromisoformat(event["known_on"]) <= cutoff:
                incidents[event["absence_id"]].append(event)
        selected = []
        games: set[int] = set()
        suspension_games: set[int] = set()
        suspension_resolved = False
        conflict = False
        resolved = False
        for revisions in incidents.values():
            latest = max(event["known_on"] for event in revisions)
            current = [event for event in revisions if event["known_on"] == latest]
            selected.extend(current)
            states = {
                (event["kind"], tuple(sorted(event["unavailable_games"]))) for event in current
            }
            if len(states) != 1:
                conflict = True
                continue
            resolved = True
            kind, unavailable = next(iter(states))
            games.update(unavailable)
            if kind == "suspension":
                suspension_games.update(unavailable)
                suspension_resolved = True
        status = (
            "conflicting_evidence"
            if conflict
            else "known_absence"
            if games
            else "specific_absence_cleared"
            if resolved
            else "unknown"
        )
        output.append(
            {
                "availability_evidence_status": status,
                "known_absence_games": len(games) if resolved else None,
                "known_absence_game_numbers": sorted(games),
                "known_available_games_cap": float(season_games(season) - len(games))
                if games
                else None,
                "known_suspension_games": len(suspension_games) if suspension_resolved else None,
                "availability_evidence_ids": sorted(event["evidence_id"] for event in selected),
                "availability_source_urls": sorted({event["source_url"] for event in selected}),
                "availability_latest_known_on": max(
                    (date.fromisoformat(event["known_on"]) for event in selected), default=None
                ),
            }
        )
    evidence = pl.DataFrame(output, schema=AVAILABILITY_SCHEMA)
    forecasts = forecasts.with_columns(
        pl.Series("forecast_cutoff_date", cutoffs, dtype=pl.Date), *evidence.get_columns()
    )
    # Retain original baseline values so reattaching a revised ledger is idempotent.
    for name in ("expected_games", "projected_availability"):
        if name not in forecasts.columns:
            continue
        original = f"{name}_before_absences"
        if original not in forecasts.columns:
            forecasts = forecasts.with_columns(pl.col(name).alias(original))
        cap = pl.col("known_available_games_cap")
        if name == "projected_availability":
            cap = cap / pl.when(pl.col("forecast_season") >= 2021).then(17).otherwise(16)
        forecasts = forecasts.with_columns(
            pl.when(pl.col(original).is_not_null() & cap.is_not_null())
            .then(pl.min_horizontal(pl.col(original), cap))
            .otherwise(pl.col(original))
            .alias(name)
        )
    return forecasts


def constrain_games(value: float, row: dict[str, Any]) -> float:
    """A known absence is a ceiling, not a second subtraction from estimated games."""
    cap = row.get("known_available_games_cap")
    if cap is None:
        return value
    if not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
        raise ValueError("Invalid known available-games cap")
    return min(value, float(cap))


def coverage_audit(forecasts: pl.DataFrame) -> tuple[dict[str, Any], pl.DataFrame]:
    """Coverage and a review queue ranked only by preseason evidence, not outcomes."""
    rows = forecasts.to_dicts()
    queue = []
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        status = row["availability_evidence_status"]
        roster = row.get("cutoff_state_resolution") or "no_transaction_feature"
        keys = (row["forecast_season"], row.get("position"), row.get("player_population"))
        group = groups.setdefault(
            keys,
            {
                "season": keys[0],
                "position": keys[1],
                "population": keys[2],
                "rows": 0,
                "known_absence": 0,
                "unknown": 0,
                "conflicting_evidence": 0,
                "specific_absence_cleared": 0,
                "observed_roster": 0,
            },
        )
        group["rows"] += 1
        group[status] += 1
        group["observed_roster"] += int(roster == "observed")
        reasons = []
        if status == "unknown":
            reasons.append("availability_not_reviewed")
        if status == "conflicting_evidence":
            reasons.append("conflicting_absence_evidence")
        if row.get("cutoff_suspended") == 1 and row["known_suspension_games"] is None:
            reasons.append("restriction_duration_or_classification_missing")
        if roster != "observed":
            reasons.append(roster)
        if reasons:
            queue.append(
                {
                    "forecast_season": row["forecast_season"],
                    "player_id": row["player_id"],
                    "player_display_name": row.get("player_display_name"),
                    "position": row.get("position"),
                    "forecast_cutoff_date": row["forecast_cutoff_date"],
                    "availability_evidence_status": status,
                    "roster_evidence_status": roster,
                    "review_reasons": reasons,
                    "priority": 0
                    if status == "conflicting_evidence"
                    or "restriction_duration_or_classification_missing" in reasons
                    else 1,
                    "prior_ppg": row.get("historical_ppg_prior"),
                }
            )
    review = pl.DataFrame(queue, infer_schema_length=None) if queue else pl.DataFrame()
    if review.height:
        review = review.sort(
            ["priority", "prior_ppg", "forecast_season", "player_id"],
            descending=[False, True, True, False],
            nulls_last=True,
        )
    counts = {
        status: sum(row["availability_evidence_status"] == status for row in rows)
        for status in (
            "known_absence",
            "unknown",
            "conflicting_evidence",
            "specific_absence_cleared",
        )
    }
    return {
        "forecast_rows": len(rows),
        "availability_evidence": counts,
        "roster_evidence": {
            str(r["cutoff_state_resolution"] or "no_transaction_feature"): r["len"]
            for r in forecasts.group_by("cutoff_state_resolution").len().to_dicts()
        }
        if "cutoff_state_resolution" in forecasts.columns
        else {"no_transaction_feature": len(rows)},
        "by_season_position_population": sorted(groups.values(), key=lambda r: str(r)),
        "review_queue_rows": review.height,
        "coverage_complete": False,
        "limitations": [
            "Curated positive absence evidence is not an exhaustive availability review.",
            "Observed roster transactions do not certify health or absence coverage.",
            "Unknown absence duration remains unknown; no default active status is imputed.",
            "Review priority uses prior production only, never realized forecast-season outcomes.",
        ],
    }, review


def audit_absence_constraints(
    forecasts: pl.DataFrame, events: list[dict[str, Any]], game_columns: list[str]
) -> dict[str, Any]:
    """Validate sourced constraints independently of the learned games estimates."""
    replay = attach_known_absences(forecasts, events)
    replay_mismatches = sum(
        original != rebuilt
        for original, rebuilt in zip(
            forecasts.select(*AVAILABILITY_SCHEMA).iter_rows(),
            replay.select(*AVAILABILITY_SCHEMA).iter_rows(),
            strict=True,
        )
    )
    violations = {
        name: forecasts.filter(
            pl.col("known_available_games_cap").is_not_null()
            & (pl.col(name) > pl.col("known_available_games_cap") + 1e-9)
        ).height
        for name in game_columns
        if name in forecasts.columns
    }
    future = forecasts.filter(
        pl.col("availability_latest_known_on") > pl.col("forecast_cutoff_date")
    ).height
    keys = set(forecasts.select("player_id", "forecast_season").iter_rows())
    return {
        "evidence_records": len(events),
        "replay_mismatches": replay_mismatches,
        "future_evidence_rows": future,
        "games_above_known_cap": violations,
        "evidence_without_candidate": sorted(
            {
                event["absence_id"]
                for event in events
                if (event["player_id"], event["season"]) not in keys
            }
        ),
        "accepted": replay_mismatches == 0 and future == 0 and not any(violations.values()),
    }
