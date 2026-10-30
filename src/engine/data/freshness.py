"""Offline freshness checks over served inputs, independent of timer success.

No provider calls occur on requests. A recent provider check is distinct from an
accepted observation cutoff; a recent injury capture never refreshes game results.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb

from engine.chairstats.application import DEFAULT_CAMPAIGN
from engine.chairstats.store import CampaignNotFound, CampaignStore
from engine.data.capture import default_season
from engine.tables.application import connect, query_scope
from engine.tables.observations import observation_cutoff

CHECK_MAX_HOURS = 6
INJURY_MAX_HOURS = 2


def age_hours(value, now: datetime) -> float | None:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp > now:
            return None
        return (now - timestamp).total_seconds() / 3600
    except (AttributeError, TypeError, ValueError):
        return None


def read_receipt(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {"status": "unavailable"}
    except (OSError, ValueError):
        return {"status": "unavailable"}


def observation_state(data: Path, season: int, now: datetime) -> dict:
    with query_scope(data), connect() as db:
        cutoff = observation_cutoff(db)
        # A schedule warning is deliberately conservative: two full days after
        # the week's final scheduled date. It does not assert provider finality.
        rows = db.execute(
            "SELECT week, max(CAST(gameday AS DATE)) FROM analytics.current_schedule "
            "WHERE season = ? AND game_type = 'REG' GROUP BY week ORDER BY week",
            [season],
        ).fetchall()
    expected = 0
    deadline = (now - timedelta(days=2)).date()
    for week, last_game in rows:
        if week != expected + 1 or last_game is None or last_game > deadline:
            break
        expected = week
    return {**cutoff, "schedule_expected_through_week": expected}


def forecast_state(data: Path, campaign: str, season: int, origin: int) -> dict:
    manifest = CampaignStore(data / "campaigns").campaign(campaign)
    latest = {}
    missing = []
    for model in manifest.get("models", []):
        if model.get("position") not in ("QB", "RB", "WR", "TE"):
            continue
        if model.get("horizon") not in ("weekly", "remaining", "season"):
            continue
        name = f"{model['position']}/{model['horizon']}"
        origins = [
            f["origin_week"] for f in model.get("forecasts", []) if f.get("season") == season
        ]
        latest[name] = max(origins, default=-1)
    for position in ("QB", "RB", "WR", "TE"):
        for horizon in ("weekly", "remaining", "season"):
            name = f"{position}/{horizon}"
            if latest.get(name, -1) != origin:
                missing.append(name)
    return {
        "campaign": campaign,
        "generated_at": manifest.get("generated_at"),
        "required_origin_week": origin,
        "latest_origins": latest,
        "mismatched_models": missing,
        "stale": bool(missing),
    }


def freshness(
    data: Path,
    *,
    campaign: str = DEFAULT_CAMPAIGN,
    now: datetime | None = None,
    season: int | None = None,
) -> dict:
    now = now or datetime.now(UTC)
    season = season if season is not None else default_season()
    warnings = []
    refresh = read_receipt(data / ".runtime/observation_refresh.json")
    checked_age = age_hours(refresh.get("provider_checked_at"), now)
    refresh["provider_check_age_hours"] = checked_age
    if refresh.get("status") == "failed":
        warnings.append("The latest full refresh failed; saved inputs may be incomplete.")
    if checked_age is None or checked_age > CHECK_MAX_HOURS:
        warnings.append("Game results have not been checked with the provider in the last 6 hours.")
    try:
        observations = observation_state(data, season, now)
        week = observations.get("through_week") or 0
        old_season = observations.get("season") != season
        behind = week < observations["schedule_expected_through_week"]
        provider_week = refresh.get("provider_through_week", 0)
        if refresh.get("season") == season and provider_week > week:
            behind = True
        observations["stale"] = (
            old_season or behind or checked_age is None or checked_age > CHECK_MAX_HOURS
        )
        if old_season or behind:
            warnings.append("Published game results lag the expected completed week.")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, duckdb.Error) as exc:
        observations = {"stale": True, "issue": str(exc)}
        warnings.append("Published game results are unavailable.")
    try:
        origin = observations.get("through_week")
        if origin is None or observations.get("season") != season:
            raise ValueError("No current-season observation cutoff to verify forecasts")
        if origin >= 18:
            forecasts = {"campaign": campaign, "stale": False, "status": "season_complete"}
        else:
            forecasts = forecast_state(data, campaign, season, origin)
        if forecasts["stale"]:
            warnings.append(
                "Weekly, remaining-season or season forecasts lag the observation cutoff."
            )
    except (OSError, ValueError, KeyError, TypeError, CampaignNotFound) as exc:
        forecasts = {"campaign": campaign, "stale": True, "issue": str(exc)}
        warnings.append("Forecast freshness could not be verified.")
    captures = data / "research/opening_drive_v32_injury_captures"
    latest = max((p for p in captures.glob("*/summary.json")), default=None)
    injury = read_receipt(latest) if latest else {}
    injury_age = age_hours(injury.get("observed_at"), now)
    injuries = {
        "observed_at": injury.get("observed_at"),
        "age_hours": injury_age,
        "stale": injury_age is None or injury_age > INJURY_MAX_HOURS,
        "kind": "provider_observations_not_official_game_status",
    }
    if injuries["stale"]:
        warnings.append("Injury observations have not been collected in the last 2 hours.")
    return {
        "checked_at": now.isoformat(),
        "season": season,
        "stale": bool(warnings),
        "warnings": warnings,
        "refresh": refresh,
        "observations": observations,
        "forecasts": forecasts,
        "injuries": injuries,
    }
