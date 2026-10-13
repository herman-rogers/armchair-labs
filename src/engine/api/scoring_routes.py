"""Weekly actuals versus campaign pace, rebuilt from the current pinned inputs."""

from typing import Annotated

import duckdb
from fastapi import APIRouter, HTTPException, Query, Response

from engine.api.chairstats_routes import store
from engine.api.league_routes import ServiceDep, snapshot_observations
from engine.chairstats.application import DEFAULT_CAMPAIGN
from engine.chairstats.store import CampaignNotFound
from engine.metrics.scoring_trends import ForecastTimeline, league_trends, player_trends
from engine.tables import observations
from engine.tables.application import connect

router = APIRouter(prefix="/api/scoring", tags=["Scoring trends"])


def unavailable(exc):
    return HTTPException(404 if isinstance(exc, CampaignNotFound) else 409, str(exc))


@router.get("/players")
def players(
    response: Response,
    campaign: str = DEFAULT_CAMPAIGN,
    position: str = "ALL",
    search: str = "",
    limit: Annotated[int, Query(ge=1, le=48)] = 12,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    try:
        with connect() as db:
            cutoff = observations.observation_cutoff(db)
            season = cutoff["season"]
            if season is None:
                raise ValueError("Weekly observations are unavailable")
            schedule = observations.schedule(db, season)
            actuals = observations.scoring_weeks(db, season)
        timeline = ForecastTimeline(
            store(), campaign, season, cutoff["through_week"] or 0, schedule
        )
        result = player_trends(
            timeline, actuals, position=position, search=search, offset=offset, limit=limit
        )
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Campaign-Manifest"] = result["manifest_sha256"]
        return result
    except (OSError, ValueError, KeyError, TypeError, CampaignNotFound, duckdb.Error) as exc:
        raise unavailable(exc) from exc


@router.get("/league")
def league(response: Response, service: ServiceDep, campaign: str = DEFAULT_CAMPAIGN):
    snapshot, stale, age = snapshot_observations(service)
    try:
        with connect() as db:
            cutoff = observations.observation_cutoff(db)
            if cutoff["season"] != snapshot.season:
                raise ValueError("League and weekly observations are for different seasons")
            schedule = observations.schedule(db, snapshot.season)
            ids = observations.espn_ids(db)
        timeline = ForecastTimeline(
            store(),
            campaign,
            snapshot.season,
            min(cutoff["through_week"] or 0, snapshot.week - 1),
            schedule,
        )
        result = league_trends(timeline, snapshot, ids)
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Campaign-Manifest"] = result["manifest_sha256"]
        return dict(result, stale=stale, age_seconds=round(age))
    except (OSError, ValueError, KeyError, TypeError, CampaignNotFound, duckdb.Error) as exc:
        raise unavailable(exc) from exc
