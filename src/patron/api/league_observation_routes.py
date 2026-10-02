"""Everyday league tools operate on ESPN observations without archived model boards."""

from fastapi import APIRouter, HTTPException, Query

from patron.api.league_routes import ServiceDep
from patron.espn.observations import weekly_matchups
from patron.espn.service import NotAuthenticatedError

router = APIRouter(prefix="/api/nextgen/league", tags=["league observations"])


def snapshot_observations(service, *, force=False):
    try:
        return service.observations(force=True) if force else service.observations()
    except NotAuthenticatedError as exc:
        raise HTTPException(401, str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/matchups")
def matchups(service: ServiceDep, week: int | None = Query(None, ge=1, le=25)):
    snapshot, stale, age = snapshot_observations(service)
    return {
        **weekly_matchups(snapshot, week or snapshot.week),
        "stale": stale,
        "age_seconds": round(age),
    }


@router.post("/refresh")
def refresh(service: ServiceDep):
    snapshot, stale, age = snapshot_observations(service, force=True)
    return {"captured_at": snapshot.captured_at, "stale": stale, "age_seconds": round(age)}
