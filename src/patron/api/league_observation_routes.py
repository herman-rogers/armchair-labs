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


@router.get("/forecasts")
def forecasts(service: ServiceDep):
    from patron.api.ranking_routes import ranking_release, rankings
    from patron.config.settings import get_settings
    from patron.data.releases import load_gold
    from patron.data.weekly_points import load_weekly_points
    from patron.espn.forecast_log import record_forecast
    from patron.metrics.weekly_lineup import weekly_forecast

    snapshot, stale, age = snapshot_observations(service)
    _, manifest = ranking_release()
    settings = get_settings()
    gold = load_gold(settings.data_dir, manifest["gold"]["version"])
    ids = {}
    for row in gold.read("players").select("espn_id", "gsis_id").drop_nulls().to_dicts():
        try:
            ids[int(float(row["espn_id"]))] = row["gsis_id"]
        except (ValueError, TypeError):
            continue
    ranks = rankings(horizon="next4", limit=1000, offset=0)
    while len(ranks["rankings"]) < ranks["total"]:
        ranks["rankings"].extend(
            rankings(horizon="next4", limit=1000, offset=len(ranks["rankings"]))["rankings"]
        )
    try:
        weekly = load_weekly_points(settings.data_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"Weekly model unavailable: {exc}") from exc
    if weekly and weekly["manifest"]["analysis"]["version"] != ranks["version"]:
        raise HTTPException(409, "Analysis changed while loading weekly points; retry.")
    result = weekly_forecast(
        snapshot, ranks, ids, gold.read("current_schedule").to_dicts(), weekly=weekly
    )
    # Fail closed if publication changed between the two verified model reads.
    if result["version"] != manifest["version"]:
        raise HTTPException(409, "Analysis changed while forecasting; retry.")
    record = record_forecast(settings.outputs_dir / "weekly_forecasts", snapshot, result)
    return dict(record, stale=stale, age_seconds=round(age))


@router.get("/forecast-accuracy")
def forecast_accuracy(service: ServiceDep):
    from patron.config.settings import get_settings
    from patron.espn.forecast_log import accuracy

    snapshot, stale, age = snapshot_observations(service)
    return dict(
        accuracy(snapshot, get_settings().outputs_dir / "weekly_forecasts"),
        stale=stale,
        age_seconds=round(age),
    )


@router.get("/forecast-evidence")
def forecast_evidence(service: ServiceDep):
    from patron.config.settings import get_settings
    from patron.data.weekly_points import load_weekly_points

    try:
        product = load_weekly_points(get_settings().data_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"Weekly evidence unavailable: {exc}") from exc
    if product is None:
        raise HTTPException(404, "A dedicated weekly forecaster has not been published")
    from patron.data.releases import load_gold
    from patron.metrics.weekly_replay import season_replay

    snapshot, _, _ = snapshot_observations(service)
    gold = load_gold(get_settings().data_dir, product["manifest"]["gold"]["version"])
    ids = {}
    for row in gold.read("players").select("espn_id", "gsis_id").drop_nulls().to_dicts():
        try:
            ids[int(float(row["espn_id"]))] = row["gsis_id"]
        except (ValueError, TypeError):
            continue
    replay = season_replay(snapshot, product, ids, gold.read("current_schedule").to_dicts())
    return dict(product["report"], replay=replay)
