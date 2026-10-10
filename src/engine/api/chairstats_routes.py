"""ChairStats: every saved campaign model, queryable at any time.

There is no selected or promoted model. The request names the campaign; the
response carries the manifest checksum it was read from.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response

from engine.chairstats import CampaignStore, views
from engine.chairstats.store import CampaignNotFound
from engine.config.settings import get_settings

router = APIRouter(prefix="/api/chairstats", tags=["ChairStats"])


def store() -> CampaignStore:
    return CampaignStore(get_settings().data_dir / "campaigns")


def _read(operation):
    try:
        return operation()
    except CampaignNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, "Campaign artifacts are unavailable or invalid.") from exc


def _campaign(campaign_id: str, request: Request, response: Response):
    manifest = _read(lambda: store().campaign(campaign_id))
    token = manifest["manifest_sha256"]
    requested = request.headers.get("X-Campaign-Manifest")
    if requested is not None and requested != token:
        raise HTTPException(409, "Campaign manifest changed; refresh campaign metadata.")
    response.headers["X-Campaign-Manifest"] = token
    response.headers["X-Artifact-Disposition"] = manifest.get("disposition", "research_preview")
    response.headers["Cache-Control"] = "no-cache"
    return manifest


@router.get("/campaigns")
def campaigns(response: Response):
    """Every available campaign with its models and latest saved forecasts."""
    response.headers["Cache-Control"] = "no-cache"
    manifests = _read(lambda: store().campaigns())
    return {"campaigns": [views.summary(m) for m in manifests]}


@router.get("/campaigns/{campaign_id}")
def campaign(campaign_id: str, request: Request, response: Response):
    return _campaign(campaign_id, request, response)


@router.get("/campaigns/{campaign_id}/models")
def models(campaign_id: str, request: Request, response: Response):
    manifest = _campaign(campaign_id, request, response)
    return {"campaign_id": campaign_id, "models": manifest["models"]}


@router.get("/campaigns/{campaign_id}/models/{model_id}")
def model(campaign_id: str, model_id: str, request: Request, response: Response):
    _campaign(campaign_id, request, response)
    return _read(lambda: store().model(campaign_id, model_id))


@router.get("/campaigns/{campaign_id}/models/{model_id}/metrics")
def metrics(campaign_id: str, model_id: str, request: Request, response: Response):
    _campaign(campaign_id, request, response)
    return {
        "campaign_id": campaign_id,
        "model_id": model_id,
        "metrics": _read(lambda: store().metrics(campaign_id, model_id)),
    }


@router.get("/campaigns/{campaign_id}/models/{model_id}/forecasts")
def forecasts(
    campaign_id: str,
    model_id: str,
    request: Request,
    response: Response,
    season: Annotated[int, Query(ge=2000, le=2100)],
    origin_week: Annotated[int, Query(ge=0, le=18)],
    player_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Query a saved forecast using an explicit season and observation cutoff.

    Per-game averages, when supplied, divide a horizon prediction by scheduled
    games; they are not separately trained weekly matchup predictions.
    """
    manifest = _campaign(campaign_id, request, response)
    result = _read(lambda: store().forecast(campaign_id, model_id, season, origin_week))
    rows = result["rows"]
    if player_id is not None:
        rows = [row for row in rows if row["player_id"] == player_id]
    return {
        "campaign_id": campaign_id,
        "model_id": model_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "artifact_sha256": result["artifact_sha256"],
        "metadata": result["metadata"],
        "disposition": manifest["disposition"],
        "season": season,
        "origin_week": origin_week,
        "total": len(rows),
        "limit": limit,
        "offset": offset,
        "rows": rows[offset : offset + limit],
    }


@router.get("/campaigns/{campaign_id}/models/{model_id}/targets")
def targets(
    campaign_id: str,
    model_id: str,
    request: Request,
    response: Response,
    season: Annotated[int | None, Query(ge=2000, le=2100)] = None,
    origin_week: Annotated[int | None, Query(ge=0, le=18)] = None,
):
    """One model's latest (or chosen) forecast with every declared target."""
    _campaign(campaign_id, request, response)
    return _read(
        lambda: views.targets(
            store(), campaign_id, model_id, season=season, origin_week=origin_week
        )
    )


@router.get("/rankings")
def rankings(
    request: Request,
    response: Response,
    campaign: str,
    horizon: str,
    season: Annotated[int | None, Query(ge=2000, le=2100)] = None,
    origin_week: Annotated[int | None, Query(ge=0, le=18)] = None,
    position: str = "ALL",
    search: str = "",
    player_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=2000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """All models of one campaign at one horizon, ranked together by predicted points."""
    _campaign(campaign, request, response)
    result = _read(
        lambda: views.rankings(
            store(),
            campaign,
            horizon=horizon,
            season=season,
            origin_week=origin_week,
            position=position,
            search=search,
            player_id=player_id,
        )
    )
    rows = result.pop("rankings")
    return {
        **result,
        "total": len(rows),
        "limit": limit,
        "offset": offset,
        "rankings": rows[offset : offset + limit],
    }


@router.get("/players/{player_id}")
def player(player_id: str, response: Response):
    """One player's latest forecast from every model of every campaign."""
    response.headers["Cache-Control"] = "no-cache"
    return _read(lambda: views.player(store(), player_id))
