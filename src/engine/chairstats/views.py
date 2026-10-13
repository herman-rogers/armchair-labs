"""Manifest-driven views over saved campaign forecasts.

Nothing here selects a model. Every view takes the campaign (and, when needed, the
model, season and observation cutoff) from the caller. Views only assume what every
forecast row already guarantees: a ``player_id``, a ``predictions`` mapping with the
model's declared targets, and optional descriptive fields. Campaign kinds that
forecast something other than players simply do not appear in the player views.
"""

from __future__ import annotations

from typing import Any

from engine.chairstats.store import CampaignNotFound, CampaignStore

POINTS = "points"


def latest_forecast(model: dict) -> dict | None:
    """The most recent saved forecast for a model: latest season, then latest cutoff."""
    forecasts = model.get("forecasts", [])
    if not forecasts:
        return None
    return max(forecasts, key=lambda f: (f["season"], f["origin_week"]))


def horizons(manifest: dict) -> list[str]:
    seen: dict[str, None] = {}
    for model in manifest["models"]:
        seen.setdefault(model["horizon"], None)
    return list(seen)


def summary(manifest: dict) -> dict:
    """What the frontend needs to offer a campaign as a choice."""
    models = []
    for model in manifest["models"]:
        latest = latest_forecast(model)
        models.append(
            {
                "model_id": model["model_id"],
                "position": model.get("position"),
                "horizon": model["horizon"],
                "origin_week": model["origin_week"],
                "targets": model["targets"],
                "training_through_season": model["training_through_season"],
                "latest_forecast": (
                    {"season": latest["season"], "origin_week": latest["origin_week"]}
                    if latest
                    else None
                ),
            }
        )
    return {
        "campaign_id": manifest["campaign_id"],
        "name": manifest["name"],
        "description": manifest.get("description"),
        "kind": manifest.get("kind"),
        "disposition": manifest.get("disposition"),
        "generated_at": manifest.get("generated_at"),
        "manifest_sha256": manifest["manifest_sha256"],
        "positions": sorted({m["position"] for m in manifest["models"] if m.get("position")}),
        "horizons": horizons(manifest),
        "limitations": manifest.get("limitations", []),
        "models": models,
    }


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _row(model: dict, forecast: dict, row: dict, manifest: dict) -> dict:
    predictions = row["predictions"]
    points = _number(predictions.get(POINTS))
    games = _number(row.get("scheduled_games"))
    per_game = _number(row.get("points_per_scheduled_game"))
    if per_game is None and points is not None and games:
        per_game = points / games
    raw_availability = row.get("availability")
    availability: dict = raw_availability if isinstance(raw_availability, dict) else {}
    distribution = row.get("distribution", {})
    return {
        "campaign_id": manifest["campaign_id"],
        "model_id": model["model_id"],
        "player_id": row["player_id"],
        "player_display_name": row.get("player_name") or row.get("player_display_name"),
        "position": row.get("position") or model.get("position"),
        "team": row.get("team"),
        "campaign_name": manifest["name"],
        "season": forecast["season"],
        "origin_week": forecast["origin_week"],
        "rolling_origin": forecast["origin_week"] != model["origin_week"],
        "horizon": model["horizon"],
        "scheduled_games": games,
        "calendar_weeks": row.get("calendar_weeks"),
        "prediction": points,
        "points_per_game": per_game,
        "position_rank": row.get("rank"),
        "overall_rank": None,
        "predictions": predictions,
        "candidate_basis": row.get("candidate_basis"),
        "availability_status": availability.get("status"),
        "availability_as_of": availability.get("as_of"),
        "prediction_std": _number(distribution.get("std")),
        "prediction_interval_80": (
            [distribution["q10"], distribution["q90"]]
            if _number(distribution.get("q10")) is not None
            and _number(distribution.get("q90")) is not None
            else None
        ),
    }


def _model_forecast(model: dict, season: int | None, origin_week: int | None) -> dict | None:
    if season is None or origin_week is None:
        return latest_forecast(model)
    return next(
        (
            f
            for f in model.get("forecasts", [])
            if (f["season"], f["origin_week"]) == (season, origin_week)
        ),
        None,
    )


def rankings(
    store: CampaignStore,
    campaign_ids: list[str] | str,
    *,
    horizon: str,
    season: int | None = None,
    origin_week: int | None = None,
    position: str = "ALL",
    search: str = "",
    player_id: str | None = None,
) -> dict:
    """Models from one or more campaigns at one horizon, ranked together by points.

    The first campaign that covers a position at the horizon supplies it; later ones
    only fill positions the earlier ones lack. Campaigns are never added implicitly.
    Without an explicit season/cutoff each model
    contributes its latest saved forecast; the response lists every vintage.
    """
    ids = [campaign_ids] if isinstance(campaign_ids, str) else list(campaign_ids)
    manifests = [store.campaign(c) for c in ids]
    if not any(m["horizon"] == horizon for m in manifests[0]["models"]):
        raise CampaignNotFound(f"Campaign {ids[0]} has no {horizon!r} horizon")
    rows: list[dict] = []
    vintages = []
    covered: set[str | None] = set()
    for manifest in manifests:
        for model in manifest["models"]:
            if model["horizon"] != horizon:
                continue
            slot = model.get("position")
            if slot in covered or (position != "ALL" and slot != position):
                continue
            forecast = _model_forecast(model, season, origin_week)
            if forecast is None:
                continue
            result = store.forecast(
                manifest["campaign_id"],
                model["model_id"],
                forecast["season"],
                forecast["origin_week"],
            )
            covered.add(slot)
            vintages.append(
                {
                    "campaign_id": manifest["campaign_id"],
                    "campaign_name": manifest["name"],
                    "model_id": model["model_id"],
                    "position": slot,
                    "season": forecast["season"],
                    "origin_week": forecast["origin_week"],
                    "training_origin_week": model["origin_week"],
                    "rolling_origin": forecast["origin_week"] != model["origin_week"],
                    "generated_at": result["metadata"].get("generated_at"),
                    "artifact_sha256": result["artifact_sha256"],
                    "rows": len(result["rows"]),
                }
            )
            rows.extend(_row(model, forecast, row, manifest) for row in result["rows"])
    ranked = sorted(
        (r for r in rows if r["prediction"] is not None),
        key=lambda r: (-r["prediction"], r["player_id"]),
    )
    for index, row in enumerate(ranked, start=1):
        row["overall_rank"] = index
    rows = ranked + [r for r in rows if r["prediction"] is None]
    if player_id is not None:
        rows = [r for r in rows if r["player_id"] == player_id]
    if search:
        needle = search.lower()
        rows = [r for r in rows if needle in (r["player_display_name"] or "").lower()]
    primary = manifests[0]
    used = {v["campaign_id"]: v["campaign_name"] for v in vintages}
    limitations: list[str] = []
    for manifest in manifests:
        if manifest["campaign_id"] in used:
            limitations += [x for x in manifest.get("limitations", []) if x not in limitations]
    return {
        "campaign_id": primary["campaign_id"],
        "campaign_name": primary["name"],
        "campaigns": [{"campaign_id": c, "name": n} for c, n in used.items()],
        "manifest_sha256": primary["manifest_sha256"],
        "disposition": primary.get("disposition"),
        "horizon": horizon,
        "models": vintages,
        "limitations": limitations,
        "rankings": rows,
    }


def player(store: CampaignStore, player_id: str) -> dict:
    """One player's latest saved forecast from every model of every campaign."""
    forecasts = []
    for manifest in store.campaigns():
        for model in manifest["models"]:
            latest = latest_forecast(model)
            if latest is None:
                continue
            result = store.forecast(
                manifest["campaign_id"], model["model_id"], latest["season"], latest["origin_week"]
            )
            match = next((r for r in result["rows"] if r["player_id"] == player_id), None)
            if match is None:
                continue
            row = _row(model, latest, match, manifest)
            row["candidates"] = len(result["rows"])
            forecasts.append(row)
    return {"player_id": player_id, "forecasts": forecasts}


def targets(
    store: CampaignStore,
    campaign_id: str,
    model_id: str,
    *,
    season: int | None = None,
    origin_week: int | None = None,
) -> dict:
    """One model's forecast with every declared target as a column."""
    manifest = store.campaign(campaign_id)
    model = store.model(campaign_id, model_id)
    forecast = (
        latest_forecast(model)
        if season is None or origin_week is None
        else {"season": season, "origin_week": origin_week}
    )
    if forecast is None:
        raise CampaignNotFound(f"Model {model_id} has no saved forecasts")
    result = store.forecast(campaign_id, model_id, forecast["season"], forecast["origin_week"])
    rows = [_row(model, forecast, row, manifest) for row in result["rows"]]
    rows.sort(key=lambda r: (r["position_rank"] is None, r["position_rank"] or 0, r["player_id"]))
    return {
        "campaign_id": campaign_id,
        "campaign_name": manifest["name"],
        "manifest_sha256": manifest["manifest_sha256"],
        "model": {
            "model_id": model["model_id"],
            "position": model.get("position"),
            "horizon": model["horizon"],
            "targets": model["targets"],
            "scoring": model.get("scoring"),
            "training_through_season": model["training_through_season"],
        },
        "season": forecast["season"],
        "origin_week": forecast["origin_week"],
        "generated_at": result["metadata"].get("generated_at"),
        "artifact_sha256": result["artifact_sha256"],
        "rows": rows,
    }
