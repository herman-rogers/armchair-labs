"""Read sealed weekly distributions without loading or fitting model weights."""

from __future__ import annotations

from engine.chairstats.store import CampaignNotFound, CampaignStore


def weekly_inputs(store: CampaignStore, campaign_id: str, season: int, week: int) -> dict:
    """Require all four positions at the exact observation cutoff for this week."""
    if not 1 <= week <= 18:
        raise ValueError("Weekly forecasts require a regular-season week")
    manifest = store.campaign(campaign_id)
    models = [m for m in manifest["models"] if m["horizon"] == "weekly"]
    if len(models) != 4 or {m["position"] for m in models} != {"QB", "RB", "WR", "TE"}:
        raise CampaignNotFound("A complete weekly campaign is not available")
    if any(
        not any(f["season"] == season and f["origin_week"] == week - 1 for f in m["forecasts"])
        for m in models
    ):
        raise CampaignNotFound(
            f"{manifest['name']} has no complete weekly forecast for {season} Week {week}. "
            f"Refresh data to update NFL observations and forecasts through Week {week - 1}."
        )
    players, receipts, calibrations = [], [], []
    for model in models:
        saved = store.forecast(campaign_id, model["model_id"], season, week - 1)
        for row in saved["rows"]:
            distribution = row.get("distribution")
            if not isinstance(distribution, dict):
                raise ValueError("Weekly forecasts require a saved predictive distribution")
            players.append(
                {
                    **distribution,
                    "player_id": row["player_id"],
                    "player_display_name": row.get("player_name") or row.get("player_display_name"),
                    "position": row["position"],
                    "team": row.get("team"),
                    "opponent": row.get("opponent"),
                    "prediction": row["predictions"]["points"],
                    "season": season,
                    "week": week,
                }
            )
        receipts.append(saved["metadata"])
        ref = model.get("artifacts", {}).get("lineup_calibration")
        if ref is not None:
            calibrations.append((ref["sha256"], store.artifact(campaign_id, ref)))
    if len({p["player_id"] for p in players}) != len(players):
        raise ValueError("Weekly positions contain duplicate player identities")
    if not calibrations or len({sha for sha, _ in calibrations}) != 1:
        raise ValueError("Weekly lineup calibration is missing or inconsistent")
    generated = [r["generated_at"] for r in receipts if r.get("generated_at")]
    return {
        "players": players,
        "calibration": calibrations[0][1],
        "campaign_id": campaign_id,
        "campaign_name": manifest["name"],
        "manifest_sha256": manifest["manifest_sha256"],
        "horizon": "weekly",
        "forecast_origins": [{"season": season, "origin_week": week - 1}],
        "through_week": week - 1,
        "published_at": max(generated) if len(generated) == len(models) else None,
        "limitations": manifest.get("limitations", []),
    }
