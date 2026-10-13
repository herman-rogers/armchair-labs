"""One application campaign, with verified TE artifacts and recoverable history."""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine.api import chairstats_routes
from engine.chairstats import CampaignStore, publish_campaign, views
from engine.chairstats.application import LATEST, OpeningDriveStore
from engine.chairstats.consolidate import RETIRED, TE_SOURCE, consolidate_opening_drive


def bundle(root, campaign_id, positions):
    work = root.parent / "inputs" / campaign_id
    work.mkdir(parents=True)
    metrics = work / "metrics.json"
    metrics.write_text('{"rmse": 4.2}')
    weights = work / "weights.joblib"
    weights.write_bytes(b"preserve trained weights without deserialization")
    models = []
    for position in positions:
        for horizon, origin in (("season", 0), ("remaining", 2)):
            model_id = f"{position.lower()}_{horizon}"
            forecasts = []
            for week in [0] if origin == 0 else [2, 4]:
                path = work / f"{model_id}_{week}.json"
                path.write_text(
                    json.dumps(
                        {
                            "campaign_id": campaign_id,
                            "model_id": model_id,
                            "season": 2026,
                            "origin_week": week,
                            "horizon": horizon,
                            "rows": [
                                {
                                    "player_id": position,
                                    "position": position,
                                    "season": 2026,
                                    "origin_week": week,
                                    "predictions": {"points": 123.4},
                                    "points_per_scheduled_game": 9.5,
                                }
                            ],
                        }
                    )
                )
                forecasts.append(
                    {
                        "season": 2026,
                        "origin_week": week,
                        "path": path,
                        "rolling_origin": week == 4,
                    }
                )
            models.append(
                {
                    "model_id": model_id,
                    "position": position,
                    "horizon": horizon,
                    "origin_week": origin,
                    "targets": ["points"],
                    "training_through_season": 2025,
                    "metrics": metrics,
                    "forecasts": forecasts,
                    "artifacts": {"weights": weights},
                }
            )
    publish_campaign(
        root,
        {
            "campaign_id": campaign_id,
            "name": campaign_id,
            "positions": positions,
            "targets": {p: ["points"] for p in positions},
            "scoring": {p: "PPR" for p in positions},
            "availability": {"version": 1},
        },
        models,
    )


@pytest.fixture
def registry(tmp_path):
    root = tmp_path / "campaigns"
    bundle(root, LATEST, ["QB", "RB", "WR"])
    bundle(root, TE_SOURCE, ["TE"])
    bundle(root, "opening_drive_v1", ["QB", "RB", "WR"])
    bundle(root, "scout_team_v1", ["QB", "RB"])
    return root, tmp_path / "campaign_revisions"


def test_merge_preserves_predictions_weights_and_archives_then_is_idempotent(registry):
    root, archive = registry
    store = CampaignStore(root)
    original = {m["campaign_id"]: m for m in store.campaigns()}
    old_rows = {}
    for campaign_id in (LATEST, TE_SOURCE):
        for model in original[campaign_id]["models"]:
            for ref in model["forecasts"]:
                key = model["model_id"], ref["origin_week"]
                old_rows[key] = store.forecast(campaign_id, key[0], 2026, key[1])["rows"]
    result = consolidate_opening_drive(root, archive)
    assert {r["campaign_id"] for r in result["archived"]} == set(RETIRED)
    assert [m["campaign_id"] for m in store.campaigns()] == [LATEST]
    manifest = store.campaign(LATEST)
    assert set(manifest["positions"]) == {"QB", "RB", "WR", "TE"}
    assert len(manifest["models"]) == 8
    assert manifest["availability"] == original[LATEST]["availability"]
    for model in manifest["models"]:
        source_id = TE_SOURCE if model["position"] == "TE" else LATEST
        source_model = next(
            m for m in original[source_id]["models"] if m["model_id"] == model["model_id"]
        )
        assert (
            model["artifacts"]["weights"]["sha256"]
            == source_model["artifacts"]["weights"]["sha256"]
        )
        for ref in model["forecasts"]:
            forecast = store.forecast(LATEST, model["model_id"], 2026, ref["origin_week"])
            assert forecast["rows"] == old_rows[model["model_id"], ref["origin_week"]]
            assert forecast["metadata"]["campaign_id"] == LATEST
    for campaign_id, old in original.items():
        saved = archive / campaign_id / old["manifest_sha256"]
        assert (saved / "manifest.sha256").read_text().strip() == old["manifest_sha256"]
    assert store.check(LATEST)["status"] == "verified"
    assert consolidate_opening_drive(root, archive)["archived"] == []
    assert store.campaign(LATEST)["manifest_sha256"] == manifest["manifest_sha256"]
    for horizon in ("remaining", "season"):
        ranks = views.rankings(store, LATEST, horizon=horizon)
        assert {r["position"] for r in ranks["rankings"]} == {"QB", "RB", "WR", "TE"}
        assert {r["campaign_id"] for r in ranks["rankings"]} == {LATEST}


def test_bad_te_checksum_does_not_change_or_retire_any_bundle(registry):
    root, archive = registry
    before = (root / LATEST / "manifest.json").read_bytes()
    (root / TE_SOURCE / "te_remaining/artifacts/weights.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        consolidate_opening_drive(root, archive)
    assert (root / LATEST / "manifest.json").read_bytes() == before
    assert all((root / campaign_id).exists() for campaign_id in RETIRED)


def test_missing_te_does_not_retire_v1(tmp_path):
    root = tmp_path / "campaigns"
    bundle(root, LATEST, ["QB", "RB", "WR"])
    bundle(root, "opening_drive_v1", ["QB"])
    with pytest.raises(ValueError, match="TE source"):
        consolidate_opening_drive(root, tmp_path / "archive")
    assert (root / "opening_drive_v1").exists()


def test_application_only_exposes_latest_even_with_old_bundles(registry, monkeypatch):
    root, archive = registry
    application = OpeningDriveStore(root)
    assert [m["campaign_id"] for m in application.campaigns()] == [LATEST]
    monkeypatch.setattr(
        chairstats_routes, "get_settings", lambda: SimpleNamespace(data_dir=root.parent)
    )
    app = FastAPI()
    app.include_router(chairstats_routes.router)
    client = TestClient(app)
    for retired in RETIRED:
        assert client.get(f"/api/chairstats/campaigns/{retired}").status_code == 404
    consolidate_opening_drive(root, archive)
    response = client.get("/api/chairstats/campaigns").json()["campaigns"]
    assert len(response) == 1 and response[0]["campaign_id"] == LATEST
    assert len(response[0]["models"]) == 8
    te = client.get(f"/api/chairstats/campaigns/{LATEST}/models/te_remaining/targets")
    assert te.status_code == 200 and te.json()["origin_week"] == 4
    player = client.get("/api/chairstats/players/TE").json()["forecasts"]
    assert {f["campaign_id"] for f in player} == {LATEST}
