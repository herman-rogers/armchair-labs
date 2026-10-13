"""Sealed weekly input contracts fail closed on missing origins and corrupt JSON."""

import json

import pytest

from engine.chairstats.application import OpeningDriveStore
from engine.chairstats.store import CampaignNotFound, CampaignStore, publish_campaign
from engine.chairstats.weekly import weekly_inputs


@pytest.fixture
def weekly_registry(tmp_path):
    root = tmp_path / "campaigns"
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"correlations": [], "special_teams": {}}))
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}")
    models = []
    for pos in ("QB", "RB", "WR", "TE"):
        path = tmp_path / f"{pos}.json"
        path.write_text(
            json.dumps(
                {
                    "campaign_id": "opening_drive_v3",
                    "season": 2026,
                    "origin_week": 4,
                    "generated_at": "2026-10-11T12:00:00+00:00",
                    "rows": [
                        {
                            "player_id": pos,
                            "position": pos,
                            "season": 2026,
                            "origin_week": 4,
                            "predictions": {"points": 12},
                            "distribution": {"std": 5, "q10": 4, "q90": 20},
                        }
                    ],
                }
            )
        )
        models.append(
            dict(
                model_id=f"{pos.lower()}_weekly",
                position=pos,
                horizon="weekly",
                origin_week=0,
                targets=["points"],
                training_through_season=2025,
                metrics=metrics,
                artifacts={"lineup_calibration": calibration},
                forecasts=[dict(season=2026, origin_week=4, rolling_origin=True, path=path)],
            )
        )
    publish_campaign(root, dict(campaign_id="opening_drive_v3", name="Opening Drive v3"), models)
    return root


def test_exact_week_inputs_and_application_access(weekly_registry):
    store = OpeningDriveStore(weekly_registry)
    assert [m["campaign_id"] for m in store.campaigns()] == ["opening_drive_v3"]
    result = weekly_inputs(store, "opening_drive_v3", 2026, 5)
    assert len(result["players"]) == 4
    assert result["through_week"] == 4
    assert {r["week"] for r in result["players"]} == {5}
    assert result["published_at"] == "2026-10-11T12:00:00+00:00"
    with pytest.raises(CampaignNotFound, match="2026 Week 6.*Refresh data.*Week 5"):
        weekly_inputs(store, "opening_drive_v3", 2026, 6)


def test_tampered_calibration_cannot_be_served(weekly_registry):
    store = CampaignStore(weekly_registry)
    model = store.model("opening_drive_v3", "qb_weekly")
    path = weekly_registry / "opening_drive_v3" / model["artifacts"]["lineup_calibration"]["path"]
    path.write_text('{"correlations":[{"correlation":1}]}')
    with pytest.raises(ValueError, match="checksum"):
        weekly_inputs(store, "opening_drive_v3", 2026, 5)


def test_unknown_campaign_and_postseason_rejected(weekly_registry):
    store = OpeningDriveStore(weekly_registry)
    with pytest.raises(CampaignNotFound):
        store.campaign("scout_team_v1")
    with pytest.raises(ValueError, match="regular-season"):
        weekly_inputs(store, "opening_drive_v3", 2026, 19)


def test_scoring_timeline_uses_weekly_only_for_its_target_week():
    from engine.metrics.scoring_trends import ForecastTimeline

    timeline = object.__new__(ForecastTimeline)
    timeline.cells = {
        "QB": [
            {"origin": 2, "horizon": "remaining", "model_id": "original"},
            {"origin": 4, "horizon": "remaining", "model_id": "rescored"},
            {"origin": 4, "horizon": "weekly", "model_id": "weekly"},
        ]
    }
    assert timeline.cell("QB", 4)["model_id"] == "original"
    assert timeline.cell("QB", 5)["model_id"] == "weekly"
    assert timeline.cell("QB", 6)["model_id"] == "rescored"
