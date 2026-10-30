import json
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v4.horizons import TARGETS
from experiments.future_player_lab.campaigns.weekly_v41 import publisher
from experiments.future_player_lab.campaigns.weekly_v41.track import merge_rows
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine.api import chairstats_routes
from engine.chairstats import views
from engine.chairstats.application import OpeningDriveStore
from engine.chairstats.weekly import weekly_inputs

SEASON, EARNED = 2026, 30.0


def projection(origin, bump=0.0):
    """Two players per position, three future games, coherent across horizons."""
    output = {h: [] for h in ("weekly", "remaining", "season")}
    future = [origin + 1, origin + 2, origin + 3]
    for position, targets in TARGETS.items():
        for i in range(2):
            pid = f"{position}{i}"
            weekly_points, later = 10.0 + i + bump, 8.0 + i
            weekly_c = {t: 2.0 for t in targets}
            later_c = {t: 1.0 for t in targets}
            curve = [
                dict(
                    week=w,
                    points=weekly_points if w == origin + 1 else later,
                    components=weekly_c if w == origin + 1 else later_c,
                    basis="v41_weekly" if w == origin + 1 else "v41_learned_horizon_extension",
                )
                for w in future
            ]
            remaining = {t: sum(g["components"][t] for g in curve) for t in targets}
            remaining["points"] = sum(g["points"] for g in curve)
            plan = dict(
                method=publisher.CAMPAIGN + "_residual_components",
                future_weeks=curve,
                remaining_points=remaining["points"],
                actual_points_to_date=EARNED,
                actual_components_to_date={t: 3.0 for t in targets},
                remaining_scheduled_games=3,
                season_scheduled_games=origin + 3 - 1,
                anchor_points_per_game=later,
                anchor_basis="v41_learned_residual",
            )
            for horizon, count in (("weekly", 1), ("remaining", 3), ("season", origin + 2)):
                values = (
                    dict(weekly_c, points=weekly_points)
                    if horizon == "weekly"
                    else remaining
                    if horizon == "remaining"
                    else {t: v + (EARNED if t == "points" else 3.0) for t, v in remaining.items()}
                )
                points = values["points"]
                row = dict(
                    player_id=pid,
                    player_name=f"Player {pid}",
                    position=position,
                    team="KC",
                    season=SEASON,
                    origin_week=origin,
                    horizon=horizon,
                    predictions=values,
                    scheduled_games=count,
                    points_per_scheduled_game=points / count,
                    base_campaign=publisher.CAMPAIGN,
                    distribution=dict(
                        samples=[points - 1, points + 1],
                        std=1.0,
                        q10=points - 1,
                        q90=points + 1,
                    ),
                )
                if horizon != "weekly":
                    row["projection"] = plan
                output[horizon].append(row)
    output["provenance"] = dict(lineage="test")
    return output


@pytest.fixture
def research(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps(dict(kind="opening_drive_v41_research")))
    (run / "protocol.json").write_text("{}")
    pd.DataFrame(dict(position=["QB"], rmse=[1.0])).to_csv(run / "horizon_scores.csv", index=False)
    pd.DataFrame(dict(position=["QB"], crps=[1.0])).to_csv(
        run / "uncertainty_scores.csv", index=False
    )
    pd.DataFrame(dict(a=["v41"], b=["control_v4"])).to_csv(
        run / "judging_horizons.csv", index=False
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}")
    return run, snapshot


def publish(tmp_path, research, origin, bump=0.0):
    run, snapshot = research
    players = pd.DataFrame(dict(player_id=["QB0"], forecast_horizon=["early_week"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # coherence checks must pass, not just warn
        return publisher.publish(
            tmp_path / "data/campaigns",
            projection(origin, bump),
            players,
            run,
            snapshot,
            archive=tmp_path / "archive",
            outputs=tmp_path / "outputs",
        )


def test_publish_accumulates_origins_and_serves(tmp_path, research, monkeypatch):
    registry = tmp_path / "data/campaigns"
    assert publish(tmp_path, research, 6)["status"] == "published"
    assert publish(tmp_path, research, 7)["status"] == "published"
    publish(tmp_path, research, 7, bump=1.0)  # same origin replaces

    store = OpeningDriveStore(registry)
    manifest = store.campaign(publisher.CAMPAIGN)
    assert manifest["status"] == "active"
    assert manifest["horizon_contract"] == dict(
        version=1, components_required=True, base_campaign=publisher.CAMPAIGN
    )
    assert len(manifest["models"]) == 12
    for model in manifest["models"]:
        assert model["training_through_season"] == 2025
        assert [f["origin_week"] for f in model["forecasts"]] == [6, 7]
        assert {"lineup_calibration", "projection_policy", "projection_evaluation"} <= set(
            model["artifacts"]
        )
    assert publisher.CAMPAIGN in [m["campaign_id"] for m in store.campaigns()]
    weekly = store.forecast(publisher.CAMPAIGN, "qb_weekly", SEASON, 7)
    assert weekly["metadata"]["forecast_kind"] == "opening_drive_v41_weekly"
    assert weekly["rows"][0]["predictions"]["points"] == 12.0  # replaced, ranked first

    ranked = views.rankings(store, publisher.CAMPAIGN, horizon="remaining")
    assert len(ranked["rankings"]) == 8 and ranked["rankings"][0]["origin_week"] == 7
    older = views.rankings(
        store, publisher.CAMPAIGN, horizon="season", season=SEASON, origin_week=6
    )
    assert {r["origin_week"] for r in older["rankings"]} == {6}

    inputs = weekly_inputs(store, publisher.CAMPAIGN, SEASON, 8)
    assert len(inputs["players"]) == 8
    assert inputs["calibration"]["campaign_id"] == publisher.CAMPAIGN

    monkeypatch.setattr(
        chairstats_routes, "get_settings", lambda: SimpleNamespace(data_dir=registry.parent)
    )
    app = FastAPI()
    app.include_router(chairstats_routes.router)
    client = TestClient(app)
    listed = client.get("/api/chairstats/campaigns").json()["campaigns"]
    assert [c["campaign_id"] for c in listed] == [publisher.CAMPAIGN]
    response = client.get(
        "/api/chairstats/rankings",
        params=dict(campaign=publisher.CAMPAIGN, horizon="weekly", market=False),
    )
    assert response.status_code == 200
    assert len(response.json()["rankings"]) == 8


def test_pregame_merge_replaces_only_updated_players():
    base = pd.DataFrame(dict(player_id=["a", "b", "c"], prediction=[1.0, 2.0, 3.0]))
    update = pd.DataFrame(dict(player_id=["c", "x"], prediction=[9.0, 9.0]))
    merged = merge_rows(base, update)
    assert merged.player_id.tolist() == ["a", "b", "c"]
    assert merged.prediction.tolist() == [1.0, 2.0, 9.0]
