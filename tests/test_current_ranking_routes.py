from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import HTTPException

from engine.api import ranking_routes as routes
from engine.data.releases import write_json
from engine.metrics.current_rankings import rank_frame


@pytest.fixture
def ranking_api(tmp_path, monkeypatch):
    # Route policy fixtures are independent of a real catalog and its response cache.
    monkeypatch.setattr(
        "engine.data.serving.get_settings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    entry = dict(
        id="ranking:rest_of_season:QB",
        target="league_points",
        horizon="rest_of_season",
        validity="verified",
        serving="approved",
        allowed_uses=["ranking", "forecast"],
        positions=["QB"],
        populations=["returner"],
        dependencies=["nfl_player_weeks.league_points"],
    )
    write_json(tmp_path / "registry.json", [entry])
    write_json(tmp_path / "incidents.json", [])
    write_json(tmp_path / "ranking_report.json", {"generated_at": "2026-09-23"})
    frame = pl.DataFrame(
        [
            dict(
                player_id=pid,
                player_display_name=name,
                horizon="rest_of_season",
                position="QB",
                population="returner",
                entry_id=entry["id"],
                prediction=points,
                rank_eligible=True,
            )
            for pid, name, points in [("a", "Alpha", 200.0), ("b", "Beta", 100.0)]
        ]
    )
    rank_frame(frame).write_parquet(tmp_path / "rankings.parquet")
    monkeypatch.setattr(
        routes,
        "release",
        lambda: (
            tmp_path,
            {"ranking_schema_version": 1, "version": "test", "gold": {"version": "test"}},
        ),
    )
    monkeypatch.setattr(
        routes,
        "load_gold",
        lambda *a: SimpleNamespace(
            manifest={"current_observations": {"season": 2026}},
            read=lambda _, **kwargs: pl.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        routes, "market_references", lambda *a: pl.DataFrame({"player_id": ["a", "b"]})
    )
    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    return tmp_path


def test_rankings_search_export_and_incident_use_one_policy(ranking_api):
    root = ranking_api
    response = routes.rankings(search="Beta", limit=1, offset=0)
    assert response["rankings"][0]["position_rank"] == 2
    export = routes.rankings(format="csv", limit=1, offset=0)
    assert b"Alpha" in export.body and b"Beta" in export.body
    write_json(
        root / "incidents.json",
        [
            dict(
                status="open",
                reason="Scoring incident",
                dependencies=["nfl_player_weeks.league_points"],
            )
        ],
    )
    response = routes.rankings(limit=100, offset=0)
    assert response["rankings"] == []
    assert response["excluded"][0]["reason"].startswith("Suspended")
    assert b"Alpha" not in routes.rankings(format="csv", limit=100, offset=0).body


def test_shadow_cannot_enter_rankings_or_exports(ranking_api):
    root = ranking_api
    import json

    entries = json.loads((root / "registry.json").read_text())
    entries[0]["serving"] = "shadow"
    write_json(root / "registry.json", entries)
    assert routes.rankings(limit=100, offset=0)["total"] == 0
    assert b"Alpha" not in routes.rankings(format="csv", limit=100, offset=0).body


def test_ranking_endpoint_rejects_wrong_horizons_and_unpublished_release(ranking_api, monkeypatch):
    root = ranking_api
    with pytest.raises(HTTPException) as exc:
        routes.rankings(horizon="preseason_season", limit=100, offset=0)
    assert exc.value.status_code == 422
    monkeypatch.setattr(routes, "release", lambda: (root, {"version": "old"}))
    with pytest.raises(HTTPException) as exc:
        routes.rankings(limit=100, offset=0)
    assert exc.value.status_code == 409
