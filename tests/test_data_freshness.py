"""Collection recency must not hide old results or an incomplete campaign."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from engine.data import freshness as module

NOW = datetime(2026, 10, 27, 18, tzinfo=UTC)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def setup_state(tmp_path, monkeypatch, *, observed=7, expected=7):
    write(
        tmp_path / ".runtime/observation_refresh.json",
        {
            "status": "current",
            "provider_checked_at": NOW.isoformat(),
            "season": 2026,
            "provider_through_week": expected,
        },
    )
    write(
        tmp_path / "research/opening_drive_v32_injury_captures/latest/summary.json",
        {
            "observed_at": NOW.isoformat(),
        },
    )
    monkeypatch.setattr(
        module,
        "observation_state",
        lambda *args: {
            "season": 2026,
            "through_week": observed,
            "schedule_expected_through_week": expected,
        },
    )
    monkeypatch.setattr(module, "forecast_state", lambda *args: {"stale": False})


def test_new_injuries_do_not_hide_old_results(tmp_path, monkeypatch):
    setup_state(tmp_path, monkeypatch, observed=5)
    state = module.freshness(tmp_path, now=NOW, season=2026)
    assert state["stale"]
    assert state["observations"]["stale"]
    assert not state["injuries"]["stale"]
    assert any("game results lag" in message for message in state["warnings"])


def test_recent_receipt_is_not_a_substitute_for_provider_check(tmp_path, monkeypatch):
    setup_state(tmp_path, monkeypatch)
    write(
        tmp_path / ".runtime/observation_refresh.json",
        {
            "status": "current",
            "completed_at": NOW.isoformat(),
        },
    )
    state = module.freshness(tmp_path, now=NOW, season=2026)
    assert state["observations"]["stale"]
    assert any("not been checked" in message for message in state["warnings"])


def test_healthy_inputs_and_failed_run_are_independent(tmp_path, monkeypatch):
    setup_state(tmp_path, monkeypatch)
    assert not module.freshness(tmp_path, now=NOW, season=2026)["stale"]
    path = tmp_path / ".runtime/observation_refresh.json"
    payload = json.loads(path.read_text())
    payload["status"] = "failed"
    write(path, payload)
    state = module.freshness(tmp_path, now=NOW, season=2026)
    assert state["stale"]
    assert not state["injuries"]["stale"]
    assert "latest full refresh failed" in state["warnings"][0]


def test_each_position_and_horizon_must_match_cutoff(tmp_path, monkeypatch):
    models = [
        {
            "position": position,
            "horizon": horizon,
            "forecasts": [{"season": 2026, "origin_week": 7}],
        }
        for position in ("QB", "RB", "WR", "TE")
        for horizon in ("weekly", "remaining", "season")
    ]
    models[-1]["forecasts"][0]["origin_week"] = 5
    monkeypatch.setattr(
        module, "CampaignStore", lambda _: SimpleNamespace(campaign=lambda _: {"models": models})
    )
    state = module.forecast_state(tmp_path, "opening_drive_v32", 2026, 7)
    assert state["stale"]
    assert state["mismatched_models"] == ["TE/season"]


def test_missing_campaign_and_broken_catalog_do_not_break_health_endpoint(tmp_path, monkeypatch):
    from engine.api import app as api
    from engine.api import freshness_routes

    monkeypatch.setattr(
        freshness_routes, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    monkeypatch.setattr(
        module,
        "observation_state",
        lambda *args: {
            "season": 2026,
            "through_week": 7,
            "schedule_expected_through_week": 7,
        },
    )
    response = TestClient(api.app).get("/api/freshness?campaign=does_not_exist")
    assert response.status_code == 200
    assert response.json()["forecasts"]["stale"]


def test_future_or_naive_capture_timestamp_is_unknown():
    assert module.age_hours("2026-10-28T00:00:00+00:00", NOW) is None
    assert module.age_hours("2026-10-27T00:00:00", NOW) is None
