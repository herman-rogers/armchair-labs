"""Application refresh retries missing forecasts and permits catalog publication."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from engine.data import refresh


@pytest.fixture
def current_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(refresh, "latest_snapshot", lambda data: {"college_source": "college"})
    source = SimpleNamespace(manifest={"current_observations": {"season": 2026, "through_week": 5}})
    monkeypatch.setattr(refresh, "load_tables", lambda data: source)
    monkeypatch.setattr(refresh, "completed_week", lambda season: 5)
    monkeypatch.setattr(refresh, "finish_tables", lambda *args: {"status": "current"})
    monkeypatch.setattr(refresh, "capture_raw", Mock(side_effect=AssertionError("No new week")))
    args = SimpleNamespace(season=2026, version="retry", force=False, no_rescore=False)
    return tmp_path, args


def test_current_observations_still_retry_forecasts(current_refresh, monkeypatch):
    _, args = current_refresh
    scorer = Mock(return_value={"returncode": 0})
    monkeypatch.setattr(refresh, "rescore_campaigns", scorer)
    result = refresh.run(args)
    scorer.assert_called_once()
    assert result["status"] == "current"
    assert result["rescore"]["returncode"] == 0


def test_failed_forecasts_are_reported_and_retryable(current_refresh, monkeypatch):
    data, args = current_refresh
    scorer = Mock(side_effect=[{"returncode": 1}, {"returncode": 0}])
    monkeypatch.setattr(refresh, "rescore_campaigns", scorer)
    with pytest.raises(RuntimeError, match="forecast generation failed"):
        refresh.run(args)
    state = json.loads((data / ".runtime/observation_refresh.json").read_text())
    assert state["status"] == "failed"
    assert state["rescore"]["returncode"] == 1
    assert refresh.run(args)["status"] == "current"


def test_no_rescore_option_is_preserved(current_refresh, monkeypatch):
    _, args = current_refresh
    args.no_rescore = True
    scorer = Mock(side_effect=AssertionError("Rescoring disabled"))
    monkeypatch.setattr(refresh, "rescore_campaigns", scorer)
    assert refresh.run(args)["status"] == "current"
    scorer.assert_not_called()


def test_preflight_failure_replaces_previous_success(current_refresh, monkeypatch):
    data, args = current_refresh
    monkeypatch.setattr(refresh, "rescore_campaigns", lambda _: {"returncode": 0})
    successful = refresh.run(args)
    monkeypatch.setattr(refresh, "load_tables", Mock(side_effect=ValueError("Invalid catalog")))
    with pytest.raises(ValueError, match="Invalid catalog"):
        refresh.run(args)
    failed = json.loads((data / ".runtime/observation_refresh.json").read_text())
    assert failed["status"] == "failed"
    assert failed["stage"] == "preflight"
    assert failed["last_success_at"] == successful["completed_at"]


def test_provider_check_is_recorded_even_when_rescore_fails(current_refresh, monkeypatch):
    data, args = current_refresh
    monkeypatch.setattr(refresh, "rescore_campaigns", lambda _: {"returncode": 1})
    with pytest.raises(RuntimeError):
        refresh.run(args)
    state = json.loads((data / ".runtime/observation_refresh.json").read_text())
    assert state["provider_through_week"] == 5
    assert state["provider_checked_at"]
    assert state["stage"] == "forecasts"


@pytest.mark.parametrize("failure", [False, True])
def test_league_refresh_updates_data_without_pinning_old_catalog(monkeypatch, failure):
    from engine.api import app as api
    from engine.api import league_routes

    snapshot = SimpleNamespace(captured_at="2026-10-13T14:00:00+00:00")
    snapshots = Mock(return_value=(snapshot, False, 0))
    monkeypatch.setattr(league_routes, "snapshot_observations", snapshots)
    update = Mock(side_effect=RuntimeError("Forecast failed") if failure else None)
    monkeypatch.setattr(refresh, "refresh_local", update)
    monkeypatch.setattr(api, "catalog_token", Mock(side_effect=AssertionError("Read-only pin")))
    client = TestClient(api.app)
    response = client.post("/api/league/refresh", headers={"X-Data-Catalog": "old-catalog"})
    update.assert_called_once_with()
    assert snapshots.call_args.kwargs == {"force": True}
    if failure:
        assert response.status_code == 503
        assert "ESPN refreshed; Forecast failed" in response.json()["detail"]
    else:
        assert response.status_code == 200
        assert response.json()["captured_at"] == snapshot.captured_at
