import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from engine.api import research_routes as research


def report_fixture(tmp_path, monkeypatch):
    data = {
        "season": 2026,
        "saved_at": "2026-09-22T12:00:00+00:00",
        "through_week": 2,
        "players": [
            {"player_id": "a", "espn_id": 1, "position": "WR"},
            {"player_id": "b", "espn_id": 2, "position": "WR"},
            {"player_id": "c", "espn_id": 3, "position": "RB"},
        ],
    }
    (tmp_path / "rookie_watch_2026.json").write_text(json.dumps(data))
    monkeypatch.setattr(research, "get_settings", lambda: SimpleNamespace(outputs_dir=tmp_path))
    monkeypatch.setattr(research, "get_league", lambda: SimpleNamespace(draft_season=2026))
    return data


def test_rookie_ownership_distinguishes_unobserved_and_free_agent(tmp_path, monkeypatch):
    report_fixture(tmp_path, monkeypatch)
    state = SimpleNamespace(
        snapshot=SimpleNamespace(
            season=2026,
            week=4,
            my_team_id=3,
            captured_at="2026-09-22",
            players=[
                SimpleNamespace(
                    espn_id=1,
                    owner_team_id=None,
                    owner_team_name=None,
                    injury_status="ACTIVE",
                    espn_draft_rank=40,
                ),
                SimpleNamespace(
                    espn_id=3,
                    owner_team_id=3,
                    owner_team_name="Mine",
                    injury_status="ACTIVE",
                    espn_draft_rank=80,
                ),
            ],
        ),
        stale=False,
    )
    monkeypatch.setattr(research, "_state", lambda *args: state)
    result = research.rookie_watch(None)
    rows = result["players"]
    assert [r["availability"] for r in rows] == ["free_agent", "unknown", "rostered"]
    assert rows[2]["is_mine"] and not rows[0]["is_mine"]
    assert result["newer_week_possible"]
    assert len(research.rookie_watch(None, position="WR")["players"]) == 2


def test_rookies_remain_visible_without_private_league(tmp_path, monkeypatch):
    report_fixture(tmp_path, monkeypatch)

    def unavailable(*args):
        raise HTTPException(401, "Not signed in")

    monkeypatch.setattr(research, "_state", unavailable)
    data = research.rookie_watch(None)
    assert len(data["players"]) == 3
    assert not data["ownership_available"]
    assert all(r["availability"] == "unknown" for r in data["players"])


def test_absent_artifact_does_not_fabricate_forecasts(tmp_path, monkeypatch):
    report_fixture(tmp_path, monkeypatch)
    (tmp_path / "rookie_watch_2026.json").unlink()
    with pytest.raises(HTTPException) as error:
        research.rookie_watch(None)
    assert error.value.status_code == 404
