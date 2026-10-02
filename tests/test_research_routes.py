from types import SimpleNamespace

import polars as pl
import pytest

from patron.api import research_routes as research


def test_actual_rank_keeps_unscored_winner_and_pending_unknown():
    rows = [
        {"player_id": "a", "score": 10, "actual": 5, "outcome_complete": True},
        {"player_id": "b", "score": None, "actual": 20, "outcome_complete": True},
        {"player_id": "c", "score": 10, "actual": 0, "outcome_complete": False},
    ]
    result = {r["player_id"]: r for r in research.compare_rows(rows, "score", "actual", "score")}
    assert result["b"]["actual_rank"] == 1
    assert result["b"]["model_rank"] is None
    assert result["a"]["actual_rank"] == 2
    assert result["a"]["model_rank"] == result["c"]["model_rank"] == 1
    assert result["c"]["actual_value"] is None
    assert result["c"]["rank_gap"] is None


def test_player_search_and_coverage_do_not_renumber(tmp_path, monkeypatch):
    path = tmp_path / "predictions.parquet"
    pl.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "player_display_name": ["Alpha", "Beta", "Missing"],
            "forecast_season": [2025] * 3,
            "position": ["WR"] * 3,
            "outcome_complete": [True] * 3,
            "fitted_season_points": [100.0, 80.0, None],
            "actual_season_points": [10.0, 20.0, 30.0],
        }
    ).write_parquet(path)
    monkeypatch.setattr(research, "prediction_path", lambda dataset="legacy": path)
    monkeypatch.setattr(
        research,
        "catalog",
        lambda dataset="legacy": {
            "saved_at": "2026-09-22",
            "seasons": [{"season": 2025, "complete": True}],
            "models": [{"id": "fitted_season_points", "seasons": [2025]}],
        },
    )
    row = research.players(2025, search="Beta")["players"][0]
    assert row["model_rank"] == 2
    assert row["actual_rank"] == 2
    missing = research.players(2025, coverage="missing")["players"][0]
    assert missing["actual_rank"] == 1


def test_scenario_preserves_baseline_and_changes_rank_with_weights():
    board = pl.DataFrame(
        {
            "player_id": ["a", "b"],
            "player_display_name": ["Alpha", "Beta"],
            "owner_team_id": [1, 2],
            "position": ["WR", "RB"],
            "forecast_season_points": [170.0, 136.0],
            "v2_overall_vor": [5.0, 4.0],
        }
    )
    teams = [SimpleNamespace(team_id=i, team_name=str(i), wins=1, losses=0) for i in [1, 2]]
    weights = {"WR": 1.0, "RB": 1.0}
    args = (board, teams, {"RB/WR/TE": 1}, {"a": 170.0, "b": 136.0}, "season points")
    baseline = research.scenario_teams(*args, weights, "production", 17)
    assert [r["team_id"] for r in baseline] == [1, 2]
    assert all(r["value_change"] == pytest.approx(0) for r in baseline)
    scenario = research.scenario_teams(*args, {**weights, "RB": 2.0}, "production", 17)
    assert scenario[0]["team_id"] == 2
    assert scenario[0]["rank_change"] == 1
    assert scenario[0]["scenario_value"] == pytest.approx(12)
    fallback = research.scenario_teams(
        board, teams, {"RB/WR/TE": 1}, {}, "season points", weights, "production", 17
    )
    assert all(r["fallback_players"] == 1 for r in fallback)
    assert all(r["value_change"] == 0 for r in fallback)
    excluded = research.scenario_teams(
        board, teams, {"RB/WR/TE": 1}, {}, "season points", weights, "exclude", 17
    )
    assert all(r["missing_players"] == 1 and r["scenario_value"] == 0 for r in excluded)


def test_current_observations_deduplicate_weeks_and_keep_unknown(monkeypatch):
    entry = SimpleNamespace(espn_id=7, points=12.0)
    future = SimpleNamespace(espn_id=7, points=100.0)
    snapshot = SimpleNamespace(
        season=2026,
        week=2,
        week_lineups=[
            SimpleNamespace(week=1, home_lineup=[entry], away_lineup=[entry]),
            SimpleNamespace(week=2, home_lineup=[entry], away_lineup=[]),
            SimpleNamespace(week=3, home_lineup=[future], away_lineup=[]),
        ],
    )
    state = SimpleNamespace(
        snapshot=snapshot,
        age_seconds=5,
        stale=False,
        tagged_board=pl.DataFrame({"player_id": ["a", "b"], "espn_id": [7, 8]}),
    )
    monkeypatch.setattr(research, "_state", lambda *args: state)
    rows = research.compare_rows(
        [
            {"player_id": "a", "score": 10},
            {"player_id": "b", "score": 20},
        ],
        "score",
        "actual",
        "score",
    )
    monkeypatch.setattr(research, "players", lambda **kwargs: {"players": rows})
    result = {r["player_id"]: r for r in research.current_players(None)["players"]}
    assert result["a"]["actual_value"] == 24
    assert result["a"]["actual_rank"] == 1
    assert result["b"]["actual_value"] is None
    assert result["b"]["actual_rank"] is None
