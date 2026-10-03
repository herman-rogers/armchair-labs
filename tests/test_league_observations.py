"""Everyday league access must not load a retired forecast board."""

from types import SimpleNamespace

from patron.api.league_observation_routes import matchups, refresh
from patron.espn.observations import weekly_matchups
from patron.espn.sync import LeagueSnapshot, LineupEntry, ScheduleEntry, TeamState, WeekLineups


def snapshot():
    return LeagueSnapshot(
        captured_at="2026-09-23T00:00:00+00:00",
        league_id=1,
        league_name="Fixture",
        season=2026,
        week=3,
        my_team_id=1,
        regular_season_weeks=14,
        roster_slots={"QB": 1},
        teams=[
            TeamState(
                i,
                str(i),
                "Owner",
                1,
                1,
                100,
                schedule=[ScheduleEntry(2, 3 - i, 0.0, "U"), ScheduleEntry(4, 3 - i, 0.0, "U")],
            )
            for i in (1, 2)
        ],
        week_lineups=[
            WeekLineups(
                2,
                1,
                2,
                0.0,
                10.0,
                home_lineup=[
                    LineupEntry(11, "Historical starter", "QB", "QB", 0.0, 12.0, "SF", False),
                    LineupEntry(12, "Historical bench", "QB", "BE", 25.0, 20.0, "NYJ", False),
                ],
            )
        ],
    )


def test_historical_zero_and_bench_preserved_and_future_unknown():
    snap = snapshot()
    games = weekly_matchups(snap, 2)["matchups"]
    assert len(games) == 1  # Reciprocal schedule rows don't duplicate games.
    assert games[0]["home"]["score"] == 0.0
    assert [r["started"] for r in games[0]["home"]["lineup"]] == [True, False]
    future = weekly_matchups(snap, 4)["matchups"][0]
    assert future["status"] == "scheduled"
    assert future["home"]["score"] is None
    assert "espn_projection" not in future["home"]
    assert future["home"]["lineup"] == []


def test_observation_api_exposes_actuals_without_provider_predictions():
    game = weekly_matchups(snapshot(), 2)["matchups"][0]
    for side in (game["home"], game["away"]):
        assert "espn_projection" not in side
        for player in side["lineup"]:
            assert "projected_points" not in player
    assert game["home"]["lineup"][0]["points"] == 0.0
    assert game["home"]["lineup"][1]["points"] == 25.0


def test_routes_use_observations_only_and_refresh_is_explicit():
    calls = []

    def observations(**kwargs):
        calls.append(kwargs)
        return snapshot(), True, 45

    # No .get(), board, registry or model methods exist on this service.
    service = SimpleNamespace(observations=observations)
    assert matchups(service, week=2)["stale"] is True
    assert refresh(service)["age_seconds"] == 45
    assert calls == [{}, {"force": True}]


def test_draft_identity_is_linked_even_after_player_leaves_current_pool(monkeypatch, tmp_path):
    import polars as pl

    from patron.api import nextgen_routes as routes
    from patron.espn.crosswalk import JoinReport
    from patron.espn.sync import DraftPick

    snap = snapshot()
    snap.draft = [DraftPick(1, 1, 1, 1, "Mine", 99, "Drafted then dropped")]
    identities = pl.DataFrame({"espn_id": [99], "gsis_id": ["draft-id"]})
    pl.DataFrame(
        {
            "period": ["current"],
            "player_id": ["other"],
            "player_display_name": ["Other player"],
            "position": ["WR"],
        }
    ).write_parquet(tmp_path / "players.parquet")
    resolved = pl.DataFrame(
        {
            "espn_id": [11],
            "player_id": ["other"],
            "player_display_name": ["Other player"],
            "position": ["WR"],
            "owner_team_id": [1],
            "injury_status": ["ACTIVE"],
            "lineup_slot": ["WR"],
        }
    )
    monkeypatch.setattr(
        routes, "release", lambda: (tmp_path, {"version": "test", "gold": {"version": "gold"}})
    )
    monkeypatch.setattr(
        routes, "load_gold", lambda *a: SimpleNamespace(read=lambda table: identities)
    )
    monkeypatch.setattr(routes, "resolve_player_ids", lambda *a: (resolved, JoinReport()))
    monkeypatch.setattr(routes, "reviewed_news", lambda *a: ([], None))
    result = routes.league(SimpleNamespace(observations=lambda: (snap, False, 1)))
    assert result["draft"][0]["player_id"] == "draft-id"
    assert result["players"][0]["player_id"] == "other"
