"""Weekly lineup forecasts spread a campaign's horizon forecast per scheduled game."""

from engine.espn.sync import LeagueSnapshot, LineupEntry, TeamState, WeekLineups
from engine.metrics.weekly_lineup import weekly_forecast


def snapshot(week=3):
    return LeagueSnapshot(
        captured_at="2026-09-23T00:00:00+00:00",
        league_id=1,
        league_name="Fixture",
        season=2026,
        week=week,
        my_team_id=1,
        regular_season_weeks=14,
        roster_slots={"QB": 1, "K": 1},
        teams=[TeamState(i, str(i), "Owner", 1, 1, 100) for i in (1, 2)],
        week_lineups=[
            WeekLineups(
                2,
                1,
                2,
                0.0,
                10.0,
                home_lineup=[LineupEntry(31, "Kicker", "K", "K", 9.0, 8.0, "SF", False)],
            ),
            WeekLineups(
                week,
                1,
                2,
                None,
                None,
                home_lineup=[
                    LineupEntry(11, "Josh Allen", "QB", "QB", None, None, "BUF", False),
                    LineupEntry(31, "Kicker", "K", "K", None, None, "SF", False),
                ],
                away_lineup=[
                    LineupEntry(12, "Unknown Passer", "QB", "QB", None, None, "NYJ", False),
                    LineupEntry(13, "Bye Guy", "QB", "BE", None, None, "LA", True),
                ],
            ),
        ],
    )


def rankings(season=2026):
    return {
        "campaign_id": "fixture_v1",
        "campaign_name": "Fixture",
        "manifest_sha256": "a" * 64,
        "horizon": "remaining",
        "limitations": ["fixture"],
        "rankings": [
            {
                "player_id": "qb-allen",
                "player_display_name": "Josh Allen",
                "position": "QB",
                "team": "BUF",
                "season": season,
                "origin_week": 2,
                "horizon": "remaining",
                "points_per_game": 20.0,
            },
            {
                "player_id": "qb-bye",
                "player_display_name": "Bye Guy",
                "position": "QB",
                "team": "LA",
                "season": season,
                "origin_week": 2,
                "horizon": "remaining",
                "points_per_game": 15.0,
            },
        ],
    }


def schedule():
    return [
        {
            "season": 2026,
            "week": 3,
            "game_type": "REG",
            "gameday": "2026-09-27",
            "home_team": "BUF",
            "away_team": "SF",
        },
        {
            "season": 2026,
            "week": 3,
            "game_type": "REG",
            "gameday": "2026-09-27",
            "home_team": "NYJ",
            "away_team": "MIA",
        },
        {
            "season": 2026,
            "week": 4,
            "game_type": "REG",
            "gameday": "2026-10-04",
            "home_team": "LA",
            "away_team": "ARI",
        },
    ]


def test_per_game_points_identity_fallback_and_kicker_reference():
    result = weekly_forecast(snapshot(), rankings(), {11: "qb-allen"}, schedule())
    game = result["matchups"][0]
    home = {p["player_display_name"]: p for p in game["home"]["forecasts"]}
    away = {p["player_display_name"]: p for p in game["away"]["forecasts"]}
    assert home["Josh Allen"]["prediction"] == 20.0
    assert "Fixture remaining points per scheduled game" in home["Josh Allen"]["basis"]
    assert home["Kicker"]["prediction"] == 9.0
    assert away["Unknown Passer"]["prediction"] is None
    assert "No campaign forecast" in away["Unknown Passer"]["unavailable_reason"]
    assert away["Bye Guy"]["prediction"] == 0.0 and away["Bye Guy"]["basis"] == "Scheduled bye"
    assert game["home"]["model_projection"] == 29.0
    assert game["away"]["model_projection"] is None  # a missing starter withholds the total
    assert result["campaign_id"] == "fixture_v1"
    assert result["forecast_origins"] == [{"season": 2026, "origin_week": 2}]
    assert result["pregame_deadline"] == "2026-09-27T00:00:00+00:00"
    assert "fixture" in result["limitations"]


def test_a_forecast_for_another_season_is_not_used():
    result = weekly_forecast(snapshot(), rankings(season=2025), {11: "qb-allen"}, schedule())
    allen = result["matchups"][0]["home"]["forecasts"][0]
    assert allen["prediction"] is None
    assert "current season" in allen["unavailable_reason"]
