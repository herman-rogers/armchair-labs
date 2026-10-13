"""Campaign offensive points plus weekly ESPN kicker/defense estimates."""

import pytest

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
        "models": [{"model_id": "qb", "generated_at": "2026-09-20T12:00:00+00:00"}],
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
    assert result["through_week"] == 2
    assert result["published_at"] == "2026-09-20T12:00:00+00:00"
    assert result["pregame_deadline"] == "2026-09-27T00:00:00+00:00"
    assert "fixture" in result["limitations"]


def test_a_forecast_for_another_season_is_not_used():
    result = weekly_forecast(snapshot(), rankings(season=2025), {11: "qb-allen"}, schedule())
    allen = result["matchups"][0]["home"]["forecasts"][0]
    assert allen["prediction"] is None
    assert "current season" in allen["unavailable_reason"]


def test_weekly_espn_special_teams_override_history_but_not_offensive_forecasts():
    data = snapshot()
    current = data.week_lineups[-1]
    current.home_lineup[0].projected_points = 99.0
    current.home_lineup[1].projected_points = 7.5
    # No historical Bengals entry: its weekly estimate must be enough.
    current.home_lineup.append(
        LineupEntry(41, "Bengals D/ST", "D/ST", "D/ST", 100.0, 6.89, "BUF", False)
    )
    result = weekly_forecast(data, rankings(), {11: "qb-allen"}, schedule())
    side = result["matchups"][0]["home"]
    assert [p["prediction"] for p in side["forecasts"]] == [20.0, 7.5, 6.89]
    assert side["model_projection"] == pytest.approx(34.39)
    assert side["covered_starters"] == side["starter_count"] == 3
    assert side["forecasts"][-1]["basis"] == "ESPN Week 3 projection · K/DST"


@pytest.mark.parametrize("projected", [0.0, -2.5])
def test_zero_and_negative_weekly_estimates_are_not_missing(projected):
    data = snapshot()
    data.week_lineups[-1].home_lineup[1].projected_points = projected
    result = weekly_forecast(data, rankings(), {11: "qb-allen"}, schedule())
    assert result["matchups"][0]["home"]["model_projection"] == 20 + projected


@pytest.mark.parametrize("projected", [None, float("nan"), float("inf")])
def test_missing_weekly_estimates_fall_back_only_to_previous_scores(projected):
    data = snapshot()
    data.week_lineups[-1].home_lineup[1].projected_points = projected
    data.week_lineups[-1].home_lineup[1].points = 100.0
    result = weekly_forecast(data, rankings(), {11: "qb-allen"}, schedule())
    kicker = result["matchups"][0]["home"]["forecasts"][1]
    assert kicker["prediction"] == 9.0
    assert "fallback" in kicker["basis"]


def test_special_teams_bye_and_missing_data_remain_explicit():
    data = snapshot()
    data.week_lineups[-1].home_lineup[1].on_bye = True
    data.week_lineups[-1].home_lineup[1].projected_points = 7.5
    data.week_lineups[-1].home_lineup.append(
        LineupEntry(41, "Unknown D/ST", "D/ST", "D/ST", 100.0, None, "BUF", False)
    )
    result = weekly_forecast(data, rankings(), {11: "qb-allen"}, schedule())
    side = result["matchups"][0]["home"]
    assert side["forecasts"][1]["prediction"] == 0.0
    assert side["forecasts"][-1]["prediction"] is None
    assert "No ESPN weekly projection" in side["forecasts"][-1]["unavailable_reason"]
    assert side["model_projection"] is None
