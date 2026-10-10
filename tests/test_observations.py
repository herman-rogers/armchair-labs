"""Observation SQL over the analytics schema, independent of any model output."""

import duckdb
import polars as pl
import pytest

from engine.tables import observations


@pytest.fixture
def db():
    connection = duckdb.connect()
    connection.execute("CREATE SCHEMA analytics")
    frames = {
        "players": pl.DataFrame(
            {
                "gsis_id": ["a", "b", "r"],
                "display_name": ["Alpha", "Bravo", "Rookie"],
                "birth_date": ["1999-01-01", None, "2004-05-05"],
                "espn_id": ["1001", "1002", None],
                "pfr_id": [None, None, None],
                "rookie_season": [2021, 2020, 2026],
                "draft_year": [2021, 2020, 2026],
                "draft_round": [1, None, 2],
                "draft_pick": [5, None, 40],
                "draft_team": ["SF", None, "LA"],
            }
        ),
        "nfl_player_weeks": pl.DataFrame(
            {
                "player_id": ["a", "a", "b"],
                "player_display_name": ["Alpha", "Alpha", "Bravo"],
                "position": ["RB", "RB", "WR"],
                "season": [2025, 2025, 2024],
                "week": [1, 2, 17],
                "team": ["SF", "SF", "SEA"],
                "opponent_team": ["LA", "ARI", "SF"],
                "completions": [0, 0, 0],
                "attempts": [0, 0, 0],
                "passing_yards": [0, 0, 0],
                "passing_tds": [0, 0, 0],
                "passing_interceptions": [0, 0, 0],
                "carries": [10, 12, 0],
                "rushing_yards": [50, 60, 0],
                "rushing_tds": [1, 0, 0],
                "targets": [2, 3, 8],
                "receptions": [2, 2, 6],
                "receiving_yards": [15, 20, 80],
                "receiving_tds": [0, 0, 1],
                "league_points": [14.5, 10.0, 20.0],
            }
        ),
        "nfl_player_seasons": pl.DataFrame(
            {"player_id": ["a"], "season": [2025], "position": ["RB"], "ppg": [12.25]}
        ),
        "current_players": pl.DataFrame(
            {
                "player_id": ["r"],
                "player_display_name": ["Rookie"],
                "position": ["WR"],
                "season": [2026],
                "draft_pick": [40],
                "espn_id": [1003],
                "team": ["LA"],
            }
        ),
        "current_schedule": pl.DataFrame(
            {
                "game_id": ["g1"],
                "season": [2026],
                "week": [1],
                "game_type": ["REG"],
                "gameday": ["2026-09-10"],
                "home_team": ["SF"],
                "away_team": ["LA"],
            }
        ),
        "nfl_schedule": pl.DataFrame(
            {
                "game_id": ["g0"],
                "season": [2025],
                "week": [1],
                "game_type": ["REG"],
                "gameday": ["2025-09-10"],
                "home_team": ["SF"],
                "away_team": ["LA"],
            }
        ),
        "college_identity_links": pl.DataFrame(
            {
                "college_id": ["c-rookie"],
                "college_name": ["Rookie"],
                "first_college_season": [2022],
                "last_college_season": [2025],
                "player_id": ["r"],
                "status": ["linked"],
            }
        ),
        "college_annual": pl.DataFrame(
            {"college_id": ["c-rookie"], "season": [2025], "receiving_yards": [1200.0]}
        ),
    }
    weeks = frames["nfl_player_weeks"]
    frames["current_weeks"] = weeks.head(1).with_columns(
        pl.lit("a").alias("player_id"),
        pl.lit(2026).alias("season"),
        pl.lit(1).alias("week"),
        pl.lit("LA").alias("team"),
    )
    for name, frame in frames.items():
        connection.register(f"src_{name}", frame.to_arrow())
        connection.execute(f"CREATE TABLE analytics.{name} AS SELECT * FROM src_{name}")
    return connection


def test_cutoff_candidates_and_identities_follow_latest_observation(db):
    assert observations.observation_cutoff(db) == {"season": 2026, "through_week": 1}
    latest = {r["player_id"]: r for r in observations.candidates(db, 2026).to_dicts()}
    assert latest["a"]["team"] == "LA"  # the current-season week wins over 2025
    assert latest["r"]["position"] == "WR"  # rookies without games are still candidates
    assert "b" not in latest  # last seen before the season window
    assert observations.espn_ids(db) == {1001: "a", 1002: "b", 1003: "r"}
    assert observations.identity(db, "a")["draft_team"] == "SF"
    assert observations.identity(db, "zzz") is None


def test_directory_player_stats_and_rookies(db):
    listing = observations.directory(db, search="alp")
    assert listing["player_id"].to_list() == ["a"]
    weeks = observations.weeks(db, "a")
    assert [(w["season"], w["week"]) for w in weeks] == [(2025, 1), (2025, 2), (2026, 1)]
    assert observations.weeks(db, "a", season=2026)[0]["team"] == "LA"
    assert observations.seasons(db, "a")[0]["ppg"] == 12.25
    rookies = observations.rookies(db, 2026).to_dicts()
    assert rookies[0]["college_id"] == "c-rookie" and rookies[0]["draft_round"] == 2
    assert observations.college_seasons(db, "c-rookie")[0]["receiving_yards"] == 1200.0
    assert [g["game_id"] for g in observations.schedule(db, 2026)] == ["g1"]
