"""Pure tests for nflverse projection enrichment transforms."""

from __future__ import annotations

import polars as pl

from patron.metrics.enrichment import (
    build_injury_history,
    build_player_usage,
    build_team_volume,
    current_depth_chart,
)


def test_participation_and_pbp_build_route_and_high_value_usage() -> None:
    pbp = pl.DataFrame(
        [
            {
                "game_id": "2025_01_A_B",
                "play_id": 1.0,
                "season": 2025,
                "season_type": "REG",
                "posteam": "A",
                "pass_attempt": 1.0,
                "rush_attempt": 0.0,
                "qb_dropback": 1.0,
                "qb_scramble": 0.0,
                "qb_kneel": 0.0,
                "yardline_100": 10.0,
                "air_yards": 10.0,
                "pass_touchdown": 1.0,
                "rush_touchdown": 0.0,
                "receiver_player_id": "wr",
                "rusher_player_id": None,
                "two_point_attempt": 0.0,
            },
            {
                "game_id": "2025_01_A_B",
                "play_id": 2.0,
                "season": 2025,
                "season_type": "REG",
                "posteam": "A",
                "pass_attempt": 0.0,
                "rush_attempt": 0.0,
                "qb_dropback": 1.0,
                "qb_scramble": 0.0,
                "qb_kneel": 0.0,
                "yardline_100": 50.0,
                "air_yards": None,
                "pass_touchdown": 0.0,
                "rush_touchdown": 0.0,
                "receiver_player_id": None,
                "rusher_player_id": None,
                "two_point_attempt": 0.0,
            },
            {
                "game_id": "2025_01_A_B",
                "play_id": 3.0,
                "season": 2025,
                "season_type": "REG",
                "posteam": "A",
                "pass_attempt": 0.0,
                "rush_attempt": 1.0,
                "qb_dropback": 0.0,
                "qb_scramble": 0.0,
                "qb_kneel": 0.0,
                "yardline_100": 3.0,
                "air_yards": None,
                "pass_touchdown": 0.0,
                "rush_touchdown": 1.0,
                "receiver_player_id": None,
                "rusher_player_id": "rb",
                "two_point_attempt": 0.0,
            },
        ]
    )
    participation = pl.DataFrame(
        [
            {
                "nflverse_game_id": "2025_01_A_B",
                "play_id": float(play_id),
                "possession_team": "A",
                "offense_players": "qb;wr;te;rb",
                "offense_positions": "QB;WR;TE;RB",
            }
            for play_id in (1, 2, 3)
        ]
    )
    weeks = pl.DataFrame(
        {
            "player_id": ["wr", "rb"],
            "season": [2025, 2025],
            "targets": [1.0, 0.0],
        }
    )

    usage = build_player_usage(weeks, pbp, participation)
    wr = usage.filter(pl.col("player_id") == "wr").to_dicts()[0]
    rb = usage.filter(pl.col("player_id") == "rb").to_dicts()[0]

    assert wr["route_opportunities"] == 2
    assert wr["route_participation"] == 1.0
    assert wr["targets_per_route_opportunity"] == 0.5
    assert wr["red_zone_targets"] == 1
    assert wr["end_zone_targets"] == 1
    assert wr["end_zone_receiving_tds"] == 1
    assert rb["goal_line_carries"] == 1
    assert rb["goal_line_rushing_tds"] == 1


def test_team_volume_uses_official_attempts_and_sacks() -> None:
    weeks = pl.DataFrame(
        {
            "season": [2025, 2025],
            "team": ["A", "A"],
            "week": [1, 2],
            "attempts": [30, 35],
            "sacks_suffered": [2, 3],
            "carries": [25, 20],
        }
    )
    row = build_team_volume(weeks).to_dicts()[0]

    assert row["team_games"] == 2
    assert row["team_pass_attempts"] == 65
    assert row["team_dropbacks"] == 70
    assert row["official_team_carries"] == 45


def test_injury_history_excludes_rest_designations() -> None:
    injuries = pl.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 2],
            "gsis_id": ["player", "player"],
            "report_primary_injury": [None, "Hamstring"],
            "report_status": [None, "Out"],
            "practice_primary_injury": ["Not injury related - resting player", "Hamstring"],
            "practice_status": ["DNP", "DNP"],
        }
    )
    row = build_injury_history(injuries).to_dicts()[0]

    assert row["injury_report_weeks"] == 1
    assert row["out_report_weeks"] == 1


def test_current_depth_chart_keeps_latest_player_snapshot() -> None:
    depth = pl.DataFrame(
        {
            "dt": ["2025-09-01", "2026-03-01"],
            "team": ["OLD", "NEW"],
            "gsis_id": ["player", "player"],
            "pos_abb": ["WR", "WR"],
            "pos_name": ["Wide Receiver", "Wide Receiver"],
            "pos_rank": [1, 2],
        }
    )
    row = current_depth_chart(depth).to_dicts()[0]

    assert row["current_team"] == "NEW"
    assert row["depth_chart_rank"] == 2
    assert row["depth_chart_date"] == "2026-03-01"
