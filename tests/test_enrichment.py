"""Pure tests for nflverse projection enrichment transforms."""

from __future__ import annotations

import polars as pl
import pytest

from patron.data.nflverse import canonicalize_depth_charts
from patron.metrics.enrichment import (
    build_expected_opportunity,
    build_injury_history,
    build_market_rankings,
    build_nextgen_features,
    build_player_efficiency,
    build_player_usage,
    build_team_tendencies,
    build_team_volume,
    current_depth_chart,
    normalize_ppg_for_active_games,
    validate_participation_usage,
)


def test_player_efficiency_uses_attempt_weighted_rates() -> None:
    weeks = pl.DataFrame(
        {
            "player_id": ["qb", "qb"],
            "season": [2025, 2025],
            "attempts": [10, 30],
            "passing_epa": [2.0, 4.0],
            "passing_cpoe": [10.0, -2.0],
            "passing_first_downs": [4, 9],
            "carries": [2, 3],
            "rushing_epa": [1.0, -0.5],
            "rushing_first_downs": [1, 1],
            "targets": [0, 0],
            "receiving_epa": [0.0, 0.0],
            "receiving_first_downs": [0, 0],
        }
    )

    row = build_player_efficiency(weeks).to_dicts()[0]

    assert row["passing_epa_per_attempt"] == pytest.approx(0.15)
    assert row["passing_cpoe"] == pytest.approx(1.0)
    assert row["passing_first_down_rate"] == pytest.approx(13 / 40)
    assert row["rushing_first_down_rate"] == pytest.approx(0.4)


def test_team_tendencies_keep_only_neutral_scrimmage_plays() -> None:
    pbp = pl.DataFrame(
        {
            "game_id": ["g", "g", "g"],
            "season": [2025, 2025, 2025],
            "season_type": ["REG", "REG", "REG"],
            "posteam": ["A", "A", "A"],
            "qtr": [1, 2, 4],
            "down": [1.0, 3.0, 1.0],
            "score_differential": [0.0, 7.0, 0.0],
            "no_huddle": [1.0, 0.0, 0.0],
            "qb_dropback": [1.0, 0.0, 1.0],
            "rush_attempt": [0.0, 1.0, 0.0],
            "qb_kneel": [0.0, 0.0, 0.0],
            "epa": [0.2, -0.1, 4.0],
            "pass_oe": [5.0, -5.0, 20.0],
        }
    )

    row = build_team_tendencies(pbp).to_dicts()[0]

    assert row["neutral_plays"] == 2
    assert row["neutral_pass_rate"] == 0.5
    assert row["neutral_early_down_pass_rate"] == 1.0
    assert row["neutral_epa_per_play"] == pytest.approx(0.05)


def test_expected_opportunity_excludes_non_regular_games_and_null_players() -> None:
    opportunity = pl.DataFrame(
        {
            "season": ["2025", "2025", "2025"],
            "game_id": ["reg", "post", "reg"],
            "player_id": ["p", "p", None],
            "total_fantasy_points_exp": [12.0, 30.0, 99.0],
            "total_yards_gained_exp": [80.0, 200.0, 500.0],
            "total_touchdown_exp": [0.5, 2.0, 4.0],
            "total_first_down_exp": [4.0, 10.0, 20.0],
        }
    )
    weeks = pl.DataFrame({"season": [2025], "game_id": ["reg"]})

    row = build_expected_opportunity(opportunity, weeks).to_dicts()[0]

    assert row["player_id"] == "p"
    assert row["xfp_games"] == 1
    assert row["xfp_pg"] == 12.0


def test_nextgen_features_join_position_specific_season_summaries() -> None:
    common = {"season": [2025], "season_type": ["REG"], "week": [0]}
    passing = pl.DataFrame(
        {
            **common,
            "player_gsis_id": ["qb"],
            "completion_percentage_above_expectation": [3.0],
            "avg_time_to_throw": [2.7],
            "avg_intended_air_yards": [8.0],
            "aggressiveness": [12.0],
        }
    )
    receiving = pl.DataFrame(
        {
            **common,
            "player_gsis_id": ["wr"],
            "avg_separation": [3.2],
            "avg_yac_above_expectation": [0.8],
            "avg_intended_air_yards": [11.0],
        }
    )
    rushing = pl.DataFrame(
        {
            **common,
            "player_gsis_id": ["rb"],
            "rush_yards_over_expected_per_att": [0.4],
            "rush_pct_over_expected": [0.52],
            "percent_attempts_gte_eight_defenders": [0.2],
        }
    )

    rows = build_nextgen_features(passing, receiving, rushing)

    assert rows.height == 3
    assert rows.filter(pl.col("player_id") == "qb")["ngs_cpoe"][0] == 3.0
    assert rows.filter(pl.col("player_id") == "wr")["ngs_separation"][0] == 3.2
    assert rows.filter(pl.col("player_id") == "rb")["ngs_ryoe_per_att"][0] == 0.4


def test_market_rankings_select_latest_preseason_snapshot_and_crosswalk_id() -> None:
    rankings = pl.DataFrame(
        {
            "page_type": ["redraft-qb", "redraft-qb", "redraft-qb"],
            "id": ["10", "10", "10"],
            "pos": ["QB", "QB", "QB"],
            "ecr": [4.0, 2.0, 1.0],
            "sd": [1.0, 0.5, 0.2],
            "rank_delta": [None, 1.0, 2.0],
            "scrape_date": ["2025-08-01", "2025-08-29", "2025-09-05"],
        }
    )
    crosswalk = pl.DataFrame({"fantasypros_id": [10], "gsis_id": ["qb"]})

    row = build_market_rankings(rankings, crosswalk).to_dicts()[0]

    assert row["forecast_season"] == 2025
    assert row["market_ecr"] == 2.0
    assert row["market_ecr_score"] == -2.0
    assert row["market_snapshot"] == "2025-08-29"


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


def test_old_participation_uses_player_week_position_fallback() -> None:
    pbp = pl.DataFrame(
        {
            "game_id": ["2019_01_A_B"],
            "play_id": [1.0],
            "season": [2019],
            "season_type": ["REG"],
            "posteam": ["A"],
            "pass_attempt": [1.0],
            "rush_attempt": [0.0],
            "qb_dropback": [1.0],
            "qb_scramble": [0.0],
            "qb_kneel": [0.0],
            "yardline_100": [50.0],
            "air_yards": [8.0],
            "pass_touchdown": [0.0],
            "rush_touchdown": [0.0],
            "receiver_player_id": ["wr"],
            "rusher_player_id": [None],
            "two_point_attempt": [0.0],
        }
    )
    participation = pl.DataFrame(
        {
            "nflverse_game_id": ["2019_01_A_B"],
            "play_id": [1.0],
            "possession_team": ["A"],
            "offense_players": ["qb;wr"],
            "offense_positions": [None],
        }
    )
    weeks = pl.DataFrame(
        {
            "player_id": ["qb", "wr"],
            "season": [2019, 2019],
            "position": ["QB", "WR"],
            "targets": [0.0, 1.0],
        }
    )

    usage = build_player_usage(weeks, pbp, participation)
    wr = usage.filter(pl.col("player_id") == "wr").to_dicts()[0]

    assert wr["active_games"] == 1
    assert wr["route_opportunities"] == 1
    validate_participation_usage(usage, [2019])


def test_participation_validation_rejects_silent_zero_routes() -> None:
    usage = pl.DataFrame({"season": [2019], "route_opportunities": [0]})

    with pytest.raises(ValueError, match="2019"):
        validate_participation_usage(usage, [2019])


def test_v2_ppg_uses_active_games_without_overwriting_production_games() -> None:
    seasons = pl.DataFrame(
        {"games": [3], "active_games": [5], "season_pts": [20.0], "ppg": [20.0 / 3]}
    )

    row = normalize_ppg_for_active_games(seasons).to_dicts()[0]

    assert row["games"] == 3
    assert row["ppg_denominator_games"] == 5.0
    assert row["ppg"] == 4.0


def test_injury_history_excludes_rest_designations() -> None:
    injuries = pl.DataFrame(
        {
            # Historical nflverse injury partitions have used Float64 here.
            "season": [2025.0, 2025.0],
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
    assert row["season"] == 2025


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


def test_legacy_depth_chart_uses_week_one_as_preseason_proxy() -> None:
    legacy = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "club_code": ["BUF", "BUF"],
            "gsis_id": ["player", "player"],
            "position": ["WR", "WR"],
            "depth_position": ["WR", "WR"],
            "depth_team": ["2", "1"],
        }
    )

    rows = canonicalize_depth_charts(legacy).to_dicts()

    assert rows == [
        {
            "dt": "2024-08-31T00:00:00Z",
            "team": "BUF",
            "gsis_id": "player",
            "pos_abb": "WR",
            "pos_name": "WR",
            "pos_rank": 2,
        }
    ]
