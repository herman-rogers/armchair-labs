from __future__ import annotations

import polars as pl
import pytest

from patron.metrics.rich_weekly import (
    build_rich_weekly_features,
    build_weekly_route_panel,
)


def test_route_panel_uses_player_position_fallback_for_legacy_participation() -> None:
    weeks = pl.DataFrame(
        {
            "player_id": ["wr"],
            "season": [2020],
            "position": ["WR"],
        }
    )
    plays = pl.DataFrame(
        {
            "game_id": ["2020_01_A_B"],
            "play_id": [10.0],
            "season": [2020],
            "week": [1],
            "season_type": ["REG"],
            "posteam": ["A"],
            "qb_dropback": [1.0],
            "rush_attempt": [0.0],
            "qb_scramble": [0.0],
            "qb_kneel": [0.0],
            "pass_attempt": [1.0],
            "receiver_player_id": ["wr"],
            "rusher_player_id": [None],
            "yardline_100": [15.0],
            "two_point_attempt": [0.0],
        }
    )
    participation = pl.DataFrame(
        {
            "nflverse_game_id": ["2020_01_A_B"],
            "play_id": [10.0],
            "possession_team": ["A"],
            "offense_players": ["wr"],
            "offense_positions": [None],
        }
    )

    result = build_weekly_route_panel(weeks, plays, participation)

    assert result["route_opportunities"].item() == 1
    assert result["red_zone_opportunities"].item() == 1


def test_rich_features_keep_missing_feed_null_and_capture_role_momentum() -> None:
    rows = []
    for season in (2023, 2024, 2025):
        for week in range(1, 19):
            snap_available = season >= 2024
            rows.append(
                {
                    "player_id": "ascending",
                    "season": season,
                    "week": week,
                    "position": "WR",
                    "team": "A",
                    "points": float(week),
                    "opportunities": float(week),
                    "target_share": week / 100,
                    "carry_share": 0.0,
                    "air_yards_share": week / 100,
                    "snap_pct": week / 18 if snap_available else None,
                        "route_participation": week / 18,
                    "targets_per_route": 0.2,
                    "xfp": float(week),
                    "expected_td": week / 20,
                    "team_dropbacks": 35.0,
                    "team_carries": 25.0,
                    "team_pass_rate": 35 / 60,
                    "injury_severity": 0.0,
                    "practice_limit": 0.0,
                    "roster_score": 1.0,
                    "red_zone_opportunities": week / 5,
                    "passing_epa_rate": 0.0,
                    "rushing_epa_rate": 0.0,
                    "receiving_epa_rate": week / 50,
                    "snap_data_available": snap_available,
                        "route_data_available": snap_available,
                    "xfp_data_available": True,
                    "injury_data_available": True,
                    "roster_data_available": True,
                }
            )
    result = build_rich_weekly_features(
        pl.DataFrame(rows),
        pl.DataFrame({"forecast_season": [2026], "player_id": ["ascending"]}),
    )

    assert result["rich_s0_snap_pct_slope"].item() > 0
    assert result["rich_s2_snap_pct_mean"].item() is None
    assert result["rich_s2_route_participation_mean"].item() is None
    assert result["rich_s0_points_last4_delta"].item() > 0
    assert result["rich_points_three_year_trend"].item() == pytest.approx(0)
