from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from engine.metrics.rich_weekly import (
    _descriptor,
    build_rich_weekly_features,
    build_weekly_route_panel,
)


def test_missing_route_observations_do_not_become_zero_in_recent_descriptors():
    values = np.full(18, np.nan)
    values[-2:] = [0.8, 0.8]
    summary = _descriptor(values)
    assert summary["mean"] == pytest.approx(0.8)
    assert summary["last4"] == pytest.approx(0.8)
    assert np.isnan(summary["last4_delta"])
    assert summary["coverage"] == pytest.approx(2 / 18)


@pytest.mark.parametrize("available", [False, True])
def test_route_panel_uses_player_position_fallback_for_legacy_participation(
    available: bool,
) -> None:
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

    result = build_weekly_route_panel(
        weeks, plays, participation if available else participation.head(0)
    )

    assert result["route_opportunities"].item() == (1 if available else None)
    assert result["red_zone_opportunities"].item() == 1
    assert result["route_team_dropbacks"].item() == (1 if available else None)
    duplicate = build_weekly_route_panel(
        weeks,
        pl.concat([plays, plays]),
        pl.concat([participation, participation]) if available else participation.head(0),
    )
    assert duplicate.equals(result)
    # A scramble is a dropback for BOTH numerator and denominator, even though it
    # is neither an official passing attempt nor a sack.
    scramble = plays.with_columns(
        pl.lit(11.0).alias("play_id"),
        pl.lit(1.0).alias("qb_scramble"),
        pl.lit(0.0).alias("pass_attempt"),
    )
    scramble_participation = participation.with_columns(pl.lit(11.0).alias("play_id"))
    combined = build_weekly_route_panel(
        weeks,
        pl.concat([plays, scramble]),
        pl.concat([participation, scramble_participation]),
    )
    assert combined["route_opportunities"].item() == 2
    assert combined["route_team_dropbacks"].item() == 2
    assert combined["route_expected_dropbacks"].item() == 2


@pytest.mark.parametrize("history", [1, 2, 3, 4])
def test_rich_features_keep_missing_feed_null_and_capture_role_momentum(history: int) -> None:
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
                    "play_data_available": True,
                    "xfp_data_available": True,
                    "injury_data_available": True,
                    "roster_data_available": True,
                }
            )
    result = build_rich_weekly_features(
        pl.DataFrame(rows),
        pl.DataFrame({"forecast_season": [2026], "player_id": ["ascending"]}),
        history_seasons=history,
    )

    assert result["rich_s0_snap_pct_slope"].item() > 0
    if history >= 3:
        assert result["rich_s2_snap_pct_mean"].item() is None
        assert result["rich_s2_route_participation_mean"].item() is None
        assert result["rich_s2_red_zone_opportunities_mean"].item() > 0
    assert result["rich_s0_points_last4_delta"].item() > 0
    if history > 1:
        assert result["rich_points_three_year_trend"].item() == pytest.approx(0)
    else:
        assert result["rich_points_three_year_trend"].item() is None
        assert result["rich_points_mean_yoy"].item() is None
    assert f"rich_s{history - 1}_points_mean" in result.columns


def test_weekly_panel_separates_play_coverage_from_participation() -> None:
    from engine.metrics.rich_weekly import build_rich_weekly_panel

    keys = {"player_id": ["wr", "wr"], "season": [2015, 2016], "week": [1, 1]}
    stats = pl.DataFrame(
        {
            **keys,
            "position": ["WR"] * 2,
            "team": ["A"] * 2,
            **{
                column: [1.0, 1.0]
                for column in (
                    "fantasy_points_ppr",
                    "attempts",
                    "passing_epa",
                    "carries",
                    "rushing_epa",
                    "targets",
                    "receiving_epa",
                    "target_share",
                    "air_yards_share",
                )
            },
        }
    )
    team = pl.DataFrame(
        {
            "season": [2015, 2016],
            "week": [1, 1],
            "team": ["A"] * 2,
            "attempts": [10.0] * 2,
            "sacks_suffered": [0.0] * 2,
            "carries": [10.0] * 2,
        }
    )
    snaps = pl.DataFrame(
        {
            "season": [2016],
            "week": [1],
            "game_type": ["REG"],
            "pfr_player_id": ["WR"],
            "position": ["WR"],
            "team": ["A"],
            "offense_snaps": [1.0],
            "offense_pct": [0.5],
        }
    )
    players = pl.DataFrame({"pfr_id": ["WR"], "gsis_id": ["wr"]})
    rosters = pl.DataFrame(
        {
            "season": [2016],
            "week": [1],
            "game_type": ["REG"],
            "gsis_id": ["wr"],
            "position": ["WR"],
            "team": ["A"],
            "status": ["ACT"],
        }
    )
    injuries = pl.DataFrame(
        {
            "season": [2016],
            "week": [1],
            "gsis_id": ["wr"],
            "team": ["A"],
            "report_status": [""],
            "practice_status": [""],
        }
    )
    expected = pl.DataFrame(
        {
            **keys,
            "posteam": ["A"] * 2,
            "total_fantasy_points_exp": [1.0] * 2,
            "total_touchdown_exp": [0.0] * 2,
        }
    )
    route = pl.DataFrame(
        {
            **keys,
            "route_team": [None, "A"],
            "route_opportunities": [None, 1.0],
            "offensive_plays": [None, 1.0],
            "route_team_dropbacks": [None, 10.0],
            "route_expected_dropbacks": [None, 10.0],
            "red_zone_opportunities": [1.0, 1.0],
        }
    ).with_columns(pl.col("season", "week").cast(pl.Int32))
    panel = build_rich_weekly_panel(stats, team, snaps, players, injuries, rosters, expected, route)
    assert panel["route_data_available"].to_list() == [False, True]
    assert panel["play_data_available"].to_list() == [True, True]
    assert panel["route_participation"].to_list() == [None, 0.1]
    partial = build_rich_weekly_panel(
        stats,
        team,
        snaps,
        players,
        injuries,
        rosters,
        expected,
        route.with_columns(pl.lit(11.0).alias("route_expected_dropbacks")),
    )
    assert partial["route_data_available"].to_list() == [False, False]
    assert partial["route_participation"].null_count() == 2
    nonfinite = build_rich_weekly_panel(
        stats.with_columns(pl.lit(float("inf")).alias("air_yards_share")),
        team,
        snaps,
        players,
        injuries,
        rosters,
        expected,
        route,
    )
    assert nonfinite["air_yards_share"].null_count() == 2
    features = build_rich_weekly_features(
        panel, pl.DataFrame({"forecast_season": [2017], "player_id": ["wr"]})
    )
    assert features["rich_s1_route_participation_mean"].item() is None
    assert features["rich_s1_red_zone_opportunities_mean"].item() == pytest.approx(1 / 18)
