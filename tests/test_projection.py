"""Forward projection tests: uncertainty, team moves, and teammate context."""

from __future__ import annotations

import polars as pl
import pytest

from patron.metrics.projection import (
    PlayerProjectionOverride,
    ProjectionAssumptions,
    TeamProjectionOverride,
    UnmatchedProjectionError,
    build_projection_board,
)


def season_row(
    player_id: str,
    name: str,
    position: str,
    team: str,
    ppg: float,
    games: int = 17,
    targets: float = 0.0,
    carries: float = 0.0,
    passing_yards: float = 0.0,
    passing_tds: float = 0.0,
    **stats: float,
) -> dict:
    row = {
        "player_id": player_id,
        "player_display_name": name,
        "position": position,
        "season": 2025,
        "team": team,
        "games": games,
        "ppg": ppg,
        "bonus_pts": 0.0,
        "passing_yards": passing_yards,
        "passing_tds": passing_tds,
        "passing_interceptions": 0.0,
        "rushing_tds": 0.0,
        "carries": carries,
        "targets": targets,
        "wtd_opp": carries + 2.2 * targets,
        "td_over_exp": 0.0,
    }
    row.update(stats)
    return row


def v1_board(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                **row,
                "rank": index,
                "vor": row["ppg"] - 10.0,
                "repl_ppg": 10.0,
                "adj_vor": row["ppg"] - 10.0,
                "override_delta": 0.0,
                "override_reason": None,
                "age_at_season": 26.0,
                "flags": "",
                "metric_version": "v1",
            }
            for index, row in enumerate(rows, 1)
        ]
    )


def wr_only_config(league_config):
    return league_config.model_copy(
        update={
            "vor_baseline_rank": {"WR": 2},
            "board_positions": ["WR"],
        }
    )


def test_small_samples_are_shrunk_more_than_full_samples(league_config) -> None:
    small = season_row("small", "Small Star", "WR", "LOW", 25.0, games=4, targets=40)
    full = season_row("full", "Full Star", "WR", "LOW", 25.0, games=17, targets=140)
    baseline = season_row("base", "Baseline", "WR", "LOW", 10.0, games=17, targets=90)
    qb = season_row("qb-low", "Low QB", "QB", "LOW", 15.0, passing_yards=3400, passing_tds=20)
    seasons = pl.DataFrame([small, full, baseline, qb])

    board = build_projection_board(
        seasons,
        v1_board(small, full, baseline),
        wr_only_config(league_config),
        assumptions=ProjectionAssumptions(),
    )
    rows = {row["player_id"]: row for row in board.iter_rows(named=True)}

    assert rows["small"]["projection_confidence"] < rows["full"]["projection_confidence"]
    assert rows["small"]["individual_prior_ppg"] < rows["full"]["individual_prior_ppg"]


def test_a_receiver_move_uses_the_destination_quarterback_context(league_config) -> None:
    receiver = season_row("wr", "Moving Receiver", "WR", "LOW", 14.0, targets=120)
    low_mate = season_row("low-mate", "Low Mate", "WR", "LOW", 9.0, targets=90)
    high_mate = season_row("high-mate", "High Mate", "WR", "HIGH", 12.0, targets=90)
    low_qb = season_row("qb-low", "Low QB", "QB", "LOW", 14.0, passing_yards=3000, passing_tds=15)
    high_qb = season_row(
        "qb-high", "High QB", "QB", "HIGH", 24.0, passing_yards=5000, passing_tds=42
    )
    seasons = pl.DataFrame([receiver, low_mate, high_mate, low_qb, high_qb])
    base = v1_board(receiver, low_mate, high_mate)
    config = wr_only_config(league_config)

    stay = (
        build_projection_board(seasons, base, config, assumptions=ProjectionAssumptions())
        .filter(pl.col("player_id") == "wr")
        .to_dicts()[0]
    )
    move_assumptions = ProjectionAssumptions(
        players={
            ("moving receiver", "WR"): PlayerProjectionOverride(
                player="Moving Receiver", position="WR", projected_team="HIGH"
            )
        }
    )
    move = (
        build_projection_board(seasons, base, config, assumptions=move_assumptions)
        .filter(pl.col("player_id") == "wr")
        .to_dicts()[0]
    )

    assert move["projected_team"] == "HIGH"
    assert move["qb_context"] > stay["qb_context"]
    assert move["team_context_factor"] > stay["team_context_factor"]
    assert move["proj_ppg"] > stay["proj_ppg"]


def test_more_projected_teammate_competition_reduces_receiver_projection(
    league_config,
) -> None:
    receiver = season_row("wr", "Context Receiver", "WR", "SEA", 14.0, targets=120)
    mate = season_row("mate", "Target Hog", "WR", "SEA", 13.0, targets=130)
    qb = season_row("qb", "Quarterback", "QB", "SEA", 20.0, passing_yards=4200, passing_tds=30)
    seasons = pl.DataFrame([receiver, mate, qb])
    base = v1_board(receiver, mate)
    config = wr_only_config(league_config)

    neutral = (
        build_projection_board(seasons, base, config, assumptions=ProjectionAssumptions())
        .filter(pl.col("player_id") == "wr")
        .to_dicts()[0]
    )
    crowded = (
        build_projection_board(
            seasons,
            base,
            config,
            assumptions=ProjectionAssumptions(
                teams={
                    "SEA": TeamProjectionOverride(team="SEA", target_availability_multiplier=0.80)
                }
            ),
        )
        .filter(pl.col("player_id") == "wr")
        .to_dicts()[0]
    )

    assert crowded["teammate_competition"] > neutral["teammate_competition"]
    assert crowded["team_context_factor"] < neutral["team_context_factor"]
    assert crowded["proj_ppg"] < neutral["proj_ppg"]


def test_v2_reports_true_season_team_shares(league_config) -> None:
    receiver = season_row("wr", "Share Receiver", "WR", "SEA", 14.0, targets=100)
    mate = season_row("mate", "Share Mate", "WR", "SEA", 10.0, targets=300)
    qb = season_row("qb", "Quarterback", "QB", "SEA", 20.0, passing_yards=4000, passing_tds=28)
    board = build_projection_board(
        pl.DataFrame([receiver, mate, qb]),
        v1_board(receiver, mate),
        wr_only_config(league_config),
        assumptions=ProjectionAssumptions(),
    )
    row = board.filter(pl.col("player_id") == "wr").to_dicts()[0]

    assert row["season_target_share"] == 0.25


def test_an_unmatched_future_player_assumption_fails_loudly(league_config) -> None:
    receiver = season_row("wr", "Real Receiver", "WR", "SEA", 14.0, targets=100)
    assumptions = ProjectionAssumptions(
        players={
            ("typo receiver", "WR"): PlayerProjectionOverride(
                player="Typo Receiver", position="WR", projected_team="BUF"
            )
        }
    )

    with pytest.raises(UnmatchedProjectionError, match="Typo Receiver"):
        build_projection_board(
            pl.DataFrame([receiver]),
            v1_board(receiver),
            wr_only_config(league_config),
            assumptions=assumptions,
        )


def test_v2_blends_historical_and_component_branches(league_config) -> None:
    receiver = season_row(
        "wr",
        "Blend Receiver",
        "WR",
        "SEA",
        16.0,
        targets=140,
        receptions=95,
        receiving_yards=1250,
        receiving_tds=8,
        wopr=0.65,
        air_yards_share=0.32,
        floor=9.0,
        volatility=7.0,
        season_pts=272.0,
    )
    mate = season_row("mate", "Mate", "WR", "SEA", 10.0, targets=100)
    qb = season_row("qb", "Quarterback", "QB", "SEA", 20.0, passing_yards=4200, passing_tds=30)
    row = (
        build_projection_board(
            pl.DataFrame([receiver, mate, qb]),
            v1_board(receiver, mate),
            wr_only_config(league_config),
            assumptions=ProjectionAssumptions(),
        )
        .filter(pl.col("player_id") == "wr")
        .to_dicts()[0]
    )

    component_weight = league_config.metrics.projection_component_weight
    expected = (1.0 - component_weight) * row["prior_branch_ppg"] + component_weight * row[
        "component_proj_ppg"
    ]
    assert row["proj_ppg"] == pytest.approx(expected)
    assert row["projected_targets_pg"] > 0
    assert row["projected_receptions_pg"] > 0
    assert row["projected_receiving_yards_pg"] > 0


def test_receiving_role_metrics_drive_projected_opportunity(league_config) -> None:
    alpha = season_row(
        "alpha",
        "Alpha",
        "WR",
        "SEA",
        12.0,
        targets=150,
        receptions=90,
        receiving_yards=1100,
        wopr=0.75,
        air_yards_share=0.38,
    )
    secondary = season_row(
        "secondary",
        "Secondary",
        "WR",
        "SEA",
        12.0,
        targets=75,
        receptions=50,
        receiving_yards=700,
        wopr=0.35,
        air_yards_share=0.16,
    )
    qb = season_row("qb", "Quarterback", "QB", "SEA", 20.0, passing_yards=4200, passing_tds=30)
    rows = build_projection_board(
        pl.DataFrame([alpha, secondary, qb]),
        v1_board(alpha, secondary),
        wr_only_config(league_config),
        assumptions=ProjectionAssumptions(),
    ).to_dicts()
    by_id = {row["player_id"]: row for row in rows}

    assert by_id["alpha"]["projected_target_share"] > by_id["secondary"]["projected_target_share"]
    assert by_id["alpha"]["projected_targets_pg"] > by_id["secondary"]["projected_targets_pg"]
    assert (
        by_id["alpha"]["projected_air_yards_share"]
        > by_id["secondary"]["projected_air_yards_share"]
    )


def test_balanced_score_uses_mean_floor_and_ceiling_vor(league_config) -> None:
    receiver = season_row(
        "wr",
        "Distribution Receiver",
        "WR",
        "SEA",
        15.0,
        targets=130,
        receptions=85,
        receiving_yards=1100,
        floor=10.0,
        volatility=6.0,
    )
    mate = season_row(
        "mate", "Distribution Mate", "WR", "SEA", 10.0, targets=90, floor=5.0, volatility=7.0
    )
    qb = season_row("qb", "Quarterback", "QB", "SEA", 20.0, passing_yards=4000, passing_tds=28)
    row = (
        build_projection_board(
            pl.DataFrame([receiver, mate, qb]),
            v1_board(receiver, mate),
            wr_only_config(league_config),
            assumptions=ProjectionAssumptions(),
        )
        .filter(pl.col("player_id") == "wr")
        .to_dicts()[0]
    )
    metrics = league_config.metrics
    expected = (
        (
            metrics.projection_mean_weight * row["proj_vor"]
            + metrics.projection_floor_weight * row["floor_vor"]
            + metrics.projection_ceiling_weight * row["ceiling_vor"]
        )
        / (
            metrics.projection_mean_weight
            + metrics.projection_floor_weight
            + metrics.projection_ceiling_weight
        )
        * row["availability_factor"]
    )

    assert row["projected_floor"] < row["proj_ppg"] < row["projected_ceiling"]
    assert row["v2_score"] == pytest.approx(expected)


def test_route_participation_and_tprr_drive_targets(league_config) -> None:
    full_route = season_row(
        "full",
        "Full Route",
        "WR",
        "SEA",
        12.0,
        targets=100,
        receptions=65,
        receiving_yards=850,
        route_opportunities=400,
        route_participation=0.90,
        targets_per_route_opportunity=0.25,
    )
    part_route = season_row(
        "part",
        "Part Route",
        "WR",
        "SEA",
        12.0,
        targets=100,
        receptions=65,
        receiving_yards=850,
        route_opportunities=400,
        route_participation=0.50,
        targets_per_route_opportunity=0.25,
    )
    qb = season_row(
        "qb",
        "Quarterback",
        "QB",
        "SEA",
        20.0,
        passing_yards=4200,
        passing_tds=30,
        attempts=600,
        team_pass_attempts=600,
        team_dropbacks=640,
        team_games=17,
    )
    rows = build_projection_board(
        pl.DataFrame([full_route, part_route, qb]),
        v1_board(full_route, part_route),
        wr_only_config(league_config),
        assumptions=ProjectionAssumptions(),
    ).to_dicts()
    by_id = {row["player_id"]: row for row in rows}

    assert (
        by_id["full"]["projected_route_opportunities_pg"]
        > by_id["part"]["projected_route_opportunities_pg"]
    )
    assert by_id["full"]["projected_targets_pg"] > by_id["part"]["projected_targets_pg"]


def test_current_depth_chart_changes_team_and_role_automatically(league_config) -> None:
    receiver = season_row("wr", "Moved Receiver", "WR", "OLD", 14.0, targets=120)
    old_qb = season_row("old-qb", "Old QB", "QB", "OLD", 16.0, passing_yards=3200)
    new_qb = season_row("new-qb", "New QB", "QB", "NEW", 22.0, passing_yards=4600)
    new_mate = season_row("new-mate", "New Mate", "WR", "NEW", 10.0, targets=90)
    current = pl.DataFrame(
        {
            "player_id": ["wr"],
            "current_team": ["NEW"],
            "depth_chart_rank": [2],
            "depth_chart_position": ["Wide Receiver"],
            "depth_chart_date": ["2026-08-01"],
        }
    )

    row = (
        build_projection_board(
            pl.DataFrame([receiver, old_qb, new_qb, new_mate]),
            v1_board(receiver, new_mate),
            wr_only_config(league_config),
            assumptions=ProjectionAssumptions(),
            current_players=current,
        )
        .filter(pl.col("player_id") == "wr")
        .to_dicts()[0]
    )

    assert row["projected_team"] == "NEW"
    assert row["depth_chart_rank"] == 2
    assert row["depth_role_factor"] < 1.0
    assert "nflverse depth chart" in row["projection_reason"]


def test_injury_and_participation_history_reduce_expected_games_and_v2_score(
    league_config,
) -> None:
    healthy = season_row(
        "healthy",
        "Healthy Receiver",
        "WR",
        "SEA",
        14.0,
        targets=120,
        active_games=17,
        team_games=17,
    )
    injured = season_row(
        "injured",
        "Injured Receiver",
        "WR",
        "SEA",
        14.0,
        targets=120,
        active_games=8,
        team_games=17,
        injury_report_weeks=9,
        out_report_weeks=5,
    )
    baseline = season_row(
        "baseline",
        "Baseline Receiver",
        "WR",
        "SEA",
        8.0,
        targets=70,
        active_games=17,
        team_games=17,
    )
    qb = season_row("qb", "Quarterback", "QB", "SEA", 20.0, passing_yards=4200)
    config = wr_only_config(league_config).model_copy(update={"vor_baseline_rank": {"WR": 3}})
    rows = build_projection_board(
        pl.DataFrame([healthy, injured, baseline, qb]),
        v1_board(healthy, injured, baseline),
        config,
        assumptions=ProjectionAssumptions(),
    ).to_dicts()
    by_id = {row["player_id"]: row for row in rows}

    assert by_id["injured"]["proj_ppg"] == pytest.approx(by_id["healthy"]["proj_ppg"])
    assert by_id["injured"]["expected_games"] < by_id["healthy"]["expected_games"]
    assert (
        by_id["injured"]["availability_adjusted_vor"]
        < by_id["healthy"]["availability_adjusted_vor"]
    )
    assert by_id["injured"]["v2_score"] < by_id["healthy"]["v2_score"]
