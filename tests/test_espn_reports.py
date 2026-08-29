"""Unit tests for the wire and roster reports."""

from __future__ import annotations

import polars as pl
import pytest

from patron.espn.crosswalk import (
    CHANGED_TEAM,
    ESPN_ID,
    INJURY_STATUS,
    IS_FREE_AGENT,
    IS_MINE,
    OWNER_TEAM_ID,
    OWNER_TEAM_NAME,
    PERCENT_OWNED,
)
from patron.espn.reports import (
    WIRE_VOR,
    opponent_weaknesses,
    ranked_wire,
    roster_health,
    team_strengths,
    unrankable_players,
    wire_replacement_levels,
)

BASELINES = {"QB": 2, "RB": 2, "WR": 2, "TE": 1}


def tagged(*rows: dict) -> pl.DataFrame:
    base = {
        "player_display_name": "Player",
        "position": "RB",
        "team": "SF",
        "espn_team": "SF",
        "ppg": 10.0,
        "adj_vor": 1.0,
        "flags": "",
        IS_FREE_AGENT: True,
        IS_MINE: False,
        CHANGED_TEAM: False,
        INJURY_STATUS: "ACTIVE",
        PERCENT_OWNED: 10.0,
        OWNER_TEAM_NAME: None,
        "player_id": None,
    }
    return pl.DataFrame(
        [{**base, **row, "player_id": row.get("player_id", f"g{i}")} for i, row in enumerate(rows)]
    )


class TestWireReplacement:
    def test_replacement_comes_from_the_free_pool_not_the_board(self) -> None:
        """ "Value over replacement" only means something if replacement means "what I
        can have for free right now". Once a league has drafted, the real bar is far
        below the preseason one, and reusing it understates every pickup."""
        frame = tagged(
            {"position": "RB", "ppg": 20.0, IS_FREE_AGENT: False},  # rostered, ignored
            {"position": "RB", "ppg": 9.0},
            {"position": "RB", "ppg": 5.0},
        )
        levels = wire_replacement_levels(frame, {"RB": 2})

        assert levels["RB"] == pytest.approx(5.0), "second-best FREE back, not the rostered one"

    def test_a_thin_pool_falls_back_to_the_worst_available(self) -> None:
        levels = wire_replacement_levels(tagged({"position": "TE", "ppg": 7.0}), {"TE": 12})
        assert levels["TE"] == pytest.approx(7.0)

    def test_a_position_with_nobody_free_is_omitted(self) -> None:
        levels = wire_replacement_levels(
            tagged({"position": "RB", "ppg": 9.0, IS_FREE_AGENT: False}), {"RB": 1}
        )
        assert "RB" not in levels


class TestRankedWire:
    def test_v2_wire_uses_availability_adjusted_projection(self) -> None:
        wire = ranked_wire(
            tagged(
                {
                    "player_display_name": "Durable",
                    "ppg": 8.0,
                    "season_equivalent_ppg": 14.0,
                    "v2_score": 5.0,
                },
                {
                    "player_display_name": "Fragile",
                    "ppg": 20.0,
                    "season_equivalent_ppg": 9.0,
                    "v2_score": 3.0,
                },
                {
                    "player_display_name": "Bar",
                    "ppg": 5.0,
                    "season_equivalent_ppg": 5.0,
                    "v2_score": 0.0,
                },
            ),
            {"RB": 3},
            min_vor=None,
        )

        assert wire["player_display_name"].to_list() == ["Durable", "Fragile", "Bar"]
        assert wire[WIRE_VOR][0] == pytest.approx(9.0)

    def test_rostered_players_never_appear(self) -> None:
        wire = ranked_wire(
            tagged(
                {"player_display_name": "Rostered", "ppg": 20.0, IS_FREE_AGENT: False},
                {"player_display_name": "Free", "ppg": 9.0},
                {"player_display_name": "Also Free", "ppg": 5.0},
            ),
            {"RB": 2},
            min_vor=None,
        )
        assert "Rostered" not in wire["player_display_name"].to_list()

    def test_ranked_by_value_against_the_free_pool(self) -> None:
        wire = ranked_wire(
            tagged(
                {"player_display_name": "Best", "ppg": 15.0},
                {"player_display_name": "Mid", "ppg": 9.0},
                {"player_display_name": "Bar", "ppg": 5.0},
            ),
            {"RB": 3},
            min_vor=None,
        )
        assert wire["player_display_name"].to_list() == ["Best", "Mid", "Bar"]
        assert wire[WIRE_VOR][0] == pytest.approx(10.0)

    def test_min_vor_floors_the_list(self) -> None:
        wire = ranked_wire(
            tagged(
                {"player_display_name": "Good", "ppg": 15.0},
                {"player_display_name": "Bar", "ppg": 5.0},
            ),
            {"RB": 2},
            min_vor=1.0,
        )
        assert wire["player_display_name"].to_list() == ["Good"]

    def test_injured_players_are_kept_by_default(self) -> None:
        """An OUT player at the top of the wire is information — a cheap stash — not an
        error. Hiding him is opt-in."""
        wire = ranked_wire(
            tagged(
                {"player_display_name": "Hurt Star", "ppg": 15.0, INJURY_STATUS: "OUT"},
                {"player_display_name": "Bar", "ppg": 5.0},
            ),
            {"RB": 2},
            min_vor=None,
        )
        assert "Hurt Star" in wire["player_display_name"].to_list()

    @pytest.mark.parametrize("tag", ["OUT", "INJURY_RESERVE", "SUSPENSION"])
    def test_healthy_only_hides_the_unstartable(self, tag: str) -> None:
        wire = ranked_wire(
            tagged(
                {"player_display_name": "Hurt Star", "ppg": 15.0, INJURY_STATUS: tag},
                {"player_display_name": "Bar", "ppg": 5.0},
            ),
            {"RB": 2},
            min_vor=None,
            healthy_only=True,
        )
        assert "Hurt Star" not in wire["player_display_name"].to_list()

    def test_questionable_survives_the_healthy_filter(self) -> None:
        """Questionable players still play most weeks; dropping them would hide real
        value behind a tag that resolves on Sunday morning."""
        wire = ranked_wire(
            tagged(
                {"player_display_name": "Q Guy", "ppg": 15.0, INJURY_STATUS: "QUESTIONABLE"},
                {"player_display_name": "Bar", "ppg": 5.0},
            ),
            {"RB": 2},
            min_vor=None,
            healthy_only=True,
        )
        assert "Q Guy" in wire["player_display_name"].to_list()


class TestRosterHealth:
    def test_v2_roster_sorts_on_v2_score(self) -> None:
        health = roster_health(
            tagged(
                {
                    "player_display_name": "Historical",
                    "adj_vor": 10.0,
                    "v2_score": 1.0,
                    IS_MINE: True,
                },
                {
                    "player_display_name": "Projected",
                    "adj_vor": 1.0,
                    "v2_score": 10.0,
                    IS_MINE: True,
                },
            ),
            my_team_id=3,
        )

        assert health["player_display_name"].to_list() == ["Projected", "Historical"]

    def test_decisions_surface_above_non_decisions(self) -> None:
        health = roster_health(
            tagged(
                {"player_display_name": "Fine", "adj_vor": 9.0, IS_MINE: True},
                {"player_display_name": "Moved", "adj_vor": 5.0, IS_MINE: True, CHANGED_TEAM: True},
                {
                    "player_display_name": "Hurt",
                    "adj_vor": 1.0,
                    IS_MINE: True,
                    INJURY_STATUS: "QUESTIONABLE",
                },
            ),
            my_team_id=3,
        )
        assert health["player_display_name"].to_list() == ["Hurt", "Moved", "Fine"]

    def test_only_my_players(self) -> None:
        health = roster_health(
            tagged(
                {"player_display_name": "Mine", IS_MINE: True},
                {"player_display_name": "Theirs", IS_MINE: False},
            ),
            my_team_id=3,
        )
        assert health["player_display_name"].to_list() == ["Mine"]


class TestOpponentWeaknesses:
    def test_reports_each_team_s_best_at_a_position(self) -> None:
        weak = opponent_weaknesses(
            tagged(
                {
                    "player_display_name": "A",
                    "position": "TE",
                    "adj_vor": -2.0,
                    IS_FREE_AGENT: False,
                    OWNER_TEAM_NAME: "Rivals",
                },
                {
                    "player_display_name": "B",
                    "position": "RB",
                    "adj_vor": 8.0,
                    IS_FREE_AGENT: False,
                    OWNER_TEAM_NAME: "Rivals",
                },
            ),
            BASELINES,
        )
        rows = {r["position"]: r for r in weak.iter_rows(named=True)}

        assert rows["TE"]["best_vor"] == pytest.approx(-2.0)
        assert rows["RB"]["best_player"] == "B"

    def test_free_agents_are_not_anyone_s_weakness(self) -> None:
        weak = opponent_weaknesses(tagged({"position": "TE", IS_FREE_AGENT: True}), BASELINES)
        assert weak.height == 0


class TestTeamStrengths:
    def test_optimizes_the_roster_instead_of_trusting_current_lineup_slots(self) -> None:
        board = pl.DataFrame(
            {
                ESPN_ID: [1, 2, 3, 4],
                "player_display_name": ["Low starter", "Bench star", "Mid", "Low bench"],
                "position": ["WR", "WR", "WR", "WR"],
                "proj_ppg": [5.0, 20.0, 12.0, 4.0],
                "projected_volatility": [0.0, 0.0, 0.0, 0.0],
                "projected_availability": [1.0, 1.0, 1.0, 1.0],
            }
        )
        espn = pl.DataFrame(
            {
                ESPN_ID: [1, 2, 3, 4],
                "player_display_name": ["Low starter", "Bench star", "Mid", "Low bench"],
                "position": ["WR", "WR", "WR", "WR"],
                OWNER_TEAM_ID: [1, 1, 2, 2],
                "projected_points": [85.0, 340.0, 204.0, 68.0],
                "lineup_slot": ["WR", "BE", "WR", "BE"],
            }
        )

        strengths = team_strengths(board, espn, [1, 2])

        assert strengths[1]["expected_weekly_points"] == pytest.approx(20.0)
        assert strengths[2]["expected_weekly_points"] == pytest.approx(12.0)
        assert strengths[1]["team_rank"] == 1
        assert strengths[2]["team_rank"] == 2

    def test_availability_uses_the_bench_and_adds_weekly_risk(self) -> None:
        board = pl.DataFrame(
            {
                ESPN_ID: [1, 2],
                "player_display_name": ["Fragile star", "Steady bench"],
                "position": ["WR", "WR"],
                "proj_ppg": [20.0, 10.0],
                "projected_volatility": [4.0, 2.0],
                "projected_availability": [0.5, 1.0],
            }
        )
        espn = pl.DataFrame(
            {
                ESPN_ID: [1, 2],
                "player_display_name": ["Fragile star", "Steady bench"],
                "position": ["WR", "WR"],
                OWNER_TEAM_ID: [1, 1],
                "projected_points": [340.0, 170.0],
                "lineup_slot": ["WR", "BE"],
            }
        )

        team = team_strengths(board, espn, [1])[1]

        assert team["expected_weekly_points"] == pytest.approx(15.0, abs=0.2)
        assert team["lineup_coverage"] == pytest.approx(1.0)
        assert team["bench_rescue_points"] == pytest.approx(5.0, abs=0.2)
        assert team["weekly_risk"] > 5.0

    def test_weekly_risk_is_separate_from_equal_expected_points(self) -> None:
        board = pl.DataFrame(
            {
                ESPN_ID: [1, 2],
                "player_display_name": ["Volatile", "Steady"],
                "position": ["WR", "WR"],
                "proj_ppg": [10.0, 10.0],
                "projected_volatility": [6.0, 1.0],
                "projected_availability": [1.0, 1.0],
            }
        )
        espn = pl.DataFrame(
            {
                ESPN_ID: [1, 2],
                "player_display_name": ["Volatile", "Steady"],
                "position": ["WR", "WR"],
                OWNER_TEAM_ID: [1, 2],
                "projected_points": [170.0, 170.0],
                "lineup_slot": ["WR", "WR"],
            }
        )

        strengths = team_strengths(board, espn, [1, 2])

        assert strengths[1]["team_score"] == pytest.approx(5.0)
        assert strengths[2]["team_score"] == pytest.approx(5.0)
        assert strengths[1]["weekly_risk"] == pytest.approx(6.0)
        assert strengths[2]["weekly_risk"] == pytest.approx(1.0)

    def test_unmatched_rookie_uses_espn_projection_with_visible_fallback(self) -> None:
        board = pl.DataFrame(
            {
                ESPN_ID: [1],
                "player_display_name": ["Veteran"],
                "position": ["WR"],
                "proj_ppg": [10.0],
                "projected_volatility": [4.0],
                "projected_availability": [1.0],
            }
        )
        espn = pl.DataFrame(
            {
                ESPN_ID: [99],
                "player_display_name": ["Rookie"],
                "position": ["WR"],
                OWNER_TEAM_ID: [1],
                "projected_points": [255.0],
                "lineup_slot": ["WR"],
            }
        )

        team = team_strengths(board, espn, [1])[1]

        assert team["fallback_players"] == 1
        assert team["expected_weekly_points"] == pytest.approx(14.1, abs=0.2)


class TestUnrankable:
    def test_lists_players_with_no_prior_tape(self) -> None:
        frame = pl.DataFrame(
            {
                "player_id": [None, "g1"],
                "player_display_name": ["Rookie", "Veteran"],
                "position": ["WR", "WR"],
                "espn_team": ["NO", "SF"],
                PERCENT_OWNED: [49.8, 10.0],
                OWNER_TEAM_NAME: [None, None],
                INJURY_STATUS: ["ACTIVE", "ACTIVE"],
            }
        )
        unrankable = unrankable_players(frame)

        assert unrankable["player_display_name"].to_list() == ["Rookie"]

    def test_sorted_by_roster_rate_so_the_hyped_ones_lead(self) -> None:
        frame = pl.DataFrame(
            {
                "player_id": [None, None],
                "player_display_name": ["Quiet", "Hyped"],
                "position": ["WR", "WR"],
                "espn_team": ["NO", "GB"],
                PERCENT_OWNED: [2.0, 49.8],
                OWNER_TEAM_NAME: [None, None],
                INJURY_STATUS: ["ACTIVE", "ACTIVE"],
            }
        )
        assert unrankable_players(frame)["player_display_name"].to_list() == ["Hyped", "Quiet"]
