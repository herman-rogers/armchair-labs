"""Unit tests for the component scorer.

Every expected value here is computed by hand from the league sheet in §3 of the plan
and written out in the assertion message, so a failure tells you which rule broke
rather than just which number moved.
"""

from __future__ import annotations

import polars as pl
import pytest

from patron.scoring.columns import MissingColumnsError
from patron.scoring.engine import (
    BASE_POINTS,
    COMPONENT_POINTS,
    LEAGUE_POINTS,
    add_league_points,
    base_points_from_nflverse,
    score_components,
)
from tests.factories import player_week, player_weeks


def points(**stats: float) -> float:
    """Score a single player-week built from `stats`."""
    frame = score_components(player_weeks(player_week(**stats)))
    return float(frame[COMPONENT_POINTS][0])


class TestPassing:
    def test_yardage_is_one_point_per_25(self) -> None:
        assert points(passing_yards=300.0) == pytest.approx(12.0)

    def test_touchdowns_are_four(self) -> None:
        assert points(passing_tds=3.0) == pytest.approx(12.0)

    def test_interceptions_cost_two(self) -> None:
        assert points(passing_interceptions=2.0) == pytest.approx(-4.0)

    def test_two_point_conversions_score_two(self) -> None:
        assert points(passing_2pt_conversions=1.0) == pytest.approx(2.0)

    def test_full_stat_line(self) -> None:
        # 312 yds (12.48) + 2 TD (8) - 1 INT (-2) = 18.48
        assert points(
            passing_yards=312.0, passing_tds=2.0, passing_interceptions=1.0
        ) == pytest.approx(18.48)


class TestRushing:
    def test_yardage_is_one_point_per_ten(self) -> None:
        assert points(rushing_yards=87.0) == pytest.approx(8.7)

    def test_touchdowns_are_six(self) -> None:
        assert points(rushing_tds=2.0) == pytest.approx(12.0)

    def test_negative_yardage_subtracts(self) -> None:
        assert points(rushing_yards=-4.0) == pytest.approx(-0.4)


class TestReceiving:
    def test_full_ppr(self) -> None:
        assert points(receptions=8.0) == pytest.approx(8.0)

    def test_yardage_is_one_point_per_ten(self) -> None:
        assert points(receiving_yards=95.0) == pytest.approx(9.5)

    def test_touchdowns_are_six(self) -> None:
        assert points(receiving_tds=1.0) == pytest.approx(6.0)

    def test_full_stat_line(self) -> None:
        # 8 rec (8) + 95 yds (9.5) + 1 TD (6) = 23.5
        assert points(receptions=8.0, receiving_yards=95.0, receiving_tds=1.0) == pytest.approx(
            23.5
        )


class TestFumbles:
    @pytest.mark.parametrize(
        "column",
        ["rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"],
    )
    def test_every_lost_fumble_costs_two(self, column: str) -> None:
        assert points(**{column: 1.0}) == pytest.approx(-2.0)

    def test_sack_fumbles_are_not_forgotten(self) -> None:
        """A sack fumble is a lost fumble; nflverse just tracks it in its own column.

        Regression guard: it is the easiest of the three to leave out of the sum.
        """
        assert points(rushing_fumbles_lost=1.0, sack_fumbles_lost=1.0) == pytest.approx(-4.0)


class TestNullHandling:
    def test_nulls_score_as_zero_not_null(self) -> None:
        """A receiver's passing columns are null, not zero. Nulls must not poison the sum."""
        row = player_week(receptions=5.0, receiving_yards=50.0)
        row["passing_yards"] = None
        row["passing_tds"] = None

        scored = score_components(pl.DataFrame([row]))
        assert scored[COMPONENT_POINTS][0] == pytest.approx(10.0)


class TestValidation:
    def test_missing_columns_raise_and_name_themselves(self) -> None:
        frame = pl.DataFrame({"passing_yards": [100.0]})
        with pytest.raises(MissingColumnsError) as excinfo:
            score_components(frame)

        assert "receptions" in str(excinfo.value)
        assert "columns.py" in str(excinfo.value), "error should say where to fix a rename"


class TestLeaguePoints:
    def test_base_plus_bonus(self) -> None:
        frame = pl.DataFrame({BASE_POINTS: [18.5], "bonus_pts": [3.0]})
        assert add_league_points(frame)[LEAGUE_POINTS][0] == pytest.approx(21.5)

    def test_missing_bonus_column_scores_base_alone(self) -> None:
        frame = pl.DataFrame({BASE_POINTS: [18.5]})
        assert add_league_points(frame)[LEAGUE_POINTS][0] == pytest.approx(18.5)

    def test_null_bonus_is_zero_not_null(self) -> None:
        """Most player-weeks earn no bonus and carry a null after the left join."""
        frame = pl.DataFrame({BASE_POINTS: [18.5], "bonus_pts": [None]}, strict=False)
        assert add_league_points(frame)[LEAGUE_POINTS][0] == pytest.approx(18.5)

    def test_base_points_read_from_nflverse_column(self) -> None:
        frame = pl.DataFrame({"fantasy_points_ppr": [22.4]})
        assert base_points_from_nflverse(frame)[BASE_POINTS][0] == pytest.approx(22.4)


class TestSpecialTeams:
    """Return touchdowns.

    Found by `tests/test_parity.py`, not by reading the league sheet: the design doc's
    §3 summary omits them entirely, and 19 player-weeks in 2025 disagreed with
    nflverse by exactly six points each until this was added.
    """

    def test_a_return_touchdown_scores_six(self) -> None:
        assert points(special_teams_tds=1.0) == pytest.approx(6.0)

    def test_return_touchdowns_stack_with_the_rest_of_the_line(self) -> None:
        # 4 rec (4) + 40 yds (4) + 1 return TD (6) = 14
        assert points(receptions=4.0, receiving_yards=40.0, special_teams_tds=1.0) == pytest.approx(
            14.0
        )
