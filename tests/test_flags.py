"""Unit tests for flag composition."""

from __future__ import annotations

import polars as pl
import pytest

from patron.board.flags import FLAGS_COLUMN, add_flags


def flags(
    buy: bool = False,
    td_luck: bool = False,
    age: bool = False,
    games: int = 17,
    small_sample_games: int = 10,
) -> str:
    frame = pl.DataFrame(
        {
            "td_regress_up": [buy],
            "td_regress_down": [td_luck],
            "rb_age_cliff": [age],
            "games": [games],
        }
    )
    return add_flags(frame, small_sample_games=small_sample_games)[FLAGS_COLUMN][0]


class TestSingleFlags:
    def test_no_flags_renders_empty_not_null(self) -> None:
        assert flags() == ""

    def test_buy(self) -> None:
        assert flags(buy=True) == "BUY"

    def test_td_luck(self) -> None:
        assert flags(td_luck=True) == "TD-luck"

    def test_age(self) -> None:
        assert flags(age=True) == "age"

    def test_small_sample_names_the_games_count(self) -> None:
        assert flags(games=8) == "8gms"


class TestComposition:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"td_luck": True, "age": True}, "TD-luck/age"),
            ({"buy": True, "age": True}, "BUY/age"),
            ({"td_luck": True, "games": 9}, "TD-luck/9gms"),
            ({"age": True, "games": 9}, "age/9gms"),
            ({"buy": True, "age": True, "games": 4}, "BUY/age/4gms"),
        ],
    )
    def test_order_is_regression_then_age_then_sample(self, kwargs: dict, expected: str) -> None:
        """Every combination in the published board renders exactly as it does there,
        so the column stays scannable at draft speed."""
        assert flags(**kwargs) == expected


class TestSampleThreshold:
    @pytest.mark.parametrize(("games", "expected"), [(9, "9gms"), (10, ""), (11, ""), (17, "")])
    def test_threshold_is_exclusive(self, games: int, expected: str) -> None:
        assert flags(games=games) == expected

    def test_threshold_is_configurable(self) -> None:
        assert flags(games=12, small_sample_games=14) == "12gms"


class TestExclusivity:
    def test_buy_wins_if_both_somehow_fire(self) -> None:
        """They cannot both be true by construction, but a silent double-render would
        be worse than a deterministic choice if that ever changed."""
        assert flags(buy=True, td_luck=True) == "BUY"
