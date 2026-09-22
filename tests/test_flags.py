"""Unit tests for flag composition."""

from __future__ import annotations

import polars as pl
import pytest

from patron.board.flags import FLAGS_COLUMN, add_flags


def flags(
    age: bool = False,
    games: int = 17,
    small_sample_games: int = 10,
) -> str:
    frame = pl.DataFrame(
        {
            "rb_age_cliff": [age],
            "games": [games],
        }
    )
    return add_flags(frame, small_sample_games=small_sample_games)[FLAGS_COLUMN][0]


class TestSingleFlags:
    def test_no_flags_renders_empty_not_null(self) -> None:
        assert flags() == ""

    def test_age(self) -> None:
        assert flags(age=True) == "age"

    def test_small_sample_names_the_games_count(self) -> None:
        assert flags(games=8) == "8gms"


class TestComposition:
    def test_order_is_age_then_sample(self) -> None:
        """The fixed order keeps the column scannable at draft speed."""
        assert flags(age=True, games=9) == "age/9gms"


class TestSampleThreshold:
    @pytest.mark.parametrize(("games", "expected"), [(9, "9gms"), (10, ""), (11, ""), (17, "")])
    def test_threshold_is_exclusive(self, games: int, expected: str) -> None:
        assert flags(games=games) == expected

    def test_threshold_is_configurable(self) -> None:
        assert flags(games=12, small_sample_games=14) == "12gms"
