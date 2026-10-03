"""Unit tests for name normalization."""

from __future__ import annotations

import pytest

from engine.names import match_key, normalize


class TestNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Puka Nacua", "puka nacua"),
            ("PUKA NACUA", "puka nacua"),
            ("  Puka   Nacua  ", "puka nacua"),
        ],
    )
    def test_case_and_whitespace(self, raw: str, expected: str) -> None:
        assert normalize(raw) == expected

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("D.J. Moore", "DJ Moore"),
            ("A.J. Brown", "AJ Brown"),
            ("T.J. Hockenson", "TJ Hockenson"),
            ("D.K. Metcalf", "DK Metcalf"),
        ],
    )
    def test_initials_match_with_or_without_periods(self, left: str, right: str) -> None:
        """The canonical case from the plan's §8, and the one most likely to silently
        drop a player from the ranked wire."""
        assert normalize(left) == normalize(right)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("Ja'Marr Chase", "JaMarr Chase"),
            ("Wan'Dale Robinson", "WanDale Robinson"),
            ("De'Von Achane", "DeVon Achane"),
        ],
    )
    def test_apostrophes_are_removed(self, left: str, right: str) -> None:
        assert normalize(left) == normalize(right)

    def test_curly_apostrophes_match_straight_ones(self) -> None:
        assert normalize("Ja’Marr Chase") == normalize("Ja'Marr Chase")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Harold Fannin Jr.", "harold fannin"),
            ("Deebo Samuel Sr.", "deebo samuel"),
            ("Kenneth Walker III", "kenneth walker"),
            ("Oronde Gadsden II", "oronde gadsden"),
            ("Marvin Harrison Jr.", "marvin harrison"),
        ],
    )
    def test_suffixes_are_dropped(self, raw: str, expected: str) -> None:
        assert normalize(raw) == expected

    def test_hyphenated_names_collapse(self) -> None:
        assert normalize("Amon-Ra St. Brown") == "amonra st brown"

    def test_accents_are_stripped(self) -> None:
        assert normalize("Audric Estimé") == normalize("Audric Estime")

    def test_a_one_word_name_that_looks_like_a_suffix_survives(self) -> None:
        """Guard against stripping a name down to nothing."""
        assert normalize("Jr") == "jr"


class TestMatchKey:
    def test_position_is_part_of_the_key(self) -> None:
        assert match_key("Josh Allen", "QB") == ("josh allen", "QB")

    def test_same_name_different_position_does_not_collide(self) -> None:
        """There really are two Josh Allens in the league."""
        assert match_key("Josh Allen", "QB") != match_key("Josh Allen", "LB")

    def test_position_case_is_normalized(self) -> None:
        assert match_key("Josh Allen", "qb") == match_key("Josh Allen", "QB")
