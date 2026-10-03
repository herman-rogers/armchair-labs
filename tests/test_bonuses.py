"""Unit tests for big-play touchdown bonuses.

This module carries three of the defects found in the draft-prep script, so the tests
are written as regression guards against each one specifically:

* bonuses keyed by season AND week, so weekly league points can include them;
* brackets exclusive, so a 55-yard touchdown pays 3 and not 5;
* null player ids dropped explicitly and counted, not silently grouped away.
"""

from __future__ import annotations

import polars as pl
import pytest

from engine.scoring.bonuses import (
    BONUS_POINTS,
    extract_touchdown_bonuses,
    season_bonus_totals,
)
from engine.scoring.columns import MissingColumnsError
from tests.factories import pbp, touchdown_play

QB = "00-0000QB1"
WR = "00-0000WR1"
RB = "00-0000RB1"


def bonus_for(player_id: str, frame: pl.DataFrame) -> float:
    matched = frame.filter(pl.col("player_id") == player_id)
    return float(matched[BONUS_POINTS].sum()) if matched.height else 0.0


class TestBrackets:
    @pytest.mark.parametrize(
        ("yards", "expected"),
        [
            (1.0, 0.0),
            (39.0, 0.0),  # just under the first bracket
            (40.0, 2.0),  # lower edge of 40-49
            (49.0, 2.0),  # upper edge of 40-49
            (50.0, 3.0),  # lower edge of 50+
            (75.0, 3.0),
            (99.0, 3.0),
        ],
    )
    def test_bracket_edges(self, yards: float, expected: float) -> None:
        bonuses, _ = extract_touchdown_bonuses(pbp(touchdown_play(yards, kind="rush")))
        assert bonus_for(RB, bonuses) == pytest.approx(expected)

    def test_brackets_are_exclusive_not_cumulative(self) -> None:
        """A 55-yard touchdown clears both thresholds but pays the higher one only.

        Summing the brackets instead of picking one would pay 5. The league sheet says
        "higher bracket only".
        """
        bonuses, _ = extract_touchdown_bonuses(pbp(touchdown_play(55.0, kind="rush")))
        assert bonus_for(RB, bonuses) == pytest.approx(3.0)

    def test_non_scoring_plays_are_ignored(self) -> None:
        long_gain = touchdown_play(60.0, kind="rush", touchdown=0)
        bonuses, audit = extract_touchdown_bonuses(pbp(long_gain))

        assert bonuses.height == 0
        assert audit.scoring_plays == 0


class TestCreditAttribution:
    def test_passing_touchdown_pays_passer_and_receiver(self) -> None:
        bonuses, _ = extract_touchdown_bonuses(pbp(touchdown_play(52.0, kind="pass")))

        assert bonus_for(QB, bonuses) == pytest.approx(3.0)
        assert bonus_for(WR, bonuses) == pytest.approx(3.0)

    def test_rushing_touchdown_pays_only_the_rusher(self) -> None:
        bonuses, _ = extract_touchdown_bonuses(pbp(touchdown_play(45.0, kind="rush")))

        assert bonus_for(RB, bonuses) == pytest.approx(2.0)
        assert bonuses.height == 1, "a rushing touchdown must not credit anyone else"

    def test_a_player_accumulates_across_plays(self) -> None:
        bonuses, _ = extract_touchdown_bonuses(
            pbp(
                touchdown_play(41.0, kind="rush", week=1),
                touchdown_play(60.0, kind="rush", week=1),
            )
        )
        assert bonus_for(RB, bonuses) == pytest.approx(5.0)


class TestWeeklyKeying:
    def test_bonuses_are_split_by_week(self) -> None:
        """The defect this module fixes: aggregating straight to a season total meant
        weekly league points silently excluded the bonus, so floor and volatility were
        computed on different scoring than the PPG beside them."""
        bonuses, _ = extract_touchdown_bonuses(
            pbp(
                touchdown_play(45.0, kind="rush", week=1),
                touchdown_play(55.0, kind="rush", week=7),
            )
        )

        assert set(bonuses.columns) == {"season", "week", "player_id", BONUS_POINTS}
        by_week = {int(row["week"]): row[BONUS_POINTS] for row in bonuses.to_dicts()}
        assert by_week == {1: pytest.approx(2.0), 7: pytest.approx(3.0)}

    def test_season_totals_collapse_the_weeks(self) -> None:
        bonuses, _ = extract_touchdown_bonuses(
            pbp(
                touchdown_play(45.0, kind="rush", week=1),
                touchdown_play(55.0, kind="rush", week=7),
            )
        )
        totals = season_bonus_totals(bonuses)

        assert totals.height == 1
        assert totals[BONUS_POINTS][0] == pytest.approx(5.0)

    def test_seasons_do_not_bleed_together(self) -> None:
        bonuses, _ = extract_touchdown_bonuses(
            pbp(
                touchdown_play(45.0, kind="rush", season=2024),
                touchdown_play(55.0, kind="rush", season=2025),
            )
        )
        totals = season_bonus_totals(bonuses)
        by_season = {int(row["season"]): row[BONUS_POINTS] for row in totals.to_dicts()}

        assert by_season == {2024: pytest.approx(2.0), 2025: pytest.approx(3.0)}


class TestSeasonTypeFiltering:
    def test_postseason_is_excluded_by_default(self) -> None:
        """Weekly stats are filtered to REG, so bonuses must be too — otherwise a
        playoff run inflates a regular-season PPG."""
        bonuses, audit = extract_touchdown_bonuses(
            pbp(touchdown_play(55.0, kind="rush", season_type="POST"))
        )

        assert bonuses.height == 0
        assert audit.scoring_plays == 0

    def test_postseason_can_be_requested(self) -> None:
        bonuses, _ = extract_touchdown_bonuses(
            pbp(touchdown_play(55.0, kind="rush", season_type="POST")),
            season_types=("REG", "POST"),
        )
        assert bonus_for(RB, bonuses) == pytest.approx(3.0)


class TestReconciliation:
    def test_audit_balances_when_every_id_is_present(self) -> None:
        _, audit = extract_touchdown_bonuses(
            pbp(
                touchdown_play(52.0, kind="pass"),  # 3 pts x 2 credits = 6
                touchdown_play(45.0, kind="rush"),  # 2 pts x 1 credit  = 2
            )
        )

        assert audit.scoring_plays == 2
        assert audit.expected_points == pytest.approx(8.0)
        assert audit.credited_points == pytest.approx(8.0)
        assert audit.dropped_credits == 0
        assert audit.is_balanced

    def test_a_null_id_is_counted_not_swallowed(self) -> None:
        """Grouping without a null filter collects these under a null key that joins to
        nothing, and the points vanish with no error. The audit is what makes that
        loss visible."""
        _, audit = extract_touchdown_bonuses(
            pbp(touchdown_play(52.0, kind="pass", receiver_player_id=None))
        )

        assert audit.dropped_credits == 1
        assert audit.expected_points == pytest.approx(6.0)
        assert audit.credited_points == pytest.approx(3.0), "only the passer was credited"
        assert not audit.is_balanced
        assert "UNBALANCED" in audit.summary()

    def test_null_key_never_reaches_the_output(self) -> None:
        bonuses, _ = extract_touchdown_bonuses(
            pbp(touchdown_play(52.0, kind="pass", receiver_player_id=None))
        )
        assert bonuses.filter(pl.col("player_id").is_null()).height == 0


class TestValidation:
    def test_missing_columns_raise(self) -> None:
        with pytest.raises(MissingColumnsError, match="yards_gained"):
            extract_touchdown_bonuses(pl.DataFrame({"touchdown": [1]}))
