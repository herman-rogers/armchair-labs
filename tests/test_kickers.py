"""Unit tests for kicker scoring and its bucket-alignment guard."""

from __future__ import annotations

import polars as pl
import pytest

from patron.scoring.kickers import (
    KICKER_POINTS,
    BracketAlignmentError,
    aggregate_kicker_seasons,
    score_kicker_weeks,
    validate_bracket_alignment,
)
from patron.scoring.rules import ScoringRules, get_rules

KICKING_DEFAULTS = {
    "fg_made_0_19": 0.0,
    "fg_made_20_29": 0.0,
    "fg_made_30_39": 0.0,
    "fg_made_40_49": 0.0,
    "fg_made_50_59": 0.0,
    "fg_made_60_": 0.0,
    "fg_missed": 0.0,
    "fg_blocked": 0.0,
    "pat_made": 0.0,
}


def kicker_week(week: int = 1, **stats: float) -> pl.DataFrame:
    row = {
        "player_id": "00-0000K01",
        "player_display_name": "Test Kicker",
        "position": "K",
        "season": 2025,
        "week": week,
        "team": "SEA",
        **KICKING_DEFAULTS,
        **stats,
    }
    return pl.DataFrame([row])


def points(**stats: float) -> float:
    return float(score_kicker_weeks(kicker_week(**stats))[KICKER_POINTS][0])


class TestDistanceBrackets:
    @pytest.mark.parametrize(
        ("column", "expected"),
        [
            ("fg_made_0_19", 3.0),
            ("fg_made_20_29", 3.0),
            ("fg_made_30_39", 3.0),
            ("fg_made_40_49", 4.0),
            ("fg_made_50_59", 5.0),
            ("fg_made_60_", 6.0),
        ],
    )
    def test_each_bucket_scores_its_bracket(self, column: str, expected: float) -> None:
        assert points(**{column: 1.0}) == pytest.approx(expected)

    def test_a_full_day(self) -> None:
        # 2x30-39 (6) + 1x40-49 (4) + 1x50-59 (5) + 3 PAT (3) - 1 miss (-1) = 17
        assert points(
            fg_made_30_39=2.0,
            fg_made_40_49=1.0,
            fg_made_50_59=1.0,
            pat_made=3.0,
            fg_missed=1.0,
        ) == pytest.approx(17.0)

    def test_the_long_range_spread_is_real(self) -> None:
        """Three 55-yarders outscore three 35-yarders by six points — the spread the
        room ignores when it treats kickers as random."""
        assert points(fg_made_50_59=3.0) - points(fg_made_30_39=3.0) == pytest.approx(6.0)


class TestExtraPointsAndMisses:
    def test_extra_points_score_one(self) -> None:
        assert points(pat_made=4.0) == pytest.approx(4.0)

    def test_misses_cost_one(self) -> None:
        assert points(fg_missed=3.0) == pytest.approx(-3.0)

    def test_blocked_kicks_are_free_by_default(self) -> None:
        assert points(fg_blocked=2.0) == pytest.approx(0.0)

    def test_blocked_kicks_can_be_scored_as_misses(self) -> None:
        rules = get_rules().model_copy(deep=True)
        rules.kicking.blocked_counts_as_miss = True

        scored = score_kicker_weeks(kicker_week(fg_blocked=2.0), rules=rules)
        assert float(scored[KICKER_POINTS][0]) == pytest.approx(-2.0)


class TestNullHandling:
    def test_nulls_score_as_zero(self) -> None:
        frame = kicker_week(fg_made_40_49=1.0).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("pat_made")
        )
        assert float(score_kicker_weeks(frame)[KICKER_POINTS][0]) == pytest.approx(4.0)


class TestBracketAlignment:
    def test_configured_league_brackets_align(self) -> None:
        validate_bracket_alignment()

    def test_a_bracket_edge_inside_a_bucket_is_rejected(self) -> None:
        """A 45-yard threshold would split nflverse's 40-49 bucket, so bucket counts
        could no longer be scored exactly. Better to fail than to approximate silently.
        """
        rules = ScoringRules.model_validate(
            {
                **get_rules().model_dump(),
                "kicking": {
                    "field_goals": [
                        {"min_yards": 0, "max_yards": 44, "points": 3},
                        {"min_yards": 45, "max_yards": None, "points": 5},
                    ],
                    "extra_point": 1,
                    "missed_field_goal": -1,
                    "blocked_counts_as_miss": False,
                },
            }
        )
        with pytest.raises(BracketAlignmentError, match="fg_made_40_49"):
            validate_bracket_alignment(rules)


class TestSeasonAggregation:
    def test_long_range_counts_survive_aggregation(self) -> None:
        weeks = pl.concat(
            [
                kicker_week(fg_made_50_59=1.0, pat_made=2.0),
                kicker_week(fg_made_60_=1.0, pat_made=3.0, week=2),
            ]
        )
        seasons = aggregate_kicker_seasons(score_kicker_weeks(weeks))

        assert seasons.height == 1
        row = seasons.to_dicts()[0]
        assert row["games"] == 2
        assert row["fg_50s"] == pytest.approx(1.0)
        assert row["fg_60s"] == pytest.approx(1.0)
        assert row[KICKER_POINTS] == pytest.approx(16.0)  # 5+2 then 6+3
        assert row["ppg"] == pytest.approx(8.0)
