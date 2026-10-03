"""Unit tests for the rate, opportunity, and age metrics."""

from __future__ import annotations

import polars as pl
import pytest

from engine.metrics.age import AGE_COLUMN, add_age, add_age_flags
from engine.metrics.opportunity import aggregate_opportunity
from engine.metrics.rates import weekly_rates
from engine.scoring.engine import LEAGUE_POINTS


def weeks(points: list[float], player_id: str = "p1", season: int = 2025) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [player_id] * len(points),
            "season": [season] * len(points),
            "week": list(range(1, len(points) + 1)),
            LEAGUE_POINTS: points,
        }
    )


class TestWeeklyRates:
    def test_games_total_and_ppg(self) -> None:
        row = weekly_rates(weeks([10.0, 20.0, 30.0])).to_dicts()[0]

        assert row["games"] == 3
        assert row["total_pts"] == pytest.approx(60.0)
        assert row["ppg"] == pytest.approx(20.0)

    def test_ppg_not_totals_is_the_currency(self) -> None:
        """Eight excellent games and seventeen mediocre ones can share a total. Only
        the per-game view separates them — the Skattebo/Nabers lesson."""
        eight_good = weekly_rates(weeks([20.0] * 8, player_id="hurt")).to_dicts()[0]
        seventeen_meh = weekly_rates(weeks([160.0 / 17] * 17, player_id="full")).to_dicts()[0]

        assert eight_good["total_pts"] == pytest.approx(seventeen_meh["total_pts"])
        assert eight_good["ppg"] > seventeen_meh["ppg"]

    def test_floor_is_the_quantile_of_weekly_scores(self) -> None:
        row = weekly_rates(weeks([0.0, 10.0, 20.0, 30.0, 40.0])).to_dicts()[0]
        assert row["floor"] == pytest.approx(10.0)

    def test_volatility_separates_the_same_average(self) -> None:
        """Two players, one average, different weekly shapes. Floor is what decides who
        starts in a week you are favoured to win."""
        steady = weekly_rates(weeks([15.0, 15.0, 15.0, 15.0], player_id="steady")).to_dicts()[0]
        spiky = weekly_rates(weeks([0.0, 0.0, 30.0, 30.0], player_id="spiky")).to_dicts()[0]

        assert steady["ppg"] == pytest.approx(spiky["ppg"])
        assert steady["volatility"] < spiky["volatility"]
        assert steady["floor"] > spiky["floor"]

    def test_single_game_volatility_is_null_not_zero(self) -> None:
        """Unknown and perfectly-consistent must not read the same on the board."""
        assert weekly_rates(weeks([15.0])).to_dicts()[0]["volatility"] is None

    def test_rates_run_on_league_points_including_bonuses(self) -> None:
        """Regression guard for the defect this module fixes: floor and volatility used
        to be computed on bonus-exclusive points while PPG included bonuses, so the
        three numbers on a row described different scoring systems."""
        frame = weeks([10.0, 10.0, 10.0]).with_columns(
            pl.Series("fantasy_points_ppr", [7.0, 7.0, 7.0])
        )
        row = weekly_rates(frame).to_dicts()[0]

        assert row["ppg"] == pytest.approx(10.0), "must read league_pts, not fantasy_points_ppr"
        assert row["floor"] == pytest.approx(10.0)


class TestOpportunity:
    def _frame(self, rows: list[dict[str, float]]) -> pl.DataFrame:
        base = {
            "carries": 0.0,
            "targets": 0.0,
            "receptions": 0.0,
            "target_share": 0.0,
            "air_yards_share": 0.0,
        }
        return pl.DataFrame(
            [
                {"player_id": "p1", "season": 2025, "week": i + 1, **base, **row}
                for i, row in enumerate(rows)
            ]
        )

    def test_raw_volume_is_summed(self) -> None:
        row = aggregate_opportunity(
            self._frame([{"carries": 10.0, "targets": 5.0}, {"carries": 4.0, "targets": 3.0}])
        ).to_dicts()[0]

        assert row["carries"] == pytest.approx(14.0)
        assert row["targets"] == pytest.approx(8.0)

    def test_shares_are_usage_weighted_not_flat_averaged(self) -> None:
        """The defect this fixes: a flat mean lets a 2-target injury week count as
        heavily as a 12-target week, understating players who missed time.

        Flat mean of (0.30, 0.10) is 0.20. Weighted by targets (12, 2) it is
        (0.30*12 + 0.10*2) / 14 = 0.271.
        """
        row = aggregate_opportunity(
            self._frame(
                [
                    {"targets": 12.0, "target_share": 0.30},
                    {"targets": 2.0, "target_share": 0.10},
                ]
            )
        ).to_dicts()[0]

        assert row["target_share"] == pytest.approx(0.2714, abs=1e-4)
        assert row["target_share"] > 0.20, "a flat mean would give exactly 0.20"

    def test_weeks_with_no_targets_do_not_vote(self) -> None:
        row = aggregate_opportunity(
            self._frame(
                [
                    {"targets": 10.0, "target_share": 0.25},
                    {"targets": 0.0, "target_share": 0.0},
                ]
            )
        ).to_dicts()[0]
        assert row["target_share"] == pytest.approx(0.25)


class TestAge:
    def _players(self, birth_dates: list[str | None], position: str = "RB") -> pl.DataFrame:
        return pl.DataFrame(
            {
                "player_id": [f"p{i}" for i in range(len(birth_dates))],
                "position": [position] * len(birth_dates),
                "birth_date": birth_dates,
            },
            schema_overrides={"birth_date": pl.String},
        ).with_columns(pl.col("birth_date").str.to_date())

    def test_age_is_measured_at_september_first(self) -> None:
        row = add_age(self._players(["1998-09-01"]), season=2026).to_dicts()[0]
        assert row[AGE_COLUMN] == pytest.approx(28.0, abs=0.05)

    def test_a_birthday_just_after_the_cutoff_still_counts_as_younger(self) -> None:
        before, after = add_age(self._players(["1998-08-31", "1998-09-02"]), season=2026)[
            AGE_COLUMN
        ].to_list()
        assert before > after

    def test_missing_birth_date_yields_null_not_zero(self) -> None:
        """A blank on the board beats silently treating an unknown player as young."""
        assert add_age(self._players([None]), season=2026).to_dicts()[0][AGE_COLUMN] is None

    def test_rb_cliff_flag_is_inclusive_at_the_threshold(self) -> None:
        frame = pl.DataFrame({"position": ["RB", "RB", "RB"], AGE_COLUMN: [27.4, 27.5, 30.0]})
        assert add_age_flags(frame)["rb_age_cliff"].to_list() == [False, True, True]

    def test_the_cliff_applies_to_backs_only(self) -> None:
        """Receivers age gently to 31-32 and quarterbacks later; the cliff is a running
        back phenomenon and flagging a 30-year-old receiver would be noise."""
        frame = pl.DataFrame(
            {"position": ["RB", "WR", "TE", "QB"], AGE_COLUMN: [29.0, 29.0, 29.0, 29.0]}
        )
        assert add_age_flags(frame)["rb_age_cliff"].to_list() == [True, False, False, False]

    def test_null_age_does_not_flag(self) -> None:
        frame = pl.DataFrame({"position": ["RB"], AGE_COLUMN: [None]}, strict=False)
        assert add_age_flags(frame)["rb_age_cliff"].to_list() == [False]
