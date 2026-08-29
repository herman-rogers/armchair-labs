"""Unit tests for the rate, opportunity, regression, and age metrics."""

from __future__ import annotations

import polars as pl
import pytest

from patron.metrics.age import AGE_COLUMN, add_age, add_age_flags
from patron.metrics.opportunity import WEIGHTED_OPPORTUNITY, aggregate_opportunity
from patron.metrics.rates import weekly_rates
from patron.metrics.regression import (
    EXPECTED_TDS,
    TD_OVER_EXPECTATION,
    add_regression_flags,
    add_td_over_expectation,
    positional_td_rates,
)
from patron.scoring.engine import LEAGUE_POINTS


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
            "wopr": 0.0,
        }
        return pl.DataFrame(
            [
                {"player_id": "p1", "season": 2025, "week": i + 1, **base, **row}
                for i, row in enumerate(rows)
            ]
        )

    def test_weighted_opportunity_prices_a_target_above_a_carry(self) -> None:
        row = aggregate_opportunity(self._frame([{"carries": 10.0, "targets": 5.0}])).to_dicts()[0]

        # 10 carries + 2.2 x 5 targets = 21
        assert row[WEIGHTED_OPPORTUNITY] == pytest.approx(21.0)

    def test_target_weight_is_configurable(self) -> None:
        row = aggregate_opportunity(
            self._frame([{"carries": 10.0, "targets": 5.0}]), target_weight=1.0
        ).to_dicts()[0]
        assert row[WEIGHTED_OPPORTUNITY] == pytest.approx(15.0)

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


class TestTdOverExpectation:
    def _seasons(self, rows: list[dict]) -> pl.DataFrame:
        base = {"season": 2025, "carries": 0.0, "targets": 0.0, "total_tds": 0.0}
        return pl.DataFrame([{"player_id": f"p{i}", **base, **row} for i, row in enumerate(rows)])

    def test_positional_rate_is_touchdowns_per_touch(self) -> None:
        frame = self._seasons(
            [
                {"position": "RB", "carries": 200.0, "targets": 50.0, "total_tds": 10.0},
                {"position": "RB", "carries": 100.0, "targets": 50.0, "total_tds": 10.0},
            ]
        )
        # 20 TDs on 400 touches = 0.05
        assert positional_td_rates(frame).to_dicts()[0]["pos_td_rate"] == pytest.approx(0.05)

    def test_expectation_is_touches_times_the_positional_rate(self) -> None:
        frame = self._seasons(
            [
                {"position": "RB", "carries": 200.0, "targets": 0.0, "total_tds": 20.0},
                {"position": "RB", "carries": 200.0, "targets": 0.0, "total_tds": 0.0},
            ]
        )
        # Pooled: 20 TDs / 400 carries = 0.05; each expects 200 * 0.05 = 10.
        rows = add_td_over_expectation(frame).sort("player_id").to_dicts()

        assert rows[0][EXPECTED_TDS] == pytest.approx(10.0)
        assert rows[0][TD_OVER_EXPECTATION] == pytest.approx(10.0)
        assert rows[1][TD_OVER_EXPECTATION] == pytest.approx(-10.0)

    def test_window_all_pools_seasons_window_season_does_not(self) -> None:
        frame = pl.DataFrame(
            [
                {
                    "player_id": "a",
                    "season": 2024,
                    "position": "RB",
                    "carries": 100.0,
                    "targets": 0.0,
                    "total_tds": 20.0,
                },
                {
                    "player_id": "b",
                    "season": 2025,
                    "position": "RB",
                    "carries": 100.0,
                    "targets": 0.0,
                    "total_tds": 0.0,
                },
            ]
        )
        pooled = positional_td_rates(frame, window="all").to_dicts()
        assert len(pooled) == 1
        assert pooled[0]["pos_td_rate"] == pytest.approx(0.10)

        per_season = {
            row["season"]: row["pos_td_rate"]
            for row in positional_td_rates(frame, window="season").to_dicts()
        }
        assert per_season == {2024: pytest.approx(0.20), 2025: pytest.approx(0.0)}

    def test_an_unknown_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="window must be"):
            positional_td_rates(self._seasons([{"position": "RB"}]), window="rolling")


class TestRegressionFlags:
    def _row(self, tdoe: float, opportunity: float) -> dict:
        frame = pl.DataFrame(
            {"player_id": ["p1"], TD_OVER_EXPECTATION: [tdoe], "wtd_opp": [opportunity]}
        )
        return add_regression_flags(frame).to_dicts()[0]

    @pytest.mark.parametrize(("tdoe", "expected"), [(3.9, False), (4.0, True), (6.0, True)])
    def test_sell_flag_boundary(self, tdoe: float, expected: bool) -> None:
        assert self._row(tdoe, 200.0)["td_regress_down"] is expected

    @pytest.mark.parametrize(("tdoe", "expected"), [(-2.4, False), (-2.5, True), (-5.0, True)])
    def test_buy_flag_boundary(self, tdoe: float, expected: bool) -> None:
        assert self._row(tdoe, 200.0)["td_regress_up"] is expected

    def test_buy_flag_requires_volume(self) -> None:
        """Without the volume gate this fires on every deep reserve who caught four
        passes, and a good signal turns into noise."""
        assert self._row(-5.0, 149.0)["td_regress_up"] is False
        assert self._row(-5.0, 150.0)["td_regress_up"] is True

    def test_volume_alone_does_not_fire_the_buy_flag(self) -> None:
        assert self._row(0.0, 400.0)["td_regress_up"] is False

    def test_flags_are_mutually_exclusive(self) -> None:
        for tdoe in (-5.0, 0.0, 5.0):
            row = self._row(tdoe, 200.0)
            assert not (row["td_regress_up"] and row["td_regress_down"])


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
