"""Forecast-cutoff, exposure and horizon safeguards for the QB variation search."""

from copy import deepcopy

import numpy as np
import polars as pl
import pytest

from patron.metrics.qb_variations import (
    augment_panel,
    bridge_candidates,
    choose,
    fitted,
    policy_ranges,
    score,
)


def example(future_yards=200, future_attempts=25):
    panel = [
        dict(
            player_id="qb",
            season=2020,
            through_week=2,
            horizon="rest_of_season",
            end_week=7,
            scheduled_games=4,
            x_horizon_games=4,
            x_elapsed=2,
            execution_ypa=7.0,
            team="NE",
            actual=400.0,
        )
    ]
    weeks = pl.DataFrame(
        [
            dict(
                player_id="qb",
                season=2019,
                week=1,
                season_type="REG",
                team="NE",
                attempts=30,
                passing_yards=240,
                league_points=12.0,
            ),
            dict(
                player_id="qb",
                season=2020,
                week=1,
                season_type="REG",
                team="NE",
                attempts=20,
                passing_yards=140,
                league_points=10.0,
            ),
            dict(
                player_id="qb",
                season=2020,
                week=3,
                season_type="REG",
                team="NE",
                attempts=future_attempts,
                passing_yards=future_yards,
                league_points=11.0,
            ),
            dict(
                player_id="qb",
                season=2020,
                week=7,
                season_type="REG",
                team="NE",
                attempts=25,
                passing_yards=200,
                league_points=12.0,
            ),
        ]
    )
    features = pl.DataFrame(
        [dict(player_id="qb", forecast_season=2020, ngs_cpoe=1.0, team_changed=False)]
    )
    ranking = [
        dict(
            player_id="qb",
            position="QB",
            season=2020,
            through_week=2,
            horizon="next4",
            end_week=6,
            scheduled_games=3,
            actual=11.0,
        )
    ]
    return augment_panel(panel, weeks, features, ranking)


def test_future_outcomes_do_not_change_any_rate_or_profile_feature():
    before, _ = example()
    after, _ = example(future_yards=900, future_attempts=80)

    def predictors(r):
        return {k: v for k, v in r.items() if k.startswith(("rate_", "f_", "e_", "x_"))}

    assert predictors(before[0]) == predictors(after[0])
    assert before[0]["rate_reference"] == 7.0


def test_ranking_bridge_uses_four_calendar_weeks_and_checks_labels():
    _, ranking = example()
    assert ranking[0]["horizon"] == "ranking_next4"
    assert ranking[0]["actual"] == 200
    assert ranking[0]["actual_points"] == 11
    assert ranking[0]["scheduled_games"] == ranking[0]["x_horizon_games"] == 3


def test_undefined_rates_do_not_become_zero_efficiency_observations():
    rows = [
        dict(season=2020, actual=0.0, actual_attempts=0, prediction=7.0),
        dict(season=2020, actual=200.0, actual_attempts=25, prediction=7.0),
        dict(season=2021, actual=6.0, actual_attempts=1, prediction=7.0),
    ]
    rate = score(rows, "prediction", "rate")
    total = score(rows, "prediction", "yards")
    assert rate["n"] == 2
    assert total["n"] == 3
    assert rate["mse"] == pytest.approx(1.0)


def test_selection_uses_supplied_completed_folds_and_season_equal_weights():
    rows = [
        dict(
            season=y,
            horizon="next_game",
            through_week=1,
            actual=7.0,
            actual_attempts=1,
            a=7.0,
            b=9.0,
        )
        for y in (2010, 2011, 2012)
    ]
    assert choose(rows, "next_game", "weekly", ["a", "b"], "b", "rate") == "a"
    assert choose(rows[:2], "next_game", "weekly", ["a", "b"], "b", "rate") == "b"
    duplicated = deepcopy(rows) + [deepcopy(rows[0]) for _ in range(100)]
    assert score(duplicated, "a", "rate")["mse"] == score(rows, "a", "rate")["mse"]


def test_bridge_rejects_same_year_upstream_training():
    with pytest.raises(ValueError, match="strictly earlier"):
        bridge_candidates([dict(season=2020)], [dict(season=2020)], 0.04)


def test_future_tracking_coverage_cannot_enable_an_untrained_feature():
    train = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    test = np.array([[2.0, 0.0], [2.0, 10000.0]])
    result = fitted("ridge", 10, train, np.array([6.0, 7.0, 8.0]), test)
    assert result[0] == pytest.approx(result[1])


def test_ranges_cannot_learn_their_own_season_errors():
    rows = [
        dict(
            player_id=str(i),
            season=year,
            through_week=2,
            horizon="next4",
            role_group="current_substantial",
            scheduled_games=4,
            allowed_fraction=1.0,
            actual=900.0,
            yards_policy=800.0,
        )
        for year in (2020, 2021, 2022)
        for i in range(100)
    ]
    before = policy_ranges(rows)
    for r in rows:
        if r["season"] == 2021:
            r["actual"] = 0.0
    after = policy_ranges(rows)
    for a, b in zip(before, after, strict=True):
        if a["season"] <= 2021:
            assert (a["lower"], a["upper"]) == (b["lower"], b["upper"])
    assert before[-1]["lower"] != after[-1]["lower"]
