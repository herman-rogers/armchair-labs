from datetime import date

import numpy as np
import polars as pl
import pytest
from nextgen_signal_audit import (
    admitted,
    college_features,
    coverage,
    fit_predict,
    quarantine_targets,
    season_summary,
)

from patron.metrics.player_profile import career_features


def training():
    return pl.DataFrame(
        {
            "forecast_season": [2004] * 12 + [2005] * 12 + [2006] * 12,
            "position": ["RB"] * 36,
            "player_population": ["returner"] * 36,
            "target": [float(i % 12) for i in range(36)],
            "signal": [float(i % 12) for i in range(36)],
            "known_available_games_cap": [None] * 36,
        }
    )


def heldout():
    return training().head(3).with_columns(pl.lit(2007).alias("forecast_season"))


def test_predictions_ignore_heldout_targets_and_use_missing_features():
    train, test = training(), heldout().with_columns(pl.Series("signal", [None, 5.0, 10.0]))
    expected = fit_predict(train, test, ["signal"])
    changed = test.with_columns(pl.lit(999999.0).alias("target"))
    np.testing.assert_array_equal(expected, fit_predict(train, changed, ["signal"]))
    assert len(expected) == 3 and np.isfinite(expected).all()


def test_rejects_future_training_and_population_mixing():
    with pytest.raises(ValueError, match="overlaps"):
        fit_predict(training(), training().head(2), ["signal"])
    mixed = training().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("rookie"))
        .otherwise(pl.col("player_population"))
        .alias("player_population")
    )
    with pytest.raises(ValueError, match="Mixed"):
        fit_predict(mixed, heldout(), ["signal"])


def test_future_profile_weeks_cannot_change_career_features():
    before = dict(season=2003, week=1, stat_recorded=True, league_points=8.0, snap_share=None)
    after = dict(season=2004, week=1, stat_recorded=True, league_points=9999.0, snap_share=0.9)
    assert career_features([before], 2004) == career_features([before, after], 2004)
    assert career_features([before, after], 2005)["completed_observed_weeks"] == 2


def test_full_career_retains_older_observations_than_recent_three():
    observations = [
        dict(season=y, week=1, stat_recorded=True, league_points=points, snap_share=None)
        for y, points in [(2001, 100.0), (2005, 10.0)]
    ]
    result = career_features(observations, 2006)
    assert result["completed_observed_weeks"] == 2
    assert result["career_points_per_observed_week"] == 55
    assert result["recent3_points_per_observed_week"] == 10


def test_family_requires_three_substantively_observed_seasons():
    assert admitted(training(), ["signal"])
    assert not admitted(training().filter(pl.col("forecast_season") > 2004), ["signal"])
    assert not admitted(training().with_columns(pl.lit(None).alias("signal")), ["signal"])
    assert not admitted(training().group_by("forecast_season").head(9), ["signal"])


def test_false_boolean_is_not_counted_as_observed_positive_fact():
    frame = heldout().with_columns(pl.Series("fact", [True, False, None]))
    result = coverage(frame, ["fact"]).to_dicts()[0]
    assert result["observed"] == 1 and result["candidates"] == 3
    assert result["semantics"] == "true"


def test_nan_is_not_numeric_coverage():
    frame = heldout().with_columns(pl.Series("signal", [float("nan"), None, 0.0]))
    result = coverage(frame, ["signal"]).to_dicts()[0]
    assert result["observed"] == 1 and result["semantics"] == "finite"


def test_quarantine_retains_early_candidates_and_masks_only_affected_inputs():
    weeks = pl.DataFrame(
        {
            "season": [2003] * 10 + [2009],
            "position": ["WR"] * 11,
            "targets": [0.0] * 10 + [5.0],
            "receptions": [2.0] * 11,
        }
    )
    features = pl.DataFrame(
        {
            "source_season": [2003, 2009],
            "targets": [0.0, 5.0],
            "receptions": [2.0, 2.0],
            "target_share": [0.0, 0.2],
            "air_yards_share": [0.1, 0.3],
            "receiving_first_down_rate": [2.0, 0.4],
            "rich_s0_opportunities_mean": [0.0, 5.0],
            "ppg": [10.0, 12.0],
        }
    )
    view, observations, quality = quarantine_targets(features, weeks)
    assert quality["source_seasons"] == [2003]
    assert view.height == features.height and observations.height == weeks.height
    assert view["targets"].to_list() == [None, 5.0]
    assert view["rich_s0_opportunities_mean"].to_list() == [None, 5.0]
    assert view["ppg"].to_list() == [10.0, 12.0]
    assert observations["targets"].null_count() == 10


def test_full_absence_forces_zero_without_double_discounting_partial_absence():
    test = heldout().with_columns(pl.Series("known_available_games_cap", [0.0, 8.0, None]))
    values = fit_predict(training(), test, [])
    assert values[0] == 0 and values[1] == values[2] > 0


def test_college_filters_future_incomplete_and_ambiguous_history():
    candidates = pl.DataFrame(
        {
            "player_id": ["p"],
            "forecast_season": [2010],
            "player_population": ["rookie"],
            "forecast_cutoff_date": [date(2010, 8, 1)],
        }
    )
    links = pl.DataFrame({"player_id": ["p"], "college_id": ["c"], "status": ["linked"]})
    base = dict(
        college_id="c",
        season=2009,
        season_end_date="2010-01-01",
        complete_team_season=True,
        team_count=1,
        passing_yards=100.0,
        share_receiving_yards=0.3,
        share_carries=0.4,
    )
    annual = pl.DataFrame([base])
    expected = college_features(candidates, annual, links).to_dicts()
    future = pl.DataFrame(
        [
            {**base, "season": 2010, "passing_yards": 9999.0},
            {**base, "season": 2008, "season_end_date": "2010-09-01"},
            {**base, "season": 2007, "complete_team_season": False},
        ]
    )
    assert college_features(candidates, pl.concat([annual, future]), links).to_dicts() == expected
    assert expected[0]["college_latest_passing_yards"] == 100
    ambiguous = college_features(candidates, pl.concat([annual, annual]), links)
    assert ambiguous["college_latest_passing_yards"][0] is None


def test_uncertainty_keeps_one_year_unquantified():
    assert season_summary([2])["bootstrap_95"] is None
    result = season_summary([1, 2, 3])
    assert result["positive_seasons"] == 3 and result["mean"] == 2
    assert result["leave_one_year_out"] == [1.5, 2.5]
