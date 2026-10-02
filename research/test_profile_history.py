"""Leakage, coverage and all-history-training tests for profile forecasts."""

import copy

import numpy as np
import polars as pl
import pytest
from profile_history import (
    build_features,
    college_features,
    fit_predict,
    injury_features,
    nfl_features,
    predict,
    specifications,
    tree_matrices,
    verify_cross_version,
)


def candidate(pid="a", year=2019, population="returner"):
    return dict(
        player_id=pid,
        forecast_season=year,
        forecast_cutoff_date=f"{year}-08-01",
        position="QB",
        player_population=population,
        games=4,
        age_at_season=25,
        player_experience=2,
        actual_season_points=100.0,
        actual_games=10,
        actual_availability_value=5.0,
        market_ecr=20.0,
        market_overall_ecr=120.0,
        market_overall_ecr_score=-120.0,
        outcome_complete=True,
    )


def week(year=2018, points=10, share=0.8):
    return dict(
        player_id="a",
        season=year,
        week=1,
        team="BAL",
        position="QB",
        league_points=float(points),
        attempts=20.0,
        passing_yards=150.0,
        passing_tds=1.0,
        carries=5.0,
        rushing_yards=30.0,
        rushing_tds=0.0,
        targets=0.0,
        receptions=0.0,
        receiving_yards=0.0,
        receiving_tds=0.0,
        bonus_pts=0.0,
        stat_recorded=True,
        offense_snaps=50.0,
        snap_share=share,
    )


def college_row(year=2016):
    return dict(
        college_id="c",
        season=year,
        team_count=1,
        complete_team_season=True,
        coverage=1.0,
        season_end_date=f"{year + 1}-01-01",
        passing_yards=2000.0,
        passing_tds=20.0,
        passing_ints=5.0,
        carries=100.0,
        rushing_yards=700.0,
        rushing_tds=5.0,
        receptions=0.0,
        receiving_yards=0.0,
        receiving_tds=0.0,
        share_receiving_yards=0.0,
        share_receptions=0.0,
        share_rushing_yards=0.5,
        share_carries=0.3,
    )


def inputs():
    return (
        pl.DataFrame([candidate(), candidate("missing", population="rookie")]),
        pl.DataFrame([week(2010), week(2018), week(2019)]),
        pl.DataFrame([dict(player_id="a", rookie_season=2010, draft_year=2010, draft_pick=20)]),
        pl.DataFrame([college_row(2009), college_row(2019)]),
        pl.DataFrame([dict(player_id="a", college_id="c", status="linked", method="provider_id")]),
        pl.DataFrame(
            [
                dict(
                    player_id="a",
                    season=y,
                    week=1,
                    report_status="Out",
                    practice_status="Did Not Participate In Practice",
                )
                for y in (2010, 2019)
            ]
        ),
        [],
    )


def test_missing_profile_and_rookie_are_retained():
    rows = build_features(*inputs())
    assert len(rows) == 2
    rookie = next(r for r in rows if r["player_id"] == "missing")
    assert rookie["eligible"]
    assert not rookie["profile_identity_known"]
    assert rookie["career_observed_weeks"] == 0
    assert rookie["career_points_per_observed_week"] is None
    assert rookie["college_linked"] == 0
    assert rookie["context_cutoff_state_observed"] is None
    assert rookie["evidence_state"] == "unknown"
    assert all(
        not f.startswith(("actual_", "pred_"))
        for fields in specifications(rows).values()
        for f in fields
    )


def test_early_tree_folds_keep_missingness_without_learning_from_future_values():
    train = np.array([[np.nan, 1, 5, 0], [np.nan, 1, np.nan, 1]], dtype=float)
    test = np.array([[99, 2, 5, 2]], dtype=float)
    left, right = tree_matrices(train, test)
    assert left.tolist() == [[0, 0], [1, 1]]
    assert right.tolist() == [[0, 2]]
    rows = synthetic_rows(range(2004, 2007))
    for r in rows:
        r["unavailable"] = None
    future = [{**r, "unavailable": 999} for r in synthetic_rows([2007])]
    assert np.isfinite(fit_predict(rows, future, ("unavailable",), "boost")).all()


def test_future_nfl_college_injuries_do_not_change_features():
    args = list(inputs())
    before = build_features(*args)
    args[1] = args[1].with_columns(
        pl.when(pl.col("season") == 2019)
        .then(9999.0)
        .otherwise(pl.col("league_points"))
        .alias("league_points")
    )
    args[3] = args[3].with_columns(
        pl.when(pl.col("season") == 2019)
        .then(99999.0)
        .otherwise(pl.col("passing_yards"))
        .alias("passing_yards")
    )
    args[5] = args[5].with_columns(
        pl.when(pl.col("season") == 2019)
        .then(pl.lit("Questionable"))
        .otherwise(pl.col("report_status"))
        .alias("report_status")
    )
    assert before == build_features(*args)


def test_old_career_history_is_kept_alongside_recent_history():
    c, identity = candidate(), {"rookie_season": 2010}
    before = nfl_features([week(2010, 20), week(2018, 10)], c, identity, 2001)
    after = nfl_features([week(2010, 100), week(2018, 10)], c, identity, 2001)
    assert before["career_points_per_observed_week"] != after["career_points_per_observed_week"]
    assert before["career_attempts"] == 40
    assert (
        before["career_recent3_points_per_observed_week"]
        == after["career_recent3_points_per_observed_week"]
    )
    assert (
        before["basic_prior_points_per_observed_week"]
        == after["basic_prior_points_per_observed_week"]
    )
    assert before["career_high_observed_weeks"] == 2


def test_early_missing_snap_and_injury_feeds_stay_unknown():
    r = nfl_features([week(2003, share=None)], candidate(year=2004), {"rookie_season": 1998}, 2001)
    assert r["career_snap_share"] is None
    assert r["career_unknown_observed_weeks"] == 1
    assert r["career_left_truncated"] == 1
    assert injury_features([], 2004, 2009)["injury_prior_reported_weeks"] is None
    assert injury_features([], 2020, 2009)["injury_prior_reported_weeks"] == 0


def test_college_quality_gaps_preserve_earlier_observations():
    good = college_row(2015)
    bad = {**college_row(2016), "complete_team_season": False}
    got = college_features([good, bad], candidate(), {}, ["provider_id"], {"career_attempts": 0})
    assert got["college_partial"] == 1
    assert got["college_complete_seasons"] == 1
    assert got["college_career_passing_yards"] == 2000
    assert got["college_latest_passing_yards"] == 2000
    assert got["college_prior_weight"] == 1
    experienced = college_features([good], candidate(), {}, [], {"career_attempts": 800})
    assert experienced["college_prior_weight"] == pytest.approx(0.2)
    assert experienced["college_career_passing_yards"] == 2000


def test_future_context_and_duplicate_candidates_fail_closed():
    args = list(inputs())
    args[0] = args[0].with_columns(pl.lit("2020-01-01").alias("availability_latest_known_on"))
    with pytest.raises(ValueError, match="Future context"):
        build_features(*args)
    args = list(inputs())
    args[0] = pl.concat([args[0], args[0].head(1)])
    with pytest.raises(ValueError, match="Duplicate"):
        build_features(*args)


def synthetic_rows(years):
    return [
        dict(
            player_id=str(i),
            forecast_season=y,
            position="QB",
            basic_signal=float(i),
            actual_points_per_scheduled_game=i / 10 + (y - 2000) / 20,
            known_available_games_cap=None,
            market_ecr=None,
        )
        for y in years
        for i in range(35)
    ]


def test_expanding_fit_uses_all_earlier_years_and_five_year_control_does_not():
    rows = synthetic_rows(range(2004, 2020))
    specs = {"profile": ("basic_signal",)}
    before, folds, skipped = predict(rows, specs, architectures=("ridge",))
    fold = next(r for r in folds if r["season"] == 2019)
    assert fold["first_train_season"] == 2004
    assert fold["train"] == 15 * 35
    assert fold["five_year_train"] == 5 * 35
    assert skipped[0]["season"] == 2004
    changed = copy.deepcopy(rows)
    for r in changed:
        if r["forecast_season"] == 2004:
            r["actual_points_per_scheduled_game"] += 20
    after, _, _ = predict(changed, specs, architectures=("ridge",))
    left = [r for r in before if r["forecast_season"] == 2019]
    right = [r for r in after if r["forecast_season"] == 2019]
    assert left[0]["pred_ridge_profile"] != right[0]["pred_ridge_profile"]
    assert [r["pred_ridge_profile_five_year"] for r in left] == [
        r["pred_ridge_profile_five_year"] for r in right
    ]


@pytest.mark.parametrize("architecture", ["ridge", "boost"])
def test_held_out_outcomes_are_not_inputs_and_absence_is_not_double_discounted(architecture):
    train = synthetic_rows(range(2004, 2007))
    test = synthetic_rows([2007])
    test[0]["known_available_games_cap"] = 0
    test[1]["known_available_games_cap"] = 8
    before = fit_predict(train, test, ("basic_signal",), architecture)
    changed = [{**r, "actual_points_per_scheduled_game": 99999} for r in test]
    after = fit_predict(train, changed, ("basic_signal",), architecture)
    assert np.array_equal(before, after)
    assert before[0] == 0
    changed[1]["known_available_games_cap"] = None
    assert fit_predict(train, changed, ("basic_signal",), architecture)[1] == before[1]
    with pytest.raises(ValueError, match="overlaps"):
        fit_predict(train + test, test, ("basic_signal",), architecture)


def test_cross_version_only_allows_restoring_missing_cutoffs():
    old = pl.DataFrame([candidate()])
    latest = old.with_columns(pl.lit(None, dtype=pl.String).alias("forecast_cutoff_date"))
    verify_cross_version(latest, old)
    with pytest.raises(ValueError, match="cutoffs"):
        verify_cross_version(old, latest)
    with pytest.raises(ValueError, match="invariants"):
        verify_cross_version(old, old.with_columns(pl.lit(999.0).alias("actual_season_points")))
    with pytest.raises(ValueError, match="cutoffs"):
        verify_cross_version(
            old, old.with_columns(pl.lit("2019-08-02").alias("forecast_cutoff_date"))
        )
