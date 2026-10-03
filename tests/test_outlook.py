import copy

import polars as pl
import pytest

from engine.metrics.outlook import (
    OUTLOOK,
    TARGETS,
    build_rows,
    evaluate,
    predict_fold,
    publication_rows,
)


def inputs():
    points = pl.DataFrame(
        [
            dict(
                player_id="a",
                season=2025,
                week=w,
                position="RB",
                team="BUF",
                league_points=float(w * 3),
                carries=w,
                targets=2,
                attempts=0,
                bonus_pts=0,
            )
            for w in range(1, 7)
        ]
    )
    snaps = points.select("player_id", "season", "week").with_columns(
        pl.lit(30).alias("offense_snaps"), pl.lit(0.5).alias("offense_pct")
    )
    schedule = pl.DataFrame(
        [
            dict(season=2025, week=w, home_team="BUF", away_team="MIA", game_type="REG")
            for w in range(1, 7)
        ]
    )
    candidates = pl.DataFrame(
        [dict(player_id="a", season=2025, position="RB", team="BUF", population="rookie")]
    )
    return points, snaps, schedule, candidates


def rows(points, snaps, schedule, candidates, **kw):
    return build_rows(
        points,
        snaps,
        schedule,
        candidates,
        [2],
        completed_seasons={2025},
        snap_complete_weeks={(2025, w) for w in range(1, 7)},
        snap_identities=kw.pop("snap_identities", {"a"}),
        **kw,
    )


@pytest.mark.parametrize("cutoff,remaining", [(14, 4), (16, 2), (17, 1), (18, 0)])
def test_late_season_outlook_ends_at_regular_season(cutoff, remaining):
    points, snaps, _, candidates = inputs()
    schedule = pl.DataFrame(
        [
            dict(season=2025, week=w, home_team="BUF", away_team="MIA", game_type="REG")
            for w in range(1, 19)
        ]
    )
    result = build_rows(
        points,
        snaps,
        schedule,
        candidates,
        [cutoff],
        completed_seasons={2025},
        snap_complete_weeks={(2025, w) for w in range(1, 19)},
        snap_identities={"a"},
    )
    assert result[0]["future_team_games"] == remaining
    assert result[0]["next4_points"] == 0
    assert result[0]["next4_offensive_weeks"] == 0


def test_future_changes_cannot_change_features_or_eligibility():
    points, snaps, schedule, candidates = inputs()
    before = rows(points, snaps, schedule, candidates)
    changed = points.with_columns(
        pl.when(pl.col("week") > 2)
        .then(900)
        .otherwise(pl.col("league_points"))
        .alias("league_points"),
        pl.when(pl.col("week") > 2)
        .then(pl.lit("WR"))
        .otherwise(pl.col("position"))
        .alias("position"),
        pl.when(pl.col("week") > 2).then(pl.lit("MIA")).otherwise(pl.col("team")).alias("team"),
    )
    # An unknown player who appears only later must not enter the decision pool.
    changed = pl.concat(
        [
            changed,
            changed.filter(pl.col("week") > 2).with_columns(pl.lit("future").alias("player_id")),
        ]
    )
    after = rows(changed, snaps, schedule, candidates)
    assert len(before) == len(after) == 1
    assert before[0]["next4_points"] != after[0]["next4_points"]
    assert {k: v for k, v in before[0].items() if k not in TARGETS} == {
        k: v for k, v in after[0].items() if k not in TARGETS
    }


def test_missing_identity_is_unknown_not_zero_participation():
    points, snaps, schedule, candidates = inputs()
    row = rows(points, snaps.head(0), schedule, candidates, snap_identities=set())[0]
    assert row["recent_snap_share"] is None
    assert row["past_offensive_fraction"] is None
    assert row["next4_offensive_weeks"] is None
    assert row["eligible"]  # Box-score usage still provides evidence.
    known = rows(points, snaps.head(0), schedule, candidates)[0]
    assert known["recent_snap_share"] == 0
    assert known["next4_offensive_weeks"] == 0


def test_no_appearance_has_no_forecast_or_false_trend():
    points, snaps, schedule, candidates = inputs()
    row = rows(points.head(0), snaps.head(0), schedule, candidates)[0]
    assert not row["eligible"]
    assert row["points_pace_next4"] is None
    assert row["opportunity_trend"] == row["role_evidence"] == "insufficient history"


def test_live_prior_sample_and_pending_outcomes():
    points, snaps, schedule, candidates = inputs()
    prior = snaps.with_columns(pl.lit(2024).alias("season"))
    schedule = pl.concat(
        [schedule, schedule.with_columns(pl.lit(2024).alias("season"))], how="vertical_relaxed"
    )
    row = build_rows(
        points.filter(pl.col("week") <= 2),
        snaps.head(2),
        schedule,
        candidates,
        [2],
        completed_seasons=set(),
        snap_complete_weeks={(y, w) for y in [2024, 2025] for w in range(1, 7)},
        snap_identities={"a"},
        prior_snaps=prior,
    )[0]
    assert row["prior_offensive_weeks"] == 6
    assert all(row[key] is None for key in TARGETS)


def test_bye_is_not_an_extra_future_game():
    points, snaps, schedule, candidates = inputs()
    schedule = schedule.filter(pl.col("week") != 4)
    row = rows(points, snaps, schedule, candidates)[0]
    assert row["future_team_games"] == 3
    assert row["points_pace_next4"] == row["recent_points_pg"] * 3


def test_duplicate_scoring_fails_closed():
    points, snaps, schedule, candidates = inputs()
    with pytest.raises(ValueError, match="Duplicate"):
        rows(pl.concat([points, points.head(1)]), snaps, schedule, candidates)


def model_rows():
    return [
        dict(
            **{f: (i % 10) + 0.1 * (y - 2013) for f in OUTLOOK},
            player_id=str(i),
            season=y,
            position="RB",
            cutoff_week=2,
            eligible=True,
            snap_feed_complete=True,
            population="rookie" if i < 5 else "returner",
            next4_points=i + 3.0,
            next4_offensive_weeks=3,
            next4_active_snap_share=0.5,
            points_pace_next4=i + 10.0,
            participation_pace_next4=4.0,
            recent_active_snap_share=0.6,
        )
        for y in range(2013, 2026)
        for i in range(40)
    ]


def test_calibration_and_test_years_never_fit_or_change_prediction():
    history = model_rows()
    test = [r for r in history if r["season"] == 2025]
    first = predict_fold(history, test)
    assert first[0]["train_seasons"] == list(range(2013, 2023))
    assert first[0]["calibration_seasons"] == [2023, 2024]
    changed = copy.deepcopy(history)
    for row in changed:
        if row["season"] >= 2023:
            row["next4_points"] = 10000.0
    second = predict_fold(changed, test)
    assert first[0]["forecast_next4"] == second[0]["forecast_next4"]
    assert first[0]["research_range_high"] != second[0]["research_range_high"]
    changed = copy.deepcopy(history)
    for row in changed:
        if row["season"] == 2025:
            row["next4_points"] = 10000.0
    assert predict_fold(changed, test) == first
    assert all(r["confidence_score"] is None for r in first)


def test_ineligible_players_never_receive_estimates():
    history = model_rows()
    test = [{**history[-1], "eligible": False}]
    prediction = predict_fold(history, test)[0]
    assert prediction["forecast_next4"] is None
    assert prediction["expected_offensive_weeks"] is None
    assert prediction["expected_active_snap_share"] is None


def test_ranges_require_every_applicable_gate_and_never_publish_confidence():
    row = dict(
        position="RB",
        cutoff_week=2,
        population="rookie",
        prior_offensive_weeks=0,
        snap_feed_complete=True,
        forecast_next4=20.0,
        research_range_low=-1.0,
        research_range_high=60.0,
        usage_range_low=0.0,
        usage_range_high=60.0,
        next4_points=99.0,
    )
    gates = [
        dict(position="RB", cutoff_week=2, cohort=c, range_gate_passed=True)
        for c in ("all", "rookie", "small_prior_sample")
    ]
    public = publication_rows([row], gates[:-1])[0]
    assert public["outcome_range"] is None
    assert not any(k in public for k in ("next4_points", "research_range_low", "usage_range_low"))
    public = publication_rows([row], gates)[0]
    assert public["outcome_range"] == dict(low=-1.0, high=60.0, nominal_coverage=0.8)
    assert public["confidence_score"] is None
    assert (
        publication_rows([{**row, "prior_offensive_weeks": None}], gates)[0]["outcome_range"]
        is None
    )


def test_incomplete_snap_feed_withholds_participation_not_points():
    history = model_rows()
    prediction = predict_fold(history, [{**history[-1], "snap_feed_complete": False}])[0]
    assert prediction["forecast_next4"] is not None
    assert prediction["expected_offensive_weeks"] is None
    assert prediction["expected_active_snap_share"] is None


def test_established_players_require_checks_for_their_actual_published_cohort():
    row = dict(
        position="QB",
        cutoff_week=2,
        population="returner",
        prior_offensive_weeks=17,
        snap_feed_complete=True,
        forecast_next4=80,
        research_range_low=50,
        research_range_high=110,
    )
    gate = dict(position="QB", cutoff_week=2, cohort="all", range_gate_passed=True)
    assert publication_rows([row], [gate])[0]["outcome_range"] is None
    qualified = {**gate, "cohort": "established_prior_sample"}
    assert publication_rows([row], [gate, qualified])[0]["outcome_range"] is not None


def test_evaluation_uses_earlier_folds_and_cannot_pass_short_history():
    history = model_rows()
    predictions = predict_fold(history, [r for r in history if r["season"] == 2025])
    summary = evaluate(predictions)[0]
    assert summary["folds"][0]["season"] == 2025
    assert not summary["publication_checks"]["enough_seasons"]
    assert not summary["range_gate_passed"]


def test_mixed_forecast_fold_is_rejected():
    history = model_rows()
    with pytest.raises(ValueError, match="one season"):
        predict_fold(history, [history[0], history[-1]])


def test_participation_forecast_cannot_exceed_scheduled_games():
    history = model_rows()
    prediction = predict_fold(history, [{**history[-1], "future_team_games": 1}])[0]
    assert 0 <= prediction["expected_offensive_weeks"] <= 1


def test_participation_baseline_uses_recent_window_not_season_average():
    points, snaps, schedule, candidates = inputs()
    snaps = snaps.with_columns(
        pl.when(pl.col("week") >= 3)
        .then(0)
        .otherwise(pl.col("offense_snaps"))
        .alias("offense_snaps")
    )
    row = build_rows(
        points,
        snaps,
        schedule,
        candidates,
        [5],
        completed_seasons=set(),
        snap_complete_weeks={(2025, w) for w in range(1, 7)},
        snap_identities={"a"},
    )[0]
    assert row["past_offensive_fraction"] == 0.4
    assert row["recent_offensive_fraction"] == row["participation_pace_next4"] == 0
