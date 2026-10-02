"""Time-boundary and scenario checks for the offline QB experiment."""

import copy

import numpy as np
import polars as pl
import pytest
from qb_role_transition import (
    NEWS,
    ROLE,
    SPECS,
    add_efficiency,
    attach_evidence,
    audit_adp_windows,
    enrich_history,
    in_review,
    opening_starters,
    predict,
    resolve_evidence,
)


def row():
    return dict(
        player_id="a",
        season=2019,
        forecast_cutoff_date="2019-07-17",
        position="QB",
        player_population="returner",
        role_games=7,
        market_ecr=17,
        eligible=True,
    )


def evidence(state="starter", published="2019-07-16", event=""):
    return dict(
        player_id="a",
        season=2019,
        state=state,
        published_on=published,
        event_on=event,
        captured_on="2026-09-23",
        source_url="https://example.com/source",
    )


def test_cutoff_uses_publication_and_event_not_capture():
    r = row()
    assert resolve_evidence(r, [evidence()])[0] == "starter"
    assert resolve_evidence(r, [evidence(published="2019-07-18")])[0] == "unknown"
    assert resolve_evidence(r, [evidence(event="2019-07-18")])[0] == "unknown"
    same = evidence(published="2019-07-17")
    assert resolve_evidence(r, [same])[0] == "starter"
    assert resolve_evidence(r, [same], strict_day=True)[0] == "unknown"


def test_late_or_undated_adp_windows_are_not_backdated():
    windows = [
        dict(forecast_season=2019, source="ffc", window="2019-09-02..2019-09-04"),
        dict(forecast_season=2019, source="mfl", window="2019 after AUG1 pre-kickoff"),
        dict(forecast_season=2019, source="fixture", window="2019-07-01..2019-07-17"),
    ]
    audited = audit_adp_windows(windows, {2019: "2019-07-17"})
    assert [r["admissible"] for r in audited] == [False, False, True]
    assert audited[0]["reason"] == "window_ends_after_cutoff"
    assert audited[1]["window_end"] is None


def test_unknown_conflicts_and_supersession_are_explicit():
    r = row()
    assert resolve_evidence(r, [])[0] == "unknown"
    conflict = [evidence(), evidence("backup")]
    assert resolve_evidence(r, conflict)[:2] == ("unknown", "conflict")
    earlier = evidence("competition", published="2019-07-01")
    late = evidence("backup", published="2019-08-01")
    assert resolve_evidence(r, [earlier, evidence(), late])[0] == "starter"


def test_cohort_and_news_do_not_depend_on_future_success():
    before = row()
    after = {
        **before,
        "actual_season_points": -100,
        "actual_role_games": 0,
        "actual_opening_starter": 0,
    }
    assert in_review(before) and in_review(after)
    assert resolve_evidence(before, [evidence()]) == resolve_evidence(after, [evidence()])
    outside = {**before, "market_ecr": 33}
    assert resolve_evidence(outside, [evidence()])[:2] == ("unknown", "outside_review")
    assert all(attach_evidence([before], [])[0][f] == 0 for f in NEWS)


def schedule():
    return pl.DataFrame(
        [
            dict(
                game_id="x",
                season=2019,
                game_type="REG",
                gameday="2019-09-08",
                home_team="A",
                away_team="B",
                home_qb_id="a",
                away_qb_id="b",
            ),
            dict(
                game_id="y",
                season=2019,
                game_type="REG",
                gameday="2019-09-15",
                home_team="A",
                away_team="C",
                home_qb_id="replacement",
                away_qb_id="c",
            ),
        ]
    )


def test_opening_job_uses_first_team_game_not_first_week_or_full_season():
    assert opening_starters(schedule())[2019] == {"a", "b", "c"}
    missing = schedule().with_columns(pl.lit(None).alias("home_qb_id"))
    with pytest.raises(ValueError, match="Unknown"):
        opening_starters(missing)


def test_future_efficiency_and_snaps_cannot_change_prior_features():
    raw = pl.DataFrame(
        [
            dict(
                player_id="a",
                season=y,
                week=1,
                season_type="REG",
                attempts=20,
                carries=5,
                passing_yards=200,
                rushing_yards=40,
            )
            for y in (2018, 2019)
        ]
    )
    snaps = pl.DataFrame(
        [
            dict(player_id="a", season=y, week=1, offense_snaps=60, offense_pct=1.0)
            for y in (2018, 2019)
        ]
    )
    rules = dict(
        passing=dict(yards_per_point=25, touchdown=4, interception=-2, two_point_conversion=2),
        rushing=dict(points_per_yard=0.1, touchdown=6, two_point_conversion=2),
        misc=dict(fumble_lost=-2),
    )
    before = enrich_history([row()], raw, snaps, schedule(), rules)[0]
    changed_raw = raw.with_columns(
        pl.when(pl.col("season") == 2019)
        .then(9999)
        .otherwise(pl.col("passing_yards"))
        .alias("passing_yards")
    )
    changed_snaps = snaps.with_columns(
        pl.when(pl.col("season") == 2019)
        .then(0)
        .otherwise(pl.col("offense_snaps"))
        .alias("offense_snaps")
    )
    changed_schedule = schedule().with_columns(pl.lit("z").alias("home_qb_id"))
    after = enrich_history([row()], changed_raw, changed_snaps, changed_schedule, rules)[0]
    assert before["actual_opening_starter"] != after["actual_opening_starter"]
    assert before["actual_offensive_games"] != after["actual_offensive_games"]
    assert {k: v for k, v in before.items() if not k.startswith("actual_")} == {
        k: v for k, v in after.items() if not k.startswith("actual_")
    }


def synthetic_rows():
    rows = []
    for year in range(2014, 2021):
        for i in range(20):
            r = {f: (i + 1) / 5 for f in ROLE}
            r.update(row())
            r.update(
                player_id=str(i),
                season=year,
                ppg=i,
                games=i % 16,
                prior3_attempts=i * 15,
                prior3_carries=i * 3,
                prior3_pass_points=i * 7,
                prior3_rush_points=i * 2,
                actual_opening_starter=int(i >= 10),
                actual_season_points=i * 15,
                actual_attempts=i * 25,
                actual_carries=i * 3,
                actual_offensive_games=i % 16,
            )
            rows.append(r)
    return attach_evidence(rows, [])


def test_efficiency_shrinks_empty_samples_and_never_uses_test_outcomes():
    rows = synthetic_rows()
    train, test = rows[:100], rows[100:]
    _, before, prior = add_efficiency(train, test)
    zero = next(r for r in before if r["prior3_attempts"] == 0)
    assert zero["pass_efficiency"] == pytest.approx(prior["pass"])
    changed = [{**r, "actual_season_points": 99999} for r in test]
    _, after, prior2 = add_efficiency(train, changed)
    assert prior == prior2
    assert [r["pass_efficiency"] for r in before] == [r["pass_efficiency"] for r in after]


def test_walk_forward_job_labels_and_targets_cannot_leak_into_same_fold():
    rows = synthetic_rows()
    before, folds = predict(rows)
    changed = copy.deepcopy(rows)
    for r in changed:
        if r["season"] >= 2019:
            r.update(
                actual_opening_starter=1 - r["actual_opening_starter"],
                actual_season_points=9999,
                actual_attempts=9999,
                actual_carries=9999,
                actual_offensive_games=0,
            )
    after, _ = predict(changed)
    for left, right in zip(before, after, strict=True):
        if left["season"] == 2019:
            for key in left:
                if not key.startswith("actual_"):
                    assert left[key] == right[key]
            for spec in SPECS:
                p = left[f"{spec}_starter_probability"]
                assert 0 <= p <= 1
                expected = (
                    p * left["conditional_1_season_points"]
                    + (1 - p) * left["conditional_0_season_points"]
                )
                assert left[f"mixture_{spec}_season_points"] == pytest.approx(expected)
    assert all(f["last_train_season"] < f["season"] for f in folds)
    assert not any(f.startswith("actual_") for spec in SPECS.values() for f in spec)
    assert np.isfinite(before[0]["mixture_market_evidence_season_points"])
