"""Temporal and denominator checks for the preseason role experiment."""

import copy

import polars as pl
import pytest
from preseason_role_workload import ROLE, build_rows, predict, role_week


def fixture():
    points = pl.DataFrame(
        [
            dict(
                player_id="a",
                season=year,
                week=w,
                team="BAL",
                attempts=20 if w > 2 else 1,
                carries=5,
                targets=0,
                league_points=20.0 if w > 2 else 2.0,
            )
            for year in (2018, 2019)
            for w in range(1, 7)
        ]
    )
    snaps = points.select("player_id", "season", "week", "team").with_columns(
        pl.when(pl.col("week") > 2).then(0.95).otherwise(0.1).alias("offense_pct"),
        pl.when(pl.col("week") > 2).then(60).otherwise(6).alias("offense_snaps"),
    )
    candidates = pl.DataFrame(
        [
            dict(
                player_id="a",
                forecast_season=2019,
                position="QB",
                player_population="returner",
                outcome_complete=True,
                ppg=14.0,
                games=6,
                actual_season_points=84.0,
                age_at_season=23,
                player_experience=1,
            )
        ]
    )
    schedule = pl.DataFrame(
        [
            dict(season=y, week=w, home_team="BAL", away_team="CIN", game_type="REG")
            for y in (2018, 2019)
            for w in range(1, 7)
        ]
    )
    return candidates, points, snaps, schedule


def make(inputs, identities=None, complete=None):
    return build_rows(
        *inputs,
        {"a"} if identities is None else identities,
        {2018, 2019} if complete is None else complete,
    )


def test_future_usage_cannot_change_past_role_or_eligibility():
    c, p, s, schedule = fixture()
    before = make((c, p, s, schedule))[0]
    changed_points = p.with_columns(
        pl.when(pl.col("season") == 2019).then(999).otherwise(pl.col("attempts")).alias("attempts"),
        pl.when(pl.col("season") == 2019)
        .then(pl.lit("NE"))
        .otherwise(pl.col("team"))
        .alias("team"),
    )
    changed_snaps = s.with_columns(
        pl.when(pl.col("season") == 2019)
        .then(0)
        .otherwise(pl.col("offense_pct"))
        .alias("offense_pct")
    )
    after = make((c, changed_points, changed_snaps, schedule))[0]
    assert before["actual_attempts"] != after["actual_attempts"]
    assert {k: v for k, v in before.items() if not k.startswith("actual_")} == {
        k: v for k, v in after.items() if not k.startswith("actual_")
    }
    assert before["role_games"] == 4
    assert before["role_ppg"] == 20
    assert before["other_ppg"] == 2


def test_late_window_keeps_absences_and_excludes_postseason():
    c, p, s, schedule = fixture()
    p, s = p.filter(pl.col("week") <= 4), s.filter(pl.col("week") <= 4)
    postseason = schedule.filter(pl.col("week") == 6).with_columns(
        pl.lit(7, dtype=pl.Int64).alias("week"), pl.lit("POST").alias("game_type")
    )
    row = make((c, p, s, pl.concat([schedule, postseason])))[0]
    assert row["late_weeks"] == [3, 4, 5, 6]
    assert row["late_points_pg"] == 10
    assert row["late_snap_share"] == pytest.approx(0.475)
    assert row["late_role_fraction"] == 0.5


def test_missing_identity_or_feed_is_not_known_zero_usage():
    inputs = fixture()
    for row in (make(inputs, identities=set())[0], make(inputs, complete={2019})[0]):
        assert not row["eligible"]
        assert row["role_games"] is None
        assert row["prior_snap_share"] is None
        assert row["late_snaps_pg"] is None
    row = make(inputs, complete={2018})[0]
    assert row["eligible"]  # Target coverage cannot decide feature eligibility.


def test_conflicting_week_keys_fail_closed():
    c, p, s, schedule = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        make((c, pl.concat([p, p.head(1)]), s, schedule))


def test_role_proxies_require_position_specific_opportunity():
    assert role_week("QB", {"offense_pct": 0.8}, {})
    assert not role_week("RB", {"offense_pct": 0.8}, {"carries": 1, "targets": 1})
    assert role_week("RB", {"offense_pct": 0.6}, {"carries": 10, "targets": 2})
    assert not role_week("TE", {"offense_pct": 1}, {"targets": 1})
    assert role_week("TE", {"offense_pct": 0.4}, {"targets": 4})


def test_walk_forward_predictions_ignore_held_out_outcomes():
    base = make(fixture())[0]
    rows = []
    for year in range(2014, 2021):
        for i in range(10):
            rows.append(
                {
                    **base,
                    "player_id": str(i),
                    "season": year,
                    "ppg": float(i),
                    "actual_season_points": float(i * 20),
                    "actual_attempts": float(i * 30),
                    "actual_carries": float(i * 4),
                }
            )
    before, folds = predict(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["season"] >= 2019:
            row["actual_season_points"] = 9999
            row["actual_attempts"] = 9999
    after, _ = predict(changed)
    for left, right in zip(before, after, strict=True):
        if left["season"] == 2019:
            for key in left:
                if not key.startswith("actual_"):
                    assert left[key] == right[key]
    assert all(f["last_train_season"] < f["season"] for f in folds)
    assert not any(f.startswith("actual_") for f in ROLE)
