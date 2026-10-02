import copy

import polars as pl
import pytest

from patron.data.player_profiles import prepare_weeks
from patron.metrics.player_profile import (
    career_features,
    college_history,
    nfl_seasons,
    role_periods,
    summarize,
)


def observation(season=2024, week=1, points=10.0, share=0.8, **extra):
    return {
        "player_id": "p1",
        "season": season,
        "week": week,
        "team": "A",
        "position": "WR",
        "stat_recorded": True,
        "league_points": points,
        "offense_snaps": 40.0,
        "snap_share": share,
        "targets": 5.0,
        "carries": 0.0,
        "receiving_yards": 50.0,
        **extra,
    }


def test_full_career_and_recent_window_are_distinct_and_exposure_weighted():
    rows = [
        observation(2015, points=5),
        observation(2023, points=20),
        observation(2024, points=10),
        observation(2024, week=2, points=30),
        observation(2025, points=50),
    ]
    features = career_features(rows, 2025, 1)
    assert features["completed_seasons_observed"] == 3
    assert features["career_points_per_observed_week"] == 65 / 4
    assert features["recent3_points_per_observed_week"] == pytest.approx(
        (20 * 0.55 + 40) / (2 + 0.55)
    )
    assert features["recent3_observed_weeks"] == 3


def test_future_weeks_and_seasons_cannot_change_earlier_profile_features_or_periods():
    rows = [observation(2023), observation(2024), observation(2024, week=2), observation(2025)]
    changed = copy.deepcopy(rows)
    changed[-1].update(league_points=10000, snap_share=0.1, team="FUTURE")
    changed[-2].update(league_points=10000, snap_share=0.1, team="FUTURE")
    for fn in [career_features, nfl_seasons, role_periods]:
        assert fn(rows, 2024, 1) == fn(changed, 2024, 1)


def test_participation_context_separates_backup_usage_and_breaks_gaps_and_team_changes():
    rows = [
        observation(week=1, points=2, share=0.1),
        observation(week=2, points=3, share=0.1),
        observation(week=4, points=20),
        observation(week=5, points=22, team="B"),
    ]
    features = career_features(rows, 2025)
    assert features["limited_points_per_observed_week"] == 2.5
    assert features["high_points_per_observed_week"] == 21
    periods = role_periods(rows, 2025, 0)
    assert [(r["start_week"], r["end_week"], r["team"]) for r in periods] == [
        (1, 2, "A"),
        (4, 4, "A"),
        (5, 5, "B"),
    ]


def test_missing_is_not_zero_and_efficiency_uses_matched_denominators():
    summary = summarize(
        [
            observation(share=None, offense_snaps=None, targets=None),
            observation(week=2, receiving_yards=None, share=None, offense_snaps=None),
        ]
    )
    assert summary["offensive_weeks"] is None
    assert summary["snap_share"] is None
    assert summary["targets_per_observed_week"] is None
    assert summary["receiving_yards_per_target"] is None
    assert summarize([])["league_points"] is None


def test_college_transfers_are_kept_and_future_seasons_excluded():
    rows = [
        {"season": 2023, "team_id": "A"},
        {"season": 2023, "team_id": "B"},
        {"season": 2024, "team_id": "C"},
    ]
    assert college_history(rows, 2024) == rows[:2]


def test_normalization_preserves_audited_points_and_rejects_duplicate_joins():
    points = pl.DataFrame([observation(league_points=12, bonus_pts=2)]).drop(
        "snap_share", "offense_snaps", "stat_recorded"
    )
    stats = points.with_columns(pl.lit(999).alias("league_points"), pl.lit(11).alias("attempts"))
    snaps = points.select("player_id", "season", "week").with_columns(
        pl.lit(40).alias("offense_snaps"), pl.lit(0.8).alias("offense_pct")
    )
    weeks = prepare_weeks(points, stats, snaps)
    assert weeks["league_points"][0] == 12
    assert weeks["attempts"][0] == 11
    assert weeks["snap_share"][0] == 0.8
    with pytest.raises(ValueError, match="Conflicting profile input keys"):
        prepare_weeks(points, pl.concat([stats, stats]), snaps)
    with pytest.raises(ValueError, match="Invalid offensive snap"):
        prepare_weeks(points, stats, snaps.with_columns(pl.lit(1.2).alias("offense_pct")))


def test_snap_only_weeks_count_without_fabricating_stat_volume():
    points = pl.DataFrame([observation(week=1)]).drop(
        "snap_share", "offense_snaps", "stat_recorded"
    )
    snaps = pl.DataFrame(
        {
            "player_id": ["p1"],
            "season": [2024],
            "week": [2],
            "offense_snaps": [10.0],
            "offense_pct": [0.2],
        }
    )
    weeks = prepare_weeks(points, points, snaps)
    summary = summarize(weeks.to_dicts())
    assert summary["observed_weeks"] == 2
    assert summary["league_points"] == 10
    assert summary["targets_per_observed_week"] is None
