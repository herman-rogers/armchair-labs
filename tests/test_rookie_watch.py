from copy import deepcopy

import polars as pl
import pytest

from engine.metrics.rookies import analog_forecast, backtest, completed_cutoff, rookie_rows


def inputs():
    rookies = pl.DataFrame(
        {
            "player_id": ["a", "b"],
            "player_display_name": ["Alpha", "Unobserved"],
            "season": [2026, 2026],
            "position": ["WR", "WR"],
            "draft_pick": [10, None],
        }
    )
    weeks = pl.DataFrame(
        {
            "player_id": ["a"] * 3,
            "season": [2026] * 3,
            "week": [1, 2, 3],
            "league_points": [10.0, 20.0, 900.0],
            "targets": [4, 6, 50],
            "carries": [0, 0, 10],
            "receptions": [2, 4, 50],
        }
    )
    snaps = pl.DataFrame(
        {
            "player_id": ["a"] * 3,
            "season": [2026] * 3,
            "week": [1, 2, 3],
            "offense_snaps": [30.0, 40.0, 80.0],
            "offense_pct": [0.5, 0.6, 1.0],
        }
    )
    return rookies, weeks, snaps


def test_future_production_and_snaps_cannot_change_current_features():
    rookies, weeks, snaps = inputs()
    before = rookie_rows(rookies, weeks, snaps, 2, completed_seasons=set())
    changed = weeks.with_columns(
        pl.when(pl.col("week") > 2)
        .then(1e8)
        .otherwise(pl.col("league_points"))
        .alias("league_points")
    )
    after = rookie_rows(
        rookies, changed, snaps.filter(pl.col("week") <= 2), 2, completed_seasons=set()
    )
    assert before == after
    assert before[0]["points_to_date"] == 30
    assert before[0]["targets"] == 10
    assert before[0]["next4_actual"] is None
    assert before[1]["snap_share"] is None
    assert not before[1]["eligible"]
    assert analog_forecast(before[1], [])["forecast_next4"] is None


def test_completed_outcomes_include_zero_weeks_without_calling_them_games():
    rookies, weeks, snaps = inputs()
    rows = rookie_rows(rookies, weeks, snaps, 2, completed_seasons={2026})
    assert rows[0]["next4_actual"] == 900
    assert rows[0]["observed_stat_weeks"] == 2
    assert rows[1]["next4_actual"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        rookie_rows(rookies, pl.concat([weeks, weeks]), snaps, 2, completed_seasons=set())


def test_zero_stat_line_without_offensive_snaps_is_not_forecast_evidence():
    rookies, weeks, snaps = inputs()
    weeks = weeks.with_columns(
        pl.lit(0).alias(f)
        for f in (
            "league_points",
            "targets",
            "carries",
            "receptions",
        )
    )
    rows = rookie_rows(rookies, weeks, snaps.head(0), 2, completed_seasons=set())
    assert rows[0]["observed_stat_weeks"] == 2
    assert not rows[0]["eligible"]
    assert analog_forecast(rows[0], [])["forecast_next4"] is None


def test_cutoff_requires_finished_games_and_all_teams_stats():
    schedule = pl.DataFrame(
        {
            "season": [2026] * 3,
            "game_type": ["REG"] * 3,
            "week": [1, 2, 3],
            "home_team": ["A"] * 3,
            "away_team": ["B"] * 3,
            "home_score": [10.0, 20.0, None],
            "away_score": [12.0, 13.0, None],
        }
    )
    weeks = pl.DataFrame({"season": [2026] * 4, "week": [1, 1, 2, 2], "team": ["A", "B", "A", "B"]})
    assert completed_cutoff(schedule, weeks, 2026) == 2
    assert completed_cutoff(schedule, weeks.head(3), 2026) == 1
    assert completed_cutoff(schedule, weeks.tail(2), 2026) == 0


def history_rows():
    rookie, weeks, snaps = inputs()
    row = rookie_rows(rookie, weeks, snaps, 2, completed_seasons=set())[0]
    history = [
        {
            **row,
            "player_id": str(i),
            "season": 2013 + i % 10,
            "points_per_week": float(i),
            "next4_actual": float(i * 2),
        }
        for i in range(40)
    ]
    return row, history


def test_neighbors_ignore_current_future_seasons_other_positions_and_cutoffs():
    row, history = history_rows()
    expected = analog_forecast(row, history)
    contaminants = [
        {**history[0], "season": 2026, "next4_actual": 1e9},
        {**history[0], "season": 2027, "next4_actual": 1e9},
        {**history[0], "position": "RB", "next4_actual": 1e9},
        {**history[0], "cutoff_week": 3, "next4_actual": 1e9},
        {**history[0], "next4_actual": None},
    ]
    assert analog_forecast(row, history + contaminants) == expected
    assert expected["neighbor_count"] == 25
    assert expected["history_count"] == 40
    assert expected["forecast_next4"] is not None
    assert analog_forecast({**row, "next4_actual": 1e9}, history) == expected
    assert analog_forecast(row, history[:5])["forecast_next4"] is None


def test_missing_snaps_stay_unknown_and_forecast_is_finite():
    row, history = history_rows()
    missing = {**row, "snap_share": None}
    value = analog_forecast(missing, history)["forecast_next4"]
    assert value is not None and 0 <= value <= 80
    assert missing["snap_share"] is None


def test_backtest_reports_identical_scored_pool_and_unscored_coverage():
    row, history = history_rows()
    result = backtest([*history, {**row, "next4_actual": 60}], first_season=2026)
    position = result["positions"][0]
    prediction = analog_forecast(row, history)["forecast_next4"]
    assert position["scored"] == position["eligible"] == 1
    assert position["forecast_next4_mae"] == pytest.approx(abs(prediction - 60))
    assert position["pace_next4_mae"] == 0
    # An unscored early season stays in eligibility coverage rather than disappearing.
    early = deepcopy(row)
    early.update(season=2000, next4_actual=20)
    result = backtest([early, *history], first_season=2000)["positions"][0]
    assert result["eligible"] > result["scored"]
