"""Chronology, missingness, and population tests for weekly v3 data."""

import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v3.data import (
    STAT_COLUMNS,
    availability_lookup,
    construct_panel,
    history_summary,
    score_points,
)
from pandas.testing import assert_frame_equal


def sources():
    rows = []
    for year, week, pid, team in [
        (2020, 1, "known", "A"),
        (2020, 1, "other", "B"),
        (2021, 1, "known", "A"),
        (2021, 1, "other", "B"),
        (2021, 2, "new", "A"),
        (2021, 2, "other", "B"),
        (2021, 3, "known", "A"),
        (2021, 3, "other", "B"),
    ]:
        row = dict(
            season=year,
            week=week,
            player_id=pid,
            player_name=pid,
            player_display_name=pid,
            position="RB",
            team=team,
            **{s: 0.0 for s in STAT_COLUMNS},
        )
        row.update(carries=10.0, rushing_yards=50.0)
        rows.append(row)
    games = []
    for year in [2020, 2021]:
        for week in [1, 2, 3, 4]:
            games.append(
                dict(
                    season=year,
                    week=week,
                    game_id=f"{year}_{week}",
                    home_team="A",
                    away_team="B",
                    gameday=f"{year}-09-{week * 7:02d}",
                    home_score=10 if week < 4 else np.nan,
                    away_score=3 if week < 4 else np.nan,
                    game_type="REG",
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(games)


def panel(box, schedule):
    return construct_panel(
        box, schedule, {}, {}, start_year=2021, end_year=2021, current_season=2021, through_week=3
    )


def test_future_observation_perturbation_cannot_change_earlier_features():
    box, schedule = sources()
    original, features = panel(box, schedule)
    box.loc[(box.season == 2021) & (box.week == 3), "rushing_yards"] = 99999.0
    modified, _ = panel(box, schedule)
    cols = ["player_id", "season", "week"] + features
    assert_frame_equal(
        original.loc[original.week <= 3, cols], modified.loc[modified.week <= 3, cols]
    )
    assert (
        original.loc[(original.week == 3) & (original.player_id == "known"), "target_points"].item()
        == 5
    )
    assert modified.loc[
        (modified.week == 3) & (modified.player_id == "known"), "target_points"
    ].item() == pytest.approx(9999.9)


def test_newcomer_not_admitted_by_future_appearance_and_missing_appearance_is_zero():
    box, schedule = sources()
    f, _ = panel(box, schedule)
    assert not ((f.player_id == "new") & (f.week <= 2)).any()
    assert ((f.player_id == "new") & (f.week == 3)).any()
    missed = f[(f.player_id == "known") & (f.week == 2)].iloc[0]
    assert missed.target_points == 0 and not missed.appeared
    assert f.loc[f.week == 4, "target_points"].isna().all()


def test_uncaptured_completed_team_game_does_not_become_absence_zero():
    box, schedule = sources()
    box = box[~((box.season == 2021) & (box.week == 2) & (box.team == "A"))]
    f, _ = panel(box, schedule)
    assert np.isnan(f[(f.player_id == "known") & (f.week == 2)].iloc[0].target_points)


def test_missing_score_component_remains_missing_and_invalid_targets_masked():
    box, schedule = sources()
    ix = (box.season == 2021) & (box.week == 1) & (box.player_id == "known")
    box.loc[ix, "receiving_yards"] = np.nan
    box.loc[ix, "receptions"] = 3
    box.loc[ix, "targets"] = 0
    f, features = panel(box, schedule)
    assert np.isnan(f[(f.player_id == "known") & (f.week == 1)].iloc[0].target_points)
    assert "target_points" not in features and "appeared" not in features
    # Invalid week1 target is ignored in the finite mean, leaving prior valid zero.
    assert f[(f.player_id == "known") & (f.week == 2)].iloc[0].recent4_targets_mean == 0


def test_history_variance_is_sample_variance_and_includes_missed_games():
    rows = [
        dict(season=2021, week=i + 1, points=p, appeared=p > 0)
        for i, p in enumerate([0.0, 10.0, 20.0])
    ]
    summary = history_summary(rows, 2021, 4)
    assert summary["player_history_count"] == 3
    assert summary["player_history_variance"] == 100
    assert summary["recent4_appearance_rate"] == 2 / 3


def test_availability_conflicts_do_not_assert_status_or_team():
    stamp = pd.Timestamp("2021-09-01", tz="UTC")
    events = pd.DataFrame(
        [
            dict(player_id="a", available_at=stamp, status="ACT", team="A"),
            dict(player_id="a", available_at=stamp, status="RET", team="B"),
        ]
    )
    _, states = availability_lookup(events)["a"]
    assert states[0]["status"] == "UNKNOWN" and pd.isna(states[0]["team"])


def test_scoring_preserves_v2_contract():
    rows = [dict(position=p, **{s: 0.0 for s in STAT_COLUMNS}) for p in ["QB", "RB"]]
    rows[0].update(passing_yards=250, passing_tds=2, passing_interceptions=1, rushing_yards=20)
    rows[1].update(rushing_yards=50, receiving_yards=30, receptions=4, receiving_tds=1)
    assert score_points(pd.DataFrame(rows)).tolist() == [18.0, 18.0]


def test_future_retirement_is_not_applied_at_earlier_cutoff():
    box, schedule = sources()
    events = pd.DataFrame(
        [
            dict(
                player_id="known",
                available_at=pd.Timestamp("2021-09-17", tz="UTC"),
                status="RET",
                team="A",
            )
        ]
    )
    # Parquet-derived datetimes can have microsecond units; search must normalize.
    events["available_at"] = events.available_at.astype("datetime64[us, UTC]")
    f, _ = construct_panel(
        box,
        schedule,
        {},
        {},
        events,
        start_year=2021,
        end_year=2021,
        current_season=2021,
        through_week=3,
    )
    assert ((f.player_id == "known") & (f.week == 3)).any()
    assert not ((f.player_id == "known") & (f.week == 4)).any()


def test_injury_report_requires_known_timestamp_and_same_season():
    from experiments.future_player_lab.campaigns.weekly_v3.data import attach_injuries

    panel = pd.DataFrame(
        [
            dict(player_id="a", season=2021, feature_cutoff="2021-09-01T12:00:00+00:00"),
            dict(player_id="a", season=2021, feature_cutoff="2021-09-10T12:00:00+00:00"),
            dict(player_id="b", season=2021, feature_cutoff="2021-09-10T12:00:00+00:00"),
        ]
    )
    injuries = pd.DataFrame(
        [
            dict(
                gsis_id="a", season=2020, date_modified="2020-12-01T00:00:00Z", report_status="Out"
            ),
            dict(
                gsis_id="a",
                season=2021,
                date_modified="2021-09-05T00:00:00Z",
                report_status="Questionable",
            ),
            dict(gsis_id="b", season=2021, date_modified=None, report_status="Out"),
        ]
    )
    result = attach_injuries(panel, injuries)
    assert result.injury_report_known.tolist() == [0.0, 1.0, 0.0]
    assert result.injury_questionable.iloc[1] == 1
    assert result.injury_out.iloc[[0, 2]].isna().all()


def test_exact_v2_baseline_uses_frozen_forecast_team_schedule():
    box, schedule = sources()
    # A historical v2 forecast uses old team C, whose known bye is week 2.
    schedule = pd.concat(
        [
            schedule,
            pd.DataFrame(
                [
                    dict(
                        season=2021,
                        week=1,
                        game_id="C1",
                        home_team="C",
                        away_team="D",
                        gameday="2021-09-07",
                        home_score=1,
                        away_score=0,
                        game_type="REG",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    baseline = {(2021, 0, "known"): (10.0, "C", "RB", "known", "v2_frozen_historical_ensemble")}
    f, _ = construct_panel(
        box,
        schedule,
        {},
        baseline,
        start_year=2021,
        end_year=2021,
        current_season=2021,
        through_week=3,
    )
    assert f.loc[(f.player_id == "known") & (f.week == 1), "v2_weekly_mean"].item() == 10
    assert f.loc[(f.player_id == "known") & (f.week == 2), "v2_weekly_mean"].item() == 0


def test_historical_roster_team_aliases_are_canonicalized():
    box, schedule = sources()
    box.loc[box.team == "A", "team"] = "BAL"
    schedule.loc[schedule.home_team == "A", "home_team"] = "BAL"
    events = pd.DataFrame(
        [
            dict(
                player_id="known",
                available_at=pd.Timestamp("2021-09-01", tz="UTC"),
                status="ACT",
                team="BLT",
            )
        ]
    )
    f, _ = construct_panel(
        box,
        schedule,
        {},
        {},
        events,
        start_year=2021,
        end_year=2021,
        current_season=2021,
        through_week=3,
    )
    known = f[f.player_id == "known"]
    assert (known.team == "BAL").all() and (known.scheduled_game == 1).all()


def test_source_wide_target_defect_masks_even_plausible_rows():
    box, schedule = sources()
    box["season"] = box.season.replace({2020: 2002, 2021: 2003})
    schedule["season"] = schedule.season.replace({2020: 2002, 2021: 2003})
    schedule["gameday"] = schedule.gameday.str.replace("2020", "2002").str.replace("2021", "2003")
    # A plausible individual observation cannot repair the deficient year capture.
    box.loc[(box.season == 2003) & (box.week == 1), "targets"] = 2.0
    f, _ = construct_panel(
        box, schedule, {}, {}, start_year=2003, end_year=2003, current_season=2003, through_week=3
    )
    row = f[(f.player_id == "known") & (f.week == 2)].iloc[0]
    assert np.isnan(row.current_targets_mean)
    assert row.current_points_mean == 5.0
