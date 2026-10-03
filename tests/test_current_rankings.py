import numpy as np
import polars as pl

from engine.metrics.current_rankings import (
    BASELINES,
    CHALLENGERS,
    apply_constraints,
    choose,
    fit_candidates,
    make_panel,
    metrics,
    rank_frame,
)
from engine.metrics.nextgen import COUNTERS


def panel_inputs():
    features = pl.DataFrame(
        [
            dict(
                player_id="a",
                player_display_name="A",
                position="QB",
                player_population="returner",
                forecast_season=2005,
                forecast_cutoff_date="2005-09-01",
                cutoff_preseason_team="TB",
            )
        ]
    )
    rows = []
    for year, week, pid, points in [
        (2004, 1, "a", 15),
        (2005, 1, "a", 20),
        (2005, 2, "b", 12),
        (2005, 3, "a", 30),
    ]:
        row = {c: 0.0 for c in COUNTERS}
        row.update(
            player_id=pid,
            player_display_name=pid.upper(),
            season=year,
            week=week,
            position="QB",
            team="TB",
            league_points=points,
            snap_share=1.0,
        )
        rows.append(row)
    schedule = pl.DataFrame(
        [
            dict(season=2005, week=w, home_team="TB", away_team="CAR", game_type="REG")
            for w in [1, 2, 4, 5, 6, 7]
        ]
    )
    people = pl.DataFrame(
        [
            dict(
                gsis_id="a",
                birth_date="1980-01-01",
                draft_year=2004,
                draft_pick=10,
                rookie_season=2004,
            )
        ]
    )
    college = pl.DataFrame(schema={"college_id": pl.String})
    links = pl.DataFrame(
        schema={"college_id": pl.String, "player_id": pl.String, "status": pl.String}
    )
    return features, pl.DataFrame(rows), schedule, people, college, links


def test_future_observations_cannot_admit_candidates_or_change_predictors():
    inputs = panel_inputs()
    original = make_panel(*inputs, season=2006, through_week=2)
    changed = inputs[1].to_dicts()
    changed[-1]["league_points"] = 900
    changed[-1]["team"] = "NYJ"
    changed.append(dict(changed[-1], player_id="future", player_display_name="Future"))
    revised = make_panel(inputs[0], pl.DataFrame(changed), *inputs[2:], season=2006, through_week=2)
    assert {r["player_id"] for r in original} == {"a", "b"}
    assert {r["player_id"] for r in revised} == {"a", "b"}
    assert [{k: v for k, v in r.items() if k != "actual"} for r in original] == [
        {k: v for k, v in r.items() if k != "actual"} for r in revised
    ]
    assert any(a["actual"] != b["actual"] for a, b in zip(original, revised, strict=True))


def test_schedule_counts_byes_and_pending_outcomes_remain_unknown():
    rows = make_panel(*panel_inputs(), season=2005, through_week=2)
    row = next(
        r for r in rows if r["player_id"] == "a" and r["horizon"] == "next4" and r["season"] == 2005
    )
    assert row["scheduled_games"] == 3
    assert row["actual"] is None
    assert row["current_weeks"] == 1
    assert row["current_rate"] == 10  # Includes the elapsed week with no recorded production.


def test_role_uses_observed_opportunity_not_prior_sample_size():
    inputs = panel_inputs()
    weeks = inputs[1].with_columns(
        pl.when((pl.col("season") == 2005) & (pl.col("player_id") == "a"))
        .then(30.0)
        .otherwise(0.0)
        .alias("attempts")
    )
    rows = make_panel(inputs[0], weeks, *inputs[2:], season=2005, through_week=2)
    starter = next(r for r in rows if r["player_id"] == "a" and r["season"] == 2005)
    backup = next(r for r in rows if r["player_id"] == "b")
    assert starter["prior_weeks"] == 1
    assert starter["role_group"] == "observed_workload"
    assert backup["role_group"] == "limited_workload"


def test_training_and_selection_do_not_use_test_labels():
    train = [
        dict(
            season=2004 + i % 3,
            population="returner",
            prior_rate=i % 20,
            current_rate=i % 21,
            elapsed_games=2,
            scheduled_games=4,
            actual=i % 20 * 3.0,
            x_prior=i % 20,
            x_current=i % 21,
            e_extra=None,
        )
        for i in range(90)
    ]
    test = [dict(train[0], actual=1000), dict(train[3], actual=-500)]
    a = fit_candidates(train, test)
    b = fit_candidates(train, [dict(r, actual=0) for r in test])
    assert all(np.allclose(a[m], b[m]) for m in a)
    assert set(a) == set(BASELINES + CHALLENGERS)
    assert choose([], "QB") == ("blend", "blend")


def test_selection_rejects_lower_error_with_worse_top_k_capture():
    rows = []
    for year in range(2007, 2011):
        for i in range(30):
            actual = 100 if i < 10 else 10
            r = dict(player_id=str(i), season=year, actual=actual)
            r.update({m: actual + 50 for m in BASELINES})
            r.update({m: 40 if i < 10 else 41 for m in CHALLENGERS})
            rows.append(r)
    reference, selected = choose(rows, "QB")
    assert selected == reference


def test_current_constraints_are_dated_and_do_not_invent_injury_durations():
    row = dict(
        player_id="a",
        season=2026,
        through_week=2,
        prediction=200,
        position="QB",
        horizon="rest_of_season",
    )
    news = dict(
        player_id="a",
        season=2026,
        known_on="2026-09-23",
        season_ending_reported=True,
        affected_period="remaining_2026_regular_season",
        earliest_affected_week=3,
        summary="Reported season-ending surgery",
        source_url="https://example.test",
    )
    assert apply_constraints([row], [news], "2026-09-22")[0]["prediction"] == 200
    constrained = apply_constraints([row], [news], "2026-09-23")[0]
    assert constrained["prediction"] == 0 and not constrained["rank_eligible"]
    assert constrained["unconstrained_prediction"] == 200
    ordinary = dict(news, season_ending_reported=False, injury_status="OUT")
    assert apply_constraints([row], [ordinary], "2026-09-23")[0]["rank_eligible"]
    assert row["prediction"] == 200


def test_ranks_preserve_ties_exclude_suspended_and_survive_filtering():
    rows = [
        dict(
            player_id=str(i),
            position=pos,
            horizon="rest_of_season",
            prediction=p,
            rank_eligible=eligible,
        )
        for i, (pos, p, eligible) in enumerate(
            [("QB", 200, True), ("QB", 200, True), ("RB", 150, True), ("QB", 0, False)]
        )
    ]
    frame = rank_frame(pl.DataFrame(rows))
    assert frame["position_rank"].to_list() == [1, 1, 1, None]
    assert frame.filter(pl.col("position") == "RB")["overall_rank"].to_list() == [3]
    assert (
        metrics([dict(player_id="a", actual=10, p=1), dict(player_id="b", actual=20, p=2)], "p", 1)[
            "capture"
        ]
        == 1
    )
