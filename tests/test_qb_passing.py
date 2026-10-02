from datetime import date

import polars as pl
import pytest

from patron.metrics.qb_passing import (
    build_panel,
    choose_recipe,
    compile_events,
    constrain_totals,
    execution_evidence,
    news_known_at_issue,
    summarize,
)


def test_short_season_does_not_erase_career_execution():
    career = [dict(season=2007, attempts=578, passing_yards=4806)]
    before = execution_evidence(career, 2009)
    after = execution_evidence(career + [dict(season=2008, attempts=11, passing_yards=76)], 2009)
    assert after["evidence_attempts"] == 589
    assert abs(after["execution_ypa"] - before["execution_ypa"]) < 0.05
    assert after["execution_ypa"] > 7.9
    zero = execution_evidence(career + [dict(season=2008, attempts=0, passing_yards=0)], 2009)
    assert zero == before


def test_partial_absence_does_not_discount_an_unconditional_forecast_twice():
    values = constrain_totals(
        [100, 100, 100, -10], [dict(allowed_fraction=f) for f in [1, 0.5, 0, 1]]
    )
    assert values.tolist() == [100, 100, 0, 0]


def test_same_day_news_updates_require_capture_before_the_forecast_issue():
    issue = "2026-09-24T17:00:00+00:00"
    note = dict(known_on="2026-09-24", retrieved_at="2026-09-24T16:00:00Z")
    assert news_known_at_issue(note, issue)
    assert not news_known_at_issue({**note, "retrieved_at": "2026-09-24T18:00:00Z"}, issue)
    assert not news_known_at_issue(dict(known_on="2026-09-24"), issue)
    assert news_known_at_issue(dict(known_on="2026-09-23"), issue)


def mini_panel(*, future_yards=200, events=(), future_position="QB"):
    features = pl.DataFrame(
        [
            dict(
                player_id="a",
                player_display_name="Test QB",
                position="QB",
                forecast_season=y,
                forecast_cutoff_date=f"{y}-08-25",
                player_population="returner",
                cutoff_preseason_team="NE",
                cutoff_preseason_status="unknown",
                cutoff_state_resolution="unknown",
            )
            for y in [2004, 2005]
        ]
    )
    rows = [
        dict(
            player_id="a",
            player_display_name="Test QB",
            position="QB",
            season_type="REG",
            season=y,
            week=w,
            team="NE",
            attempts=a,
            passing_yards=v,
        )
        for y, w, a, v in [
            (2003, 1, 30, 240),
            (2004, 1, 1, 8),
            (2004, 2, 30, future_yards),
            (2005, 1, 30, 200),
        ]
    ]
    for row in rows:
        if row["season"] == 2004 and row["week"] == 2:
            row["position"] = future_position
    schedule = pl.DataFrame(
        [
            dict(
                game_type="REG",
                season=y,
                week=w,
                gameday=f"{y}-09-{7 * w:02d}",
                home_team="NE",
                away_team="NYJ",
            )
            for y in [2003, 2004, 2005]
            for w in [1, 2]
        ]
    )
    identities = pl.DataFrame(
        [
            dict(
                gsis_id="a",
                display_name="Test QB",
                draft_year=2002,
                draft_pick=100,
                birth_date="1980-01-01",
            )
        ]
    )
    return build_panel(
        features,
        pl.DataFrame(rows),
        schedule,
        identities,
        events,
        season=2005,
        through_week=1,
        current_issue_date="2005-09-08",
    )


def test_future_outcome_cannot_change_features_or_earlier_candidate_pool():
    before, after = mini_panel(), mini_panel(future_yards=1000)
    a = next(
        r
        for r in before
        if r["season"] == 2004 and r["through_week"] == 1 and r["horizon"] == "next_game"
    )
    b = next(
        r
        for r in after
        if r["season"] == 2004 and r["through_week"] == 1 and r["horizon"] == "next_game"
    )
    assert a["actual"] == 200 and b["actual"] == 1000
    assert {k: v for k, v in a.items() if not k.startswith("actual")} == {
        k: v for k, v in b.items() if not k.startswith("actual")
    }
    assert a["current_attempts"] == 1
    assert a["execution_ypa"] > 7
    assert all(r["actual"] is None for r in before if r["season"] == 2005)


def test_dated_retirement_changes_opportunity_after_known_date_not_execution():
    event = dict(
        player_id="a",
        known_on="2004-08-20",
        status="off",
        action="retire",
        source_url="https://example.test/event",
        team=None,
        resolution="observed",
    )
    base, constrained = mini_panel(), mini_panel(events=[event])
    before = next(r for r in constrained if r["season"] == 2004 and r["through_week"] == 0)
    after = next(r for r in constrained if r["season"] == 2004 and r["through_week"] == 1)
    assert before["retired_known"] and before["allowed_fraction"] == 0
    assert (
        before["execution_ypa"]
        == next(r for r in base if r["season"] == 2004 and r["through_week"] == 0)["execution_ypa"]
    )
    assert not after["retired_known"]  # Subsequent actual participation supersedes retirement.
    event["known_on"] = "2004-09-08"
    rows = mini_panel(events=[event])
    current = next(r for r in rows if r["season"] == 2005 and r["through_week"] == 0)
    # A later observed game cancels the stale retirement constraint.
    assert current["allowed_fraction"] == 1
    assert (
        after["execution_ypa"]
        == next(r for r in base if r["season"] == 2004 and r["through_week"] == 1)["execution_ypa"]
    )


def test_passing_outcomes_survive_a_change_to_te_or_wr():
    rows = mini_panel(future_position="TE")
    r = next(
        r
        for r in rows
        if r["season"] == 2004 and r["through_week"] == 1 and r["horizon"] == "next_game"
    )
    assert r["actual"] == 200
    assert r["actual_attempts"] == 30


def test_official_event_parser_excludes_bad_dates_and_preserves_return():
    candidates = pl.DataFrame(dict(player_id=["a"], player_display_name=["Test Quarterback"]))
    identities = pl.DataFrame(dict(gsis_id=["a"], display_name=["Test Quarterback"]))
    transactions = pl.DataFrame(
        [
            dict(
                transaction_date=date(2020, 1, day),
                transaction_year=2020,
                category=cat,
                description=desc,
                source_url="https://example.test/transactions",
                source_team="NE",
                from_team=None,
                to_team=None,
            )
            for day, cat, desc in [
                (1, "espn", "Test Quarterback retired."),
                (2, "official", "QB Test Quarterback retired."),
                (3, "official", "Signed QB Test Quarterback."),
            ]
        ]
    )
    events = compile_events(transactions, identities, candidates)
    assert [r["action"] for r in events] == ["retire", "sign"]


def test_nested_selection_and_equal_season_loss():
    rows = [
        dict(season=y, horizon="next_game", through_week=1, actual=100.0, a=a, b=b)
        for y, a, b in [(2010, 0.0, 100.0), (2011, 0.0, 100.0), (2012, 0.0, 100.0)]
    ]
    assert choose_recipe(rows[:2], "next_game", "weekly", ("a", "b"), "a") == "a"
    assert choose_recipe(rows, "next_game", "weekly", ("a", "b"), "a") == "b"
    result = summarize(rows + rows[-1:], "b", "a")
    assert result["rmse"] == 0
    assert result["mse_gain"] == pytest.approx(10000)
