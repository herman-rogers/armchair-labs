"""Independent fixed examples for cutoff, revision, and no-double-discount rules."""

from datetime import UTC, date, datetime

import polars as pl
from experiments.future_player_lab.notebooks.rb_availability import (
    adjust_forecasts,
    annotate,
    selected_events,
)


def event(**changes):
    return {
        "player_id": "rb",
        "season": 2025,
        "absence_id": "injury",
        "evidence_id": "first",
        "source_published_on": "2025-08-01",
        "known_on": "2025-08-01",
        "unavailable_games": list(range(1, 18)),
        **changes,
    }


def inputs(origin=0, end=18):
    panel = pl.DataFrame(
        [
            {
                "player_id": "rb",
                "year": 2025,
                "origin": origin,
                "end": end,
                "forecast_cutoff_date": date(2025, 8, 29),
            }
        ]
    )
    schedule = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": w,
                "game_type": "REG",
                "gameday": date(2025, 9, 1).fromordinal(date(2025, 9, 1).toordinal() + w * 7),
                "home_team": "A",
                "away_team": "B",
            }
            for w in range(1, 19)
            if w != 5
        ]
    )
    preseason = pl.DataFrame(
        [
            {
                "player_id": "rb",
                "forecast_season": 2025,
                "forecast_cutoff_date": date(2025, 8, 29),
                "cutoff_preseason_team": "A",
                "cutoff_state_observed": True,
            }
        ]
    )
    return panel, schedule, preseason


def test_future_evidence_cannot_change_past_and_uncertain_revision_clears_rule():
    e = event()
    later = event(
        evidence_id="later",
        known_on="2025-09-01",
        source_published_on="2025-09-01",
        certainty="uncertain",
    )
    assert selected_events([e, later], "rb", 2025, "2025-08-29") == ([e], [])
    assert selected_events([e, later], "rb", 2025, "2025-09-02") == ([], [])


def test_same_day_conflicts_and_input_order_never_force_zero():
    panel, schedule, preseason = inputs()
    a, b = event(), event(evidence_id="conflict", unavailable_games=[1, 2])
    first = annotate(panel, [a, b], schedule, preseason)
    assert first.equals(annotate(panel, [b, a], schedule, preseason))
    assert not first["availability_force_zero"][0]


def test_partial_absence_does_not_scale_totals_and_actuals_stay_unchanged():
    panel, schedule, preseason = inputs()
    forecast = pl.DataFrame(
        {"actual": [5.0], "prediction": [150.0], "persistence_prediction": [170.0]}
    )
    annotations = annotate(panel, [event(unavailable_games=[1, 2, 3, 4])], schedule, preseason)
    assert not annotations["availability_force_zero"][0]
    partial = adjust_forecasts(forecast, annotations)
    assert partial["prediction"][0] == 150
    full = adjust_forecasts(forecast, annotate(panel, [event()], schedule, preseason))
    assert full["prediction"][0] == full["persistence_prediction"][0] == 0
    assert full["raw_prediction"][0] == 150
    assert full["raw_persistence_prediction"][0] == 170
    assert full["actual"].equals(forecast["actual"])


def test_calendar_week_absence_only_zeros_matching_period():
    panel, schedule, preseason = inputs(origin=2, end=3)
    e = event(unavailable_games=[], unavailable_weeks=[3])
    assert annotate(panel, [e], schedule, preseason)["availability_force_zero"][0]
    longer = panel.with_columns(pl.lit(6).alias("end"))
    assert not annotate(longer, [e], schedule, preseason)["availability_force_zero"][0]


def test_missing_report_timestamp_is_unknown_and_future_reports_do_not_leak():
    panel, schedule, preseason = inputs(origin=2)
    base = {
        "gsis_id": "rb",
        "season": 2025,
        "week": 2,
        "game_type": "REG",
        "report_status": "Out",
        "practice_status": "Did Not Participate",
    }
    rows = [
        dict(base, date_modified=None),
        dict(base, date_modified=datetime(2025, 12, 1, tzinfo=UTC)),
    ]
    a = annotate(panel, [], schedule, preseason, pl.DataFrame(rows))
    assert a["availability_dated_reports"][0] == 0
    assert a["availability_latest_out"][0] is None
    assert a["availability_known_absence_games"][0] is None
    assert not a["availability_force_zero"][0]


def test_partial_game_ordinals_do_not_assume_current_team():
    panel, schedule, preseason = inputs(origin=2, end=3)
    assert not annotate(panel, [event(unavailable_games=[3])], schedule, preseason)[
        "availability_force_zero"
    ][0]


def test_uncertain_absence_is_an_input_but_never_a_zero_override():
    panel, schedule, preseason = inputs()
    a = annotate(panel, [event(certainty="uncertain")], schedule, preseason)
    assert a["availability_uncertain_absence"][0] == 1.0
    assert a["availability_known_absence_games"][0] is None
    assert not a["availability_force_zero"][0]


def test_status_corroboration_does_not_invent_practice_participation():
    panel, schedule, preseason = inputs(origin=2)
    reports = pl.DataFrame(
        [
            dict(
                gsis_id="rb",
                season=2025,
                week=2,
                game_type="REG",
                date_modified=datetime(2025, 9, 15, tzinfo=UTC),
                report_status="Out",
                practice_status=None,
            )
        ]
    )
    a = annotate(panel, [], schedule, preseason, reports)
    assert a["availability_latest_out"][0] == 1
    assert a["availability_latest_dnp"][0] is None


def test_annotation_order_is_checked_before_adjustment():
    import pytest

    panel, schedule, preseason = inputs()
    panel = panel.with_columns(pl.lit("season").alias("horizon"))
    a = annotate(panel, [event()], schedule, preseason)
    forecast = panel.with_columns(
        pl.lit("different-player").alias("player_id"),
        pl.lit(1.0).alias("prediction"),
        pl.lit(2.0).alias("persistence_prediction"),
    )
    with pytest.raises(ValueError, match="identities/order"):
        adjust_forecasts(forecast, a)


def test_recovery_matches_exact_injury_status_and_uses_last_update():
    # Import CLI module without requiring installation of research as a package.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
    from recover_injury_dates import corroborate

    html = """<script>{"datePublished":"2025-09-17T18:00:00Z",
    "dateModified":"2025-09-20T19:00:00Z"}</script>
    <li><strong>OUT:</strong> RB Will Shipley (oblique), WR Other Player (ankle)</li>"""
    row = dict(
        record_id="original",
        season=2025,
        week=3,
        game_type="REG",
        gsis_id="shipley",
        position="RB",
        full_name="Will Shipley",
        team="PHI",
        report_primary_injury="Oblique",
        report_status="Out",
        date_modified=None,
    )
    reports = pl.DataFrame([row])
    schedule = pl.DataFrame(
        [
            dict(season=2025, week=3, game_type="REG", gameday="2025-09-18"),
            dict(season=2025, week=3, game_type="REG", gameday="2025-09-22"),
        ]
    )
    article = dict(
        title="NFL Week 3 injury report",
        season=2025,
        sha256="a" * 64,
        url="https://www.nfl.com/news/example",
        capture_file="capture.gz",
        retrieved_at="2026-09-25",
    )
    found = corroborate(article, html, reports, schedule)
    assert len(found) == 1
    assert found[0]["date_modified"].isoformat().startswith("2025-09-20")
    assert found[0]["practice_status"] is None
    assert reports["date_modified"][0] is None
    assert not corroborate(article, html.replace("oblique", "ankle"), reports, schedule)
    assert not corroborate(article, html.replace("OUT:", "QUESTIONABLE:"), reports, schedule)
    assert not corroborate(article, html.replace("2025-09-20", "2026-09-20"), reports, schedule)


def test_notebook_training_wiring_without_fitting_a_model():
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    from experiments.future_player_lab.notebooks import rb_availability as availability
    from experiments.future_player_lab.notebooks import rb_probability as probability

    path = (
        Path(__file__).resolve().parents[1] / "experiments/future_player_lab/notebooks/rb_stats.py"
    )
    module = ast.parse(path.read_text())
    cell = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef)
        and any(
            isinstance(child, ast.FunctionDef) and child.name == "train_stat" for child in n.body
        )
    )
    cell.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), str(path), "exec"), namespace)
    keys = ["player_id", "position", "year", "origin", "horizon"]
    panel = pl.DataFrame(
        [
            dict(
                player_id="rb",
                position="RB",
                year=y,
                origin=0,
                horizon="season",
                name="Example RB",
                population="veteran",
                actual=0.0,
                persistence_prediction=50.0,
                exposure=18.0,
                availability_force_zero=y == 2025,
                availability_row_key=f"rb|{y}|0|season",
            )
            for y in [2022, 2023, 2024, 2025]
        ]
    )

    def metrics(actual, pred):
        err = pred - actual
        return dict(
            rmse=float(np.sqrt(np.mean(err**2))),
            mae=float(np.mean(np.abs(err))),
            bias=float(np.mean(err)),
            rows=len(err),
        )

    def fake_fit(x, y, c, rounds):
        return None, np.array([0, 1])

    def fake_predict(model, x, c, counts, target):
        return {n: np.ones(len(x)) for n in counts}

    train_stat = namespace["_"](
        availability,
        SimpleNamespace(KEYS=keys, stat_panel=lambda *args: args[0]),
        metrics,
        fake_fit,
        np,
        pl,
        fake_predict,
        probability,
    )[0]
    result = train_stat(
        panel,
        np.zeros((4, 2)),
        ["one", "two"],
        None,
        "carries",
        dict(year=2025, rounds=20, objective="rmse", learning_curve=False),
    )
    assert result["selected"] == 10
    assert result["forecasts"]["actual"][0] == 0
    assert result["forecasts"]["raw_prediction"][0] == 18
    assert result["forecasts"]["prediction"][0] == 0
    assert result["scores"].height == 4
    assert result["capacity"].filter(pl.col("split") == "Test")["rmse"].to_list() == [0, 0]
    assert result["probability"]["calibration"]["year"].to_list() == [2023]
    assert result["probability"]["calibration"]["fit_last_year"].to_list() == [2022]
    changed = panel.with_columns(
        pl.when(pl.col("year") == 2025).then(999.0).otherwise(pl.col("actual")).alias("actual")
    )
    replay = train_stat(
        changed,
        np.zeros((4, 2)),
        ["one", "two"],
        None,
        "carries",
        dict(year=2025, rounds=20, objective="crps", learning_curve=False),
    )
    assert replay["selected"] == result["selected"]
    for name, draws in result["probability"]["samples"].items():
        assert np.array_equal(draws, replay["probability"]["samples"][name])

    # A concrete case where choosing by CRPS should differ from choosing by RMSE.
    def tradeoff_predictions(model, x, c, counts, target):
        return {
            n: np.where(x[:, 0] == 2023, 100 / 18 if n == 20 else 0, 1 if n == 20 else 0)
            for n in counts
        }

    tradeoff_train = namespace["_"](
        availability,
        SimpleNamespace(KEYS=keys, stat_panel=lambda *args: args[0]),
        metrics,
        fake_fit,
        np,
        pl,
        tradeoff_predictions,
        probability,
    )[0]
    tradeoff_panel = panel.with_columns(
        pl.when(pl.col("year") == 2023).then(100.0).otherwise(pl.col("actual")).alias("actual")
    )
    tradeoff_x = np.column_stack([panel["year"].to_numpy(), np.zeros(4)])
    rmse = tradeoff_train(
        tradeoff_panel,
        tradeoff_x,
        ["one", "two"],
        None,
        "carries",
        dict(year=2025, rounds=20, objective="rmse", learning_curve=False),
    )
    crps = tradeoff_train(
        tradeoff_panel,
        tradeoff_x,
        ["one", "two"],
        None,
        "carries",
        dict(year=2025, rounds=20, objective="crps", learning_curve=False),
    )
    assert rmse["selected"] == 10
    assert crps["selected"] == 20
