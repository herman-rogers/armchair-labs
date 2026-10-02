"""Safety checks for the research-only audit and role pilot."""

import json
import math

import polars as pl
import pytest
from artifact_inputs import require_audited_version
from data_integrity_audit import clean_json, digest, duplicate_rows
from role_transition_pilot import BASE, ROLE, USAGE, fit_predict, make_rows


def test_audit_reports_duplicates_and_serializes_nonfinite():
    frame = pl.DataFrame({"player_id": ["a", "a", "b"], "season": [2024, 2024, 2024]})
    assert duplicate_rows(frame, ["player_id", "season"]) == 2
    assert clean_json({"v": [math.nan, math.inf, -math.inf, 1.0]}) == {"v": [None, None, None, 1.0]}


def test_future_results_do_not_change_features_or_eligibility():
    points = pl.DataFrame(
        [
            {
                "player_id": "a",
                "season": 2024,
                "week": week,
                "position": "RB",
                "league_points": float(week),
                "carries": week,
                "targets": 1,
            }
            for week in range(1, 10)
        ]
    )
    snaps = pl.DataFrame(
        [
            {
                "player_id": "a",
                "season": 2024,
                "week": week,
                "offense_snaps": 20,
                "offense_pct": 0.3,
                "player": "A",
            }
            for week in range(1, 10)
        ]
    )
    adp = pl.DataFrame({"gsis_id": ["a"], "forecast_season": [2024], "adp": [150.0]})
    before = next(r for r in make_rows(points, snaps, adp) if r["week"] == 3)
    changed_points = points.with_columns(
        pl.when(pl.col("week") > 3)
        .then(1000.0)
        .otherwise(pl.col("league_points"))
        .alias("league_points"),
        pl.when(pl.col("week") > 3)
        .then(pl.lit("LB"))
        .otherwise(pl.col("position"))
        .alias("position"),
    )
    changed_snaps = snaps.with_columns(
        pl.when(pl.col("week") > 3).then(0.99).otherwise(pl.col("offense_pct")).alias("offense_pct")
    )
    after = next(r for r in make_rows(changed_points, changed_snaps, adp) if r["week"] == 3)
    for feature in set(BASE + ROLE + USAGE):
        assert before[feature] == after[feature]
    assert before["next4_points"] == 4 + 5 + 6 + 7
    assert after["next4_points"] == 4000
    assert before["small_prior_sample"] == 0


def test_ridge_prediction_does_not_read_test_outcomes():
    train = [{"x": float(i), "next4_points": float(i * 2)} for i in range(20)]
    a = fit_predict(train, [{"x": 3.0, "next4_points": 0}], ["x"])
    b = fit_predict(train, [{"x": 3.0, "next4_points": 1000000}], ["x"])
    assert a.tolist() == b.tolist()


def test_quarantined_fields_cannot_enter_research_models():
    for feature in ROLE:
        assert not any(s in feature for s in ["contract", "transaction", "route", "injury", "xfp"])


@pytest.mark.parametrize("change", ["forecast", "weekly", "failed_check", "empty_checks"])
def test_research_refuses_changed_or_unaccepted_inputs(tmp_path, change):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    prediction = outputs / "metric_backtest_predictions.parquet"
    weekly = outputs / "research_audited_weekly_points.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(prediction)
    pl.DataFrame({"x": [1]}).write_parquet(weekly)
    manifest = {
        "status": "rebuilt_pending_audit",
        "protected_unchanged": True,
        "version": "test",
        "source_sha256": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    audit = {
        "accepted": True,
        "acceptance_checks": {"valid": True},
        "prediction_sha256": digest(prediction),
        "weekly_points_sha256": digest(weekly),
    }
    if change in {"forecast", "weekly"}:
        pl.DataFrame({"x": [2]}).write_parquet(prediction if change == "forecast" else weekly)
    elif change == "failed_check":
        audit["acceptance_checks"]["valid"] = False
    else:
        audit["acceptance_checks"] = {}
    (outputs / "data_integrity_audit_2026-09-22.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        require_audited_version(tmp_path)
