"""Regression cases from the September stat-system review."""

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from patron.artifacts import artifact_status, publish_draft, publish_provenance, verify_draft
from patron.espn.reports import wire_lineup_improvements, wire_replacement_levels
from patron.metrics.backtest import _overall_rows, _rank_fold
from patron.metrics.fit import (
    FitConfig,
    ModelSpec,
    _has_required,
    fit_fingerprint,
    fit_ridge,
    production_config,
)
from patron.metrics.forecast import attach_forecast


def test_omitting_best_actual_player_is_not_a_perfect_ranker():
    rows = [{"score": None, "actual": 100.0}, {"score": 10.0, "actual": 10.0}]
    result = _rank_fold(rows, "score", "actual", 1, 2)
    assert result["hit_rate"] == 0
    assert result["coverage"] == 0.5
    assert result["ndcg"] < 1
    assert _rank_fold(rows, "score", "actual", 1, 2, False)["hit_rate"] == 1
    assert len(_overall_rows(rows, "score", {})) == 2


def test_rookie_rows_cannot_change_returner_coefficients():
    spec = ModelSpec(name="fitted_ppg", population="returner", features=("signal",))
    rows = [
        {"signal": float(i), "actual_ppg": float(i), "returning_indicator": 1} for i in range(10)
    ]
    rookie = {"signal": 99.0, "actual_ppg": 999.0, "rookie_indicator": 1}
    assert not _has_required(rookie, spec)
    a = fit_ridge(rows, spec.features, spec.target, 0.1, spec)
    b = fit_ridge([*rows, rookie], spec.features, spec.target, 0.1, spec)
    assert a.record() == b.record()


def test_research_spec_does_not_change_production_fingerprint():
    config = FitConfig()
    altered = replace(
        config, models=(*config.models, ModelSpec(name="research", features=("new",)))
    )
    assert fit_fingerprint(production_config(config)) == fit_fingerprint(production_config(altered))


def test_draft_archive_is_exact_and_verified(tmp_path):
    archive = Path(__file__).parents[1] / "data/static/draft_2026"
    original = verify_draft(archive).read_bytes()
    import hashlib

    assert (
        hashlib.sha256(original).hexdigest()
        == "ad726895d69481d0f4e2a423d90d300908875b031717ebd7ee92d8536bd772a9"
    )
    publish_draft(archive, tmp_path)
    assert (tmp_path / "board.json").read_bytes() == original
    assert (tmp_path / "board_v1.json").read_bytes() == original


def test_provenance_detects_drift(tmp_path):
    board, dependency = tmp_path / "board.json", tmp_path / "model.json"
    board.write_text("[]")
    dependency.write_text("before")
    publish_provenance(board, [dependency])
    assert artifact_status(board)["state"] == "current"
    dependency.write_text("after")
    assert artifact_status(board)["state"] == "stale"


def test_canonical_adaptive_does_not_invent_a_decomposition():
    frame = pl.DataFrame(
        {
            "fitted_ppg": [10.0],
            "fitted_games": [12.0],
            "fitted_season_points": [120.0],
            "adaptive_season_points": [200.0],
        }
    )
    row = attach_forecast(frame, "2026-08-30").row(0, named=True)
    assert row["forecast_active_ppg"] is None
    assert row["forecast_expected_games"] is None
    assert row["forecast_season_points"] == 200
    assert row["simulation_source"] == "v2_separate_estimate"


def test_wire_replacement_is_best_of_thirty():
    frame = pl.DataFrame(
        {"position": ["RB"] * 30, "ppg": list(range(30, 0, -1)), "is_free_agent": [True] * 30}
    )
    assert wire_replacement_levels(frame, {"RB": 25}) == {"RB": 30.0}


def test_lineup_gain_respects_existing_starters_and_flex():
    board = pl.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "player_display_name": ["A", "B", "C"],
            "position": ["RB"] * 3,
            "ppg": [20.0, 5.0, 10.0],
            "is_mine": [True, True, False],
        }
    )
    wire = board.tail(1)
    assert wire_lineup_improvements(wire, board, {"RB": 1})["lineup_improvement"][0] == 0
    assert wire_lineup_improvements(wire, board, {"RB": 1, "RB/WR/TE": 1})["lineup_improvement"][
        0
    ] == pytest.approx(5)


def test_live_depth_chart_uses_recorded_artifact_date(monkeypatch):
    from patron import pipeline
    from patron.config.league import get_league
    from patron.metrics.backtest import MetricReportConfig

    cutoffs = []
    monkeypatch.setattr(pipeline.nflverse, "load_depth_charts", lambda seasons: pl.DataFrame())

    def select(frame, season, cutoff):
        cutoffs.append(cutoff)
        return None

    monkeypatch.setattr(pipeline, "depth_chart_as_of", select)
    pipeline.load_draft_depth_chart(
        get_league(), MetricReportConfig.from_config(), snapshot_date="2026-08-28T19:13:13Z"
    )
    assert cutoffs == ["08-28"]


def test_production_artifact_loads_without_research_fingerprint(tmp_path):
    import json

    from patron.metrics.backtest import MetricReportConfig
    from patron.metrics.fit import RidgeModel, build_fit_artifact
    from patron.pipeline import load_fitted_models

    config = MetricReportConfig.from_config()
    predictions = pl.DataFrame({"forecast_season": [2025, 2026], "outcome_complete": [True, False]})
    artifact = build_fit_artifact(config.fit, predictions, config.depth_chart_cutoff)
    artifact.pop("fingerprint")
    record = RidgeModel(
        features=["ppg"],
        means=[10.0],
        scales=[1.0],
        coefficients=[1.0],
        intercept=10.0,
        n_train=100,
        ridge_lambda=1.0,
    ).record()
    record.update(model="fitted_ppg", position="QB", forecast_season=2026)
    path = tmp_path / "production_model.json"
    path.write_text(json.dumps({"fitted_artifact": artifact, "fitted_models": [record]}))
    assert "QB" in load_fitted_models(path, 2026, config, None)["fitted_ppg"]


def test_export_preserves_model_input_precision(tmp_path):
    from patron.pipeline import export_board

    frame = pl.DataFrame(
        {
            "player_id": ["a"],
            "age_factor": [0.987654321],
            "forecast_source": ["production_returner_v1"],
            "forecast_active_ppg": [17.1234567],
        }
    )
    path = tmp_path / "board.json"
    export_board(frame, path)
    saved = pl.read_json(path)
    assert saved["age_factor"][0] == frame["age_factor"][0]
    assert saved["forecast_active_ppg"][0] == frame["forecast_active_ppg"][0]
