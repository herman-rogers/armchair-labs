"""Approved profile evidence must preserve missingness, scope and historical cutoffs."""

import json
from types import SimpleNamespace

import polars as pl
import pytest

from engine.data import profile_forecasts as forecasts
from engine.metrics.profile_tracking import TRACKING_SCHEMA, tracking_seasons


def test_tracking_preserves_provider_aggregates_zeros_and_unknowns():
    source = pl.DataFrame(
        [dict(player_id="p", season=2025, position="WR", ngs_separation=0.0, ngs_yac_oe=-1.2)],
        schema=TRACKING_SCHEMA,
    )
    row = tracking_seasons(source).row(0, named=True)
    assert row["ngs_separation"] == 0
    assert row["ngs_cpoe"] is None
    assert row["ngs_yac_oe"] == -1.2
    with pytest.raises(ValueError, match="Conflicting"):
        tracking_seasons(pl.concat([source, source]))
    with pytest.raises(ValueError, match="Nonfinite"):
        tracking_seasons(source.with_columns(pl.lit(float("nan")).alias("ngs_separation")))


def forecast_fixture(tmp_path, monkeypatch):
    report = {"version": "profiles", "gold": {"version": "gold"}, "season": 2026}
    manifest = {"version": "analysis", "profiles": {"version": "profiles"}, "gold": report["gold"]}
    monkeypatch.setattr(forecasts, "published", lambda _: True)
    monkeypatch.setattr(forecasts, "load_analysis", lambda _: (tmp_path, manifest))
    schedule = pl.DataFrame(
        {
            "season": [2024, 2024, 2026],
            "week": [1, 18, 1],
            "game_type": ["REG"] * 3,
            "gameday": ["2024-09-08", "2025-01-05", "2026-09-13"],
        }
    )
    monkeypatch.setattr(forecasts, "load_gold", lambda *_: SimpleNamespace(read=lambda _: schedule))
    registry = [
        dict(
            id=f"forecast:{model}:receiving_yards",
            label=model,
            kind="forecast",
            positions=["WR"],
            target="receiving_yards",
            horizon="preseason_season",
            validity="verified",
            allowed_uses=["forecast"] if model == "baseline" else [],
            serving=model,
            reason="Scoped baseline",
            dependencies=["receiving_yards"],
        )
        for model in ["baseline", "ridge"]
    ]
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    (tmp_path / "incidents.json").write_text("[]")
    rows = [
        dict(
            player_id="p",
            season=season,
            cutoff=cutoff,
            position=position,
            model=model,
            target="receiving_yards",
            horizon="preseason_season",
            prediction=100.0,
            actual=200.0,
            complete=True,
            unit="yards",
        )
        for season, cutoff, position, model in [
            (2024, "2024-08-30", "WR", "baseline"),
            (2024, "2024-08-30", "WR", "ridge"),
            (2024, "2024-08-30", "QB", "baseline"),
            (2024, "2024-10-01", "WR", "baseline"),
            (2026, "2026-08-28", "WR", "baseline"),
        ]
    ]
    pl.DataFrame(rows).write_parquet(tmp_path / "predictions.parquet")
    return report


def test_profile_forecasts_apply_registry_position_date_and_outcome_cutoffs(tmp_path, monkeypatch):
    report = forecast_fixture(tmp_path, monkeypatch)
    rows, version = forecasts.approved_profile_forecasts(tmp_path, report, "p", 2024, 1)
    assert version == "analysis"
    assert len(rows) == 1
    assert rows[0]["model"] == "baseline"
    assert rows[0]["actual"] is None
    assert rows[0]["complete"] is False
    rows, _ = forecasts.approved_profile_forecasts(tmp_path, report, "p", 2024, 18)
    assert all(r["actual"] == 200 for r in rows)
    rows, _ = forecasts.approved_profile_forecasts(tmp_path, report, "p", 2026, 1)
    assert rows[0]["actual"] is None
    assert forecasts.approved_profile_forecasts(tmp_path, report, "p", 2024, 0)[0] == []
    assert forecasts.approved_profile_forecasts(tmp_path, report, None, 2024, 18)[0] == []


def test_profile_forecasts_honor_incidents_and_exact_release(tmp_path, monkeypatch):
    report = forecast_fixture(tmp_path, monkeypatch)
    (tmp_path / "incidents.json").write_text(
        json.dumps(
            [{"status": "open", "dependencies": ["receiving_yards"], "reason": "Invalid input"}]
        )
    )
    assert forecasts.approved_profile_forecasts(tmp_path, report, "p", 2024, 18)[0] == []
    with pytest.raises(ValueError, match="different releases"):
        forecasts.approved_profile_forecasts(tmp_path, {**report, "version": "old"}, "p", 2024, 18)
