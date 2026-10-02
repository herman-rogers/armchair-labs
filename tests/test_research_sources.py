"""Version selection must not mix accepted research with live or failed artifacts."""

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest
from fastapi import HTTPException

from patron.api import research_routes as routes
from patron.api import research_sources as sources
from patron.config.settings import Settings


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(sources, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    version = tmp_path / "research/repaired_test"
    outputs = version / "outputs"
    outputs.mkdir(parents=True)
    settings.outputs_dir.mkdir()
    raw = version / "raw.csv"
    raw.write_text("frozen input")
    predictions = outputs / "metric_backtest_predictions.parquet"
    pl.DataFrame(
        {
            "player_id": ["rookie", "returner"],
            "player_display_name": ["Rookie", "Returner"],
            "position": ["RB", "RB"],
            "player_population": ["rookie", "returner"],
            "forecast_season": [2025, 2025],
            "outcome_complete": [True, True],
            "forecast_cutoff_date": ["2025-08-29", "2025-08-29"],
            "archived_only_model": [90.0, 100.0],
            "fitted_season_points": [None, 100.0],
            "actual_season_points": [80.0, 60.0],
        }
    ).write_parquet(predictions)
    weekly = outputs / "research_audited_weekly_points.parquet"
    pl.DataFrame({"player_id": ["rookie"], "points": [80.0]}).write_parquet(weekly)
    report = outputs / "metric_report.json"
    write_json(report, {"title": "repaired report"})
    audit = outputs / "data_integrity_audit_2026-09-22.json"
    checks = {"scoring_parity": True}
    write_json(
        audit,
        {
            "accepted": True,
            "acceptance_checks": checks,
            "prediction_sha256": sha(predictions),
            "weekly_points_sha256": sha(weekly),
        },
    )
    manifest = version / "manifest.json"
    write_json(
        manifest,
        {
            "version": version.name,
            "status": "rebuilt_pending_audit",
            "protected_unchanged": True,
            "source_sha256": {"raw.csv": sha(raw)},
            "output_sha256": {
                str(path.relative_to(version)): sha(path) for path in (predictions, weekly, report)
            },
            "report_config": {
                "fit": {
                    "models": [
                        {
                            "name": "archived_only_model",
                            "kind": "ridge",
                            "target": "actual_season_points",
                        }
                    ]
                }
            },
        },
    )
    write_json(
        version / "acceptance.json",
        {
            "version": version.name,
            "status": "accepted_for_research_not_promoted",
            "generated_at": "2026-09-22T23:00:00+00:00",
            "forecast_rows": 2,
            "acceptance_checks": checks,
            "audited_input": {
                "manifest_sha256": sha(manifest),
                "audit_sha256": sha(audit),
                "prediction_sha256": sha(predictions),
            },
        },
    )
    # Legacy files deliberately disagree, making accidental fallback visible.
    pl.DataFrame({"legacy": [1]}).write_parquet(
        settings.outputs_dir / "metric_backtest_predictions.parquet"
    )
    write_json(settings.outputs_dir / "metric_report.json", {"title": "legacy report"})
    return version


def test_only_accepted_rebuilds_are_default_and_failed_runs_stay_hidden(accepted):
    version = accepted
    failed = version.parent / "failed_build"
    failed.mkdir()
    write_json(failed / "manifest.json", {"status": "failed"})
    payload = routes.sources()
    assert payload["default_id"] == version.name
    assert [row["id"] for row in payload["datasets"]] == [version.name, "legacy"]
    assert payload["unavailable"] == []


def test_catalog_players_and_report_use_same_dataset_and_saved_definitions(accepted):
    version = accepted
    catalog = routes.catalog(dataset=version.name)
    assert catalog["dataset"]["id"] == version.name
    model = next(row for row in catalog["models"] if row["id"] == "archived_only_model")
    assert model["unit"] == "season points"
    assert catalog["seasons"][0]["cutoff_dates"] == ["2025-08-29"]
    assert not any(row["id"] == "frozen_adaptive_2026" for row in catalog["models"])
    data = routes.players(dataset=version.name, model=model["id"], season=2025, population="rookie")
    assert data["model_unit"] == "season points"
    assert data["pool_size"] == 1
    assert data["players"][0]["model_value"] == 90
    assert data["players"][0]["historical_context"]["cutoff_date"] == "2025-08-29"
    assert (
        json.loads(Path(routes.report(dataset=version.name).path).read_text())["title"]
        == "repaired report"
    )
    assert json.loads(Path(routes.report().path).read_text())["title"] == "legacy report"


@pytest.mark.parametrize(
    "filename",
    [
        "raw.csv",
        "manifest.json",
        "outputs/metric_report.json",
        "outputs/metric_backtest_predictions.parquet",
        "outputs/research_audited_weekly_points.parquet",
        "outputs/data_integrity_audit_2026-09-22.json",
    ],
)
def test_changed_artifacts_fail_closed_even_after_cached_verification(accepted, filename):
    version = accepted
    sources.accepted_version(version.name)
    with (version / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(HTTPException) as exc:
        routes.catalog(dataset=version.name)
    assert exc.value.status_code == 409
    payload = routes.sources()
    assert [row["id"] for row in payload["datasets"]] == ["legacy"]
    assert payload["unavailable"][0]["id"] == version.name


@pytest.mark.parametrize("name", ["../outputs", "/tmp/anything", "../../research/repaired_test"])
def test_dataset_identifier_cannot_escape_root(accepted, name):
    with pytest.raises(HTTPException) as exc:
        routes.catalog(dataset=name)
    assert exc.value.status_code == 422


def test_missing_acceptance_cannot_be_selected(accepted):
    version = accepted
    (version / "acceptance.json").unlink()
    with pytest.raises(HTTPException) as exc:
        sources.accepted_version(version.name)
    assert exc.value.status_code == 409
