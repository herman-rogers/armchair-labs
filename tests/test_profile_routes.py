import json
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import HTTPException

from patron.api import profile_routes as routes
from patron.api import profile_sources as sources
from patron.metrics.player_profile import METRICS


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "patron.data.serving.get_settings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    root = tmp_path / "research/profiles"
    root.mkdir(parents=True)
    (root / "implementation").mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    history = {"version": "history"}
    report = {
        "version": "profiles",
        "schema_version": 1,
        "history": history,
        "season": 2026,
        "through_week": 2,
        "college_version": "college",
        "outlook_version": "outlook",
        "observations_saved_at": "2026-09-22",
        "metrics": METRICS,
        "limits": [],
    }
    (root / "report.json").write_text(json.dumps(report))
    pl.DataFrame(
        {
            "player_id": ["p1"],
            "player_display_name": ["Player [One]"],
            "position": ["WR"],
            "current_candidate": [True],
            "college_linked": [True],
            "rookie_season": [2024],
        }
    ).write_parquet(root / "players.parquet")
    pl.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "season": [2024, 2026],
            "week": [1, 1],
            "team": ["A", "B"],
            "stat_recorded": [True, True],
            "league_points": [10.0, 999.0],
            "snap_share": [0.8, 0.2],
        }
    ).write_parquet(root / "nfl_weeks.parquet")
    pl.DataFrame(
        schema={"player_id": pl.String, "season": pl.Int64, "week": pl.Int64}
    ).write_parquet(root / "injuries.parquet")
    (root / "profile_features.parquet").write_text("features")
    college = tmp_path / "research/college"
    college.mkdir()
    pl.DataFrame(
        {
            "college_id": ["c1", "c2"],
            "college_name": ["Player [One]", "Same Name"],
            "player_id": ["p1", None],
            "status": ["linked", "review"],
        }
    ).write_parquet(college / "identity_links.parquet")
    pl.DataFrame(
        {
            "college_id": ["c1", "c2"],
            "season": [2023, 2023],
            "team_id": ["A", "A"],
            "college_team": ["School A", "School A"],
            "complete_team_season": [True, True],
        }
    ).write_parquet(college / "college_seasons.parquet")
    pl.DataFrame(
        {
            "player_id": ["p1"],
            "forecast_year": [2024],
            "target": ["nfl_first3_points"],
            "actual": [1000.0],
        }
    ).write_parquet(college / "nfl_predictions.parquet")
    pl.DataFrame(
        {
            "college_id": ["c1"],
            "forecast_year": [2024],
            "position": ["receiving_yards"],
            "actual": [1000.0],
        }
    ).write_parquet(college / "college_predictions.parquet")
    historical = tmp_path / "research/history/outputs"
    historical.mkdir(parents=True)
    pl.DataFrame(
        {"player_id": ["p1"], "forecast_season": [2024], "fitted_ppg": [11.0]}
    ).write_parquet(historical / "metric_backtest_predictions.parquet")
    outlook = tmp_path / "research/outlook"
    outlook.mkdir()
    (outlook / "outlook.json").write_text(
        json.dumps({"players": [{"player_id": "p1", "forecast_next4": 60}]})
    )
    dependency = college / "identity_links.parquet"
    manifest = {
        "version": "profiles",
        "kind": "player_profiles",
        "status": "complete",
        "protected_artifacts_unchanged": True,
        "history": history,
        "source_sha256": {str(dependency.relative_to(tmp_path)): sources.digest(dependency)},
        "output_sha256": {name: sources.digest(root / name) for name in sources.REQUIRED},
        "implementation_sha256": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (outputs / "player_profiles.json").write_text(
        json.dumps(
            {"version": "profiles", "manifest_sha256": sources.digest(root / "manifest.json")}
        )
    )
    settings = SimpleNamespace(data_dir=tmp_path, outputs_dir=outputs)
    monkeypatch.setattr(sources, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        sources, "accepted_version", lambda _: (tmp_path, {"audited_input": history}, {})
    )
    return root, dependency


def test_profile_cutoff_excludes_future_production_forecasts_and_immature_outcomes(
    tmp_path, monkeypatch
):
    fixture(tmp_path, monkeypatch)
    data = routes.profile(player_id="p1", season=2024, week=18, scope="research")
    assert data["career"]["league_points"] == 10
    assert data["current_outlook"] is None
    assert data["college_forecasts"][0]["actual"] is None
    assert data["college_development_forecasts"][0]["actual"] is None
    assert len(data["college_seasons"]) == 1
    assert data["features"]["completed_seasons_observed"] == 1
    assert data["nfl_seasons"][0]["partial"] is False
    assert data["available_through"] == {"season": 2026, "week": 2}
    partial = routes.profile(player_id="p1", season=2024, week=1)
    assert partial["features"]["completed_seasons_observed"] == 0
    assert partial["nfl_seasons"][0]["partial"] is True
    preseason = routes.profile(player_id="p1", season=2024, week=0)
    assert preseason["preseason_forecasts"] == []
    assert preseason["college_forecasts"] == []
    assert preseason["college_development_forecasts"] == []
    assert preseason["career"]["observed_weeks"] == 0
    with pytest.raises(HTTPException) as exc:
        routes.profile(player_id="p1", season=2026, week=3)
    assert exc.value.status_code == 422


def test_review_college_identity_never_borrows_nfl_data(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    data = routes.profile(college_id="c2", week=None)
    assert data["identity"]["player_id"] is None
    assert data["nfl_seasons"] == []
    assert data["current_outlook"] is None
    assert len(data["college_seasons"]) == 1


@pytest.mark.parametrize("changed", ["output", "source"])
def test_profile_mutations_fail_closed(tmp_path, monkeypatch, changed):
    root, dependency = fixture(tmp_path, monkeypatch)
    assert sources.load_profiles()[0] == root
    (root / "nfl_weeks.parquet" if changed == "output" else dependency).write_text("changed")
    with pytest.raises(HTTPException) as exc:
        sources.load_profiles()
    assert exc.value.status_code == 409


def test_directory_literal_search_and_explicit_missing_id(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    assert routes.players(search="[One]", limit=100, offset=0)["total"] == 1
    with pytest.raises(HTTPException) as exc:
        routes.profile(player_id="missing", week=None)
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        routes.profile(player_id="p1", college_id="c1", week=None)
    assert exc.value.status_code == 422


def test_directory_population_filters_before_pagination(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path, monkeypatch)
    report = json.loads((root / "report.json").read_text())
    players = pl.read_parquet(root / "players.parquet")
    pl.concat(
        [
            players,
            players.with_columns(
                pl.lit("rookie").alias("player_id"),
                pl.lit(2026, dtype=pl.Int64).alias("rookie_season"),
            ),
            players.with_columns(
                pl.lit("unknown").alias("player_id"),
                pl.lit(None, dtype=pl.Int64).alias("rookie_season"),
            ),
        ]
    ).write_parquet(root / "players.parquet")
    monkeypatch.setattr(routes, "load_profiles", lambda: (root, report))
    assert (
        routes.players(population="rookie", limit=1, offset=0)["players"][0]["player_id"]
        == "rookie"
    )
    assert routes.players(population="returner", limit=1, offset=0)["total"] == 1
    assert routes.players(population="rookie", limit=1, offset=1)["players"] == []
    with pytest.raises(HTTPException) as exc:
        routes.players(population="invalid", limit=100, offset=0)
    assert exc.value.status_code == 422


def test_nextgen_profile_does_not_read_or_serve_archived_forecasts(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    scan = pl.scan_parquet

    def observations_only(path, *args, **kwargs):
        assert "predictions.parquet" not in str(path)
        return scan(path, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", observations_only)
    data = routes.profile(player_id="p1", week=None)
    assert data["preseason_forecasts"] == []
    assert data["college_forecasts"] == []
    assert data["college_development_forecasts"] == []
    assert data["current_outlook"] is None
    assert data["career"]["observed_weeks"] > 0
    assert data["forecast_scope"] == "approved_analysis"
    assert data["approved_forecasts"] == []


def test_tracking_hides_unfinished_and_future_seasons(tmp_path, monkeypatch):
    from patron.metrics.profile_tracking import TRACKING_SCHEMA

    root, _ = fixture(tmp_path, monkeypatch)
    report = json.loads((root / "report.json").read_text())
    report["schema_version"] = 2
    monkeypatch.setattr(routes, "load_profiles", lambda: (root, report))
    pl.DataFrame(
        [
            {"player_id": "p1", "season": year, "position": "WR", "ngs_separation": value}
            for year, value in [(2023, None), (2024, 0.0), (2025, 3.0), (2026, 99.0)]
        ],
        schema=TRACKING_SCHEMA,
    ).write_parquet(root / "tracking_seasons.parquet")
    partial = routes.profile(player_id="p1", season=2024, week=1)
    assert [r["season"] for r in partial["tracking_seasons"]] == [2023]
    complete = routes.profile(player_id="p1", season=2024, week=18)
    assert [r["ngs_separation"] for r in complete["tracking_seasons"]] == [None, 0.0]
    latest = routes.profile(player_id="p1", week=None)
    assert max(r["season"] for r in latest["tracking_seasons"]) == 2025


def test_profile_rejects_unavailable_forecast_policy(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)

    def broken(*args):
        raise ValueError("Profile and approved forecasts use different releases")

    monkeypatch.setattr(routes, "approved_profile_forecasts", broken)
    with pytest.raises(HTTPException) as exc:
        routes.profile(player_id="p1", week=None)
    assert exc.value.status_code == 409
