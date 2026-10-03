"""Serving, provenance and temporal safety gates for the NextGen system."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from engine.api import app as api
from engine.api import nextgen_routes as routes
from engine.data import nextgen as policy
from engine.data.catalog import publish_catalog
from engine.data.releases import digest, reference, write_json
from engine.metrics.nextgen import COUNTERS, build_panel, distribution, fit_fold, paired_summary


def entry(model="baseline"):
    return dict(
        id=f"forecast:{model}:receiving_yards",
        label=model,
        kind="forecast",
        target="receiving_yards",
        horizon="preseason_season",
        unit="yards",
        positions=["WR"],
        populations=["returner"],
        validity="verified",
        evidence="reference_baseline" if model == "baseline" else "exploratory",
        serving="baseline" if model == "baseline" else "shadow",
        allowed_uses=["forecast"] if model == "baseline" else [],
        dependencies=["nfl_player_weeks.receiving_yards"],
        reason="Declared scope only",
    )


def test_serving_requires_the_exact_use_outcome_position_and_horizon():
    baseline = entry()
    assert policy.eligible(baseline, use="forecast", target="receiving_yards", position="WR")[0]
    for scope in (
        dict(use="decision"),
        dict(use="forecast", target="season_points"),
        dict(use="forecast", position="QB"),
        dict(use="forecast", population="rookie"),
        dict(use="forecast", horizon="next_four_weeks"),
    ):
        assert not policy.eligible(baseline, **scope)[0]
    assert not policy.eligible(entry("ridge"), use="forecast")[0]
    for disposition in ("archive", "shadow", "suspended"):
        assert not policy.eligible({**baseline, "serving": disposition}, use="forecast")[0]
    baseline["validity"] = "quarantined"
    assert not policy.eligible(baseline, use="forecast")[0]


def test_incident_suspends_derivatives_but_keeps_unaffected_measurements():
    incident = dict(
        status="open", dependencies=["nfl_player_weeks.receiving_yards"], reason="defect"
    )
    assert not policy.eligible(entry(), use="forecast", incidents=[incident])[0]
    unrelated = {**entry(), "dependencies": ["nfl_player_weeks.league_points"]}
    assert policy.eligible(unrelated, use="forecast", incidents=[incident])[0]
    incident["status"] = "resolved"
    assert policy.eligible(entry(), use="forecast", incidents=[incident])[0]


def test_distribution_preserves_zeros_negatives_and_undefined_samples():
    assert distribution([None])["observations"] == 0
    assert distribution([0])["mean"] == 0
    assert distribution([0])["std"] is None
    assert distribution([-4, 0, 2])["cv"] is None
    assert distribution([-4, 0, 2])["top2_positive_share"] == 1
    assert distribution([40, 40] + [4] * 8)["mean_without_top2"] == 4
    assert distribution([1, 2])["mean_without_top2"] is None


def test_panel_uses_all_prior_history_keeps_unknowns_and_never_future_features():
    features = pl.DataFrame(
        [
            dict(
                player_id="p",
                player_display_name="Player",
                position="WR",
                player_population="returner",
                forecast_season=y,
                forecast_cutoff_date=f"{y}-08-01",
                games=1,
            )
            for y in (2019, 2026)
        ]
    )
    outcomes = features.select("player_id", "forecast_season", "forecast_cutoff_date").with_columns(
        pl.Series("outcome_complete", [True, False]), pl.Series("actual_games", [1, None])
    )
    weeks = pl.DataFrame(
        [
            {
                "player_id": "p",
                "season": y,
                **{c: 3.0 for c in COUNTERS},
                "targets": None if y == 2003 else 1.0,
            }
            for y in (2003, 2018, 2019, 2026)
        ]
    )
    panel = build_panel(features, outcomes, weeks)
    assert panel[0]["x_career_weeks"] == 2
    assert panel[0]["x_career_targets_rate"] is None
    assert panel[0]["y_receiving_yards"] == 3
    assert panel[1]["y_receiving_yards"] is None
    changed = weeks.with_columns(
        pl.when(pl.col("season") >= 2019)
        .then(99999.0)
        .otherwise(pl.col("receiving_yards"))
        .alias("receiving_yards")
    )
    updated = build_panel(features, outcomes, changed)
    assert {k: v for k, v in panel[0].items() if k.startswith("x_")} == {
        k: v for k, v in updated[0].items() if k.startswith("x_")
    }
    no_opportunity = weeks.with_columns(pl.lit(0.0).alias("targets"))
    assert build_panel(features, outcomes, no_opportunity)[0]["y_receiving_efficiency"] is None


def test_heldout_labels_do_not_change_models_or_training_imputation():
    train = [
        dict(
            forecast_season=2004 + i // 20,
            player_population="returner",
            x_prior=i,
            x_missing=None if i % 4 else i,
            y_receiving_yards=float(i * 3),
        )
        for i in range(60)
    ]
    test = [
        dict(
            forecast_season=2026,
            player_population="returner",
            x_prior=22,
            x_missing=1e6,
            prior_receiving_yards=None,
            y_receiving_yards=None,
        )
    ]
    fits, n = fit_fold(train, test, "receiving_yards", ["x_prior", "x_missing"])
    changed = deepcopy(test)
    changed[0]["y_receiving_yards"] = -99999
    updated, count = fit_fold(train, changed, "receiving_yards", ["x_prior", "x_missing"])
    assert n == count == 60
    assert set(fits) == {"baseline", "ridge", "boost"}
    for model in fits:
        np.testing.assert_array_equal(fits[model], updated[model])
    assert fits["baseline"][0] == np.mean([r["y_receiving_yards"] for r in train])


def test_probability_evaluation_reports_brier_and_calibration_with_zero_events():
    report = paired_summary(
        [
            dict(season=2024, actual=0, prediction=0.25, baseline=0.5),
            dict(season=2025, actual=1, prediction=0.75, baseline=0.5),
            dict(season=2025, actual=None, prediction=0.75, baseline=0.5),
        ],
        probability=True,
    )
    assert report["error"] == 0.0625
    assert report["metric"] == "Brier score"
    assert sum(r["n"] for r in report["calibration"]) == 2
    assert report["prediction_coverage"] == 1
    assert report["observed_outcomes"] == 2
    assert report["paired_coverage"] == 1


def route_fixture(tmp_path, monkeypatch):
    write_json(tmp_path / "registry.json", [entry(), entry("ridge")])
    write_json(tmp_path / "incidents.json", [])
    pl.DataFrame(
        [
            dict(
                player_id="p",
                player_display_name="Player",
                target="receiving_yards",
                position="WR",
                population="returner",
                model=m,
                prediction=100.0,
            )
            for m in ("baseline", "ridge")
        ]
    ).write_parquet(tmp_path / "forecasts.parquet")
    monkeypatch.setattr(routes, "release", lambda: (tmp_path, {"version": "test"}))


def test_direct_api_and_csv_use_the_same_policy(tmp_path, monkeypatch):
    route_fixture(tmp_path, monkeypatch)
    for output in ("json", "csv"):
        with pytest.raises(HTTPException) as exc:
            routes.forecasts(
                target="receiving_yards", model="ridge", format=output, limit=100, offset=0
            )
        assert exc.value.status_code == 409
        archived = routes.forecasts(
            target="receiving_yards",
            model="ridge",
            format=output,
            limit=100,
            offset=0,
            scope="research",
        )
        if output == "json":
            assert archived["entry"]["serving"] == "shadow"
        else:
            assert b"shadow" in archived.body
    approved = routes.forecasts(target="receiving_yards", limit=100, offset=0, format="csv")
    assert b"baseline,test" in approved.body
    with pytest.raises(HTTPException):
        routes.forecasts(target="receiving_yards", horizon="weekly", limit=100, offset=0)


def test_legacy_boundary_requires_explicit_archive_scope_and_labels_response(monkeypatch):
    monkeypatch.setattr(api, "get_settings", lambda: SimpleNamespace(data_dir=Path("unused")))
    monkeypatch.setattr(api, "current_catalog", lambda _: {"products": {"analysis": {}}})
    calls = []

    async def handler(_):
        calls.append(True)
        return JSONResponse({"archived": True})

    for path in ("/api/board", "/api/league/wire", "/api/research/outlook", "/api/metric-report"):
        request = SimpleNamespace(url=SimpleNamespace(path=path), headers={}, query_params={})
        response = asyncio.run(api.analysis_boundary(request, handler))
        assert response.status_code == 409
        assert calls == []
    request.headers = {"X-Analysis-Scope": "research"}
    response = asyncio.run(api.analysis_boundary(request, handler))
    assert response.status_code == 200
    assert response.headers["X-Artifact-Disposition"] == "archived_not_for_analysis"
    assert calls == [True]
    monkeypatch.setattr(api, "current_catalog", lambda _: json.loads("broken"))
    response = asyncio.run(api.analysis_boundary(request, handler))
    assert response.status_code == 409
    assert calls == [True]


def test_catalog_cannot_silently_drop_nextgen_policy(tmp_path, monkeypatch):
    import engine.data.catalog as catalog

    monkeypatch.setattr(
        catalog,
        "load_gold",
        lambda *_: SimpleNamespace(manifest={"current_observations": {"season": 2026}}),
    )
    monkeypatch.setattr(catalog, "current_catalog", lambda _: {"products": {"analysis": {}}})
    with pytest.raises(ValueError, match="silently drop"):
        publish_catalog(tmp_path, "new", {"college": "x", "profiles": "x", "outlook_2026": "x"})


def test_analysis_verifies_artifacts_and_exact_profile_dependency(tmp_path, monkeypatch):
    root = tmp_path / "research/a"
    root.mkdir(parents=True)
    profile = tmp_path / "research/p"
    profile.mkdir()
    write_json(profile / "manifest.json", {})
    for name in policy.REQUIRED:
        (root / name).write_text("{}")
    gold_ref = {"version": "gold", "manifest_sha256": "g"}
    manifest = dict(
        kind="nextgen_analysis",
        status="complete",
        gold=gold_ref,
        profiles=reference(profile),
        evidence={},
        files={n: digest(root / n) for n in policy.REQUIRED},
    )
    write_json(root / "manifest.json", manifest)
    ref = reference(root)
    monkeypatch.setattr(
        policy,
        "load_gold",
        lambda *_: SimpleNamespace(ref=gold_ref, manifest={"files": {"target_quality.json": "t"}}),
    )
    monkeypatch.setattr(policy, "verify_evidence", lambda *_: None)
    monkeypatch.setattr(
        policy,
        "current_catalog",
        lambda _: dict(gold=gold_ref, products={"analysis": ref, "profiles": reference(profile)}),
    )
    assert policy.load_analysis(tmp_path)[0] == root
    (root / "registry.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        policy.load_analysis(tmp_path)
    (root / "registry.json").write_text("{}")
    write_json(profile / "manifest.json", {"changed": True})
    with pytest.raises(ValueError, match="profiles use different"):
        policy.load_analysis(tmp_path)


def test_no_fit_is_explicit_missing_coverage_not_a_zero_prediction():
    report = paired_summary(
        [
            dict(season=2009, actual=150, prediction=None, baseline=None),
            dict(season=2008, actual=None, prediction=None, baseline=None),
        ]
    )
    assert report["eligible_rows"] == 2
    assert report["observed_outcomes"] == 1
    assert report["prediction_coverage"] == 0
    assert report["n"] == 0
    assert report["evidence"] == "untested"
