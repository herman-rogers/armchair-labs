"""Workload diagnostics preserve cutoff grouping, coverage and equal-season weights."""

import polars as pl
import pytest

from engine.api import nextgen_routes as routes
from engine.data.releases import write_json
from engine.metrics.evidence_diagnostics import passing_workload_diagnostics


def test_workload_groups_use_prior_attempts_keep_unknowns_and_weight_seasons_equally():
    predictions = pl.DataFrame(
        [
            dict(player_id=p, season=y, actual=a, prediction=f, baseline=f, complete=True)
            for p, y, a, f in [
                ("a", 2020, 0.0, 100.0),
                ("b", 2021, 0.0, 300.0),
                ("c", 2021, 0.0, 300.0),
                ("d", 2021, 4000.0, 100.0),
                ("e", 2021, 0.0, 0.0),
                ("f", 2021, 4000.0, 1000.0),
            ]
        ]
    )
    features = pl.DataFrame(
        [
            dict(player_id=p, forecast_season=y, x_prior_attempts=n)
            for p, y, n in [
                ("a", 2020, 300),
                ("b", 2021, 500),
                ("c", 2021, 400),
                ("d", 2021, 25),
                ("e", 2021, 0),
                ("f", 2021, None),
            ]
        ]
    )
    result = passing_workload_diagnostics(predictions, features)
    assert result["n"] == 6
    assert result["zero_outcomes"] == 4
    assert sum(g["n"] for g in result["workload_groups"]) == 6
    high, low, zero, unknown = result["workload_groups"]
    assert high["n"] == 3 and high["years"] == 2
    assert high["error"] == 200.0  # 100 and 300 receive equal season weight.
    assert low["error"] == 3900.0
    assert zero["n"] == unknown["n"] == 1
    updated = passing_workload_diagnostics(
        predictions.with_columns(pl.lit(9000.0).alias("actual")), features
    )
    assert [(g["label"], g["n"]) for g in updated["workload_groups"]] == [
        (g["label"], g["n"]) for g in result["workload_groups"]
    ]


def test_missing_predictions_and_pending_outcomes_do_not_enter_diagnostics():
    predictions = pl.DataFrame(
        [
            dict(player_id=p, season=2025, actual=a, prediction=f, baseline=10.0, complete=c)
            for p, a, f, c in [
                ("a", 20.0, 30.0, True),
                ("b", None, 30.0, True),
                ("c", 20.0, None, True),
                ("d", 20.0, 30.0, False),
            ]
        ]
    )
    features = pl.DataFrame(dict(player_id=["a"], forecast_season=[2025], x_prior_attempts=[300]))
    result = passing_workload_diagnostics(predictions, features)
    assert result["n"] == 1
    assert result["workload_groups"][0]["error"] == pytest.approx(10)


def test_evidence_diagnostics_respect_scope_population_and_window(tmp_path, monkeypatch):
    write_json(tmp_path / "incidents.json", [])
    write_json(
        tmp_path / "registry.json",
        [
            dict(
                id=f"forecast:{m}:passing_yards",
                validity="verified",
                serving="baseline" if m == "baseline" else "shadow",
                allowed_uses=["forecast"],
                target="passing_yards",
                positions=["QB"],
                populations=["all", "returner"],
                dependencies=[],
            )
            for m in ["baseline", "boost"]
        ],
    )
    write_json(
        tmp_path / "evaluations.json",
        [
            dict(
                target="passing_yards",
                position="QB",
                population="returner",
                window="modern",
                model=m,
            )
            for m in ["baseline", "boost"]
        ],
    )
    pl.DataFrame(
        [
            dict(
                player_id=p,
                season=y,
                model=m,
                target="passing_yards",
                position="QB",
                population=population,
                actual=100.0,
                prediction=110.0,
                baseline=120.0,
                complete=True,
            )
            for p, y, population in [
                ("old", 2018, "returner"),
                ("veteran", 2021, "returner"),
                ("new", 2021, "rookie"),
            ]
            for m in ["baseline", "boost"]
        ]
    ).write_parquet(tmp_path / "predictions.parquet")
    pl.DataFrame(
        dict(
            player_id=["old", "veteran", "new"],
            forecast_season=[2018, 2021, 2021],
            x_prior_attempts=[100, 350, None],
        )
    ).write_parquet(tmp_path / "features.parquet")
    monkeypatch.setattr(
        routes,
        "release",
        lambda: (
            tmp_path,
            {"version": "test", "files": {"features.parquet": "verified-by-release"}},
        ),
    )
    normal = routes.evidence(target="passing_yards", population="returner", window="modern")
    assert [r["model"] for r in normal["comparisons"]] == ["baseline"]
    diagnostic = normal["comparisons"][0]["diagnostics"]
    assert diagnostic["n"] == 1
    assert diagnostic["workload_groups"][0]["label"] == "300+ prior-season pass attempts"
    archived = routes.evidence(
        target="passing_yards", population="returner", window="modern", scope="research"
    )
    assert {r["model"] for r in archived["comparisons"]} == {"baseline", "boost"}
