"""Prospective scoring preserves frozen constraints and undefined rates."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from engine.data.releases import write_json


def test_frozen_zero_opportunity_is_not_zero_efficiency_and_current_schedule_is_used(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "qb_score", Path(__file__).resolve().parents[1] / "research/score_qb_variations.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "research/frozen"
    (root / "qb_passing").mkdir(parents=True)
    (root / "qb_variations").mkdir()
    manifest = dict(
        version="frozen",
        qb_variations_schema_version=1,
        gold=dict(version="gold"),
        generated_at="2026-09-24T18:00:00Z",
    )
    write_json(root / "manifest.json", manifest)
    write_json(root / "qb_variations/report.json", dict(season=2026, through_week=2))
    passing = dict(
        player_id="q",
        horizon="next_game",
        season=2026,
        team="NE",
        schedule_known=True,
        through_week=2,
        end_week=3,
        execution_ypa=7.0,
        execution_reference=6.5,
        prediction=0.0,
        reference=250.0,
        constraint_reason="Dated absence",
        issued_at="2026-09-24T18:00:00Z",
    )
    pl.DataFrame([passing]).write_parquet(root / "qb_passing/current.parquet")
    ranking = dict(
        player_id="q",
        position="QB",
        horizon="next4",
        season=2026,
        team="NE",
        schedule_known=True,
        through_week=2,
        end_week=6,
        prediction=0.0,
        rank_eligible=False,
    )
    pl.DataFrame([ranking]).write_parquet(root / "rankings.parquet")
    pl.DataFrame([dict(player_id="q", season=2026, horizon="next4", reference=50.0)]).write_parquet(
        root / "qb_variations/ranking_predictions.parquet"
    )
    weeks = pl.DataFrame(
        schema={
            "player_id": pl.String,
            "season": pl.Int64,
            "season_type": pl.String,
            "week": pl.Int64,
            "passing_yards": pl.Float64,
            "attempts": pl.Float64,
            "league_points": pl.Float64,
        }
    )
    schedule = pl.DataFrame(
        [
            dict(
                game_id="g",
                season=2026,
                week=3,
                game_type="REG",
                gameday="2026-09-27",
                home_team="NE",
                away_team="BUF",
            )
        ]
    )
    tables = dict(
        nfl_player_weeks=weeks,
        current_weeks=weeks,
        nfl_schedule=schedule.head(0),
        current_schedule=schedule,
    )
    gold = SimpleNamespace(
        ref=dict(version="gold"),
        manifest=dict(current_observations=dict(season=2026, through_week=6)),
        read=lambda name: tables[name],
    )
    monkeypatch.setattr(module, "load_analysis", lambda *_: (root, manifest))
    monkeypatch.setattr(module, "load_gold", lambda *_: gold)
    module.score(tmp_path, "frozen", "scored")
    result = json.loads((tmp_path / "research/scored/report.json").read_text())
    assert result["scored"] == 2
    assert result["undefined_rates"] == 1
    assert result["excluded_same_day_or_unknown_timing"] == 0
    assert {r["target"] for r in result["outcomes"]} == {"yards", "league_points"}
    assert all(r["reference"] == r["prediction"] == r["actual"] == 0 for r in result["outcomes"])
