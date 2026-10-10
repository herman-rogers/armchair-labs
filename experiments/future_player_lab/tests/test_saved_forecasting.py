"""Cutoff isolation and artifact contracts for reusing saved research models."""

import json
from datetime import date

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.forecasting.data import (
    add_supplements,
    feature_matrix,
    feature_rows,
    snapshot_inputs,
)
from experiments.future_player_lab.forecasting.registry import load_spec, model_contract
from experiments.future_player_lab.forecasting.run import rank_predictions
from experiments.future_player_lab.gpu_lab.run import digest
from experiments.future_player_lab.notebooks import raw_distribution_data as raw


@pytest.fixture
def tables():
    box = pl.DataFrame(
        {
            "player_id": ["a", "a", "future_only"],
            "season": [2024, 2025, 2025],
            "week": [1, 1, 3],
            "team": ["ATL"] * 3,
            "position": ["RB"] * 3,
            "player_display_name": ["Player A", "Player A", "Future Player"],
            **{s: [1.0, 2.0, 1000.0] for s in raw.STATS},
        }
    )
    return {
        "box": box,
        "player": raw.prefixed(box, raw.KEY, "box"),
        "team": pl.DataFrame(
            {
                "team": ["ATL"] * 3,
                "season": [2024, 2025, 2025],
                "week": [1, 1, 3],
                "carries": [20.0, 21.0, 1000.0],
            }
        ),
        "schedule": pl.DataFrame(
            {
                "season": [2024, 2025, 2025, 2025],
                "week": [18, 1, 2, 18],
                "home_team": ["ATL"] * 4,
                "away_team": ["CAR"] * 4,
                "home_score": [0.0, 0.0, 0.0, None],
                "away_score": [0.0, 0.0, 0.0, None],
            }
        ),
        "combine": pl.DataFrame(
            {
                "pfr_id": ["rookie"],
                "draft_year": [2025],
                "pos": ["RB"],
                "draft_team": ["ATL"],
                "player_name": ["Rookie"],
                "season": [2025],
                "forty": [4.5],
            }
        ),
        "mapping": pl.DataFrame({"pfr_id": ["rookie"], "player_id": ["r"]}),
        "identity": pl.DataFrame(
            {"gsis_id": ["a", "r"], "birth_date": [date(2000, 1, 1), date(2003, 1, 1)]}
        ),
        "college": pl.DataFrame(
            {"player_id": ["r", "r"], "season": [2024, 2025], "yards": [100.0, 10000.0]}
        ),
    }


def test_features_do_not_require_outcomes_and_future_observations_cannot_change_them(tables):
    expected = feature_rows(tables, "RB", 2025, 2)
    changed = dict(tables)
    for name in ("box", "player", "team"):
        f = changed[name]
        cols = raw.numeric(f, ["player_id"])
        changed[name] = f.with_columns(
            *[
                pl.when((pl.col("season") == 2025) & (pl.col("week") > 2))
                .then(-999999)
                .otherwise(pl.col(c))
                .alias(c)
                for c in cols
            ]
        )
    changed["schedule"] = changed["schedule"].with_columns(
        pl.lit(99999.0).alias("home_score"), pl.lit(99999.0).alias("away_score")
    )
    assert feature_rows(changed, "RB", 2025, 2).equals(expected)
    assert set(expected["player_id"]) == {"a", "r"}
    assert not any(c.startswith("y_") for c in expected.columns)
    assert expected.filter(pl.col("player_id") == "a")["history_games"][0] == 1
    assert expected.filter(pl.col("player_id") == "r")["college__yards"][0] == 100


def test_future_supplement_values_and_players_are_cut_off(tmp_path, tables):
    tables["player"] = tables["player"].with_columns(
        pl.lit(None, pl.Float64).alias("ngs__efficiency")
    )
    manifest = {"status": "complete", "season": 2025, "sources": {}}
    ngs = pl.DataFrame(
        {
            "player_gsis_id": ["a", "future_only"],
            "season": [2025, 2025],
            "week": [1, 3],
            "efficiency": [4.0, 1000.0],
        }
    )
    injuries = pl.DataFrame({"gsis_id": ["a"], "season": [2025], "week": [1]})
    depth = pl.DataFrame({"gsis_id": ["a"], "dt": ["2025-01-01"]})
    for name, f in (("ngs_rushing", ngs), ("injuries", injuries), ("depth", depth)):
        path = tmp_path / f"{name}.parquet"
        f.write_parquet(path)
        manifest["sources"][name] = {
            "status": "available",
            "path": path.name,
            "sha256": digest(path),
        }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    actual, audit = add_supplements(tables, tmp_path, 2025, 2)
    values = actual["player"].filter(pl.col("ngs__efficiency").is_not_null())
    assert values["ngs__efficiency"].to_list() == [4.0]
    assert values["week"].max() <= 2
    assert {r["source"] for r in audit["sources"] if r["status"] == "withheld"} == {
        "injuries",
        "depth",
    }
    (tmp_path / "ngs_rushing.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        add_supplements(tables, tmp_path, 2025, 2)


def test_feature_names_are_required_but_unknown_values_are_preserved():
    frame = pl.DataFrame({"age": [20.0, None], "history_games": [0, 2]})
    values = feature_matrix(frame, ["history_games", "age"])
    assert values[0, 0] == 0
    assert np.isnan(values[1, 1])
    with pytest.raises(ValueError, match="Missing feature"):
        feature_matrix(frame, ["age", "prior__box__carries"])
    with pytest.raises(ValueError, match="Outcomes"):
        feature_matrix(frame, ["y_points"])


def test_registry_refuses_wrong_horizon_and_unregistered_positions(tmp_path):
    spec = load_spec("rb_ppr_v1")
    with pytest.raises(ValueError, match="supports completed weeks"):
        model_contract(spec, 2026, 4)
    with pytest.raises(ValueError, match="not registered"):
        load_spec("wr_ppr_v1")
    with pytest.raises(ValueError, match="Unsafe"):
        load_spec("../rb_ppr_v1")
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        model_contract(
            {**spec, "model_directory": str(tmp_path), "model_manifest_sha256": "wrong"}, 2026, 2
        )


def test_snapshot_cutoff_cannot_exceed_available_observations(tmp_path, tables):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "status": "accepted",
                "current_observations": {
                    "season": 2025,
                    "through_week": 2,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="exceeds"):
        snapshot_inputs(tables, path, 2025, 3)


def test_ranking_retains_unobserved_candidates_and_unknown_schedules(tables):
    panel = feature_rows(tables, "RB", 2025, 2).with_columns(
        pl.when(pl.col("player_id") == "r")
        .then(None)
        .otherwise(pl.col("scheduled_games"))
        .alias("scheduled_games")
    )
    predictions = {
        name: np.full(panel.height, 20.0)
        for name in ["ensemble_top3", "selected", "ensemble_diverse"]
    }
    result = rank_predictions(panel.reverse(), predictions, "ensemble_top3")
    assert result["player_id"].to_list() == ["a", "r"]
    assert result["rank"].to_list() == [1, 2]
    rookie = result.filter(pl.col("player_id") == "r")
    assert not rookie["observed_this_season"][0]
    assert rookie["ppr_per_scheduled_game"][0] is None
    with pytest.raises(ValueError, match="nonfinite"):
        rank_predictions(panel, {"ensemble_top3": np.array([np.nan, 20])}, "ensemble_top3")
