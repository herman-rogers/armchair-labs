"""Offline bundle replay works after original run files are removed."""

import hashlib
import json
import shutil

import joblib
import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.campaigns import replay
from experiments.future_player_lab.campaigns.data import SCORING, STATS
from experiments.future_player_lab.gpu_lab.run import packages

from engine.chairstats import publish_campaign


class ConstantPredictor:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value, dtype=float)


class ColumnPredictor:
    def predict(self, x):
        return x[:, 0]


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "original_run"
    source.mkdir()
    targets = [*STATS["RB"], "points"]
    identity = {
        "position": "RB",
        "horizon": "remaining",
        "origin_week": 2,
        "targets": targets,
        "training_through_season": 2025,
        "scoring": SCORING["RB"],
    }
    members = {
        target: [{"indices": [0], "model": ConstantPredictor(value)}]
        for target, value in {
            "carries": -1,
            "rushing_tds": 1,
            "targets": 0,
            "receptions": 1,
            "receiving_yards": 2,
            "receiving_tds": 0,
        }.items()
    }
    members["carries"].append({"indices": [0], "model": ConstantPredictor(1)})
    members["rushing_yards"] = [{"indices": [0], "model": ColumnPredictor()}]
    members["points"] = [{"indices": [1], "model": ColumnPredictor()}]
    names = ["history_rushing_yards", "history_receptions"]
    model = {**identity, "schema_version": 1, "feature_names": names, "models": members}
    joblib.dump(model, source / "models.joblib")
    (source / "feature_names.json").write_text(json.dumps(names))
    panel = pl.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "year": [2026, 2026],
            "position": ["RB", "RB"],
            "horizon": ["remaining", "remaining"],
            "origin": [2, 2],
            "weeks": [16, 16],
            "scheduled_games": [15, 0],
            names[0]: [-1.0, 3.0],
            names[1]: [10.0, 20.0],
        }
    )
    panel.write_parquet(source / "features.parquet")
    saved = {
        **identity,
        "schema_version": 1,
        "campaign_id": "replay_test",
        "model_id": "rb_remaining",
        "packages": packages(),
    }
    (source / "manifest.json").write_text(json.dumps(saved))
    (source / "metrics.json").write_text("{}")
    rows = []
    for pid, rank, rushing_yards, points, components, games in [
        ("p1", 2, -16.0, 160.0, 113.6, 15),
        ("p2", 1, 48.0, 320.0, 120.0, 0),
    ]:
        rows.append(
            {
                "player_id": pid,
                "position": "RB",
                "season": 2026,
                "origin_week": 2,
                "rank": rank,
                "scheduled_games": games,
                "calendar_weeks": 16,
                "component_points": components,
                "points_per_scheduled_game": points / games if games else None,
                "predictions": {
                    "carries": 8.0,
                    "rushing_yards": rushing_yards,
                    "rushing_tds": 16.0,
                    "targets": 0.0,
                    "receptions": 16.0,
                    "receiving_yards": 32.0,
                    "receiving_tds": 0.0,
                    "points": points,
                },
            }
        )
    receipt = {
        "rows": rows,
        "scoring": SCORING["RB"],
        "model_manifest_sha256": hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest(),
        "feature_parquet_sha256": hashlib.sha256(
            (source / "features.parquet").read_bytes()
        ).hexdigest(),
    }
    (source / "forecast.json").write_text(json.dumps(receipt))
    root = tmp_path / "campaigns"
    publish_campaign(
        root,
        {"campaign_id": "replay_test", "name": "Replay"},
        [
            {
                **identity,
                "model_id": "rb_remaining",
                "metrics": source / "metrics.json",
                "forecasts": [{"season": 2026, "origin_week": 2, "path": source / "forecast.json"}],
                "artifacts": {
                    "weights": source / "models.joblib",
                    "model_manifest": source / "manifest.json",
                    "feature_contract": source / "feature_names.json",
                    "forecast_features": source / "features.parquet",
                },
            }
        ],
    )
    shutil.rmtree(source)
    return root


def mutate(root, artifact, mutation):
    """Produce a consistently sealed, semantically incompatible local test bundle."""
    directory = root / "replay_test"
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["models"][0]
    ref = entry["forecasts"][0] if artifact == "forecast" else entry["artifacts"][artifact]
    file = directory / ref["path"]
    doc = json.loads(file.read_text())
    mutation(doc)
    file.write_text(json.dumps(doc))
    ref["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    (directory / "manifest.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def test_replay_copied_bundle_without_original_run(bundle):
    result = replay.replay_campaign(bundle, "replay_test")
    assert result["status"] == "passed"
    assert result["replay"][0]["rows"] == 2
    assert result["replay"][0]["max_absolute_difference"] == 0


def test_corruption_is_rejected_before_deserialization(bundle, monkeypatch):
    weight = bundle / "replay_test/rb_remaining/artifacts/weights.joblib"
    weight.write_bytes(b"corrupted model")
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("must not deserialize"))
    with pytest.raises(ValueError, match="checksum"):
        replay.replay_campaign(bundle, "replay_test")


@pytest.mark.parametrize(
    "key,value",
    [
        ("component_points", 999),
        ("points_per_scheduled_game", 999),
        ("rank", 1),
        ("scheduled_games", 14),
    ],
)
def test_replay_checks_derived_outputs_and_exposure(bundle, key, value):
    mutate(bundle, "forecast", lambda doc: doc["rows"][0].update({key: value}))
    with pytest.raises((ValueError, AssertionError)):
        replay.replay_campaign(bundle, "replay_test")


def test_replay_checks_feature_contract(bundle):
    mutate(bundle, "feature_contract", lambda doc: doc.reverse())
    with pytest.raises(ValueError, match="feature/schema"):
        replay.replay_campaign(bundle, "replay_test")


def test_replay_checks_receipt_scoring(bundle):
    mutate(bundle, "forecast", lambda doc: doc.update(scoring="Different scoring"))
    with pytest.raises(ValueError, match="scoring"):
        replay.replay_campaign(bundle, "replay_test")


def test_unknown_schedule_remains_null_and_cannot_gain_a_per_game_forecast(bundle):
    directory = bundle / "replay_test"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["models"][0]
    feature_ref = entry["artifacts"]["forecast_features"]
    feature_path = directory / feature_ref["path"]
    panel = pl.read_parquet(feature_path).with_columns(
        pl.when(pl.col("player_id") == "p1")
        .then(None)
        .otherwise(pl.col("scheduled_games"))
        .alias("scheduled_games")
    )
    panel.write_parquet(feature_path)
    feature_ref["sha256"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    forecast_ref = entry["forecasts"][0]
    forecast_path = directory / forecast_ref["path"]
    forecast = json.loads(forecast_path.read_text())
    forecast["feature_parquet_sha256"] = feature_ref["sha256"]
    forecast["rows"][0].update(scheduled_games=None, points_per_scheduled_game=None)
    forecast_path.write_text(json.dumps(forecast))
    forecast_ref["sha256"] = hashlib.sha256(forecast_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    (directory / "manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert replay.replay_campaign(bundle, "replay_test")["status"] == "passed"
    mutate(bundle, "forecast", lambda doc: doc["rows"][0].update(points_per_scheduled_game=10.0))
    with pytest.raises(ValueError, match="Unknown or zero"):
        replay.replay_campaign(bundle, "replay_test")


def test_forecast_only_campaign_explains_replay_limit_before_loading_weights(bundle, monkeypatch):
    path = bundle / "replay_test/manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(
        bundle_replay_supported=False, replay_note="Original QB estimators not retained"
    )
    path.write_text(json.dumps(manifest))
    (path.parent / "manifest.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("must not deserialize"))
    with pytest.raises(ValueError, match="Original QB estimators not retained"):
        replay.replay_campaign(bundle, "replay_test")
