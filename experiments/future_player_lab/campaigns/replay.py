"""Replay predictions from a self-contained published campaign without raw data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from engine.chairstats import CampaignStore

from .data import ROOT, SCORING
from .run import predict
from .runtime import verify_environment


def replay_campaign(root, campaign_id):
    """Trust only a local sealed bundle; API requests never execute this operation."""
    root = Path(root)
    store = CampaignStore(root)
    verification = store.check(campaign_id)
    manifest = store.campaign(campaign_id)
    if manifest.get("bundle_replay_supported") is False:
        raise ValueError(manifest["replay_note"])
    receipts = []
    for entry in manifest["models"]:
        if len(entry["forecasts"]) != 1:
            raise ValueError("Bundle replay requires one explicitly bound feature matrix per model")
        artifacts = entry["artifacts"]
        directory = root / campaign_id

        def content(name, artifacts=artifacts, directory=directory):
            ref = artifacts[name]
            path = (directory / ref["path"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Replay artifact escaped campaign directory")
            result = path.read_bytes()
            if hashlib.sha256(result).hexdigest() != ref["sha256"]:
                raise ValueError(f"Replay artifact checksum changed: {name}")
            return result

        saved = json.loads(content("model_manifest"))
        names = json.loads(content("feature_contract"))
        if saved["schema_version"] != 1:
            raise ValueError("Unsupported saved model schema")
        for key in (
            "model_id",
            "position",
            "horizon",
            "origin_week",
            "targets",
            "training_through_season",
            "scoring",
        ):
            if saved[key] != entry[key]:
                raise ValueError(f"Saved model manifest differs from registry: {key}")
        if saved["campaign_id"] != campaign_id or entry["scoring"] != SCORING[entry["position"]]:
            raise ValueError("Campaign or scoring contract differs from supported inference")
        # Matrix replay does not rebuild raw features, so historical source files can
        # be archived. Numerical library versions must still match the training run.
        verify_environment(saved, check_sources=False)
        # Deserialize the exact bytes just verified, avoiding a second filesystem
        # read between checksum verification and trusted local joblib execution.
        model = joblib.load(io.BytesIO(content("weights")))
        if model["schema_version"] != 1 or model["feature_names"] != names:
            raise ValueError("Saved weights differ from feature/schema contract")
        for key in (
            "position",
            "horizon",
            "origin_week",
            "targets",
            "training_through_season",
            "scoring",
        ):
            if model[key] != entry[key]:
                raise ValueError(f"Saved weights differ from model contract: {key}")
        panel = pl.read_parquet(io.BytesIO(content("forecast_features")))
        ref = entry["forecasts"][0]
        if set(panel["year"].to_list()) != {ref["season"]}:
            raise ValueError("Saved feature season differs from forecast receipt")
        if panel["player_id"].n_unique() != panel.height:
            raise ValueError("Saved features contain duplicate player identifiers")
        outputs = predict(model, panel)
        forecast = store.forecast(campaign_id, entry["model_id"], ref["season"], ref["origin_week"])
        metadata = forecast["metadata"]
        if metadata.get("scoring") != entry["scoring"]:
            raise ValueError("Forecast receipt scoring differs from model contract")
        for key, artifact in (
            ("model_manifest_sha256", "model_manifest"),
            ("feature_parquet_sha256", "forecast_features"),
        ):
            if metadata.get(key) != artifacts[artifact]["sha256"]:
                raise ValueError(f"Forecast receipt artifact binding differs: {key}")
        by_id = {r["player_id"]: r for r in forecast["rows"]}
        if set(by_id) != set(panel["player_id"].to_list()):
            raise ValueError("Saved features and forecast populations differ")
        errors = []
        ranks = np.empty(panel.height, dtype=int)
        ranks[np.argsort(-outputs["points"], kind="stable")] = np.arange(1, panel.height + 1)
        for index, player_id in enumerate(panel["player_id"]):
            row = by_id[player_id]
            if row["rank"] != int(ranks[index]):
                raise ValueError(
                    "Saved forecast ranking differs from stable direct points ordering"
                )
            recorded_games = panel["scheduled_games"][index]
            games = int(recorded_games) if recorded_games is not None else None
            if row["scheduled_games"] != games or row["calendar_weeks"] != int(
                panel["weeks"][index]
            ):
                raise ValueError("Forecast exposure differs from saved features")
            for target in entry["targets"]:
                actual, expected = outputs[target][index], row["predictions"][target]
                np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
                errors.append(abs(float(actual) - expected))
            np.testing.assert_allclose(
                outputs["component_points"][index], row["component_points"], atol=1e-7, rtol=1e-7
            )
            if games:
                np.testing.assert_allclose(
                    outputs["points"][index] / games,
                    row["points_per_scheduled_game"],
                    atol=1e-7,
                    rtol=1e-7,
                )
            elif row["points_per_scheduled_game"] is not None:
                raise ValueError("Unknown or zero scheduled games must have no per-game forecast")
        receipts.append(
            {
                "model_id": entry["model_id"],
                "rows": panel.height,
                "targets": len(entry["targets"]),
                "max_absolute_difference": max(errors),
            }
        )
    return {
        "campaign_id": campaign_id,
        "verified_bundle": verification,
        "replay": receipts,
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "data/campaigns")
    parser.add_argument("--campaign", default="opening_drive_v1")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        print(json.dumps(replay_campaign(args.registry, args.campaign), indent=2))


if __name__ == "__main__":
    main()
