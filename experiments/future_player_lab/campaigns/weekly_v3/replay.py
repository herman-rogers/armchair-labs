"""Offline replay of sealed v3 weekly weights, distributions, and preserved v2 rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from engine.chairstats.store import CampaignStore

from . import modeling
from .modeling import QCOLS, QUANTILES, predict_distribution, predict_mean


def rows_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


def replay_campaign(root, campaign_id="opening_drive_v3"):
    root = Path(root)
    store = CampaignStore(root)
    verification = store.check(campaign_id)
    manifest = store.campaign(campaign_id)
    receipts = []
    lineup_receipt = None
    for model in manifest["models"]:
        if model["horizon"] != "weekly":
            expected = model["original_forecast_rows_sha256"]
            for forecast in model["forecasts"]:
                rows = store.forecast(
                    campaign_id, model["model_id"], forecast["season"], forecast["origin_week"]
                )["rows"]
                key = f"{forecast['season']}_{forecast['origin_week']}"
                if rows_digest(rows) != expected[key]:
                    raise ValueError("Inherited v2 forecast values changed")
            receipts.append(dict(model_id=model["model_id"], status="original_rows_verified"))
            continue

        def content(name, model=model):
            ref = model["artifacts"][name]
            value = (root / campaign_id / ref["path"]).read_bytes()
            if hashlib.sha256(value).hexdigest() != ref["sha256"]:
                raise ValueError("Replay input changed after verification")
            return value

        if content("source_modeling") != Path(modeling.__file__).read_bytes():
            raise ValueError("Weekly inference code differs from the sealed source snapshot")
        training = json.loads(content("run_manifest"))
        for package, version in training["packages"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Replay package differs from training: {package}")
        if lineup_receipt is None:
            from engine.espn.sync import LeagueSnapshot
            from engine.metrics import weekly_distribution

            if content("source_lineup_runtime") != Path(weekly_distribution.__file__).read_bytes():
                raise ValueError("Lineup inference code differs from sealed source")
            case = json.loads(content("lineup_replay_case"))
            actual = weekly_distribution.forecast_snapshot(
                LeagueSnapshot.from_dict(case["snapshot"]),
                case["players"],
                {int(k): v for k, v in case["identities"].items()},
                case["schedule"],
                case["calibration"],
            )
            if rows_digest(actual) != rows_digest(case["expected"]):
                raise ValueError("Sealed lineup aggregation did not reproduce exactly")
            sides = [game[side] for game in actual["matchups"] for side in ("home", "away")]
            lineup_receipt = dict(status="passed", teams=len(sides), exact_json_match=True)
        # These are local, checksummed artifacts produced by this training pipeline.
        mean = pickle.loads(content("mean_model"))
        spread = pickle.loads(content("distribution_model"))
        panel = pd.read_parquet(io.BytesIO(content("forecast_matrix")))
        errors = []
        count = 0
        for ref in model["forecasts"]:
            week = ref["origin_week"] + 1
            matrix_name = ref.get("feature_matrix_artifact", "forecast_matrix")
            matrix = (
                panel
                if matrix_name == "forecast_matrix"
                else pd.read_parquet(io.BytesIO(content(matrix_name)))
            )
            frame = matrix[
                matrix.season.eq(ref["season"])
                & matrix.week.eq(week)
                & matrix.position.eq(model["position"])
            ].copy()
            scheduled = frame[frame.scheduled_game.eq(1)].copy()
            scheduled["prediction"] = predict_mean(mean, scheduled)
            q, std = predict_distribution(spread, scheduled)
            scheduled[QCOLS] = q
            scheduled["std"] = std
            byes = frame[frame.is_bye.eq(1)].copy()
            byes["prediction"], byes["std"] = 0.0, 0.0
            byes[QCOLS] = 0.0
            predicted = pd.concat([scheduled, byes]).set_index("player_id")
            rows = store.forecast(
                campaign_id, model["model_id"], ref["season"], ref["origin_week"]
            )["rows"]
            if set(predicted.index) != {r["player_id"] for r in rows}:
                raise ValueError("Replay forecast population differs")
            for row in rows:
                p = predicted.loc[row["player_id"]]
                expected = np.array(
                    [
                        row["predictions"]["points"],
                        row["distribution"]["std"],
                        *[row["distribution"][c] for c in QCOLS],
                    ]
                )
                actual = np.array([p.prediction, p["std"], *p[QCOLS]])
                np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
                samples = np.interp(
                    (np.arange(256) + 0.5) / 256, QUANTILES, p[QCOLS].to_numpy(dtype=float)
                )
                np.testing.assert_allclose(
                    samples, row["distribution"]["samples"], atol=1e-8, rtol=0
                )
                errors.append(float(np.max(np.abs(actual - expected))))
            count += len(rows)
        receipts.append(
            dict(
                model_id=model["model_id"],
                rows=count,
                max_absolute_difference=max(errors, default=0.0),
            )
        )
    return dict(
        status="passed",
        campaign_id=campaign_id,
        verified_bundle=verification,
        replay=receipts,
        lineup_replay=lineup_receipt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("data/campaigns"))
    parser.add_argument("--campaign", default="opening_drive_v3")
    args = parser.parse_args()
    print(json.dumps(replay_campaign(args.registry, args.campaign), indent=2))
