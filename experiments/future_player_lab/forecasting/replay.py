"""Verify a forecast run's hashes and reproduce predictions from its saved features."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from ..gpu_lab.refit import predict_policy
from ..gpu_lab.run import ROOT, digest
from .data import feature_matrix
from .registry import load_registered_policy, model_contract
from .run import POLICIES, rank_predictions


def replay(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Forecast run is incomplete")
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Forecast artifact changed: {name}")
    for name, expected in manifest["sources"].items():
        # Output rendering/capture and this audit wrapper do not implement model
        # inference. Their archived copies are still verified above; presentation
        # edits must not invalidate an otherwise reproducible saved forecast.
        if name in {
            "experiments/future_player_lab/forecasting/report.py",
            "experiments/future_player_lab/forecasting/capture.py",
            "experiments/future_player_lab/forecasting/replay.py",
        }:
            continue
        if digest(ROOT / name) != expected:
            raise ValueError(f"Forecast implementation changed: {name}; use the saved source copy")
    spec = json.loads((directory / "model.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    _, names = model_contract(spec, summary["season"], summary["origin"])
    panel = pl.read_parquet(directory / "features.parquet")
    x = feature_matrix(panel, names)
    predictions = {}
    with threadpool_limits(limits=4):
        for policy in POLICIES:
            predictions[policy] = predict_policy(
                load_registered_policy(spec, policy), x, names, panel["weeks"].to_numpy()
            )
    actual = rank_predictions(panel, predictions, spec["default_policy"])
    expected = pl.read_parquet(directory / "rankings.parquet")
    assert actual["player_id"].equals(expected["player_id"]), "Rank order changed"
    columns = [f"ppr_{p}" for p in POLICIES] + ["predicted_ppr"]
    np.testing.assert_allclose(
        actual.select(columns).to_numpy(), expected.select(columns).to_numpy(), rtol=1e-6, atol=1e-5
    )
    result = {
        "status": "passed",
        "rows": actual.height,
        "policies": len(POLICIES),
        "verified_files": len(manifest["files"]),
    }
    print(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    replay(parser.parse_args().directory)


if __name__ == "__main__":
    main()
