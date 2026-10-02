#!/usr/bin/env python3
"""Read-only audit of saved forecast pairing and chronological model choices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["protected_inputs_unchanged"] and manifest["publication_unchanged"]
    for name, expected in manifest["files"].items():
        assert digest(root / name) == expected, name
    for parent, expected in manifest.get("parents", {}).items():
        assert digest(root.parent / parent / "manifest.json") == expected
        verify(root.parent / parent)
    df = pl.read_parquet(root / "predictions.parquet")
    keys = ["position", "season", "player_id", "forecast_cutoff_date"]
    assert not df.select(*keys, "model").is_duplicated().any()
    base = df.filter(pl.col("model") == "basic_tree").select(*keys, "actual")
    for model in df["model"].unique():
        block = df.filter(pl.col("model") == model).select(*keys, "actual")
        assert block.sort(keys).equals(base.sort(keys)), model
    assert df["prediction"].is_finite().all() and df["actual"].is_finite().all()
    assert df["season"].max() <= 2025
    annual = df.group_by("position", "season", "model").agg(
        (pl.col("prediction") - pl.col("actual")).pow(2).mean().alias("mse")
    )
    losses = {(r["position"], r["season"], r["model"]): r["mse"] for r in annual.to_dicts()}
    folds = json.loads((root / "folds.json").read_text())
    checked_losses = 0
    for fold in folds:
        year, pos, winner = fold["season"], fold["position"], fold["chosen_model"]
        assert fold["train_min"] == 2004 and fold["train_max"] == year - 1
        assert fold["inner_years"] == list(range(year - 3, year))
        scores = fold["validation_mse"]
        assert winner == min(scores, key=lambda name: (scores[name], name))
        for model, score in scores.items():
            if all((pos, y, model) in losses for y in fold["inner_years"]):
                expected = np.mean([losses[pos, y, model] for y in fold["inner_years"]])
                assert np.isclose(score, expected, rtol=1e-12), (pos, year, model)
                checked_losses += 1
        current = df.filter((pl.col("position") == pos) & (pl.col("season") == year))
        selected = current.filter(pl.col("model") == fold["search"]).sort(keys)
        fixed = current.filter(pl.col("model") == winner).sort(keys)
        assert selected["prediction"].equals(fixed["prediction"]), (pos, year, winner)
        assert selected.height == fold["test_n"]
    result = {
        "run": root.name,
        "unique_forecasts": base.height,
        "models": df["model"].n_unique(),
        "position_seasons": base.select("position", "season").unique().height,
        "selection_decisions_checked": len(folds),
        "validation_losses_recomputed": checked_losses,
        "passed": True,
    }
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run)
