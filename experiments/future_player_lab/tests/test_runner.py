import json

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.models import recipes
from experiments.future_player_lab.run import task


def prepare(folder, change_future=False):
    from dataclasses import asdict

    folder.mkdir()
    rng = np.random.default_rng(123)
    rows = [
        {
            "player_id": f"p{i}",
            "name": f"Player {i}",
            "position": "QB",
            "year": year,
            "horizon": "season",
            "origin": 0,
            "end": 18,
            "exposure": 18,
            "population": "returning",
        }
        for year in range(2020, 2026)
        for i in range(35)
    ]
    panel = pl.DataFrame(rows)
    x = rng.normal(size=(len(rows), 2))
    y = np.column_stack(
        [
            10 + 2 * x[:, 0],
            30 + 3 * x[:, 0],
            100 + x[:, 1],
            np.ones(len(rows)),
            np.full(len(rows), 10),
        ]
    )
    if change_future:
        y[panel["year"].to_numpy() == 2025, 0] += 10000
    panel.write_parquet(folder / "panel.parquet")
    np.save(folder / "x.npy", x)
    np.save(folder / "y.npy", y)
    np.save(folder / "baseline.npy", np.ones_like(y))
    (folder / "features.json").write_text(json.dumps({"names": ["static:a", "summary:b"]}))
    config = {
        "evaluation_years": [2024, 2025],
        "inner_seasons": 1,
        "seed": 123,
        "threads_per_worker": 1,
        "interval_alpha": 0.2,
        "families": ["ridge"],
        "trials_per_family": 1,
    }
    return config, [asdict(r) for r in recipes(config)]


def test_outer_labels_cannot_change_forecasts_choices_or_current_intervals(tmp_path):
    folders = [tmp_path / "original", tmp_path / "perturbed"]
    frames, choices = [], []
    for folder, mutate in zip(folders, [False, True], strict=True):
        config, candidates = prepare(folder, mutate)
        task(str(folder), "QB", "season", config, candidates)
        base = folder / "tasks" / "QB_season"
        frames.append(pl.read_parquet(base / "predictions.parquet").filter(pl.col("year") == 2025))
        choices.append(json.loads((base / "choices.json").read_text()))
    columns = [c for c in frames[0].columns if not c.startswith("actual_")]
    assert frames[0].select(columns).equals(frames[1].select(columns))
    assert choices[0] == choices[1]
    assert frames[0].filter(pl.col("model") == "selected_mse")["low_points"].null_count() == 0


def test_resume_checks_complete_artifact_integrity(tmp_path):
    folder = tmp_path / "trial"
    config, candidates = prepare(folder)
    task(str(folder), "QB", "season", config, candidates)
    assert "cached" in task(str(folder), "QB", "season", config, candidates)
    path = folder / "tasks" / "QB_season" / "choices.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="artifacts changed"):
        task(str(folder), "QB", "season", config, candidates)
