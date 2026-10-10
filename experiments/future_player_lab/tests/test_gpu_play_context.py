"""Handoff-only source features, directional geometry, selection and empirical scores."""

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.gpu_lab import play_context as play


def synthetic_play():
    return pl.DataFrame(
        {
            "GameId": [2017090700] * 22,
            "PlayId": [100] * 22,
            "NflId": list(range(22)),
            "NflIdRusher": [0] * 22,
            "Team": ["home"] * 11 + ["away"] * 11,
            "Position": ["RB"] + ["OL"] * 10 + ["LB"] * 11,
            "Season": [2017] * 22,
            "Yards": [5] * 22,
            "X": [40.0 + i / 2 for i in range(22)],
            "Y": [20.0 + i / 3 for i in range(22)],
            "S": [3.0 + i / 20 for i in range(22)],
            "A": [1.0] * 22,
            "Dis": [0.2] * 22,
            "Dir": [90.0] * 22,
            "Orientation": [70.0] * 22,
            "PlayDirection": ["right"] * 22,
            "PlayerWeight": [220] * 22,
            "TimeSnap": ["2017-09-07T20:00:00.000Z"] * 22,
            "TimeHandoff": ["2017-09-07T20:00:01.000Z"] * 22,
            "PlayerBirthDate": ["01/15/1995"] * 22,
            "PlayerHeight": ["6-0"] * 22,
        }
    )


def test_context_features_never_use_yards_or_ids_as_predictors():
    source = synthetic_play()
    first, contract = play.build_features(source)
    changed, other = play.build_features(source.with_columns(pl.lit(99).alias("Yards")))
    columns = contract["numeric"] + contract["categorical"]
    assert contract == other
    assert first.select(columns).equals(changed.select(columns))
    assert first["actual"][0] == 5 and changed["actual"][0] == 99
    assert not set(columns) & {"Yards", "actual", "NflId", "GameId", "PlayId", "Season"}
    assert first["teammates__count"][0] == 10
    assert first["defenders__count"][0] == 11
    assert first["snap_to_handoff_seconds"][0] == 1


def test_mirrored_direction_preserves_attack_normalized_geometry():
    source = synthetic_play()
    first, contract = play.build_features(source)
    mirrored = source.with_columns(
        (120 - pl.col("X")).alias("X"),
        (160 / 3 - pl.col("Y")).alias("Y"),
        ((pl.col("Dir") + 180) % 360).alias("Dir"),
        ((pl.col("Orientation") + 180) % 360).alias("Orientation"),
        pl.lit("left").alias("PlayDirection"),
    )
    second, _ = play.build_features(mirrored)
    np.testing.assert_allclose(
        first.select(contract["numeric"]).to_numpy(),
        second.select(contract["numeric"]).to_numpy(),
        equal_nan=True,
        atol=1e-10,
    )


def test_duplicate_or_incomplete_snapshot_rejected():
    with pytest.raises(ValueError, match="22 players"):
        play.build_features(synthetic_play().head(21))
    with pytest.raises(ValueError, match="Duplicate"):
        play.build_features(pl.concat([synthetic_play(), synthetic_play().head(1)]))


def test_exact_residual_crps_matches_explicit_pairwise_formula():
    observed = np.array([-3.0, 0.0, 7.0, 20.0])
    pred = np.array([1.0, 2.0, 3.0, 10.0])
    cal_pred = np.array([1.0, 3.0, 4.0, 5.0, 6.0])
    cal_actual = np.array([-2.0, 1.0, 4.0, 8.0, 18.0])
    samples = pred[:, None] + (cal_actual - cal_pred)[None, :]
    exact = np.abs(samples - observed[:, None]).mean(axis=1)
    exact -= 0.5 * np.abs(samples[:, :, None] - samples[:, None, :]).mean(axis=(1, 2))
    result = play.residual_scores(observed, pred, cal_actual, cal_pred)
    np.testing.assert_allclose(result["crps"], exact, atol=1e-12)
    changed = play.residual_scores(observed + 100, pred, cal_actual, cal_pred)
    np.testing.assert_array_equal(result["lower80"], changed["lower80"])
    np.testing.assert_array_equal(result["upper80"], changed["upper80"])


def test_validation_selection_tie_is_stable():
    rows = [
        {"config_index": 2, "mse": 4.0},
        {"config_index": 1, "mse": 4.0},
        {"config_index": 0, "mse": 5.0},
    ]
    assert play.select_recipe(rows) == 1
