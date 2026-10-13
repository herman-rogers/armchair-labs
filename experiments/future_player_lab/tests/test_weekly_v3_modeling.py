import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v3.modeling import (
    QCOLS,
    QUANTILES,
    _quantile_moments,
    feature_columns,
    fit_distribution,
    fit_mean,
    predict_distribution,
    predict_mean,
    score_predictions,
    select_candidate,
)


def calibration():
    rng = np.random.default_rng(4)
    n = 1000
    return pd.DataFrame(
        {
            "season": np.repeat([2017, 2018, 2019, 2020], n // 4),
            "player_id": np.tile(np.arange(25).astype(str), n // 25),
            "prediction": rng.uniform(3, 20, n),
            "target_points": rng.normal(12, 5, n),
            "player_history_variance": rng.uniform(2, 15, n),
            "usage": rng.uniform(0, 20, n),
        }
    )


def test_cutoff_rejects_contemporaneous_outcomes():
    cal = calibration()
    with pytest.raises(ValueError, match="cutoff"):
        fit_mean(cal, ["usage"], "lgb_direct_full", 2020)
    with pytest.raises(ValueError, match="earlier-season"):
        fit_distribution(cal, ["usage"], "pooled", 2020)


def test_selection_never_uses_evaluation_years_and_weights_years_equally():
    scores = pd.DataFrame(
        {
            "season": [2018, 2019, 2020, 2021] * 2,
            "method": ["a"] * 4 + ["b"] * 4,
            "rmse": [1, 1, 1, 99, 2, 2, 2, 0],
        }
    )
    assert select_candidate(scores, "rmse") == "a"


def test_variance_ablation_and_no_v2_are_explicit():
    cols = ["usage", "player_history_variance", "v2_weekly_mean", "v2_exact_available"]
    assert feature_columns(cols, "conditional_no_variability") == [
        "usage",
        "v2_weekly_mean",
        "v2_exact_available",
    ]
    assert feature_columns(cols, "lgb_direct_no_v2") == ["usage", "player_history_variance"]


@pytest.mark.parametrize(
    "method",
    [
        "pooled",
        "mean_bin",
        "player_shrinkage",
        "conditional_scale",
        "conditional_no_variability",
        "quantile",
    ],
)
def test_distribution_replays_orders_quantiles_and_personalizes_unseen_players(method):
    cal = calibration()
    fit = fit_distribution(cal, ["usage", "player_history_variance"], method, 2021)
    future = cal.iloc[:5].copy()
    future["player_id"] = "rookie"
    q, std = predict_distribution(fit, future)
    q2, std2 = predict_distribution(fit, future)
    assert np.isfinite(q).all() and np.isfinite(std).all()
    assert (std > 0).all() and (np.diff(q, axis=1) >= 0).all()
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_array_equal(std, std2)
    future[QCOLS] = q
    scores = score_predictions(future)
    assert scores["n"] == 5 and 0 <= scores["coverage_80"] <= 1


def test_mean_bias_does_not_inflate_pooled_variance():
    cal = calibration()
    first = fit_distribution(cal, ["usage"], "pooled", 2021)
    cal.target_points += 100
    second = fit_distribution(cal, ["usage"], "pooled", 2021)
    assert second["bias"] - first["bias"] == pytest.approx(100)
    assert first["pooled_std"] == pytest.approx(second["pooled_std"])
    np.testing.assert_allclose(first["quantiles"], second["quantiles"])


def test_recent_baseline_does_not_change_when_proxy_becomes_exact_v2():
    frame = pd.DataFrame(
        {
            "recent4_points_mean": [10.0],
            "player_history_mean": [8.0],
            "player_history_count": [20.0],
            "v2_weekly_mean": [5.0],
        }
    )
    expected = predict_mean({"method": "recent"}, frame)
    frame.v2_weekly_mean = 25.0
    np.testing.assert_array_equal(expected, predict_mean({"method": "recent"}, frame))


def test_moments_match_linearly_interpolated_quantile_samples():
    mean, std = _quantile_moments(QUANTILES[None, :])
    expected_second = 0.01 * 0.01**2 + (0.99**3 - 0.01**3) / 3 + 0.01 * 0.99**2
    assert mean[0] == pytest.approx(0.5)
    assert std[0] ** 2 == pytest.approx(expected_second - 0.25)
    _, zero_std = _quantile_moments(np.zeros((1, len(QUANTILES))))
    assert zero_std[0] == 0
