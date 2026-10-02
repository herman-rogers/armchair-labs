"""Proper-score identities, forecast support, and independence from test outcomes."""

import numpy as np
import pytest
from experiments.future_player_lab.notebooks.rb_probability import (
    constrain,
    empirical_crps,
    evaluate,
    interval,
    residual_distribution,
)


def test_crps_matches_brute_force_definition():
    rng = np.random.default_rng(14)
    draws, actual = rng.normal(size=(9, 13)), rng.normal(size=9)
    expected = np.mean(abs(draws - actual[:, None]), axis=1) - 0.5 * np.mean(
        abs(draws[:, :, None] - draws[:, None, :]), axis=(1, 2)
    )
    np.testing.assert_allclose(empirical_crps(actual, draws), expected, atol=1e-14)
    np.testing.assert_allclose(empirical_crps(actual, draws[:, ::-1]), expected)


def test_known_scores_and_point_mass_limit():
    np.testing.assert_allclose(empirical_crps([1], [[0, 2]]), [0.5])
    np.testing.assert_allclose(empirical_crps([0, 3], [[0, 0], [2, 2]]), [0, 1])
    # Honest uncertainty beats a confidently wrong forecast; needless width costs.
    assert empirical_crps([0], [[0, 100]])[0] == 25
    assert empirical_crps([0], [[100, 100]])[0] == 100
    assert empirical_crps([0], [[-100, 100]])[0] == 50


def test_score_translation_and_scale():
    y, x = np.array([2.0]), np.array([[0.0, 1.0, 3.0]])
    np.testing.assert_allclose(empirical_crps(y + 10, x + 10), empirical_crps(y, x))
    np.testing.assert_allclose(empirical_crps(4 * y, 4 * x), 4 * empirical_crps(y, x))


def test_neighborhood_uses_predicted_rate_and_counts_clip_but_yards_do_not():
    args = ([1], [10], [0, 1, 100], [-2, 3, 300])
    signed = residual_distribution(*args, neighbors=2, count_stat=False)
    np.testing.assert_allclose(signed, [[30, -10]])
    counts = residual_distribution(*args, neighbors=2, count_stat=True)
    np.testing.assert_allclose(counts, [[30, 0]])
    # Small calibration sets use available rows without sampling duplicates.
    assert residual_distribution(*args, neighbors=100).shape == (1, 3)


def test_absences_create_identical_point_mass_and_do_not_mutate_raw_forecasts():
    raw = np.array([[1, 2, 3], [-1, 3, 8]])
    adjusted = constrain(raw, [True, False])
    np.testing.assert_array_equal(raw[0], [1, 2, 3])
    np.testing.assert_array_equal(adjusted[0], [0, 0, 0])
    np.testing.assert_array_equal(adjusted[1], raw[1])
    assert empirical_crps([7], adjusted[:1])[0] == 7  # Do not hide a failed zero rule.


def test_scores_change_but_distributions_do_not_when_test_outcomes_change():
    args = dict(
        model_rate=[2, 4],
        baseline_rate=[1, 3],
        exposure=[10, 10],
        calibration_model=[1, 3, 4],
        calibration_baseline=[2, 4, 5],
        calibration_actual=[0, 2, 6],
        force_zero=[False, True],
        target="carries",
        neighbors=3,
    )
    a, b = evaluate(np.array([5, 0]), **args), evaluate(np.array([90, 20]), **args)
    for name in a["samples"]:
        np.testing.assert_array_equal(a["samples"][name], b["samples"][name])
    assert not a["scores"].equals(b["scores"])


def test_intervals_are_quantiles_of_the_reported_distribution():
    low, high = interval(np.arange(1, 101)[None, :], 0.8)
    assert low[0] == 10 and high[0] == 90


@pytest.mark.parametrize(
    "actual,draws", [([1], [[]]), ([1], [[np.nan]]), ([np.nan], [[1]]), ([1, 2], [[1]])]
)
def test_invalid_scores_fail_explicitly(actual, draws):
    with pytest.raises(ValueError):
        empirical_crps(actual, draws)
