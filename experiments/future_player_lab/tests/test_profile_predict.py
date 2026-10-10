"""Profile predictive means respect fixed priors, soft memberships and label masks."""

import numpy as np
import pytest
from experiments.future_player_lab.gpu_lab.profile_predict import ProfileMeans, recipe_library


def test_hard_profile_means_shrink_by_actual_observed_target_counts():
    memberships = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float)
    y = np.array([[0, np.nan], [2, 4], [8, 10], [10, 12]], float)
    plain = ProfileMeans(0).fit(memberships, y)
    np.testing.assert_allclose(plain.cluster_mean_, [[1, 4], [9, 11]])
    shrunk = ProfileMeans(2).fit(memberships, y)
    np.testing.assert_allclose(shrunk.cluster_mean_[:, 0], [3, 7])
    np.testing.assert_allclose(
        shrunk.cluster_mean_[:, 1], [(4 + 2 * 26 / 3) / 3, (22 + 2 * 26 / 3) / 4]
    )
    np.testing.assert_array_equal(shrunk.weighted_count_, [[2, 1], [2, 2]])


def test_soft_memberships_produce_weighted_means_and_effective_counts():
    weights = np.array([[0.8, 0.2], [0.2, 0.8]])
    model = ProfileMeans(0).fit(weights, [[0], [10]])
    np.testing.assert_allclose(model.cluster_mean_, [[2], [8]])
    np.testing.assert_allclose(model.predict([[0.5, 0.5]]), [[5]])
    np.testing.assert_allclose(model.effective_count_, np.full((2, 1), 1 / 0.68))


def test_empty_hard_cluster_uses_global_mean_without_fabricated_zero():
    model = ProfileMeans(0).fit([[1, 0], [1, 0]], [[2], [6]])
    np.testing.assert_allclose(model.predict([[0, 1]]), [[4]])


def test_invalid_probabilities_unknown_labels_and_priors_fail_explicitly():
    with pytest.raises(ValueError, match="sum to one"):
        ProfileMeans().fit([[0.8, 0.8]], [[1]])
    with pytest.raises(ValueError, match="observed training label"):
        ProfileMeans().fit([[1]], [[np.nan]])
    with pytest.raises(ValueError, match="nonnegative"):
        ProfileMeans(-1)
    recipes = recipe_library()
    assert len(recipes) == 12
    assert len({r["id"] for r in recipes}) == 12
