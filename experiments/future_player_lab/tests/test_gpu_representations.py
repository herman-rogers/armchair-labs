"""Validate missingness semantics and fold isolation of unsupervised discovery."""

import json
import pickle

import numpy as np
import pytest
from experiments.future_player_lab.gpu_lab.representations import (
    KINDS,
    fit_representation,
)


def test_raw_retains_zero_versus_absent_without_learning_test_eligibility():
    train = np.array([[0, 3, np.nan], [2, 3, np.nan], [np.nan, 3, np.nan], [4, 3, np.nan]])
    model = fit_representation(train, ["production", "constant", "future_only"], {"kind": "raw"})
    before = pickle.dumps(model)
    actual = model.transform([[0, 99, 8], [np.nan, -99, -8], [10000, 3, 1]])
    assert model.get_feature_names_out().tolist() == ["production", "missing:production"]
    np.testing.assert_array_equal(actual, [[0, 0], [2, 1], [10000, 0]])
    assert pickle.dumps(model) == before
    assert model.summary()["constant_in_training"] == ["constant", "future_only"]


def test_constant_observed_value_keeps_informative_missingness():
    model = fit_representation([[7], [np.nan], [7]], ["sparse"], {"kind": "raw"})
    assert model.get_feature_names_out().tolist() == ["missing:sparse"]
    np.testing.assert_array_equal(model.transform([[7], [np.inf], [0]]), [[0], [1], [0]])


@pytest.mark.parametrize("kind", KINDS)
def test_representations_are_finite_deterministic_and_serializable(kind):
    rng = np.random.default_rng(131)
    train = rng.normal(size=(80, 20)).astype(np.float32)
    train[::4, 1] = np.nan
    train[:, 18] = train[:, 0]
    train[:, 19] = 0
    names = [f"prior__box__{i}" for i in range(train.shape[1])]
    spec = {"kind": kind, "n_components": 4, "max_fit_rows": 30}
    first = fit_representation(train, names, spec, seed=51)
    second = fit_representation(train, names, spec, seed=51)
    test = rng.normal(size=(10, 20))
    test[::2, 1] = np.nan
    transformed = first.transform(test)
    assert transformed.shape == (10, len(first.get_feature_names_out()))
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed, second.transform(test))
    np.testing.assert_allclose(transformed, pickle.loads(pickle.dumps(first)).transform(test))
    assert json.loads(json.dumps(first.summary()))["fit_uses_targets"] is False


def test_correlation_removes_only_training_redundancy_and_preserves_sign_relatives():
    rng = np.random.default_rng(71)
    a = rng.normal(size=90)
    b = rng.normal(size=90)
    train = np.column_stack([a, a * 10, -a, b])
    model = fit_representation(
        train,
        ["a", "scaled_a", "negative_a", "b"],
        {"kind": "correlation", "correlation_threshold": 0.99},
    )
    assert model.get_feature_names_out().tolist() == ["a", "b"]
    assert model.transform([[1, 200, 300, 2]]).tolist() == [[1, 2]]


def test_raw_pca_preserves_the_entire_raw_anchor():
    rng = np.random.default_rng(3)
    train = rng.normal(size=(30, 7))
    train[:4, 2] = np.nan
    names = [f"f{i}" for i in range(7)]
    raw = fit_representation(train, names, {"kind": "raw"})
    combined = fit_representation(train, names, {"kind": "raw_pca", "n_components": 3})
    np.testing.assert_array_equal(combined.transform(train)[:, :8], raw.transform(train))
    assert combined.transform(train).shape[1] == 11


def test_source_balancing_is_invariant_to_uniform_duplication_inside_one_source():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(70, 4))
    names = ["prior__box__a", "prior__box__b", "prior__pbp_rush__a", "prior__pbp_rush__b"]
    original = fit_representation(x, names, {"kind": "group_pca", "n_components": 4})
    more = np.column_stack([x, x[:, 2:]])
    duplicate = fit_representation(
        more,
        [*names, "current__pbp_rush__a", "current__pbp_rush__b"],
        {"kind": "group_pca", "n_components": 4},
    )
    a, b = original.transform(x), duplicate.transform(more)
    np.testing.assert_allclose(a @ a.T, b @ b.T, atol=1e-5)
    assert original.summary()["source_balance_groups"] == {"box": 2, "plays": 2}


@pytest.mark.parametrize("kind", KINDS)
def test_all_constant_or_missing_training_inputs_have_explicit_intercept_fallback(kind):
    model = fit_representation([[1, np.nan], [1, np.nan]], ["one", "empty"], {"kind": kind})
    np.testing.assert_array_equal(model.transform([[55, 9]]), [[0]])
    assert model.summary()["degenerate_training_inputs"] is True


def test_feature_cluster_summary_maps_all_varying_features_to_native_representatives():
    rng = np.random.default_rng(5)
    factors = rng.normal(size=(60, 3))
    train = np.column_stack([factors[:, i % 3] + rng.normal(0, 0.01, 60) for i in range(18)])
    names = [f"x{i}" for i in range(18)]
    model = fit_representation(train, names, {"kind": "cluster", "n_components": 3})
    clusters = model.summary()["cluster_representatives"]
    assert sorted(n for c in clusters for n in c["members"]) == sorted(names)
    assert sorted(c["representative"] for c in clusters) == sorted(model.get_feature_names_out())
    assert all(c["representative"] in c["members"] for c in clusters)


def test_invalid_contracts_fail_instead_of_silently_changing_representation():
    with pytest.raises(ValueError, match="Unknown representation options"):
        fit_representation([[0], [1]], ["a"], {"kind": "raw", "top_k": 3})
    with pytest.raises(ValueError, match="unique"):
        fit_representation([[0, 1], [1, 2]], ["a", "a"], {"kind": "raw"})
    model = fit_representation([[0], [1]], ["a"], {"kind": "raw"})
    with pytest.raises(ValueError, match="feature order"):
        model.transform([[0, 1]])
