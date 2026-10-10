"""Offline backend contracts; actual GPU fits use the separate smoke command."""

import json
import pickle
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.future_player_lab import gpu_backends as backends


@pytest.mark.parametrize("family", ["random_forest", "extra_trees", "hist"])
def test_cpu_fit_is_portable_serializable_and_records_effective_device(monkeypatch, family):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU estimator attempted to load an optional GPU dependency")

    monkeypatch.setattr(backends, "probe_cuda", forbidden)
    monkeypatch.setattr(backends, "_optional", forbidden)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(150, 4))
    y = x[:, 0] * 5
    model, metadata = backends.build_estimator(
        family, {"n_estimators": 30, "max_depth": 5}, seed=3, device="cpu", threads=1
    )
    assert metadata["effective_device"] is None
    model.fit(x[:120], y[:120])
    pred = model.predict(x[120:])
    assert np.mean((pred - y[120:]) ** 2) < np.mean((y[120:].mean() - y[120:]) ** 2)
    assert metadata["effective_device"] == "cpu"
    assert metadata["device_verified"] is True
    assert metadata["fit_seconds"] > 0
    json.dumps(metadata, allow_nan=False)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(x[120:]), pred)


@pytest.mark.parametrize("family", ["xgboost", "catboost", "cuml_rf"])
def test_explicit_gpu_request_fails_before_import_without_device(monkeypatch, family):
    def unavailable(ordinal):
        raise backends.BackendUnavailableError("No visible CUDA device")

    monkeypatch.setattr(backends, "probe_cuda", unavailable)
    with pytest.raises(backends.BackendUnavailableError, match="No visible CUDA device"):
        backends.build_estimator(family, device="cuda:2")


@pytest.mark.parametrize("family", ["hist", "extra_trees", "random_forest", "lightgbm"])
def test_gpu_request_never_silently_uses_cpu_family(family):
    with pytest.raises(backends.BackendUnavailableError, match="CPU-only"):
        backends.build_estimator(family, device="cuda")


def test_execution_controls_cannot_override_logged_device():
    with pytest.raises(ValueError, match="execution controls"):
        backends.build_estimator("xgboost", {"device": "cpu"}, device="cuda")


def test_xgboost_cpu_fallback_is_detected_after_fit():
    class FallingBack:
        def fit(self, x, y):
            return self

        def get_booster(self):
            return self

        def save_config(self):
            return json.dumps({"learner": {"generic_param": {"device": "cpu"}}})

    metadata = {"family": "xgboost", "configured_device": "cuda:0", "threads": 1}
    model = backends.TreeEstimator(FallingBack(), metadata)
    with pytest.raises(backends.BackendUnavailableError, match="fitted estimator used cpu"):
        model.fit(np.zeros((5, 2)), np.zeros(5))
    assert metadata["device_verified"] is False
    assert metadata["effective_device"] is None


def test_cuml_forwards_sample_weights_supported_by_pinned_release(monkeypatch):
    class Weighted:
        def fit(self, x, y, sample_weight):
            self.weight = sample_weight

    stream = SimpleNamespace(synchronize=lambda: None)
    cupy = SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda: stream))
    monkeypatch.setattr(backends, "_optional", lambda name: cupy)
    model = backends.TreeEstimator(Weighted(), {"family": "cuml_rf", "configured_device": "cuda:0"})
    monkeypatch.setattr(model, "_context", nullcontext)
    model.fit(np.zeros((5, 2)), np.zeros(5), sample_weight=[1, 2, 3, 4, 5])
    np.testing.assert_array_equal(model.estimator.weight, [1, 2, 3, 4, 5])
    assert model.estimator.weight.dtype == np.float64


def test_cuml_stream_count_is_explicit_and_recorded(monkeypatch):
    class Forest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    cupy = SimpleNamespace(cuda=SimpleNamespace(Device=lambda ordinal: nullcontext()))
    monkeypatch.setattr(
        backends, "probe_cuda", lambda ordinal: {"ordinal": ordinal, "compute_capability": "8.9"}
    )
    monkeypatch.setattr(
        backends,
        "_optional",
        lambda name: cupy if name == "cupy" else SimpleNamespace(RandomForestRegressor=Forest),
    )
    default, _ = backends.build_estimator("cuml_rf")
    assert default.estimator.kwargs["n_streams"] == 1
    parallel, metadata = backends.build_estimator("cuml_rf", {"n_streams": 4})
    assert parallel.estimator.kwargs["n_streams"] == metadata["effective_params"]["n_streams"] == 4
    with pytest.raises(ValueError, match="only supported"):
        backends.build_estimator("xgboost", {"n_streams": 4}, device="cpu")


def test_catboost_parameter_aliases_are_recorded(monkeypatch):
    class FakeCatBoost:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Module:
        CatBoostRegressor = FakeCatBoost

    monkeypatch.setattr(backends, "_optional", lambda name: Module)
    model, metadata = backends.build_estimator(
        "cat", {"n_estimators": 17, "max_depth": 4, "max_bin": 63}, device="cpu"
    )
    assert metadata["effective_params"]["iterations"] == 17
    assert metadata["effective_params"]["depth"] == 4
    assert model.estimator.kwargs["border_count"] == 63
    with pytest.raises(ValueError, match="Specify only one"):
        backends.build_estimator("cat", {"n_estimators": 17, "iterations": 20}, device="cpu")
