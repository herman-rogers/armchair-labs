"""Optional, explicit tree backends for reproducible local research.

Importing this module never imports CUDA or optional tree libraries.  GPU requests
fail instead of silently using a CPU.  Keep the returned metadata with each fit;
``fit`` updates it with the verified effective device and timing.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata
import json
import platform
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np


class BackendUnavailableError(RuntimeError):
    """A requested compute backend cannot execute on this machine."""


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _optional(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as exc:
        raise BackendUnavailableError(
            f"Cannot load {module}: {exc}. Install the research extra; for cuML "
            "also install the research-gpu extra on supported NVIDIA/Linux hardware."
        ) from exc


def _device(device: str) -> tuple[str, int | None]:
    if device == "cpu":
        return "cpu", None
    if device in {"cuda", "gpu"}:
        return "cuda:0", 0
    if device.startswith("cuda:") and device[5:].isdigit():
        ordinal = int(device[5:])
        return f"cuda:{ordinal}", ordinal
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<ordinal>'")


def probe_cuda(ordinal: int = 0) -> dict[str, Any]:
    """Query the driver without requiring CuPy, respecting CUDA_VISIBLE_DEVICES."""
    try:
        driver = ctypes.CDLL("nvcuda.dll" if platform.system() == "Windows" else "libcuda.so.1")

        def check(result: int, operation: str) -> None:
            if result:
                label = ctypes.c_char_p()
                driver.cuGetErrorName(result, ctypes.byref(label))
                name = label.value.decode() if label.value else "unknown"
                raise BackendUnavailableError(
                    f"CUDA {operation} failed: {name} (driver error {result})"
                )

        check(driver.cuInit(0), "initialization")
        count = ctypes.c_int()
        check(driver.cuDeviceGetCount(ctypes.byref(count)), "device count")
        if ordinal < 0 or ordinal >= count.value:
            raise BackendUnavailableError(
                f"Requested CUDA device {ordinal}, but {count.value} devices are visible"
            )
        handle = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(handle), ordinal), "device lookup")
        name = ctypes.create_string_buffer(256)
        check(driver.cuDeviceGetName(name, len(name), handle), "device name")
        memory = ctypes.c_size_t()
        check(driver.cuDeviceTotalMem_v2(ctypes.byref(memory), handle), "device memory")
        major, minor, version = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        check(
            driver.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), handle),
            "compute capability",
        )
        check(driver.cuDriverGetVersion(ctypes.byref(version)), "driver version")
        try:
            system_driver = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                .stdout.strip()
                .splitlines()[0]
            )
        except (OSError, subprocess.SubprocessError, IndexError):
            system_driver = None
        return {
            "ordinal": ordinal,
            "name": name.value.decode(),
            "total_memory_bytes": memory.value,
            "compute_capability": f"{major.value}.{minor.value}",
            "driver_cuda_version": version.value,
            "nvidia_driver_version": system_driver,
            "visible_device_count": count.value,
        }
    except (OSError, AttributeError) as exc:
        raise BackendUnavailableError(f"NVIDIA CUDA driver unavailable: {exc}") from exc


class TreeEstimator:
    """Small serializable adapter with NumPy input/output and device verification.

    The native estimator remains available as ``estimator``. For cuML, imputing
    missing inputs is the caller's responsibility and must use training data only.
    """

    def __init__(self, estimator: Any, metadata: dict[str, Any]):
        self.estimator = estimator
        self.metadata = metadata

    def __getattr__(self, name: str) -> Any:
        # Avoid recursion while pickle is restoring instance attributes.
        estimator = self.__dict__.get("estimator")
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)

    def _context(self) -> Any:
        from threadpoolctl import threadpool_limits

        stack = ExitStack()
        stack.enter_context(threadpool_limits(limits=self.metadata["threads"]))
        try:
            if self.metadata["family"] == "cuml_rf":
                stack.enter_context(_optional("cupy").cuda.Device(self.metadata["gpu"]["ordinal"]))
        except Exception:
            stack.close()
            raise
        return stack

    def _array(self, values: Any) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32, order="C")
        if self.metadata["family"] == "cuml_rf" and not np.isfinite(result).all():
            raise ValueError(
                "cuML random forest requires finite inputs; fit an imputer on train only"
            )
        return result

    def fit(self, x: Any, y: Any, **kwargs: Any) -> TreeEstimator:
        self.metadata["device_verified"] = False
        self.metadata["effective_device"] = None
        if kwargs.get("sample_weight") is not None:
            kwargs["sample_weight"] = np.asarray(kwargs["sample_weight"], dtype=np.float64)
        start = time.perf_counter()
        with self._context():
            self.estimator.fit(self._array(x), self._array(y), **kwargs)
            if self.metadata["family"] == "cuml_rf":
                _optional("cupy").cuda.get_current_stream().synchronize()
        effective = self.metadata["configured_device"]
        if self.metadata["family"] == "xgboost":
            config = json.loads(self.estimator.get_booster().save_config())
            effective = config["learner"]["generic_param"]["device"]
        if effective != self.metadata["configured_device"]:
            raise BackendUnavailableError(
                f"Requested {self.metadata['configured_device']}, "
                f"but fitted estimator used {effective}"
            )
        self.metadata.update(
            effective_device=effective,
            device_verified=True,
            fit_seconds=time.perf_counter() - start,
        )
        return self

    def predict(self, x: Any, **kwargs: Any) -> np.ndarray:
        with self._context():
            prediction = self.estimator.predict(self._array(x), **kwargs)
            if hasattr(prediction, "get"):
                prediction = prediction.get()
        return np.asarray(prediction)


def build_estimator(
    family: str,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    device: str = "cuda",
    threads: int = 4,
) -> tuple[TreeEstimator, dict[str, Any]]:
    """Construct a tree regressor and its JSON-serializable execution record.

    Families: xgboost, catboost, cuml_rf, lightgbm, random_forest, extra_trees,
    hist. The last four currently support CPU only. Parameters are native model
    parameters except that CatBoost also accepts n_estimators/max_depth/max_bin.
    Device, seed and CPU thread controls belong in explicit arguments.
    """
    requested_family = family
    family = {
        "xgb": "xgboost",
        "cat": "catboost",
        "lgb": "lightgbm",
        "forest": "random_forest",
        "rf": "random_forest",
        "extra": "extra_trees",
        "cuml": "cuml_rf",
        "rapids_rf": "cuml_rf",
    }.get(family, family)
    supported = {
        "xgboost",
        "catboost",
        "cuml_rf",
        "lightgbm",
        "random_forest",
        "extra_trees",
        "hist",
    }
    if family not in supported:
        raise ValueError(f"Unknown tree family {family!r}; choose from {sorted(supported)}")
    if threads < 1:
        raise ValueError("threads must be a positive integer")
    configured, ordinal = _device(device)
    if ordinal is not None and family not in {"xgboost", "catboost", "cuml_rf"}:
        raise BackendUnavailableError(f"{family} is CPU-only in this adapter; request device='cpu'")
    if ordinal is None and family == "cuml_rf":
        raise BackendUnavailableError("cuML random forest requires CUDA; use random_forest for CPU")
    supplied = dict(params or {})
    controls = {
        "device",
        "devices",
        "device_type",
        "task_type",
        "n_jobs",
        "thread_count",
        "random_state",
        "random_seed",
        "seed",
        "output_type",
        "handle",
    }
    if controls.intersection(supplied):
        raise ValueError(
            f"Pass execution controls as arguments: {sorted(controls.intersection(supplied))}"
        )
    if "n_streams" in supplied and family != "cuml_rf":
        raise ValueError("n_streams is only supported by the cuML random forest backend")
    gpu = probe_cuda(ordinal) if ordinal is not None else None
    common = {"n_estimators": 300, **supplied}
    if family == "xgboost":
        package = "xgboost"
        module = _optional(package)
        if ordinal is not None and not module.build_info().get("USE_CUDA", False):
            raise BackendUnavailableError("Installed XGBoost was built without CUDA support")
        if supplied.get("tree_method", "hist") != "hist":
            raise ValueError("This adapter requires XGBoost tree_method='hist' for comparable runs")
        options = dict(
            common, tree_method="hist", device=configured, n_jobs=threads, random_state=seed
        )
        estimator = module.XGBRegressor(**options)
    elif family == "catboost":
        package = "catboost"
        options = {"iterations": 300, "verbose": False, "allow_writing_files": False}
        aliases = {"n_estimators": "iterations", "max_depth": "depth", "max_bin": "border_count"}
        for key, value in supplied.items():
            target = aliases.get(key, key)
            if target != key and target in supplied:
                raise ValueError(f"Specify only one of {key!r} and {target!r}")
            options[target] = value
        options.update(random_seed=seed, thread_count=threads, task_type="GPU" if gpu else "CPU")
        if gpu:
            options.setdefault("gpu_ram_part", 0.7)
            options["devices"] = str(ordinal)
        estimator = _optional(package).CatBoostRegressor(**options)
    elif family == "cuml_rf":
        package = "cuml-cu13"
        if tuple(map(int, gpu["compute_capability"].split("."))) < (7, 0):
            raise BackendUnavailableError("cuML requires NVIDIA compute capability 7.0 or newer")
        options = dict(
            {"max_depth": 16, "n_bins": 128, "n_streams": 1, **common},
            random_state=seed,
            output_type="numpy",
        )
        with _optional("cupy").cuda.Device(ordinal):
            estimator = _optional("cuml.ensemble").RandomForestRegressor(**options)
    elif family == "lightgbm":
        package = "lightgbm"
        options = dict(common, n_jobs=threads, random_state=seed, device_type="cpu", verbosity=-1)
        estimator = _optional(package).LGBMRegressor(**options)
    else:
        package = "scikit-learn"
        module = importlib.import_module("sklearn.ensemble")
        if family == "hist":
            options = {"max_iter": 300, "early_stopping": False, **supplied, "random_state": seed}
            if "n_estimators" in options:
                options["max_iter"] = options.pop("n_estimators")
            estimator = module.HistGradientBoostingRegressor(**options)
        else:
            options = dict(common, n_jobs=threads, random_state=seed)
            cls = (
                module.RandomForestRegressor
                if family == "random_forest"
                else module.ExtraTreesRegressor
            )
            estimator = cls(**options)
    metadata = {
        "requested_family": requested_family,
        "family": family,
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__name__}",
        "package": package,
        "package_version": _version(package),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": device,
        "configured_device": configured,
        "effective_device": None,
        "device_verified": False,
        "threads": threads,
        "seed": seed,
        "requested_params": supplied,
        "effective_params": options,
        "gpu": gpu,
    }
    # Fail early if an experiment supplied a non-serializable config.
    json.dumps(metadata, allow_nan=False)
    return TreeEstimator(estimator, metadata), metadata


def smoke_test(families: list[str], device: str = "cuda", repeats: int = 1) -> dict[str, Any]:
    """Run real fits; this is deliberately separate from portable unit tests."""
    rng = np.random.default_rng(421)
    x = rng.normal(size=(1024, 32)).astype(np.float32)
    y = (3 * x[:, 0] + x[:, 1] ** 2 + rng.normal(0, 0.1, len(x))).astype(np.float32)
    records = []
    for family in families:
        for repeat in range(repeats):
            model, metadata = build_estimator(
                family, {"n_estimators": 40, "max_depth": 6}, device=device
            )
            model.fit(x[:800], y[:800])
            prediction = model.predict(x[800:])
            if prediction.shape != y[800:].shape or not np.isfinite(prediction).all():
                raise RuntimeError(f"{family} returned invalid smoke predictions")
            records.append(
                {
                    **metadata,
                    "repeat": repeat,
                    "rmse": float(np.sqrt(np.mean((prediction - y[800:]) ** 2))),
                    "baseline_rmse": float(np.sqrt(np.mean((y[:800].mean() - y[800:]) ** 2))),
                }
            )
    return {"dataset": "synthetic_smoke_1024x32_seed421", "fits": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=["xgboost", "catboost", "cuml_rf"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(smoke_test(args.families, args.device, args.repeats), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
