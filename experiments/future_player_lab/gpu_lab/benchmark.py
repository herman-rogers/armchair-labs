"""Matched RB CPU/CUDA timing with isolated first fits and repeated warm fits."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gpu_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return [int(value) for value in result.stdout.splitlines()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _process_memory():
    """VmHWM starts at exec; getrusage can retain the parent's pre-exec peak."""
    path = Path("/proc/self/status")
    if not path.exists():
        return {"process_peak_rss_kib": None, "process_rss_kib": None}
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith(("VmHWM:", "VmRSS:")):
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0])
    return {
        "process_peak_rss_kib": values.get("VmHWM"),
        "process_rss_kib": values.get("VmRSS"),
    }


def worker(config_path):
    import joblib

    from ..gpu_backends import build_estimator

    config = json.loads(Path(config_path).read_text())
    with np.load(config["arrays"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    records, predictions = [], []
    for repeat in range(config["repeats"]):
        before_memory = _gpu_memory()
        started = time.perf_counter()
        model, metadata = build_estimator(
            config["family"],
            config["params"],
            seed=20261003,
            device=config["device"],
            threads=config["threads"],
        )
        construction_seconds = time.perf_counter() - started
        model.fit(arrays["train_x"], arrays["train_y"])
        prediction_start = time.perf_counter()
        pred = model.predict(arrays["test_x"]) * arrays["test_exposure"]
        prediction_seconds = time.perf_counter() - prediction_start
        total_seconds = time.perf_counter() - started
        if not np.isfinite(pred).all():
            raise RuntimeError("Benchmark produced nonfinite predictions")
        # Confirm the exact artifact strategy used by the campaign after timing.
        if repeat == 0:
            buffer = io.BytesIO()
            joblib.dump(model, buffer, compress=3)
            buffer.seek(0)
            restored = joblib.load(buffer)
            restored_pred = restored.predict(arrays["test_x"]) * arrays["test_exposure"]
            np.testing.assert_allclose(restored_pred, pred, rtol=1e-5, atol=1e-5)
            del restored, buffer
        error = pred - arrays["test_y"]
        records.append(
            {
                "id": config["id"],
                "repeat": repeat,
                "timing_state": "first_fit_in_fresh_process" if repeat == 0 else "warm_fit",
                "construction_seconds": construction_seconds,
                "fit_seconds": metadata["fit_seconds"],
                "prediction_seconds": prediction_seconds,
                "total_seconds": total_seconds,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "fits_per_hour": 3600 / total_seconds,
                "resource_lifetime_peak_rss_kib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                **_process_memory(),
                "system_gpu_used_mib_before": before_memory,
                "system_gpu_used_mib_after": _gpu_memory(),
                "execution": metadata,
                "artifact_roundtrip_verified": repeat == 0,
            }
        )
        predictions.append(pred)
        del model
        gc.collect()
    output = Path(config["output"])
    output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    np.save(output.with_suffix(".npy"), np.stack(predictions))


def benchmark(output, cache, year=2025, horizon="season", target="rushing_yards", repeats=3):
    from threadpoolctl import threadpool_limits

    from .data import STATS, load_dataset
    from .recipes import recipe_library
    from .representations import fit_representation

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = load_dataset(horizon, cache_dir=cache)
    years = dataset.panel["year"].to_numpy()
    y = dataset.targets[:, STATS.index(target)]
    train = (years < year) & np.isfinite(y)
    test = (years == year) & np.isfinite(y)
    if not train.any() or not test.any():
        raise ValueError("Benchmark requires nonempty known training and evaluation rows")
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        rep = fit_representation(dataset.x[train], dataset.names, {"kind": "raw"})
        train_x = rep.transform(dataset.x[train])
        test_x = rep.transform(dataset.x[test])
    transform_seconds = time.perf_counter() - started
    arrays = output / "arrays.npz"
    np.savez(
        arrays,
        train_x=train_x,
        train_y=(y[train] / dataset.exposure[train]).astype(np.float32),
        test_x=test_x,
        test_y=y[test],
        test_exposure=dataset.exposure[test],
    )
    # The first recipe for each family is its prespecified raw anchor.
    recipe_defaults = {}
    for recipe in recipe_library():
        recipe_defaults.setdefault(recipe["family"], recipe["params"])
    configurations = [
        {"id": f"xgboost_cpu_{n}", "family": "xgboost", "device": "cpu", "threads": n}
        for n in [1, 2, 4, 8]
    ] + [
        {"id": f"{family}_cuda", "family": family, "device": "cuda", "threads": 4}
        for family in ["xgboost", "catboost", "cuml_rf"]
    ]
    records = []
    all_predictions = {}
    for config in configurations:
        config.update(
            params=recipe_defaults[config["family"]],
            arrays=str(arrays),
            repeats=repeats,
            output=str(output / f"{config['id']}.json"),
        )
        config_path = output / f"{config['id']}.config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Benchmarking {config['id']} ({repeats} fits)", flush=True)
        with (output / f"{config['id']}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.future_player_lab.gpu_lab.benchmark",
                    "--worker-config",
                    str(config_path),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        records.extend(json.loads(Path(config["output"]).read_text()))
        all_predictions[config["id"]] = np.load(Path(config["output"]).with_suffix(".npy"))
    reference = all_predictions["xgboost_cpu_1"][0]
    for record in records:
        pred = all_predictions[record["id"]][record["repeat"]]
        record["prediction_rmse_vs_xgboost_cpu_1"] = float(
            np.sqrt(np.mean((pred - reference) ** 2))
        )
        record["prediction_max_abs_delta_vs_xgboost_cpu_1"] = float(np.max(abs(pred - reference)))
    result = {
        "horizon": horizon,
        "evaluation_year": year,
        "target": target,
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "raw_features": len(dataset.names),
        "model_features": train_x.shape[1],
        "representation_seconds": transform_seconds,
        "arrays_sha256": _sha(arrays),
        "source_sha256": {
            str(path): _sha(path)
            for path in [
                Path(__file__),
                Path(__file__).parents[1] / "gpu_backends.py",
                Path(__file__).with_name("recipes.py"),
                Path(__file__).with_name("representations.py"),
            ]
        },
        "dataset_provenance": dataset.provenance,
        "feature_names": rep.output_names_.tolist(),
        "fits": records,
        "notes": [
            "Each configuration starts a fresh process, followed by repeated warm fits.",
            "First-fit timing includes library construction but excludes interpreter startup.",
            "Cold driver/OS filesystem caches and other GPU processes are not controlled.",
            "GPU memory readings include display and unrelated processes and are not fit peaks.",
            "Only XGBoost CPU/CUDA runs match statistical settings; other algorithms differ.",
            "One representative target; throughput does not establish whole-campaign speedup.",
        ],
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                in {
                    "horizon",
                    "target",
                    "train_rows",
                    "test_rows",
                    "raw_features",
                    "model_features",
                }
            }
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--input-cache", default="experiments/future_player_lab/runs/gpu_raw_inputs_001"
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--horizon", default="season")
    parser.add_argument("--target", default="rushing_yards")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.worker_config:
        worker(args.worker_config)
    else:
        if not args.output:
            parser.error("--output is required")
        if args.repeats < 3:
            parser.error("at least three repetitions are required")
        benchmark(args.output, args.input_cache, args.year, args.horizon, args.target, args.repeats)


if __name__ == "__main__":
    main()
