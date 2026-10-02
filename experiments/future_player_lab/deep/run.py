"""Execute immutable deep-search paths; each checkpoint is one chronological fit."""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from ..data import digest
from ..run import write_json
from .models import fit_path, names_for, paths

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[1]
_CACHE = {}


def source_hashes():
    files = [LAB / "deep" / n for n in ("__init__.py", "models.py", "data.py", "run.py")] + [
        LAB / n for n in ("data.py", "models.py", "evaluate.py", "run.py", "report.py")
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def load_task(data, position, horizon):
    key = (str(data), position, horizon)
    if key not in _CACHE:
        _CACHE.clear()
        panel = pl.read_parquet(data / "panel.parquet")
        ix = np.flatnonzero(
            (panel["position"].to_numpy() == position) & (panel["horizon"].to_numpy() == horizon)
        )
        _CACHE[key] = (
            panel[ix],
            np.load(data / "x.npy", mmap_mode="r")[ix],
            np.load(data / "targets.npy", mmap_mode="r")[ix],
        )
    return _CACHE[key]


def fit_job(data_path, run_path, position, horizon, path, config):
    data, run = Path(data_path), Path(run_path)
    panel, x, y = load_task(data, position, horizon)
    metadata = json.loads((data / "features.json").read_text())
    cols = metadata["views"][path["view"]]
    x = x[:, cols]
    years, exposure = panel["year"].to_numpy(), panel["exposure"].to_numpy()
    folder = run / "fits" / f"{position}_{horizon}"
    folder.mkdir(parents=True, exist_ok=True)
    count, elapsed = 0, 0.0
    with threadpool_limits(limits=1):
        for year in range(config["warmup_start"], max(config["evaluation_years"]) + 1):
            filename = f"{year}_{path['name']}"
            checkpoint, diagfile = folder / f"{filename}.npz", folder / f"{filename}.json"
            test, train = years == year, years < year
            if checkpoint.exists() and diagfile.exists():
                diag = json.loads(diagfile.read_text())
                if digest(checkpoint) != diag["sha256"]:
                    raise ValueError("Checkpoint changed")
                continue
            predictions, diag = fit_path(
                path,
                config,
                x[train],
                y[train],
                exposure[train],
                years[train],
                x[test],
                exposure[test],
            )
            diag.update(
                position=position,
                horizon=horizon,
                year=year,
                train_last_year=int(years[train].max()),
                train_rows=int(train.sum()),
                test_rows=int(test.sum()),
                path=path,
                feature_view_indices=cols,
            )
            np.savez_compressed(checkpoint, **predictions)
            diag["sha256"] = digest(checkpoint)
            write_json(diagfile, diag)
            count += len(predictions)
            elapsed += diag["fit_seconds"]
    return f"{position}/{horizon}/{path['name']}: {count} capacity/fold predictions, {elapsed:.1f}s"


def validate_data(data):
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete prepared data")
    for name, sha in manifest["artifacts"].items():
        if digest(data / name) != sha:
            raise ValueError("Prepared data changed")
    for name, sha in manifest["inputs"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Underlying source changed")
    return digest(data / "manifest.json")


def main():
    sys.path.insert(0, str(LAB / ".packages"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="deep_data_001")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(Path(s).name != s or s in {".", ".."} for s in [args.data, args.run_id]):
        raise ValueError("Invalid run name")
    data, run = LAB / "runs" / args.data, LAB / "runs" / args.run_id
    config = json.loads((data / "config.json").read_text())
    sha = validate_data(data)
    protocol = {
        "config": config,
        "data_manifest": sha,
        "data_run": args.data,
        "source": source_hashes(),
        "environment": {
            p: version(p)
            for p in ["numpy", "polars", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost"]
        },
        "paths": paths(config),
    }
    if args.resume:
        if json.loads((run / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume requires identical code, data and environment")
    else:
        run.mkdir(exist_ok=False)
        write_json(run / "protocol.json", protocol)
        for name in protocol["source"]:
            dest = run / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
    manifest = {
        "status": "running",
        "started": datetime.now(UTC).isoformat(),
        "research_only": True,
        "primary_target": "league points",
        "data_run": args.data,
        "catalog_start_sha256": digest(ROOT / "data/current.json"),
        "candidate_count": sum(len(names_for(p, config)) for p in paths(config)),
        "paths": len(paths(config)),
    }
    write_json(run / "manifest.json", manifest)
    started = time.monotonic()
    try:
        futures = []
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            for position in config["positions"]:
                for horizon in config["horizons"]:
                    for path in paths(config):
                        futures.append(
                            pool.submit(
                                fit_job, str(data), str(run), position, horizon, path, config
                            )
                        )
            for i, future in enumerate(as_completed(futures), 1):
                print(f"[{i}/{len(futures)}] {future.result()}", flush=True)
        if source_hashes() != protocol["source"] or validate_data(data) != sha:
            raise ValueError("Inputs or implementation changed during run")
        manifest.update(
            status="fits_complete",
            completed=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            catalog_end_sha256=digest(ROOT / "data/current.json"),
        )
        manifest["fit_checkpoints"] = len(list((run / "fits").rglob("*.npz")))
        manifest["artifacts"] = {
            str(p.relative_to(run)): digest(p) for p in (run / "fits").rglob("*") if p.is_file()
        }
        write_json(run / "manifest.json", manifest)
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc())
        write_json(run / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
