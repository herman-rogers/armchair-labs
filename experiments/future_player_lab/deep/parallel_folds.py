"""Redistribute verified unfinished fits by year without changing numerical code."""

import argparse
import json
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from ..data import digest
from ..run import write_json
from .capacity import task as capacity_task
from .run import LAB, ROOT, fit_job, source_hashes, validate_data


def fold_config(config, year):
    # These two fields only delimit the loop in fit_job/capacity_task. Every
    # numerical fit still receives exactly the same earlier rows and settings.
    return {**config, "warmup_start": year, "evaluation_years": [year]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data", default="deep_data_001")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or any(
        Path(s).name != s or s in {".", ".."} for s in [args.source, args.run_id, args.data]
    ):
        raise ValueError("Invalid execution arguments")
    old, run, data = [LAB / "runs" / n for n in [args.source, args.run_id, args.data]]
    prior = json.loads((old / "protocol.json").read_text())
    state = json.loads((old / "manifest.json").read_text())
    if state["status"] == "running":
        raise ValueError("Stop the old executor before reusing checkpoints")
    fixed = "paths" in prior
    config = prior["config"]
    if config != json.loads((data / "config.json").read_text()):
        raise ValueError("Prepared configuration changed")
    data_sha = validate_data(data)
    if data_sha != prior["data_manifest" if fixed else "data_sha256"]:
        raise ValueError("Prepared data changed")
    numerical = (
        source_hashes()
        if fixed
        else {
            str((LAB / "deep" / n).relative_to(ROOT)): digest(LAB / "deep" / n)
            for n in ["capacity.py", "run.py", "models.py"]
        }
    )
    if any(prior["source"].get(name) != sha for name, sha in numerical.items()):
        raise ValueError("Numerical implementation changed")
    if any(version(package) != value for package, value in prior["environment"].items()):
        raise ValueError("Numerical package versions changed")
    implementation = {**numerical, str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    protocol = {**prior, "source": implementation, "data_run": args.data}
    run.mkdir(exist_ok=False)
    write_json(run / "protocol.json", protocol)
    for name in implementation:
        dest = run / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    reused = {}
    for file in (old / "fits").rglob("*.npz"):
        diagfile = file.with_suffix(".json")
        if not diagfile.exists():
            continue
        diag = json.loads(diagfile.read_text())
        if digest(file) != diag["sha256"]:
            raise ValueError("Checkpoint changed")
        relative = file.relative_to(old)
        dest = run / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
        shutil.copy2(diagfile, dest.with_suffix(".json"))
        reused[str(relative)] = diag["sha256"]
    write_json(
        run / "reuse.json",
        dict(
            source=args.source,
            source_protocol_sha256=digest(old / "protocol.json"),
            source_manifest_sha256=digest(old / "manifest.json"),
            checkpoints=reused,
        ),
    )
    record = {
        k: state[k]
        for k in ["candidate_count", "paths", "primary_target", "catalog_start_sha256"]
        if k in state
    }
    record.update(
        status="running",
        started=datetime.now(UTC).isoformat(),
        research_only=True,
        data_run=args.data,
        executor="independent annual folds",
        workers=args.workers,
        reused_checkpoints=len(reused),
    )
    write_json(run / "manifest.json", record)
    try:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        futures = []
        try:
            for position in config["positions"]:
                for horizon in config["horizons"]:
                    recipes = (
                        [(p["name"], p) for p in prior["paths"]]
                        if fixed
                        else [
                            (f"auto_{e}_{v}", (e, v))
                            for e in prior["engines"]
                            for v in prior["views"]
                        ]
                    )
                    for name, recipe in recipes:
                        for year in range(
                            config["warmup_start"], max(config["evaluation_years"]) + 1
                        ):
                            path = run / "fits" / f"{position}_{horizon}" / f"{year}_{name}.npz"
                            if str(path.relative_to(run)) in reused:
                                continue
                            scoped = fold_config(config, year)
                            if fixed:
                                future = pool.submit(
                                    fit_job, str(data), str(run), position, horizon, recipe, scoped
                                )
                            else:
                                engine, view = recipe
                                future = pool.submit(
                                    capacity_task,
                                    str(run),
                                    str(data),
                                    position,
                                    horizon,
                                    engine,
                                    view,
                                    scoped,
                                )
                            futures.append(future)
            print(
                f"Reused {len(reused)} checkpoints; queued {len(futures)} annual fits", flush=True
            )
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if i % 25 == 0 or i == len(futures):
                    print(f"[{i}/{len(futures)}] {result}", flush=True)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        if validate_data(data) != data_sha or any(
            digest(ROOT / p) != sha for p, sha in implementation.items()
        ):
            raise ValueError("Source or data changed during fitting")
        record.update(
            status="fits_complete" if fixed else "complete",
            completed=datetime.now(UTC).isoformat(),
            fit_checkpoints=len(list((run / "fits").rglob("*.npz"))),
            catalog_end_sha256=digest(ROOT / "data/current.json"),
            artifacts={
                str(p.relative_to(run)): digest(p) for p in (run / "fits").rglob("*") if p.is_file()
            },
        )
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write_json(run / "manifest.json", record)


if __name__ == "__main__":
    main()
