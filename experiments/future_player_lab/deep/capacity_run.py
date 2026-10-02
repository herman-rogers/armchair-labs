"""Configurable execution and verified checkpoint reuse for adaptive tree counts."""

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
from .capacity import ENGINES, VIEWS, task
from .run import LAB, ROOT, validate_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="deep_data_001")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reuse")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or any(
        Path(s).name != s or s in {".", ".."}
        for s in [args.data, args.run_id, args.reuse or "none"]
    ):
        raise ValueError("Invalid execution arguments")
    data, run = LAB / "runs" / args.data, LAB / "runs" / args.run_id
    config = json.loads((data / "config.json").read_text())
    files = [
        Path(__file__),
        Path(__file__).with_name("capacity.py"),
        Path(__file__).with_name("run.py"),
        Path(__file__).with_name("models.py"),
    ]
    source = {str(p.relative_to(ROOT)): digest(p) for p in files}
    protocol = dict(
        config=config,
        data_sha256=validate_data(data),
        source=source,
        ceiling=4096,
        patience=80,
        engines=ENGINES,
        views=VIEWS,
        environment={
            p: version(p)
            for p in ["numpy", "scipy", "polars", "scikit-learn", "lightgbm", "xgboost", "catboost"]
        },
    )
    run.mkdir(exist_ok=False)
    write_json(run / "protocol.json", protocol)
    for name in source:
        dest = run / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    reused = {}
    if args.reuse:
        old = LAB / "runs" / args.reuse
        prior = json.loads((old / "protocol.json").read_text())
        state = json.loads((old / "manifest.json").read_text())
        if state["status"] == "running":
            raise ValueError("Stop the old executor before reusing its checkpoints")
        for key in [
            "config",
            "data_sha256",
            "ceiling",
            "patience",
            "engines",
            "views",
            "environment",
        ]:
            if prior[key] != json.loads(json.dumps(protocol[key])):
                raise ValueError(f"Adaptive checkpoint reuse changed {key}")
        for file in ["capacity.py", "run.py", "models.py"]:
            name = str((LAB / "deep" / file).relative_to(ROOT))
            if prior["source"][name] != source[name]:
                raise ValueError("Adaptive numerical implementation changed")
        for file in (old / "fits").rglob("*.npz"):
            diagfile = file.with_suffix(".json")
            if not diagfile.exists():
                continue
            diag = json.loads(diagfile.read_text())
            if digest(file) != diag["sha256"]:
                raise ValueError("Adaptive checkpoint changed")
            relative = file.relative_to(old)
            dest = run / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
            shutil.copy2(diagfile, dest.with_suffix(".json"))
            reused[str(relative)] = diag["sha256"]
        write_json(run / "reuse.json", dict(source=args.reuse, checkpoints=reused))
    record = dict(
        status="running",
        started=datetime.now(UTC).isoformat(),
        research_only=True,
        workers=args.workers,
        reused_checkpoints=len(reused),
    )
    write_json(run / "manifest.json", record)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(task, str(run), str(data), p, h, engine, view, config)
                for p in config["positions"]
                for h in config["horizons"]
                for engine in ENGINES
                for view in VIEWS
            ]
            for i, future in enumerate(as_completed(futures), 1):
                print(f"[{i}/{len(futures)}] {future.result()}", flush=True)
        if validate_data(data) != protocol["data_sha256"] or any(
            digest(ROOT / p) != sha for p, sha in source.items()
        ):
            raise ValueError("Adaptive source/data changed")
        record.update(
            status="complete",
            ended=datetime.now(UTC).isoformat(),
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
