"""Continue from a running fit campaign into its reports without a manual gap."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="deep_search_002")
    parser.add_argument("--capacity", default="deep_capacity_002")
    parser.add_argument("--report", default="deep_report_001")
    parser.add_argument("--diagnostics", default="deep_diagnostics_001")
    args = parser.parse_args()
    if any(Path(s).name != s or s in {".", ".."} for s in vars(args).values()):
        raise ValueError("Invalid run id")
    lab = Path(__file__).resolve().parents[1]
    source = lab / "runs" / args.source
    capacity = lab / "runs" / args.capacity
    last_update = 0.0
    while True:
        manifest = json.loads((source / "manifest.json").read_text())
        cap = json.loads((capacity / "manifest.json").read_text())
        if manifest["status"] == "fits_complete" and cap["status"] == "complete":
            break
        if manifest["status"] not in {"running", "fits_complete"}:
            raise RuntimeError(f"Fit campaign stopped with status {manifest['status']}")
        if cap["status"] not in {"running", "complete"}:
            raise RuntimeError(f"Adaptive capacity stopped with status {cap['status']}")
        if time.monotonic() - last_update >= 60:
            print(
                f"Progress: {len(list((source / 'fits').rglob('*.json')))}/11232 fit checkpoints",
                flush=True,
            )
            print(
                f"Adaptive capacity: {len(list((capacity / 'fits').rglob('*.json')))}/1248",
                flush=True,
            )
            last_update = time.monotonic()
        time.sleep(10)
    for module, options in [
        (
            "analyze",
            ["--source", args.source, "--capacity", args.capacity, "--run-id", args.report],
        ),
        (
            "diagnostics",
            [
                "--source",
                args.source,
                "--report",
                args.report,
                "--run-id",
                args.diagnostics,
            ],
        ),
    ]:
        print("Starting", module, flush=True)
        subprocess.run(
            [sys.executable, "-u", "-m", f"experiments.future_player_lab.deep.{module}", *options],
            check=True,
            cwd=lab.parents[1],
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
    print("Fits, ensemble evaluation and diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
