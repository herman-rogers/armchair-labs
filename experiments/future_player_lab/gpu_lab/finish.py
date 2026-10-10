"""Run analysis, diagnostics, final refits and figures after all campaign fits complete."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .run import ROOT, RUNS, json_write


def finish(out, wait=False):
    out = Path(out)
    while True:
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["status"] == "complete":
            break
        if manifest["status"] != "running" or not wait:
            raise ValueError(f"Training is not complete: {manifest['status']}")
        time.sleep(30)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "POLARS_MAX_THREADS": "4",
    }
    for step in ["analyze", "diagnostics", "refit", "figures"]:
        print(f"Postprocessing {step}", flush=True)
        json_write(out / "postprocess.json", {"status": "running", "step": step})
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                f"experiments.future_player_lab.gpu_lab.{step}",
                "--run-id",
                out.name,
            ],
            cwd=ROOT,
            env=env,
            check=False,
        )
        if result.returncode:
            json_write(
                out / "postprocess.json",
                {"status": "failed", "step": step, "exit_code": result.returncode},
            )
            raise RuntimeError(f"Postprocessing {step} failed")
    for name in ["diagnostics.py", "figures.py", "finish.py"]:
        shutil.copyfile(Path(__file__).with_name(name), out / "analysis" / name)
    # Resolve this after a potentially long wait, alongside the other late-loaded steps.
    from .analyze import seal_analysis

    seal_analysis(out)
    json_write(
        out / "postprocess.json",
        {"status": "complete", "steps": ["analyze", "diagnostics", "refit", "figures"]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    finish(RUNS / args.run_id, args.wait)
