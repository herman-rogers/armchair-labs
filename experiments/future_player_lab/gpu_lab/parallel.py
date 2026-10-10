"""One bounded CPU worker alongside one GPU worker, each with isolated checkpoints."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .run import ROOT, digest, json_write, packages, run_campaign, source_paths, verify_checkpoint


def run_parallel(out, config, resume=False):
    ids = [r["id"] for r in config["recipes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Recipe IDs must be unique across workers")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / f".{out.name}.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            if json.loads((out / "config.json").read_text()) != config:
                raise ValueError("Parallel resume configuration differs")
        else:
            out.mkdir(exist_ok=False)
            json_write(out / "config.json", config)
        signature = {
            "sources": {str(p.relative_to(ROOT)): digest(p) for p in source_paths()},
            "packages": packages(),
        }
        if resume and json.loads((out / "signature.json").read_text()) != signature:
            raise ValueError("Parallel resume code or packages changed")
        json_write(out / "signature.json", signature)
        for p in source_paths():
            target = out / "implementation" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        processes, logs = {}, []
        began = time.monotonic()
        for worker in ["cpu", "gpu"]:
            partial = {
                **config,
                "recipes": [
                    r for r in config["recipes"] if (r["device"] == "cpu") == (worker == "cpu")
                ],
            }
            if not partial["recipes"]:
                continue
            workdir = out / "workers" / worker
            cfgpath = out / f"{worker}_config.json"
            json_write(cfgpath, partial)
            log = (out / f"{worker}.log").open("a" if resume else "w")
            logs.append(log)
            command = [
                sys.executable,
                "-u",
                "-m",
                "experiments.future_player_lab.gpu_lab.parallel",
                "--worker",
                str(workdir),
                "--config",
                str(cfgpath),
            ]
            if resume and workdir.exists():
                command.append("--resume")
            env = {
                **os.environ,
                "OMP_NUM_THREADS": str(config["threads"]),
                "OPENBLAS_NUM_THREADS": str(config["threads"]),
                "POLARS_MAX_THREADS": str(config["threads"]),
            }
            processes[worker] = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env
            )
        json_write(
            out / "manifest.json",
            {
                "status": "running",
                "gpu_workers": 1,
                "cpu_workers": 1,
                "worker_pids": {k: p.pid for k, p in processes.items()},
            },
        )
        try:
            while any(p.poll() is None for p in processes.values()):
                progress = {}
                for worker in processes:
                    path = out / "workers" / worker / "progress.json"
                    if path.exists():
                        progress[worker] = json.loads(path.read_text())
                json_write(
                    out / "progress.json",
                    {"workers": progress, "seconds": time.monotonic() - began},
                )
                print(json.dumps(progress), flush=True)
                time.sleep(30)
        except BaseException:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                process.wait()
            json_write(
                out / "manifest.json",
                {"status": "interrupted", "reason": "Parent interrupted; workers stopped"},
            )
            raise
        finally:
            for log in logs:
                log.close()
        if any(p.returncode for p in processes.values()):
            json_write(
                out / "manifest.json",
                {
                    "status": "interrupted",
                    "exit_codes": {k: p.returncode for k, p in processes.items()},
                },
            )
            raise RuntimeError(
                "A worker failed; inspect cpu.log/gpu.log, then resume unchanged inputs"
            )
        children = []
        for worker in processes:
            workdir = out / "workers" / worker
            child = json.loads((workdir / "manifest.json").read_text())
            child_signature = json.loads((workdir / "signature.json").read_text())
            if not child["code_unchanged"] or any(
                child_signature[k] != signature[k] for k in ["sources", "packages"]
            ):
                json_write(out / "manifest.json", {"status": "invalid_source_change"})
                raise ValueError("Worker source/packages changed during campaign")
            for receipt in workdir.glob("*/fits/*/*/complete.json"):
                verify_checkpoint(receipt.parent)
            children.append(child)
            for horizon in config["horizons"]:
                source, target = workdir / horizon, out / horizon
                target.mkdir(exist_ok=True)
                for name in ["inputs.json", "baselines.parquet", "target_audit.parquet"]:
                    if (target / name).exists() and digest(target / name) != digest(source / name):
                        raise ValueError(f"Workers disagree on input/baseline {horizon}/{name}")
                    shutil.copyfile(source / name, target / name)
                shutil.copytree(source / "fits", target / "fits", dirs_exist_ok=True)
        failures = [
            f
            for w in processes
            for f in json.loads((out / "workers" / w / "failures.json").read_text())
        ]
        json_write(out / "failures.json", failures)
        json_write(
            out / "manifest.json",
            {
                "status": "complete_with_failures" if failures else "complete",
                "completed_fits": sum(c["completed_fits"] for c in children),
                "failed_folds": len(failures),
                "seconds": time.monotonic() - began,
                "code_unchanged": all(c["code_unchanged"] for c in children),
                "gpu_workers": 1,
                "cpu_workers": 1,
                "worker_manifests": children,
                "configuration_sha256": digest(out / "config.json"),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_campaign(args.worker, json.loads(args.config.read_text()), args.resume)
