#!/usr/bin/env python3
# ruff: noqa: E402
"""Protocol 2 stage runner: all mutations stay in a create-only lab run."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import importlib
import json
import os
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from followup_common import CONFIG, Inputs
from safety import install_write_guard, new_run, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["audit", "calibration", "signals", "structure", "distributions"],
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    name = args.run_id or args.stage + "_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    root = new_run(LAB, name)
    install_write_guard(root)
    manifest = dict(
        run_id=name,
        stage=args.stage,
        protocol=2,
        status="running",
        research_only=True,
        started_at=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        packages={k: version(k) for k in ("numpy", "polars", "scipy", "scikit-learn")},
    )
    write_json(root / "manifest.json", manifest)
    inputs = None
    try:
        stage = importlib.import_module("followup_" + args.stage)
        inputs = Inputs()
        modules = [
            Path(m.__file__).resolve()
            for m in list(sys.modules.values())
            if getattr(m, "__file__", None) and Path(m.__file__).resolve().parent == LAB
        ]
        files = set(modules + [LAB / "config.json", LAB / "followup_protocol.md"])
        for path in files:
            inputs.bind(path)
            destination = root / "source" / path.name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(path.read_bytes())
        catalog = REPO / "data/current.json"
        catalog_before = sha256(catalog)
        write_json(root / "catalog_at_start.json", json.loads(catalog.read_text()))
        write_json(root / "config.json", CONFIG)
        summary = stage.run(inputs, root)
        inputs.verify()
        manifest.update(
            status="complete",
            completed_at=datetime.now(UTC).isoformat(),
            protected_inputs_unchanged=True,
            inputs=inputs.hashes,
            summary=summary,
            current_catalog_changed=sha256(catalog) != catalog_before,
            outputs={
                str(p.relative_to(root)): sha256(p)
                for p in root.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        write_json(root / "manifest.json", manifest)
        print(f"COMPLETE {args.stage}: {root}", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        if inputs:
            manifest["inputs"] = inputs.hashes
        write_json(root / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
