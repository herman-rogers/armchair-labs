#!/usr/bin/env python3
"""Combine independently executed position runs without fitting or choosing models."""

# ruff: noqa: E402
import sys

sys.dont_write_bytecode = True
import argparse
import json

import polars as pl
from report import build_report
from run import LAB, REPO
from safety import install_write_guard, new_run, sha256, write_json


def combine(run_ids, destination):
    manifests, predictions, folds, permutations, diagnostics = [], [], [], [], []
    positions = []
    config = inventory = None
    parents = {}
    for run_id in run_ids:
        root = LAB / "runs" / run_id
        if root.resolve().parent != (LAB / "runs").resolve():
            raise ValueError("Invalid child run ID")
        m = json.loads((root / "manifest.json").read_text())
        if (
            m["status"] != "complete"
            or not m["protected_inputs_unchanged"]
            or not m["publication_unchanged"]
        ):
            raise ValueError("Incomplete or invalid child run")
        consumed = ["predictions.parquet", "folds.json", "permutations.json", "inventory.json"]
        consumed += [f"diagnostics_{p}.json" for p in m["config"]["positions"]]
        for name in consumed:
            if sha256(root / name) != m["files"][name]:
                raise ValueError(f"Child checksum mismatch: {name}")
        child_config = {k: v for k, v in m["config"].items() if k != "positions"}
        child_inventory = json.loads((root / "inventory.json").read_text())
        if config is not None and (child_config != config or child_inventory != inventory):
            raise ValueError("Children have different protocols or input matrices")
        if manifests and m["protected_before"] != manifests[0]["protected_before"]:
            raise ValueError("Children have different source snapshots")
        config, inventory = child_config, child_inventory
        manifests.append(m)
        positions.extend(m["config"]["positions"])
        predictions.extend(pl.read_parquet(root / "predictions.parquet").to_dicts())
        folds.extend(json.loads((root / "folds.json").read_text()))
        permutations.extend(json.loads((root / "permutations.json").read_text()))
        for p in m["config"]["positions"]:
            diagnostics.extend(json.loads((root / f"diagnostics_{p}.json").read_text()))
        parents[run_id] = sha256(root / "manifest.json")
    if sorted(positions) != ["QB", "RB", "TE", "WR"]:
        raise ValueError("Need each position exactly once")
    config["positions"] = ["QB", "RB", "WR", "TE"]
    for path, expected in manifests[0]["protected_before"].items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"Protected input changed: {path}")
    pl.DataFrame(predictions).write_parquet(destination / "predictions.parquet")
    write_json(destination / "folds.json", folds)
    write_json(destination / "inventory.json", inventory)
    write_json(destination / "permutations.json", permutations)
    build_report(destination, predictions, folds, permutations, diagnostics, config, inventory)
    m = {
        "status": "complete",
        "research_only": True,
        "protected_inputs_unchanged": True,
        "publication_unchanged": True,
        "parents": parents,
        "config": config,
        "protected_before": manifests[0]["protected_before"],
    }
    for path, expected in manifests[0]["publication_before"].items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"Publication changed: {path}")
    m["files"] = {
        str(p.relative_to(destination)): sha256(p) for p in destination.rglob("*") if p.is_file()
    }
    write_json(destination / "manifest.json", m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("parents", nargs="+")
    args = parser.parse_args()
    destination = new_run(LAB, args.run_id)
    install_write_guard(destination)
    combine(args.parents, destination)
    print(destination)
