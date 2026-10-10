"""Explicit model contracts; adding a position requires its own registered artifact."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..gpu_lab.refit import load_policy
from ..gpu_lab.run import ROOT, digest, packages

REGISTRY = Path(__file__).with_name("registry")


def load_spec(model_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", model_id):
        raise ValueError("Unsafe model identifier")
    path = REGISTRY / f"{model_id}.json"
    if not path.exists():
        raise ValueError(f"Model is not registered: {model_id}")
    spec = json.loads(path.read_text())
    if spec["id"] != model_id or spec["schema_version"] != 1:
        raise ValueError("Invalid model registry entry")
    return spec


def model_contract(spec, season, origin):
    if origin not in spec["supported_origins"]:
        raise ValueError(
            f"{spec['id']} supports completed weeks {spec['supported_origins']}; got {origin}"
        )
    directory = ROOT / spec["model_directory"]
    path = directory / "manifest.json"
    if digest(path) != spec["model_manifest_sha256"]:
        raise ValueError("Registered model manifest checksum mismatch")
    manifest = json.loads(path.read_text())
    if manifest["status"] != "complete" or manifest["training_cutoff_year"] >= season:
        raise ValueError("Forecast season must follow the completed model training period")
    inputs = manifest["input_provenance"][spec["horizon"]]
    if inputs["contract"]["position"] != spec["position"]:
        raise ValueError("Registered position does not match the trained model")
    if origin not in inputs["contract"]["origins"]:
        raise ValueError("Registered origin does not match model training")
    installed = packages()
    for package in ("numpy", "scikit-learn", "xgboost", "catboost", "lightgbm"):
        if installed[package] != manifest["packages"][package]:
            raise ValueError(f"Saved-model package version changed: {package}")
    return manifest, inputs["names"]


def load_registered_policy(spec, policy):
    return load_policy(ROOT / spec["model_directory"], spec["horizon"], spec["target"], policy)
