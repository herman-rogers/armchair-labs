"""V4 run locations and immutable inference contracts."""

import platform
from importlib.metadata import version
from pathlib import Path

from ..weekly_v3.data import ROOT
from ..weekly_v31.artifacts import digest, verify

CAMPAIGN = "opening_drive_v4"
BASE = "opening_drive_v32"
RUNS = ROOT / "experiments/future_player_lab/runs"
DEFAULT_HORIZONS = RUNS / "opening_drive_v4_horizons_20261027_003"
DEFAULT_UNCERTAINTY = RUNS / "opening_drive_v4_uncertainty_20261027_002"
DEFAULT_INTEGRATION = RUNS / "opening_drive_v4_integration_20261027_001"
DEFAULT_LOG = ROOT / "data/outputs/opening_drive_v4_shadow"


def environment():
    return {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "lightgbm",
                "joblib",
                "pyarrow",
            )
        },
    }


def check(directory):
    directory = Path(directory)
    contract = verify(directory)
    if contract["environment"] != environment():
        raise ValueError("V4 numerical environment changed; seal a new run")
    if contract.get("base_campaign") != BASE:
        raise ValueError("V4 must retain its declared v3.2 base")
    for name, reference in contract["dependencies"].items():
        path = Path(reference["path"])
        verify(path)
        if digest(path / "manifest.json") != reference["sha256"]:
            raise ValueError(f"V4 {name} dependency changed")
    for name, expected in contract["inference_sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"V4 inference source changed: {name}; seal a new run")
    return contract
