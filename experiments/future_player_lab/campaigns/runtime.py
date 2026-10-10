"""Compatibility gates for trusted local model artifacts."""

from pathlib import Path

from ..gpu_lab.run import digest, packages
from .data import ROOT


def verify_environment(manifest, *, check_sources=True):
    """Fail before unpickling or rebuilding features under a changed contract."""
    current = packages()
    for name in ("numpy", "polars", "scikit-learn", "xgboost", "lightgbm"):
        expected = manifest["packages"][name]
        if current[name] != expected:
            raise ValueError(f"Package version differs for {name}: {current[name]} != {expected}")
    if check_sources:
        for relative, sha in manifest["source_files"].items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or digest(path) != sha:
                raise ValueError(f"Inference source contract changed: {relative}")
    return True


def source_artifacts(manifest):
    return {
        f"source_{index:02d}_{Path(relative).stem}": ROOT / relative
        for index, relative in enumerate(manifest["source_files"])
    }
