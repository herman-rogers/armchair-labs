"""Seal v3.2 serving weights, uncertainty evidence and baseline replay dependencies."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from engine.chairstats import CampaignStore

from ..weekly_v31.artifacts import digest, now, publish_directory, seal, verify, write
from . import data, uncertainty
from .run import DEFAULT_RUN

DEFAULT_INTEGRATION = (
    data.ROOT / "experiments/future_player_lab/runs/opening_drive_v32_serving_20261013_001"
)


def build(
    output=DEFAULT_INTEGRATION,
    research=DEFAULT_RUN,
    registry=data.DATA / "campaigns",
    evaluated=None,
):
    output, research, registry = map(Path, (output, research, registry))
    verify(research)
    if output.exists():
        raise FileExistsError("Use a new immutable integration version")
    for name in ("data.py", "modeling.py"):
        if (research / "source" / name).read_bytes() != Path(__file__).with_name(name).read_bytes():
            raise ValueError("Research inference source changed")
    store = CampaignStore(registry)
    store.check("opening_drive_v3")
    manifest = store.campaign("opening_drive_v3")
    research_baseline = json.loads((data.V3_RUN / "manifest.json").read_text())
    original_weights = set(research_baseline["artifact_hashes"].values())
    for entry in manifest["models"]:
        if entry["horizon"] == "weekly":
            for key in ("mean_model", "distribution_model"):
                if entry["artifacts"][key]["sha256"] not in original_weights:
                    raise ValueError("Published baseline weights differ from v3 research")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v32-serving-") as tmp:
        staging = Path(tmp) / "run"
        staging.mkdir()
        write(
            staging / "protocol.json",
            dict(
                mean_policy="Unchanged v3.2 chronological selections; no model reselection",
                uncertainty="256 centered empirical earlier-year errors; "
                "position and fixed mean bins",
                evaluation="2023–25, seeded by 2021–22 errors; years reused for development",
                lineup="Matched v3-ranked and v3.2-ranked artificial offense lineups, 10 per week",
                timing="Early week first capture; per-game late capture "
                "90–60 minutes before kickoff",
                fallback="No usable official game-week evidence: retain early-week forecast",
                publication="Selectable preview, no default change; K/DST uncertainty withheld",
            ),
        )
        if evaluated is None:
            selected = pd.read_parquet(
                research / "predictions.parquet", filters=[("method", "==", "selected")]
            )
            v3 = pd.read_parquet(data.V3_RUN / "selected_predictions.parquet")
            scores, lineups, contracts, deps = uncertainty.evaluate(
                selected, v3, pd.read_parquet(research / "panel.parquet")
            )
            scores.to_csv(staging / "uncertainty_scores.csv", index=False)
            lineups.to_parquet(staging / "lineup_predictions.parquet", index=False)
            write(staging / "uncertainty.json", contracts)
            write(staging / "dependence.json", deps)
        else:
            for name in (
                "uncertainty_scores.csv",
                "lineup_predictions.parquet",
                "uncertainty.json",
                "dependence.json",
            ):
                shutil.copy2(Path(evaluated) / name, staging / name)
        shutil.copy2(research / "selection.json", staging / "selection.json")
        shutil.copy2(research / "scores.csv", staging / "point_scores.csv")
        (staging / "models").mkdir()
        for path in (research / "models").glob("*_2026.joblib"):
            shutil.copy2(path, staging / "models" / path.name)
        entries = []
        for entry in manifest["models"]:
            if entry["horizon"] != "weekly":
                continue
            refs = {}
            for name in (
                "mean_model",
                "distribution_model",
                "source_modeling",
                "source_data",
                "run_manifest",
            ):
                source = registry / "opening_drive_v3" / entry["artifacts"][name]["path"]
                relative = f"{entry['position']}/{name}{source.suffix}"
                target = staging / "baseline" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                refs[name] = dict(path=relative, sha256=digest(target))
            entries.append(dict(position=entry["position"], artifacts=refs))
        write(staging / "baseline/entries.json", entries)
        calibration = next(m for m in manifest["models"] if m["horizon"] == "weekly")["artifacts"][
            "lineup_calibration"
        ]
        shutil.copy2(
            registry / "opening_drive_v3" / calibration["path"],
            staging / "baseline_lineup_calibration.json",
        )
        shutil.copytree(
            Path(__file__).parent, staging / "source", ignore=shutil.ignore_patterns("__pycache__")
        )
        # Pin runtime helpers too: the serving contract must not silently drift after sealing.
        from engine.metrics import weekly_distribution

        from ..weekly_v3 import lineups as v3_lineups
        from ..weekly_v3 import rescore as v3_rescore

        dependencies = [
            Path(v3_rescore.__file__),
            Path(v3_lineups.__file__),
            Path(weekly_distribution.__file__),
        ]
        write(staging / "dependencies.json", {str(p): digest(p) for p in dependencies})
        seal(
            staging,
            dict(
                status="v32_shadow_serving",
                forecast_season=2026,
                generated_at=now().isoformat(),
                research_run=str(research.resolve()),
                research_manifest_sha256=digest(research / "manifest.json"),
                baseline_manifest_sha256=manifest["manifest_sha256"],
                packages={
                    p: importlib.metadata.version(p)
                    for p in ("numpy", "pandas", "lightgbm", "scikit-learn", "joblib")
                },
            ),
        )
        publish_directory(staging, output)
    return output


def check(directory):
    directory = Path(directory)
    receipt = verify(directory)
    for name, version in receipt["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"V3.2 environment changed: {name}")
    # Freeze numerical inference. Collection/publication orchestration may receive
    # reviewed fixes without invalidating already locked feature-matrix replays.
    for name in ("data.py", "modeling.py", "live.py", "uncertainty.py"):
        path = directory / "source" / name
        if path.read_bytes() != Path(__file__).with_name(path.name).read_bytes():
            raise ValueError(f"V3.2 serving source changed: {path.name}; build a new version")
    for path, sha in json.loads((directory / "dependencies.json").read_text()).items():
        if digest(path) != sha:
            raise ValueError(f"Serving dependency changed: {path}")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_INTEGRATION)
    parser.add_argument("--research", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    print(build(args.output, args.research))
