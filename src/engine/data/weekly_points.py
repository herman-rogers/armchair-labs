"""Verified, analysis-bound dedicated weekly model products."""

from pathlib import Path

from engine.data.frames import read_frame
from engine.data.nextgen import load_analysis
from engine.data.releases import digest, identifier, inside, reference
from engine.data.verification import observe, read_json, verified


def load_weekly_points(data: Path):
    pointer = data / "weekly/current.json"
    observe(pointer)
    if not pointer.exists():
        return None
    ref = read_json(pointer)
    analysis, _ = load_analysis(data)

    def load():
        root = data / "research" / identifier(ref["version"])
        if digest(root / "manifest.json") != ref["manifest_sha256"]:
            raise ValueError("Weekly model manifest changed")
        manifest = read_json(root / "manifest.json")
        if manifest.get("kind") != "nextgen_weekly_points" or manifest.get("status") != "complete":
            raise ValueError("Weekly model is not complete")
        if manifest["analysis"] != reference(analysis):
            raise ValueError("Weekly model needs a build for the current analysis release")
        for name, expected in manifest["files"].items():
            if digest(inside(root, name)) != expected:
                raise ValueError(f"Weekly model artifact changed: {name}")
        return root, manifest

    root, manifest = verified(
        (
            "weekly-points",
            str(data.absolute()),
            tuple(sorted(ref.items())),
            digest(analysis / "manifest.json"),
        ),
        load,
    )
    from engine.tables.application import read_json as query_json

    return dict(
        manifest=manifest,
        report=query_json(root / "report.json"),
        predictions=read_frame(root / "predictions.parquet").to_dicts(),
    )
