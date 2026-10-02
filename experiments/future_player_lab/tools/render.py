"""Render a completed run into a separate, versioned analysis artifact."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from ..data import digest
from ..report import generate

LAB = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    for value in (args.source, args.run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
            raise ValueError("Unsafe run identifier")
    source, output = LAB / "runs" / args.source, LAB / "runs" / args.run_id
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] not in ("complete", "complete_with_failures"):
        raise ValueError("Source experiment must be complete")
    inputs = {}
    for task in (source / "tasks").iterdir():
        complete = json.loads((task / "complete.json").read_text())
        for name, sha in complete["artifacts"].items():
            if digest(task / name) != sha:
                raise ValueError("Source task artifacts changed")
        for name in ("predictions.parquet", "metrics.parquet"):
            inputs[str((task / name).relative_to(LAB))] = digest(task / name)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((source / "config.json").read_text())
    generate(source, config, destination=output)
    for name in ("config.json", "candidates.json", "panel_audit.json", "features.json"):
        shutil.copy2(source / name, output / name)
    (output / "source").mkdir()
    for path in (LAB / "report.py", LAB / "evaluate.py", Path(__file__)):
        shutil.copy2(path, output / "source" / path.name)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_run": args.source,
                "source_manifest_sha256": digest(source / "manifest.json"),
                "inputs": inputs,
                "matched_row_differences": True,
                "analysis_sha256": {p.name: digest(p) for p in (output / "source").iterdir()},
                "source_candidate_failures": manifest.get("candidate_failures", 0),
            },
            indent=2,
        )
    )
    print(output / "index.html")


if __name__ == "__main__":
    main()
