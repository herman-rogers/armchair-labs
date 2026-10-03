"""Read notebook inputs from a restored workspace without training or networking."""

import argparse
import json
import socket
from pathlib import Path

from engine.data.releases import load_gold


def verify(root: Path) -> dict:
    from experiments.future_player_lab.notebooks import raw_distribution_data as raw
    from experiments.future_player_lab.notebooks import rb_availability as availability
    from experiments.future_player_lab.notebooks import run_raw_distributions as saved
    from experiments.future_player_lab.notebooks import workbench as work

    def no_network(*args, **kwargs):
        raise RuntimeError("Input verification must work offline")

    original_socket = socket.socket
    socket.socket = no_network
    try:
        root = root.resolve()
        work.ROOT = root
        work.LAB = root / "experiments/future_player_lab"
        for field, name in {
            "DATA": "deep_data_001",
            "FITS": "deep_search_003",
            "CAPACITY": "deep_capacity_002",
            "REPORT": "deep_report_001",
            "DIAGNOSTICS": "deep_diagnostics_001",
        }.items():
            setattr(work, field, work.LAB / "runs" / name)
        gold = load_gold(root / "data", "canonical_targets_20260923_r3")
        panel, x = work.task_data("RB", "season")
        weeks = work.target_weeks()
        events, schedule, preseason, injuries = availability.load_inputs(root)
        comparisons = {
            name: work.one_forecast("RB", "season", name).height
            for name in ("persistence", "auto_lgb_all", "new:top_3")
        }
        original_panel, original_x, _, _ = work.sandbox_data("QB", "next_four", "original_profile")
        raw.ROOT = root
        raw.SNAPSHOT = root / "data/raw/snapshots/canonical_20260923_r4/manifest.json"
        catalog = raw.catalog()
        result = saved.load_saved(work.LAB / "runs/raw_distributions_001")
        # Every pinned input hash used by the prepared-data builder must be restorable.
        prepared = json.loads((work.DATA / "manifest.json").read_text())
        for name, expected in prepared["inputs"].items():
            if work.digest(root / name) != expected:
                raise ValueError(f"Prepared input differs: {name}")
        auxiliary = {}
        source_catalog = root / "data/source_catalog.json"
        if source_catalog.exists():
            import pyarrow.parquet as pq

            for source_id, ref in json.loads(source_catalog.read_text())["sources"].items():
                release = load_gold(root / "data", ref["version"])
                if release.ref != ref or release.manifest["source"]["id"] != source_id:
                    raise ValueError(f"Source catalog mismatch: {source_id}")
                tables = release.manifest["tables"]
                for name, spec in tables.items():
                    if pq.ParquetFile(release.path(name)).metadata.num_rows != spec["rows"]:
                        raise ValueError(f"Source row count mismatch: {source_id}/{name}")
                auxiliary[source_id] = {"version": ref["version"], "tables": len(tables)}
        return {
            "auxiliary_sources_verified": auxiliary,
            "network": "disabled",
            "training": "not run",
            "gold_tables": len(gold.manifest["tables"]),
            "rb_examples": panel.height,
            "rb_matrix_shape": list(x.shape),
            "weekly_target_rows": weeks.height,
            "reviewed_absence_events": len(events),
            "injury_rows": injuries.height,
            "schedule_rows": schedule.height,
            "preseason_rows": preseason.height,
            "saved_comparison_rows": comparisons,
            "original_model_examples": original_panel.height,
            "original_model_inputs": original_x.shape[1],
            "raw_assets": catalog.height,
            "saved_distribution_rows": result["rows"].height,
            "prepared_input_hashes_verified": len(prepared["inputs"]),
        }
    finally:
        socket.socket = original_socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
