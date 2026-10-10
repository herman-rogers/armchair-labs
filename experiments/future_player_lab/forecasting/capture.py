"""Capture current provider inputs once, then forecast offline from verified bytes."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from ..gpu_lab.run import digest, json_write


def capture(output, season, ngs_snapshot=None):
    import nflreadpy as nfl

    from engine.data.nflverse import configure_cache

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configure_cache()
    loaders = {
        "plays": lambda: nfl.load_pbp([season]),
        "team_box": lambda: nfl.load_team_stats([season], summary_level="week"),
        "ngs_rushing": lambda: nfl.load_nextgen_stats([season], stat_type="rushing"),
        "ngs_receiving": lambda: nfl.load_nextgen_stats([season], stat_type="receiving"),
        "ngs_passing": lambda: nfl.load_nextgen_stats([season], stat_type="passing"),
        "rosters": lambda: nfl.load_rosters_weekly([season]),
        "injuries": lambda: nfl.load_injuries([season]),
        "depth": lambda: nfl.load_depth_charts([season]),
    }
    source_refs = {}
    if ngs_snapshot is not None:
        import polars as pl

        from ..notebooks.raw_distribution_data import ROOT

        saved = json.loads(Path(ngs_snapshot).read_text())
        if saved.get("status") != "accepted":
            raise ValueError("NGS snapshot must be accepted")
        for stat in ("rushing", "receiving", "passing"):
            asset = saved["assets"][f"ngs_{stat}.parquet"]
            source = ROOT / "data" / asset["object"]
            if digest(source) != asset["sha256"]:
                raise ValueError("NGS source hash mismatch")
            loaders[f"ngs_{stat}"] = lambda source=source: pl.read_parquet(source)
            source_refs[f"ngs_{stat}"] = {
                "raw_manifest": str(Path(ngs_snapshot).resolve()),
                "manifest_sha256": digest(ngs_snapshot),
                "asset_sha256": asset["sha256"],
            }
    manifest = {
        "schema_version": 1,
        "status": "capturing",
        "season": season,
        "provider": "nflverse via nflreadpy (provider cache permitted)",
        "captured_at": datetime.now(UTC).isoformat(),
        "sources": {},
        "notice": "Capture time is not publication time; reconstructed inputs may be revised.",
    }
    for name, loader in loaders.items():
        try:
            frame = loader()
            path = output / f"{name}.parquet"
            frame.write_parquet(path)
            manifest["sources"][name] = {
                "status": "available",
                "path": path.name,
                "sha256": digest(path),
                "rows": frame.height,
                "columns": frame.columns,
                "source_reference": source_refs.get(name),
            }
            print(f"Captured {name}: {frame.height:,} rows", flush=True)
        except Exception as error:
            manifest["sources"][name] = {"status": "unavailable", "error": str(error)}
            print(f"Unavailable {name}: {error}", flush=True)
        json_write(output / "manifest.json", manifest)
    manifest["status"] = "complete"
    json_write(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--ngs-snapshot", type=Path)
    args = parser.parse_args()
    capture(args.output, args.season, args.ngs_snapshot)


if __name__ == "__main__":
    main()
