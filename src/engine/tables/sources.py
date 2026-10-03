"""Import validated pipeline outputs into the ordinary named-table catalog.

Legacy release layout is confined to the release loader. Observed schedules are
separate from the schedule features used in pregame forecasting.
"""

from pathlib import Path

from engine.data.releases import TableRelease, load_manifest


def source_path(data_dir: Path, release: TableRelease, name: str) -> Path:
    if name in release.manifest["tables"]:
        return release.path(name)
    # Older batches retained postgame QB identities/scores only in enrichment.
    schedules = {
        "nfl_observed_schedule": "nfl_schedule",
        "current_observed_schedule": "current_schedule",
    }
    if name not in schedules:
        raise ValueError(f"Source batch has no table: {name}")
    root, manifest = load_manifest(data_dir, "enriched", release.manifest["input"])
    return root / manifest["tables"][schedules[name]]
