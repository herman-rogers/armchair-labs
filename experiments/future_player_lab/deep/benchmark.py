"""Re-score saved production forecasts against position-independent player outcomes."""

import json

import numpy as np
import polars as pl

from ..data import digest
from ..run import write_json
from .data import KEYS, verified_files

EXISTING = (
    "prior",
    "current",
    "blend",
    "calibrated_current",
    "calibrated_blend",
    "profile_ridge",
    "profile_boost",
    "enriched_boost",
    "policy",
)


def existing_library(root, config):
    directory = config["published_root"]
    hashes = verified_files(
        root,
        directory,
        ["ranking_predictions.parquet", "qb_variations/ranking_predictions.parquet"],
        config["pinned_manifests"],
    )

    def keys(frame):
        return frame.rename({"season": "year", "through_week": "origin"}).with_columns(
            pl.col("horizon").replace({"next4": "next_four", "rest_of_season": "remaining"})
        )

    base = keys(pl.read_parquet(root / directory / "ranking_predictions.parquet"))
    variations = keys(
        pl.read_parquet(root / directory / "qb_variations/ranking_predictions.parquet")
    )
    columns = [c for c in variations.columns if c.startswith("points_")]
    if len(columns) != 36:
        raise ValueError("Review changed QB variation library")
    base = base.select(*KEYS, *[pl.col(c).alias("existing:" + c) for c in EXISTING])
    variations = variations.select(*KEYS, *[pl.col(c).alias("existing:" + c) for c in columns])
    return base.join(variations, on=KEYS, how="left", validate="1:1"), hashes


def reconcile_published(benchmark, weeks):
    weeks = weeks.filter(pl.col("season_type") == "REG")
    if weeks.select("player_id", "season", "week").is_duplicated().any():
        raise ValueError("Duplicate weekly gold outcome")
    lookup = {}
    for key, block in weeks.partition_by(["player_id", "season"], as_dict=True).items():
        lookup[key] = (
            block["week"].to_numpy(),
            block["league_points"].to_numpy(),
            block["position"].is_in(["QB", "RB", "WR", "TE"]).fill_null(False).to_numpy(),
        )
    corrected, changes = [], []
    for row in benchmark.iter_rows(named=True):
        if row["published_actual"] is None:
            corrected.append(None)
            continue
        week, points, allowed = lookup.get(
            (row["player_id"], row["year"]), (np.array([]), np.array([]), np.array([], dtype=bool))
        )
        future = (week > row["origin"]) & (week <= row["published_end"])
        old = float(points[future & allowed].sum())
        total = float(points[future].sum())
        if not np.isclose(old, row["published_actual"], rtol=0, atol=1e-7):
            raise ValueError(
                "Published labels differ for a reason other than the verified position filter"
            )
        corrected.append(total)
        if not np.isclose(total, old, rtol=0, atol=1e-7):
            changes.append(
                {
                    **{k: row[k] for k in ["player_id", "position", "year", "origin", "horizon"]},
                    "original_actual": old,
                    "canonical_actual": total,
                    "reason": "Original outcome feed excludes weekly positions outside QB/RB/WR/TE",
                }
            )
    return benchmark.rename({"published_actual": "published_original_actual"}).with_columns(
        pl.Series("published_actual", corrected)
    ), changes


def prepare_benchmark(root, data, output):
    config = json.loads((data / "config.json").read_text())
    gold = root / config["gold_root"]
    if digest(gold / "manifest.json") != config["gold_manifest_sha256"]:
        raise ValueError("Gold manifest changed")
    manifest = json.loads((gold / "manifest.json").read_text())
    spec = manifest["tables"]["nfl_player_weeks"]
    if digest(gold / spec["path"]) != spec["sha256"]:
        raise ValueError("Weekly gold changed")
    revised, changes = reconcile_published(
        pl.read_parquet(data / "published.parquet"), pl.read_parquet(gold / spec["path"])
    )
    library, library_hashes = existing_library(root, config)
    revised = revised.join(library, on=KEYS, how="left", validate="1:1")
    revised.write_parquet(output / "published_reconciled.parquet")
    write_json(
        output / "published_target_audit.json",
        dict(
            rows=revised.height,
            changes=changes,
            changed_rows=len(changes),
            predictions_changed=False,
            original_benchmark_sha256=digest(data / "published.parquet"),
            gold_week_sha256=spec["sha256"],
            existing_library_hashes=library_hashes,
            policy=(
                "Score all saved predictions against all future player points; "
                "no outcome-based row exclusions"
            ),
        ),
    )
    return revised
