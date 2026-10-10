"""Build a standalone rankings page from checksum-verified saved campaign forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import polars as pl

from engine.chairstats.store import CampaignStore


def actual_rankings(
    weeks: pl.DataFrame, season: int, through_week: int, positions=("QB", "RB", "WR")
) -> list[dict]:
    observed = weeks.filter(
        (pl.col("season") == season)
        & (pl.col("season_type") == "REG")
        & pl.col("week").is_between(1, through_week)
        & pl.col("position").is_in(positions)
    ).drop_nulls("player_id")
    if observed.is_empty():
        raise ValueError("No actual observations for requested season/cutoff")
    if observed.select(pl.struct("player_id", "week").is_duplicated().any()).item():
        raise ValueError("Duplicate player/week outcomes")
    if observed.filter(
        pl.col("fantasy_points_ppr").is_null() | ~pl.col("fantasy_points_ppr").is_finite()
    ).height:
        raise ValueError("Missing/nonfinite observed fantasy points")
    return (
        observed.sort("week")
        .group_by("player_id")
        .agg(
            pl.col("position").last().alias("actual_position"),
            pl.col("fantasy_points_ppr").sum().round(4).alias("actual_points"),
        )
        .with_columns(
            pl.col("actual_points").rank("min", descending=True).alias("actual_overall_rank"),
            pl.col("actual_points")
            .rank("min", descending=True)
            .over("actual_position")
            .alias("actual_position_rank"),
        )
        .sort(["actual_overall_rank", "player_id"])
        .to_dicts()
    )


def load_actuals(
    manifest_path: Path, data_root: Path, season: int, positions=("QB", "RB", "WR")
) -> dict:
    content = manifest_path.read_bytes()
    manifest = json.loads(content)
    cutoff = manifest["current_observations"]
    if manifest.get("status") != "accepted" or cutoff["season"] != season:
        raise ValueError("Actual results need an accepted snapshot for the requested season")
    frames, sources = {}, {}
    for name in ("weeks", "schedule"):
        ref = manifest["assets"][f"current/{name}.parquet"]
        path = (data_root / ref["object"]).resolve()
        if not path.is_relative_to(data_root.resolve()):
            raise ValueError("Actual source escapes data directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError("Actual source checksum mismatch")
        frames[name] = pl.read_parquet(path)
        sources[name] = ref
    schedule = frames["schedule"]
    type_col = "season_type" if "season_type" in schedule.columns else "game_type"
    completed = schedule.filter(
        (pl.col("season") == season)
        & (pl.col(type_col) == "REG")
        & pl.col("week").is_between(1, cutoff["through_week"])
    )
    if (
        completed.is_empty()
        or completed.filter(pl.col("home_score").is_null() | pl.col("away_score").is_null()).height
    ):
        raise ValueError("Actual cutoff includes incomplete games")
    rows = actual_rankings(frames["weeks"], season, cutoff["through_week"], positions)
    return {
        "season": season,
        "through_week": cutoff["through_week"],
        "scoring": "nflverse fantasy_points_ppr; standard PPR, no league bonuses",
        "population": (
            f"All observed {'/'.join(positions)} players, "
            "including players outside model candidates"
        ),
        "positions": list(positions),
        "ties": "Competition ranks: equal point totals share a rank; next rank skips",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "sources": sources,
        "saved_at": cutoff["saved_at"],
        "rows": rows,
    }


def build(
    root: Path, campaign_id: str, season: int, output: Path, actual_snapshot: Path | None = None
) -> Path:
    store = CampaignStore(root)
    campaign = store.campaign(campaign_id)
    positions = campaign["positions"]
    actuals = (
        load_actuals(actual_snapshot, root.parent, season, positions) if actual_snapshot else None
    )
    actual_by_id = {r["player_id"]: r for r in actuals["rows"]} if actuals else {}
    groups: dict[tuple[str, int], dict] = {}
    for model in campaign["models"]:
        for ref in model["forecasts"]:
            if ref["season"] != season:
                continue
            forecast = store.forecast(campaign_id, model["model_id"], season, ref["origin_week"])
            key = model["horizon"], ref["origin_week"]
            group = groups.setdefault(
                key,
                {
                    "horizon": key[0],
                    "origin_week": key[1],
                    "rows": [],
                    "sources": [],
                },
            )
            group["sources"].append(
                {
                    "model_id": model["model_id"],
                    "path": ref["path"],
                    "sha256": forecast["artifact_sha256"],
                    "metadata": forecast["metadata"],
                }
            )
            group["rows"].extend({**row, "model_id": model["model_id"]} for row in forecast["rows"])
    if not groups:
        raise ValueError(f"No saved forecasts for {season}")
    snapshots = []
    for group in sorted(groups.values(), key=lambda g: g["origin_week"], reverse=True):
        rows = group["rows"]
        if len({r["player_id"] for r in rows}) != len(rows):
            raise ValueError("Player appears in multiple position models for one snapshot")
        rows.sort(key=lambda r: (-r["predictions"]["points"], r["player_id"]))
        counts: Counter = Counter()
        for rank, row in enumerate(rows, 1):
            counts[row["position"]] += 1
            row["overall_rank"] = rank
            row["position_rank"] = counts[row["position"]]
            row.update(actual_by_id.get(row["player_id"], {}))
        group["counts"] = dict(counts)
        snapshots.append(group)
    payload = {
        "campaign_id": campaign_id,
        "campaign_name": campaign["name"],
        "season": season,
        "manifest_sha256": campaign["manifest_sha256"],
        "scoring": campaign["scoring"],
        "snapshots": snapshots,
        "actuals": actuals,
    }
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("rankings_browser.html").read_text()
    document = template.replace("__RANKINGS_DATA__", encoded)
    output.mkdir(parents=True, exist_ok=True)
    (output / "rankings.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    destination = output / "index.html"
    destination.write_text(document)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/campaigns"))
    parser.add_argument("--campaign", default="opening_drive_v1")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actual-snapshot", type=Path)
    args = parser.parse_args()
    print(build(args.root, args.campaign, args.season, args.output, args.actual_snapshot).resolve())


if __name__ == "__main__":
    main()
