"""Profile forecasts obey the same published registry as the NextGen workspace."""

from pathlib import Path

import polars as pl

from patron.data.frames import read_frame
from patron.data.nextgen import eligible, load_analysis, published, read_json
from patron.data.releases import load_gold


def approved_profile_forecasts(
    data: Path,
    report: dict,
    player_id: str | None,
    season: int,
    week: int,
) -> tuple[list[dict], str | None]:
    if not published(data):
        return [], None
    root, manifest = load_analysis(data)
    if manifest["profiles"]["version"] != report["version"] or manifest["gold"] != report.get(
        "gold"
    ):
        raise ValueError("Profile and approved forecasts use different releases")
    version = manifest["version"]
    if not player_id or week == 0:
        return [], version
    gold = load_gold(data, manifest["gold"]["version"])
    # A season/week request must not expose a forecast dated later in that season.
    schedule = gold.read("nfl_schedule")
    if season == report["season"]:
        schedule = pl.concat([schedule, gold.read("current_schedule")], how="diagonal_relaxed")
    dates = (
        schedule.filter(
            (pl.col("season") == season) & (pl.col("game_type") == "REG") & (pl.col("week") <= week)
        )["gameday"]
        .cast(pl.String)
        .str.to_date()
    )
    through = dates.max()
    if through is None:
        return [], version
    rows = (
        read_frame(root / "predictions.parquet", player_id=player_id)
        .lazy()
        .filter(
            (pl.col("player_id") == player_id)
            & (pl.col("season") == season)
            & (pl.col("cutoff").cast(pl.String).str.to_date() <= through)
        )
        .collect()
        .to_dicts()
    )
    registry = {entry["id"]: entry for entry in read_json(root / "registry.json")}
    incidents = read_json(root / "incidents.json")
    forecasts = []
    for row in rows:
        entry = registry.get(f"forecast:{row['model']}:{row['target']}")
        if (
            not entry
            or not eligible(
                entry,
                use="forecast",
                target=row["target"],
                position=row["position"],
                population=row.get("population"),
                horizon=row["horizon"],
                incidents=incidents,
            )[0]
        ):
            continue
        # A full-season outcome cannot be exposed in a midseason profile.
        complete = bool(row["complete"] and season < report["season"] and week == 18)
        forecasts.append(
            {
                **row,
                "actual": row["actual"] if complete else None,
                "complete": complete,
                "label": entry["label"],
                "serving": entry["serving"],
                "reason": entry["reason"],
                "analysis_version": version,
            }
        )
    return sorted(forecasts, key=lambda row: (row["target"], row["model"])), version
