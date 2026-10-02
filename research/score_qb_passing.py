"""Score a frozen QB passing issue on later observations; never refit or promote it."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from patron.data.releases import digest, identifier, inside, load_gold, reference, write_json
from patron.metrics.qb_passing import MODELS


def score(data: Path, source_version: str, output_version: str):
    source = data / "research" / identifier(source_version)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("kind") != "qb_passing_research" or manifest.get("status") != "complete":
        raise ValueError("A complete frozen QB research issue is required")
    for name, expected in manifest["files"].items():
        if digest(inside(source, name)) != expected:
            raise ValueError(f"Frozen issue changed: {name}")
    gold = load_gold(data)
    frozen_gold = load_gold(data, manifest["gold"]["version"])
    report = json.loads((source / "report.json").read_text())
    season = report["season"]
    observed = gold.manifest["current_observations"]
    through = (
        observed["through_week"]
        if observed["season"] == season
        else 99
        if observed["season"] > season
        else -1
    )
    cols = ["player_id", "season", "week", "passing_yards", "season_type"]
    weeks = pl.concat(
        [gold.read("nfl_player_weeks").select(cols), gold.read("current_weeks").select(cols)],
        how="vertical_relaxed",
    ).unique(["player_id", "season", "week"])
    schedule = (
        pl.concat([frozen_gold.read("nfl_schedule"), frozen_gold.read("current_schedule")])
        .unique("game_id")
        .filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    )
    frozen = pl.read_parquet(source / "current.parquet")
    predictions = pl.read_parquet(source / "predictions.parquet").filter(
        (pl.col("season") == season) & (pl.col("through_week") == report["through_week"])
    )
    by_key = {(r["player_id"], r["horizon"]): r for r in predictions.to_dicts()}
    outcomes = []
    pending = excluded = 0
    for row in frozen.to_dicts():
        if row["end_week"] > through:
            pending += 1
            continue
        games = schedule.filter(
            (pl.col("week") > row["through_week"]) & (pl.col("week") <= row["end_week"])
        )
        if row["schedule_known"]:
            games = games.filter(
                (pl.col("home_team") == row["team"]) | (pl.col("away_team") == row["team"])
            )
        first_day = games["gameday"].min() if games.height else None
        # Day-only schedules cannot certify that a same-day issue preceded kickoff.
        if first_day is None or row["issued_at"][:10] >= str(first_day)[:10]:
            excluded += 1
            continue
        target = weeks.filter(
            (pl.col("player_id") == row["player_id"])
            & (pl.col("season") == season)
            & (pl.col("week") > row["through_week"])
            & (pl.col("week") <= row["end_week"])
            & (pl.col("season_type") == "REG")
        )
        if target["passing_yards"].null_count():
            raise ValueError("Undefined future passing outcome")
        raw = by_key[row["player_id"], row["horizon"]]
        values = {m: 0.0 if row["constraint_reason"] else raw[m] for m in MODELS}
        outcomes.append(
            dict(
                player_id=row["player_id"],
                horizon=row["horizon"],
                actual=target["passing_yards"].sum(),
                reference=row["prediction"],
                **values,
            )
        )
    summaries = []
    for horizon in sorted({r["horizon"] for r in outcomes}):
        rows = [r for r in outcomes if r["horizon"] == horizon]
        for model in (*MODELS, "reference"):
            error = np.array([r[model] - r["actual"] for r in rows])
            summaries.append(
                dict(
                    horizon=horizon,
                    model=model,
                    n=len(rows),
                    mae=float(np.abs(error).mean()),
                    mse=float((error**2).mean()),
                )
            )
    root = data / "research" / identifier(output_version)
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "report.json",
        dict(
            source=reference(source),
            gold=gold.ref,
            scored_at=datetime.now(UTC).isoformat(),
            status="partial_observation" if outcomes else "pending_outcomes",
            scored=len(outcomes),
            pending=pending,
            excluded_same_day_or_unknown_timing=excluded,
            comparisons=summaries,
            outcomes=outcomes,
            serving="research_only",
            note="No refitting, promotion or independence claim. Same-day origins lack "
            "verified kickoff timestamps; excluded conservatively.",
        ),
    )
    write_json(
        root / "manifest.json",
        dict(
            version=output_version,
            kind="qb_passing_prospective_score",
            status="complete",
            files={"report.json": digest(root / "report.json")},
        ),
    )
    print(json.dumps(dict(scored=len(outcomes), pending=pending, excluded=excluded)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    score(args.data, args.source, args.version)
