"""Score a frozen delivery's QB rates, yards and fantasy points on later outcomes."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from engine.data.nextgen import load_analysis
from engine.data.releases import digest, identifier, load_gold, reference, write_json


def score(data, source_version, version):
    source, manifest = load_analysis(
        data, reference(data / "research" / identifier(source_version))
    )
    if manifest.get("qb_variations_schema_version") != 1:
        raise ValueError("A frozen QB variation delivery is required")
    frozen_gold = load_gold(data, manifest["gold"]["version"])
    gold = load_gold(data)
    report = json.loads((source / "qb_variations/report.json").read_text())
    year = report["season"]
    observed = gold.manifest["current_observations"]
    through = (
        observed["through_week"]
        if observed["season"] == year
        else 99
        if observed["season"] > year
        else -1
    )
    cols = [
        "player_id",
        "season",
        "season_type",
        "week",
        "passing_yards",
        "attempts",
        "league_points",
    ]
    weeks = pl.concat(
        [gold.read("nfl_player_weeks").select(cols), gold.read("current_weeks").select(cols)],
        how="vertical_relaxed",
    ).unique(["player_id", "season", "week"])
    schedule = (
        pl.concat([frozen_gold.read("nfl_schedule"), frozen_gold.read("current_schedule")])
        .unique("game_id")
        .filter((pl.col("season") == year) & (pl.col("game_type") == "REG"))
    )
    pending = excluded = undefined_rates = 0
    outcomes = []
    passing = pl.read_parquet(source / "qb_passing/current.parquet").to_dicts()
    ranking = (
        pl.read_parquet(source / "rankings.parquet").filter(pl.col("position") == "QB").to_dicts()
    )
    previous_rank = {
        (r["player_id"], r["horizon"]): r
        for r in pl.read_parquet(source / "qb_variations/ranking_predictions.parquet")
        .filter(pl.col("season") == year)
        .to_dicts()
    }
    for row in passing + ranking:
        is_passing = "execution_ypa" in row
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
        first = games["gameday"].min() if games.height else None
        # Date-only schedules cannot certify a same-day forecast preceded kickoff.
        issue = row.get("issued_at", manifest["generated_at"])
        if first is None or issue[:10] >= str(first)[:10]:
            excluded += 1
            continue
        future = weeks.filter(
            (pl.col("player_id") == row["player_id"])
            & (pl.col("season") == year)
            & (pl.col("season_type") == "REG")
            & (pl.col("week") > row["through_week"])
            & (pl.col("week") <= row["end_week"])
        )
        if any(future[c].null_count() for c in ("passing_yards", "attempts", "league_points")):
            raise ValueError("Undefined prospective QB outcomes")
        common = dict(player_id=row["player_id"], horizon=row["horizon"], season=year)
        if is_passing:
            constrained = row.get("constraint_reason") or row.get("retired_known")
            outcomes.append(
                dict(
                    common,
                    target="yards",
                    actual=future["passing_yards"].sum(),
                    prediction=row["prediction"],
                    reference=0.0 if constrained else row["reference"],
                    weight=1.0,
                )
            )
            attempts = future["attempts"].sum()
            if attempts:
                outcomes.append(
                    dict(
                        common,
                        target="rate",
                        actual=future["passing_yards"].sum() / attempts,
                        prediction=row["execution_ypa"],
                        reference=row["execution_reference"],
                        weight=float(attempts),
                    )
                )
            else:
                undefined_rates += 1
        else:
            old = previous_rank[row["player_id"], row["horizon"]]
            outcomes.append(
                dict(
                    common,
                    target="league_points",
                    actual=future["league_points"].sum(),
                    prediction=row["prediction"],
                    reference=old["reference"] if row["rank_eligible"] else 0.0,
                    weight=1.0,
                )
            )
    summaries = []
    for target, horizon in sorted({(r["target"], r["horizon"]) for r in outcomes}):
        rows = [r for r in outcomes if (r["target"], r["horizon"]) == (target, horizon)]
        for method in ("prediction", "reference"):
            error = np.array([r[method] - r["actual"] for r in rows])
            summaries.append(
                dict(
                    target=target,
                    horizon=horizon,
                    model=method,
                    n=len(rows),
                    mae=float(np.abs(error).mean()),
                    mse=float(np.average(error**2, weights=[r["weight"] for r in rows])),
                )
            )
    root = data / "research" / identifier(version)
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
            undefined_rates=undefined_rates,
            comparisons=summaries,
            outcomes=outcomes,
            serving="research_only",
            note="Frozen forecasts; no refitting or promotion. "
            "Same-day issues require verified kickoff times.",
        ),
    )
    write_json(
        root / "manifest.json",
        dict(
            version=version,
            kind="qb_variations_prospective_score",
            status="complete",
            files={"report.json": digest(root / "report.json")},
        ),
    )
    print(json.dumps(dict(scored=len(outcomes), pending=pending, excluded=excluded)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    score(args.data, args.source, args.version)
