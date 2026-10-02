"""In-season rookie analogs. Pure, exploratory, and separate from draft forecasts.

Compare the same position at the same calendar-week cutoff, training on earlier
seasons only. Outcomes include zero-scoring absences/byes over four calendar weeks.
No future observation determines eligibility, features, scaling, or neighbors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl

POSITIONS = ("QB", "RB", "WR", "TE")
NEIGHBORS = 25
MIN_HISTORY = 20


def completed_cutoff(schedule: pl.DataFrame, weeks: pl.DataFrame, season: int) -> int:
    """Only consecutive finished weeks with stat coverage for every scheduled team."""
    games = schedule.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    stats = weeks.filter(pl.col("season") == season)
    cutoff = 0
    for week in range(1, 19):
        now = games.filter(pl.col("week") == week)
        if not now.height or now["home_score"].null_count() or now["away_score"].null_count():
            break
        teams = set(now["home_team"]) | set(now["away_team"])
        if not teams.issubset(set(stats.filter(pl.col("week") == week)["team"])):
            break
        cutoff = week
    return cutoff


def _unique(frame: pl.DataFrame, keys: list[str]) -> None:
    if frame.height and frame.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError(f"Duplicate rookie input keys: {keys}")


def rookie_rows(
    rookies: pl.DataFrame,
    weeks: pl.DataFrame,
    snaps: pl.DataFrame,
    cutoff: int,
    *,
    completed_seasons: set[int],
) -> list[dict[str, Any]]:
    """Retain unobserved rookies with null forecasts; never infer games from box scores.

    The caller certifies complete season/week source coverage. A missing scoring row
    then means no recorded points, not a missed game. Snap share stays null if absent.
    """
    if not 1 <= cutoff <= 13:
        raise ValueError("Rookie research requires a completed cutoff from Week 1 to 13")
    _unique(rookies, ["player_id", "season"])
    _unique(weeks, ["player_id", "season", "week"])
    _unique(snaps, ["player_id", "season", "week"])
    point_map: dict[tuple, dict] = defaultdict(dict)
    snap_map: dict[tuple, dict] = defaultdict(dict)
    for row in weeks.to_dicts():
        point_map[row["player_id"], row["season"]][row["week"]] = row
    for row in snaps.to_dicts():
        snap_map[row["player_id"], row["season"]][row["week"]] = row
    rows = []
    for player in rookies.to_dicts():
        if player["position"] not in POSITIONS:
            continue
        season = int(player["season"])
        observed = point_map[player["player_id"], season]
        past = [r for w, r in observed.items() if 1 <= w <= cutoff]
        snap_past = [
            r
            for w, r in snap_map[player["player_id"], season].items()
            if 1 <= w <= cutoff and r.get("offense_snaps", 0) > 0
        ]

        def total(field: str, past: list[dict] = past) -> float:
            return sum(float(r.get(field) or 0) for r in past)

        shares = [float(r["offense_pct"]) for r in snap_past if r.get("offense_pct") is not None]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in shares):
            raise ValueError("Offensive snap shares must be finite fractions")
        latest = observed.get(cutoff, {})
        pick = player.get("draft_pick")
        points = total("league_points")
        rows.append(
            {
                **player,
                "cutoff_week": cutoff,
                "observed_stat_weeks": len(past),
                "snap_observations": len(shares),
                "eligible": bool(snap_past)
                or any(
                    any(
                        float(r.get(f) or 0) != 0
                        for f in (
                            "league_points",
                            "targets",
                            "carries",
                            "receptions",
                        )
                    )
                    for r in past
                ),
                "points_to_date": points,
                "points_per_week": points / cutoff,
                "targets": total("targets"),
                "targets_per_week": total("targets") / cutoff,
                "carries": total("carries"),
                "carries_per_week": total("carries") / cutoff,
                "receptions": total("receptions"),
                "latest_targets": float(latest.get("targets") or 0),
                "latest_carries": float(latest.get("carries") or 0),
                "snap_share": float(np.mean(shares)) if shares else None,
                "log_pick": math.log(float(pick) if pick and pick > 0 else 300),
                "draft_pick_missing": float(not pick or pick <= 0),
                "next4_actual": sum(
                    float(observed.get(w, {}).get("league_points") or 0)
                    for w in range(cutoff + 1, cutoff + 5)
                )
                if season in completed_seasons
                else None,
            }
        )
    return rows


def features(position: str) -> list[str]:
    usage = (
        ["targets_per_week", "latest_targets"]
        if position in {"WR", "TE"}
        else [
            "carries_per_week",
            "latest_carries",
        ]
    )
    if position == "RB":
        usage += ["targets_per_week", "latest_targets"]
    return ["points_per_week", *usage, "snap_share", "log_pick", "draft_pick_missing"]


def analog_forecast(row: dict, history: list[dict]) -> dict:
    pool = sorted(
        [
            r
            for r in history
            if r["season"] < row["season"]
            and r["position"] == row["position"]
            and r["cutoff_week"] == row["cutoff_week"]
            and r["eligible"]
            and r["next4_actual"] is not None
        ],
        key=lambda r: (r["season"], r["player_id"]),
    )
    result = {
        "forecast_next4": None,
        "analog_p10": None,
        "analog_p90": None,
        "history_count": len(pool),
        "neighbor_count": 0,
        "analogs": [],
        "pace_next4": row["points_per_week"] * 4 if row["eligible"] else None,
        "historical_mean_next4": None,
        "forecast_status": "No observed offensive production or snaps",
    }
    if not row["eligible"]:
        return result
    if len(pool) < MIN_HISTORY:
        return {**result, "forecast_status": "Insufficient earlier rookie history"}
    names = features(row["position"])
    # If today's snap feed has no observation, compare using the other observed
    # features; never manufacture a zero snap share. Training-only imputation.
    names = [name for name in names if row.get(name) is not None]
    x = np.array([[r.get(f, np.nan) for f in names] for r in pool], dtype=float)
    current = np.array([row[f] for f in names], dtype=float)
    missing = ~np.isfinite(x)
    for j in range(len(names)):
        valid = x[np.isfinite(x[:, j]), j]
        x[missing[:, j], j] = np.median(valid) if valid.size else 0
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1
    distance = np.mean(((x - current) / scale) ** 2, axis=1) + missing.mean(axis=1)
    indices = np.argsort(distance, kind="stable")[:NEIGHBORS]
    neighbors = [pool[int(i)] for i in indices]
    values = np.array([r["next4_actual"] for r in neighbors], dtype=float)
    return {
        **result,
        "forecast_next4": float(values.mean()),
        "analog_p10": float(np.quantile(values, 0.1)),
        "analog_p90": float(np.quantile(values, 0.9)),
        "neighbor_count": len(neighbors),
        "historical_mean_next4": float(np.mean([r["next4_actual"] for r in pool])),
        "forecast_status": "Exploratory historical analog forecast",
        "analogs": [
            {
                "player_display_name": r["player_display_name"],
                "season": r["season"],
                "points_per_week": r["points_per_week"],
                "targets": r["targets"],
                "carries": r["carries"],
                "snap_share": r["snap_share"],
                "next4_actual": r["next4_actual"],
            }
            for r in neighbors[:5]
        ],
    }


def backtest(rows: list[dict], first_season: int = 2018) -> dict:
    """Walk forward at one fixed cutoff. Each rookie appears once per evaluation."""
    predictions = []
    for row in rows:
        if row["season"] < first_season or row["next4_actual"] is None or not row["eligible"]:
            continue
        forecast = analog_forecast(row, rows)
        predictions.append({**row, **forecast})
    summaries = []
    for position in POSITIONS:
        candidates = [r for r in predictions if r["position"] == position]
        scored = [r for r in candidates if r["forecast_next4"] is not None]
        if not candidates:
            continue
        summary = {"position": position, "eligible": len(candidates), "scored": len(scored)}
        for key in ("forecast_next4", "pace_next4", "historical_mean_next4"):
            errors = [r[key] - r["next4_actual"] for r in scored]
            summary[key + "_mae"] = float(np.mean(np.abs(errors))) if errors else None
        summary["seasons"] = sorted({r["season"] for r in scored})
        summary["folds"] = [
            {
                "season": season,
                "players": sum(r["season"] == season for r in scored),
                "analog_mae": float(
                    np.mean(
                        [
                            abs(r["forecast_next4"] - r["next4_actual"])
                            for r in scored
                            if r["season"] == season
                        ]
                    )
                ),
                "pace_mae": float(
                    np.mean(
                        [
                            abs(r["pace_next4"] - r["next4_actual"])
                            for r in scored
                            if r["season"] == season
                        ]
                    )
                ),
            }
            for season in summary["seasons"]
        ]
        summaries.append(summary)
    return {
        "positions": summaries,
        "first_season": first_season,
        "basis": "Earlier seasons only; same cutoff and scored pool for every baseline.",
    }
