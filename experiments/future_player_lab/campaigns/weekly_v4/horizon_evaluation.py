"""Score immutable component and total forecasts only after their labels mature.

Whole-horizon actuals follow a player across teams, as in horizon training. They
therefore wait for the entire regular season, even if the origin team's final
scheduled game is earlier. No partial-season outcome is scored as a final total.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..weekly_v3.data import ROOT, TEAM_ALIASES, score_points
from ..weekly_v3.lineups import empirical_crps, interval_score
from ..weekly_v31.artifacts import digest, timestamp, verify


def _assets(snapshot):
    document = json.loads(Path(snapshot).read_text())
    if document.get("status") != "accepted":
        raise ValueError("Horizon outcomes require an accepted observation snapshot")
    root = (ROOT / "data").resolve()
    assets = {}
    for name in ("weeks", "schedule"):
        reference = document["assets"][f"current/{name}.parquet"]
        path = (root / reference["object"]).resolve()
        if not path.is_relative_to(root) or digest(path) != reference["sha256"]:
            raise ValueError(f"Accepted outcome asset changed: {name}")
        assets[name] = pd.read_parquet(path)
    season = int(document["current_observations"]["season"])
    cutoff = int(document["current_observations"]["through_week"])
    box = assets["weeks"]
    box = box[box.season.eq(season) & box.season_type.eq("REG") & box.week.le(cutoff)].copy()
    schedule = assets["schedule"]
    schedule = schedule[schedule.season.eq(season) & schedule.game_type.eq("REG")].copy()
    if schedule.empty:
        raise ValueError("No regular-season schedule for accepted outcome season")
    box["points"] = score_points(box)
    if {"targets", "receptions"} <= set(box):
        box.loc[(box.targets < box.receptions) | (box.targets < 0), "targets"] = np.nan
    complete = set()
    for week, games in schedule.groupby("week"):
        expected = {TEAM_ALIASES.get(t, t) for t in set(games.home_team) | set(games.away_team)}
        observed = {TEAM_ALIASES.get(t, t) for t in box.loc[box.week.eq(week), "team"].dropna()}
        if (
            week <= cutoff
            and games[["home_score", "away_score"]].notna().all().all()
            and expected <= observed
        ):
            complete.add(int(week))
    return season, cutoff, box, schedule, complete


def _distribution(draws, prediction, actual):
    values = np.asarray(draws, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Frozen point distribution must contain finite samples")
    if not np.isclose(values.mean(), prediction, rtol=1e-10, atol=1e-8):
        raise ValueError("Frozen distribution mean disagrees with its point forecast")
    result = {"crps": empirical_crps(values, actual)}
    for coverage in (80, 90):
        alpha = 1 - coverage / 100
        lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        result.update(
            {
                f"coverage{coverage}": float(lo <= actual <= hi),
                f"width{coverage}": float(hi - lo),
                f"interval_score{coverage}": float(interval_score(actual, lo, hi, alpha)),
                f"lower_miss{coverage}": float(actual < lo),
                f"upper_miss{coverage}": float(actual > hi),
            }
        )
    return result


def score_projection(directory, snapshot):
    """Return scored/pending rows for all saved weekly and season target forecasts.

    Missing components in an observed box row stay unknown independently of the
    other components. Entirely absent player rows count as zero only for complete
    accepted game weeks. Duplicate player-week identities remain unscored.
    """
    directory, snapshot = Path(directory), Path(snapshot)
    receipt = verify(directory)
    if receipt.get("status") != "prospective" or not timestamp(
        receipt["evidence_cutoff"]
    ) <= timestamp(receipt["generated_at"]) < timestamp(receipt["deadline"]):
        raise ValueError("Component scoring requires an immutable prospective forecast")
    projection = json.loads((directory / "projections.json").read_text())
    season, cutoff, box, schedule, complete = _assets(snapshot)
    final_week = int(schedule.week.max())
    scheduled_weeks = set(schedule.week.astype(int))
    rows = []
    source_hash = digest(snapshot)
    for horizon in ("weekly", "remaining", "season"):
        for player in projection[horizon]:
            origin = int(player["origin_week"])
            week = origin + 1
            same_season = int(player["season"]) == season
            mature = week <= cutoff if horizon == "weekly" else final_week <= cutoff
            required = (
                {week}
                if horizon == "weekly"
                else {w for w in scheduled_weeks if w > origin}
                if horizon == "remaining"
                else scheduled_weeks
            )
            source_complete = required <= complete
            selected = box[box.player_id.eq(player["player_id"])]
            selected = (
                selected[selected.week.eq(week)]
                if horizon == "weekly"
                else (selected[selected.week.gt(origin)] if horizon == "remaining" else selected)
            )
            ambiguous = bool(selected.duplicated(["player_id", "week"]).any())
            for target, prediction in player["predictions"].items():
                record = dict(
                    capture=str(directory),
                    method="v4",
                    player_id=player["player_id"],
                    position=player["position"],
                    season=int(player["season"]),
                    origin_week=origin,
                    horizon=horizon,
                    target=target,
                    prediction=float(prediction),
                    actual=None,
                    status="pending",
                    outcome_snapshot_sha256=source_hash,
                    through_week=cutoff,
                )
                reason = (
                    "season_not_in_current_snapshot"
                    if not same_season
                    else "horizon_not_complete"
                    if not mature
                    else "incomplete_source_week"
                    if not source_complete
                    else "ambiguous_player_week"
                    if ambiguous
                    else None
                )
                if reason is None:
                    if target not in box:
                        reason = "missing_component_column"
                    elif len(selected) and not np.isfinite(selected[target].to_numpy(float)).all():
                        reason = "missing_component_label"
                if reason:
                    record["pending_reason"] = reason
                else:
                    actual = float(selected[target].sum()) if len(selected) else 0.0
                    error = float(prediction) - actual
                    record.update(
                        actual=actual,
                        status="scored",
                        pending_reason=None,
                        error=error,
                        squared_error=error**2,
                        absolute_error=abs(error),
                    )
                    if target == "points":
                        draws = player.get("distribution", {}).get("samples")
                        if draws is not None:
                            record.update(_distribution(draws, float(prediction), actual))
                        else:
                            record["distribution_status"] = "not_saved"
                rows.append(record)
    return pd.DataFrame(rows)
