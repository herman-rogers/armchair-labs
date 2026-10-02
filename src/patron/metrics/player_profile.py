"""Canonical observed player histories, independent of any forecasting model.

All summaries use an explicit season/week cutoff. Missing participation is unknown;
college production and NFL points never share a denominator. Role bands describe
observed offensive participation, not a starter designation or a talent rating.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any

import polars as pl

KEYS = ["player_id", "season", "week"]
TOTALS = (
    "league_points",
    "bonus_pts",
    "attempts",
    "passing_yards",
    "passing_tds",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)
METRICS = {
    "league_points": {
        "label": "League points",
        "unit": "points",
        "definition": "Audited NFL scoring including the league's long-touchdown bonuses.",
    },
    "observed_weeks": {
        "label": "Observed weeks",
        "unit": "weeks",
        "definition": (
            "Weeks with a scoring-source record or positive offensive snaps. Not a "
            "medical availability or starts count."
        ),
    },
    "points_per_observed_week": {
        "label": "Points / observed week",
        "unit": "points/week",
        "definition": (
            "League points divided by observed weeks; unknown/offense-unobserved weeks "
            "are not invented games."
        ),
    },
    "offensive_weeks": {
        "label": "Offensive weeks",
        "unit": "weeks",
        "definition": (
            "Weeks with observed positive offensive snaps. Null when snap evidence is "
            "absent; counts may be partial."
        ),
    },
    "snap_share": {
        "label": "Mean offensive snap share",
        "unit": "fraction",
        "definition": (
            "Mean of recorded offensive snap shares in observed weeks. Includes recorded "
            "zero shares; not routes run."
        ),
    },
    "targets_per_observed_week": {
        "label": "Targets / observed week",
        "unit": "targets/week",
        "definition": (
            "Recorded targets divided by observed weeks, only when targets cover every such week."
        ),
    },
    "carries_per_observed_week": {
        "label": "Carries / observed week",
        "unit": "carries/week",
        "definition": (
            "Recorded carries divided by observed weeks, only when carries cover every such week."
        ),
    },
    "receiving_yards_per_target": {
        "label": "Receiving yards / target",
        "unit": "yards/target",
        "definition": (
            "Ratio of totals on weeks with both receiving yards and targets; descriptive "
            "efficiency, not an isolated talent estimate."
        ),
    },
    "rushing_yards_per_carry": {
        "label": "Rushing yards / carry",
        "unit": "yards/carry",
        "definition": (
            "Ratio of totals on weeks with both rushing yards and carries; descriptive efficiency."
        ),
    },
    "passing_yards_per_attempt": {
        "label": "Passing yards / attempt",
        "unit": "yards/attempt",
        "definition": "Ratio of totals on weeks with both passing yards and attempts.",
    },
    "recent3_points_per_observed_week": {
        "label": "Recent three-season baseline",
        "unit": "points/week",
        "definition": (
            "Completed seasons only: points / observed weeks with 0.55 annual decay and "
            "exposure weighting. Current partial season is separate."
        ),
    },
}
ROLE_LABELS = {
    "high": "High participation",
    "rotation": "Rotational participation",
    "limited": "Limited participation",
    "unknown": "Participation unknown",
}


def unique(frame: pl.DataFrame, keys: list[str]) -> None:
    if frame.height and frame.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError(f"Conflicting profile input keys: {keys}")


def finite(value):
    return float(value) if isinstance(value, (float, int)) and math.isfinite(value) else None


def role_band(share):
    if share is None:
        return "unknown"
    if not 0 <= share <= 1:
        raise ValueError("Offensive snap share outside [0, 1]")
    return "high" if share >= 0.70 else "rotation" if share >= 0.35 else "limited"


def before(rows: list[dict], season: int, week: int = 18) -> list[dict]:
    """Filter BEFORE deriving aggregates or role episodes."""
    return [r for r in rows if (r["season"], r["week"]) <= (season, week)]


def summarize(rows: list[dict]) -> dict:
    """One definition for career, season, stint and participation-band summaries."""
    active = [r for r in rows if r.get("stat_recorded") or (r.get("offense_snaps") or 0) > 0]
    shares = [r["snap_share"] for r in active if r.get("snap_share") is not None]
    result: dict[str, Any] = {"observed_weeks": len(active), "snap_observations": len(shares)}
    for field in TOTALS:
        values = [finite(r.get(field)) for r in active]
        valid = [v for v in values if v is not None]
        # A partial career counter is not a complete total. Coverage is kept separately.
        result[field] = sum(valid) if active and len(valid) == len(active) else None
        result[f"{field}_observations"] = len(valid)
    result["offensive_weeks"] = (
        sum((r.get("offense_snaps") or 0) > 0 for r in active)
        if any(r.get("offense_snaps") is not None for r in active)
        else None
    )
    result["snap_share"] = mean(shares) if shares else None
    for field, key in [("league_points", "points"), ("targets", "targets"), ("carries", "carries")]:
        result[f"{key}_per_observed_week"] = (
            result[field] / len(active)
            if active and result[f"{field}_observations"] == len(active)
            else None
        )
    for numerator, denominator, name in [
        ("receiving_yards", "targets", "receiving_yards_per_target"),
        ("rushing_yards", "carries", "rushing_yards_per_carry"),
        ("passing_yards", "attempts", "passing_yards_per_attempt"),
    ]:
        paired = [(finite(r.get(numerator)), finite(r.get(denominator))) for r in active]
        paired = [(n, d) for n, d in paired if n is not None and d is not None]
        count = sum(d for _, d in paired)
        result[name] = sum(n for n, _ in paired) / count if count > 0 else None
        result[f"{name}_sample"] = count
    return result


def nfl_seasons(rows: list[dict], season: int, week: int) -> list[dict]:
    groups = defaultdict(list)
    for row in before(rows, season, week):
        groups[row["season"]].append(row)
    output = []
    for year, observations in sorted(groups.items()):
        teams = list(
            dict.fromkeys(
                r["team"] for r in sorted(observations, key=lambda r: r["week"]) if r.get("team")
            )
        )
        output.append(
            {
                "season": year,
                "teams": teams,
                "positions": sorted({r["position"] for r in observations if r.get("position")}),
                "through_week": max(r["week"] for r in observations),
                "partial": year == season,
                **summarize(observations),
            }
        )
    return output


def role_periods(rows: list[dict], season: int, week: int) -> list[dict]:
    periods: list[list[dict]] = []
    for row in sorted(before(rows, season, week), key=lambda r: (r["season"], r["week"])):
        band = role_band(row.get("snap_share"))
        prior = periods[-1][-1] if periods else None
        if (
            prior
            and (prior["season"], prior.get("team"), role_band(prior.get("snap_share")))
            == (row["season"], row.get("team"), band)
            and row["week"] == prior["week"] + 1
        ):
            periods[-1].append(row)
        else:
            periods.append([row])
    return [
        {
            "season": period[0]["season"],
            "start_week": period[0]["week"],
            "end_week": period[-1]["week"],
            "team": period[0].get("team"),
            "role": role_band(period[0].get("snap_share")),
            **summarize(period),
        }
        for period in periods
    ]


def career_features(rows: list[dict], season: int, week: int = 18) -> dict:
    """Reusable model-ready observations; no future outcomes or forecast aliases."""
    history = before(rows, season, week)
    completed = [r for r in history if r["season"] < season]
    seasons = nfl_seasons(completed, season, week)
    recent = [
        r
        for r in completed
        if r["season"] >= season - 3
        and (r.get("stat_recorded") or (r.get("offense_snaps") or 0) > 0)
    ]
    weighted = [(r, 0.55 ** (season - 1 - r["season"])) for r in recent]
    numerator = sum((r.get("league_points") or 0) * w for r, w in weighted)
    denominator = sum(w for _, w in weighted)
    summary = summarize(completed)
    result = {
        "completed_seasons_observed": len(seasons),
        "completed_observed_weeks": summary["observed_weeks"],
        "career_points_per_observed_week": summary["points_per_observed_week"],
        "career_targets": summary["targets"],
        "career_carries": summary["carries"],
        "career_attempts": summary["attempts"],
        "recent3_points_per_observed_week": numerator / denominator
        if denominator and all(r.get("league_points") is not None for r, _ in weighted)
        else None,
        "recent3_observed_weeks": len(recent),
    }
    for band in ROLE_LABELS:
        subset = [r for r in completed if role_band(r.get("snap_share")) == band]
        segment = summarize(subset)
        result[f"{band}_observed_weeks"] = segment["observed_weeks"]
        result[f"{band}_points_per_observed_week"] = segment["points_per_observed_week"]
    return result


def college_history(rows: list[dict], season: int) -> list[dict]:
    """Keep source stints and incomplete rows; never silently pool transfers."""
    return sorted(
        [r for r in rows if r["season"] < season], key=lambda r: (r["season"], r["team_id"])
    )


def transitions(seasons: list[dict], college: list[dict], identity: dict) -> list[dict]:
    events = []
    for level, rows, team_key in [("college", college, "college_team"), ("NFL", seasons, "teams")]:
        previous = None
        for row in rows:
            teams = row.get(team_key) or []
            teams = teams if isinstance(teams, list) else [teams]
            if previous and set(teams) != set(previous[1]):
                events.append(
                    {
                        "season": row["season"],
                        "kind": "team_change",
                        "level": level,
                        "description": (
                            f"Observed team change: {', '.join(previous[1])} → {', '.join(teams)}"
                        ),
                    }
                )
            if teams:
                previous = (row["season"], teams)
    entry = identity.get("rookie_season")
    if entry:
        events.append(
            {
                "season": entry,
                "kind": "nfl_entry",
                "level": "NFL",
                "description": "Recorded NFL entry season",
            }
        )
    for old, new in zip(seasons, seasons[1:], strict=False):
        if new["season"] != old["season"] + 1 or new["partial"]:
            continue
        if min(old["snap_observations"], new["snap_observations"]) >= 4:
            delta = new["snap_share"] - old["snap_share"]
            if abs(delta) >= 0.15:
                events.append(
                    {
                        "season": new["season"],
                        "kind": "role_change",
                        "level": "NFL",
                        "description": (
                            f"Mean offensive snap share changed {delta * 100:+.0f} "
                            "percentage points"
                        ),
                    }
                )
    return sorted(events, key=lambda r: (r["season"], r["kind"]))
