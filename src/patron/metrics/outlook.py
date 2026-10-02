"""Four-calendar-week research outlooks with strictly earlier-season validation.

Points already include absences and byes. Participation and role estimates are
separate diagnostics, never multipliers applied to the point forecast a second time.
Offensive participation is not a diagnosis, medical availability, or an injury risk.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import polars as pl

POSITIONS = ("QB", "RB", "WR", "TE")
USAGE = (
    "recent_points_pg",
    "recent_targets_pg",
    "recent_carries_pg",
    "recent_attempts_pg",
    "recent_snap_share",
    "future_team_games",
)
OUTLOOK = (
    *USAGE,
    "season_points_pg",
    "usage_change",
    "snap_change",
    "snap_std",
    "points_std",
    "spike_share",
    "bonus_share",
    "observed_offensive_weeks",
    "past_offensive_fraction",
    "prior_offensive_weeks",
)
SCALE = (
    "recent_points_pg",
    "points_std",
    "snap_std",
    "recent_snap_share",
    "prior_offensive_weeks",
    "future_team_games",
)
TARGETS = ("next4_points", "next4_offensive_weeks", "next4_active_snap_share")


def team_name(value):
    return {"OAK": "LV", "STL": "LA", "LAR": "LA", "SD": "LAC", "WSH": "WAS"}.get(value, value)


def number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def mean(values):
    return float(np.mean(values)) if values else None


def unique(frame, keys):
    if frame.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError(f"Duplicate outlook input keys: {keys}")


def build_rows(
    points: pl.DataFrame,
    snaps: pl.DataFrame,
    schedule: pl.DataFrame,
    candidates: pl.DataFrame,
    cutoffs: list[int],
    *,
    completed_seasons: set[int],
    snap_complete_weeks: set[tuple[int, int]],
    snap_identities: set[str],
    prior_snaps: pl.DataFrame | None = None,
) -> list[dict]:
    """Features use observations at/before the decision. Full outcomes stay by ID.

    Scoring coverage must be certified by the caller. Missing snap rows count as
    no observed offense only with a matched snap identity and complete weekly feed.
    Future snapshots never define candidate eligibility, position, team, or features.
    """
    if any(not 1 <= cutoff <= 18 for cutoff in cutoffs):
        raise ValueError("Outlook cutoffs must be Weeks 1–18")
    points = points.filter(pl.col("player_id").is_not_null())
    unique(points, ["player_id", "season", "week"])
    unique(snaps, ["player_id", "season", "week"])
    unique(candidates, ["player_id", "season"])
    games = defaultdict(set)
    for row in schedule.filter(pl.col("game_type") == "REG").to_dicts():
        for team in (row["home_team"], row["away_team"]):
            games[int(row["season"]), team_name(team)].add(int(row["week"]))
    season_weeks = defaultdict(set)
    for (year, _), weeks in games.items():
        season_weeks[year].update(weeks)
    complete_snap_seasons = {
        year
        for year, weeks in season_weeks.items()
        if weeks and all((year, w) in snap_complete_weeks for w in weeks)
    }
    scoring: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    participation: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    for row in points.to_dicts():
        scoring[row["player_id"], int(row["season"])][int(row["week"])] = row
    for row in snaps.to_dicts():
        share = number(row.get("offense_pct"))
        if share is not None and not 0 <= share <= 1:
            raise ValueError("Offensive snap share must be a finite fraction")
        participation[row["player_id"], int(row["season"])][int(row["week"])] = row
    predefined = {(r["player_id"], int(r["season"])): r for r in candidates.to_dicts()}
    prior_counts: dict[tuple[str, int], int] = defaultdict(int)
    prior_source = prior_snaps if prior_snaps is not None else snaps
    unique(prior_source, ["player_id", "season", "week"])
    for row in prior_source.to_dicts():
        prior_counts[row["player_id"], int(row["season"])] += (row.get("offense_snaps") or 0) > 0
    results = []
    for player_id, season in sorted(set(scoring) | set(participation) | set(predefined)):
        scored = scoring[player_id, season]
        observed = participation[player_id, season]
        prior = predefined.get((player_id, season), {})
        for cutoff in sorted(set(cutoffs)):
            past = sorted(w for w in scored if w <= cutoff)
            snap_past = sorted(w for w in observed if w <= cutoff)
            if not past and not snap_past and not prior:
                continue
            latest = scored[past[-1]] if past else {}
            latest_snap = observed[snap_past[-1]] if snap_past else {}
            position = (
                prior.get("position") or latest.get("position") or latest_snap.get("position")
            )
            if position not in POSITIONS:
                continue
            latest_team_rows = sorted(
                [(w, scored[w]) for w in past] + [(w, observed[w]) for w in snap_past],
                key=lambda item: item[0],
                reverse=True,
            )
            team = team_name(
                next((r["team"] for _, r in latest_team_rows if r.get("team")), prior.get("team"))
            )
            known_games = games.get((season, team), set())
            if not known_games:
                continue
            past_games = sorted(w for w in known_games if w <= cutoff)
            # An observed appearance on another club remains in the denominator.
            past_games = sorted(
                set(past_games)
                | set(past)
                | {w for w in snap_past if (observed[w].get("offense_snaps") or 0) > 0}
            )
            recent = [w for w in past_games if w >= cutoff - 2]
            earlier = [w for w in past_games if cutoff - 4 <= w < cutoff - 2]
            future = [w for w in range(cutoff + 1, cutoff + 5) if w in season_weeks[season]]

            def stat(w, key, scored=scored):
                return number(scored.get(w, {}).get(key)) or 0.0

            def share(w, observed=observed, player_id=player_id, season=season):
                if w in observed:
                    return number(observed[w].get("offense_pct"))
                return (
                    0.0
                    if player_id in snap_identities and (season, w) in snap_complete_weeks
                    else None
                )

            def shares(weeks):
                return [v for w in weeks if (v := share(w)) is not None]

            def usage(w, position=position):
                fields = ("attempts", "carries") if position == "QB" else ("targets", "carries")
                return sum(stat(w, key) for key in fields)

            recent_shares, earlier_shares = shares(recent), shares(earlier)
            recent_points = [stat(w, "league_points") for w in recent]
            past_points = [stat(w, "league_points") for w in past_games]
            offense_weeks = [w for w in snap_past if (observed[w].get("offense_snaps") or 0) > 0]
            eligible = bool(offense_weeks) or any(usage(w) > 0 for w in past)
            fraction = (
                len(offense_weeks) / len(past_games)
                if past_games
                and (
                    player_id in snap_identities
                    and all((season, w) in snap_complete_weeks for w in past_games)
                )
                else None
            )
            recent_share = mean(recent_shares)
            recent_offense_fraction = (
                sum((observed.get(w, {}).get("offense_snaps") or 0) > 0 for w in recent)
                / len(recent)
                if recent
                and player_id in snap_identities
                and all((season, w) in snap_complete_weeks for w in recent)
                else None
            )
            snap_change = (
                recent_share - float(mean(earlier_shares))
                if (
                    len(recent_shares) >= 2
                    and len(earlier_shares) >= 2
                    and recent_share is not None
                )
                else None
            )
            usage_change = (
                float(mean([usage(w) for w in recent])) - float(mean([usage(w) for w in earlier]))
                if recent and earlier
                else None
            )
            recent_pg = mean(recent_points)
            horizon_games = len(set(future) & known_games)
            total_points = sum(max(v, 0) for v in recent_points)
            future_known = season in completed_seasons
            future_snaps_known = (
                future_known
                and player_id in snap_identities
                and all((season, w) in snap_complete_weeks for w in future)
            )
            future_active = [
                observed[w] for w in future if (observed.get(w, {}).get("offense_snaps") or 0) > 0
            ]
            future_active_shares = [
                v for r in future_active if (v := number(r.get("offense_pct"))) is not None
            ]
            prior_weeks = (
                prior_counts[player_id, season - 1]
                if (player_id in snap_identities and season - 1 in complete_snap_seasons)
                else None
            )
            results.append(
                {
                    "player_id": player_id,
                    "season": season,
                    "cutoff_week": cutoff,
                    "player_display_name": latest.get("player_display_name")
                    or latest_snap.get("player")
                    or prior.get("player_display_name")
                    or player_id,
                    "position": position,
                    "team": team,
                    "population": prior.get("population", "unknown"),
                    "eligible": eligible,
                    "past_team_games": len(past_games),
                    "recent_points_pg": recent_pg,
                    "recent_targets_pg": mean([stat(w, "targets") for w in recent]),
                    "recent_carries_pg": mean([stat(w, "carries") for w in recent]),
                    "recent_attempts_pg": mean([stat(w, "attempts") for w in recent]),
                    "recent_snap_share": recent_share,
                    "future_team_games": horizon_games,
                    "season_points_pg": mean(past_points),
                    "usage_change": usage_change,
                    "snap_change": snap_change,
                    "snap_std": float(np.std(shares(past_games)))
                    if len(shares(past_games)) >= 3
                    else None,
                    "points_std": float(np.std(recent_points)) if len(recent_points) >= 3 else None,
                    "spike_share": max(recent_points) / total_points
                    if len(recent_points) >= 3 and total_points > 0
                    else None,
                    "bonus_share": sum(stat(w, "bonus_pts") for w in recent) / total_points
                    if total_points > 0
                    else None,
                    "observed_offensive_weeks": len(offense_weeks),
                    "past_offensive_fraction": fraction,
                    "recent_offensive_fraction": recent_offense_fraction,
                    "prior_offensive_weeks": prior_weeks,
                    "snap_observations": len(
                        [w for w in snap_past if number(observed[w].get("offense_pct")) is not None]
                    ),
                    "snap_feed_complete": player_id in snap_identities
                    and all((season, w) in snap_complete_weeks for w in past_games),
                    "opportunity_trend": "insufficient history"
                    if usage_change is None
                    else "rising"
                    if usage_change > 1
                    else "falling"
                    if usage_change < -1
                    else "little change",
                    "role_evidence": "insufficient history"
                    if snap_change is None
                    else "expanding"
                    if snap_change > 0.1
                    else "contracting"
                    if snap_change < -0.1
                    else "little change",
                    "points_pace_next4": recent_pg * horizon_games
                    if recent_pg is not None and eligible
                    else None,
                    "participation_pace_next4": recent_offense_fraction * horizon_games
                    if recent_offense_fraction is not None and eligible
                    else None,
                    "next4_points": sum(stat(w, "league_points") for w in future)
                    if future_known
                    else None,
                    "next4_offensive_weeks": len(future_active) if future_snaps_known else None,
                    "next4_active_snap_share": mean(future_active_shares)
                    if future_snaps_known
                    else None,
                    "recent_active_snap_share": mean(
                        [
                            float(observed[w]["offense_pct"])
                            for w in snap_past
                            if w >= cutoff - 2
                            and (observed[w].get("offense_snaps") or 0) > 0
                            and observed[w].get("offense_pct") is not None
                        ]
                    ),
                }
            )
    return results


@dataclass
class Ridge:
    names: tuple[str, ...]
    fill: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, rows):
        x = np.array([[r.get(f) for f in self.names] for r in rows], dtype=float)
        missing = ~np.isfinite(x)
        x = np.where(missing, self.fill, x)
        x = np.column_stack((x, missing.astype(float)))
        return (x - self.center) / self.scale @ self.coef + self.intercept


def fit(rows, names, target):
    train = [r for r in rows if r.get(target) is not None and r["eligible"]]
    if len(train) < 30:
        return None
    x = np.array([[r.get(f) for f in names] for r in train], dtype=float)
    missing = ~np.isfinite(x)
    fill = np.array(
        [
            np.median(x[~missing[:, j], j]) if (~missing[:, j]).any() else 0
            for j in range(len(names))
        ]
    )
    x = np.column_stack((np.where(missing, fill, x), missing.astype(float)))
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    x = (x - center) / scale
    y = np.array([r[target] for r in train], dtype=float)
    coef = np.linalg.solve(x.T @ x + 10 * np.eye(x.shape[1]), x.T @ (y - y.mean()))
    return Ridge(tuple(names), fill, center, scale, coef, float(y.mean()))


def predict_fold(history, test):
    """Final two earlier seasons calibrate intervals, never fit the point/scale model."""
    if not test:
        return []
    season = test[0]["season"]
    position, cutoff = test[0]["position"], test[0]["cutoff_week"]
    if any(
        (r["season"], r["position"], r["cutoff_week"]) != (season, position, cutoff) for r in test
    ):
        raise ValueError("A forecast fold must have one season, position and cutoff")
    earlier = [
        r
        for r in history
        if r["season"] < season
        and r["position"] == position
        and r["cutoff_week"] == cutoff
        and r["eligible"]
        and r["next4_points"] is not None
    ]
    seasons = sorted({r["season"] for r in earlier})
    if len(seasons) < 6:
        return [
            {**r, "forecast_next4": None, "forecast_status": "Insufficient earlier-season history"}
            for r in test
        ]
    calibration_seasons = seasons[-2:]
    train = [r for r in earlier if r["season"] < calibration_seasons[0]]
    calibration = [r for r in earlier if r["season"] in calibration_seasons]
    models = {
        "outlook": fit(train, OUTLOOK, "next4_points"),
        "usage": fit(train, USAGE, "next4_points"),
    }
    if any(model is None for model in models.values()) or len(calibration) < 30:
        return [
            {**r, "forecast_next4": None, "forecast_status": "Insufficient calibration sample"}
            for r in test
        ]
    # Learn heterogeneity from errors of earlier-season forecasts, not in-sample errors.
    errors = []
    train_years = sorted({r["season"] for r in train})
    for year in train_years[3:]:
        held = [r for r in train if r["season"] == year]
        model = fit([r for r in train if r["season"] < year], OUTLOOK, "next4_points")
        if model is not None:
            errors += [
                {**r, "log_abs_error": math.log1p(abs(r["next4_points"] - max(0, float(p))))}
                for r, p in zip(held, model.predict(held), strict=True)
            ]
    scale_model = fit(errors, SCALE, "log_abs_error")

    def scales(rows):
        return (
            np.maximum(5, np.expm1(np.clip(scale_model.predict(rows), 0, 6)))
            if scale_model
            else np.ones(len(rows))
        )

    values, bands = {}, {}
    for name, model in models.items():
        cp = np.maximum(0, model.predict(calibration))
        values[name] = np.maximum(0, model.predict(test))
        cs = scales(calibration) if name == "outlook" else np.ones(len(calibration))
        ts = scales(test) if name == "outlook" else np.ones(len(test))
        residuals = (np.array([r["next4_points"] for r in calibration]) - cp) / cs
        low, high = np.quantile(residuals, [0.1, 0.9])
        bands[name] = (values[name] + low * ts, values[name] + high * ts)
    availability = fit(train, OUTLOOK, "next4_offensive_weeks")
    role = fit(train, OUTLOOK, "next4_active_snap_share")
    av = np.clip(availability.predict(test), 0, 4) if availability else [None] * len(test)
    roles = np.clip(role.predict(test), 0, 1) if role else [None] * len(test)
    results = []
    for i, row in enumerate(test):
        usable = row["eligible"]
        participation_usable = usable and row.get("snap_feed_complete") is True
        results.append(
            {
                **row,
                "forecast_next4": float(values["outlook"][i]) if usable else None,
                "usage_baseline_next4": float(values["usage"][i]) if usable else None,
                "expected_offensive_weeks": min(float(av[i]), row["future_team_games"])
                if participation_usable and av[i] is not None
                else None,
                "expected_active_snap_share": number(roles[i]) if participation_usable else None,
                "research_range_low": float(bands["outlook"][0][i]) if usable else None,
                "research_range_high": float(bands["outlook"][1][i]) if usable else None,
                "usage_range_low": float(bands["usage"][0][i]) if usable else None,
                "usage_range_high": float(bands["usage"][1][i]) if usable else None,
                "train_seasons": train_years,
                "calibration_seasons": calibration_seasons,
                "scale_training_rows": len(errors),
                "confidence_score": None,
                "forecast_status": "Exploratory four-week outlook"
                if usable
                else "No observed offensive role; unscored",
            }
        )
    return results


def interval_score(actual, low, high):
    return high - low + 10 * max(low - actual, 0) + 10 * max(actual - high, 0)


def evaluate(predictions):
    """Common-row baselines; year-block uncertainty, never iid player-week inference."""

    def in_cohort(row, cohort):
        if cohort == "all":
            return True
        if cohort == "rookie":
            return row.get("population") == "rookie"
        prior = row.get("prior_offensive_weeks")
        return prior is not None and (prior < 10 if cohort == "small_prior_sample" else prior >= 10)

    summaries = []
    for position in POSITIONS:
        for cutoff in sorted({r["cutoff_week"] for r in predictions}):
            for cohort in ("all", "rookie", "small_prior_sample", "established_prior_sample"):
                rows = [
                    r
                    for r in predictions
                    if r["position"] == position
                    and r["cutoff_week"] == cutoff
                    and r.get("forecast_next4") is not None
                    and r.get("next4_points") is not None
                    and in_cohort(r, cohort)
                ]
                if not rows:
                    continue
                mae = {
                    name: mean([abs(r[key] - r["next4_points"]) for r in rows])
                    for name, key in (
                        ("outlook", "forecast_next4"),
                        ("recent_usage", "usage_baseline_next4"),
                        ("recent_points", "points_pace_next4"),
                    )
                }
                folds = []
                for season in sorted({r["season"] for r in rows}):
                    fold = [r for r in rows if r["season"] == season]
                    folds.append(
                        {
                            "season": season,
                            "n": len(fold),
                            "mae_improvement_vs_usage": mean(
                                [
                                    abs(r["usage_baseline_next4"] - r["next4_points"])
                                    - abs(r["forecast_next4"] - r["next4_points"])
                                    for r in fold
                                ]
                            ),
                        }
                    )
                diffs = np.array([r["mae_improvement_vs_usage"] for r in folds])
                rng = np.random.default_rng(20260922)
                ci = np.quantile(
                    rng.choice(diffs, (5000, len(diffs)), replace=True).mean(axis=1), [0.025, 0.975]
                ).tolist()
                coverage = mean(
                    [
                        r["research_range_low"] <= r["next4_points"] <= r["research_range_high"]
                        for r in rows
                    ]
                )
                scores = {
                    name: mean([interval_score(r["next4_points"], r[low], r[high]) for r in rows])
                    for name, low, high in (
                        ("outlook", "research_range_low", "research_range_high"),
                        ("recent_usage", "usage_range_low", "usage_range_high"),
                    )
                }
                av_rows = [
                    r
                    for r in rows
                    if r.get("next4_offensive_weeks") is not None
                    and r.get("expected_offensive_weeks") is not None
                    and r.get("participation_pace_next4") is not None
                ]
                role_rows = [
                    r
                    for r in rows
                    if r.get("next4_active_snap_share") is not None
                    and r.get("expected_active_snap_share") is not None
                    and r.get("recent_active_snap_share") is not None
                ]
                checks = {
                    "enough_rows": len(rows) >= 200,
                    "enough_seasons": len(folds) >= 5,
                    "interval_coverage": 0.75 <= coverage <= 0.85,
                    "interval_score_vs_usage": scores["outlook"] <= scores["recent_usage"],
                    "point_mae_vs_points": mae["outlook"] <= mae["recent_points"],
                    "point_mae_vs_usage": ci[0] > 0,
                }
                summaries.append(
                    {
                        "position": position,
                        "cutoff_week": cutoff,
                        "cohort": cohort,
                        "n": len(rows),
                        "mae": mae,
                        "season_bootstrap_95_improvement_vs_usage": ci,
                        "folds": folds,
                        "nominal_coverage": 0.8,
                        "observed_coverage": coverage,
                        "interval_score": scores,
                        "availability": {
                            "n": len(av_rows),
                            "outlook_mae": mean(
                                [
                                    abs(r["expected_offensive_weeks"] - r["next4_offensive_weeks"])
                                    for r in av_rows
                                ]
                            ),
                            "recent_participation_mae": mean(
                                [
                                    abs(r["participation_pace_next4"] - r["next4_offensive_weeks"])
                                    for r in av_rows
                                ]
                            ),
                        },
                        "role": {
                            "n": len(role_rows),
                            "outlook_mae": mean(
                                [
                                    abs(
                                        r["expected_active_snap_share"]
                                        - r["next4_active_snap_share"]
                                    )
                                    for r in role_rows
                                ]
                            ),
                            "hold_recent_share_mae": mean(
                                [
                                    abs(
                                        r["recent_active_snap_share"] - r["next4_active_snap_share"]
                                    )
                                    for r in role_rows
                                ]
                            ),
                        },
                        "publication_checks": checks,
                        "range_gate_passed": all(checks.values()),
                    }
                )
    return summaries


def publication_rows(predictions, summaries):
    """Unpassed ranges stay in diagnostics, never the public player payload."""
    lookup = {(r["position"], r["cutoff_week"], r["cohort"]): r for r in summaries}
    result = []
    for row in predictions:
        cohorts = ["all"]
        if row.get("population") == "rookie":
            cohorts.append("rookie")
        if row.get("prior_offensive_weeks") is None or row["prior_offensive_weeks"] < 10:
            cohorts.append("small_prior_sample")
        else:
            cohorts.append("established_prior_sample")
        gates = [lookup.get((row["position"], row["cutoff_week"], c)) for c in cohorts]
        passed = (
            row.get("snap_feed_complete") is True
            and row.get("prior_offensive_weeks") is not None
            and all(g and g["range_gate_passed"] for g in gates)
        )
        public = {
            k: v
            for k, v in row.items()
            if not k.startswith(("research_range_", "usage_range_")) and k not in TARGETS
        }
        public.update(
            {
                "outcome_range": {
                    "low": row.get("research_range_low"),
                    "high": row.get("research_range_high"),
                    "nominal_coverage": 0.8,
                }
                if passed and row.get("forecast_next4") is not None
                else None,
                "range_status": "Historical gates passed; prospective validation pending"
                if passed
                else "Withheld: data coverage or calibration/incremental-value gates not passed",
                "confidence_score": None,
            }
        )
        result.append(public)
    return result
