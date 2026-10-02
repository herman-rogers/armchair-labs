"""Cutoff-safe current-season forecasts and nested chronological model selection."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from itertools import product

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from patron.metrics.nextgen import COUNTERS, POSITIONS, season_length

HORIZONS = ("rest_of_season", "next4")
RAW_BASELINES = ("prior", "current", "blend")
BASELINES = RAW_BASELINES + ("calibrated_current", "calibrated_blend")
CHALLENGERS = ("profile_ridge", "profile_boost", "enriched_boost")
TOP_K = {"QB": 10, "RB": 20, "WR": 30, "TE": 10}
TRACKING = ("ngs_cpoe", "ngs_separation", "ngs_yac_oe", "ngs_ryoe_per_att")


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def total(rows, column):
    values = [finite(r.get(column)) for r in rows]
    return sum(values) if values and all(v is not None for v in values) else None


def team_code(value):
    return {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV"}.get(value, value)


def make_panel(features, weeks, schedule, identities, college, links, *, season, through_week):
    """Outcome-only suffixes cannot change candidates, teams, or predictor values."""
    by_player = defaultdict(list)
    by_season = defaultdict(list)
    for row in weeks.sort("season", "week").to_dicts():
        if row["position"] in POSITIONS:
            by_player[row["player_id"]].append(row)
            by_season[row["season"]].append(row)
    games = defaultdict(set)
    for row in schedule.filter(pl.col("game_type") == "REG").to_dicts():
        for team in (row["home_team"], row["away_team"]):
            games[row["season"], team_code(team)].add(row["week"])
    people = {r["gsis_id"]: r for r in identities.to_dicts()}
    link = {r["college_id"]: r["player_id"] for r in links.to_dicts() if r["status"] == "linked"}
    college_history = defaultdict(list)
    for row in college.filter(pl.col("college_id").is_in(list(link))).to_dicts():
        pid = link.get(row["college_id"])
        if (
            pid
            and row["college_position"] in POSITIONS
            and row["observed_stat_games"] > 0
            and row["complete_team_season"]
            and row["team_count"] == 1
        ):
            college_history[pid].append(row)
    candidates = defaultdict(dict)
    feature_columns = [
        "player_id",
        "player_display_name",
        "position",
        "player_population",
        "forecast_season",
        "forecast_cutoff_date",
        "cutoff_preseason_team",
        *TRACKING,
    ]
    for r in features.select([c for c in feature_columns if c in features.columns]).to_dicts():
        year = r["forecast_season"]
        if 2004 <= year <= season and r["position"] in POSITIONS:
            candidates[year][r["player_id"]] = r
    for year in range(2004, season + 1):
        for r in by_season[year]:
            if r["week"] <= through_week and r["player_id"] not in candidates[year]:
                candidates[year][r["player_id"]] = dict(
                    player_id=r["player_id"],
                    player_display_name=r["player_display_name"],
                    position=r["position"],
                    player_population="observed_by_cutoff",
                    forecast_cutoff_date=f"{year}-09-01",
                )
    panel = []
    for year, players in sorted(candidates.items()):
        for pid, candidate in players.items():
            history = by_player[pid]
            prior = [r for r in history if r["season"] == year - 1]
            recent = [r for r in history if year - 3 <= r["season"] < year]
            career = [r for r in history if r["season"] < year]
            prefix = [r for r in history if r["season"] == year and r["week"] <= through_week]
            person = people.get(pid, {})
            draft_year = person.get("draft_year")
            draft_known = draft_year is not None and draft_year <= year
            population = (
                "rookie" if draft_year == year else candidate.get("player_population", "returner")
            )
            team = team_code(
                prefix[-1]["team"] if prefix else candidate.get("cutoff_preseason_team")
            )
            schedule_weeks = games.get((year, team), set())
            elapsed = (
                sum(w <= through_week for w in schedule_weeks) if schedule_weeks else through_week
            )
            # Expansion/early byes cannot turn an unknown exposure into an invented appearance.
            current_rate = (total(prefix, "league_points") or 0) / max(elapsed, 1)
            birth = person.get("birth_date")
            age = (
                (date(year, 9, 1) - date.fromisoformat(str(birth)[:10])).days / 365.25
                if birth
                else None
            )
            row = dict(
                player_id=pid,
                player_display_name=candidate["player_display_name"],
                position=candidate["position"],
                population=population,
                season=year,
                through_week=through_week,
                team=team,
                prior_weeks=len(prior),
                current_weeks=len(prefix),
                elapsed_games=elapsed,
                schedule_known=bool(schedule_weeks),
                prior_rate=(total(prior, "league_points") / season_length(year - 1))
                if prior
                else None,
                current_rate=current_rate,
                current_points=total(prefix, "league_points") or 0,
                x_age=age,
                x_draft=person.get("draft_pick") if draft_known else None,
                x_experience=year - person["rookie_season"]
                if person.get("rookie_season") and person["rookie_season"] <= year
                else None,
                x_rookie=float(population == "rookie"),
                x_prior_weeks=len(prior),
                x_career_weeks=len(career),
                x_current_weeks=len(prefix),
                x_elapsed=elapsed,
            )
            for name, subset in (("prior", prior), ("recent3", recent), ("career", career)):
                for c in COUNTERS:
                    value = total(subset, c)
                    row[f"x_{name}_{c}_rate"] = value / len(subset) if value is not None else None
            for c in COUNTERS:
                value = total(prefix, c)
                row[f"x_current_{c}_rate"] = value / max(elapsed, 1) if value is not None else None
                row[f"x_last_{c}"] = prefix[-1].get(c) if prefix else None
            if row["position"] == "QB":
                workload = row["x_current_attempts_rate"]
                exposed = workload is not None and workload >= 10
            elif row["position"] == "RB":
                workload = (row["x_current_carries_rate"] or 0) + (
                    row["x_current_targets_rate"] or 0
                )
                exposed = workload >= 5
            else:
                workload = row["x_current_targets_rate"]
                exposed = workload is not None and workload >= 2
            row["role_group"] = "observed_workload" if exposed else "limited_workload"
            for name, subset in (("prior", prior), ("current", prefix)):
                shares = [r["snap_share"] for r in subset if r.get("snap_share") is not None]
                row[f"x_{name}_snap"] = float(np.mean(shares)) if shares else None
            row["x_last_snap"] = prefix[-1].get("snap_share") if prefix else None
            for c in TRACKING:
                row["e_" + c] = finite(candidate.get(c))
            eligible_college = [
                r
                for r in college_history[pid]
                if r["season"] < year
                and str(r["season_end_date"])[:10] < str(candidate["forecast_cutoff_date"])[:10]
            ]
            latest = max(eligible_college, key=lambda r: r["season"], default={})
            for c in (
                "passing_yards",
                "passing_tds",
                "passing_ints",
                "rushing_yards",
                "rushing_tds",
                "receiving_yards",
                "receptions",
            ):
                row["e_college_" + c] = latest.get(c)
            row["e_college_games"] = latest.get("observed_stat_games")
            for horizon in HORIZONS:
                last_week = 18 if year >= 2021 else 17
                end = min(through_week + 4, last_week) if horizon == "next4" else last_week
                future = [
                    r for r in history if r["season"] == year and through_week < r["week"] <= end
                ]
                count = (
                    sum(through_week < w <= end for w in schedule_weeks)
                    if schedule_weeks
                    else (4 if horizon == "next4" else max(season_length(year) - elapsed, 0))
                )
                panel.append(
                    dict(
                        row,
                        horizon=horizon,
                        end_week=end,
                        scheduled_games=count,
                        actual=(total(future, "league_points") or 0) if year < season else None,
                    )
                )
    return panel


def fit_candidates(train, test):
    if not train or not test:
        raise ValueError("A ranking fold needs earlier training rows and cutoff candidates")
    group = defaultdict(list)
    for r in train:
        if r["prior_rate"] is not None:
            group[r["population"]].append(r["prior_rate"])
    fallback = float(np.mean([r["prior_rate"] for r in train if r["prior_rate"] is not None]))
    results = {m: [] for m in RAW_BASELINES}
    training_references = {m: [] for m in RAW_BASELINES}
    for index, r in enumerate(train + test):
        prior = r["prior_rate"]
        if prior is None:
            prior = float(np.mean(group[r["population"]])) if group[r["population"]] else fallback
        current = r["current_rate"] if r["elapsed_games"] else prior
        rates = (
            prior,
            current,
            (4 * prior + r["elapsed_games"] * current) / (4 + r["elapsed_games"]),
        )
        target = training_references if index < len(train) else results
        for name, rate in zip(RAW_BASELINES, rates, strict=True):
            target[name].append(max(0, rate * r["scheduled_games"]))
    y = np.array([r["actual"] for r in train])
    # An affine rate calibration learns regression toward typical future output
    # from earlier seasons only; it does not select a cap from current player names.
    for base in ("current", "blend"):
        with threadpool_limits(limits=1):
            predictions = np.zeros(len(test))
            for role in {r.get("role_group", "limited_workload") for r in test}:
                ti = [
                    i
                    for i, r in enumerate(train)
                    if r.get("role_group", "limited_workload") == role
                ]
                if len(ti) < 50:
                    ti = list(range(len(train)))
                zi = [
                    i for i, r in enumerate(test) if r.get("role_group", "limited_workload") == role
                ]
                calibration = Ridge(alpha=100, positive=True)
                calibration.fit(np.array(training_references[base])[ti].reshape(-1, 1), y[ti])
                predictions[zi] = np.maximum(
                    calibration.predict(np.array(results[base])[zi].reshape(-1, 1)), 0
                )
            results["calibrated_" + base] = predictions.tolist()
    core = sorted(c for c in train[0] if c.startswith("x_"))
    enriched = core + sorted(c for c in train[0] if c.startswith("e_"))
    for name in CHALLENGERS:
        columns = enriched if name == "enriched_boost" else core
        # Schedule is known at the forecast cutoff and belongs to every recipe.
        columns = columns + ["scheduled_games"]
        x = np.array(
            [[r.get(c) if r.get(c) is not None else np.nan for c in columns] for r in train],
            dtype=float,
        )
        z = np.array(
            [[r.get(c) if r.get(c) is not None else np.nan for c in columns] for r in test],
            dtype=float,
        )
        varying = [i for i in range(x.shape[1]) if len(np.unique(x[np.isfinite(x[:, i]), i])) > 1]
        if not varying:
            raise ValueError("Ranking training set has no varying predictors")
        with threadpool_limits(limits=1):
            if name == "profile_ridge":
                model = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                    Ridge(alpha=100),
                )
            else:
                model = HistGradientBoostingRegressor(
                    max_iter=120,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=10,
                    max_bins=63,
                    early_stopping=False,
                    random_state=20260923,
                )
            model.fit(x[:, varying], y)
            results[name] = np.maximum(model.predict(z[:, varying]), 0).tolist()
    for values in results.values():
        for i, r in enumerate(test):
            if r["scheduled_games"] == 0:
                values[i] = 0.0
    return results


def metrics(rows, field, k):
    actual = np.array([r["actual"] for r in rows])
    predicted = np.array([r[field] for r in rows])
    selected = sorted(rows, key=lambda r: (-r[field], r["player_id"]))[:k]
    oracle = sum(sorted(actual, reverse=True)[:k])
    return dict(
        mae=float(np.mean(np.abs(predicted - actual))),
        mse=float(np.mean((predicted - actual) ** 2)),
        capture=sum(r["actual"] for r in selected) / oracle if oracle > 0 else 0.0,
    )


def choose(earlier, position):
    """Select only from completed out-of-year predictions before the test season."""
    if not earlier:
        return "blend", "blend"
    years = sorted({r["season"] for r in earlier})
    scores = {}
    for recipe in BASELINES + CHALLENGERS:
        annual = [
            metrics([r for r in earlier if r["season"] == year], recipe, TOP_K[position])
            for year in years
        ]
        scores[recipe] = {
            m: float(np.mean([r[m] for r in annual])) for m in ("mae", "mse", "capture")
        }
    baseline = min(BASELINES, key=lambda m: (scores[m]["mae"], m))
    if len(years) < 4:
        return baseline, baseline
    b = scores[baseline]
    qualified = [
        m
        for m in CHALLENGERS
        if b["mae"] - scores[m]["mae"] >= max(0.5, 0.01 * b["mae"])
        and scores[m]["mse"] <= b["mse"]
        and scores[m]["capture"] >= b["capture"]
    ]
    return baseline, min(qualified, key=lambda m: (scores[m]["mae"], m)) if qualified else baseline


def choose_by_role(earlier, position):
    """Observed opportunity changes the recipe without excluding any candidates."""
    return {
        group: choose([r for r in earlier if r["role_group"] == group], position)
        for group in ("observed_workload", "limited_workload")
    }


def compare(rows, position):
    annual = []
    for year in sorted({r["season"] for r in rows}):
        subset = [r for r in rows if r["season"] == year]
        p, b = (metrics(subset, f, TOP_K[position]) for f in ("policy", "reference"))
        annual.append(
            dict(
                season=year,
                n=len(subset),
                error=p["mae"],
                baseline_error=b["mae"],
                gain=b["mae"] - p["mae"],
                mse_gain=b["mse"] - p["mse"],
                capture=p["capture"],
                baseline_capture=b["capture"],
                capture_gain=p["capture"] - b["capture"],
            )
        )
    if not annual:
        return dict(years=0, n=0, annual=[])
    gains = np.array([r["gain"] for r in annual])
    boot = np.random.default_rng(20260923).choice(gains, (10000, len(gains))).mean(axis=1)
    # Exact sign flips for the seven primary modern seasons; older window is a guardrail.
    p_value = None
    if len(gains) <= 10:
        flips = np.array(list(product((-1, 1), repeat=len(gains))))
        p_value = float(np.mean(np.abs((flips * gains).mean(axis=1)) >= abs(gains.mean()) - 1e-12))
    return dict(
        years=len(annual),
        n=len(rows),
        annual=annual,
        error=float(np.mean([r["error"] for r in annual])),
        baseline_error=float(np.mean([r["baseline_error"] for r in annual])),
        improvement=float(gains.mean()),
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
        mse_improvement=float(np.mean([r["mse_gain"] for r in annual])),
        capture_improvement=float(np.mean([r["capture_gain"] for r in annual])),
        p_value=p_value,
    )


def evaluate_scopes(predictions):
    scopes = []
    for position, horizon in product(POSITIONS, HORIZONS):
        rows = [
            r
            for r in predictions
            if r["position"] == position
            and r["horizon"] == horizon
            and r["actual"] is not None
            and r["season"] >= 2011
        ]
        modern = [r for r in rows if r["season"] >= 2019]
        cohorts = {}
        for name, subset in (
            ("rookies", [r for r in modern if r["population"] == "rookie"]),
            ("small_prior", [r for r in modern if r["prior_weeks"] < 10]),
        ):
            cohorts[name] = compare(subset, position)
        scopes.append(
            dict(
                position=position,
                horizon=horizon,
                modern=compare(modern, position),
                all_history=compare(rows, position),
                cohorts=cohorts,
            )
        )
    ordered = sorted(scopes, key=lambda r: r["modern"].get("p_value") or 1.0)
    running = 1.0
    for index in range(len(ordered) - 1, -1, -1):
        p = ordered[index]["modern"].get("p_value") or 1.0
        running = min(running, p * len(ordered) / (index + 1))
        ordered[index]["q_value"] = running
    for scope in scopes:
        m, a = scope["modern"], scope["all_history"]
        reasons = []
        if m["years"] < 7 or a["years"] < 8:
            reasons.append("Insufficient evaluation years")
        elif (
            m["improvement"] < max(0.5, 0.01 * m["baseline_error"])
            or m["ci_low"] <= 0
            or scope["q_value"] > 0.05
        ):
            reasons.append(
                "Modern improvement did not clear the declared uncertainty and multiplicity gates"
            )
        for label, view in (("Modern", m), ("Full-history", a)):
            if view["years"] and (
                view["mse_improvement"] < 0
                or view["capture_improvement"] < 0
                or view["improvement"] < 0
            ):
                reasons.append(label + " error or ranking guardrail failed")
        for name, view in scope["cohorts"].items():
            if (
                view["n"] >= 100
                and view["years"] >= 5
                and view["improvement"] < -max(1, 0.05 * view["baseline_error"])
            ):
                reasons.append(name + " error guardrail failed")
        scope["approved_challenger_policy"] = not reasons
        scope["reason"] = (
            "; ".join(reasons)
            if reasons
            else "Nested chronological forecast and ranking checks passed"
        )
    return scopes


def apply_constraints(rows, news, as_of):
    """Current dated reports constrain delivery; they never change historical fits."""
    result = []
    for original in rows:
        row = dict(
            original,
            unconstrained_prediction=original["prediction"],
            rank_eligible=True,
            constraint=None,
            constraint_source=None,
            constraint_known_on=None,
        )
        relevant = [
            n
            for n in news
            if n.get("player_id") == row["player_id"]
            and n.get("season") == row["season"]
            and n.get("known_on", "9999") <= as_of
            and n.get("season_ending_reported")
            and n.get("affected_period") == f"remaining_{row['season']}_regular_season"
            and n.get("earliest_affected_week", 999) <= row["through_week"] + 1
        ]
        if relevant:
            evidence = max(relevant, key=lambda n: n["known_on"])
            row.update(
                prediction=0.0,
                rank_eligible=False,
                constraint=evidence["summary"],
                constraint_source=evidence["source_url"],
                constraint_known_on=evidence["known_on"],
            )
        result.append(row)
    return result


def rank_frame(frame):
    eligible = frame.filter(pl.col("rank_eligible")).with_columns(
        pl.col("prediction")
        .rank(method="min", descending=True)
        .over("horizon")
        .alias("overall_rank"),
        pl.col("prediction")
        .rank(method="min", descending=True)
        .over("horizon", "position")
        .alias("position_rank"),
    )
    ranks = eligible.select("player_id", "horizon", "overall_rank", "position_rank")
    return frame.drop("overall_rank", "position_rank", strict=False).join(
        ranks, on=["player_id", "horizon"], how="left", validate="1:1"
    )
