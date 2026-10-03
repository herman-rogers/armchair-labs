"""Dedicated next-calendar-week learner and chronological evaluation utilities."""

from collections import defaultdict
from itertools import product

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from patron.metrics.current_rankings import TOP_K, team_code

REFERENCES = ("prior", "current", "blend", "last3")
CHALLENGERS = ("weekly_ridge", "weekly_boost")
SEED = 20261002


def add_context(rows, weeks, schedule):
    """Enrich prefix features; target-week production is read only as the outcome."""
    history = defaultdict(list)
    totals = defaultdict(float)
    opponent = {}
    for game in schedule:
        if game["game_type"] == "REG":
            home, away = team_code(game["home_team"]), team_code(game["away_team"])
            opponent[game["season"], game["week"], home] = (away, 1)
            opponent[game["season"], game["week"], away] = (home, 0)
    actual = {}
    for r in weeks:
        history[r["player_id"], r["season"]].append(r)
        actual[r["player_id"], r["season"], r["week"]] = r["league_points"]
        op = opponent.get((r["season"], r["week"], team_code(r["team"])))
        if op and r["league_points"] is not None:
            totals[r["season"], r["week"], op[0], r["position"]] += r["league_points"]
    allowed = defaultdict(list)
    for (year, week, team, position), value in totals.items():
        allowed[year, team, position].append((week, value))
    for values in history.values():
        values.sort(key=lambda r: r["week"])
    for row in rows:
        year, cutoff, team = row["season"], row["through_week"], team_code(row["team"])
        prefix = [r for r in history[row["player_id"], year] if r["week"] <= cutoff]
        recent = prefix[-3:]
        row["x_last3_points"] = (
            np.mean([r["league_points"] or 0 for r in recent]) if recent else None
        )
        row["x_last3_targets"] = np.mean([r["targets"] or 0 for r in recent]) if recent else None
        row["x_last3_carries"] = np.mean([r["carries"] or 0 for r in recent]) if recent else None
        op = opponent.get((year, cutoff + 1, team))
        row["x_home"] = op[1] if op else None
        # Current-prefix opponent production only; preseason uses the prior season.
        op_rows = (
            [
                value
                for w, value in allowed[(year if cutoff else year - 1), op[0], row["position"]]
                if not cutoff or w <= cutoff
            ]
            if op
            else []
        )
        row["x_opponent_points_allowed"] = float(np.mean(op_rows)) if op_rows else None
        row["target_week"] = cutoff + 1
        row["observed_actual"] = actual.get((row["player_id"], year, cutoff + 1), 0.0)
    return rows


def fit_fold(train, test):
    """No test outcomes are used for fitting, imputation, or positional priors."""
    priors = [r["prior_rate"] for r in train if r["prior_rate"] is not None]
    fallback = float(np.mean(priors)) if priors else 0.0
    result = {k: [] for k in REFERENCES}
    for r in test:
        prior = r["prior_rate"] if r["prior_rate"] is not None else fallback
        current = r["current_rate"] if r["elapsed_games"] else prior
        values = [
            prior,
            current,
            (4 * prior + r["elapsed_games"] * current) / (4 + r["elapsed_games"]),
            r["x_last3_points"] if r["x_last3_points"] is not None else prior,
        ]
        for name, value in zip(REFERENCES, values, strict=True):
            result[name].append(max(0.0, value) if r["scheduled_games"] else 0.0)
    columns = sorted(k for k in train[0] if k.startswith("x_"))
    x = np.array([[r.get(k) if r.get(k) is not None else np.nan for k in columns] for r in train])
    z = np.array([[r.get(k) if r.get(k) is not None else np.nan for k in columns] for r in test])
    varying = np.array([len(np.unique(v[np.isfinite(v)])) > 1 for v in x.T])
    y = np.array([r["actual"] for r in train])
    with threadpool_limits(limits=2):
        ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100))
        boost = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=10,
            early_stopping=False,
            random_state=SEED,
        )
        for name, model in zip(CHALLENGERS, (ridge, boost), strict=True):
            model.fit(x[:, varying], y)
            values = np.maximum(model.predict(z[:, varying]), 0)
            result[name] = [
                float(v) if r["scheduled_games"] else 0.0 for v, r in zip(values, test, strict=True)
            ]
    return result


def choose(earlier):
    years = sorted({r["season"] for r in earlier})
    if not years:
        return "blend", "blend"
    scores = {
        name: np.mean(
            [
                np.mean([abs(r[name] - r["actual"]) for r in earlier if r["season"] == year])
                for year in years
            ]
        )
        for name in REFERENCES + CHALLENGERS
    }
    reference = min(REFERENCES, key=lambda k: (scores[k], k))
    challenger = min(CHALLENGERS, key=lambda k: (scores[k], k))
    return reference, challenger if len(years) >= 3 else reference


def metrics(rows, field, position):
    if not rows:
        return dict(n=0, mae=None, mse=None, bias=None, capture=None)
    errors = np.array([r[field] - r["actual"] for r in rows])
    groups = defaultdict(list)
    for r in rows:
        groups[r["season"], r["target_week"]].append(r)
    captures = []
    for group in groups.values():
        oracle = sum(sorted((r["actual"] for r in group), reverse=True)[: TOP_K[position]])
        selected = sorted(group, key=lambda r: (-r[field], r["player_id"]))[: TOP_K[position]]
        if oracle > 0:
            captures.append(sum(r["actual"] for r in selected) / oracle)
    return dict(
        n=len(rows),
        mae=float(np.abs(errors).mean()),
        mse=float((errors**2).mean()),
        bias=float(errors.mean()),
        capture=float(np.mean(captures)) if captures else 0.0,
    )


def evaluate(rows, position):
    modern = [r for r in rows if 2019 <= r["season"] <= 2025]
    annual = []
    for year in sorted({r["season"] for r in modern}):
        subset = [r for r in modern if r["season"] == year]
        annual.append(
            dict(
                season=year,
                model=metrics(subset, "policy", position),
                reference=metrics(subset, "reference", position),
            )
        )
    gains = np.array([r["reference"]["mae"] - r["model"]["mae"] for r in annual])
    boot = np.random.default_rng(SEED).choice(gains, (10000, len(gains))).mean(axis=1)
    flips = np.array(list(product((-1, 1), repeat=len(gains))))
    p = float(np.mean((flips * gains).mean(axis=1) >= gains.mean() - 1e-12))
    cohorts = {
        name: dict(
            model=metrics(subset, "policy", position),
            reference=metrics(subset, "reference", position),
        )
        for name, subset in [
            ("observed_workload", [r for r in modern if r["role_group"] == "observed_workload"]),
            ("rookies", [r for r in modern if r["population"] == "rookie"]),
        ]
    }
    full = [r for r in rows if r["season"] >= 2011]
    return dict(
        position=position,
        modern=dict(
            model=metrics(modern, "policy", position),
            reference=metrics(modern, "reference", position),
        ),
        annual=annual,
        mean_season_gain=float(gains.mean()),
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
        p_value=p,
        cohorts=cohorts,
        full_history=dict(
            model=metrics(full, "policy", position), reference=metrics(full, "reference", position)
        ),
    )


def promotion(reports):
    running = 0.0
    for index, report in enumerate(sorted(reports, key=lambda r: r["p_value"])):
        running = max(running, min(1.0, (len(reports) - index) * report["p_value"]))
        report["adjusted_p"] = running
    for report in reports:
        m, b = report["modern"]["model"], report["modern"]["reference"]
        full = report["full_history"]
        checks = [
            len(report["annual"]) == 7,
            report["mean_season_gain"] >= max(0.10, 0.02 * b["mae"]),
            report["ci_low"] > 0,
            report["adjusted_p"] <= 0.05,
            m["mse"] <= b["mse"],
            m["capture"] >= b["capture"],
            full["model"]["mae"] <= full["reference"]["mae"],
        ]
        checks.extend(
            c["model"]["n"] < 100
            or c["model"]["mae"] - c["reference"]["mae"] <= max(0.25, 0.05 * c["reference"]["mae"])
            for c in report["cohorts"].values()
        )
        report["approved"] = all(checks)
        report["reason"] = (
            "Chronological weekly policy passed all declared gates."
            if all(checks)
            else "Weekly reference retained: "
            "at least one declared accuracy/uncertainty/cohort gate failed."
        )
    return reports
