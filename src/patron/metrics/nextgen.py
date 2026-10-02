"""Outcome-specific NextGen baselines and chronological research comparisons.

Observations, predictions and decisions have separate contracts. No model selection
uses held-out outcomes; the fixed reference baseline remains the serving default.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

POSITIONS = ("QB", "RB", "WR", "TE")
COUNTERS = (
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "attempts",
    "carries",
    "targets",
    "receptions",
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "league_points",
)
TARGETS = {
    "season_points": ("Season fantasy points", "points", POSITIONS, "league_points", None),
    "passing_yards": ("Passing yards", "yards", ("QB",), "passing_yards", None),
    "rushing_yards": ("Rushing yards", "yards", ("QB", "RB", "WR"), "rushing_yards", None),
    "receiving_yards": ("Receiving yards", "yards", ("RB", "WR", "TE"), "receiving_yards", None),
    "attempts": ("Pass attempts", "attempts", ("QB",), "attempts", None),
    "carries": ("Carries", "carries", ("QB", "RB", "WR"), "carries", None),
    "targets": ("Receiving targets", "targets", ("RB", "WR", "TE"), "targets", None),
    "receptions": ("Receptions", "receptions", ("RB", "WR", "TE"), "receptions", None),
    "scoring_appearances": (
        "Scoring appearances",
        "games",
        POSITIONS,
        "actual_games",
        None,
    ),
    "season_appearance": (
        "At least one scoring appearance",
        "probability",
        POSITIONS,
        "actual_games",
        "binary",
    ),
    "passing_efficiency": (
        "Passing yards / attempt",
        "yards/attempt",
        ("QB",),
        "passing_yards",
        "attempts",
    ),
    "rushing_efficiency": (
        "Rushing yards / carry",
        "yards/carry",
        ("QB", "RB"),
        "rushing_yards",
        "carries",
    ),
    "receiving_efficiency": (
        "Receiving yards / target",
        "yards/target",
        ("RB", "WR", "TE"),
        "receiving_yards",
        "targets",
    ),
}


def season_length(year: int) -> int:
    return 17 if year >= 2021 else 16


def distribution(values) -> dict:
    x = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)])
    result = dict.fromkeys(
        ("mean", "median", "std", "variance", "cv", "top2_positive_share", "mean_without_top2")
    )
    result["observations"] = len(x)
    if len(x):
        result.update(mean=float(x.mean()), median=float(np.median(x)))
    if len(x) >= 2:
        positive = np.maximum(x, 0)
        result.update(
            std=float(x.std()),
            variance=float(x.var()),
            cv=float(x.std() / x.mean()) if x.mean() > 0 else None,
            top2_positive_share=(
                float(np.sort(positive)[-2:].sum() / positive.sum()) if positive.sum() else None
            ),
        )
    if len(x) >= 3:
        result["mean_without_top2"] = float(np.sort(x)[:-2].mean())
    return result


def annual_observations(weeks: pl.DataFrame) -> pl.DataFrame:
    """Never present a partially observed counter as a complete total."""
    if "season_type" in weeks.columns:
        weeks = weeks.filter(pl.col("season_type") == "REG")
    return weeks.group_by("player_id", "season").agg(
        pl.len().alias("observed_weeks"),
        *[
            pl.when(pl.col(c).is_not_null().all()).then(pl.col(c).sum()).otherwise(None).alias(c)
            for c in COUNTERS
        ],
    )


def build_panel(features: pl.DataFrame, outcomes: pl.DataFrame, weeks: pl.DataFrame):
    """Join outcomes explicitly; build every player feature from earlier years only."""
    annual = annual_observations(weeks).sort("player_id", "season")
    history = defaultdict(list)
    for row in annual.to_dicts():
        history[row["player_id"]].append(row)
    panel = features.join(
        outcomes, on=["player_id", "forecast_season", "forecast_cutoff_date"], validate="1:1"
    ).filter(pl.col("position").is_in(POSITIONS))
    rows = []
    for original in panel.to_dicts():
        year = original["forecast_season"]
        previous = [r for r in history[original["player_id"]] if r["season"] < year]
        last = next((r for r in previous if r["season"] == year - 1), {})
        current = next((r for r in history[original["player_id"]] if r["season"] == year), {})
        row = {
            k: original.get(k)
            for k in (
                "player_id",
                "player_display_name",
                "position",
                "player_population",
                "forecast_season",
                "forecast_cutoff_date",
                "outcome_complete",
                "known_available_games_cap",
                "cutoff_preseason_team",
                "cutoff_state_resolution",
                "known_absence_games",
            )
        }
        row["forecast_cutoff_date"] = str(row["forecast_cutoff_date"])
        row["x_year"] = year
        row["x_age"] = original.get("age_at_season")
        row["x_draft"] = original.get("rookie_draft_pick")
        row["x_rookie"] = float(original["player_population"] == "rookie")
        row["x_market_only"] = float(original["player_population"] == "market_only")
        row["x_snap"] = original.get("source_offense_snap_pct")
        row["x_absence"] = original.get("known_absence_games")
        row["x_prior_weeks"] = last.get("observed_weeks", 0)
        row["x_career_weeks"] = sum(r["observed_weeks"] for r in previous)
        row["x_career_years"] = len(previous)
        row["x_gap"] = year - previous[-1]["season"] if previous else None
        for c in COUNTERS:
            row[f"x_prior_{c}"] = last.get(c)
            for period, subset in (
                ("career", previous),
                ("recent3", [r for r in previous if r["season"] >= year - 3]),
            ):
                values = [r[c] for r in subset]
                total = sum(values) if values and all(v is not None for v in values) else None
                exposure = sum(r["observed_weeks"] for r in subset)
                row[f"x_{period}_{c}_rate"] = (
                    total / exposure if total is not None and exposure else None
                )
        for target, (_, _, _, counter, denominator) in TARGETS.items():
            if counter == "actual_games":
                actual = original.get("actual_games")
                prior = original.get("games")
            else:
                actual = current.get(counter) if current else 0.0
                prior = last.get(counter)
            if denominator == "binary":
                actual = float(actual > 0) if actual is not None else None
                # A prior appearance is history, not a certain future event.
                prior = None
            elif denominator:
                d = current.get(denominator)
                actual = actual / d if actual is not None and d is not None and d > 0 else None
                d = last.get(denominator)
                prior = prior / d if prior is not None and d is not None and d > 0 else None
            elif prior is not None:
                prior *= season_length(year) / season_length(year - 1)
            row[f"y_{target}"] = actual if original["outcome_complete"] else None
            row[f"prior_{target}"] = prior
        rows.append(row)
    return rows


def constrained(values, rows, target):
    values = np.asarray(values, dtype=float).copy()
    values = np.maximum(values, 0)
    if target == "season_appearance":
        values = np.minimum(values, 1)
    for i, row in enumerate(rows):
        cap = row.get("known_available_games_cap")
        if target == "scoring_appearances":
            values[i] = min(
                values[i], season_length(row["forecast_season"]), cap if cap is not None else 17
            )
        if cap == 0:
            values[i] = np.nan if TARGETS[target][4] not in (None, "binary") else 0
    return values


def fit_fold(train, test, target, columns):
    """All transforms use training rows; no post-result selection among recipes."""
    train = [r for r in train if r.get(f"y_{target}") is not None]
    if not train:
        return {}, 0
    y = np.array([r[f"y_{target}"] for r in train])
    by_population = defaultdict(list)
    for r, value in zip(train, y, strict=True):
        by_population[r["player_population"]].append(value)
    baseline = []
    for r in test:
        prior = r.get(f"prior_{target}")
        group = by_population.get(r["player_population"], [])
        mean = float(np.mean(group)) if group else float(y.mean())
        baseline.append(prior if prior is not None else mean)
    result = {"baseline": constrained(baseline, test, target)}
    if len(train) < 30 or len({r["forecast_season"] for r in train}) < 3:
        return result, len(train)

    def matrix(data):
        return np.array(
            [[float(r[c]) if r.get(c) is not None else np.nan for c in columns] for r in data]
        )

    x, z = matrix(train), matrix(test)
    # Constant/all-null columns cannot form a tree split in the installed binning code.
    varying = [i for i in range(x.shape[1]) if len(np.unique(x[np.isfinite(x[:, i]), i])) > 1]
    if not varying:
        return result, len(train)
    with threadpool_limits(limits=1):
        ridge = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=100)
        )
        ridge.fit(x[:, varying], y)
        result["ridge"] = constrained(ridge.predict(z[:, varying]), test, target)
        booster = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=10,
            max_bins=63,
            early_stopping=False,
            random_state=20260923,
        )
        booster.fit(x[:, varying], y)
        result["boost"] = constrained(booster.predict(z[:, varying]), test, target)
    return result, len(train)


def paired_summary(rows, *, probability=False):
    coverage = dict(
        eligible_rows=len(rows),
        observed_outcomes=sum(r["actual"] is not None for r in rows),
        prediction_coverage=sum(r["prediction"] is not None for r in rows) / len(rows)
        if rows
        else None,
    )
    by_year = defaultdict(list)
    for r in rows:
        if r["actual"] is not None and r["prediction"] is not None and r["baseline"] is not None:
            by_year[r["season"]].append(r)
    annual = []
    for year, values in sorted(by_year.items()):
        actual = np.array([r["actual"] for r in values])
        pred = np.array([r["prediction"] for r in values])
        baseline = np.array([r["baseline"] for r in values])
        power = 2 if probability else 1
        loss = float(np.mean(np.abs(pred - actual) ** power))
        reference = float(np.mean(np.abs(baseline - actual) ** power))
        annual.append(
            dict(
                season=year,
                n=len(values),
                error=loss,
                baseline_error=reference,
                gain=reference - loss,
                mse_gain=float(np.mean((baseline - actual) ** 2 - (pred - actual) ** 2)),
            )
        )
    if not annual:
        return dict(n=0, years=0, annual=[], evidence="untested", **coverage)
    gains = np.array([r["gain"] for r in annual])
    boots = np.random.default_rng(20260923).choice(gains, (10000, len(gains))).mean(axis=1)
    matched = [r for values in by_year.values() for r in values]
    calibration = []
    if probability:
        for index in range(10):
            group = [r for r in matched if min(int(r["prediction"] * 10), 9) == index]
            calibration.append(
                dict(
                    lower=index / 10,
                    upper=(index + 1) / 10,
                    n=len(group),
                    predicted=float(np.mean([r["prediction"] for r in group])) if group else None,
                    observed=float(np.mean([r["actual"] for r in group])) if group else None,
                )
            )
    return dict(
        n=sum(r["n"] for r in annual),
        years=len(annual),
        annual=annual,
        metric="Brier score" if probability else "MAE",
        error=float(np.mean([r["error"] for r in annual])),
        baseline_error=float(np.mean([r["baseline_error"] for r in annual])),
        improvement=float(gains.mean()),
        ci_low=float(np.quantile(boots, 0.025)),
        ci_high=float(np.quantile(boots, 0.975)),
        mse_improvement=float(np.mean([r["mse_gain"] for r in annual])),
        positive_years=int((gains > 0).sum()),
        loo_min=float(min(np.delete(gains, i).mean() for i in range(len(gains))))
        if len(gains) > 1
        else None,
        **coverage,
        paired_coverage=len(matched) / coverage["observed_outcomes"],
        calibration=calibration,
        outside_training_range=sum(r.get("outside_training_range", False) for r in matched),
    )
