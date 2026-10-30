"""Isolated in-season residual retraining challenger to the frozen v3.2 model.

All feature transforms are fixed, and fits admit only earlier seasons or earlier
completed weeks. This challenger is never selected as the serving default here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..weekly_v32 import uncertainty as base
from .uncertainty import metrics

FEATURES = (
    "prediction",
    "player_history_count",
    "report_known",
    "report_questionable",
    "room_out_usage",
)
SCALE = np.array([20.0, 32.0, 1.0, 1.0, 20.0])
ALPHA = 100.0


def matrix(frame):
    # Missingness is explicit; fixed scaling does not consume any evaluation rows.
    values = frame.reindex(columns=FEATURES).to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    return np.column_stack([np.ones(len(frame)), np.where(missing, 0, values) / SCALE, missing])


def fit(history, season, week, *, in_season=True):
    history = history.copy()
    if history.empty:
        raise ValueError("Retraining requires completed earlier forecasts")
    forbidden = history.season.gt(season) | (history.season.eq(season) & history.week.ge(week))
    if forbidden.any():
        raise ValueError("Retraining would consume current or future outcomes")
    history = history[history.season.ge(season - 3)]
    if not in_season:
        history = history[history.season.lt(season)]
    known = np.isfinite(history.target_points) & np.isfinite(history.prediction)
    history = history[known]
    if len(history) < 100:
        raise ValueError("Retraining requires at least 100 completed forecasts")
    x = matrix(history)
    y = (history.target_points - history.prediction).to_numpy()
    weights = np.where(history.season.eq(season), 5.0, 1.0)
    coef = np.linalg.solve(
        x.T @ (weights[:, None] * x) + ALPHA * np.eye(x.shape[1]), x.T @ (weights * y)
    )
    return dict(
        season=int(season),
        week=int(week),
        coefficients=coef.tolist(),
        n=len(history),
        current_season_n=int(history.season.eq(season).sum()),
        max_training_season=int(history.season.max()),
        max_current_week=int(history.loc[history.season.eq(season), "week"].max())
        if history.season.eq(season).any()
        else None,
        features=list(FEATURES),
        alpha=ALPHA,
        current_season_weight=5,
        policy="in_season_residual_ridge" if in_season else "prior_seasons_residual_ridge",
    )


def predict(frame, contract):
    if not frame.season.eq(contract["season"]).all() or not frame.week.ge(contract["week"]).all():
        raise ValueError("Residual model cannot predict before its training cutoff")
    delta = matrix(frame) @ np.asarray(contract["coefficients"])
    # Fixed bounded correction retains the v3.2 base and limits sparse-week updates.
    delta = np.clip(delta, -5, 5)
    deterministic = frame.get("report_out", pd.Series(0, index=frame.index)).eq(1)
    deterministic |= frame.get("is_bye", pd.Series(0, index=frame.index)).eq(1)
    result = frame.prediction.to_numpy() + delta
    result[deterministic.to_numpy()] = 0
    return result


def apply(players, fitted_run, completed_forecasts=None):
    history = pd.read_parquet(Path(fitted_run) / "history.parquet")
    contract = json.loads((Path(fitted_run) / "retraining_contract.json").read_text())
    if contract["features"] != list(FEATURES) or contract["alpha"] != ALPHA:
        raise ValueError("Retraining inference policy changed")
    out = players.copy()
    column = "forecast_horizon" if "forecast_horizon" in out else "horizon"
    if completed_forecasts is not None and len(completed_forecasts):
        added = completed_forecasts.copy()
        if "forecast_horizon" in added and "horizon" not in added:
            added["horizon"] = added.forecast_horizon
        keys = ["season", "week", "player_id", "horizon"]
        if added.duplicated(keys).any():
            raise ValueError("Duplicate completed forecast labels")
        if added.season.lt(contract["forecast_season"]).any():
            raise ValueError("Live additions cannot replace frozen historical rows")
        history = pd.concat([history, added], ignore_index=True)
    all_draws = np.stack(out.samples).copy()
    means = out.prediction.to_numpy().copy()
    for (horizon, position, season, week), indices in out.groupby(
        [column, "position", "season", "week"]
    ).indices.items():
        prior = history[history.horizon.eq(horizon) & history.position.eq(position)]
        # Reject future additions, rather than silently interpreting them as admissible.
        bundle = fit(prior, int(season), int(week))
        part = out.iloc[indices]
        corrected = predict(part, bundle)
        all_draws[indices] += (corrected - part.prediction.to_numpy())[:, None]
        deterministic = part.get("report_out", pd.Series(0, index=part.index)).eq(1)
        deterministic |= part.get("is_bye", pd.Series(0, index=part.index)).eq(1)
        all_draws[indices[deterministic.to_numpy()]] = 0
        means[indices] = corrected
    out["prediction"] = means
    out = base.frame_with_samples(out, all_draws)
    out["candidate_method"] = "v4_in_season_residual_ridge"
    return out


def evaluate(history, distributions):
    forecasts, receipts, scores = [], [], []
    for method, name in (("v32", "v32_frozen"), ("v4", "v4_calibration_only")):
        control = distributions[distributions.distribution_method.eq(method)].copy()
        control["candidate_method"] = name
        forecasts.append(control)
    # A prior-only residual fit isolates the effect of consuming current-season labels.
    for (horizon, year, position), part in distributions[
        distributions.distribution_method.eq("v4")
    ].groupby(["horizon", "season", "position"]):
        pool = history[history.horizon.eq(horizon) & history.position.eq(position)]
        for week, test in part.groupby("week"):
            test = test.copy()
            prior = pool[pool.season.lt(year) | (pool.season.eq(year) & pool.week.lt(week))]
            for name, current in (("prior_only_residual", False), ("in_season_residual", True)):
                contract = fit(prior, year, week, in_season=current)
                receipts.append(dict(horizon=horizon, position=position, **contract))
                prediction = predict(test, contract)
                draws = np.stack(test.samples) + (prediction - test.prediction.to_numpy())[:, None]
                deterministic = test.report_out.eq(1).to_numpy()
                draws[deterministic] = 0
                test_copy = test.copy()
                test_copy["prediction"] = prediction
                test_copy = base.frame_with_samples(test_copy, draws)
                test_copy["candidate_method"] = name
                forecasts.append(test_copy)
    output = pd.concat(forecasts, ignore_index=True)
    for (horizon, year, position, method), part in output.groupby(
        ["horizon", "season", "position", "candidate_method"]
    ):
        for cohort, subset in (
            ("all", part),
            ("points_5_plus", part[part.base_prediction.ge(5)]),
            ("reported", part[part.report_known.eq(1)]),
        ):
            if len(subset):
                scores.append(
                    dict(
                        horizon=horizon,
                        season=year,
                        position=position,
                        method=method,
                        cohort=cohort,
                        **metrics(subset, np.stack(subset.samples)),
                    )
                )
    return pd.DataFrame(scores), output, receipts
