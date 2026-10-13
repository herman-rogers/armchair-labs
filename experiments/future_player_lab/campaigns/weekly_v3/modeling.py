"""Chronological weekly mean and distribution estimators for Opening Drive v3.

All residual estimators consume forecasts made by models fitted before the outcome
season. Distribution location is kept at the mean forecast: predictable bias and
dispersion are recorded separately rather than treating squared bias as variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

QUANTILES = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
QCOLS = [f"q{round(q * 100):02d}" for q in QUANTILES]
MEAN_METHODS = [
    "v2",
    "recent",
    "lgb_direct_full",
    "lgb_correction_full",
    "lgb_direct_5y",
    "lgb_direct_decay",
    "lgb_direct_no_variability",
    "lgb_direct_no_v2",
    "xgb_direct_full",
]
DIST_METHODS = [
    "pooled",
    "mean_bin",
    "player_shrinkage",
    "conditional_scale",
    "conditional_no_variability",
    "quantile",
]
VARIABILITY_TOKENS = ("variance", "volatility", "std", "cv_", "ceiling", "floor")


def feature_columns(features, method):
    if method == "lgb_direct_no_v2":
        return [c for c in features if not c.startswith("v2_")]
    if "no_variability" in method:
        return [c for c in features if not any(t in c.lower() for t in VARIABILITY_TOKENS)]
    return list(features)


def matrix(frame, features):
    return frame.reindex(columns=features).replace([np.inf, -np.inf], np.nan).astype(float)


def tree(*, quantile=None, n_estimators=160):
    args = dict(
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=15,
        max_depth=5,
        min_child_samples=60,
        reg_lambda=15.0,
        n_jobs=4,
        verbosity=-1,
        random_state=20261011,
        deterministic=True,
        force_col_wise=True,
    )
    if quantile is not None:
        args.update(objective="quantile", alpha=float(quantile))
    return LGBMRegressor(**args)


def fit_mean(train, features, method, year):
    """Fit using earlier seasons only; callers cannot silently cross the cutoff."""
    if len(train) and int(train.season.max()) >= year:
        raise ValueError("Mean training crosses the forecast-season cutoff")
    chosen = feature_columns(features, method)
    if method in ("v2", "recent"):
        return {"method": method, "features": chosen, "model": None, "year": year}
    if method.endswith("5y"):
        train = train[train.season >= year - 5]
    weights = (
        np.power(0.5, (year - 1 - train.season.to_numpy()) / 4)
        if method.endswith("decay")
        else None
    )
    target = train.target_points.to_numpy().copy()
    if "correction" in method:
        target -= train.v2_weekly_mean.to_numpy()
    if method.startswith("xgb"):
        model = XGBRegressor(
            n_estimators=180,
            max_depth=3,
            min_child_weight=40,
            reg_lambda=20,
            learning_rate=0.045,
            tree_method="hist",
            device="cpu",
            n_jobs=4,
            random_state=20261011,
        )
    else:
        model = tree()
    model.fit(matrix(train, chosen), target, sample_weight=weights)
    return {"method": method, "features": chosen, "model": model, "year": year}


def predict_mean(bundle, frame):
    method = bundle["method"]
    if method == "v2":
        return frame.v2_weekly_mean.to_numpy()
    if method == "recent":
        cols = [
            c
            for c in (
                "recent4_points_mean",
                "recent_points_mean",
                "points_mean_4",
                "player_history_mean",
                "history_points_mean",
                "points_mean_8",
            )
            if c in frame
        ]
        if not cols:
            raise ValueError("Recent baseline requires an explicitly named historical mean")
        history = frame.player_history_mean.fillna(0).to_numpy()
        recent = frame[cols[0]].fillna(frame.player_history_mean).fillna(0).to_numpy()
        count = frame.get("player_history_count", pd.Series(4.0, index=frame.index)).to_numpy()
        weight = np.minimum(count, 4) / (np.minimum(count, 4) + 2)
        return weight * recent + (1 - weight) * history
    value = bundle["model"].predict(matrix(frame, bundle["features"]))
    if "correction" in method:
        value += frame.v2_weekly_mean.to_numpy()
    return np.maximum(value, 0)


def _quantile_moments(q):
    # Flat tails make a deliberately bounded numerical approximation, common to
    # every candidate. Report this as quantile-integrated CRPS, not exact CRPS.
    p = np.r_[0, QUANTILES, 1]
    v = np.column_stack([q[:, 0], q, q[:, -1]])
    mean = np.trapezoid(v, p, axis=1)
    left, right = v[:, :-1], v[:, 1:]
    second = np.sum(np.diff(p) * (left * left + left * right + right * right) / 3, axis=1)
    variance = np.maximum(second - mean * mean, 0)
    return mean, np.sqrt(variance)


def fit_distribution(calibration, features, method, year):
    """Fit dispersion to strictly previous, out-of-time forecast errors."""
    if calibration.empty or int(calibration.season.max()) >= year:
        raise ValueError("Calibration must contain only earlier-season forecasts")
    cal = calibration[calibration.season >= year - 5].copy()
    error = cal.target_points.to_numpy() - cal.prediction.to_numpy()
    bias = float(error.mean())
    centered = error - bias
    pooled_std = max(float(centered.std()), 1.0)
    base = dict(
        method=method,
        year=year,
        bias=bias,
        pooled_std=pooled_std,
        quantiles=np.quantile(centered, QUANTILES),
        features=list(features),
        calibration_min_year=int(cal.season.min()),
        calibration_max_year=int(cal.season.max()),
        calibration_n=len(cal),
    )
    edges = np.unique(np.quantile(cal.prediction, [0, 0.2, 0.4, 0.6, 0.8, 1]))
    edges[0], edges[-1] = -np.inf, np.inf
    groups = np.digitize(cal.prediction, edges[1:-1])
    base["edges"] = edges
    base["bin_quantiles"] = {}
    base["bin_std"] = {}
    for group in np.unique(groups):
        e = error[groups == group]
        e = e - e.mean()
        base["bin_quantiles"][int(group)] = np.quantile(e, QUANTILES)
        base["bin_std"][int(group)] = max(float(e.std()), 1.0)
    if method == "player_shrinkage":
        cal["error"] = error
        stats = cal.groupby("player_id").error.agg(["count", "var"])
        base["player_stats"] = {
            str(k): (int(v["count"]), float(v["var"])) for k, v in stats.fillna(0).iterrows()
        }
    elif method.startswith("conditional"):
        chosen = feature_columns(features, method) + ["prediction"]
        base["features"] = chosen
        # Last calibration year is held out from the scale fit and supplies the
        # standardized residual shape. Otherwise fitted errors shrink coverage.
        fit = cal.season < cal.season.max()
        if fit.sum() < 200:
            base["method"] = "mean_bin"
            base["fallback"] = "insufficient earlier scale-fit rows"
            return base
        model = tree(n_estimators=110)
        model.fit(matrix(cal.loc[fit], chosen), np.log1p(centered[fit] ** 2))
        scale = np.sqrt(np.maximum(np.expm1(model.predict(matrix(cal.loc[~fit], chosen))), 1))
        standardized = centered[~fit] / scale
        standardized -= standardized.mean()
        base.update(model=model, standardized_quantiles=np.quantile(standardized, QUANTILES))
    elif method == "quantile":
        chosen = list(features) + ["prediction"]
        base["features"] = chosen
        base["models"] = []
        for q in QUANTILES:
            model = tree(quantile=q, n_estimators=110)
            model.fit(matrix(cal, chosen), centered)
            base["models"].append(model)
    return base


def predict_distribution(bundle, frame, *, diagnostics=False):
    method = bundle["method"]
    n = len(frame)
    groups = np.digitize(frame.prediction, bundle["edges"][1:-1])
    if method == "pooled":
        q = np.tile(bundle["quantiles"], (n, 1))
    elif method in ("mean_bin", "player_shrinkage"):
        q = np.vstack([bundle["bin_quantiles"].get(int(g), bundle["quantiles"]) for g in groups])
        if method == "player_shrinkage":
            # 24 historical weekly errors are the fixed prior equivalent sample
            # size. Player variance uses its own mean-centered residuals.
            scales = []
            for player, g in zip(frame.player_id, groups, strict=True):
                count, variance = bundle["player_stats"].get(str(player), (0, 0.0))
                prior = bundle["bin_std"].get(int(g), bundle["pooled_std"])
                variance = (max(count - 1, 0) * variance + 24 * prior**2) / (max(count - 1, 0) + 24)
                scales.append(np.sqrt(variance) / prior)
            q *= np.asarray(scales)[:, None]
    elif method.startswith("conditional"):
        scale = np.sqrt(
            np.maximum(np.expm1(bundle["model"].predict(matrix(frame, bundle["features"]))), 1)
        )
        q = scale[:, None] * bundle["standardized_quantiles"][None, :]
    elif method == "quantile":
        q = np.column_stack(
            [m.predict(matrix(frame, bundle["features"])) for m in bundle["models"]]
        )
    else:
        raise ValueError(method)
    crossings = (np.diff(q, axis=1) < 0).any(axis=1)
    q = np.sort(q, axis=1)
    center, std = _quantile_moments(q)
    q += (frame.prediction.to_numpy() - center)[:, None]
    return (q, std, crossings) if diagnostics else (q, std)


def score_predictions(frame):
    valid = frame.target_points.notna()
    frame = frame.loc[valid]
    if frame.empty:
        return {"n": 0}
    y = frame.target_points.to_numpy()
    error = y - frame.prediction.to_numpy()
    result = dict(
        n=len(frame),
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.mean(error**2))),
        bias=float(-error.mean()),
    )
    if all(c in frame for c in QCOLS):
        q = frame[QCOLS].to_numpy()
        delta = y[:, None] - q
        loss = np.maximum(QUANTILES * delta, (QUANTILES - 1) * delta)
        result["pinball"] = float(loss.mean())
        result["crps_quantile"] = float(
            2
            * np.trapezoid(
                np.column_stack([loss[:, 0], loss, loss[:, -1]]), np.r_[0, QUANTILES, 1], axis=1
            ).mean()
        )
        for interval, low, high in ((80, "q10", "q90"), (90, "q05", "q95")):
            result[f"coverage_{interval}"] = float(((y >= frame[low]) & (y <= frame[high])).mean())
            result[f"width_{interval}"] = float((frame[high] - frame[low]).mean())
        if "raw_quantile_crossing" in frame:
            result["raw_quantile_crossing_rate"] = float(frame.raw_quantile_crossing.mean())
        for threshold in (0, 5, 10, 20):
            probability = np.array(
                [1 - np.interp(threshold, row, QUANTILES, left=0, right=1) for row in q]
            )
            result[f"brier_over_{threshold}"] = float(np.mean((probability - (y > threshold)) ** 2))
    return result


def select_candidate(scores, metric):
    """Equal-season selection, never weighting a large year more heavily."""
    scores = scores[scores.season.between(2018, 2020)]
    if scores.empty:
        raise ValueError("No selection-season scores")
    return str(scores.groupby("method")[metric].mean().sort_values().index[0])
