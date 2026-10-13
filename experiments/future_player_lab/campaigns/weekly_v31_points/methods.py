"""Past-only point corrections, matched ablations, mixtures and model blending."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..interval_research.methods import group_name, order_stat

MEMBERS = (
    "recent",
    "lgb_direct_full",
    "lgb_direct_5y",
    "lgb_direct_decay",
    "lgb_direct_no_variability",
    "lgb_direct_no_v2",
    "xgb_direct_full",
)
UNCERTAINTY = (
    "std",
    "width80",
    "width90",
    "median_offset",
    "skew80",
    "pad80",
    "pad90",
    "tail_shift80",
    "tail_width80",
    "calibration_bias",
)
FIXED = (
    "v3",
    "v31",
    "v3_residual_intervals",
    "bias_global",
    "bias_group_24",
    "bias_group_100",
    "bias_group_400",
    "bias_player_8",
    "bias_player_24",
    "bias_player_64",
    "ridge_basic",
    "ridge_uncertainty",
    "residual_basic",
    "residual_basic_half",
    "residual_uncertainty",
    "residual_uncertainty_half",
    "direct_basic",
    "direct_uncertainty",
    "mixture_basic",
    "mixture_basic_half",
    "mixture_uncertainty",
    "mixture_uncertainty_half",
    "blend_equal",
    "blend_global",
    "blend_convex",
    "blend_conditional_basic",
    "blend_conditional_uncertainty",
)
SELECTORS = ("select_rmse", "select_guarded")
FORBIDDEN = {
    "target_points",
    "appeared",
    "label_observed",
    "actual_team",
    "team_changed_in_outcome",
    "row_id",
    "player_id",
    "player_name",
    "season",
    "phase",
    "method",
}


def matrix(frame, features):
    if set(features) & FORBIDDEN:
        raise ValueError("Outcome or identity in feature allowlist")
    return frame.loc[:, features].replace([np.inf, -np.inf], np.nan).astype(float)


def tree(*, classifier=False, trees=100):
    cls = LGBMClassifier if classifier else LGBMRegressor
    return cls(
        n_estimators=trees,
        learning_rate=0.04,
        num_leaves=7,
        max_depth=3,
        min_child_samples=100,
        reg_lambda=30,
        n_jobs=2,
        verbosity=-1,
        random_state=20261013,
        deterministic=True,
        force_col_wise=True,
    )


def earlier(frame, year):
    if frame.empty or frame.season.ge(year).any():
        raise ValueError("Training requires strictly earlier seasons")
    result = frame[frame.season.ge(year - 5) & frame.target_points.notna()].copy()
    if result.empty:
        raise ValueError("No observed earlier training rows")
    return result


def fit(train, year, features):
    """One position; base forecasts supplied here must themselves be out of time."""
    train = earlier(train, year)
    if train.position.nunique() != 1:
        raise ValueError("Fit positions separately")
    if ((~train.appeared.astype(bool)) & train.target_points.ne(0)).any():
        raise ValueError("Nonappearance branch requires observed zero outcomes")
    basic = list(dict.fromkeys([*features, "base_prediction"]))
    full = list(dict.fromkeys([*basic, *UNCERTAINTY]))
    error = train.target_points.to_numpy() - train.base_prediction.to_numpy()
    groups = train.base_prediction.map(lambda x: group_name(x, True))
    global_bias = float(error.mean())
    stats = pd.DataFrame({"group": groups, "player": train.player_id, "error": error})
    group_stats = stats.groupby("group").error.agg(["sum", "count"]).to_dict("index")
    player_stats = stats.groupby("player").error.agg(["sum", "count"]).to_dict("index")
    result = dict(
        year=year,
        position=str(train.position.iloc[0]),
        min_year=int(train.season.min()),
        max_year=int(train.season.max()),
        n=len(train),
        basic=basic,
        full=full,
        global_bias=global_bias,
        group_stats=group_stats,
        player_stats=player_stats,
        models={},
    )
    for label, cols in (("basic", basic), ("uncertainty", full)):
        x = matrix(train, cols)
        # Both ridge fits see the same pregame inputs, with uncertainty added in one arm.
        ridge = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=1000.0),
        ).fit(x, error)
        result["models"][f"ridge_{label}"] = ridge
        result["models"][f"residual_{label}"] = tree().fit(x, error)
        result["models"][f"direct_{label}"] = tree().fit(x, train.target_points)
        active = train.appeared.astype(bool).to_numpy()
        probability = (
            tree(classifier=True).fit(x, active.astype(int))
            if len(np.unique(active)) == 2
            else float(active.mean())
        )
        conditional = (
            tree().fit(x.loc[active], train.loc[active, "target_points"])
            if active.sum() >= 100
            else float(train.loc[active, "target_points"].mean())
            if active.any()
            else 0.0
        )
        result["models"][f"mixture_{label}"] = (probability, conditional)
    member_cols = [f"member_{m}" for m in MEMBERS]
    forecasts = train[member_cols].to_numpy(float)
    if not np.isfinite(forecasts).all():
        raise ValueError("All ensemble members must forecast every matched row")
    errors = forecasts - train.target_points.to_numpy()[:, None]
    mse = np.maximum((errors**2).mean(axis=0), 1.0)
    result["global_weights"] = (1 / mse) / (1 / mse).sum()
    uniform = np.full(len(MEMBERS), 1 / len(MEMBERS))
    # Joint fitting accounts for correlated member errors; fixed penalty shrinks to equal weights.
    covariance = errors.T @ errors / len(errors)
    opt = minimize(
        lambda w: w @ covariance @ w + np.sum((w - uniform) ** 2),
        uniform,
        jac=lambda w: 2 * covariance @ w + 2 * (w - uniform),
        bounds=[(0, 1)] * len(MEMBERS),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 300},
    )
    if not opt.success:
        raise ValueError(f"Convex blend failed: {opt.message}")
    result["convex_weights"] = opt.x
    blend_basic = [
        "base_prediction",
        "player_history_count",
        "recent4_appearance_rate",
        "recent8_points_mean",
        "weeks_since_appearance",
        *member_cols,
    ]
    for label, cols in (("basic", blend_basic), ("uncertainty", [*blend_basic, *UNCERTAINTY])):
        result[f"blend_features_{label}"] = cols
        x = matrix(train, cols)
        result["models"][f"blend_conditional_{label}"] = [
            tree(trees=60).fit(x, errors[:, k] ** 2) for k in range(len(MEMBERS))
        ]
    return result


def predict(bundle, frame):
    """Inference uses only declared pregame features; outcomes are never read."""
    if not frame.season.eq(bundle["year"]).all():
        raise ValueError("Model year differs from forecast season")
    if not frame.position.eq(bundle["position"]).all():
        raise ValueError("Model position differs from forecast rows")
    base = frame.base_prediction.to_numpy(float)
    out = {m: base.copy() for m in ("v3", "v31", "v3_residual_intervals")}
    out["bias_global"] = base + bundle["global_bias"]
    groups = frame.base_prediction.map(lambda x: group_name(x, True))
    for prior in (24, 100, 400):
        deltas = []
        for group in groups:
            s = bundle["group_stats"].get(group, {"sum": 0, "count": 0})
            deltas.append((s["sum"] + prior * bundle["global_bias"]) / (s["count"] + prior))
        out[f"bias_group_{prior}"] = base + deltas
    for prior in (8, 24, 64):
        deltas = []
        for player, parent in zip(frame.player_id, out["bias_group_100"] - base, strict=True):
            s = bundle["player_stats"].get(player, {"sum": 0, "count": 0})
            deltas.append((s["sum"] + prior * parent) / (s["count"] + prior))
        out[f"bias_player_{prior}"] = base + deltas
    diagnostics = {}
    for label, feature_key in (("basic", "basic"), ("uncertainty", "full")):
        x = matrix(frame, bundle[feature_key])
        for kind in ("ridge", "residual", "direct"):
            name = f"{kind}_{label}"
            values = bundle["models"][name].predict(x)
            out[name] = values if kind == "direct" else base + values
        probability, conditional = bundle["models"][f"mixture_{label}"]
        p = (
            probability.predict_proba(x)[:, 1]
            if hasattr(probability, "predict_proba")
            else np.full(len(x), probability)
        )
        active_mean = (
            conditional.predict(x)
            if hasattr(conditional, "predict")
            else np.full(len(x), conditional)
        )
        out[f"mixture_{label}"] = p * active_mean
        diagnostics[f"p_appearance_{label}"] = p
        diagnostics[f"active_mean_{label}"] = active_mean
    for name in ("residual_basic", "residual_uncertainty", "mixture_basic", "mixture_uncertainty"):
        out[name + "_half"] = (out[name] + base) / 2
    values = frame[[f"member_{m}" for m in MEMBERS]].to_numpy(float)
    out["blend_equal"] = values.mean(axis=1)
    out["blend_global"] = values @ bundle["global_weights"]
    out["blend_convex"] = values @ bundle["convex_weights"]
    for label in ("basic", "uncertainty"):
        name = f"blend_conditional_{label}"
        x = matrix(frame, bundle[f"blend_features_{label}"])
        mse = np.maximum(np.column_stack([m.predict(x) for m in bundle["models"][name]]), 1.0)
        weights = 1 / mse
        weights /= weights.sum(axis=1)[:, None]
        # Partial shrinkage prevents a very small predicted MSE from dictating the entire blend.
        weights = (weights + bundle["convex_weights"]) / 2
        out[name] = np.sum(weights * values, axis=1)
    if set(out) != set(FIXED) or not np.isfinite(np.column_stack(list(out.values()))).all():
        raise ValueError("Missing or nonfinite point predictions")
    if "is_bye" in frame:
        for values in out.values():
            values[frame.is_bye.eq(1).to_numpy()] = 0
    return out, diagnostics


def choose(validation, year, *, guarded=False):
    validation = earlier(validation, year)
    validation = validation[validation.season.ge(year - 2)]
    if validation.duplicated(["row_id", "method"]).any():
        raise ValueError("Duplicate validation predictions")
    baseline = validation[validation.method.eq("v31")]
    rows = []
    for method, part in validation.groupby("method"):
        if method not in FIXED:
            continue
        if set(part.row_id) != set(baseline.row_id):
            raise ValueError("Selection methods must use identical rows")
        annual = part.assign(squared=(part.prediction - part.target_points) ** 2)
        score = float(np.sqrt(annual.groupby("season").squared.mean()).mean())
        eligible = True
        if guarded:
            masks = [baseline.base_prediction.ge(5)]
            if baseline.position.iloc[0] == "QB":
                masks.append(baseline.base_prediction.between(1, 5, inclusive="left"))
            for mask in masks:
                ids = baseline.loc[mask, "row_id"]
                if len(ids) >= 100:
                    b = baseline[baseline.row_id.isin(ids)]
                    c = part[part.row_id.isin(ids)]

                    def annual_rmse(f):
                        return float(
                            np.sqrt(
                                f.assign(e=(f.prediction - f.target_points) ** 2)
                                .groupby("season")
                                .e.mean()
                            ).mean()
                        )

                    eligible &= annual_rmse(c) <= 1.01 * annual_rmse(b)
        rows.append(dict(method=str(method), rmse=score, eligible=bool(eligible)))
    # Prefer the existing policy on exact ties, including equivalent point-control methods.
    candidates = sorted(rows, key=lambda r: (r["rmse"], r["method"] != "v31", r["method"]))
    winner = next((r["method"] for r in candidates if r["eligible"]), "v31")
    return dict(
        year=year,
        max_validation_year=int(validation.season.max()),
        method=winner,
        guarded=guarded,
        candidates=rows,
    )


def residual_intervals(past, forecast, year):
    """Calibrate changed means from earlier out-of-time errors of that same method."""
    past = earlier(past, year)
    if not forecast.season.eq(year).all():
        raise ValueError("Interval forecast year mismatch")
    errors = past.target_points - past.prediction
    groups = past.base_prediction.map(lambda x: group_name(x, True))
    lookup = {}
    for key, idx in [("all", past.index), *list(past.groupby(groups).groups.items())]:
        residual = errors.loc[idx].to_numpy()
        if len(residual) < 100:
            continue
        lookup[key] = {
            level: (
                order_stat(residual, (1 - level / 100) / 2, lower=True),
                order_stat(residual, 1 - (1 - level / 100) / 2),
            )
            for level in (80, 90)
        }
    if "all" not in lookup:
        raise ValueError("Not enough earlier residuals for interval calibration")
    result = forecast.copy()
    keys = result.base_prediction.map(lambda x: group_name(x, True))
    for level in (80, 90):
        offsets = np.array([lookup.get(k, lookup["all"])[level] for k in keys])
        result[f"lower_{level}"] = result.prediction.to_numpy() + offsets[:, 0]
        result[f"upper_{level}"] = result.prediction.to_numpy() + offsets[:, 1]
    return result
