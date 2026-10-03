"""Fixed experimental methods. Nothing is registered in Armchair Labs."""

from __future__ import annotations

import numpy as np
from data import FAMILIES, MARKET, PROXY
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from engine.metrics.nextgen import constrained


def matrix(rows, columns):
    if any(c.startswith(("y_", "prior_")) for c in columns):
        raise ValueError("Outcome/reference columns cannot be model predictors")
    return np.asarray(
        [[float(r[c]) if r.get(c) is not None else np.nan for c in columns] for r in rows],
        dtype=float,
    )


def fit_pool(train, family) -> dict:
    """Approximate normal empirical Bayes: rate variance = tau² + sigma²/exposure.

    Only historical predictor rates in the earlier training cohort are used.
    These are approximate reliability weights, not isolated talent estimates or
    posterior uncertainty guarantees; football opportunities are dependent.
    """
    observed = [r for r in train if r.get(f"rate_{family}") is not None]
    if len(observed) < 20:
        return {"mean": 0.0, "strength": 0.0, "n": len(observed), "fitted": False}
    rates = np.array([r[f"rate_{family}"] for r in observed])
    exposure = np.array([r[f"exposure_{family}"] for r in observed])
    mean = float(np.average(rates, weights=exposure))
    variance = max(float(np.var(rates)), 1e-4)

    def objective(theta):
        mu, log_between, log_within = theta
        v = np.exp(log_between) + np.exp(log_within) / exposure
        return float(np.mean(np.log(v) + (rates - mu) ** 2 / v))

    fit = minimize(
        objective,
        [mean, np.log(variance / 2), np.log(variance * np.median(exposure) / 2)],
        method="L-BFGS-B",
        bounds=[(None, None), (-14, 14), (-14, 18)],
    )
    if not fit.success or not np.isfinite(fit.fun):
        return {"mean": mean, "strength": 0.0, "n": len(observed), "fitted": False}
    return {
        "mean": float(fit.x[0]),
        "strength": float(np.exp(fit.x[2] - fit.x[1])),
        "n": len(observed),
        "fitted": True,
    }


def add_rates(rows, pools, position):
    result = []
    for original in rows:
        row = dict(original)
        for family in FAMILIES[position]:
            rate, exposure = row[f"rate_{family}"], row[f"exposure_{family}"]
            pool = pools[family]
            # Unknown stays unknown; model handles missingness. No fictional zero rate.
            weight = exposure / (exposure + pool["strength"]) if exposure else None
            row[f"raw_{family}"] = rate
            row[f"pooled_{family}"] = (
                weight * rate + (1 - weight) * pool["mean"] if weight is not None else None
            )
            row[f"n_{family}"] = np.log1p(exposure) if exposure is not None else None
        result.append(row)
    return result


def rate_columns(base, position, pooled=False):
    prefix = "pooled" if pooled else "raw"
    return base + [c for f in FAMILIES[position] for c in (f"{prefix}_{f}", f"n_{f}")]


def predict_regression(train, test, target, columns, params):
    x, z = matrix(train, columns), matrix(test, columns)
    y = np.array([r[f"y_{target}"] for r in train])
    varying = [i for i in range(x.shape[1]) if len(np.unique(x[np.isfinite(x[:, i]), i])) > 1]
    if len(train) < 30 or not varying:
        prediction = np.repeat(y.mean(), len(test))
    else:
        with threadpool_limits(limits=1):
            model = HistGradientBoostingRegressor(**params).fit(x[:, varying], y)
            prediction = model.predict(z[:, varying])
    return constrained(prediction, test, target)


def mix_predictions(probabilities, experts):
    if probabilities.shape != experts.shape or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Role probabilities and conditional experts must align and sum to one")
    return np.sum(probabilities * experts, axis=1)


def role_mixture(train, test, target, columns, params):
    x, z = matrix(train, columns), matrix(test, columns)
    labels = np.array([r["y_role_state"] for r in train])
    varying = [i for i in range(x.shape[1]) if len(np.unique(x[np.isfinite(x[:, i]), i])) > 1]
    probabilities = np.zeros((len(test), 3))
    if len(np.unique(labels)) == 1 or not varying:
        for state in range(3):
            probabilities[:, state] = np.mean(labels == state)
    else:
        with threadpool_limits(limits=1):
            gate = HistGradientBoostingClassifier(**params).fit(x[:, varying], labels)
            probabilities[:, gate.classes_.astype(int)] = gate.predict_proba(z[:, varying])
    experts = np.zeros_like(probabilities)
    for state in range(3):
        subset = [r for r in train if r["y_role_state"] == state]
        if subset:
            experts[:, state] = predict_regression(subset, test, target, columns, params)
    return constrained(mix_predictions(probabilities, experts), test, target), probabilities


def recipes(position, base):
    raw = rate_columns(base, position)
    pooled = rate_columns(base, position, pooled=True)
    result = {
        "raw_rates": (raw, "nextgen_boost"),
        "pooled_rates": (pooled, "raw_rates"),
        "market_control": (raw + MARKET, "raw_rates"),
        "market_pooled": (pooled + MARKET, "market_control"),
    }
    if position in ("WR", "TE"):
        result.update(
            participation=(raw + PROXY, "raw_rates"),
            market_participation=(raw + MARKET + PROXY, "market_control"),
        )
    if position == "QB":
        # Give the direct control the same historical role-duration feature.
        result.update(
            role_direct=(raw + ["r_prior_role_games"], "raw_rates"),
            role_mixture=(raw + ["r_prior_role_games"], "role_direct"),
            market_role_direct=(raw + MARKET + ["r_prior_role_games"], "market_control"),
            market_role_mixture=(raw + MARKET + ["r_prior_role_games"], "market_role_direct"),
        )
    return result
