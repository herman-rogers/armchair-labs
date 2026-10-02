"""Learn combinations from earlier out-of-year predictions, never fitted values."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

from ..evaluate import score


def balanced_weights(years, origins):
    groups = np.column_stack([years, origins])
    _, index, counts = np.unique(groups, axis=0, return_inverse=True, return_counts=True)
    w = 1.0 / counts[index]
    return w / w.sum()


def loss(y, p, w, metric="mse", groups=None):
    if metric == "mse":
        return float(w @ ((y - p) ** 2))
    if metric == "mae":
        return float(w @ abs(y - p))
    values = []
    for group in np.unique(groups):
        ix = groups == group
        values.append(score(y[ix], p[ix], np.zeros(ix.sum()))["ndcg24"] or 0.0)
    return -float(np.mean(values))


def convex_weights(p, y, w, shrink=0.0, prior=None):
    n = p.shape[1]
    prior = np.full(n, 1.0 / n) if prior is None else prior
    scale = max(float(w @ (y**2)), 1.0)
    gram = p.T @ (w[:, None] * p) / scale
    cross = p.T @ (w * y) / scale
    def objective(a):
        return float(a @ gram @ a - 2 * cross @ a + shrink * ((a - prior) ** 2).sum())

    def gradient(a):
        return 2 * (gram @ a - cross + shrink * (a - prior))
    fit = minimize(
        objective,
        prior,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda a: a.sum() - 1.0, "jac": lambda a: np.ones(n)},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not fit.success:
        raise ValueError(f"Convex ensemble optimization failed: {fit.message}")
    result = np.maximum(0.0, fit.x)
    return result / result.sum()


def capture_score(actual, predicted, groups, top_k):
    values = []
    for group in np.unique(groups):
        ix = groups == group
        a, p = actual[ix], predicted[ix]
        denominator = np.sort(a)[::-1][:top_k].sum()
        values.append(
            float(a[np.argsort(-p, kind="stable")[:top_k]].sum() / denominator)
            if denominator > 0
            else 0.0
        )
    return float(np.mean(values))


def greedy_weights(p, y, w, steps, metric="mse", groups=None, top_k=24):
    n = p.shape[1]
    counts, running = np.zeros(n), np.zeros(len(y))
    best_loss, best, snapshots = np.inf, None, {}
    for step in range(1, max(steps) + 1):
        trials = (running[:, None] + p) / step
        if metric == "mse":
            errors = w @ ((y[:, None] - trials) ** 2)
        elif metric == "mae":
            errors = w @ abs(y[:, None] - trials)
        else:
            gains = []
            for group in np.unique(groups):
                ix = groups == group
                actual = y[ix] if metric == "capture" else np.maximum(y[ix], 0)
                predicted = trials[ix]
                k = min(top_k if metric == "capture" else 24, len(actual))
                discount = np.ones(k) if metric == "capture" else 1 / np.log2(np.arange(k) + 2)
                denominator = np.sort(actual)[::-1][:k] @ discount
                order = np.argsort(-predicted, axis=0, kind="stable")[:k]
                gains.append(
                    (actual[order] * discount[:, None]).sum(axis=0) / denominator
                    if denominator > 0
                    else np.zeros(n)
                )
            errors = -np.mean(gains, axis=0)
        chosen = int(np.argmin(errors))
        running += p[:, chosen]
        counts[chosen] += 1
        if errors[chosen] < best_loss:
            best_loss, best = errors[chosen], counts.copy() / step
        if step in steps:
            snapshots[step] = best.copy()
    return snapshots


def percentiles(p, groups):
    result = np.empty_like(p)
    for group in np.unique(groups):
        ix = groups == group
        for j in range(p.shape[1]):
            result[ix, j] = rankdata(p[ix, j]) / (ix.sum() + 1)
    return result


def combine(
    train_p,
    train_y,
    train_years,
    train_origins,
    test_p,
    test_origins,
    train_context,
    test_context,
    names,
    seed=1,
    top_k=24,
):
    if not np.isfinite(train_p).all() or not np.isfinite(test_p).all():
        raise ValueError("Incomplete base-model forecasts; do not silently change membership")
    w = balanced_weights(train_years, train_origins)
    groups = train_years * 32 + train_origins
    losses = w @ ((train_p - train_y[:, None]) ** 2)
    order = np.argsort(losses, kind="stable")
    results, detail = {}, {}

    def weighted(name, weights):
        results[name] = test_p @ weights
        detail[name] = {
            "weights": {names[j]: float(v) for j, v in enumerate(weights) if abs(v) > 1e-8},
            "effective_models": float(1 / (weights**2).sum()),
        }

    for k in [1, 2, 3, 5, 10, 20, len(names)]:
        weights = np.zeros(len(names))
        weights[order[:k]] = 1 / min(k, len(names))
        weighted(f"top_{k}", weights)
    for power in [0.5, 1.0, 2.0]:
        weights = np.maximum(losses, 1e-8) ** -power
        weighted(f"inverse_{power}", weights / weights.sum())
    for metric in ("mse", "mae", "rank", "capture"):
        steps = [5, 20, 100] if metric in {"mse", "mae"} else [5, 20]
        for step, weights in greedy_weights(
            train_p, train_y, w, steps, metric, groups, top_k
        ).items():
            weighted(f"greedy_{metric}_{step}", weights)
    for shrink in [0.0, 0.01, 0.1, 1.0]:
        weighted(f"convex_{shrink}", convex_weights(train_p, train_y, w, shrink))
    # Residual stacking permits signed corrections but regularizes toward top-three averaging.
    anchor = train_p[:, order[:3]].mean(axis=1)
    test_anchor = test_p[:, order[:3]].mean(axis=1)
    scale = max(float(np.sqrt(w @ (train_y**2))), 1.0)
    z = (train_p - anchor[:, None]) / scale
    t = (test_p - test_anchor[:, None]) / scale
    for alpha in [0.01, 1.0, 100.0]:
        model = Ridge(alpha=alpha).fit(z, (train_y - anchor) / scale, sample_weight=w * len(w))
        name = f"ridge_stack_{alpha}"
        results[name] = test_anchor + scale * model.predict(t)
        detail[name] = {
            "coefficients": dict(zip(names, model.coef_.tolist(), strict=True)),
            "intercept_rate": float(model.intercept_),
            "anchor": [names[i] for i in order[:3]],
        }
    # Context is forecast-time origin, rookie status and observed-history count.
    for depth in [2, 4]:
        model = ExtraTreesRegressor(
            n_estimators=160,
            max_depth=depth,
            min_samples_leaf=30,
            max_features=0.7,
            n_jobs=1,
            random_state=seed,
        )
        model.fit(
            np.column_stack([train_p, train_context]), train_y - anchor, sample_weight=w * len(w)
        )
        name = f"nonlinear_stack_{depth}"
        results[name] = test_anchor + model.predict(np.column_stack([test_p, test_context]))
        detail[name] = {"depth": depth, "meta_training_rows": len(w)}
    cohort_pred = test_p @ convex_weights(train_p, train_y, w, 0.1)
    cohort_detail = {}
    for rookie in [0.0, 1.0]:
        ix, tx = train_context[:, 1] == rookie, test_context[:, 1] == rookie
        if ix.sum() < 50 or len(np.unique(train_years[ix])) < 3:
            cohort_detail[str(rookie)] = {"fallback": "global convex"}
            continue
        cw = balanced_weights(train_years[ix], train_origins[ix])
        weights = convex_weights(train_p[ix], train_y[ix], cw, 0.1)
        cohort_pred[tx] = test_p[tx] @ weights
        cohort_detail[str(rookie)] = {names[j]: float(v) for j, v in enumerate(weights) if v > 1e-8}
    results["cohort_convex"] = cohort_pred
    detail["cohort_convex"] = cohort_detail
    # Percentile combinations explicitly calibrated on earlier out-of-year outcomes.
    rank_train, rank_test = percentiles(train_p, groups), percentiles(test_p, test_origins)
    for k in [3, 10, len(names)]:
        train_rank = rank_train[:, order[:k]].mean(axis=1)
        test_rank = rank_test[:, order[:k]].mean(axis=1)
        calibration = IsotonicRegression(out_of_bounds="clip").fit(
            train_rank, train_y, sample_weight=w
        )
        results[f"rank_calibrated_{k}"] = calibration.predict(test_rank)
        detail[f"rank_calibrated_{k}"] = {
            "members": [names[i] for i in order[:k]],
            "calibration_rows": len(w),
        }
    return results, detail


def select_policy(history, year, metric, seasons=3):
    earlier = sorted(y for y in history if y < year)[-seasons:]
    if len(earlier) < seasons:
        raise ValueError("Insufficient out-of-year ensemble evaluations")
    eligible = set.intersection(*(set(history[y]) for y in earlier))
    values = {name: np.mean([history[y][name][metric] for y in earlier]) for name in eligible}
    if metric in {"ndcg24", "capture"}:
        values = {n: -v for n, v in values.items()}
    return min(values, key=lambda n: (values[n], n)), earlier, values
