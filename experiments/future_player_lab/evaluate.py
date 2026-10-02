"""Chronological selection, forecast scoring and season-cluster uncertainty."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import rankdata


def ranking(actual, predicted, origins):
    correlations, ndcgs, recalls = [], [], []
    for origin in np.unique(origins):
        ix = (origins == origin) & np.isfinite(actual) & np.isfinite(predicted)
        a, p = actual[ix], predicted[ix]
        if len(a) < 2:
            continue
        ar, pr = rankdata(a), rankdata(p)
        if ar.std() > 0 and pr.std() > 0:
            correlations.append(float(np.corrcoef(ar, pr)[0, 1]))
        k = min(24, len(a))
        ideal = np.argsort(-a, kind="stable")[:k]
        chosen = np.argsort(-p, kind="stable")[:k]
        discount = 1 / np.log2(np.arange(k) + 2)
        best = (np.maximum(0, a[ideal]) * discount).sum()
        if best > 0:
            ndcgs.append(float((np.maximum(0, a[chosen]) * discount).sum() / best))
        recalls.append(len(set(ideal) & set(chosen)) / k)
    return {
        "spearman": float(np.mean(correlations)) if correlations else None,
        "ndcg24": float(np.mean(ndcgs)) if ndcgs else None,
        "top24_recall": float(np.mean(recalls)) if recalls else None,
    }


def score(actual, predicted, origins):
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return {
            "n": 0,
            "mse": None,
            "mae": None,
            "bias": None,
            "spearman": None,
            "ndcg24": None,
            "top24_recall": None,
        }
    residual = predicted[valid] - actual[valid]
    return {
        "n": int(valid.sum()),
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(abs(residual))),
        "bias": float(np.mean(residual)),
        **ranking(actual, predicted, origins),
    }


def select(history, year, inner_seasons, metric="mse"):
    earlier = sorted({y for by_year in history.values() for y in by_year if y < year})[
        -inner_seasons:
    ]
    if len(earlier) < inner_seasons:
        raise ValueError("Not enough historical validation years")
    losses = {}
    for name, by_year in history.items():
        values = [by_year.get(y, {}).get(metric) for y in earlier]
        if all(v is not None and np.isfinite(v) for v in values):
            losses[name] = float(np.mean(values)) * (-1 if metric == "ndcg24" else 1)
    if not losses:
        raise ValueError("No candidate has complete earlier validation")
    return sorted(losses, key=lambda n: (losses[n], n)), earlier, losses


def interval(residuals, alpha):
    """Empirical symmetric intervals in per-calendar-week units; no IID guarantee."""
    if len(residuals) < 30:
        return np.nan
    r = np.sort(np.abs(residuals[np.isfinite(residuals)]))
    if len(r) < 30:
        return np.nan
    return float(r[min(len(r) - 1, int(np.ceil((len(r) + 1) * (1 - alpha))) - 1)])


def paired_interval(differences, draws, seed):
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, (draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def matched_scores(frame, model, control, target):
    keys = ["player_id", "year", "origin", "horizon", "position"]
    fields = [*keys, f"actual_{target}", f"pred_{target}"]
    left = frame.filter(pl.col("model") == model).select(fields)
    right = frame.filter(pl.col("model") == control).select(fields)
    pairs = left.join(right, on=keys, suffix="_control", validate="1:1")
    valid = (
        pl.col(f"actual_{target}").is_finite()
        & pl.col(f"pred_{target}").is_finite()
        & pl.col(f"pred_{target}_control").is_finite()
    )
    pairs = pairs.filter(valid)
    result = []
    for (year,), block in pairs.partition_by("year", as_dict=True).items():
        actual = block[f"actual_{target}"].to_numpy()
        if not np.array_equal(actual, block[f"actual_{target}_control"].to_numpy()):
            raise ValueError("Forecast artifacts disagree about actual outcomes")
        p = block[f"pred_{target}"].to_numpy()
        b = block[f"pred_{target}_control"].to_numpy()
        origin = block["origin"].to_numpy()
        ranks, base_ranks = ranking(actual, p, origin), ranking(actual, b, origin)
        result.append(
            {
                "year": year,
                "n": len(block),
                "model_mse": float(np.mean((p - actual) ** 2)),
                "control_mse": float(np.mean((b - actual) ** 2)),
                "delta_mse": float(np.mean((p - actual) ** 2 - (b - actual) ** 2)),
                "delta_mae": float(np.mean(abs(p - actual) - abs(b - actual))),
                "delta_ndcg24": (
                    ranks["ndcg24"] - base_ranks["ndcg24"]
                    if ranks["ndcg24"] is not None and base_ranks["ndcg24"] is not None
                    else None
                ),
            }
        )
    return result
