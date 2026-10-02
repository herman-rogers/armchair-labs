"""Conditional empirical residual forecasts and exact CRPS for the reported CDF.

Calibration predictions must come from earlier-season fits. No tree-spread proxy,
Gaussian likelihood assumption, or test-outcome fitting is used here.
"""

import numpy as np
import polars as pl


def empirical_crps(actual, samples):
    """E|X-y| - E|X-X'|/2 for an equally weighted finite predictive distribution.

    This scores the empirical distribution itself, not an unbiased Monte Carlo
    estimate for an unreported underlying distribution (hence denominator m²).
    """
    y, draws = np.asarray(actual, float), np.asarray(samples, float)
    if draws.ndim != 2 or y.shape != (len(draws),) or draws.shape[1] == 0:
        raise ValueError("Expected one outcome per nonempty sample row")
    if not np.isfinite(y).all() or not np.isfinite(draws).all():
        raise ValueError("CRPS requires finite outcomes and forecasts")
    ordered = np.sort(draws, axis=1)
    m = ordered.shape[1]
    weights = 2 * np.arange(1, m + 1) - m - 1
    return np.mean(np.abs(draws - y[:, None]), axis=1) - ordered @ weights / m**2


def residual_distribution(
    predicted_rate,
    exposure,
    calibration_prediction,
    calibration_actual,
    neighbors=100,
    count_stat=True,
):
    """Equal-weight residuals from nearest historical predicted production rates.

    Neighborhoods depend only on historical predictions and the query prediction,
    not query outcomes. All calibration rows, including zeros, are retained.
    Count support is censored at zero; signed yardage is allowed.
    """
    prediction = np.asarray(predicted_rate, float)
    exposure = np.asarray(exposure, float)
    historical_prediction = np.asarray(calibration_prediction, float)
    historical_actual = np.asarray(calibration_actual, float)
    if (
        prediction.ndim != 1
        or exposure.shape != prediction.shape
        or historical_prediction.ndim != 1
        or historical_actual.shape != historical_prediction.shape
    ):
        raise ValueError("Calibration and forecast arrays must align")
    if not all(
        np.isfinite(a).all()
        for a in [prediction, exposure, historical_prediction, historical_actual]
    ):
        raise ValueError("Residual distributions require finite inputs")
    if not len(historical_actual) or np.any(exposure <= 0) or int(neighbors) < 1:
        raise ValueError("Need calibration rows, positive exposure and neighbors")
    k = min(int(neighbors), len(historical_actual))
    ix = np.argsort(np.abs(prediction[:, None] - historical_prediction), axis=1, kind="stable")[
        :, :k
    ]
    residual = historical_actual - historical_prediction
    draws = (prediction[:, None] + residual[ix]) * exposure[:, None]
    return np.maximum(0, draws) if count_stat else draws


def constrain(samples, force_zero):
    result = np.asarray(samples, float).copy()
    mask = np.asarray(force_zero, bool)
    if mask.shape != (len(result),):
        raise ValueError("Absence mask must match sample rows")
    result[mask] = 0.0
    return result


def interval(samples, level):
    # Inverse empirical CDF: endpoints are support values, without interpolation.
    alpha = (1 - level) / 2
    return np.quantile(samples, [alpha, 1 - alpha], axis=1, method="inverted_cdf")


def metrics(actual, samples):
    scores = empirical_crps(actual, samples)
    result = dict(rows=len(scores), crps=float(scores.mean()))
    for level in [0.5, 0.8, 0.95]:
        low, high = interval(samples, level)
        suffix = int(level * 100)
        result[f"coverage_{suffix}"] = float(np.mean((actual >= low) & (actual <= high)))
        result[f"width_{suffix}"] = float(np.mean(high - low))
    return result


def evaluate(
    actual,
    model_rate,
    baseline_rate,
    exposure,
    calibration_model,
    calibration_baseline,
    calibration_actual,
    force_zero,
    target,
    neighbors,
):
    """Freeze distributions before computing scores; retain all four variants."""
    raw_model = residual_distribution(
        model_rate,
        exposure,
        calibration_model,
        calibration_actual,
        neighbors,
        "yards" not in target,
    )
    raw_baseline = residual_distribution(
        baseline_rate,
        exposure,
        calibration_baseline,
        calibration_actual,
        neighbors,
        "yards" not in target,
    )
    samples = {
        "Final refit · raw": raw_model,
        "Prior production rate · raw": raw_baseline,
        "Final refit · adjusted": constrain(raw_model, force_zero),
        "Prior production rate · adjusted": constrain(raw_baseline, force_zero),
    }
    scores = pl.DataFrame(
        [dict(model=name, **metrics(actual, draws)) for name, draws in samples.items()]
    )
    return dict(samples=samples, scores=scores)
