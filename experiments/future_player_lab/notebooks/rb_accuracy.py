"""Scale-aware accuracy reporting; no fitting or model dependencies."""

import numpy as np


def accuracy_context(actual, predicted, baseline, tolerance=10, positive_only=False):
    actual, predicted, baseline = (
        np.asarray(v, dtype=float) for v in (actual, predicted, baseline)
    )
    if actual.shape != predicted.shape or actual.shape != baseline.shape:
        raise ValueError("Accuracy comparisons require matched players")
    mask = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(baseline)
    if positive_only:
        mask &= actual > 0
    a, p, b = actual[mask], predicted[mask], baseline[mask]
    result = dict.fromkeys(
        [
            "actual_mean",
            "actual_median",
            "actual_p90",
            "zero_actual_pct",
            "mae",
            "rmse",
            "wape_pct",
            "baseline_mae",
            "mae_improvement_pct",
            "rmse_improvement_pct",
            "r2",
            "within_tolerance_pct",
            "baseline_within_tolerance_pct",
        ]
    )
    result["players"] = len(a)
    if not len(a):
        return result, mask
    error, base_error = p - a, b - a
    mae, base_mae = np.mean(abs(error)), np.mean(abs(base_error))
    rmse, base_rmse = np.sqrt(np.mean(error**2)), np.sqrt(np.mean(base_error**2))
    production, variation = np.sum(abs(a)), np.sum((a - a.mean()) ** 2)
    result.update(
        actual_mean=float(a.mean()),
        actual_median=float(np.median(a)),
        actual_p90=float(np.quantile(a, 0.9)),
        zero_actual_pct=float(100 * np.mean(a == 0)),
        mae=float(mae),
        rmse=float(rmse),
        baseline_mae=float(base_mae),
        wape_pct=float(100 * np.sum(abs(error)) / production) if production > 0 else None,
        mae_improvement_pct=float(100 * (1 - mae / base_mae)) if base_mae > 0 else None,
        rmse_improvement_pct=float(100 * (1 - rmse / base_rmse)) if base_rmse > 0 else None,
        r2=float(1 - np.sum(error**2) / variation) if variation > 0 else None,
        within_tolerance_pct=float(100 * np.mean(abs(error) <= tolerance)),
        baseline_within_tolerance_pct=float(100 * np.mean(abs(base_error) <= tolerance)),
    )
    return result, mask
