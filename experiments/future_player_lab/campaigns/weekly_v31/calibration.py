"""Past-only, width-scaled interval corrections; point/distribution models stay fixed.

Uses the CQR nonconformity score (Romano et al., https://arxiv.org/abs/1905.03222),
normalized by pregame interval width. Corrections only expand intervals. Repeated
players, temporal drift, and reused development years preclude a coverage guarantee.
The corrected bounds are NOT replacement quantiles of the original distribution.
"""

from __future__ import annotations

import math

import numpy as np

LEVELS = {80: ("q10", "q90"), 90: ("q05", "q95")}
POLICIES = ("position", "position_workload")


def workload(prediction):
    return "under_5" if prediction < 5 else "5_to_10" if prediction < 10 else "10_plus"


def fit(frame, before_season, *, policy="position_workload", min_rows=200, levels=(80, 90)):
    """Caller supplies earlier out-of-time forecasts, never this season's labels."""
    if policy not in POLICIES or min_rows < 1:
        raise ValueError("Invalid calibration policy/minimum support")
    if not frame.empty and (frame.season >= before_season).any():
        raise ValueError("Calibration must contain strictly earlier seasons")
    if "phase" in frame and frame.phase.isin(["train", "training", "in_sample"]).any():
        raise ValueError("Calibration requires out-of-time predictions")
    cal = frame[frame.season.ge(before_season - 5) & frame.target_points.notna()].copy()
    cal["workload"] = cal.prediction.map(workload)
    result = dict(
        schema_version=1,
        before_season=int(before_season),
        policy=policy,
        min_rows=min_rows,
        window_seasons=5,
        levels=list(levels),
        groups={},
        calibration_rows=len(cal),
        min_season=int(cal.season.min()) if len(cal) else None,
        max_season=int(cal.season.max()) if len(cal) else None,
    )
    for position, group in cal.groupby("position"):
        groups = [(str(position), group)]
        if policy == "position_workload":
            groups += [(f"{position}/{name}", part) for name, part in group.groupby("workload")]
        for key, part in groups:
            estimates = {}
            for level in levels:
                lo, hi = LEVELS[level]
                values = part[[lo, hi, "target_points"]].to_numpy(float)
                if not np.isfinite(values).all() or (values[:, 0] > values[:, 1]).any():
                    raise ValueError("Invalid calibration outcomes or interval bounds")
                n = len(part)
                rank = math.ceil((n + 1) * level / 100)
                if n < min_rows or rank > n:
                    continue
                scale = np.maximum(values[:, 1] - values[:, 0], 1.0)
                scores = (
                    np.maximum(values[:, 0] - values[:, 2], values[:, 2] - values[:, 1]) / scale
                )
                correction = max(0.0, float(np.partition(scores, rank - 1)[rank - 1]))
                estimates[str(level)] = dict(correction=correction, n=n, rank=rank)
            result["groups"][key] = estimates
    return result


def apply(frame, bundle):
    if not frame.empty and not frame.season.eq(bundle["before_season"]).all():
        raise ValueError("Calibration bundle must match forecast season")
    result = frame.copy()
    for level in bundle["levels"]:
        low, high = LEVELS[level]
        bounds, supports, groups = [], [], []
        for row in frame.to_dict("records"):
            lo, hi = float(row[low]), float(row[high])
            if not np.isfinite([lo, hi, row["prediction"]]).all() or lo > hi:
                raise ValueError("Invalid forecast interval")
            key = f"{row['position']}/{workload(row['prediction'])}"
            estimate = bundle["groups"].get(key, {}).get(str(level))
            if estimate is None:
                key = str(row["position"])
                estimate = bundle["groups"].get(key, {}).get(str(level))
            if row.get("is_bye", 0) == 1:
                lo, hi, key, estimate = 0.0, 0.0, "scheduled_bye", None
            elif estimate is not None:
                padding = estimate["correction"] * max(hi - lo, 1.0)
                lo, hi = lo - padding, hi + padding
            else:
                key = "uncalibrated_insufficient_history"
            bounds.append((lo, hi))
            supports.append(estimate["n"] if estimate else 0)
            groups.append(key)
        result[f"lower_{level}"] = [b[0] for b in bounds]
        result[f"upper_{level}"] = [b[1] for b in bounds]
        result[f"support_{level}"] = supports
        result[f"group_{level}"] = groups
    if 80 in bundle["levels"] and 90 in bundle["levels"]:
        result["lower_90"] = np.minimum(result.lower_90, result.lower_80)
        result["upper_90"] = np.maximum(result.upper_90, result.upper_80)
    return result


def score(frame, *, adjusted=False, levels=(80, 90)):
    valid = frame[frame.target_points.notna()]
    if valid.empty:
        return {"n": 0}
    y = valid.target_points.to_numpy(float)
    errors = valid.prediction.to_numpy(float) - y
    result = dict(
        n=len(valid),
        mae=float(np.abs(errors).mean()),
        rmse=float(np.sqrt(np.mean(errors**2))),
        bias=float(errors.mean()),
    )
    for level in levels:
        names = (f"lower_{level}", f"upper_{level}") if adjusted else LEVELS[level]
        lo, hi = (valid[name].to_numpy(float) for name in names)
        if not np.isfinite(np.column_stack([y, errors, lo, hi])).all():
            raise ValueError("Nonfinite scoring input")
        alpha = 1 - level / 100
        interval_score = hi - lo + 2 / alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
        result.update(
            {
                f"coverage_{level}": float(((lo <= y) & (y <= hi)).mean()),
                f"width_{level}": float((hi - lo).mean()),
                f"interval_score_{level}": float(interval_score.mean()),
                f"miss_below_{level}": float((y < lo).mean()),
                f"miss_above_{level}": float((y > hi).mean()),
            }
        )
    return result
