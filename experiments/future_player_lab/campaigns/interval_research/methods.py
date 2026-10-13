"""Bounded past-only interval experiments, with fixed means and distributions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..weekly_v31 import calibration as v31

FIXED = (
    "v3",
    "v31",
    "half_v31",
    "scaled_signed",
    "additive_signed",
    "tail_scaled",
    "tail_additive",
    "tail_scaled_recent",
    "tail_scaled_fine",
)
SELECTORS = ("selective", "selective_guarded")


def group_name(prediction, fine=False):
    if fine and prediction < 1:
        return "under_1"
    if fine and prediction < 5:
        return "1_to_5"
    return v31.workload(prediction)


def order_stat(values, probability, *, lower=False):
    """Conservative empirical rank for each endpoint, not a time-series guarantee."""
    n = len(values)
    value = (n + 1) * probability
    # Arithmetic such as (1 - .8) / 2 must not move an exact integer rank by one.
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-10):
        value = round(value)
    rank = math.floor(value) if lower else math.ceil(value)
    if not 1 <= rank <= n:
        raise ValueError("Insufficient calibration rows for the requested tail")
    return float(np.partition(values, rank - 1)[rank - 1])


def fit(past, year, method, *, levels=(80, 90), min_rows=200, lineup=False):
    if method not in FIXED:
        raise ValueError("Unknown fixed interval method")
    if not past.empty and (past.season >= year).any():
        raise ValueError("Calibration must precede the forecast season")
    if "phase" in past and past.phase.isin(["train", "training", "in_sample"]).any():
        raise ValueError("Use out-of-time forecasts")
    if method in ("v31", "half_v31"):
        return dict(
            method=method,
            v31=v31.fit(
                past,
                year,
                policy="position" if lineup else "position_workload",
                levels=levels,
                min_rows=min_rows,
            ),
            before_season=year,
            levels=list(levels),
        )
    window = 2 if method == "tail_scaled_recent" else 5
    cal = past[past.season.ge(year - window) & past.target_points.notna()].copy()
    cal["workload"] = cal.prediction.map(lambda x: group_name(x, method == "tail_scaled_fine"))
    bundle = dict(
        method=method,
        before_season=year,
        levels=list(levels),
        groups={},
        min_rows=min_rows,
        window=window,
        n=len(cal),
        max_season=int(cal.season.max()) if len(cal) else None,
    )
    if method == "v3":
        return bundle
    for position, part in cal.groupby("position"):
        groups = [(position, part)]
        if not lineup:
            groups += [(f"{position}/{key}", group) for key, group in part.groupby("workload")]
        for key, group in groups:
            if len(group) < min_rows:
                continue
            bundle["groups"][key] = {}
            for level in levels:
                lo, hi = (group[c].to_numpy(float) for c in v31.LEVELS[level])
                y = group.target_points.to_numpy(float)
                if not np.isfinite(np.c_[lo, hi, y]).all() or (lo > hi).any():
                    raise ValueError("Invalid calibration values")
                scaled = "scaled" in method
                scale = np.maximum(hi - lo, 1.0) if scaled else np.ones(len(group))
                if method.startswith("tail_"):
                    alpha = 1 - level / 100
                    low_delta = order_stat((y - lo) / scale, alpha / 2, lower=True)
                    high_delta = order_stat((y - hi) / scale, 1 - alpha / 2)
                else:
                    correction = order_stat(np.maximum(lo - y, y - hi) / scale, level / 100)
                    low_delta, high_delta = -correction, correction
                bundle["groups"][key][str(level)] = dict(
                    low_delta=low_delta, high_delta=high_delta, scaled=scaled, n=len(group)
                )
    return bundle


def apply(frame, bundle):
    if not frame.season.eq(bundle["before_season"]).all():
        raise ValueError("Calibration season does not match forecast rows")
    method = bundle["method"]
    if method in ("v31", "half_v31"):
        result = v31.apply(frame, bundle["v31"])
        if method == "half_v31":
            for level in bundle["levels"]:
                lo, hi = v31.LEVELS[level]
                result[f"lower_{level}"] = (result[f"lower_{level}"] + result[lo]) / 2
                result[f"upper_{level}"] = (result[f"upper_{level}"] + result[hi]) / 2
        return result
    result = frame.copy()
    for level in bundle["levels"]:
        lowers, uppers, supports, used = [], [], [], []
        for row in frame.to_dict("records"):
            lo, hi = (float(row[c]) for c in v31.LEVELS[level])
            if not np.isfinite([lo, hi, row["prediction"]]).all() or lo > hi:
                raise ValueError("Invalid forecast bounds")
            key = f"{row['position']}/{group_name(row['prediction'], method == 'tail_scaled_fine')}"
            estimate = bundle["groups"].get(key, {}).get(str(level))
            if estimate is None:
                key = row["position"]
                estimate = bundle["groups"].get(key, {}).get(str(level))
            if estimate:
                scale = max(hi - lo, 1.0) if estimate["scaled"] else 1.0
                lo, hi = lo + estimate["low_delta"] * scale, hi + estimate["high_delta"] * scale
                # A crossing collapses to its midpoint instead of swapping the two tails.
                if lo > hi:
                    lo = hi = (lo + hi) / 2
            if row.get("is_bye", 0) == 1:
                lo = hi = 0.0
            lowers.append(lo)
            uppers.append(hi)
            supports.append(estimate["n"] if estimate else 0)
            used.append(key if estimate else "baseline_fallback")
        result[f"lower_{level}"] = lowers
        result[f"upper_{level}"] = uppers
        result[f"support_{level}"] = supports
        result[f"group_{level}"] = used
    if 90 in bundle["levels"] and 80 in bundle["levels"]:
        result["lower_90"] = np.minimum(result.lower_90, result.lower_80)
        result["upper_90"] = np.maximum(result.upper_90, result.upper_80)
    return result


def components(frame, level):
    alpha = 1 - level / 100
    y = frame.target_points.to_numpy(float)
    lo, hi = frame[f"lower_{level}"].to_numpy(float), frame[f"upper_{level}"].to_numpy(float)
    width = hi - lo
    low_penalty = 2 / alpha * np.maximum(lo - y, 0)
    high_penalty = 2 / alpha * np.maximum(y - hi, 0)
    return dict(
        width=width,
        below_penalty=low_penalty,
        above_penalty=high_penalty,
        interval_score=width + low_penalty + high_penalty,
        covered=(lo <= y) & (y <= hi),
        miss_below=y < lo,
        miss_above=y > hi,
    )


def select(validation, year, *, levels=(80, 90), guarded=False, min_rows=100):
    """Choose separately by cohort using two earlier years of chronological validation."""
    if not validation.empty and validation.season.ge(year).any():
        raise ValueError("Selection cannot use forecast-season outcomes")
    val = validation[validation.season.ge(year - 2) & validation.target_points.notna()].copy()
    val["cohort"] = val.prediction.map(v31.workload)
    choices = {}
    for position, frame in val.groupby("position"):
        groups = [(position, frame)]
        if position != "LINEUP":
            groups += [(f"{position}/{name}", part) for name, part in frame.groupby("cohort")]
        for key, part in groups:
            candidates = []
            for method, rows in part.groupby("method"):
                if len(rows) < min_rows:
                    continue
                annual = []
                coverage = {level: [] for level in levels}
                for _, group in rows.groupby("season"):
                    metrics = {level: components(group, level) for level in levels}
                    weights = np.array([1 - level / 100 for level in levels])
                    annual.append(
                        float(
                            np.average(
                                [metrics[level]["interval_score"].mean() for level in levels],
                                weights=weights,
                            )
                        )
                    )
                    for level in levels:
                        coverage[level].append(float(metrics[level]["covered"].mean()))
                admissible = all(np.mean(coverage[level]) >= level / 100 - 0.02 for level in levels)
                if not guarded or admissible:
                    candidates.append((float(np.mean(annual)), method, len(rows)))
            if candidates:
                value, method, n = min(candidates)
                choices[key] = dict(method=method, validation_score=value, n=n)
            else:
                choices[key] = dict(method="v31", reason="No supported admissible candidate", n=0)
    return dict(
        before_season=year,
        max_validation_season=int(val.season.max()) if len(val) else None,
        guarded=guarded,
        choices=choices,
    )


def apply_selection(candidates, policy):
    if not candidates.season.eq(policy["before_season"]).all():
        raise ValueError("Policy season mismatch")
    base = candidates[candidates.method.eq("v31")]
    rows = []
    indexed = candidates.set_index(["row_id", "method"])
    if not indexed.index.is_unique:
        raise ValueError("Duplicate candidate identities")
    for row in base.to_dict("records"):
        key = f"{row['position']}/{v31.workload(row['prediction'])}"
        decision = policy["choices"].get(
            key, policy["choices"].get(row["position"], {"method": "v31"})
        )
        chosen = indexed.loc[(row["row_id"], decision["method"])].to_dict()
        chosen.update(
            row_id=row["row_id"],
            chosen_method=decision["method"],
            method="selective_guarded" if policy["guarded"] else "selective",
        )
        rows.append(chosen)
    return pd.DataFrame(rows)
