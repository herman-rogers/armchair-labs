"""V4 mean-preserving distributions selected using strictly earlier forecast errors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..weekly_v32 import uncertainty as base

SCALES = (0.85, 1.0, 1.15, 1.3)


def fit(history, before_season, scale=1.0):
    contract = base.fit(history, before_season)
    contract.update(version="v4_empirical_scale", scale=float(scale))
    return contract


def samples(frame, contract):
    draws = base.samples(frame, contract)
    mean = frame.prediction.to_numpy()[:, None]
    return mean + float(contract.get("scale", 1.0)) * (draws - mean)


def metrics(frame, draws):
    if not len(frame):
        return {"n": 0}
    result = base.metrics(frame, draws)
    actual = frame.target_points.to_numpy()
    result.update(
        mae=float(np.mean(abs(actual - frame.prediction))),
        bias=float(np.mean(frame.prediction - actual)),
    )
    if {"week", "position", "player_id"} <= set(frame):
        selected_points, ranks = [], []
        for (_, position), part in frame.groupby(["week", "position"]):
            count = 10 if position in ("QB", "TE") else 20
            ranked = part.sort_values(["prediction", "player_id"], ascending=[False, True])
            selected_points.append(float(ranked.head(count).target_points.sum()))
            a, b = part.prediction.rank(), part.target_points.rank()
            if a.std() > 0 and b.std() > 0:
                ranks.append(float(a.corr(b)))
        result["topk_actual_points_per_week"] = float(np.mean(selected_points))
        result["weekly_rank_correlation"] = float(np.mean(ranks)) if ranks else None
    for coverage in (80, 90):
        alpha = 1 - coverage / 100
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=1)
        result.update(
            {
                f"coverage{coverage}": float(np.mean((lo <= actual) & (actual <= hi))),
                f"width{coverage}": float(np.mean(hi - lo)),
                f"lower_miss{coverage}": float(np.mean(actual < lo)),
                f"upper_miss{coverage}": float(np.mean(actual > hi)),
                f"interval_score{coverage}": float(
                    np.mean(
                        hi
                        - lo
                        + 2 / alpha * (np.maximum(lo - actual, 0) + np.maximum(actual - hi, 0))
                    )
                ),
            }
        )
    return result


def _horizon(frame):
    column = "forecast_horizon" if "forecast_horizon" in frame else "horizon"
    if column not in frame:
        raise ValueError("An explicit early_week/pregame_60m horizon is required")
    return column


def apply(players, fitted_run):
    """Update distribution only; retain the v3.2 means and deterministic byes."""
    contracts = json.loads((Path(fitted_run) / "uncertainty.json").read_text())
    result = players.copy()
    horizon_column = _horizon(result)
    draws = np.zeros((len(result), 256))
    for horizon in result[horizon_column].unique():
        mask = result[horizon_column].eq(horizon).to_numpy()
        part = result.loc[mask]
        for position in part.position.unique():
            local = mask & result.position.eq(position).to_numpy()
            draws[local] = samples(result.loc[local], contracts[horizon][position])
    result = base.frame_with_samples(result, draws)
    result["distribution_method"] = "v4_prior_year_crps_selected_scale"
    return result


def evaluate(history):
    """2021 seed; 2022 selection forecasts; 2023–25 evaluation; 2026 final fits."""
    scores, decisions, predictions, final = [], [], [], {}
    for horizon, group in history.groupby("horizon"):
        final[horizon] = {}
        previous = []
        for year in range(2022, 2027):
            train = group[group.season.lt(year)]
            test = group[group.season.eq(year)].reset_index(drop=True)
            for position, prior in train.groupby("position"):
                older = [r for r in previous if r["position"] == position and r["season"] < year]
                if older:
                    by_method = pd.DataFrame(older).groupby("scale").crps.mean()
                    # Stable preference for unchanged baseline when scores tie.
                    scale = min(by_method.index, key=lambda s: (by_method[s], abs(s - 1)))
                else:
                    scale = 1.0
                decisions.append(
                    dict(
                        horizon=horizon,
                        position=position,
                        season=year,
                        scale=float(scale),
                        validation_years=sorted({r["season"] for r in older}),
                    )
                )
                contract = fit(prior, year, scale)
                if year == 2026:
                    final[horizon][position] = contract
                    continue
                part = test[test.position.eq(position)].reset_index(drop=True)
                candidates = {}
                for value in SCALES:
                    c = dict(contract, scale=value)
                    draws = samples(part, c)
                    candidates[value] = draws
                    previous.append(
                        dict(
                            position=position,
                            season=year,
                            scale=value,
                            crps=metrics(part, draws)["crps"],
                        )
                    )
                if year < 2023:
                    continue
                for method, value in (("v32", 1.0), ("v4", scale)):
                    draws = candidates[value]
                    out = base.frame_with_samples(part, draws)
                    out["distribution_method"] = method
                    predictions.append(out)
                    cohorts = {
                        "all": np.ones(len(part), dtype=bool),
                        "points_5_plus": part.base_prediction.ge(5).to_numpy(),
                        "points_below_5": part.base_prediction.lt(5).to_numpy(),
                        "reported": part.report_known.eq(1).to_numpy(),
                        "limited_history": part.player_history_count.lt(8).to_numpy(),
                    }
                    for cohort, mask in cohorts.items():
                        if mask.any():
                            scores.append(
                                dict(
                                    horizon=horizon,
                                    season=year,
                                    position=position,
                                    method=method,
                                    cohort=cohort,
                                    **metrics(part.loc[mask], draws[mask]),
                                )
                            )
    return pd.DataFrame(scores), pd.concat(predictions, ignore_index=True), decisions, final


def fit_total(history, before_season):
    """Calibrate total errors directly, preserving observed across-week dependence.

    The input must be out-of-time *fixed-origin total* forecasts. An origin bucket
    uses only that same origin; sparse scoring bins fall back to its position pool.
    This never treats weekly forecast errors as independent.
    """
    if history.empty or history.season.ge(before_season).any():
        raise ValueError("Total calibration requires earlier seasons")
    result = dict(
        before_season=int(before_season),
        origins={},
        method="centered_fixed_origin_total_errors",
        minimum_support=100,
    )
    for origin, group in history.groupby("origin_week"):
        result["origins"][str(int(origin))] = fit(group, before_season)
    return result


def total_samples(frame, contract):
    # A current-week absence/bye cannot make a multi-week total deterministic.
    frame = frame.drop(columns=["report_out", "is_bye"], errors="ignore")
    output = np.empty((len(frame), 256))
    for origin in frame.origin_week.unique():
        mask = frame.origin_week.eq(origin).to_numpy()
        key = str(int(origin))
        if key not in contract["origins"]:
            raise ValueError(f"No evaluated total calibration for origin {origin}")
        output[mask] = samples(frame.loc[mask], contract["origins"][key])
    return output


def evaluate_totals(history, evaluation_years=(2023, 2024, 2025)):
    """Compare calibrated whole-total forecasts using each method's earlier errors.

    A v32 bridge with no published ranges is an explicitly newly calibrated
    control here, not a claim that the original served bridge had distributions.
    All methods must have identical player/origin/year keys.
    """
    keys = ["season", "origin_week", "position", "player_id"]
    populations = [set(map(tuple, g[keys].to_numpy())) for _, g in history.groupby("method")]
    if any(p != populations[0] for p in populations):
        raise ValueError("Total uncertainty comparison requires matched populations")
    rows = []
    for method, group in history.groupby("method"):
        for year in evaluation_years:
            train = group[group.season.lt(year)]
            test = group[group.season.eq(year)].reset_index(drop=True)
            contract = fit_total(train, year)
            draws = total_samples(test, contract)
            for (origin, position), indices in test.groupby(
                ["origin_week", "position"]
            ).indices.items():
                rows.append(
                    dict(
                        method=method,
                        season=int(year),
                        origin_week=int(origin),
                        position=position,
                        **metrics(test.iloc[indices], draws[indices]),
                    )
                )
    return pd.DataFrame(rows)
