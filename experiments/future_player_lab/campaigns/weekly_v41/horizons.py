"""Stage 4: v4's fixed-origin horizon/component method, retrained around v4.1 means.

Label construction, targets and tree settings come from `weekly_v4.horizons`.
The anchor is v4.1's own out-of-time stage-1 mean (never a saved v3 forecast) and
the current week is v4.1's selected stage-2 forecast. Added inputs are data, not
earlier models: dated availability status (the v2 availability archive), prior
snap/participation workload, origin-week injury status, and market context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ..weekly_v4 import horizons as v4

TARGETS, COMPONENTS, ORIGINS = v4.TARGETS, v4.COMPONENTS, v4.ORIGINS
EXTRA_FEATURES = (
    *[
        f"availability_{s}"
        for s in ("known", "age_days", "act", "res", "pup", "sus", "dev", "cut", "ina", "unknown")
    ],
    *[f"prior4_{c}" for c in ("snaps", "snap_share", "played", "attempts", "carries", "targets")],
    "report_known",
    "report_out",
    "report_doubtful",
    "report_questionable",
    "practice_dnp",
    "market_implied",
    "market_spread",
    "market_total",
)
FEATURES = (*v4.FEATURES, *EXTRA_FEATURES)


def feature_matrix(frame, features=FEATURES):
    return frame.reindex(columns=list(features)).astype(float).replace([np.inf, -np.inf], np.nan)


def tree():
    # v4's fixed horizon settings; no tuning on evaluation years.
    return LGBMRegressor(
        n_estimators=70,
        num_leaves=12,
        max_depth=5,
        min_child_samples=50,
        learning_rate=0.05,
        reg_lambda=10,
        verbosity=-1,
        n_jobs=4,
        random_state=704,
        deterministic=True,
        force_col_wise=True,
    )


def fit_models(frame, before_season, features=FEATURES):
    prior = frame[frame.season.lt(before_season)]
    if prior.empty:
        raise ValueError("Horizon fit requires earlier seasons")
    result = dict(
        training_through_season=int(prior.season.max()), features=list(features), positions={}
    )
    for position, targets in TARGETS.items():
        group = prior[prior.position.eq(position)]
        models = {}
        for target in ("points", *targets):
            rows = group[group[f"label_{target}"].notna()]
            if target == "points":
                rows = rows[rows.later_games.eq(1)]
            if len(rows) < 100:
                raise ValueError(f"Insufficient earlier labels for {position}/{target}")
            label = rows[f"label_{target}"].to_numpy()
            if target == "points":
                label = label - rows.base_prediction.to_numpy()
            models[target] = tree().fit(feature_matrix(rows, features), label)
        result["positions"][position] = models
    return result


def predictions(models, frame):
    result = pd.DataFrame(index=frame.index)
    features = models["features"]
    for position, group in frame.groupby("position"):
        x = feature_matrix(group, features)
        for target, model in models["positions"][position].items():
            values = model.predict(x)
            if target == "points":
                values += group.base_prediction.to_numpy()
            if target != "points" and not target.endswith("yards"):
                values = np.maximum(values, 0)
            result.loc[group.index, target] = values
    result["completions"] = np.minimum(result.completions, result.attempts)
    result["receptions"] = np.minimum(result.receptions, result.targets)
    if "report_out" in frame:
        absent = frame.report_out.eq(1) & frame.later_games.eq(0)
        for position, targets in TARGETS.items():
            result.loc[absent & frame.position.eq(position), list(targets)] = 0.0
    return result


def evaluate(frame, weekly, years=range(2018, 2026), log=print):
    """Out-of-time total forecasts per origin; v4.1 and fixed-origin persistence control."""
    records, components = [], []
    for year in years:
        fitted = fit_models(frame, year)
        test = frame[frame.season.eq(year)].copy()
        estimates = predictions(fitted, test)
        test["weekly_points"] = test.row_id.map(weekly).fillna(test.base_prediction)
        first = test[test.later_games.eq(0)].set_index("row_id")
        later = test[test.later_games.eq(1)]
        current = first.loc[later.row_id]
        actual = current.label_points.to_numpy() + later.label_points.to_numpy() * later.horizon_games
        common = dict(
            player_id=later.player_id.to_numpy(),
            player_name=later.player_name.to_numpy(),
            season=year,
            position=later.position.to_numpy(),
            origin_week=later.origin_week.astype(int).to_numpy(),
            remaining_games=later.remaining_games.astype(int).to_numpy(),
            target_points=actual,
            row_id=later.row_id.to_numpy(),
            reference_points=later.recent8_points_mean.fillna(later.prior_points_mean).to_numpy(),
        )
        weekly_now = current.weekly_points.to_numpy()
        for method, mean in (
            ("v41", estimates.loc[later.index, "points"].to_numpy()),
            ("v41_fixed_origin", later.base_prediction.to_numpy()),
        ):
            records.append(
                pd.DataFrame(
                    dict(common, method=method, prediction=weekly_now + mean * later.horizon_games)
                )
            )
        for target in COMPONENTS:
            valid = test.position.map(lambda p, t=target: t in TARGETS[p]) & test[
                f"label_{target}"
            ].notna()
            part = test[valid]
            components.append(
                pd.DataFrame(
                    dict(
                        season=year,
                        position=part.position.to_numpy(),
                        origin_week=part.origin_week.astype(int).to_numpy(),
                        horizon=np.where(part.later_games.eq(1), "later", "weekly"),
                        target=target,
                        prediction=estimates.loc[part.index, target].to_numpy()
                        * part.horizon_games.to_numpy(),
                        actual=part[f"label_{target}"].to_numpy() * part.horizon_games.to_numpy(),
                    )
                )
            )
        log(f"horizons {year}: {len(later):,} player/origin totals")
    return pd.concat(records, ignore_index=True), pd.concat(components, ignore_index=True)
