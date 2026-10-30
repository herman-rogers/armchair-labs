"""Separate participation, active workload, points and conserved teammate transfers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import INJURY_FEATURES, OPPORTUNITIES, TEAM_FEATURES, WORK_FEATURES

STAGES = ("no_injury", "injury", "team")
METHODS = (
    "v3",
    "v3_out_rule",
    *[f"{family}_{stage}" for family in ("direct", "mixture", "workload") for stage in STAGES],
    "redistribution_half",
    "redistribution_full",
    "mixture_team_blend",
    "workload_team_blend",
)


def matrix(frame, features):
    if any(
        c.startswith("label_") or c in ("target_points", "appeared", "actual_team")
        for c in features
    ):
        raise ValueError("Outcome in inference feature allowlist")
    return frame[features].replace([np.inf, -np.inf], np.nan).astype(float)


def tree(classifier=False):
    cls = LGBMClassifier if classifier else LGBMRegressor
    return cls(
        n_estimators=140,
        learning_rate=0.04,
        num_leaves=10,
        max_depth=4,
        min_child_samples=80,
        reg_lambda=25,
        n_jobs=2,
        random_state=20261013,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def regression(x, y):
    known = pd.Series(y, index=x.index).notna()
    if known.sum() < 100 or pd.Series(y, index=x.index)[known].nunique() == 1:
        return float(pd.Series(y, index=x.index)[known].mean()) if known.any() else 0.0
    return tree().fit(x.loc[known], pd.Series(y, index=x.index)[known])


def prediction(model, x, probability=False):
    if isinstance(model, float):
        return np.full(len(x), model)
    return model.predict_proba(x)[:, 1] if probability else model.predict(x)


def fit(train, year, raw_features):
    if train.empty or train.season.ge(year).any():
        raise ValueError("Training must precede forecast season")
    train = train[train.season.ge(year - 5) & train.target_points.notna()].copy()
    base = list(dict.fromkeys([*raw_features, *WORK_FEATURES, "base_prediction"]))
    feature_sets = {
        "no_injury": base,
        "injury": [*base, *INJURY_FEATURES],
        "team": [*base, *INJURY_FEATURES, *TEAM_FEATURES],
    }
    bundle = dict(
        year=year,
        position=str(train.position.iloc[0]),
        features=feature_sets,
        n=len(train),
        min_year=int(train.season.min()),
        max_year=int(train.season.max()),
        models={},
    )
    active = train.label_played.eq(1)
    known = train.label_played.notna()
    for stage, columns in feature_sets.items():
        x = matrix(train, columns)
        p = (
            tree(True).fit(x.loc[known], train.loc[known, "label_played"].astype(int))
            if train.loc[known, "label_played"].nunique() > 1
            else float(train.loc[known, "label_played"].mean())
        )
        bundle["models"][stage] = dict(
            direct=regression(x, train.target_points),
            probability=p,
            active_points=regression(x.loc[active], train.loc[active, "target_points"]),
            workload=[
                regression(x.loc[active], train.loc[active, f"label_{c}"]) for c in OPPORTUNITIES
            ],
        )
    # Linear conditional point response: plugging in E(workload|playing, X) is
    # coherent with this linear response, unlike plugging a mean into a nonlinear tree.
    context = ["base_prediction", "recent4_points_mean", "player_history_mean", *WORK_FEATURES]
    mapper_features = [*context, *[f"workload_{c}" for c in OPPORTUNITIES]]
    mapped = train.loc[active].copy()
    for c in OPPORTUNITIES:
        mapped[f"workload_{c}"] = mapped[f"label_{c}"]
    complete = mapped[[f"workload_{c}" for c in OPPORTUNITIES]].notna().all(axis=1)
    mapper = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=100.0),
    )
    mapper.fit(matrix(mapped.loc[complete], mapper_features), mapped.loc[complete, "target_points"])
    bundle.update(mapper=mapper, mapper_features=mapper_features)
    return bundle


def points_from_workload(bundle, frame, probability, active_workload):
    x = frame.copy()
    for j, c in enumerate(OPPORTUNITIES):
        x[f"workload_{c}"] = active_workload[:, j]
    return probability * bundle["mapper"].predict(matrix(x, bundle["mapper_features"]))


def predict(bundle, frame):
    if not frame.season.eq(bundle["year"]).all():
        raise ValueError("Wrong model year")
    base = frame.base_prediction.to_numpy(float)
    output = {"v3": base.copy(), "v3_out_rule": np.where(frame.report_out.fillna(0).eq(1), 0, base)}
    diagnostic = {}
    for stage, columns in bundle["features"].items():
        model, x = bundle["models"][stage], matrix(frame, columns)
        p = prediction(model["probability"], x, True)
        if stage != "no_injury":
            p = np.where(frame.report_out.fillna(0).eq(1), 0, p)
        active = prediction(model["active_points"], x)
        workload = np.maximum(np.column_stack([prediction(m, x) for m in model["workload"]]), 0)
        if bundle["position"] != "QB":
            workload[:, 0] = 0
        if bundle["position"] == "QB":
            workload[:, 2] = 0
        output[f"direct_{stage}"] = prediction(model["direct"], x)
        if stage != "no_injury":
            output[f"direct_{stage}"] = np.where(
                frame.report_out.fillna(0).eq(1), 0, output[f"direct_{stage}"]
            )
        output[f"mixture_{stage}"] = p * active
        output[f"workload_{stage}"] = points_from_workload(bundle, frame, p, workload)
        diagnostic[stage] = dict(probability=p, active_workload=workload)
    output["mixture_team_blend"] = (output["mixture_team"] + base) / 2
    output["workload_team_blend"] = (output["workload_team"] + base) / 2
    for name in ("mixture_team_blend", "workload_team_blend"):
        output[name] = np.where(frame.report_out.fillna(0).eq(1), 0, output[name])
    return output, diagnostic


def redistribute(frame, before, after, probability, fraction):
    """Transfer a fraction of a predicted team budget shortfall; never create volume.

    Budget is for the v3 eligible pool, not the entire NFL roster. Low-probability
    players receive no transfer. Work with unconditional counts, avoiding a
    second participation multiplier. Unassigned volume is retained in diagnostics.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("Redistribution fraction must be between zero and one")
    result = np.array(after, copy=True)
    receipts = []
    for key, indexes in (
        frame.reset_index(drop=True).groupby(["season", "week", "team"]).groups.items()
    ):
        idx = np.asarray(indexes)
        for j, opportunity in enumerate(OPPORTUNITIES):
            budget, used = float(before[idx, j].sum()), float(after[idx, j].sum())
            shortage = max(budget - used, 0.0)
            weights = before[idx, j] * probability[idx] * (probability[idx] >= 0.5)
            if weights.sum() > 0:
                result[idx, j] += fraction * shortage * weights / weights.sum()
            transferred = float(result[idx, j].sum() - used)
            if transferred > shortage + 1e-8 or transferred < -1e-8:
                raise ValueError("Redistribution created workload")
            receipts.append(
                dict(
                    season=int(key[0]),
                    week=int(key[1]),
                    team=str(key[2]),
                    opportunity=opportunity,
                    baseline_budget=budget,
                    pre_transfer=used,
                    transferred=transferred,
                    unassigned=shortage - transferred,
                )
            )
    return result, receipts


def choose(history, year):
    if history.season.ge(year).any():
        raise ValueError("Selection must precede forecast season")
    history = history[history.season.ge(year - 2)]
    choices = []
    for method, group in history.groupby("method"):
        values = (
            group.assign(squared=(group.prediction - group.target_points) ** 2)
            .groupby("season")
            .squared.mean()
        )
        choices.append((float(np.sqrt(values).mean()), method != "v3", str(method)))
    return min(choices)[2]
