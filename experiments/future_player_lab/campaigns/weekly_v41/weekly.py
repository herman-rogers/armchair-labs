"""Stage 1: v3's weekly mean method, retrained on the v4.1 panel.

Reuses v3's estimators and fixed tree settings (`weekly_v3.modeling`). Candidates
are fitted on strictly earlier seasons; one method per position is selected on
2018-20 equal-year RMSE and then held fixed, exactly as in v3. A no-market arm is
an ablation for evaluation only and is never eligible for selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..weekly_v3 import modeling as v3_modeling
from .data import MARKET_FEATURES

CANDIDATES = ("recent", "lgb_direct_full", "lgb_direct_decay", "lgb_direct_no_variability")
ABLATIONS = {"ablation_no_market": "lgb_direct_full"}
SELECTION_YEARS = (2018, 2019, 2020)
FIRST_YEAR = 2014  # stage-2 fits use five earlier seasons of base predictions from 2019


def _features(features, name):
    if name in ABLATIONS:
        return [f for f in features if f not in MARKET_FEATURES]
    return list(features)


def fit(train, features, name, year):
    method = ABLATIONS.get(name, name)
    return dict(
        name=name, bundle=v3_modeling.fit_mean(train, _features(features, name), method, year)
    )


def predict(model, frame):
    return v3_modeling.predict_mean(model["bundle"], frame)


def out_of_time(panel, features, years=range(FIRST_YEAR, 2027), log=print):
    """Every candidate's forecast for each season from fits on earlier seasons only."""
    rows, final = [], {}
    names = (*CANDIDATES, *ABLATIONS)
    for position, local in panel.groupby("position"):
        for year in years:
            train = local[local.season.lt(year) & local.target_points.notna()]
            test = local[local.season.eq(year)]
            if test.empty:
                continue
            for name in names:
                model = fit(train, features, name, year)
                if year == 2026:
                    final[(position, name)] = model
                rows.append(
                    pd.DataFrame(
                        dict(
                            row_id=test.row_id.to_numpy(),
                            position=position,
                            season=year,
                            method=name,
                            prediction=predict(model, test),
                        )
                    )
                )
            log(f"stage1 {position} {year}: {len(train):,} train / {len(test):,} rows")
    return pd.concat(rows, ignore_index=True), final


def select(predictions, labels):
    """Equal-year RMSE on 2018-20 per position, among eligible candidates only."""
    frame = predictions[
        predictions.season.isin(SELECTION_YEARS) & predictions.method.isin(CANDIDATES)
    ].merge(labels, on="row_id", validate="many_to_one")
    frame = frame[frame.target_points.notna()]
    choice = {}
    for position, part in frame.groupby("position"):
        scores = (
            part.assign(se=(part.prediction - part.target_points) ** 2)
            .groupby(["method", "season"])
            .se.mean()
            .pow(0.5)
            .groupby("method")
            .mean()
        )
        choice[position] = dict(
            mean_method=str(scores.idxmin()),
            selection_rmse={k: float(v) for k, v in scores.items()},
            selection_years=list(SELECTION_YEARS),
        )
    return choice


def base_predictions(predictions, choice):
    parts = [
        predictions[predictions.position.eq(p) & predictions.method.eq(c["mean_method"])]
        for p, c in choice.items()
    ]
    base = pd.concat(parts, ignore_index=True)
    if base.row_id.duplicated().any():
        raise ValueError("Duplicate base predictions")
    if not np.isfinite(base.prediction).all():
        raise ValueError("Nonfinite base predictions")
    return base.set_index("row_id").prediction
