"""How v4.1 judges a change: relevant cohort, paired uncertainty, noise floor.

Pooled RMSE over every eligible player is dominated by fringe rows with tiny
errors and by irreducible touchdown/efficiency noise, so small method differences
look like progress or regression by chance. Each comparison therefore reports:

* `all` and `relevant` cohorts. Relevance is model-independent and past-only
  (`reference_points` >= 5, the player's recent mean), so no method can move rows
  between cohorts.
* Paired 95% intervals for the % RMSE change, resampling whole players (their
  rows are correlated across weeks/origins), plus seasons improved.
* For weekly points, an opportunity-oracle floor (a model given the *actual*
  same-week attempts/carries/targets/snaps) and the share of the reducible gap
  between the control and that floor which the challenger closes.

A change is called an improvement only when the relevant-cohort interval excludes
zero; otherwise it is reported as not distinguishable from noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

RELEVANT_POINTS = 5.0
DRAWS = 2000
SEED = 20261027


def _cohorts(frame):
    yield "all", frame
    yield "relevant", frame[frame.reference_points.fillna(0).ge(RELEVANT_POINTS)]


def paired(frame, challenger, control, group=("position",), cluster="player_id"):
    """Compare two methods on exactly matched rows (inner join on the row key)."""
    keys = [c for c in ("row_id", "player_id", "season", "week", "origin_week") if c in frame]
    keys = list(dict.fromkeys(keys))
    # Duplicate keys (e.g. repeated control rows) are dropped, never multiplied.
    a = frame[frame.method.eq(challenger)].drop_duplicates(keys)
    b = frame[frame.method.eq(control)][[*keys, "prediction"]].drop_duplicates(keys)
    joined = a.merge(b, on=keys, suffixes=("", "_control"), validate="one_to_one")
    joined = joined[joined.target_points.notna()]
    rng = np.random.default_rng(SEED)
    rows = []
    for labels, part in joined.groupby(list(group)):
        labels = labels if isinstance(labels, tuple) else (labels,)
        for cohort, sub in _cohorts(part):
            if len(sub) < 30:
                continue
            e1 = (sub.prediction - sub.target_points).to_numpy() ** 2
            e0 = (sub.prediction_control - sub.target_points).to_numpy() ** 2
            r1, r0 = np.sqrt(e1.mean()), np.sqrt(e0.mean())
            # Player-cluster bootstrap of the RMSE ratio.
            codes, uniques = pd.factorize(sub[cluster])
            s1 = np.bincount(codes, e1, len(uniques))
            s0 = np.bincount(codes, e0, len(uniques))
            n = np.bincount(codes, minlength=len(uniques)).astype(float)
            pick = rng.integers(0, len(uniques), (DRAWS, len(uniques)))
            w = np.zeros((DRAWS, len(uniques)))
            np.add.at(w, (np.arange(DRAWS)[:, None], pick), 1.0)
            ratio = np.sqrt((w @ s1) / (w @ n)) / np.sqrt((w @ s0) / (w @ n))
            lo, hi = np.quantile(100 * (ratio - 1), [0.025, 0.975])
            yearly = (
                sub.assign(e1=e1, e0=e0)
                .groupby("season")[["e1", "e0"]]
                .mean()
                .pipe(lambda t: (t.e1 < t.e0).sum())
            )
            rows.append(
                dict(
                    zip(group, labels, strict=True),
                    cohort=cohort,
                    challenger=challenger,
                    control=control,
                    n=len(sub),
                    players=len(uniques),
                    rmse_challenger=float(r1),
                    rmse_control=float(r0),
                    pct_change=float(100 * (r1 / r0 - 1)),
                    ci95_low=float(lo),
                    ci95_high=float(hi),
                    seasons_improved=int(yearly),
                    seasons=int(sub.season.nunique()),
                    verdict="improved"
                    if hi < 0
                    else "worse"
                    if lo > 0
                    else "not distinguishable from noise",
                )
            )
    return pd.DataFrame(rows)


def oracle_floor(panel, train_years=range(2010, 2021), test_years=range(2021, 2026)):
    """Weekly RMSE given the actual same-week opportunity. Not a forecast; a bound."""
    columns = ["label_attempts", "label_carries", "label_targets", "label_snaps"]
    rows, predictions = [], []
    for position, local in panel[panel.target_points.notna()].groupby("position"):
        train = local[local.season.isin(train_years)]
        test = local[local.season.isin(test_years)]
        model = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31, verbosity=-1, n_jobs=4,
            random_state=SEED,
        ).fit(train[columns].astype(float), train.target_points)
        value = model.predict(test[columns].astype(float))
        predictions.append(pd.DataFrame(dict(row_id=test.row_id.to_numpy(), oracle=value)))
        for cohort, sub in _cohorts(test.assign(oracle=value)):
            rows.append(
                dict(
                    position=position,
                    cohort=cohort,
                    n=len(sub),
                    oracle_rmse=float(np.sqrt(np.mean((sub.oracle - sub.target_points) ** 2))),
                    outcome_sd=float(sub.target_points.std()),
                )
            )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def gap_closed(comparison, floor):
    """Share of (control RMSE - oracle floor) removed by the challenger."""
    merged = comparison.merge(floor[["position", "cohort", "oracle_rmse"]], on=["position", "cohort"])
    gap = merged.rmse_control - merged.oracle_rmse
    merged["reducible_gap_closed"] = (merged.rmse_control - merged.rmse_challenger) / gap
    return merged
