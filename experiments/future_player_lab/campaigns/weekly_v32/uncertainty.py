"""Mean-preserving empirical error distributions and chronological validation.

This is a v3.2 distribution contract, independent of the paused v3.1 experiment.
Only earlier out-of-time point errors enter the empirical distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..weekly_v3.lineups import aggregate, estimate_correlations
from ..weekly_v3.modeling import QCOLS, QUANTILES

GRID = (np.arange(256) + 0.5) / 256
BINS = (2.0, 5.0, 10.0, 20.0)


def fit(frame, before_season):
    if frame.empty or frame.season.ge(before_season).any():
        raise ValueError("Uncertainty requires strictly earlier out-of-time errors")
    result = dict(before_season=int(before_season), bins=list(BINS), cells={})
    for position, part in frame.groupby("position"):
        groups = np.digitize(part.prediction, BINS)
        for bucket in range(len(BINS) + 1):
            local = part[groups == bucket]
            if len(local) < 100:
                local = part
            error = (local.target_points - local.prediction).dropna().to_numpy()
            if len(error) < 100:
                raise ValueError("Insufficient prior error support")
            samples = np.quantile(error, GRID)
            # The point forecast remains the declared mean in the API and joint draws.
            samples -= samples.mean()
            result["cells"][f"{position}:{bucket}"] = dict(n=len(error), errors=samples.tolist())
    return result


def samples(frame, contract):
    if not frame.season.ge(contract["before_season"]).all():
        raise ValueError("Uncertainty calibration contains future outcomes")
    output = []
    for row in frame.to_dict("records"):
        bucket = np.digitize(row["prediction"], contract["bins"])
        values = np.asarray(contract["cells"][f"{row['position']}:{bucket}"]["errors"])
        draw = values + row["prediction"]
        if row.get("report_out") == 1 or row.get("is_bye") == 1:
            draw = np.zeros(256)
            if abs(row["prediction"]) > 1e-10:
                raise ValueError("Deterministic absence must have zero points")
        output.append(draw)
    return np.asarray(output)


def frame_with_samples(frame, draws):
    out = frame.copy()
    out["samples"] = draws.tolist()
    out["std"] = draws.std(axis=1)
    out[QCOLS] = np.quantile(draws, QUANTILES, axis=1).T
    return out


def baseline_samples(frame):
    draws = np.asarray([np.interp(GRID, QUANTILES, q) for q in frame[QCOLS].to_numpy()])
    # Exactly the centering used by v3's existing lineup runtime.
    return draws + frame.prediction.to_numpy()[:, None] - draws.mean(axis=1)[:, None]


def metrics(frame, draws):
    actual = frame.target_points.to_numpy()
    lo, hi = np.quantile(draws, [0.1, 0.9], axis=1)
    n = draws.shape[1]
    crps = (
        np.abs(draws - actual[:, None]).mean(axis=1)
        - (np.sort(draws) * (2 * np.arange(n) - n + 1)).sum(axis=1) / n**2
    )
    return dict(
        n=len(frame),
        rmse=float(np.sqrt(np.mean((actual - frame.prediction) ** 2))),
        coverage80=float(np.mean((lo <= actual) & (actual <= hi))),
        width80=float(np.mean(hi - lo)),
        interval_score80=float(
            np.mean(hi - lo + 10 * np.maximum(lo - actual, 0) + 10 * np.maximum(actual - hi, 0))
        ),
        crps=float(crps.mean()),
    )


def dependence(frame, before_season):
    values = estimate_correlations(frame.to_dict("records"), before_season)
    return dict(
        covariance_before_season=before_season,
        correlations=[
            dict(relationship=k[0], position_a=k[1], position_b=k[2], **v)
            for k, v in values.items()
        ],
        special_teams={},
        note="V3.2 earlier out-of-time point residuals; K/DST uncertainty withheld",
    )


def estimates(contract):
    return {
        (r["relationship"], *sorted((r["position_a"], r["position_b"]))): r
        for r in contract["correlations"]
    }


def draft(frame, rank_column):
    """Same forecast-only eight-player offense shape as the v3 study."""
    pools = {
        p: part.sort_values([rank_column, "player_id"], ascending=[False, True]).player_id.tolist()
        for p, part in frame.groupby("position")
    }
    rosters = [[] for _ in range(10)]
    for position, count in (("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1)):
        for _ in range(count):
            for roster in rosters:
                if pools.get(position):
                    roster.append(pools[position].pop(0))
    remaining = sum([pools.get(p, []) for p in ("RB", "WR", "TE")], [])
    flex = (
        frame[frame.player_id.isin(remaining)]
        .sort_values([rank_column, "player_id"], ascending=[False, True])
        .player_id.tolist()
    )
    for roster in rosters:
        if flex:
            roster.append(flex.pop(0))
    return [r for r in rosters if len(r) == 8]


def evaluate(selected, baseline, panel):
    """2021–22 seed calibration; evaluate 2023–25 with only earlier errors."""
    player_scores, lineup_rows = [], []
    final, final_dependence = {}, {}
    context = panel.drop_duplicates("row_id").set_index("row_id")
    baseline = baseline.set_index("row_id")
    for horizon, group in selected.groupby("horizon"):
        group = group.copy()
        group["opponent"] = group.row_id.map(context.opponent)
        for year in (2023, 2024, 2025, 2026):
            history = group[group.season.lt(year)]
            contract = fit(history, year)
            dep = dependence(history, year)
            if year == 2026:
                final[horizon], final_dependence[horizon] = contract, dep
                continue
            test = group[group.season.eq(year)].reset_index(drop=True)
            old = baseline.loc[test.row_id].reset_index()
            old["base_prediction"] = old.prediction
            new = frame_with_samples(test, samples(test, contract))
            old = frame_with_samples(old, baseline_samples(old))
            for name, frame in (("v3", old), ("v3.2", new)):
                for position, part in frame.groupby("position"):
                    for cohort, subset in (
                        ("all", part),
                        ("base_points_5_plus", part[part.base_prediction.ge(5)]),
                    ):
                        player_scores.append(
                            dict(
                                horizon=horizon,
                                season=year,
                                position=position,
                                method=name,
                                cohort=cohort,
                                **metrics(subset, np.stack(subset.samples)),
                            )
                        )
            old_dep = dependence(baseline.loc[history.row_id].reset_index(), year)
            for week, part in new.groupby("week"):
                maps = {
                    "v3": old[old.week.eq(week)].set_index("player_id").to_dict("index"),
                    "v3.2": part.set_index("player_id").to_dict("index"),
                }
                for design, rank in (
                    ("v3_ranked", "base_prediction"),
                    ("v32_ranked", "prediction"),
                ):
                    for i, ids in enumerate(draft(part, rank)):
                        for name, dependence_contract in (("v3", old_dep), ("v3.2", dep)):
                            roster = [dict(player_id=p, **maps[name][p]) for p in ids]
                            actual = sum(r["target_points"] for r in roster)
                            value = aggregate(
                                roster,
                                estimates(dependence_contract),
                                seed=year * 10000 + int(week) * 10 + i,
                                actual=actual,
                            )
                            lineup_rows.append(
                                dict(
                                    horizon=horizon,
                                    season=year,
                                    week=int(week),
                                    design=design,
                                    lineup=i,
                                    method=name,
                                    target_points=actual,
                                    player_ids=",".join(ids),
                                    **value,
                                )
                            )
            print(f"Uncertainty/lineups: {horizon} {year}", flush=True)
    return pd.DataFrame(player_scores), pd.DataFrame(lineup_rows), final, final_dependence
