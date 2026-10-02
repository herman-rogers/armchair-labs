"""Paired chronological evaluation and strictly earlier OOF calibration."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def calibrated_intervals(history, row, config):
    year = row["season"]
    earlier = [
        r
        for r in history
        if year - config["calibration_window_seasons"] <= r["season"] < year
        and r["actual"] is not None
    ]
    seasons = sorted({r["season"] for r in earlier})
    result = {
        "calibration_n": len(earlier),
        "calibration_seasons": seasons,
        "median_prediction": None,
    }
    for method in ("absolute", "scaled"):
        for level in config["interval_levels"]:
            for side in ("lower", "upper"):
                result[f"{method}_{int(level * 100)}_{side}"] = None
    if len(earlier) < config["calibration_min_rows"] or (
        len(seasons) < config["calibration_min_seasons"]
    ):
        return result
    residuals = np.array([r["actual"] - r["prediction"] for r in earlier])
    scales = np.array([r["scale"] for r in earlier])
    result["median_prediction"] = float(row["prediction"] + np.median(residuals))
    if row["target"] in ("attempts", "carries", "targets"):
        result["median_prediction"] = max(0.0, result["median_prediction"])
    for method in ("absolute", "scaled"):
        scores = np.abs(residuals) / (scales if method == "scaled" else 1)
        for level in config["interval_levels"]:
            # Finite-sample corrected order statistic; temporal dependence means
            # there is NO claim of conformal/exchangeable coverage guarantees.
            index = min(len(scores), int(np.ceil((len(scores) + 1) * level))) - 1
            radius = float(np.sort(scores)[index]) * (row["scale"] if method == "scaled" else 1)
            lower, upper = row["prediction"] - radius, row["prediction"] + radius
            if row["target"] in ("attempts", "carries", "targets"):
                lower = max(0.0, lower)
            if row["known_available_games_cap"] == 0:
                lower = upper = 0.0
                result["median_prediction"] = 0.0
            result[f"{method}_{int(level * 100)}_lower"] = lower
            result[f"{method}_{int(level * 100)}_upper"] = upper
    return result


def paired_summary(rows, config):
    annual = []
    for year in sorted({r["season"] for r in rows}):
        group = [r for r in rows if r["season"] == year]
        actual = np.array([r["actual"] for r in group])
        pred = np.array([r["prediction"] for r in group])
        control = np.array([r["control_prediction"] for r in group])
        annual.append(
            {
                "season": year,
                "n": len(group),
                "mae": float(np.mean(np.abs(pred - actual))),
                "control_mae": float(np.mean(np.abs(control - actual))),
                "mse": float(np.mean((pred - actual) ** 2)),
                "control_mse": float(np.mean((control - actual) ** 2)),
            }
        )
    result = {"n": len(rows), "years": len(annual), "annual": annual}
    for loss in ("mae", "mse"):
        values = np.array([a[f"control_{loss}"] - a[loss] for a in annual])
        control = np.mean([a[f"control_{loss}"] for a in annual])
        rng = np.random.default_rng(config["seed"])
        draws = rng.choice(values, (config["bootstrap_draws"], len(values))).mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        result.update(
            {
                loss: float(np.mean([a[loss] for a in annual])),
                f"control_{loss}": float(control),
                f"{loss}_gain": float(values.mean()),
                f"{loss}_gain_pct": float(100 * values.mean() / control) if control else None,
                f"{loss}_ci_low": float(low),
                f"{loss}_ci_high": float(high),
                f"{loss}_positive_years": int((values > 0).sum()),
                f"{loss}_leave_one_year_out_min": float(
                    min(np.delete(values, i).mean() for i in range(len(values)))
                )
                if len(values) > 1
                else None,
            }
        )
    return result


def summarize(predictions, config):
    groups = defaultdict(list)
    for row in predictions:
        if row["actual"] is None or row["prediction"] is None or row["control_prediction"] is None:
            continue
        slices = ["all", row["population"]]
        if row["season"] >= config["modern_start"]:
            slices.append("modern")
        if row["market_observed"]:
            slices.append("market_covered")
        if row["proxy_half_season"]:
            slices.append("participation_covered")
        if row["prior_exposure"] is not None and 0 < row["prior_exposure"] < 50:
            slices.append("prior_exposure_1_to_49")
        if row["position"] == "QB" and row["prior_role_games"] < 8:
            slices.append("qb_prior_role_under_8_games")
        for name in slices:
            groups[row["position"], row["target"], row["model"], row["control"], name].append(row)
    summaries = []
    for (position, target, model, control, name), rows in sorted(groups.items()):
        summaries.append(
            dict(
                position=position,
                target=target,
                model=model,
                control=control,
                slice=name,
                **paired_summary(rows, config),
            )
        )
    return summaries


def interval_summary(predictions, config):
    groups = defaultdict(list)
    for row in predictions:
        if row["actual"] is None:
            continue
        for method in ("absolute", "scaled"):
            for level in config["interval_levels"]:
                lower = row[f"{method}_{int(level * 100)}_lower"]
                upper = row[f"{method}_{int(level * 100)}_upper"]
                if lower is None:
                    continue
                actual = row["actual"]
                width = upper - lower
                score = width + 2 / (1 - level) * (max(lower - actual, 0) + max(actual - upper, 0))
                for name in ("all", row["population"]):
                    key = row["position"], row["target"], row["model"], name, method, level
                    groups[key].append(
                        {
                            "season": row["season"],
                            "covered": float(lower <= actual <= upper),
                            "width": width,
                            "interval_score": score,
                            "mean_mae": abs(row["prediction"] - actual),
                            "median_mae": abs(row["median_prediction"] - actual),
                        }
                    )
    summaries = []
    for (position, target, model, name, method, level), rows in sorted(groups.items()):
        annual = []
        for year in sorted({r["season"] for r in rows}):
            group = [r for r in rows if r["season"] == year]
            annual.append(
                dict(
                    season=year,
                    n=len(group),
                    **{
                        field: float(np.mean([r[field] for r in group]))
                        for field in (
                            "covered",
                            "width",
                            "interval_score",
                            "mean_mae",
                            "median_mae",
                        )
                    },
                )
            )
        summaries.append(
            dict(
                position=position,
                target=target,
                model=model,
                slice=name,
                method=method,
                level=level,
                n=len(rows),
                years=len(annual),
                annual=annual,
                **{
                    field: float(np.mean([a[field] for a in annual]))
                    for field in ("covered", "width", "interval_score", "mean_mae", "median_mae")
                },
            )
        )
    return summaries
