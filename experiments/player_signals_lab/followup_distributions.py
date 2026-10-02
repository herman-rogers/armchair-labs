"""Earlier-OOF joint opportunity/execution simulation; no fitted residual calibration."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl
from followup_calibration import interval_score
from followup_common import columns, holm, point_summaries, predict, record, sign_test
from safety import write_json

DRAWS = 512


def crps(draws, actual):
    """Exact CRPS of the empirical distribution, including its discrete point masses."""
    values = np.sort(np.asarray(draws, float))
    n = len(values)
    coefficients = 2 * np.arange(1, n + 1) - n - 1
    return float(np.abs(values - actual).mean() - np.dot(coefficients, values) / n**2)


def earlier_history(history, year):
    return [r for r in history if year - 5 <= r["season"] < year]


def joint_draws(volume, rate, probability, errors, rng, *, paired):
    """Center earlier positive-case errors, keep coupling only for the paired variant.

    Nonnegative workload/rate clipping is common to both variants. This is an empirical
    approximation with explicit zero opportunity, not a learned copula or causal model.
    """
    errors = np.asarray(errors, float)
    errors = errors - errors.mean(axis=0)
    first = rng.integers(len(errors), size=DRAWS)
    second = first if paired else rng.integers(len(errors), size=DRAWS)
    active = rng.random(DRAWS) < probability
    return (
        active * np.maximum(0, volume + errors[first, 0]) * np.maximum(0, rate + errors[second, 1])
    )


def score_summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        slices = {"all", row["population"]}
        if row["season"] >= 2017:
            slices.add("modern")
        for name in slices:
            groups[row["position"], row["model"], name].append(row)
    output = []
    for (position, model, name), values in sorted(groups.items()):
        years = sorted({r["season"] for r in values})
        for metric in ("crps", "interval_score_80", "interval_score_90"):
            annual = []
            for year in years:
                group = [r for r in values if r["season"] == year]
                annual.append(
                    dict(
                        season=year,
                        score=float(np.mean([r[metric] for r in group])),
                        control_score=float(np.mean([r["control_" + metric] for r in group])),
                    )
                )
            gains = np.array([r["control_score"] - r["score"] for r in annual])
            reference = np.mean([r["control_score"] for r in annual])
            rng = np.random.default_rng(20260923)
            boot = rng.choice(gains, (4000, len(gains))).mean(axis=1)
            output.append(
                dict(
                    position=position,
                    model=model,
                    slice=name,
                    metric=metric,
                    n=len(values),
                    seasons=len(years),
                    first_season=min(years),
                    score=float(np.mean([r["score"] for r in annual])),
                    control_score=float(reference),
                    relative_gain=float(gains.mean() / reference),
                    gain_ci_low=float(np.quantile(boot, 0.025)),
                    gain_ci_high=float(np.quantile(boot, 0.975)),
                    p_value=sign_test(gains),
                    coverage_80=float(
                        np.mean([r["lower_80"] <= r["actual"] <= r["upper_80"] for r in values])
                    ),
                    coverage_90=float(
                        np.mean([r["lower_90"] <= r["actual"] <= r["upper_90"] for r in values])
                    ),
                    annual=annual,
                )
            )
    return output


def run(inputs, root):
    panel = inputs.panel()
    outputs, distributions, folds = [], [], []
    for position in ("WR", "TE"):
        cohort = [r for r in panel if r["position"] == position and r["y_targets"] is not None]
        features = columns(cohort)
        for row in cohort:
            row["y_positive"] = float(row["y_targets"] > 0)
        history = []
        for year in range(2007, 2026):
            train = [r for r in cohort if r["forecast_season"] < year]
            positive = [r for r in train if r["y_targets"] > 0]
            test = [r for r in cohort if r["forecast_season"] == year]
            if len({r["forecast_season"] for r in positive}) < 3:
                folds.append(
                    dict(
                        position=position,
                        season=year,
                        eligible=False,
                        reason="Fewer than three earlier seasons of observed positive targets",
                    )
                )
                continue
            chance = predict(train, test, "y_positive", features, kind="probability")
            volume = predict(positive, test, "y_targets", features)
            rate = predict(positive, test, "y_receiving_efficiency", features)
            direct = predict(train, test, "y_receiving_yards", features)
            earlier = earlier_history(history, year)
            observed = [r for r in earlier if r["target_error"] is not None]
            ready = len(observed) >= 100 and len({r["season"] for r in observed}) >= 3
            errors = (
                np.array([[r["target_error"], r["rate_error"]] for r in observed])
                if ready
                else None
            )
            direct_errors = np.array([r["direct_error"] for r in earlier]) if ready else None
            if ready:
                direct_errors -= direct_errors.mean()
            folds.append(
                dict(
                    position=position,
                    season=year,
                    train_max_season=year - 1,
                    train_rows=len(train),
                    positive_train_rows=len(positive),
                    test_rows=len(test),
                    calibration_rows=len(earlier),
                    positive_calibration_rows=len(observed),
                    eligible=ready,
                    residual_correlation=float(np.corrcoef(errors.T)[0, 1]) if ready else None,
                )
            )
            for index, row in enumerate(test):
                v, e, p, d = volume[index], rate[index], chance[index], direct[index]
                if row["known_available_games_cap"] == 0:
                    p, d = 0.0, 0.0
                actual = row["y_receiving_yards"]
                if ready:
                    seed = 20260923 + year * 10000 + index + (position == "TE") * 1000000
                    rng = np.random.default_rng(seed)
                    simulations = {
                        "direct_residual": np.maximum(0, d + rng.choice(direct_errors, DRAWS)),
                        "independent_joint": joint_draws(
                            v, e, p, errors, np.random.default_rng(seed + 1), paired=False
                        ),
                        "paired_joint": joint_draws(
                            v, e, p, errors, np.random.default_rng(seed + 1), paired=True
                        ),
                    }
                    if row["known_available_games_cap"] == 0:
                        simulations = {name: np.zeros(DRAWS) for name in simulations}
                    metrics = {}
                    for name, samples in simulations.items():
                        scores = dict(crps=crps(samples, actual))
                        for nominal in (80, 90):
                            alpha = 1 - nominal / 100
                            lower, upper = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
                            scores.update(
                                {
                                    f"lower_{nominal}": float(lower),
                                    f"upper_{nominal}": float(upper),
                                    f"interval_score_{nominal}": interval_score(
                                        actual, lower, upper, nominal / 100
                                    ),
                                }
                            )
                        metrics[name] = scores
                    for name, samples in simulations.items():
                        item = record(
                            row,
                            "receiving_yards",
                            name,
                            "direct_mean",
                            actual,
                            samples.mean(),
                            d,
                            appearance_probability=p,
                            **metrics[name],
                        )
                        for metric in ("crps", "interval_score_80", "interval_score_90"):
                            item["control_" + metric] = metrics["direct_residual"][metric]
                        distributions.append(item)
                        outputs.append(
                            record(
                                row,
                                "receiving_yards",
                                name,
                                "direct_mean",
                                actual,
                                samples.mean(),
                                d,
                            )
                        )
                    outputs.append(
                        record(
                            row,
                            "receiving_yards",
                            "product_of_means",
                            "direct_mean",
                            actual,
                            p * v * e,
                            d,
                        )
                    )
                history.append(
                    dict(
                        season=year,
                        direct_error=actual - d,
                        target_error=row["y_targets"] - v if row["y_targets"] > 0 else None,
                        rate_error=row["y_receiving_efficiency"] - e
                        if row["y_targets"] > 0
                        else None,
                    )
                )
            print(f"joint distributions {position} {year}: eligible={ready}", flush=True)
    points = point_summaries(outputs)
    scores = score_summaries(distributions)
    # One family for mean-MSE and the distribution stage's primary CRPS comparisons.
    holm(points + [r for r in scores if r["metric"] == "crps" and r["model"] != "direct_residual"])
    pl.DataFrame(outputs).write_parquet(root / "joint_predictions.parquet")
    pl.DataFrame(distributions).write_parquet(root / "joint_distributions.parquet")
    write_json(root / "joint_point_summaries.json", points)
    write_json(root / "joint_score_summaries.json", scores)
    write_json(root / "joint_folds.json", folds)
    return dict(
        point_predictions=len(outputs),
        distribution_predictions=len(distributions),
        folds=len(folds),
        draws=DRAWS,
    )
