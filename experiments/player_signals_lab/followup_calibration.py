"""Population-aware residual intervals and chronological conditional quantiles."""

from __future__ import annotations

import numpy as np
import polars as pl
from followup_common import CONFIG, columns, holm, predict, sign_test
from safety import write_json

METHODS = (
    "global_scaled",
    "population_scaled",
    "volume_scaled",
    "population_volume_scaled",
    "pooled_population_scaled",
    "population_asymmetric",
    "recent3_population",
    "adaptive_population",
)


def weighted_quantile(values, q, weights=None):
    values = np.asarray(values, float)
    if not len(values):
        raise ValueError("Cannot calibrate without earlier residuals")
    order = np.argsort(values)
    if weights is None:
        index = min(len(values), int(np.ceil((len(values) + 1) * q))) - 1
        return float(values[order[max(0, index)]])
    weights = np.asarray(weights, float)[order]
    index = np.searchsorted(np.cumsum(weights) / weights.sum(), q, side="left")
    return float(values[order[min(index, len(values) - 1)]])


def calibration_group(history, population, volume, boundaries, method, year):
    earlier = [r for r in history if year - 5 <= r["season"] < year]
    if len(earlier) < 100 or len({r["season"] for r in earlier}) < 3:
        return [], "warmup"
    group = [r for r in earlier if r["population"] == population]
    bucket = int(np.searchsorted(boundaries, volume))
    volume_group = [
        r for r in earlier if int(np.searchsorted(boundaries, r["prediction"])) == bucket
    ]
    both = [r for r in volume_group if r["population"] == population]
    recent = [r for r in group if r["season"] >= year - 3]
    choices = {
        "global_scaled": [],
        "population_scaled": [(group, "population")],
        "pooled_population_scaled": [(group, "population")],
        "population_asymmetric": [(group, "population")],
        "adaptive_population": [(group, "population")],
        "recent3_population": [(recent, "recent_population"), (group, "population")],
        "volume_scaled": [(volume_group, "volume")],
        "population_volume_scaled": [(both, "population_volume"), (group, "population")],
    }
    for subset, name in choices[method]:
        if len(subset) >= 60 and len({r["season"] for r in subset}) >= 3:
            return subset, name
    return earlier, "global"


def interval_score(actual, lower, upper, level):
    if lower > upper:
        raise ValueError("Crossed interval")
    return upper - lower + 2 / (1 - level) * (max(lower - actual, 0) + max(actual - upper, 0))


def metric_record(row, method, level, lower, upper, median, source, n):
    if row["target"] in ("attempts", "carries", "targets"):
        lower, upper, median = max(0, lower), max(0, upper), max(0, median)
    if row["known_available_games_cap"] == 0:
        lower = upper = median = 0.0
    return dict(
        player_id=row["player_id"],
        position=row["position"],
        target=row["target"],
        base=row["model"],
        method=method,
        season=row["season"],
        population=row["population"],
        prediction=row["prediction"],
        actual=row["actual"],
        level=level,
        lower=float(lower),
        upper=float(upper),
        median=float(median),
        covered=float(lower <= row["actual"] <= upper),
        width=float(upper - lower),
        score=float(interval_score(row["actual"], lower, upper, level)),
        median_absolute_error=float(abs(row["actual"] - median)),
        calibration_source=source,
        calibration_n=n,
    )


def summarize_intervals(records):
    frame = pl.DataFrame(records, infer_schema_length=None)
    keys = ["player_id", "season", "level"]
    baseline = frame.filter(pl.col("method") == "global_scaled").select(
        *keys, pl.col("score").alias("baseline_score")
    )
    frame = frame.join(baseline, on=keys, how="left", validate="m:1")
    if frame["baseline_score"].null_count():
        raise ValueError("Intervals require a matched baseline on every evaluated row")
    results = []
    slices = {
        "all": frame,
        "modern": frame.filter(pl.col("season") >= 2017),
        "covid_2020": frame.filter(pl.col("season") == 2020),
    }
    for name in frame["population"].unique():
        slices[name] = frame.filter(pl.col("population") == name)
    if "volume_quartile" in frame.columns:
        for bucket in range(4):
            slices[f"volume_quartile_{bucket}"] = frame.filter(pl.col("volume_quartile") == bucket)
    for name, subset in slices.items():
        for key, part in subset.group_by("position", "target", "base", "method", "level"):
            annual = (
                part.group_by("season")
                .agg(
                    pl.len().alias("n"),
                    pl.col("covered").mean(),
                    pl.col("width").mean(),
                    pl.col("score").mean(),
                    pl.col("baseline_score").mean(),
                    pl.col("median_absolute_error").mean(),
                )
                .sort("season")
                .to_dicts()
            )
            gain = np.array([a["baseline_score"] - a["score"] for a in annual])
            rng = np.random.default_rng(CONFIG["seed"])
            boot = rng.choice(gain, (4000, len(gain))).mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            control = float(np.mean([a["baseline_score"] for a in annual]))
            results.append(
                dict(zip(("position", "target", "base", "method", "level"), key, strict=True))
                | dict(
                    slice=name,
                    n=part.height,
                    years=len(annual),
                    annual=annual,
                    coverage=float(np.mean([a["covered"] for a in annual])),
                    width=float(np.mean([a["width"] for a in annual])),
                    interval_score=float(np.mean([a["score"] for a in annual])),
                    score_gain=float(gain.mean()),
                    score_gain_pct=float(100 * gain.mean() / control) if control else 0.0,
                    ci_low=float(low),
                    ci_high=float(high),
                    p_value=sign_test(gain),
                    fallback_fraction=float(
                        part.filter(pl.col("calibration_source") == "global").height / part.height
                    ),
                )
            )
    # WIS uses both central intervals plus the separately estimated median.
    wis = (
        frame.with_columns(((1 - pl.col("level")) / 2 * pl.col("score")).alias("weighted_score"))
        .group_by("player_id", "season", "position", "target", "base", "method", "population")
        .agg(
            (
                (pl.col("weighted_score").sum() + 0.5 * pl.col("median_absolute_error").first())
                / 2.5
            ).alias("wis")
        )
    )
    return results, frame, wis


def residual_experiments(rows):
    records = []
    adaptive = {}
    for year in sorted({r["season"] for r in rows}):
        earlier = [r for r in rows if year - 5 <= r["season"] < year]
        if len(earlier) < 100 or len({r["season"] for r in earlier}) < 3:
            continue
        boundaries = np.quantile([r["prediction"] for r in earlier], [0.25, 0.5, 0.75])
        current = [r for r in rows if r["season"] == year]
        cache = {}
        fold = []
        for row in current:
            bucket = int(np.searchsorted(boundaries, row["prediction"]))
            for method in METHODS:
                key = (method, row["population"], bucket)
                if key not in cache:
                    group, source = calibration_group(
                        earlier, row["population"], row["prediction"], boundaries, method, year
                    )
                    residual = np.array(
                        [(r["actual"] - r["prediction"]) / r["scale"] for r in group]
                    )
                    weights = None
                    if method == "pooled_population_scaled" and source != "global":
                        global_residual = np.array(
                            [(r["actual"] - r["prediction"]) / r["scale"] for r in earlier]
                        )
                        weight = len(group) / (len(group) + 100)
                        weights = np.r_[
                            np.full(len(group), weight / len(group)),
                            np.full(len(earlier), (1 - weight) / len(earlier)),
                        ]
                        residual = np.r_[residual, global_residual]
                    cache[key] = (group, source, residual, weights)
                group, source, residual, weights = cache[key]
                median = (
                    row["prediction"] + weighted_quantile(residual, 0.5, weights) * row["scale"]
                )
                if method == "global_scaled":
                    median = row["median_prediction"]
                for level in (0.8, 0.9):
                    alpha = adaptive.get((row["population"], level), 1 - level)
                    if method == "population_asymmetric":
                        lower = (
                            row["prediction"]
                            + weighted_quantile(residual, (1 - level) / 2) * row["scale"]
                        )
                        upper = (
                            row["prediction"]
                            + weighted_quantile(residual, 1 - (1 - level) / 2) * row["scale"]
                        )
                    else:
                        q = 1 - alpha if method == "adaptive_population" else level
                        radius = weighted_quantile(np.abs(residual), q, weights) * row["scale"]
                        lower, upper = row["prediction"] - radius, row["prediction"] + radius
                    item = metric_record(
                        row, method, level, lower, upper, median, source, len(group)
                    )
                    item["volume_quartile"] = bucket
                    item["calibration_max_season"] = max(r["season"] for r in group)
                    fold.append(item)
        # One batch update after all current-season bounds are fixed.
        for population in {r["population"] for r in current}:
            for level in (0.8, 0.9):
                group = [
                    r
                    for r in fold
                    if r["method"] == "adaptive_population"
                    and r["population"] == population
                    and r["level"] == level
                ]
                if group:
                    error = 1 - np.mean([r["covered"] for r in group])
                    old = adaptive.get((population, level), 1 - level)
                    adaptive[population, level] = float(
                        np.clip(old + 0.05 * ((1 - level) - error), 0.02, 0.40)
                    )
        records.extend(fold)
    return records


def quantile_experiments(panel, prior, position):
    panel = [r for r in panel if r["position"] == position]
    features = columns(panel)
    old = {(r["player_id"], r["season"]): r for r in prior}
    oof, records, folds = [], [], []
    for year in range(2007, 2026):
        train = [r for r in panel if r["forecast_season"] < year]
        test = [r for r in panel if r["forecast_season"] == year]
        q = np.array(
            [
                predict(train, test, "y_season_points", features, quantile=value, lower=None)
                for value in (0.05, 0.10, 0.50, 0.90, 0.95)
            ]
        ).T
        crossings = int(np.any(np.diff(q, axis=1) < 0, axis=1).sum())
        q = np.sort(q, axis=1)
        earlier = [r for r in oof if year - 5 <= r["season"] < year]
        eligible = len(earlier) >= 100 and len({r["season"] for r in earlier}) >= 3
        new = []
        for i, player in enumerate(test):
            row = old[player["player_id"], year]
            new.append({**row, "q": q[i]})
            if not eligible:
                continue
            group = [r for r in earlier if r["population"] == row["population"]]
            if len(group) < 60 or len({r["season"] for r in group}) < 3:
                group = earlier
            for level, lo, hi in ((0.8, 1, 3), (0.9, 0, 4)):
                records.append(
                    metric_record(
                        row,
                        "global_scaled",
                        level,
                        row[f"scaled_{int(level * 100)}_lower"],
                        row[f"scaled_{int(level * 100)}_upper"],
                        row["median_prediction"],
                        "global",
                        row["calibration_n"],
                    )
                )
                for method, calibration in (
                    ("quantile_uncalibrated", []),
                    ("quantile_cqr", earlier),
                    ("quantile_population_cqr", group),
                ):
                    scores = [
                        max(r["q"][lo] - r["actual"], r["actual"] - r["q"][hi], 0)
                        for r in calibration
                    ]
                    expansion = weighted_quantile(scores, level) if scores else 0
                    records.append(
                        metric_record(
                            row,
                            method,
                            level,
                            q[i, lo] - expansion,
                            q[i, hi] + expansion,
                            q[i, 2],
                            "population"
                            if calibration is group and group is not earlier
                            else "global",
                            len(calibration),
                        )
                    )
        oof.extend(new)
        folds.append(
            dict(
                position=position,
                season=year,
                train_n=len(train),
                test_n=len(test),
                crossing_rows=crossings,
                calibration_n=len(earlier),
            )
        )
        print(f"quantiles {position} {year}", flush=True)
    return records, folds


def run(inputs, root):
    old = inputs.previous().filter(pl.col("model").is_in(["market_control", "raw_rates"]))
    root.joinpath("parts").mkdir()
    summaries, wis_parts = [], []
    for key, frame in old.group_by("position", "target", "model", maintain_order=True):
        position, target, base = key
        records = residual_experiments(frame.to_dicts())
        summary, evaluated, wis = summarize_intervals(records)
        name = f"{position}_{target}_{base}"
        evaluated.write_parquet(root / "parts" / f"{name}.parquet")
        summaries.extend(summary)
        wis_parts.append(wis)
        print(f"residual calibration {name}: {len(records)} interval records", flush=True)
    panel = inputs.panel()
    folds = []
    for position in ("QB", "RB", "WR", "TE"):
        prior = old.filter(
            (pl.col("position") == position)
            & (pl.col("target") == "season_points")
            & (pl.col("model") == "market_control")
        ).to_dicts()
        records, diagnostics = quantile_experiments(panel, prior, position)
        summary, evaluated, wis = summarize_intervals(records)
        evaluated.write_parquet(root / "parts" / f"{position}_quantiles.parquet")
        summaries.extend(summary)
        wis_parts.append(wis)
        folds.extend(diagnostics)
    # Baseline comparisons are identities, not hypotheses in the Holm family.
    holm([r for r in summaries if r["method"] != "global_scaled"])
    write_json(root / "calibration_summaries.json", summaries)
    write_json(root / "quantile_folds.json", folds)
    pl.concat(wis_parts).unique().write_parquet(root / "weighted_interval_scores.parquet")
    return dict(comparisons=len(summaries), parts=len(list((root / "parts").glob("*.parquet"))))
