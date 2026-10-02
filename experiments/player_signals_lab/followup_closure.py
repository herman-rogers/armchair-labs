"""Post-Protocol-2 exploratory age, recency, signed calibration, and dependence checks."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import polars as pl
from followup_calibration import (
    calibration_group,
    metric_record,
    summarize_intervals,
    weighted_quantile,
)
from followup_common import LAB, columns, holm, point_summaries, predict, record
from safety import write_json


def artifact(inputs, run, name):
    folder = LAB / "runs" / run
    manifest = json.loads(inputs.bind(folder / "manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest["protected_inputs_unchanged"]:
        raise ValueError(f"Unusable prior trial: {run}")
    return inputs.bind(folder / name, manifest["outputs"][name])


def age_and_recency(inputs, root):
    panel = inputs.panel()
    outputs, folds = [], []
    targets = dict(
        QB="passing_efficiency",
        RB="rushing_efficiency",
        WR="receiving_efficiency",
        TE="receiving_efficiency",
    )
    for position, efficiency in targets.items():
        cohort = [r for r in panel if r["position"] == position]
        features = columns(cohort)
        for r in cohort:
            r["y_meaningful_role"] = float(r["y_scoring_appearances"] >= 8)
        for target in ("season_points", "meaningful_role", efficiency):
            for year in range(2007, 2026):
                train = [
                    r
                    for r in cohort
                    if r["forecast_season"] < year and r[f"y_{target}"] is not None
                ]
                test = [
                    r
                    for r in cohort
                    if r["forecast_season"] == year and r[f"y_{target}"] is not None
                ]
                if not test or len({r["forecast_season"] for r in train}) < 3:
                    continue
                kind = "probability" if target == "meaningful_role" else "mean"
                if target == "season_points":
                    base = predict(train, test, f"y_{target}", features)
                    values = predict(
                        [r for r in train if r["forecast_season"] >= year - 5],
                        test,
                        f"y_{target}",
                        features,
                    )
                    model, control = "recent_five_years", "expanding_history"
                else:
                    base = predict(
                        train, test, f"y_{target}", [f for f in features if f != "x_age"], kind=kind
                    )
                    values = predict(train, test, f"y_{target}", features, kind=kind)
                    model, control = "with_age", "without_age"
                for row, value, reference in zip(test, values, base, strict=True):
                    if row["known_available_games_cap"] == 0 and target in (
                        "season_points",
                        "meaningful_role",
                    ):
                        value = reference = 0.0
                    age = row["x_age"]
                    outputs.append(
                        record(
                            row,
                            target,
                            model,
                            control,
                            row[f"y_{target}"],
                            value,
                            reference,
                            age_band="unknown"
                            if age is None
                            else "under25"
                            if age < 25
                            else "25to29"
                            if age < 30
                            else "30plus",
                        )
                    )
                folds.append(
                    dict(
                        position=position,
                        target=target,
                        season=year,
                        train_max_season=year - 1,
                        test_n=len(test),
                    )
                )
        print(f"age/recency {position}", flush=True)
    pl.DataFrame(outputs).write_parquet(root / "age_recency_predictions.parquet")
    write_json(root / "age_recency_folds.json", folds)
    results = point_summaries(outputs)
    age_rows = [{**r, "population": r["age_band"]} for r in outputs]
    write_json(
        root / "age_band_diagnostics.json",
        [r for r in point_summaries(age_rows) if r["slice"] not in ("all", "modern")],
    )
    return results


def signed_bounds(lower, upper, correction):
    lo, hi = lower - correction, upper + correction
    crossed = lo > hi
    if crossed:
        lo = hi = (lower + upper) / 2
    return lo, hi, crossed


def signed_calibration(inputs, root):
    summaries, records_all, wis_all = [], [], []
    meta = {(r["player_id"], r["forecast_season"]): r for r in inputs.panel()}
    for position in ("QB", "RB", "WR", "TE"):
        source = pl.read_parquet(
            artifact(inputs, "calibration_001", f"parts/{position}_quantiles.parquet")
        ).drop("baseline_score")  # Recompute the matched-control join for this narrower cohort.
        rows = source.filter(pl.col("method") == "quantile_uncalibrated").to_dicts()
        baseline = {
            (r["player_id"], r["season"], r["level"]): r
            for r in source.filter(pl.col("method") == "global_scaled").to_dicts()
        }
        records = []
        for year in range(2013, 2026):
            for level in (0.8, 0.9):
                earlier = [
                    r for r in rows if year - 5 <= r["season"] < year and r["level"] == level
                ]
                if len(earlier) < 100 or len({r["season"] for r in earlier}) < 3:
                    continue
                boundaries = np.quantile([r["prediction"] for r in earlier], [0.25, 0.5, 0.75])
                cache = {}
                for row in [r for r in rows if r["season"] == year and r["level"] == level]:
                    records.append(baseline[row["player_id"], year, level])
                    records.append(row)
                    for method, grouping in (
                        ("signed_cqr_global", "global_scaled"),
                        ("signed_cqr_population", "population_scaled"),
                        ("signed_cqr_volume", "volume_scaled"),
                    ):
                        bucket = int(np.searchsorted(boundaries, row["prediction"]))
                        key = method, row["population"], bucket
                        if key not in cache:
                            group, name = calibration_group(
                                earlier,
                                row["population"],
                                row["prediction"],
                                boundaries,
                                grouping,
                                year,
                            )
                            scores = [
                                max(r["lower"] - r["actual"], r["actual"] - r["upper"])
                                for r in group
                            ]
                            cache[key] = weighted_quantile(scores, level), len(group), name
                        correction, n, name = cache[key]
                        lower, upper, crossed = signed_bounds(
                            row["lower"], row["upper"], correction
                        )
                        item = metric_record(
                            {
                                **row,
                                "model": row["base"],
                                "known_available_games_cap": meta[row["player_id"], year][
                                    "known_available_games_cap"
                                ],
                            },
                            method,
                            level,
                            lower,
                            upper,
                            row["median"],
                            name,
                            n,
                        )
                        item.update(
                            signed_correction=correction,
                            collapsed_crossing=crossed,
                            calibration_max_season=year - 1,
                        )
                        records.append(item)
        result, frame, wis = summarize_intervals(records)
        summaries.extend(result)
        records_all.append(frame)
        wis_all.append(wis)
    pl.concat(records_all, how="diagonal_relaxed").write_parquet(
        root / "signed_calibration.parquet"
    )
    pl.concat(wis_all).write_parquet(root / "signed_weighted_interval_scores.parquet")
    return summaries


def cluster_sensitivity(rows, labels, seed=20260923):
    """Resample clusters of paired losses with original equal-season row weights."""
    seasons = defaultdict(list)
    for row in rows:
        seasons[row["season"]].append(row)
    aggregates = defaultdict(lambda: [0.0, 0.0])
    for row, label in zip(rows, labels, strict=True):
        weight = 1 / len(seasons[row["season"]]) / len(seasons)
        gain = row.get("gain")
        if gain is None:
            gain = (row["actual"] - row["control_prediction"]) ** 2 - (
                row["actual"] - row["prediction"]
            ) ** 2
        aggregates[label][0] += weight * gain
        aggregates[label][1] += weight
    totals = np.array(list(aggregates.values()))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(20):
        picks = rng.integers(len(totals), size=(100, len(totals)))
        sums = totals[picks].sum(axis=1)
        estimates.extend(sums[:, 0] / sums[:, 1])
    return dict(
        clusters=len(totals),
        gain=float(totals[:, 0].sum()),
        low=float(np.quantile(estimates, 0.025)),
        high=float(np.quantile(estimates, 0.975)),
    )


def block_sensitivity(rows):
    annual = defaultdict(list)
    for r in rows:
        gain = r.get("gain")
        if gain is None:
            gain = (r["actual"] - r["control_prediction"]) ** 2 - (
                r["actual"] - r["prediction"]
            ) ** 2
        annual[r["season"]].append(gain)
    values = np.array([np.mean(annual[y]) for y in sorted(annual)])
    rng = np.random.default_rng(20260923)
    starts = rng.integers(len(values), size=(2000, int(np.ceil(len(values) / 2))))
    indices = np.stack([starts, (starts + 1) % len(values)], axis=2).reshape(2000, -1)[
        :, : len(values)
    ]
    estimates = values[indices].mean(axis=1)
    return dict(
        seasons=len(values),
        gain=float(values.mean()),
        low=float(np.quantile(estimates, 0.025)),
        high=float(np.quantile(estimates, 0.975)),
    )


def dependence_and_support(inputs, root):
    meta = {(r["player_id"], r["forecast_season"]): r for r in inputs.panel()}
    source = inputs.gold("preseason_features")
    originals = {
        (r["player_id"], r["forecast_season"]): r
        for r in source.select("player_id", "forecast_season", "source_team").to_dicts()
    }
    depend, supports, exposure = [], [], []
    files = [
        ("signals_001", "signals_predictions.parquet"),
        ("structure_001", "team_allocation_predictions.parquet"),
        ("structure_001", "weekly_roles_predictions.parquet"),
    ]
    for run, name in files:
        frame = pl.read_parquet(artifact(inputs, run, name)).filter(
            pl.col("model") != pl.col("control")
        )
        for key, part in frame.group_by("position", "target", "model", "control"):
            rows = part.to_dicts()
            description = dict(zip(("position", "target", "model", "control"), key, strict=True))
            result = dict(
                **description,
                n=len(rows),
                career=cluster_sensitivity(rows, [r["player_id"] for r in rows]),
                two_season_block=block_sensitivity(rows),
            )
            if "cutoff_team" in frame.columns:
                result["franchise"] = cluster_sensitivity(rows, [r["cutoff_team"] for r in rows])
            depend.append(result)
            if run != "signals_001":
                continue
            groups = defaultdict(list)
            for r in rows:
                original = meta[r["player_id"], r["season"]]
                age = original["x_age"]
                if age is not None and age >= 30:
                    groups["age30plus"].append(r)
                if original["m_observed"] == 0:
                    groups["missing_market"].append(r)
                team = original["cutoff_preseason_team"]
                prior_team = originals[r["player_id"], r["season"]]["source_team"]
                if team and prior_team and team != prior_team:
                    groups["changed_team_raw_abbreviation"].append(r)
                prior = meta.get((r["player_id"], r["season"] - 1))
                if prior and prior["position"] != original["position"]:
                    groups["changed_position"].append(r)
                field = (
                    "x_prior_attempts"
                    if r["position"] == "QB"
                    else "x_prior_carries"
                    if r["position"] == "RB"
                    else "x_prior_targets"
                )
                if (
                    original["m_log_ecr"] is not None
                    and original["m_log_ecr"] <= np.log(24)
                    and (original[field] or 0) < (100 if r["position"] in ("QB", "RB") else 50)
                ):
                    groups["low_prior_volume_top24_market"].append(r)
            for label, group in groups.items():
                annual = defaultdict(list)
                for r in group:
                    annual[r["season"]].append(
                        (r["actual"] - r["control_prediction"]) ** 2
                        - (r["actual"] - r["prediction"]) ** 2
                    )
                supports.append(
                    dict(
                        **description,
                        slice=label,
                        n=len(group),
                        years=len(annual),
                        mse_gain=float(np.mean([np.mean(v) for v in annual.values()])),
                        max_prediction=float(max(r["prediction"] for r in group)),
                    )
                )
            if key[1].endswith("efficiency"):
                field = {
                    "passing_efficiency": "y_attempts",
                    "rushing_efficiency": "y_carries",
                    "receiving_efficiency": "y_targets",
                }[key[1]]
                annual = []
                for year in sorted({r["season"] for r in rows}):
                    group = [r for r in rows if r["season"] == year]
                    weights = np.array([meta[r["player_id"], year][field] for r in group])
                    gains = np.array(
                        [
                            (r["actual"] - r["control_prediction"]) ** 2
                            - (r["actual"] - r["prediction"]) ** 2
                            for r in group
                        ]
                    )
                    annual.append(
                        dict(
                            season=year,
                            player_gain=float(gains.mean()),
                            exposure_gain=float(np.average(gains, weights=weights)),
                        )
                    )
                exposure.append(
                    dict(
                        **description,
                        annual=annual,
                        player_gain=float(np.mean([r["player_gain"] for r in annual])),
                        exposure_gain=float(np.mean([r["exposure_gain"] for r in annual])),
                    )
                )
    for position in ("QB", "RB", "WR", "TE"):
        frame = pl.read_parquet(
            artifact(
                inputs, "calibration_001", f"parts/{position}_season_points_market_control.parquet"
            )
        )
        rows = frame.filter(
            (pl.col("method") == "volume_scaled") & (pl.col("level") == 0.8)
        ).to_dicts()
        for r in rows:
            r["gain"] = r["baseline_score"] - r["score"]
        depend.append(
            dict(
                position=position,
                target="season_points_80_interval_score",
                model="volume_scaled",
                control="global_scaled",
                n=len(rows),
                career=cluster_sensitivity(rows, [r["player_id"] for r in rows]),
                two_season_block=block_sensitivity(rows),
                franchise=cluster_sensitivity(
                    rows,
                    [
                        meta[r["player_id"], r["season"]]["cutoff_preseason_team"] or "unknown"
                        for r in rows
                    ],
                ),
            )
        )
    write_json(root / "dependence_sensitivity.json", depend)
    write_json(root / "support_slices.json", supports)
    write_json(root / "exposure_weighted_efficiency.json", exposure)


def run(inputs, root):
    inputs.bind(LAB / "closure_protocol.md")
    age = age_and_recency(inputs, root)
    calibration = signed_calibration(inputs, root)
    holm(age + [r for r in calibration if r["method"].startswith("signed_")])
    write_json(root / "age_recency_summaries.json", age)
    write_json(root / "signed_calibration_summaries.json", calibration)
    dependence_and_support(inputs, root)
    return dict(age_comparisons=len(age), signed_calibration_comparisons=len(calibration))
