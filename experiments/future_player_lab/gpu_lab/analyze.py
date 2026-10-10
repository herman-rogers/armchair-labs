"""Chronological selection, matched evaluation, and earlier-fold residual calibration."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from experiments.future_player_lab.notebooks.rb_probability import (
    empirical_crps,
    interval,
    residual_distribution,
)
from scipy.optimize import minimize
from scipy.stats import spearmanr

from .run import COUNT_STATS, RUNS, STATS, digest, json_write, verify_checkpoint

KEY = ["horizon", "year", "origin", "player_id", "target"]


def metric_values(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if not len(y):
        return {
            "n": 0,
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
            "spearman": None,
            "normalized_mse": None,
        }
    mse = float(np.mean((y - pred) ** 2))
    variance = float(np.var(y))
    corr = float(spearmanr(y, pred).statistic) if np.ptp(y) and np.ptp(pred) else None
    return {
        "n": len(y),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(y - pred))),
        "r2": 1 - mse / variance if variance > 0 else None,
        "spearman": corr,
        "normalized_mse": mse / max(variance, 1e-8),
    }


def scores(rows, groups):
    records = []
    for key, group in rows.group_by(groups, maintain_order=True):
        record = dict(zip(groups, key, strict=True))
        record.update(metric_values(group["actual"].to_numpy(), group["prediction"].to_numpy()))
        records.append(record)
    return pl.DataFrame(records, infer_schema_length=None)


def choose_members(annual, year, lookback=3, count=3):
    """Only complete candidate histories before the forecast season may select a model."""
    prior = annual.filter((pl.col("year") >= year - lookback) & (pl.col("year") < year))
    ranked = (
        prior.filter(pl.col("normalized_mse").is_finite())
        .group_by("recipe")
        .agg(
            pl.col("year").n_unique().alias("seasons"),
            pl.col("normalized_mse").mean().alias("selection_loss"),
        )
        .filter(pl.col("seasons") == lookback)
        .sort("selection_loss", "recipe")
    )
    return ranked.head(count).to_dicts()


def blend(rows, members, policy, year, weights=None):
    selected = rows.filter(pl.col("recipe").is_in(members))
    if weights is None:
        weights = np.full(len(members), 1 / len(members))
    mapping = dict(zip(members, map(float, weights), strict=True))
    # A model missing a particular row cannot silently change ensemble membership.
    key_cols = [n for n in rows.columns if n not in {"prediction", "recipe", "fit_seconds"}]
    selected = selected.with_columns(pl.col("recipe").replace_strict(mapping).alias("weight"))
    result = selected.group_by(key_cols).agg(
        (pl.col("prediction") * pl.col("weight")).sum().alias("prediction"),
        pl.col("recipe").n_unique().alias("members_present"),
        pl.col("prediction").is_finite().fill_null(False).all().alias("predictions_known"),
    )
    result = (
        result.filter(pl.col("members_present") == len(members))
        .with_columns(
            pl.when(pl.col("predictions_known")).then(pl.col("prediction")).otherwise(None)
        )
        .drop("members_present", "predictions_known")
    )
    return result.with_columns(
        pl.lit(policy).alias("recipe"), pl.lit(0.0).alias("fit_seconds")
    ).sort(KEY)


def diverse_members(ranked, config, count=3):
    families = {r["id"]: r["family"] for r in config["recipes"]}
    result, seen = [], set()
    for record in ranked:
        name = record["recipe"]
        family = families.get(name, name)
        if family not in seen:
            result.append(name)
            seen.add(family)
        if len(result) == count:
            break
    return result


def convex_weights(prior, members):
    """Simplex weights from earlier out-of-year predictions, with fixed weak L2 regularization."""
    aligned = prior.filter(pl.col("recipe") == members[0]).select(*KEY, "actual")
    for i, name in enumerate(members):
        component = prior.filter(pl.col("recipe") == name).select(
            *KEY, pl.col("prediction").alias(f"p{i}")
        )
        aligned = aligned.join(component, on=KEY, how="inner", validate="1:1")
    aligned = aligned.filter(pl.col("actual").is_finite()).sort(KEY)
    x = aligned.select([f"p{i}" for i in range(len(members))]).to_numpy()
    y = aligned["actual"].to_numpy()
    row_weight = np.empty(len(y))
    seasons = aligned["year"].to_numpy()
    for season in np.unique(seasons):
        mask = seasons == season
        row_weight[mask] = 1 / (mask.sum() * max(np.var(y[mask]), 1e-8))
    row_weight /= len(np.unique(seasons))
    uniform = np.full(len(members), 1 / len(members))

    def objective(w):
        residual = x @ w - y
        return np.sum(row_weight * residual**2) + 0.01 * np.sum((w - uniform) ** 2)

    def gradient(w):
        return 2 * x.T @ (row_weight * (x @ w - y)) + 0.02 * (w - uniform)

    fitted = minimize(
        objective,
        uniform,
        jac=gradient,
        bounds=[(0, 1)] * len(members),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        method="SLSQP",
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Earlier-fold convex weight optimization failed: {fitted.message}")
    weights = np.maximum(0, fitted.x)
    return (weights / weights.sum()).tolist()


def policy_predictions(rows, annual, config):
    outputs, selections, calibrated = [], [], []
    for horizon in config["horizons"]:
        for target in config["targets"]:
            subset = rows.filter((pl.col("horizon") == horizon) & (pl.col("target") == target))
            candidate_scores = annual.filter(
                (pl.col("horizon") == horizon)
                & (pl.col("target") == target)
                & (pl.col("recipe") != "persistence")
            )
            for year in config["evaluation_years"]:
                ranked = choose_members(
                    candidate_scores,
                    year,
                    config["selection_years"],
                    count=len(config["recipes"]) + 1,
                )
                if not ranked:
                    continue
                policies = {
                    "selected": [ranked[0]["recipe"]],
                    "ensemble_top3": [r["recipe"] for r in ranked[:3]],
                    "ensemble_diverse": diverse_members(ranked, config),
                    "ensemble_convex": diverse_members(ranked, config),
                    "persistence_calibrated": ["persistence"],
                }
                prior = subset.filter(
                    (pl.col("year") < year) & (pl.col("year") >= year - config["selection_years"])
                )
                for policy, members in policies.items():
                    weights = (
                        convex_weights(prior, members)
                        if policy == "ensemble_convex"
                        else [1 / len(members)] * len(members)
                    )
                    test = blend(
                        subset.filter(pl.col("year") == year), members, policy, year, weights
                    )
                    cal = blend(
                        subset.filter((pl.col("year") < year) & (pl.col("year") >= year - 3)),
                        members,
                        policy,
                        year,
                        weights,
                    )
                    if not test.height:
                        continue
                    outputs.append(test)
                    selections.append(
                        {
                            "horizon": horizon,
                            "target": target,
                            "year": year,
                            "policy": policy,
                            "members": members,
                            "weights": weights,
                            "selection_max_year": year - 1,
                            "selection_loss": ranked[0]["selection_loss"]
                            if policy == "selected"
                            else None,
                        }
                    )
                    valid_cal = cal.filter(
                        pl.col("actual").is_finite() & pl.col("prediction").is_finite()
                    )
                    valid_test = test.filter(
                        pl.col("actual").is_finite() & pl.col("prediction").is_finite()
                    )
                    if valid_cal.height < 20 or not valid_test.height:
                        continue
                    samples = residual_distribution(
                        valid_test["prediction"].to_numpy() / valid_test["weeks"].to_numpy(),
                        valid_test["weeks"].to_numpy(),
                        valid_cal["prediction"].to_numpy() / valid_cal["weeks"].to_numpy(),
                        valid_cal["actual"].to_numpy() / valid_cal["weeks"].to_numpy(),
                        neighbors=100,
                        count_stat=target in COUNT_STATS,
                    )
                    y = valid_test["actual"].to_numpy()
                    q = interval(samples, 0.8)
                    calibrated.append(
                        valid_test.with_columns(
                            pl.Series("crps", empirical_crps(y, samples)),
                            pl.Series("lower80", q[0]),
                            pl.Series("upper80", q[1]),
                            pl.Series("coverage80", ((y >= q[0]) & (y <= q[1])).astype(float)),
                            pl.Series("width80", q[1] - q[0]),
                            pl.Series("distribution_mean", samples.mean(axis=1)),
                            pl.lit(int(valid_cal["year"].max())).alias("calibration_max_year"),
                        )
                    )
    if not outputs:
        raise ValueError("No policies have a complete earlier selection history for evaluation")
    policies = pl.concat(outputs, how="diagonal_relaxed")
    distribution = (
        pl.concat(calibrated, how="diagonal_relaxed")
        if calibrated
        else policies.head(0).with_columns(
            *[
                pl.Series(name, [], dtype=pl.Float64)
                for name in [
                    "crps",
                    "lower80",
                    "upper80",
                    "coverage80",
                    "width80",
                    "distribution_mean",
                ]
            ],
            pl.Series("calibration_max_year", [], dtype=pl.Int64),
        )
    )
    return policies, pl.DataFrame(selections, infer_schema_length=None), distribution


def stat_points(policies):
    """Sum independently forecast stat means; no false claim of a joint distribution."""
    weights = dict(zip(STATS[:7], [0.0, 0.1, 6.0, 0.0, 1.0, 0.1, 6.0], strict=True))
    cols = [n for n in KEY if n != "target"] + ["recipe", "weeks", "training_max_year"]
    rows = policies.filter(pl.col("target").is_in([s for s, w in weights.items() if w]))
    rows = rows.with_columns(pl.col("target").replace_strict(weights).alias("weight"))
    result = (
        rows.group_by(cols)
        .agg(
            (pl.col("actual") * pl.col("weight")).sum().alias("actual"),
            (pl.col("prediction") * pl.col("weight")).sum().alias("prediction"),
            pl.col("target").n_unique().alias("components"),
            pl.col("actual").is_finite().fill_null(False).all().alias("labels_known"),
            pl.col("prediction").is_finite().fill_null(False).all().alias("predictions_known"),
        )
        .filter((pl.col("components") == 5) & pl.col("predictions_known"))
        .with_columns(
            pl.when(pl.col("labels_known")).then(pl.col("actual")).otherwise(None).alias("actual")
        )
    )
    return result.drop("components", "labels_known", "predictions_known").with_columns(
        pl.lit("points_from_stats").alias("target"), pl.lit(0.0).alias("fit_seconds")
    )


def paired_baseline(rows, seed=20261003):
    """Season-block bootstrap on exactly matched player-origin rows, never iid player CIs."""
    baseline = rows.filter(pl.col("recipe") == "persistence").select(
        *KEY, pl.col("prediction").alias("baseline")
    )
    joined = rows.join(baseline, on=KEY, how="inner", validate="m:1").filter(
        pl.col("actual").is_finite()
        & pl.col("prediction").is_finite()
        & pl.col("baseline").is_finite()
    )
    joined = joined.with_columns(
        ((pl.col("prediction") - pl.col("actual")) ** 2).alias("se"),
        ((pl.col("baseline") - pl.col("actual")) ** 2).alias("baseline_se"),
    )
    results = []
    rng = np.random.default_rng(seed)
    joined = joined.sort("horizon", "target", "recipe", "year", "origin", "player_id")
    for key, group in joined.group_by(["horizon", "target", "recipe"], maintain_order=True):
        annual = group.group_by("year").agg(
            (pl.col("se") - pl.col("baseline_se")).mean().alias("delta")
        )
        delta = annual.sort("year")["delta"].to_numpy()
        boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
        mse, base = float(group["se"].mean()), float(group["baseline_se"].mean())
        results.append(
            {
                "horizon": key[0],
                "target": key[1],
                "recipe": key[2],
                "n": group.height,
                "seasons": len(delta),
                "rmse": np.sqrt(mse),
                "baseline_rmse": np.sqrt(base),
                "mse_improvement_pct": 100 * (1 - mse / base) if base else None,
                "mean_season_delta_mse": float(delta.mean()),
                "delta_low95": float(np.quantile(boot, 0.025)),
                "delta_high95": float(np.quantile(boot, 0.975)),
            }
        )
    return pl.DataFrame(results, infer_schema_length=None)


def analyze(out):
    out = Path(out)
    config = json.loads((out / "config.json").read_text())
    files = sorted(out.glob("*/fits/*/*/complete.json"))
    rows = []
    for receipt in files:
        verify_checkpoint(receipt.parent)
        rows.append(pl.read_parquet(receipt.parent / "predictions.parquet"))
    for horizon in config["horizons"]:
        rows.append(pl.read_parquet(out / horizon / "baselines.parquet"))
    predictions = pl.concat(rows, how="diagonal_relaxed")
    if predictions.select([*KEY, "recipe"]).is_duplicated().any():
        raise ValueError("Duplicate forecast keys")
    if predictions.filter(pl.col("training_max_year") >= pl.col("year")).height:
        raise ValueError("Training chronology violated")
    annual = scores(predictions, ["horizon", "target", "recipe", "year"])
    policies, selections, distribution = policy_predictions(predictions, annual, config)
    points = stat_points(policies)
    # Compare component-point policies with the same raw persistence points baseline.
    component_baseline = (
        predictions.filter(pl.col("target") == "points")
        .filter(pl.col("recipe") == "persistence")
        .with_columns(pl.lit("points_from_stats").alias("target"))
    )
    all_rows = pl.concat(
        [predictions, policies, points, component_baseline], how="diagonal_relaxed"
    ).sort(*KEY, "recipe")
    annual = annual.sort("horizon", "target", "recipe", "year")
    selections = selections.sort("horizon", "target", "year", "policy")
    distribution = distribution.sort(*KEY, "recipe")
    evaluation = all_rows.filter(pl.col("year").is_in(config["evaluation_years"]))
    dest = out / "analysis"
    dest.mkdir(exist_ok=True)
    shutil.copyfile(__file__, dest / "analysis_source.py")
    for name, table in {
        "predictions": all_rows,
        "annual_metrics": annual,
        "policy_selection": selections,
        "distributions": distribution,
        "pooled_metrics": scores(evaluation, ["horizon", "target", "recipe"]),
        "evaluation_annual_metrics": scores(evaluation, ["horizon", "target", "recipe", "year"]),
        "matched_baseline": paired_baseline(evaluation),
    }.items():
        table.write_parquet(dest / f"{name}.parquet")
    summary = (
        distribution.group_by("horizon", "target", "recipe")
        .agg(
            pl.len().alias("n"),
            pl.col("crps").mean(),
            pl.col("coverage80").mean(),
            pl.col("width80").mean(),
        )
        .sort("horizon", "target", "recipe")
    )
    summary.write_parquet(dest / "distribution_summary.parquet")
    metadata = pl.DataFrame(
        [
            {
                "recipe": r["id"],
                "family": r["family"],
                "representation": r["representation"]["kind"],
                "view": r["view"],
                "history": r["history"],
            }
            for r in config["recipes"]
        ]
    )
    annual.join(metadata, on="recipe", how="left").write_parquet(dest / "landscape.parquet")
    report(dest, config)
    seal_analysis(out, status="core_complete")
    return dest


def seal_analysis(out, status="complete"):
    """Seal all analysis artifacts after the last producing step, including figures."""
    out = Path(out)
    dest = out / "analysis"
    config = json.loads((out / "config.json").read_text())
    json_write(
        dest / "manifest.json",
        {
            "status": status,
            "selection_lookback": config["selection_years"],
            "files": {
                str(p.relative_to(dest)): digest(p)
                for p in sorted(dest.rglob("*"))
                if p.is_file() and p != dest / "manifest.json" and not p.name.endswith(".tmp")
            },
        },
    )


def markdown_table(frame, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.select(cols).iter_rows():
        lines.append(
            "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
        )
    return "\n".join(lines)


def report(dest, config):
    matched = pl.read_parquet(dest / "matched_baseline.parquet")
    selected = matched.filter(
        pl.col("recipe").is_in(["selected", "ensemble_top3", "ensemble_diverse", "ensemble_convex"])
    )
    selected = selected.sort("horizon", "target", "recipe")
    distributions = pl.read_parquet(dest / "distribution_summary.parquet")
    lines = [
        "# Running back raw-input GPU tree campaign",
        "",
        "Historical development evidence; 2025 was already examined in previous research.",
        "",
        f"Frozen library: {len(config['recipes'])} recipes. Evaluation: "
        f"{config['evaluation_years']}. Earlier three seasons select each policy.",
        "",
        "All source-eligible raw numeric features begin eligible; transformations, missing-value "
        "handling, and feature reduction fit only earlier seasons. "
        "Descriptive profiles are separate.",
        "",
        "## Matched results against prior-production persistence",
        "",
        "Positive pooled-row MSE improvement is better. Intervals describe the equal-season "
        "mean MSE difference, resampling seasons rather than players; these weightings differ. "
        "Five seasons give coarse uncertainty. Per-target units differ.",
        "",
        markdown_table(
            selected,
            [
                "horizon",
                "target",
                "recipe",
                "n",
                "rmse",
                "baseline_rmse",
                "mse_improvement_pct",
                "delta_low95",
                "delta_high95",
            ],
        ),
        "",
        "## Earlier-fold residual distributions",
        "",
        "CRPS scores the finite empirical CDF. Calibration uses earlier out-of-season errors; "
        "coverage is measured, not guaranteed. Selection and calibration reuse earlier years, "
        "which can make calibration optimistic. No joint fantasy distribution is claimed.",
        "",
        markdown_table(
            distributions.sort("horizon", "target", "recipe"),
            ["horizon", "target", "recipe", "n", "crps", "coverage80", "width80"],
        ),
        "",
        "## Interpretation limits",
        "",
        "Raw PPR means rushing/receiving yards ÷ 10 + receptions + 6 × rushing/receiving TDs. "
        "It excludes fumbles, passing, bonuses, returns, and conversions. Candidate populations "
        "are past observed players plus exactly linked drafted entrants; newcomer coverage and "
        "last-observed team assignment are imperfect. Historical source vintages may be revised. "
        "No same-date ECR comparison is available in this raw source contract. "
        "Feature importance and full-data profiles are descriptive, not causal. "
        "The library is finite, and its retrospective winner is not a prospective guarantee.",
    ]
    (dest / "readout.md").write_text("\n".join(lines) + "\n")
    records = selected.to_dicts()
    encoded = json.dumps(records).replace("</", "<\\/")
    (dest / "index.html").write_text(
        """<!doctype html><meta charset="utf-8">
<title>RB GPU research</title><style>
body{font:16px system-ui;margin:40px;background:#111821;color:#e6edf3}
table{border-collapse:collapse;width:100%;font-size:14px}
td,th{padding:9px;border-bottom:1px solid #344150;text-align:right}
td:first-child,th:first-child{text-align:left}select{padding:8px;margin:16px;background:#253442;color:white}
h1{color:#7dd3fc}a{color:#7dd3fc}p{max-width:1000px;line-height:1.5}</style>
<h1>Running back GPU research</h1>
<p>Raw inputs · chronological selection · historical development evidence</p>
<p>Choose a horizon. Positive MSE improvement means lower squared error than matched persistence.
Models and preprocessing only learn earlier seasons. 2025 has already been studied.</p>
<select id="h"><option>season</option><option>remaining</option></select>
<table><thead id="head"></thead><tbody id="body"></tbody></table>
<p><a href="readout.md">Complete readout</a> ·
<a href="../profiles/index.html">Player profiles</a></p>
<script>const rows="""
        + encoded
        + """;const cols=['target','recipe','n','rmse','baseline_rmse',
'mse_improvement_pct','delta_low95','delta_high95'];
document.getElementById('head').innerHTML='<tr>'+cols.map(x=>'<th>'+x+'</th>').join('')+'</tr>';
function render(){let h=document.getElementById('h').value;
document.getElementById('body').innerHTML=rows.filter(x=>x.horizon===h)
.map(r=>'<tr>'+cols.map(c=>'<td>'+
(typeof r[c]==='number'?r[c].toFixed(c==='n'?0:2):r[c])+'</td>').join('')+'</tr>').join('')}
document.getElementById('h').onchange=render;render();</script>"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(analyze(RUNS / args.run_id))
