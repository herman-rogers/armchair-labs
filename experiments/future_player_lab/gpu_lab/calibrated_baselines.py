"""Post-preview sensitivity study: two fixed chronological affine persistence baselines.

Each fit minimizes unweighted squared error in total production over player-origin
rows. Only the scalar raw persistence forecast enters the affine equation; prior
usage defines reporting cohorts but never enters a forecast or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .analyze import choose_members, scores
from .data import load_dataset
from .run import COUNT_STATS, RUNS, STATS, digest, json_write, packages, stat_arrays

KEY = ["horizon", "year", "origin", "player_id", "target"]
VARIANTS = {"affine_expanding": 0, "affine_recent3": 3}
POLICIES = ["selected", "ensemble_top3", "ensemble_diverse", "ensemble_convex"]
YEARS = list(range(2018, 2026))
EVALUATION_YEARS = list(range(2021, 2026))


class AffinePersistence:
    """Constrained least squares with unrestricted intercept and nonnegative slope."""

    def fit(self, predictor, actual):
        x, y = np.asarray(predictor, float), np.asarray(actual, float)
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError("Predictor and outcome must be aligned one-dimensional arrays")
        known = np.isfinite(x) & np.isfinite(y)
        if not known.any():
            raise ValueError("No observed predictor/outcome pairs")
        x, y = x[known], y[known]
        centered = x - x.mean()
        denominator = float(centered @ centered)
        self.slope_ = (
            max(0.0, float(centered @ (y - y.mean())) / denominator) if denominator else 0.0
        )
        self.intercept_ = float(y.mean() - self.slope_ * x.mean())
        self.training_rows_ = int(known.sum())
        self.dropped_rows_ = int((~known).sum())
        return self

    def predict(self, predictor, *, count_stat=False):
        x = np.asarray(predictor, float)
        prediction = self.intercept_ + self.slope_ * np.where(np.isfinite(x), x, np.nan)
        return np.maximum(prediction, 0) if count_stat else prediction


def fit_fold(years, predictor, actual, year, window=0):
    years = np.asarray(years)
    training = (years < year) & ((years >= year - window) if window else True)
    model = AffinePersistence().fit(np.asarray(predictor)[training], np.asarray(actual)[training])
    observed = training & np.isfinite(predictor) & np.isfinite(actual)
    return model, {
        "training_min_year": int(years[observed].min()),
        "training_max_year": int(years[observed].max()),
        "training_rows": model.training_rows_,
        "dropped_training_rows": model.dropped_rows_,
        "intercept": model.intercept_,
        "slope": model.slope_,
    }


def select_affine(predictions, years):
    annual = scores(predictions, ["horizon", "target", "recipe", "year"])
    selected, choices = [], []
    for (horizon, target), group in annual.filter(pl.col("recipe").is_in(VARIANTS)).group_by(
        "horizon", "target"
    ):
        for year in years:
            ranked = choose_members(group, year, lookback=3, count=1)
            if not ranked:
                raise ValueError(
                    f"Incomplete earlier affine validation history: {horizon}/{target}/{year}"
                )
            winner = ranked[0]
            choices.append(
                {
                    "horizon": horizon,
                    "target": target,
                    "year": year,
                    "member": winner["recipe"],
                    "selection_loss": winner["selection_loss"],
                    "selection_min_year": year - 3,
                    "selection_max_year": year - 1,
                }
            )
            selected.append(
                predictions.filter(
                    (pl.col("horizon") == horizon)
                    & (pl.col("target") == target)
                    & (pl.col("year") == year)
                    & (pl.col("recipe") == winner["recipe"])
                ).with_columns(pl.lit("affine_selected").alias("recipe"))
            )
    return pl.concat(selected), pl.DataFrame(choices), annual


def usage_tags(panel, years):
    seasons = panel["year"].to_numpy()
    denominator = panel["history_weeks"].to_numpy()
    numerator = (panel["history_carries"] + panel["history_targets"]).to_numpy()
    rate = np.divide(numerator, denominator, out=np.full(len(panel), np.nan), where=denominator > 0)
    tags, thresholds = [], []
    for year in years:
        prior = rate[(seasons < year) & np.isfinite(rate) & (rate > 0)]
        cutoff = float(np.quantile(prior, 0.75)) if len(prior) else None
        test = seasons == year
        high = rate[test] >= cutoff if cutoff is not None else np.zeros(test.sum(), bool)
        labels = np.where(
            high,
            "high_prior_usage",
            np.where(rate[test] > 0, "lower_prior_usage", "no_recorded_prior_usage"),
        )
        labels = np.where(np.isfinite(rate[test]), labels, "unknown_prior_usage")
        tags.append(
            panel.filter(pl.Series(test))
            .select(KEY[:-1])
            .with_columns(pl.Series("usage_cohort", labels))
        )
        thresholds.append(
            {
                "horizon": panel["horizon"][0],
                "year": year,
                "training_max_year": year - 1,
                "cutoff": cutoff,
            }
        )
    return pl.concat(tags), pl.DataFrame(thresholds)


def compare_predictions(predictions, controls, extra_groups=()):
    """Compare each recipe/control on identical finite outcome/predictor rows."""
    if predictions.select([*KEY, "recipe"]).is_duplicated().any():
        raise ValueError("Duplicate forecast keys")
    valid = predictions.filter(pl.col("actual").is_finite() & pl.col("prediction").is_finite())
    groups = ["horizon", "target", "recipe", *extra_groups]
    results = []
    for control in controls:
        baseline = valid.filter(pl.col("recipe") == control).select(
            *KEY,
            pl.col("actual").alias("baseline_actual"),
            pl.col("prediction").alias("baseline_prediction"),
        )
        pairs = valid.filter(pl.col("recipe") != control).join(
            baseline, on=KEY, how="inner", validate="m:1"
        )
        if pairs.filter((pl.col("actual") - pl.col("baseline_actual")).abs() > 1e-8).height:
            raise ValueError("Matched outcomes disagree")
        for keys, frame in pairs.sort(groups).group_by(groups, maintain_order=True):
            error = (frame["prediction"] - frame["actual"]).to_numpy()
            base_error = (frame["baseline_prediction"] - frame["actual"]).to_numpy()
            record = {
                **dict(zip(groups, keys, strict=True)),
                "baseline": control,
                "n": len(error),
                "mse": float(np.mean(error**2)),
                "baseline_mse": float(np.mean(base_error**2)),
                "mae": float(np.mean(abs(error))),
                "baseline_mae": float(np.mean(abs(base_error))),
                "bias": float(error.mean()),
                "baseline_bias": float(base_error.mean()),
            }
            record.update(
                rmse=float(np.sqrt(record["mse"])),
                baseline_rmse=float(np.sqrt(record["baseline_mse"])),
                delta_mse=record["mse"] - record["baseline_mse"],
                delta_mae=record["mae"] - record["baseline_mae"],
                mse_improvement_pct=100 * (1 - record["mse"] / record["baseline_mse"])
                if record["baseline_mse"]
                else None,
            )
            annual = (
                frame.with_columns(
                    pl.Series("delta_se", error**2 - base_error**2),
                    pl.Series("delta_ae", abs(error) - abs(base_error)),
                )
                .group_by("year")
                .agg(pl.col("delta_se").mean(), pl.col("delta_ae").mean())
                .sort("year")
            )
            seed = int.from_bytes(
                hashlib.sha256(repr((keys, control)).encode()).digest()[:4], "little"
            )
            rng = np.random.default_rng(seed)
            sample = rng.integers(0, len(annual), size=(3000, len(annual)))
            record["seasons"] = len(annual)
            record["positive_skill_seasons"] = int((annual["delta_se"] < 0).sum())
            for metric, column in (("mse", "delta_se"), ("mae", "delta_ae")):
                values = annual[column].to_numpy()
                record[f"mean_season_delta_{metric}"] = float(values.mean())
                bounds = (
                    np.quantile(values[sample].mean(axis=1), [0.025, 0.975])
                    if len(annual) > 1
                    else [None, None]
                )
                record[f"delta_{metric}_low95"], record[f"delta_{metric}_high95"] = bounds
            results.append(record)
    return pl.DataFrame(results, infer_schema_length=None)


def write_comparisons(out, predictions, tags):
    controls = ["persistence", *VARIANTS, "affine_selected"]
    compare_predictions(predictions, controls).write_parquet(out / "matched_comparisons.parquet")
    compare_predictions(predictions, controls, ("year",)).write_parquet(
        out / "annual_comparisons.parquet"
    )
    tagged = predictions.join(tags, on=KEY[:-1], how="left", validate="m:1")
    if tagged["usage_cohort"].null_count():
        raise ValueError("A prediction lacks its prior-input usage cohort")
    compare_predictions(tagged, controls, ("usage_cohort",)).write_parquet(
        out / "cohort_comparisons.parquet"
    )
    compare_predictions(tagged, controls, ("year", "usage_cohort")).write_parquet(
        out / "annual_cohort_comparisons.parquet"
    )


def report(out, title):
    rows = (
        pl.read_parquet(out / "matched_comparisons.parquet")
        .filter(
            pl.col("target").is_in(["points", "rushing_yards", "receiving_yards", "receptions"])
            & pl.col("baseline").is_in(["persistence", "affine_selected"])
        )
        .sort("horizon", "target", "baseline", "recipe")
    )
    lines = [
        f"# {title}",
        "",
        "This baseline-strength sensitivity check was specified after seeing the preseason "
        "tree preview. It neither changes nor promotes any recipe in the frozen tree campaign. "
        "Historical development evidence; "
        "2025 was already examined in earlier research.",
        "",
        "Two fixed baselines fit production = intercept + nonnegative slope × raw persistence. "
        "Expanding and previous-three-season fits give every observed player-origin pair "
        "equal weight in total-production squared error. Missing predictor/outcome pairs are "
        "excluded per target, unknown predictors stay unknown, and only count predictions "
        "are clipped to zero. No extra covariates enter. The affine_selected policy chooses "
        "between them using the equal-season mean normalized MSE of "
        "strictly earlier three-year out-of-season forecasts.",
        "",
        "Pooled-row MSE/MAE differences compare identical rows. Bootstrap intervals resample "
        "the five "
        "annual mean differences, a distinct equal-season estimand with coarse uncertainty. "
        "Usage cohorts use a threshold from earlier input history; they do not use "
        "future production. "
        "No calibrated predictive intervals are claimed.",
        "",
        "| Horizon | Target | Recipe | Control | N | RMSE / control | MSE improvement % "
        "| MAE difference | Positive years |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows.iter_rows(named=True):
        skill = "—" if r["mse_improvement_pct"] is None else f"{r['mse_improvement_pct']:.2f}"
        lines.append(
            f"| {r['horizon']} | {r['target']} | {r['recipe']} | {r['baseline']} | {r['n']} | "
            f"{r['rmse']:.3f} / {r['baseline_rmse']:.3f} | {skill} | {r['delta_mae']:.3f} | "
            f"{r['positive_skill_seasons']}/{r['seasons']} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "readout.md").write_text(text)
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>'
        + html.escape(title)
        + "</title><style>body{font:15px system-ui;margin:30px;max-width:1500px}"
        + "pre{white-space:pre-wrap;line-height:1.5}</style>"
        + '<p><a href="matched_comparisons.parquet">Matched data</a> · '
        + '<a href="annual_comparisons.parquet">Annual data</a> · '
        + '<a href="cohort_comparisons.parquet">Cohort data</a></p><pre>'
        + html.escape(text)
        + "</pre>"
    )


def seal(out, **metadata):
    json_write(
        out / "manifest.json",
        {
            "status": "complete",
            **metadata,
            "files": {
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p != out / "manifest.json"
            },
        },
    )


def build(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "horizons": ["season", "remaining"],
        "years": YEARS,
        "evaluation_years": EVALUATION_YEARS,
        "targets": STATS,
        "variants": VARIANTS,
        "selection_years": 3,
        "origins": [2],
        "fit_weighting": "Equal observed player-origin weight, total-production squared error",
        "selection_weighting": "Equal season mean of annual MSE / annual outcome variance",
        "specified_after_preseason_preview": True,
        "frozen_tree_campaign_modified": False,
        "predictor": "Scalar raw prior-production persistence only",
        "predictive_intervals": False,
    }
    json_write(out / "config.json", config)
    json_write(out / "manifest.json", {"status": "running"})
    forecasts, coefficients, latest, tags, thresholds, inputs = [], [], [], [], [], []
    for horizon in config["horizons"]:
        dataset = load_dataset(horizon)
        actual, predictor = stat_arrays(dataset)
        seasons = dataset.panel["year"].to_numpy()
        path = Path(dataset.provenance["cache_directory"]) / "dataset.json"
        inputs.append(
            {
                "horizon": horizon,
                "dataset_manifest": str(path),
                "dataset_manifest_sha256": digest(path),
                "provenance": dataset.provenance,
            }
        )
        cohort, cutoff = usage_tags(dataset.panel, EVALUATION_YEARS)
        tags.append(cohort)
        thresholds.append(cutoff)
        for year in YEARS:
            test = seasons == year
            keys = dataset.panel.filter(pl.Series(test)).select(KEY[:-1])
            for j, target in enumerate(STATS):
                records = {"persistence": predictor[test, j]}
                for recipe, window in VARIANTS.items():
                    model, metadata = fit_fold(seasons, predictor[:, j], actual[:, j], year, window)
                    coefficients.append(
                        {
                            "horizon": horizon,
                            "target": target,
                            "year": year,
                            "recipe": recipe,
                            **metadata,
                        }
                    )
                    records[recipe] = model.predict(
                        predictor[test, j], count_stat=target in COUNT_STATS
                    )
                for recipe, prediction in records.items():
                    forecasts.append(
                        keys.with_columns(
                            pl.lit(target).alias("target"),
                            pl.lit(recipe).alias("recipe"),
                            pl.Series("actual", actual[test, j]),
                            pl.Series("prediction", prediction),
                        )
                    )
        future_year = int(seasons.max()) + 1
        for j, target in enumerate(STATS):
            for recipe, window in VARIANTS.items():
                _, metadata = fit_fold(seasons, predictor[:, j], actual[:, j], future_year, window)
                latest.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "year": future_year,
                        "recipe": recipe,
                        **metadata,
                    }
                )
        print(
            f"Fitted {horizon}: two scalar baselines across eight targets/eight years", flush=True
        )
    predictions = pl.concat(forecasts)
    selected, choices, annual = select_affine(predictions, EVALUATION_YEARS)
    _, future_choices, _ = select_affine(predictions, [2026])
    pl.DataFrame(latest).join(
        future_choices.select("horizon", "target", "year", "member"),
        on=["horizon", "target", "year"],
        validate="m:1",
    ).with_columns(
        (pl.col("recipe") == pl.col("member")).alias("selected_for_future")
    ).write_parquet(out / "latest_coefficients.parquet")
    future_choices.write_parquet(out / "latest_selection.parquet")
    predictions = pl.concat([predictions, selected])
    predictions.write_parquet(out / "predictions.parquet")
    pl.DataFrame(coefficients).write_parquet(out / "coefficients.parquet")
    choices.write_parquet(out / "policy_selection.parquet")
    annual.write_parquet(out / "annual_metrics.parquet")
    tag_frame = pl.concat(tags)
    tag_frame.write_parquet(out / "usage_tags.parquet")
    pl.concat(thresholds).write_parquet(out / "usage_thresholds.parquet")
    json_write(out / "inputs.json", inputs)
    json_write(out / "packages.json", packages())
    sources = out / "implementation"
    sources.mkdir()
    for filename in ("calibrated_baselines.py", "analyze.py", "data.py", "run.py"):
        shutil.copyfile(Path(__file__).parent / filename, sources / filename)
    write_comparisons(out, predictions.filter(pl.col("year").is_in(EVALUATION_YEARS)), tag_frame)
    report(out, "Stronger simple RB baselines")
    seal(
        out,
        fit_count=len(coefficients),
        latest_fit_count=len(latest),
        note="Fixed post-preview sensitivity check; no changes to the frozen tree library",
    )
    return out


def integration_ready(main):
    path = Path(main) / "analysis/manifest.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    return manifest.get("status") in {
        "core_complete",
        "complete",
    } and "predictions.parquet" in manifest.get("files", {})


def integrate(out, main):
    out, main = Path(out), Path(main)
    if not integration_ready(main):
        raise ValueError("A sealed complete/core_complete analysis is required")
    own = json.loads((out / "manifest.json").read_text())
    if own.get("status") != "complete":
        raise ValueError("The affine study is not complete")
    for name in ("predictions.parquet", "usage_tags.parquet", "inputs.json"):
        if digest(out / name) != own["files"][name]:
            raise ValueError(f"Affine study artifact changed: {name}")
    source = json.loads((main / "analysis/manifest.json").read_text())
    prediction_path = main / "analysis/predictions.parquet"
    if digest(prediction_path) != source["files"]["predictions.parquet"]:
        raise ValueError("Source forecast hash mismatch")
    config = json.loads((main / "config.json").read_text())
    for item in json.loads((out / "inputs.json").read_text()):
        if item["horizon"] in config["horizons"]:
            main_input = json.loads((main / item["horizon"] / "inputs.json").read_text())
            if main_input["provenance"] != item["provenance"]:
                raise ValueError(
                    "Source forecasts and affine baselines used different raw datasets"
                )
    trees = (
        pl.read_parquet(prediction_path)
        .filter(
            pl.col("recipe").is_in(POLICIES)
            & pl.col("target").is_in(STATS)
            & pl.col("year").is_in(EVALUATION_YEARS)
        )
        .select(*KEY, "recipe", "actual", "prediction")
    )
    affine = (
        pl.read_parquet(out / "predictions.parquet")
        .filter(
            pl.col("horizon").is_in(config["horizons"]) & pl.col("year").is_in(EVALUATION_YEARS)
        )
        .select(*KEY, "recipe", "actual", "prediction")
    )
    dest = out / "comparisons" / main.name
    dest.mkdir(parents=True, exist_ok=False)
    combined = pl.concat([trees, affine], how="vertical_relaxed")
    combined.write_parquet(dest / "predictions.parquet")
    write_comparisons(dest, combined, pl.read_parquet(out / "usage_tags.parquet"))
    report(dest, "Do tree gains survive affine persistence recalibration?")
    shutil.copyfile(__file__, dest / "integration_source.py")
    seal(
        dest,
        source_run=str(main.resolve()),
        source_predictions_sha256=digest(prediction_path),
        source_affine_predictions_sha256=digest(out / "predictions.parquet"),
        source_affine_manifest_sha256=digest(out / "manifest.json"),
    )
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rb_gpu_calibrated_baselines_20261003_001")
    parser.add_argument("--integrate", metavar="SOURCE_RUN_ID")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        if args.integrate:
            source_run = RUNS / args.integrate
            if args.wait:
                print(f"Waiting for sealed analysis from {source_run}", flush=True)
                while not integration_ready(source_run):
                    time.sleep(30)
            print(integrate(RUNS / args.run_id, source_run))
        else:
            print(build(RUNS / args.run_id))
