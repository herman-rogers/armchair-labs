"""Held-out feature diagnostics and controlled representation/source comparisons."""

from __future__ import annotations

import argparse
import gc
import json

import joblib
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .analyze import KEY, metric_values
from .data import load_dataset, source_family
from .run import COUNT_STATS, RUNS, STATS, json_write, stat_arrays, verify_checkpoint


def predict_totals(model, representation, values, exposure, target):
    transformed = representation.transform(values)
    prediction = np.asarray(model.predict(transformed)).reshape(-1)
    if target in COUNT_STATS:
        prediction = np.maximum(0, prediction)
    return prediction * exposure


def controlled_comparisons(out, config):
    rows = pl.read_parquet(out / "analysis/predictions.parquet").filter(
        pl.col("year").is_in(config["evaluation_years"])
    )
    recipes = config["recipes"]
    comparisons = []
    for recipe in recipes:
        control = next(
            (
                r
                for r in recipes
                if r["family"] == recipe["family"]
                and r["representation"]["kind"] == "raw"
                and r["view"] == "all"
                and not r.get("target_mode")
                and not r["history"]
            ),
            None,
        )
        if control is None or recipe["id"] == control["id"]:
            continue
        # Hyperparameter draws are comparisons of entire recipes; factorial reps keep knobs fixed.
        left = rows.filter(pl.col("recipe") == recipe["id"])
        right = rows.filter(pl.col("recipe") == control["id"]).select(
            *KEY, pl.col("prediction").alias("control_prediction")
        )
        matched = left.join(right, on=KEY, validate="1:1").filter(pl.col("actual").is_finite())
        for (horizon, target), group in matched.group_by("horizon", "target"):
            y = group["actual"].to_numpy()
            m = metric_values(y, group["prediction"].to_numpy())
            c = metric_values(y, group["control_prediction"].to_numpy())
            comparisons.append(
                {
                    "horizon": horizon,
                    "target": target,
                    "recipe": recipe["id"],
                    "control": control["id"],
                    "family": recipe["family"],
                    "representation": recipe["representation"]["kind"],
                    "view": recipe["view"],
                    "n": m["n"],
                    "rmse": m["rmse"],
                    "control_rmse": c["rmse"],
                    "mse_improvement_pct": 100 * (1 - m["mse"] / c["mse"]) if c["mse"] else None,
                    "controlled_representation": recipe["params"] == control["params"]
                    and recipe["view"] == "all"
                    and not recipe["history"]
                    and not recipe.get("target_mode"),
                }
            )
    schema = {
        name: pl.String
        for name in ["horizon", "target", "recipe", "control", "family", "representation", "view"]
    }
    schema.update(
        n=pl.Int64,
        rmse=pl.Float64,
        control_rmse=pl.Float64,
        mse_improvement_pct=pl.Float64,
        controlled_representation=pl.Boolean,
    )
    table = (
        pl.DataFrame(comparisons, infer_schema_length=None)
        if comparisons
        else pl.DataFrame(schema=schema)
    )
    table.write_parquet(out / "analysis/controlled_comparisons.parquet")


def known_usage_cohorts(out, config):
    """Evaluate ex-ante population slices; future production never defines membership."""
    predictions = pl.read_parquet(out / "analysis/predictions.parquet").filter(
        pl.col("recipe").is_in(
            ["selected", "ensemble_top3", "ensemble_diverse", "ensemble_convex", "persistence"]
        )
    )
    tagged, thresholds = [], []
    keys = ["horizon", "year", "origin", "player_id"]
    for horizon in config["horizons"]:
        dataset = load_dataset(horizon, tuple(config["origins"]), config["input_cache"])
        panel = dataset.panel
        years = panel["year"].to_numpy()
        rate = panel["history_carries"].to_numpy() + panel["history_targets"].to_numpy()
        rate = rate / panel["history_weeks"].to_numpy()
        for year in config["evaluation_years"]:
            earlier = rate[(years < year) & np.isfinite(rate) & (rate > 0)]
            cutoff = float(np.quantile(earlier, 0.75)) if len(earlier) else None
            test = years == year
            values = rate[test]
            labels = np.where(
                (values >= cutoff) if cutoff is not None else np.zeros(len(values), dtype=bool),
                "high_prior_usage",
                np.where(values > 0, "lower_prior_usage", "no_recorded_prior_usage"),
            )
            labels = np.where(np.isfinite(values), labels, "unknown_prior_usage")
            tagged.append(
                panel.filter(pl.Series(test))
                .select(keys)
                .with_columns(pl.Series("usage_cohort", labels))
            )
            thresholds.append(
                {
                    "horizon": horizon,
                    "year": year,
                    "high_usage_cutoff_opportunities_per_week": cutoff,
                    "training_max_year": year - 1,
                }
            )
    tags = pl.concat(tagged)
    combined = predictions.join(tags, on=keys, how="inner", validate="m:1")
    baseline = combined.filter(pl.col("recipe") == "persistence").select(
        *KEY, pl.col("prediction").alias("baseline")
    )
    combined = combined.join(baseline, on=KEY, how="inner", validate="m:1").filter(
        pl.col("actual").is_finite()
        & pl.col("prediction").is_finite()
        & pl.col("baseline").is_finite()
    )
    rows = []
    for key, group in combined.group_by("horizon", "target", "recipe", "usage_cohort"):
        actual = group["actual"].to_numpy()
        scored = metric_values(actual, group["prediction"].to_numpy())
        control = metric_values(actual, group["baseline"].to_numpy())
        rows.append(
            {
                "horizon": key[0],
                "target": key[1],
                "recipe": key[2],
                "cohort": key[3],
                **scored,
                "baseline_rmse": control["rmse"],
                "mse_improvement_pct": 100 * (1 - scored["mse"] / control["mse"])
                if control["mse"]
                else None,
            }
        )
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        out / "analysis/usage_cohorts.parquet"
    )
    pl.DataFrame(thresholds).write_parquet(out / "analysis/usage_cohort_thresholds.parquet")


def permutation_diagnostics(out, config, repeats=3):
    selected = pl.read_parquet(out / "analysis/policy_selection.parquet")
    final_year = max(config["years"])
    selected = selected.filter((pl.col("year") == final_year) & (pl.col("policy") == "selected"))
    records, importance = [], []
    rng = np.random.default_rng(config["seed"])
    for horizon in config["horizons"]:
        dataset = load_dataset(horizon, tuple(config["origins"]), config["input_cache"])
        test = dataset.panel["year"].to_numpy() == final_year
        targets, _ = stat_arrays(dataset)
        for choice in selected.filter(pl.col("horizon") == horizon).iter_rows(named=True):
            recipe = choice["members"][0]
            path = out / horizon / "fits" / recipe / str(final_year)
            if not (path / "complete.json").exists():
                continue  # Training-mean baseline has no estimator.
            verify_checkpoint(path)
            target = choice["target"]
            names = json.loads((path / "feature_order.json").read_text())
            index = np.array([dataset.names.index(n) for n in names])
            x = dataset.x[test][:, index]
            representation = joblib.load(path / "preprocessing.joblib")
            model = joblib.load(path / f"{target}.joblib")

            with threadpool_limits(limits=config["threads"]):
                y = targets[test, STATS.index(target)]
                replayed = predict_totals(model, representation, x, dataset.exposure[test], target)
                forecast_keys = (
                    dataset.panel.filter(pl.Series(test))
                    .select([key for key in KEY if key != "target"])
                    .with_columns(pl.lit(target).alias("target"))
                )
                saved = (
                    pl.read_parquet(path / "predictions.parquet")
                    .filter(pl.col("target") == target)
                    .select(*KEY, "prediction")
                )
                aligned = forecast_keys.join(
                    saved, on=KEY, how="left", validate="1:1", maintain_order="left"
                )
                np.testing.assert_allclose(
                    replayed, aligned["prediction"].to_numpy(), rtol=1e-5, atol=1e-5
                )
                original = metric_values(y, replayed)["mse"]
                if original is None:
                    continue
                families = np.array([source_family(n) for n in names])
                for family in sorted(set(families)):
                    columns = families == family
                    for repeat in range(repeats):
                        perturbed = x.copy()
                        perturbed[:, columns] = x[rng.permutation(len(x))][:, columns]
                        mse = metric_values(
                            y,
                            predict_totals(
                                model, representation, perturbed, dataset.exposure[test], target
                            ),
                        )["mse"]
                        records.append(
                            {
                                "horizon": horizon,
                                "target": target,
                                "year": final_year,
                                "recipe": recipe,
                                "family": family,
                                "repeat": repeat,
                                "features": int(columns.sum()),
                                "baseline_mse": original,
                                "permuted_mse": mse,
                                "delta_mse": mse - original,
                            }
                        )
                native = getattr(model, "feature_importances_", None)
                if native is not None:
                    values = np.asarray(native).reshape(-1)
                    transformed_names = representation.get_feature_names_out()
                    for i in np.argsort(values)[-50:][::-1]:
                        importance.append(
                            {
                                "horizon": horizon,
                                "target": target,
                                "recipe": recipe,
                                "feature": str(transformed_names[i]),
                                "importance": float(values[i]),
                                "note": "Native training importance; not causal",
                            }
                        )
            del model, representation
            gc.collect()
            print(f"Permutation diagnostic {horizon}/{target}/{recipe}", flush=True)
    record_schema = {
        "horizon": pl.String,
        "target": pl.String,
        "year": pl.Int64,
        "recipe": pl.String,
        "family": pl.String,
        "repeat": pl.Int64,
        "features": pl.Int64,
        "baseline_mse": pl.Float64,
        "permuted_mse": pl.Float64,
        "delta_mse": pl.Float64,
    }
    importance_schema = {
        "horizon": pl.String,
        "target": pl.String,
        "recipe": pl.String,
        "feature": pl.String,
        "importance": pl.Float64,
        "note": pl.String,
    }
    (
        pl.DataFrame(records, infer_schema_length=None)
        if records
        else pl.DataFrame(schema=record_schema)
    ).write_parquet(out / "analysis/source_permutation.parquet")
    (
        pl.DataFrame(importance, infer_schema_length=None)
        if importance
        else pl.DataFrame(schema=importance_schema)
    ).write_parquet(out / "analysis/native_importance.parquet")
    json_write(
        out / "analysis/diagnostic_notes.json",
        {
            "selection": "Models selected using earlier three seasons only",
            "artifact_replay": (
                "Loaded pipelines reproduce saved selected-model forecasts before permutation"
            ),
            "interpretation": "Held-out diagnostics only; no same-fold feature reselection",
            "permutation": "All columns of a source jointly permuted across held-out rows",
            "limitations": "Correlated source groups can substitute; negative effects and unstable "
            "estimates are expected. Native importance is training-biased.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = RUNS / args.run_id
    config = json.loads((out / "config.json").read_text())
    controlled_comparisons(out, config)
    known_usage_cohorts(out, config)
    permutation_diagnostics(out, config)
