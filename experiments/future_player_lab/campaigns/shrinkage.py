"""Post-hoc QB preseason development diagnostic; never modifies frozen forecasts.

Select a convex mix of the raw XGBoost anchor and affine persistence using only
three earlier validation years, globally or by previously known usage cohort.
No new tree models are fit. This question was prompted by the main evaluation,
so its 2021–2025 results are further development evidence, not independent tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from ..gpu_lab.calibrated_baselines import AffinePersistence
from ..gpu_lab.run import digest, json_write
from .data import ROOT
from .run import EVALUATION_YEARS, YEARS, baseline_values, metrics

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ANCHOR = "xgb_raw_draft"


def select_alpha(history, year, cohort=None):
    """Equal-year loss selection; current/future rows never enter the choice."""
    previous = history.filter(pl.col("year").is_between(year - 3, year - 1))
    if cohort is not None:
        previous = previous.filter(pl.col("high_prior_usage") == cohort)
    if set(previous["year"].to_list()) != set(range(year - 3, year)):
        raise ValueError("Incomplete earlier selection years or empty usage cohort")
    candidates = []
    for alpha in ALPHAS:
        annual = []
        for block in previous.partition_by("year"):
            prediction = alpha * block["raw"].to_numpy() + (1 - alpha) * block["affine"].to_numpy()
            loss = metrics(block["actual"].to_numpy(), prediction)["mse"]
            if loss is None or not np.isfinite(loss):
                raise ValueError("Unknown shrinkage validation loss")
            annual.append(loss)
        candidates.append((float(np.mean(annual)), alpha))
    loss, alpha = min(candidates)
    return alpha, loss


def diagnostic(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(f"Immutable diagnostic already exists: {output}")
    cell = run / "qb_season"
    manifest = json.loads((cell / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(cell / name) != sha:
            raise ValueError(f"Model artifact changed: {name}")
    dataset = run / "inputs/qb_season"
    meta_path = dataset / "dataset.json"
    if digest(meta_path) != manifest["dataset_manifest_sha256"]:
        raise ValueError("Dataset provenance mismatch")
    meta = json.loads(meta_path.read_text())
    if digest(dataset / "panel.parquet") != meta["panel_sha256"]:
        raise ValueError("Historical panel checksum mismatch")
    panel = pl.read_parquet(dataset / "panel.parquet")
    panel = panel.with_columns(
        ((pl.col("history_attempts") + pl.col("history_carries")) / pl.col("history_weeks")).alias(
            "known_usage"
        )
    )
    years = panel["year"].to_numpy()
    y = panel["y_points"].to_numpy()
    baseline = baseline_values(panel, "QB", "points")
    pieces, thresholds = [], []
    checkpoint_hashes = {}
    for year in YEARS:
        training, testing = years < year, years == year
        calibration = AffinePersistence().fit(baseline[training], y[training])
        affine = calibration.predict(baseline[testing])
        path = cell / "checkpoints" / f"points_{year}_{ANCHOR}.npy"
        receipt = json.loads(path.with_suffix(".json").read_text())
        if digest(path) != receipt["sha256"]:
            raise ValueError("Raw-anchor checkpoint mismatch")
        raw = np.load(path, allow_pickle=False)
        if raw.shape != affine.shape or not np.isfinite(raw).all():
            raise ValueError("Raw-anchor checkpoint row shape mismatch")
        checkpoint_hashes[str(path.relative_to(run))] = receipt["sha256"]
        previous_usage = panel.filter(
            (pl.col("year") < year)
            & pl.col("known_usage").is_finite()
            & (pl.col("known_usage") > 0)
        )["known_usage"].to_numpy()
        threshold = float(np.quantile(previous_usage, 0.75))
        thresholds.append({"year": year, "threshold": threshold, "training_max_year": year - 1})
        pieces.append(
            panel.filter(pl.col("year") == year)
            .select("year", "player_id", "player_display_name", "known_usage")
            .with_columns(
                pl.Series("actual", y[testing]),
                pl.Series("raw", raw),
                pl.Series("affine", affine),
                (pl.col("known_usage") >= threshold).fill_null(False).alias("high_prior_usage"),
            )
        )
    history = pl.concat(pieces)
    main = pl.read_parquet(cell / "evaluation_predictions.parquet").filter(
        pl.col("target") == "points"
    )
    # Verify reconstructed controls reproduce their already-frozen OOF forecasts.
    for key, recipe in (("raw", ANCHOR), ("affine", "affine")):
        comparison = history.filter(pl.col("year").is_in(EVALUATION_YEARS)).join(
            main.filter(pl.col("recipe") == recipe).select("year", "player_id", "prediction"),
            on=["year", "player_id"],
            how="inner",
            validate="1:1",
        )
        np.testing.assert_allclose(
            comparison[key].to_numpy(), comparison["prediction"].to_numpy(), rtol=1e-12, atol=1e-12
        )
    predictions, choices = [], []
    for year in EVALUATION_YEARS:
        block = history.filter(pl.col("year") == year)
        alpha, loss = select_alpha(history, year)
        by_cohort = {cohort: select_alpha(history, year, cohort) for cohort in (False, True)}
        cohort_alpha = np.where(
            block["high_prior_usage"].to_numpy(), by_cohort[True][0], by_cohort[False][0]
        )
        raw, affine = block["raw"].to_numpy(), block["affine"].to_numpy()
        block = block.with_columns(
            pl.Series("global_blend", alpha * raw + (1 - alpha) * affine),
            pl.Series("cohort_blend", cohort_alpha * raw + (1 - cohort_alpha) * affine),
        ).join(
            main.filter((pl.col("year") == year) & (pl.col("recipe") == "ensemble")).select(
                "year", "player_id", pl.col("prediction").alias("frozen_ensemble")
            ),
            on=["year", "player_id"],
            how="left",
            validate="1:1",
        )
        predictions.append(block)
        choices.append(
            {
                "year": year,
                "selection_years": list(range(year - 3, year)),
                "global_alpha": alpha,
                "global_selection_mse": loss,
                "high_usage_alpha": by_cohort[True][0],
                "other_alpha": by_cohort[False][0],
            }
        )
    predictions = pl.concat(predictions)
    summary, rankings = [], []
    for recipe in ("affine", "raw", "frozen_ensemble", "global_blend", "cohort_blend"):
        for cohort, selected in (
            ("all_candidates", predictions),
            ("high_prior_usage", predictions.filter("high_prior_usage")),
            ("other", predictions.filter(~pl.col("high_prior_usage"))),
        ):
            groups = [
                (None, selected),
                *[
                    (int(key[0]), block)
                    for key, block in selected.partition_by("year", as_dict=True).items()
                ],
            ]
            for year, block in groups:
                summary.append(
                    {
                        "recipe": recipe,
                        "cohort": cohort,
                        "year": year,
                        **metrics(block["actual"].to_numpy(), block[recipe].to_numpy()),
                    }
                )
        for (year,), block in predictions.partition_by("year", as_dict=True).items():
            selected = block.sort([recipe, "player_id"], descending=[True, False]).head(12)
            rankings.append(
                {
                    "year": year,
                    "recipe": recipe,
                    "selected_top12_actual_points": float(selected["actual"].sum()),
                }
            )
    output.mkdir(parents=True, exist_ok=False)
    history.write_parquet(output / "validation_predictions.parquet")
    predictions.write_parquet(output / "evaluation_predictions.parquet")
    report = {
        "schema_version": 1,
        "kind": "post_hoc_development_diagnostic",
        "campaign_id": "opening_drive_v1",
        "position": "QB",
        "horizon": "season",
        "target": "points",
        "motivation": (
            "Main model improved all-candidate errors but weakened "
            "high-usage/top12 QB preseason results."
        ),
        "alpha_grid": ALPHAS,
        "alpha_meaning": "Weight on raw XGBoost; (1-alpha) on affine persistence",
        "cohort_definition": (
            "Prior (attempts+carries)/calendar-week >= upper quartile "
            "of positive rates in strictly earlier training seasons"
        ),
        "selection": "Minimum equal-year MSE in previous3years; ties prefer smaller raw weight",
        "new_tree_fits": 0,
        "replaces_primary_models": False,
        "limitation": (
            "Designed after inspecting these development seasons; "
            "not an untouched holdout or a promotion decision."
        ),
        "thresholds": thresholds,
        "choices": choices,
        "rows": summary,
        "rankings": rankings,
    }
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    json_write(output / "metrics.json", report)
    json_write(
        output / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "kind": "post_hoc_development_diagnostic",
            "source_sha256": digest(Path(__file__)),
            "model_manifest_sha256": digest(cell / "manifest.json"),
            "dataset_manifest_sha256": digest(meta_path),
            "checkpoint_hashes": checkpoint_hashes,
            "files": {path.name: digest(path) for path in output.iterdir() if path.is_file()},
        },
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="opening_drive_v1_001")
    parser.add_argument("--output-name", default="qb_preseason_shrinkage_002")
    args = parser.parse_args()
    run = ROOT / "experiments/future_player_lab/runs" / args.run_id
    with threadpool_limits(limits=4):
        report = diagnostic(run, run / "diagnostics" / args.output_name)
    print(json.dumps([r for r in report["rows"] if r["year"] is None], indent=2))


if __name__ == "__main__":
    main()
