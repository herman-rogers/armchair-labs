"""Reproduce the saved QB yardage challenger and diagnose its scope without promotion."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from engine.data.nextgen import load_analysis
from engine.data.releases import digest, identifier, write_json
from engine.metrics.evidence_diagnostics import passing_workload_diagnostics


def review(data: Path, version: str):
    source, manifest = load_analysis(data)
    output = data / "research" / identifier(version)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = source / "implementation/src/engine/metrics/nextgen.py"
    spec = importlib.util.spec_from_file_location("saved_nextgen_review", snapshot)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    features = pl.read_parquet(source / "features.parquet")
    all_predictions = pl.read_parquet(source / "predictions.parquet")
    predictions = all_predictions.filter(
        (pl.col("target") == "passing_yards") & (pl.col("position") == "QB") & pl.col("complete")
    )
    qb_features = features.filter(pl.col("position") == "QB").to_dicts()
    columns = sorted(c for c in features.columns if c.startswith("x_"))
    reproduction = []
    for year in sorted(predictions["season"].unique()):
        train = [r for r in qb_features if r["forecast_season"] < year and r["outcome_complete"]]
        test = [r for r in qb_features if r["forecast_season"] == year]
        fits, n = module.fit_fold(train, test, "passing_yards", columns)
        for model, values in fits.items():
            saved = predictions.filter((pl.col("season") == year) & (pl.col("model") == model))
            lookup = {r["player_id"]: r["prediction"] for r in saved.to_dicts()}
            actual = np.array([lookup[r["player_id"]] for r in test], dtype=float)
            np.testing.assert_allclose(values, actual, rtol=1e-10, atol=1e-8, equal_nan=True)
            reproduction.append(
                dict(
                    season=int(year),
                    model=model,
                    train_rows=n,
                    test_rows=len(test),
                    max_absolute_difference=float(np.nanmax(np.abs(values - actual))),
                )
            )
        print(f"Reproduced QB passing yards {year}", flush=True)

    comparisons, workloads, populations, direct, related = [], [], [], [], []
    for window, start in (("all_history", 2007), ("modern", 2019)):
        frame = predictions.filter(pl.col("season") >= start)
        for model in ("baseline", "ridge", "boost"):
            subset = frame.filter(pl.col("model") == model)
            comparisons.append(
                dict(window=window, model=model, **module.paired_summary(subset.to_dicts()))
            )
            workloads.append(
                dict(window=window, model=model, **passing_workload_diagnostics(subset, features))
            )
            for population in sorted(subset["population"].unique()):
                rows = subset.filter(pl.col("population") == population).to_dicts()
                populations.append(
                    dict(
                        window=window,
                        model=model,
                        population=population,
                        **module.paired_summary(rows),
                    )
                )
        versus_ridge = (
            frame.filter(pl.col("model") == "boost")
            .drop("baseline")
            .join(
                frame.filter(pl.col("model") == "ridge").select(
                    "player_id", "season", pl.col("prediction").alias("baseline")
                ),
                on=["player_id", "season"],
                validate="1:1",
            )
        )
        direct.append(
            dict(
                window=window,
                model="boost",
                comparator="ridge",
                **module.paired_summary(versus_ridge.to_dicts()),
            )
        )
        for target in ("attempts", "passing_efficiency", "scoring_appearances"):
            for model in ("baseline", "ridge", "boost"):
                subset = all_predictions.filter(
                    (pl.col("target") == target)
                    & (pl.col("position") == "QB")
                    & (pl.col("model") == model)
                    & pl.col("complete")
                    & (pl.col("season") >= start)
                )
                related.append(
                    dict(
                        window=window,
                        target=target,
                        model=model,
                        **module.paired_summary(subset.to_dicts()),
                    )
                )
    examples = predictions.filter(
        ((pl.col("player_display_name") == "Tom Brady") & pl.col("season").is_in([2008, 2009]))
        | ((pl.col("player_display_name") == "Patrick Mahomes") & (pl.col("season") == 2018))
        | ((pl.col("player_display_name") == "Lamar Jackson") & (pl.col("season") == 2019))
    ).select("player_id", "player_display_name", "season", "model", "prediction", "actual")
    boost = predictions.filter(pl.col("model") == "boost").with_columns(
        (pl.col("prediction") - pl.col("actual")).abs().alias("absolute_error")
    )
    report = dict(
        version=version,
        generated_at=datetime.now(UTC).isoformat(),
        disposition="research_only",
        source_version=manifest["version"],
        source_hashes={
            str(p.relative_to(source)): digest(p)
            for p in (
                source / "manifest.json",
                source / "predictions.parquet",
                source / "features.parquet",
                source / "evaluations.json",
                snapshot,
            )
        },
        method=(
            "Exact saved-code refit for all completed QB passing-yard folds, "
            "plus exploratory diagnostics."
        ),
        limitations=[
            "No new independent holdout and no promotion or publication.",
            "Workload groups use cutoff-known prior attempts; "
            "they are not actual role or injury labels.",
            "Group boundaries and named cases are diagnostic choices after seeing overall results.",
            "Intervals resample seasons, are unadjusted for multiple comparisons, "
            "and do not handle all career dependence.",
            "Other-outcome gains are not a causal decomposition of passing-yard error.",
            "Injury-caused and job-change-caused shares of total error "
            "cannot be identified from these labels.",
        ],
        reproduction=reproduction,
        features=columns,
        comparisons=comparisons,
        boost_versus_ridge=direct,
        prior_workload=workloads,
        populations=populations,
        related_outcomes=related,
        examples=examples.to_dicts(),
        largest_boost_errors=boost.sort("absolute_error", descending=True).head(25).to_dicts(),
    )
    write_json(output / "report.json", report)
    shutil.copyfile(__file__, output / "review.py")
    shutil.copyfile(snapshot, output / "saved_forecast_implementation.py")
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "src/engine/metrics/evidence_diagnostics.py",
        output / "evidence_diagnostics.py",
    )
    write_json(
        output / "manifest.json",
        dict(
            version=version,
            status="research_only",
            source_version=manifest["version"],
            files={p.name: digest(p) for p in output.iterdir() if p.is_file()},
        ),
    )
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    review(args.data, args.version)
