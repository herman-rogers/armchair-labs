"""Isolated draft-capital coverage-gap study; never changes the frozen main inputs.

Two fixed raw-tree anchors append three known-as-of event features. CPU fitting
starts immediately; CUDA fitting waits for the dimension supplement to complete.
All comparisons are development diagnostics, not new main selection policies.
"""

# Long literal readout paragraphs retain their prose layout.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import shutil
import time
import traceback
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from experiments.future_player_lab.gpu_backends import build_estimator
from experiments.future_player_lab.notebooks import raw_distribution_data as raw
from threadpoolctl import threadpool_limits

from .data import load_dataset
from .representations import fit_representation
from .run import (
    COUNT_STATS,
    ROOT,
    RUNS,
    STATS,
    digest,
    fold_masks,
    json_write,
    packages,
    row_frame,
    source_paths,
    stat_arrays,
    verify_checkpoint,
)

DEFAULT_RUN = "rb_gpu_draft_capital_20261003_001"
KEY = ["horizon", "year", "origin", "player_id", "target"]
ADDED = ["draft_event__overall_pick", "draft_event__round", "draft_event__years_since_draft"]
ANCHORS = ("r024_lightgbm_raw_all", "r000_xgboost_raw_all")


def read_json(path):
    return json.loads(Path(path).read_text())


def parse_positive_integer(frame, column, alias=None):
    """Parse mixed numeric/string event fields; missing never means undrafted."""
    text = pl.col(column).cast(pl.String).str.strip_chars()
    value = text.cast(pl.Float64, strict=False)
    bad = frame.filter(
        text.is_not_null()
        & (text != "")
        & (value.is_null() | ~value.is_finite() | (value <= 0) | (value != value.floor()))
    )
    if bad.height:
        raise ValueError(f"Invalid positive integer event field {column}")
    return frame.with_columns(value.alias(alias or column))


def unique_events(frame, columns):
    """Collapse agreeing observations, preserving unknown event values."""
    counts = frame.group_by("player_id").agg(
        [pl.col(c).drop_nulls().n_unique().alias(c) for c in columns]
    )
    if counts.filter(pl.any_horizontal([pl.col(c) > 1 for c in columns])).height:
        raise ValueError("Conflicting known draft event values for one exact player ID")
    return frame.group_by("player_id").agg(
        [pl.col(c).drop_nulls().first().alias(c) for c in columns]
    )


def draft_features(panel, combine, rosters):
    """Use only known combine/draft events or observations before each origin."""
    pieces = []
    indexed = panel.select("player_id", "year", "origin", "candidate_basis").with_row_index("row")
    for (year, origin), block in indexed.partition_by(["year", "origin"], as_dict=True).items():
        events = unique_events(
            combine.filter((pl.col("combine_year") <= year) & (pl.col("draft_year") <= year)),
            ["draft_year", "draft_round", "combine_pick"],
        )
        history = unique_events(
            rosters.filter(
                (pl.col("season") < year)
                | ((pl.col("season") == year) & (pl.col("week") <= origin))
            ),
            ["roster_pick"],
        )
        joined = block.join(events, on="player_id", how="left", validate="m:1").join(
            history, on="player_id", how="left", validate="m:1"
        )
        if joined.filter(
            pl.col("combine_pick").is_not_null()
            & pl.col("roster_pick").is_not_null()
            & (pl.col("combine_pick") != pl.col("roster_pick"))
        ).height:
            raise ValueError("Known combine and roster draft picks disagree")
        pieces.append(
            joined.with_columns(
                pl.coalesce("combine_pick", "roster_pick").alias(ADDED[0]),
                pl.col("draft_round").alias(ADDED[1]),
                (pl.col("year") - pl.col("draft_year")).alias(ADDED[2]),
            )
        )
    result = pl.concat(pieces, how="diagonal_relaxed").sort("row")
    if result.height != panel.height or not result.select("player_id", "year", "origin").equals(
        panel.select("player_id", "year", "origin")
    ):
        raise ValueError("Draft extension changed base population or row ordering")
    return result.drop("row")


def load_sources():
    combine = raw.read_family("combine")
    crosswalk = raw.read_family("crosswalk")
    roster = raw.read_family("rosters")
    mapping = crosswalk.select("pfr_id", pl.col("gsis_id").alias("player_id")).drop_nulls().unique()
    ambiguous = mapping.filter(pl.col("pfr_id").is_duplicated())["pfr_id"].n_unique()
    mapping = mapping.filter(~pl.col("pfr_id").is_duplicated())
    c = combine.join(mapping, on="pfr_id", how="inner")
    for column in ["draft_year", "draft_round", "draft_ovr"]:
        c = parse_positive_integer(c, column)
    c = c.select(
        "player_id",
        pl.col("season").alias("combine_year"),
        "draft_year",
        "draft_round",
        pl.col("draft_ovr").alias("combine_pick"),
    )
    r = (
        parse_positive_integer(roster, "draft_number", "roster_pick")
        .filter(pl.col("gsis_id").is_not_null() & (pl.col("week") > 0))
        .select(pl.col("gsis_id").alias("player_id"), "season", "week", "roster_pick")
    )
    catalog = raw.catalog().filter(pl.col("family").is_in(["combine", "crosswalk", "rosters"]))
    audit = {
        "raw_manifest_sha256": digest(raw.SNAPSHOT),
        "source_assets": catalog.to_dicts(),
        "combined_roster_draft_dtype": str(roster.schema["draft_number"]),
        "roster_rows": roster.height,
        "roster_draft_known": roster["draft_number"].is_not_null().sum(),
        "combine_rows": combine.height,
        "mapped_combine_rows": c.height,
        "mapped_combine_players": c["player_id"].n_unique(),
        "withheld_ambiguous_pfr_ids": ambiguous,
        "fields": {
            ADDED[0]: "combine.draft_ovr, else roster.draft_number from observed past weeks",
            ADDED[1]: "combine.draft_round; no inference from pick number",
            ADDED[
                2
            ]: "forecast year minus combine.draft_year; roster entry/rookie years not substituted",
        },
        "clock": "combine season and draft_year <= forecast year; roster season < year or current week <= origin",
        "null_policy": "Missing is unknown, never an undrafted indicator or zero pick",
        "gap": "Main explicitly excluded combine draft fields and numeric-only extraction omitted roster draft_number after String promotion",
    }
    return c, r, audit


def extend_dataset(base, combine, rosters):
    features = draft_features(base.panel, combine, rosters)
    values = features.select(ADDED).to_numpy().astype(np.float32)
    if any(name in base.names for name in ADDED):
        raise ValueError("Draft fields already exist in base inputs")
    extended = replace(
        base,
        x=np.column_stack([base.x, values]),
        names=[*base.names, *ADDED],
        groups=[*base.groups, *["draft_event"] * len(ADDED)],
    )
    np.testing.assert_array_equal(extended.x[:, : len(base.names)], base.x)
    return extended, features


def aligned_forecasts(candidate, control):
    ordered = []
    for frame in (candidate, control):
        if frame.select(KEY).is_duplicated().any():
            raise ValueError("Duplicate forecast keys")
        if not np.isfinite(frame["prediction"].to_numpy()).all():
            raise ValueError("Nonfinite forecast")
        if frame.filter(pl.col("training_max_year") >= pl.col("year")).height:
            raise ValueError("Training chronology violated")
        ordered.append(frame.sort(KEY))
    a, b = ordered
    if not a.select(KEY).equals(b.select(KEY)):
        raise ValueError("Candidate/control row keys differ")
    for name in ["actual", "weeks"]:
        if not np.array_equal(a[name].to_numpy(), b[name].to_numpy(), equal_nan=True):
            raise ValueError(f"Candidate/control {name} differs")
    return a.with_columns(pl.Series("control_prediction", b["prediction"]))


def control_path(main, horizon, recipe, year):
    suffix = Path(horizon) / "fits" / recipe["id"] / str(year)
    for path in [
        main / suffix,
        main / "workers" / ("cpu" if recipe["device"] == "cpu" else "gpu") / suffix,
    ]:
        if (path / "complete.json").exists():
            verify_checkpoint(path)
            return path
    raise FileNotFoundError(f"Completed raw anchor missing: {suffix}")


def fit_fold(dataset, recipe, year, config, path, final=False):
    if (path / "complete.json").exists():
        verify_checkpoint(path)
        return
    path.mkdir(parents=True, exist_ok=True)
    years = dataset.panel["year"].to_numpy()
    train, test = fold_masks(years, year) if not final else (years <= 2025, years == 2025)
    targets, _ = stat_arrays(dataset)
    with threadpool_limits(limits=config["threads"]):
        representation = fit_representation(
            dataset.x[train], dataset.names, recipe["representation"], seed=recipe["seed"]
        )
        tx = np.ascontiguousarray(representation.transform(dataset.x[train]), dtype=np.float32)
        vx = np.ascontiguousarray(representation.transform(dataset.x[test]), dtype=np.float32)
    predictions, fits = [], []
    for j, stat in enumerate(STATS):
        rate = targets[:, j] / dataset.exposure
        known = np.isfinite(rate[train])
        model, metadata = build_estimator(
            recipe["family"],
            recipe["params"],
            seed=recipe["seed"],
            device=recipe["device"],
            threads=config["threads"],
        )
        start = time.monotonic()
        with threadpool_limits(limits=config["threads"]):
            model.fit(tx[known], rate[train][known].astype(np.float32))
            rates = np.asarray(model.predict(vx), dtype=float).reshape(-1)
        if not np.isfinite(rates).all() or not metadata["device_verified"]:
            raise ValueError("Nonfinite predictions or unverified device")
        if stat in COUNT_STATS:
            rates = np.maximum(0, rates)
        elapsed = time.monotonic() - start
        fits.append(
            {
                "target": stat,
                "seconds": elapsed,
                "known_training_rows": int(known.sum()),
                "backend": metadata,
            }
        )
        if final:
            bundle = {
                "model": model,
                "preprocessing": representation,
                "features": dataset.names,
                "target": stat,
                "count_stat": stat in COUNT_STATS,
                "training_max_year": 2025,
                "output": "predicted rate times provided calendar-week exposure",
            }
            dest = path / f"{stat}.joblib"
            joblib.dump(bundle, dest, compress=3)
            restored = joblib.load(dest)
            np.testing.assert_allclose(
                restored["model"].predict(vx[:8]), model.predict(vx[:8]), rtol=1e-6, atol=1e-6
            )
        else:
            predictions.append(
                row_frame(
                    dataset,
                    test,
                    stat,
                    targets[test, j],
                    rates * dataset.exposure[test],
                    recipe["id"] + "_draft",
                    year,
                    elapsed,
                )
            )
            if year == max(config["years"]):
                dest = path / f"{stat}.joblib"
                joblib.dump(model, dest, compress=3)
                restored = joblib.load(dest)
                np.testing.assert_allclose(
                    restored.predict(vx[:8]), model.predict(vx[:8]), rtol=1e-6, atol=1e-6
                )
        del model
        gc.collect()
    if predictions:
        pl.concat(predictions).write_parquet(path / "predictions.parquet")
    if final or year == max(config["years"]):
        joblib.dump(representation, path / "preprocessing.joblib", compress=3)
        json_write(path / "feature_order.json", dataset.names)
    json_write(
        path / "metadata.json",
        {
            "recipe": recipe,
            "year": year,
            "final_refit": final,
            "training_max_year": int(years[train].max()),
            "representation": representation.summary(),
            "fits": fits,
        },
    )
    json_write(
        path / "complete.json",
        {
            "files": {
                p.name: digest(p)
                for p in path.iterdir()
                if p.is_file() and p.name != "complete.json"
            }
        },
    )
    print(
        f"{recipe['family']} {dataset.panel['horizon'][0]} {year}: {len(fits)} fits{' final' if final else ''}",
        flush=True,
    )


def score(out, main, config):
    matched, controls = [], {}
    for recipe in config["recipes"]:
        for horizon in config["horizons"]:
            for year in config["years"]:
                candidate = out / horizon / "fits" / recipe["id"] / str(year)
                verify_checkpoint(candidate)
                control = control_path(main, horizon, recipe, year)
                controls[str(control.relative_to(main))] = digest(control / "complete.json")
                aligned = aligned_forecasts(
                    pl.read_parquet(candidate / "predictions.parquet"),
                    pl.read_parquet(control / "predictions.parquet"),
                )
                rookies = pl.read_parquet(out / horizon / "draft_features.parquet").select(
                    "player_id",
                    "year",
                    "origin",
                    (pl.col("draft_year") == pl.col("year"))
                    .fill_null(False)
                    .alias("known_current_draft_year"),
                )
                aligned = aligned.join(
                    rookies, on=["player_id", "year", "origin"], how="left", validate="m:1"
                )
                matched.append(aligned.with_columns(pl.lit(recipe["family"]).alias("family")))
    frame = pl.concat(matched, how="diagonal_relaxed")
    tables = []
    for scope, subset in [
        ("all_folds", frame),
        ("evaluation", frame.filter(pl.col("year").is_in(config["evaluation_years"]))),
    ]:
        for cohort, rows in [
            ("all", subset),
            (
                "dated_draft_entrants",
                subset.filter(pl.col("candidate_basis") == "dated draft / exact ID"),
            ),
            ("known_current_draft_year", subset.filter(pl.col("known_current_draft_year"))),
        ]:
            for annual in (False, True):
                groups = ["horizon", "target", "family"] + (["year"] if annual else [])
                result = (
                    rows.filter(pl.col("actual").is_finite())
                    .group_by(groups)
                    .agg(
                        pl.len().alias("n"),
                        pl.col("year").n_unique().alias("years"),
                        ((pl.col("prediction") - pl.col("actual")) ** 2).mean().alias("mse"),
                        ((pl.col("control_prediction") - pl.col("actual")) ** 2)
                        .mean()
                        .alias("control_mse"),
                        (pl.col("prediction") - pl.col("actual")).abs().mean().alias("mae"),
                        (pl.col("control_prediction") - pl.col("actual"))
                        .abs()
                        .mean()
                        .alias("control_mae"),
                    )
                    .with_columns(
                        pl.col("mse").sqrt().alias("rmse"),
                        pl.col("control_mse").sqrt().alias("control_rmse"),
                        pl.when(pl.col("control_mse") > 0)
                        .then(100 * (1 - pl.col("mse") / pl.col("control_mse")))
                        .otherwise(None)
                        .alias("mse_improvement_pct"),
                        pl.lit(scope).alias("scope"),
                        pl.lit(cohort).alias("cohort"),
                        pl.lit(annual).alias("annual"),
                    )
                )
                tables.append(result)
    scores = pl.concat(tables, how="diagonal_relaxed")
    analysis = out / "analysis"
    analysis.mkdir(exist_ok=True)
    frame.write_parquet(analysis / "predictions.parquet")
    scores.filter(~pl.col("annual")).write_parquet(analysis / "controlled_comparisons.parquet")
    scores.filter(pl.col("annual")).write_parquet(analysis / "annual_comparisons.parquet")
    scores.write_csv(analysis / "comparisons.csv")
    json_write(out / "raw_controls.json", controls)
    shown = scores.filter(
        (pl.col("scope") == "evaluation") & ~pl.col("annual") & (pl.col("target") == "points")
    ).sort("horizon", "family", "cohort")
    lines = [
        "# Draft-capital coverage-gap study",
        "",
        "The original 4,810-input campaign omitted draft capital: combine draft-event fields were excluded and mixed roster draft_number schemas promoted the field to String. This isolated study appends exactly three known-as-of features without changing the original inputs, rows, target masks or frozen 52 policies.",
        "",
        "Two fixed raw LightGBM/XGBoost anchors retain their main parameters, seed and four threads. All 2018–2025 folds were fitted; the table evaluates 2021–2025 against verified raw anchor checkpoints on identical rows. dated_draft_entrants means the original candidate_basis='dated draft / exact ID'; known_current_draft_year also includes rookies observed in Weeks 1–2 when a verified draft_year equals forecast year. Neither cohort uses future success; neither claims complete undrafted-rookie coverage. Missing draft data stays unknown, never undrafted. These gap-driven comparisons are historical development evidence and do not promote a new policy.",
        "",
        "| Horizon | Family | Cohort | n | RMSE | Raw RMSE | MSE reduction % |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in shown.iter_rows(named=True):
        gain = "—" if row["mse_improvement_pct"] is None else f"{row['mse_improvement_pct']:.2f}"
        lines.append(
            f"| {row['horizon']} | {row['family']} | {row['cohort']} | {row['n']} | {row['rmse']:.2f} | {row['control_rmse']:.2f} | {gain} |"
        )
    lines += [
        "",
        "[All target/cohort/annual comparisons](analysis/comparisons.csv) · [Raw schema/time audit](source_audit.json) · [Coverage](coverage.json) · [Verified raw controls](raw_controls.json)",
        "",
        "256 evaluation target fits plus 32 separately counted all-through-2025 refits. Latest evaluated-fold estimators/preprocessors and final bundles are saved and replay checked. Final refits have no prospective evaluation. Historical data revisions and incomplete newcomer coverage remain limitations; missing draft fields do not establish undrafted status. Roster entry/rookie years are not substituted for actual draft year.",
    ]
    (out / "readout.md").write_text("\n".join(lines) + "\n")


def run(out, main, dimensions):
    out = Path(out)
    main = Path(main)
    main_config = read_json(main / "config.json")
    recipes = [
        copy.deepcopy(next(r for r in main_config["recipes"] if r["id"] == name))
        for name in ANCHORS
    ]
    config = {
        **main_config,
        "recipes": recipes,
        "added_features": ADDED,
        "purpose": "Fixed draft-capital gap supplement; main inputs/policies untouched",
    }
    sources = {str(p.relative_to(ROOT)): digest(p) for p in [*source_paths(), Path(__file__)]}
    signature = {
        "config": config,
        "sources": sources,
        "packages": packages(),
        "main_config_sha256": digest(main / "config.json"),
    }
    out.mkdir(parents=True, exist_ok=True)
    if (out / "signature.json").exists():
        if read_json(out / "signature.json") != signature:
            raise ValueError("Draft study resume source/configuration changed")
    else:
        json_write(out / "signature.json", signature)
        json_write(out / "config.json", config)
        implementation = out / "implementation"
        implementation.mkdir()
        for path in [*source_paths(), Path(__file__)]:
            dest = implementation / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    if (out / "manifest.json").exists() and read_json(out / "manifest.json").get(
        "status"
    ) == "complete":
        return
    json_write(out / "manifest.json", {"status": "running_cpu", "expected_fits": 256})
    combine, rosters, audit = load_sources()
    json_write(out / "source_audit.json", audit)
    datasets, coverage = {}, []
    for horizon in config["horizons"]:
        base = load_dataset(horizon, tuple(config["origins"]), config["input_cache"])
        dataset, features = extend_dataset(base, combine, rosters)
        datasets[horizon] = dataset
        dest = out / horizon
        dest.mkdir(exist_ok=True)
        features.write_parquet(dest / "draft_features.parquet")
        json_write(
            dest / "inputs.json",
            {
                "base_provenance": base.provenance,
                "base_features": len(base.names),
                "features": dataset.names,
                "rows": base.panel.height,
            },
        )
        for cohort, f in [
            ("all", features),
            (
                "dated_draft_entrants",
                features.filter(pl.col("candidate_basis") == "dated draft / exact ID"),
            ),
            ("evaluation", features.filter(pl.col("year").is_in(config["evaluation_years"]))),
            ("known_current_draft_year", features.filter(pl.col("draft_year") == pl.col("year"))),
        ]:
            coverage.append(
                {
                    "horizon": horizon,
                    "cohort": cohort,
                    "rows": f.height,
                    **{n: int(f[n].is_not_null().sum()) for n in ADDED},
                }
            )
    json_write(out / "coverage.json", coverage)
    for recipe in recipes:
        if recipe["device"] == "cuda":
            json_write(
                out / "manifest.json",
                {
                    "status": "waiting_gpu",
                    "completed_fits": 128,
                    "final_refit_fits": 16,
                    "expected_fits": 256,
                },
            )
            while (
                not (Path(dimensions) / "manifest.json").exists()
                or read_json(Path(dimensions) / "manifest.json").get("status") != "complete"
            ):
                time.sleep(30)
            if {
                str(p.relative_to(ROOT)): digest(p) for p in [*source_paths(), Path(__file__)]
            } != sources:
                raise ValueError("Study sources changed while waiting for CUDA")
            json_write(
                out / "manifest.json",
                {"status": "running_gpu", "completed_fits": 128, "expected_fits": 256},
            )
        for horizon, dataset in datasets.items():
            for year in config["years"]:
                fit_fold(
                    dataset, recipe, year, config, out / horizon / "fits" / recipe["id"] / str(year)
                )
            fit_fold(
                dataset,
                recipe,
                2026,
                config,
                out / "final_models" / horizon / recipe["id"],
                final=True,
            )
    json_write(
        out / "manifest.json", {"status": "scoring", "completed_fits": 256, "final_refit_fits": 32}
    )
    score(out, main, config)
    json_write(
        out / "manifest.json",
        {
            "status": "complete",
            "completed_fits": 256,
            "expected_fits": 256,
            "final_refit_fits": 32,
            "sources": sources,
            "files": {
                str(p.relative_to(out)): digest(p)
                for p in out.rglob("*")
                if p.is_file() and p != out / "manifest.json" and not p.name.endswith(".tmp")
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--main-run", default="rb_gpu_20261003_001")
    parser.add_argument("--dimensions-run", default="rb_gpu_dimensions_20261003_001")
    args = parser.parse_args()
    with (RUNS / f".{args.run_id}.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(RUNS / args.run_id, RUNS / args.main_run, RUNS / args.dimensions_run)
        except Exception as exc:
            destination = RUNS / args.run_id
            if destination.exists():
                previous = (
                    read_json(destination / "manifest.json")
                    if (destination / "manifest.json").exists()
                    else {}
                )
                json_write(
                    destination / "manifest.json",
                    {
                        **previous,
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            raise
