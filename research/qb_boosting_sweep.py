"""Reproducible, chronological capacity research against exact canonical QB errors."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from patron.data.nextgen import load_analysis
from patron.data.releases import digest, identifier, write_json

ROOT = Path(__file__).resolve().parents[1]
STAGES = (30, 60, 120, 240, 480)
SEED = 20260923


def configurations():
    paths = []

    def add(depth=None, leaves=15, rate=0.05, minimum=30, l2=10, features=1.0):
        p = dict(
            max_depth=depth,
            max_leaf_nodes=leaves,
            learning_rate=rate,
            min_samples_leaf=minimum,
            l2_regularization=l2,
            max_features=features,
        )
        if p not in paths:
            paths.append(p)

    for (depth, leaves), rate in product(
        [(1, 2), (2, 4), (3, 4), (3, 8), (5, 8), (5, 15), (5, 31), (None, 15), (None, 63)],
        [0.02, 0.05, 0.10],
    ):
        add(depth, leaves, rate)
    for minimum, l2 in product((10, 30, 60), (0, 10, 100)):
        add(minimum=minimum, l2=l2)
    for features in (0.5, 0.75):
        add(features=features)
    for rate in (0.05, 0.10):
        add(leaves=63, rate=rate, minimum=5, l2=0)
    return [
        dict(id=f"p{i:02d}_t{n:03d}", path=i, max_iter=n, **p)
        for i, p in enumerate(paths)
        for n in STAGES
    ]


def fixed_index(configs):
    return next(
        i
        for i, c in enumerate(configs)
        if c["max_depth"] is None
        and c["max_leaf_nodes"] == 15
        and c["learning_rate"] == 0.05
        and c["min_samples_leaf"] == 30
        and c["l2_regularization"] == 10
        and c["max_features"] == 1
        and c["max_iter"] == 120
    )


def complexity(c):
    return (
        c["max_leaf_nodes"] * c["max_iter"],
        -c["min_samples_leaf"],
        -c["l2_regularization"],
        c["max_depth"] or 999,
        c["id"],
    )


def select_config(annual, configs, year, policy, probability=False):
    """Selection cannot observe year or any later scores; inputs include all folds."""
    earlier = [r for r in annual if r["season"] < year]
    fixed = fixed_index(configs)
    if len(earlier) < 3:
        return fixed, dict(reason="fewer_than_three_earlier_folds", years=[])
    metric = "mse" if probability or policy == "mse" else "mae"
    losses = np.array([r[metric] for r in earlier])
    guard_metric = "mae" if metric == "mse" else "mse"
    guard = np.array([r[guard_metric] for r in earlier]).mean(axis=0)
    means = losses.mean(axis=0)
    eligible = (means <= means[fixed] + 1e-12) & (
        guard <= guard[fixed] * (1.02 if metric == "mse" else 1) + 1e-12
    )
    eligible[fixed] = True
    best = min(np.flatnonzero(eligible), key=lambda i: (means[i], configs[i]["id"]))
    paired = losses - losses[:, [best]]
    se = paired.std(axis=0, ddof=1) / np.sqrt(len(earlier))
    candidates = np.flatnonzero(eligible & (means <= means[best] + se + 1e-12))
    selected = min(candidates, key=lambda i: complexity(configs[i]))
    return int(selected), dict(
        reason="paired_one_standard_error",
        years=[r["season"] for r in earlier],
        metric=metric,
        best=configs[best]["id"],
        eligible=int(eligible.sum()),
        selected_inner_loss=float(means[selected]),
        best_inner_loss=float(means[best]),
        fixed_inner_loss=float(means[fixed]),
        paired_se=float(se[selected]),
    )


def saved_module(source):
    spec = importlib.util.spec_from_file_location(
        "frozen_qb_boost_source", source / "implementation/src/patron/metrics/nextgen.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def errors(predictions, actual):
    e = predictions - actual
    return dict(
        mae=np.nanmean(np.abs(e), axis=-1),
        mse=np.nanmean(e**2, axis=-1),
        bias=np.nanmean(e, axis=-1),
    )


def fit_job(source_text, out_text, target, year, configs):
    source, out = Path(source_text), Path(out_text)
    module = saved_module(source)
    frame = pl.read_parquet(source / "features.parquet").filter(pl.col("position") == "QB")
    rows = frame.to_dicts()
    columns = sorted(c for c in frame.columns if c.startswith("x_"))
    train = [r for r in rows if r["forecast_season"] < year and r["outcome_complete"]]
    test = [r for r in rows if r["forecast_season"] == year]
    assert all(r["forecast_season"] < year for r in train)
    saved = pl.read_parquet(source / "predictions.parquet").filter(
        (pl.col("position") == "QB")
        & (pl.col("target") == target)
        & (pl.col("season") == year)
        & pl.col("complete")
    )
    reproduction = []
    fits, _ = module.fit_fold(train, test, target, columns)
    for model, values in fits.items():
        subset = saved.filter(pl.col("model") == model)
        lookup = {r["player_id"]: r for r in subset.to_dicts()}
        assert len(lookup) == len(test) == len(subset)
        expected = np.array([lookup[r["player_id"]]["prediction"] for r in test], dtype=float)
        labels = np.array([lookup[r["player_id"]]["actual"] for r in test], dtype=float)
        actual = np.array([r[f"y_{target}"] for r in test], dtype=float)
        np.testing.assert_allclose(actual, labels, rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-8, equal_nan=True)
        reproduction.append(
            dict(model=model, max_difference=float(np.nanmax(abs(values - expected))))
        )
    train = [r for r in train if r[f"y_{target}"] is not None]
    x = np.array([[r[c] if r[c] is not None else np.nan for c in columns] for r in train])
    z = np.array([[r[c] if r[c] is not None else np.nan for c in columns] for r in test])
    varying = [i for i in range(x.shape[1]) if len(np.unique(x[np.isfinite(x[:, i]), i])) > 1]
    x, z = x[:, varying], z[:, varying]
    y = np.array([r[f"y_{target}"] for r in train])
    actual = np.array([r[f"y_{target}"] for r in test], dtype=float)
    predictions = np.empty((len(configs), len(test)))
    train_metrics = {k: np.empty(len(configs)) for k in ("mae", "mse", "bias")}
    # A staged prefix is the exact smaller ensemble; no test-dependent stopping.
    with threadpool_limits(limits=1):
        for path in sorted({c["path"] for c in configs}):
            indices = [i for i, c in enumerate(configs) if c["path"] == path]
            params = {
                k: v for k, v in configs[indices[0]].items() if k not in ("id", "path", "max_iter")
            }
            estimator = HistGradientBoostingRegressor(
                **params, max_iter=max(STAGES), max_bins=63, early_stopping=False, random_state=SEED
            )
            estimator.fit(x, y)
            stages = {configs[i]["max_iter"]: i for i in indices}
            for n, (a, b) in enumerate(
                zip(estimator.staged_predict(x), estimator.staged_predict(z), strict=True), 1
            ):
                if n not in stages:
                    continue
                i = stages[n]
                predictions[i] = module.constrained(b, test, target)
                for name, value in errors(module.constrained(a, train, target), y).items():
                    train_metrics[name][i] = value
    np.testing.assert_allclose(
        predictions[fixed_index(configs)], fits["boost"], rtol=1e-10, atol=1e-8, equal_nan=True
    )
    finite = np.isfinite(actual) & np.isfinite(fits["baseline"])
    assert np.array_equal(np.isfinite(predictions).all(axis=0), np.isfinite(fits["boost"]))
    scores = errors(predictions[:, finite], actual[finite])
    fold = dict(
        target=target,
        season=year,
        train_rows=len(train),
        test_rows=len(test),
        scored_rows=int(finite.sum()),
        train_min_year=min(r["forecast_season"] for r in train),
        train_max_year=max(r["forecast_season"] for r in train),
        features=len(varying),
        reproduction=reproduction,
        **{k: v.tolist() for k, v in scores.items()},
        **{"train_" + k: v.tolist() for k, v in train_metrics.items()},
    )
    prefix = out / "folds" / f"{target}_{year}"
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        predictions=predictions,
        actual=actual,
        player_ids=np.array([r["player_id"] for r in test]),
    )
    write_json(prefix.with_suffix(".json"), fold)
    return fold


def holm(pvalues):
    order = np.argsort(pvalues)
    result = np.empty(len(order))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        result[index] = previous
    return result.tolist()


def paired_summary(rows, model, comparator, metric):
    annual = []
    for year in sorted({r["season"] for r in rows}):
        subset = [
            r
            for r in rows
            if r["season"] == year
            and r["actual"] is not None
            and r[model] is not None
            and r[comparator] is not None
        ]
        if not subset:
            continue
        y = np.array([r["actual"] for r in subset])
        a = errors(np.array([r[model] for r in subset]), y)
        b = errors(np.array([r[comparator] for r in subset]), y)
        annual.append(
            dict(
                season=year,
                n=len(subset),
                **{k: float(v) for k, v in a.items()},
                **{"reference_" + k: float(v) for k, v in b.items()},
            )
        )
    if not annual:
        return dict(n=0, years=0)
    gain = np.array([r["reference_" + metric] - r[metric] for r in annual])
    rng = np.random.default_rng(SEED)
    boots = rng.choice(gain, (10000, len(gain))).mean(axis=1)
    # Circular moving blocks preserve adjacent-season dependence within each pair.
    starts = rng.integers(0, len(gain), (10000, (len(gain) + 1) // 2))
    indices = np.stack([starts, (starts + 1) % len(gain)], axis=-1).reshape(10000, -1)
    blocks = gain[indices[:, : len(gain)]].mean(axis=1)
    p = None
    if len(gain) <= 10:
        signs = np.array(list(product((-1, 1), repeat=len(gain))))
        p = float(np.mean(np.abs(signs @ gain / len(gain)) >= abs(gain.mean()) - 1e-12))
    avg = {
        k: float(np.mean([r[k] for r in annual]))
        for k in ("mae", "mse", "bias", "reference_mae", "reference_mse")
    }
    return dict(
        n=sum(r["n"] for r in annual),
        years=len(annual),
        annual=annual,
        **avg,
        rmse=float(np.sqrt(avg["mse"])),
        reference_rmse=float(np.sqrt(avg["reference_mse"])),
        gain=float(gain.mean()),
        ci_low=float(np.quantile(boots, 0.025)),
        ci_high=float(np.quantile(boots, 0.975)),
        block_ci_low=float(np.quantile(blocks, 0.025)),
        block_ci_high=float(np.quantile(blocks, 0.975)),
        positive_years=int((gain > 0).sum()),
        loo_min=float(min(np.delete(gain, i).mean() for i in range(len(gain))))
        if len(gain) > 1
        else None,
        p_value=p,
    )


def finite_or_none(value):
    return float(value) if value is not None and np.isfinite(value) else None


def analyze(source, out, configs, folds, targets):
    module = saved_module(source)
    saved = (
        pl.read_parquet(source / "predictions.parquet")
        .filter((pl.col("position") == "QB") & pl.col("complete"))
        .to_dicts()
    )
    lookup = {(r["target"], r["season"], r["player_id"], r["model"]): r for r in saved}
    features = pl.read_parquet(source / "features.parquet").filter(pl.col("position") == "QB")
    feature_lookup = {(r["forecast_season"], r["player_id"]): r for r in features.to_dicts()}
    predictions, selections, hindsight, curves = [], [], [], []
    for target in targets:
        annual = sorted([r for r in folds if r["target"] == target], key=lambda r: r["season"])
        probability = target == "season_appearance"
        metric = "mse" if probability else "mae"
        for window, start in [("all_history", 2007), ("modern", 2019)]:
            part = [r for r in annual if r["season"] >= start]
            losses = np.array([r[metric] for r in part]).mean(axis=0)
            best = int(np.argmin(losses))
            hindsight.append(
                dict(
                    target=target,
                    window=window,
                    metric=metric,
                    configuration=configs[best]["id"],
                    loss=float(losses[best]),
                    warning="Chosen using evaluation labels; not deployable-policy performance",
                )
            )
            for i, config in enumerate(configs):
                curves.append(
                    dict(
                        target=target,
                        window=window,
                        **config,
                        **{
                            k: float(np.mean([r[k][i] for r in part]))
                            for k in ("mae", "mse", "bias", "train_mae", "train_mse")
                        },
                    )
                )
        for fold in annual:
            year = fold["season"]
            artifact = np.load(out / "folds" / f"{target}_{year}.npz")
            choices = {}
            for policy in ("canonical", "mse"):
                index, details = select_config(annual, configs, year, policy, probability)
                choices["selected_" + policy] = index
                selections.append(
                    dict(
                        target=target,
                        season=year,
                        policy=policy,
                        configuration=configs[index]["id"],
                        **details,
                    )
                )
                locked, _ = select_config(annual, configs, 2023, policy, probability)
                if year >= 2023:
                    choices["locked_" + policy] = locked
            for i, pid in enumerate(artifact["player_ids"]):
                row = feature_lookup[year, str(pid)]
                prior = row["x_prior_attempts"]
                population = row["player_population"]
                item = dict(
                    target=target,
                    season=year,
                    player_id=str(pid),
                    player_display_name=row["player_display_name"],
                    population=population,
                    prior_workload="unknown"
                    if prior is None
                    else "300+"
                    if prior >= 300
                    else "1-299"
                    if prior > 0
                    else "zero",
                    prior_history="small_prior"
                    if (row["x_prior_weeks"] or 0) < 10
                    else "10plus_prior_weeks",
                    actual=finite_or_none(artifact["actual"][i]),
                )
                for model in ("baseline", "boost", "ridge"):
                    item[model] = lookup[target, year, str(pid), model]["prediction"]
                for name, index in choices.items():
                    item[name] = finite_or_none(artifact["predictions"][index, i])
                predictions.append(item)
    pl.DataFrame(predictions, infer_schema_length=None).write_parquet(
        out / "policy_predictions.parquet"
    )
    pl.DataFrame(curves).write_csv(out / "capacity_curves.csv")
    write_json(out / "selections.json", selections)
    comparisons, subgroups, checks = [], [], []
    for target in targets:
        probability = target == "season_appearance"
        for window, start in [
            ("all_history", 2007),
            ("selection_active", 2010),
            ("modern", 2019),
            ("recent_locked", 2023),
        ]:
            rows = [r for r in predictions if r["target"] == target and r["season"] >= start]
            models = ["baseline", "boost", "ridge", "selected_canonical", "selected_mse"]
            if window == "recent_locked":
                models += ["locked_canonical", "locked_mse"]
            for model in models:
                metric = "mse" if probability or model.endswith("_mse") else "mae"
                for comparator in ("baseline", "boost", "ridge"):
                    if comparator == model:
                        continue
                    result = dict(
                        target=target,
                        window=window,
                        model=model,
                        comparator=comparator,
                        metric=metric,
                        **paired_summary(rows, model, comparator, metric),
                    )
                    comparisons.append(result)
            # Reproduce the existing canonical metric aggregation, not just predictions.
            for model in ("baseline", "boost", "ridge"):
                canonical = module.paired_summary(
                    [
                        r
                        for r in saved
                        if r["target"] == target and r["model"] == model and r["season"] >= start
                    ],
                    probability=probability,
                )
                ours = paired_summary(rows, model, "baseline", "mse" if probability else "mae")
                np.testing.assert_allclose(
                    canonical["error"],
                    ours["mse" if probability else "mae"],
                    rtol=1e-12,
                    atol=1e-12,
                )
                assert canonical["n"] == ours["n"]
                checks.append(
                    dict(
                        target=target,
                        window=window,
                        model=model,
                        n=ours["n"],
                        error=canonical["error"],
                    )
                )
            if window not in ("all_history", "modern"):
                continue
            for field in ("population", "prior_workload", "prior_history"):
                for group in sorted({r[field] for r in rows}):
                    subset = [r for r in rows if r[field] == group]
                    for policy in ("canonical", "mse"):
                        metric = "mse" if probability or policy == "mse" else "mae"
                        for comparator in ("baseline", "boost"):
                            subgroups.append(
                                dict(
                                    target=target,
                                    window=window,
                                    field=field,
                                    group=group,
                                    policy=policy,
                                    comparator=comparator,
                                    metric=metric,
                                    **paired_summary(
                                        subset, "selected_" + policy, comparator, metric
                                    ),
                                )
                            )
    primary = [
        r
        for r in comparisons
        if r["window"] == "modern"
        and r["model"].startswith("selected_")
        and r["comparator"] == "boost"
    ]
    assert len(primary) == len(targets) * 2
    for r, adjusted in zip(primary, holm([r["p_value"] for r in primary]), strict=True):
        r["holm_p_value"] = adjusted
    report = dict(
        status="completed_research_only",
        source=source.name,
        configurations=len(configs),
        paths=len({c["path"] for c in configs}),
        targets=targets,
        folds=len(folds),
        reproduced_model_folds=sum(len(f["reproduction"]) for f in folds),
        maximum_reproduction_difference=max(
            r["max_difference"] for f in folds for r in f["reproduction"]
        ),
        comparisons=comparisons,
        subgroups=subgroups,
        hindsight=hindsight,
        canonical_metric_checks=checks,
        limitations=[
            "Preseason full-season study; no weekly-horizon or ranking-promotion claim.",
            "Repeated historical research; no untouched or prospective holdout.",
            "Season and two-year blocks do not fully model repeated-player dependence.",
            "Uses identical reconstructed provider vintages and known canonical source gaps.",
            "Undefined rate outcomes are excluded only from their rate target.",
            "Fixed search is bounded and cannot establish a global hyperparameter optimum.",
            "Squared-error training is evaluated under both canonical and mean-forecast losses.",
            "Holm adjustment applies to 18 modern selected-policy versus fixed-boost tests only.",
        ],
    )
    write_json(out / "report.json", report)
    return report


def run(data, version, workers):
    source, manifest = load_analysis(data)
    out = data / "research" / identifier(version)
    out.mkdir(parents=True, exist_ok=False)
    (out / "folds").mkdir()
    configs = configurations()
    module = saved_module(source)
    targets = [t for t, definition in module.TARGETS.items() if "QB" in definition[2]]
    for name in (
        "research/qb_boosting_sweep.py",
        "research/qb_boosting_protocol.md",
        "tests/test_qb_boosting_sweep.py",
    ):
        dest = out / "implementation" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    shutil.copyfile(
        source / "implementation/src/patron/metrics/nextgen.py", out / "canonical_nextgen.py"
    )
    shutil.copyfile(ROOT / "research/qb_boosting_protocol.md", out / "protocol.md")
    write_json(out / "configurations.json", configs)
    started = dict(
        started_at=datetime.now(UTC).isoformat(),
        source=source.name,
        source_manifest_sha256=digest(source / "manifest.json"),
        gold=manifest["gold"],
        source_hashes={
            name: digest(source / name)
            for name in ("features.parquet", "predictions.parquet", "evaluations.json")
        },
        catalog_sha256=digest(data / "current.json"),
        protocol_sha256=digest(out / "protocol.md"),
        python=platform.python_version(),
        sklearn=sklearn.__version__,
        numpy=np.__version__,
        polars=pl.__version__,
        workers=workers,
        targets=targets,
        configurations=len(configs),
        implementation_sha256=digest(Path(__file__)),
    )
    write_json(out / "started.json", started)
    print(f"Frozen {len(configs)} configurations × {len(targets)} targets × 19 folds", flush=True)
    folds = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fit_job, str(source), str(out), target, year, configs)
            for year in range(2007, 2026)
            for target in targets
        ]
        for future in as_completed(futures):
            fold = future.result()
            folds.append(fold)
            print(
                f"{len(folds)}/{len(futures)} {fold['target']} {fold['season']} "
                f"train={fold['train_rows']} scored={fold['scored_rows']}",
                flush=True,
            )
    report = analyze(source, out, configs, folds, targets)
    assert digest(data / "current.json") == started["catalog_sha256"], "Catalog changed during run"
    for name, expected in started["source_hashes"].items():
        assert digest(source / name) == expected, f"Source changed during run: {name}"
    write_json(
        out / "manifest.json",
        dict(
            version=version,
            status=report["status"],
            completed_at=datetime.now(UTC).isoformat(),
            source=started,
            files={
                str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()
            },
        ),
    )
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--version", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    # Bound native thread pools before spawned workers import NumPy/Polars.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[key] = "1"
    run(args.data, args.version, args.workers)
