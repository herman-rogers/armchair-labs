"""Choose tree count on an earlier season, then refit all earlier history."""

import argparse
import json
import shutil
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from ..data import digest
from ..run import write_json
from .run import LAB, ROOT, load_task, validate_data

ENGINES = ("lgb", "xgb", "cat")
VIEWS = ("all", "market")


def fit_adaptive(engine, x, y, years, test_x, *, ceiling=4096, patience=80, seed=20260926):
    validation_year = int(years.max())
    older, validation = years < validation_year, years == validation_year
    if not older.any() or not validation.any():
        raise ValueError("Tree-count selection needs an earlier chronological validation season")
    active = np.flatnonzero(np.any(np.isfinite(x[older]), axis=0))

    def model(iterations, stopping=False):
        if engine == "lgb":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=iterations,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=6,
                min_child_samples=30,
                reg_lambda=10,
                colsample_bytree=0.8,
                max_bin=63,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
            )
        if engine == "xgb":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=iterations,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=30,
                reg_lambda=10,
                colsample_bytree=0.8,
                subsample=0.8,
                max_bin=63,
                n_jobs=1,
                tree_method="hist",
                random_state=seed,
                early_stopping_rounds=patience if stopping else None,
            )
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=10,
            rsm=0.8,
            border_count=63,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
        )

    candidate = model(ceiling, True)
    val_x, val_y = x[validation][:, active], y[validation]
    if engine == "lgb":
        from lightgbm import early_stopping, log_evaluation

        candidate.fit(
            x[older][:, active],
            y[older],
            eval_set=[(val_x, val_y)],
            callbacks=[early_stopping(patience, verbose=False), log_evaluation(0)],
        )
        trees = int(candidate.best_iteration_)
    elif engine == "xgb":
        candidate.fit(x[older][:, active], y[older], eval_set=[(val_x, val_y)], verbose=False)
        trees = int(candidate.best_iteration) + 1
    else:
        candidate.fit(
            x[older][:, active],
            y[older],
            eval_set=(val_x, val_y),
            early_stopping_rounds=patience,
            use_best_model=True,
        )
        trees = int(candidate.get_best_iteration()) + 1
    trees = max(1, trees)
    # Refit uses all completed years, including the tree-count validation year.
    # Recompute available columns on that final training set only.
    active = np.flatnonzero(np.any(np.isfinite(x), axis=0))
    final = model(trees)
    final.fit(x[:, active], y)
    prediction = final.predict(test_x[:, active])
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite adaptive forecast")
    return prediction, dict(
        trees=trees,
        ceiling=ceiling,
        validation_year=validation_year,
        train_last_year=int(years.max()),
        inner_train_last_year=int(years[older].max()),
        ceiling_selected=trees == ceiling,
    )


def task(run_path, data_path, position, horizon, engine, view, config):
    run, data = Path(run_path), Path(data_path)
    panel, x, y = load_task(data, position, horizon)
    cols = json.loads((data / "features.json").read_text())["views"][view]
    x = x[:, cols]
    years, e = panel["year"].to_numpy(), panel["exposure"].to_numpy()
    folder = run / "fits" / f"{position}_{horizon}"
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for year in range(config["warmup_start"], max(config["evaluation_years"]) + 1):
            path = folder / f"{year}_auto_{engine}_{view}.npz"
            if path.exists() and path.with_suffix(".json").exists():
                saved = json.loads(path.with_suffix(".json").read_text())
                if digest(path) != saved["sha256"]:
                    raise ValueError("Adaptive capacity checkpoint changed")
                continue
            train, test = years < year, years == year
            start = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pred, diag = fit_adaptive(
                    engine, x[train], y[train, 0] / e[train], years[train], x[test]
                )
            np.savez_compressed(path, prediction=pred * e[test])
            diag.update(
                engine=engine,
                view=view,
                year=year,
                position=position,
                horizon=horizon,
                elapsed=time.monotonic() - start,
                warnings=sorted({str(w.message) for w in caught}),
                sha256=digest(path),
            )
            write_json(path.with_suffix(".json"), diag)
    return f"{position}/{horizon}/auto_{engine}_{view}: complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="deep_capacity_001")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("Invalid run id")
    data, run = LAB / "runs/deep_data_001", LAB / "runs" / args.run_id
    config = json.loads((data / "config.json").read_text())
    source = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("run.py"),
            Path(__file__).with_name("models.py"),
        ]
    }
    protocol = dict(
        config=config,
        data_sha256=validate_data(data),
        source=source,
        ceiling=4096,
        patience=80,
        engines=ENGINES,
        views=VIEWS,
        environment={
            p: version(p)
            for p in ["numpy", "scipy", "polars", "scikit-learn", "lightgbm", "xgboost", "catboost"]
        },
    )
    if args.resume:
        if json.loads((run / "protocol.json").read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("Adaptive resume requires identical code and inputs")
    else:
        run.mkdir(exist_ok=False)
        write_json(run / "protocol.json", protocol)
        for name in source:
            dest = run / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
    record = dict(status="running", started=datetime.now(UTC).isoformat(), research_only=True)
    write_json(run / "manifest.json", record)
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(task, str(run), str(data), p, h, engine, view, config)
                for p in config["positions"]
                for h in config["horizons"]
                for engine in ENGINES
                for view in VIEWS
            ]
            for i, f in enumerate(as_completed(futures), 1):
                print(f"[{i}/{len(futures)}] {f.result()}", flush=True)
        if any(digest(ROOT / p) != sha for p, sha in source.items()):
            raise ValueError("Adaptive capacity implementation changed")
        record.update(
            status="complete",
            ended=datetime.now(UTC).isoformat(),
            artifacts={
                str(p.relative_to(run)): digest(p) for p in (run / "fits").rglob("*") if p.is_file()
            },
        )
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write_json(run / "manifest.json", record)


if __name__ == "__main__":
    main()
