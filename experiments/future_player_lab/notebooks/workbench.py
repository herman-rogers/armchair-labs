"""Verified historical comparisons and chronological, editable tree experiments."""

import hashlib
import json
import re
import shutil
import time
import uuid
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from ..data import digest
from ..deep.ensemble import capture_score, convex_weights, greedy_weights
from ..deep.models import Discover, names_for
from ..deep.run import LAB, ROOT
from ..evaluate import paired_interval, score

DATA = LAB / "runs/deep_data_001"
FITS = LAB / "runs/deep_search_003"
CAPACITY = LAB / "runs/deep_capacity_002"
REPORT = LAB / "runs/deep_report_001"
DIAGNOSTICS = LAB / "runs/deep_diagnostics_001"
KEYS = ["player_id", "position", "year", "origin", "horizon"]
K = {"QB": 10, "RB": 20, "WR": 30, "TE": 10}
POSITION_TARGETS = {
    "QB": [
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ],
    "RB": [
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ],
    "WR": [
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ],
    "TE": [
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ],
}


def target_label(target):
    return {
        "points": "Fantasy points",
        "attempts": "Pass attempts",
        "passing_tds": "Passing touchdowns",
        "rushing_tds": "Rushing touchdowns",
        "receiving_tds": "Receiving touchdowns",
    }.get(target, target.replace("_", " ").capitalize())


def target_unit(target):
    return (
        "points"
        if target == "points"
        else "yards"
        if "yards" in target
        else target_label(target).lower()
    )


@lru_cache(maxsize=1)
def target_weeks():
    config = read_json(DATA / "config.json")
    root = ROOT / config["gold_root"]
    if digest(root / "manifest.json") != config["gold_manifest_sha256"]:
        raise ValueError("Gold target manifest changed")
    spec = read_json(root / "manifest.json")["tables"]["nfl_player_weeks"]
    path = root / spec["path"]
    if digest(path) != spec["sha256"]:
        raise ValueError("Gold weekly targets changed")
    return pl.read_parquet(path).filter(pl.col("season_type") == "REG")


def stat_outcomes(panel, weeks, target):
    """Exact future interval; absent completed records are zero, recorded nulls unknown."""
    if weeks.select("player_id", "season", "week").is_duplicated().any():
        raise ValueError("Duplicate weekly target observation")
    candidates = panel.with_row_index("_row")
    joined = (
        candidates.select("_row", "player_id", "year", "origin", "end")
        .join(
            weeks.select("player_id", pl.col("season").alias("year"), "week", target),
            on=["player_id", "year"],
            how="inner",
        )
        .filter((pl.col("week") > pl.col("origin")) & (pl.col("week") <= pl.col("end")))
    )
    totals = joined.group_by("_row").agg(
        pl.when(pl.col(target).is_null().any() | pl.col(target).cast(pl.Float64).is_nan().any())
        .then(None)
        .otherwise(pl.col(target).sum())
        .alias("_total"),
        pl.len().alias("_observations"),
    )
    result = candidates.join(totals, on="_row", how="left").sort("_row")
    return result.with_columns(
        pl.when(pl.col("actual").is_null() | pl.col("actual").is_nan())
        .then(None)
        .when(pl.col("_observations").is_null())
        .then(0.0)
        .otherwise(pl.col("_total"))
        .alias("actual")
    ).drop("_row", "_total", "_observations")


def stat_panel(panel, weeks, target):
    future = stat_outcomes(panel, weeks, target)
    # Persistence knows only observations at/before the forecast cutoff.
    past = panel.with_columns(
        pl.when(pl.col("origin") == 0)
        .then(pl.col("year") - 1)
        .otherwise(pl.col("year"))
        .alias("year"),
        pl.when(pl.col("origin") == 0)
        .then(pl.when(pl.col("year") - 1 >= 2021).then(18).otherwise(17))
        .otherwise(pl.col("origin"))
        .alias("end"),
        pl.lit(0).alias("origin"),
    )
    observed = stat_outcomes(past, weeks, target)
    prediction = (
        observed["actual"].to_numpy() / past["end"].to_numpy() * panel["exposure"].to_numpy()
    )
    return future.with_columns(pl.Series("persistence_prediction", prediction))


def read_json(file):
    file = Path(file)
    stat = file.stat()
    return _json_version(file, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
def _json_version(file, modified, size):
    return json.loads(file.read_text())


@lru_cache(maxsize=2048)
def _verify_version(file, expected, modified, size):
    if digest(file) != expected:
        raise ValueError(f"Changed artifact: {file}")


def checked(run, relative):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    if manifest["status"] not in {"complete", "fits_complete"}:
        raise ValueError(f"Incomplete source: {run.name}")
    file = run / relative
    stat = file.stat()
    _verify_version(file, manifest["artifacts"][relative], stat.st_mtime_ns, stat.st_size)
    return file


@lru_cache(maxsize=2)
def task_data(position, horizon):
    panel = pl.read_parquet(checked(DATA, "panel.parquet"))
    ix = np.flatnonzero(
        (panel["position"].to_numpy() == position) & (panel["horizon"].to_numpy() == horizon)
    )
    x = np.load(checked(DATA, "x.npy"), mmap_mode="r")[ix]
    y = np.load(checked(DATA, "targets.npy"), mmap_mode="r")[ix, 0]
    return panel[ix].with_columns(pl.Series("actual", y)), x


def model_menu(horizon):
    protocol = read_json(FITS / "protocol.json")
    names = (
        ["persistence"]
        + [n for p in protocol["paths"] for n in names_for(p, protocol["config"])]
        + [f"auto_{e}_{v}" for e in ["lgb", "xgb", "cat"] for v in ["all", "market"]]
    )
    names += [
        f"new:{n}" for n in ["top_3", "policy_mse", "policy_mae", "policy_rank", "policy_capture"]
    ]
    if horizon in {"next_four", "remaining"}:
        names = [
            "published",
            "existing:profile_boost",
            "existing:enriched_boost",
            "existing:profile_ridge",
        ] + names
        names += [
            f"combined:{n}"
            for n in ["top_3", "policy_mse", "policy_mae", "policy_rank", "policy_capture"]
        ]
    return names


def label(name):
    return {
        "published": "Published serving policy",
        "existing:profile_boost": "Original • profile booster",
        "existing:enriched_boost": "Original • enriched booster",
        "existing:profile_ridge": "Original • profile ridge",
        "new:top_3": "New library • top-three blend",
        "combined:top_3": "New + existing • top-three blend",
        "combined:policy_mae": "New + existing • MAE selector",
        "combined:policy_mse": "New + existing • MSE selector",
    }.get(name, name)


@lru_cache(maxsize=96)
def one_forecast(position, horizon, name):
    panel, _ = task_data(position, horizon)
    columns = KEYS + ["name", "population", "end", "actual", "exposure"]
    if name.startswith(("new:", "combined:")):
        library, method = name.split(":", 1)
        relative = "published_library/" if library == "combined" else ""
        relative += f"{position}_{horizon}/forecasts.parquet"
        frame = pl.read_parquet(checked(REPORT, relative)).filter(pl.col("model") == method)
        frame = frame.join(panel.select(*KEYS, "exposure"), on=KEYS, validate="1:1")
        frame = frame.select(*columns, "prediction")
    elif name == "published" or name.startswith("existing:"):
        old = pl.read_parquet(checked(REPORT, "published_reconciled.parquet"))
        old = old.filter((pl.col("position") == position) & (pl.col("horizon") == horizon))
        frame = panel.join(
            old.select(*KEYS, "published_end", "published_actual", name),
            on=KEYS,
            how="inner",
            validate="1:1",
        )
        if (frame["end"] != frame["published_end"]).any() or not np.allclose(
            frame["actual"], frame["published_actual"], atol=1e-7, rtol=0
        ):
            raise ValueError("Historical baseline label or horizon mismatch")
        frame = frame.select(*columns, pl.col(name).alias("prediction"))
    else:
        years = panel["year"].to_numpy()
        keep = years >= 2013
        if name == "persistence":
            entire = pl.read_parquet(checked(DATA, "panel.parquet"))
            ix = (entire["position"].to_numpy() == position) & (
                entire["horizon"].to_numpy() == horizon
            )
            pred = np.load(checked(DATA, "baseline.npy"), mmap_mode="r")[ix, 0][keep]
        else:
            pred = np.empty(keep.sum())
            for year in sorted(set(years[keep])):
                path = name if name.startswith("auto_") else name.rsplit("_t", 1)[0]
                run = CAPACITY if name.startswith("auto_") else FITS
                relative = f"fits/{position}_{horizon}/{year}_{path}.npz"
                with np.load(checked(run, relative)) as saved:
                    pred[years[keep] == year] = saved["prediction" if run == CAPACITY else name]
        frame = (
            panel.filter(pl.Series(keep))
            .select(columns)
            .with_columns(pl.Series("prediction", pred))
        )
    if not np.isfinite(frame["prediction"].to_numpy()).all():
        raise ValueError(f"Incomplete forecasts: {name}")
    return frame.with_columns(pl.lit(name).alias("model"))


def comparison_frame(position, horizon, names, first=2019, last=2025, cohort="all"):
    if not names:
        raise ValueError("Select at least one model")
    frames = [
        one_forecast(position, horizon, n).filter(pl.col("year").is_between(first, last))
        for n in names
    ]
    if cohort != "all":
        frames = [f.filter(pl.col("population") == cohort) for f in frames]
    common = frames[0].select(KEYS)
    for f in frames[1:]:
        common = common.join(f.select(KEYS), on=KEYS, how="inner", validate="1:1")
    if not common.height:
        raise ValueError("No shared forecast rows for these filters")
    return pl.concat([f.join(common, on=KEYS, how="inner", validate="1:1") for f in frames])


def annual_scores(frame):
    records = []
    for (model, year), part in frame.partition_by(["model", "year"], as_dict=True).items():
        part = part.sort("player_id")
        y, pred, origin = (part[c].to_numpy() for c in ["actual", "prediction", "origin"])
        stats = score(y, pred, origin)
        records.append(
            dict(
                model=model,
                year=year,
                **stats,
                capture=capture_score(y, pred, origin, K[part["position"][0]]),
            )
        )
    return pl.DataFrame(records)


def summary_scores(annual, reference):
    rows = []
    baseline = annual.filter(pl.col("model") == reference).sort("year")
    for (model,), part in annual.partition_by("model", as_dict=True).items():
        part = part.sort("year")
        if not part["year"].equals(baseline["year"]):
            raise ValueError("Annual comparison has unmatched years")
        gains = (baseline["mae"] - part["mae"]).to_numpy()
        interval = paired_interval(gains, 5000, 20260925)
        rows.append(
            dict(
                model=model,
                label=label(model),
                rows=int(part["n"].sum()),
                years=part.height,
                rmse=float(np.sqrt(part["mse"].mean())),
                mae=part["mae"].mean(),
                ndcg24=part["ndcg24"].mean(),
                capture=part["capture"].mean(),
                mae_gain=float(gains.mean()),
                mae_gain_low=interval[0],
                mae_gain_high=interval[1],
                mse_reduction_pct=100 * (1 - part["mse"].mean() / baseline["mse"].mean()),
            )
        )
    return pl.DataFrame(rows)


def data_profile(position, horizon):
    panel, x = task_data(position, horizon)
    meta = read_json(checked(DATA, "features.json"))
    finite = np.isfinite(x)
    features = pl.DataFrame(
        {
            "feature": meta["names"],
            "group": meta["groups"],
            "observed_fraction": finite.mean(axis=0),
            "missing_fraction": 1 - finite.mean(axis=0),
        }
    )
    return panel, features


def saved_capacity(position, horizon, path):
    panel, _ = task_data(position, horizon)
    records = []
    for year in range(2019, 2026):
        relative = f"fits/{position}_{horizon}/{year}_{path}"
        diag = read_json(checked(FITS, relative + ".json"))
        part = panel.filter(pl.col("year") == year)
        e = part["exposure"].to_numpy()
        with np.load(checked(FITS, relative + ".npz")) as saved:
            for cap in diag["capacity"]:
                n = cap["trees"]
                test_mse = float(
                    np.mean(((saved[f"{path}_t{n}"] - part["actual"].to_numpy()) / e) ** 2)
                )
                for split, mse in [("Training", cap["train_mse_rate"]), ("Later season", test_mse)]:
                    records.append(
                        dict(
                            year=year,
                            trees=n,
                            split=split,
                            rmse_rate=float(np.sqrt(mse)),
                            train_rows=diag["train_rows"],
                        )
                    )
    return pl.DataFrame(records)


def original_root():
    config = read_json(DATA / "config.json")
    return ROOT / config["published_root"]


def sandbox_data(position, horizon, view):
    if view not in {"original_profile", "original_enriched"}:
        panel, x = task_data(position, horizon)
        meta = read_json(checked(DATA, "features.json"))
        cols = meta["views"][view]
        return panel, x[:, cols], [meta["names"][j] for j in cols], False
    if horizon not in {"next_four", "remaining"}:
        raise ValueError("Original production inputs exist only for next-four/remaining forecasts")
    root = original_root()
    # The pinned published manifest binds the original input panel as well as predictions.
    config = read_json(DATA / "config.json")
    from ..deep.data import verified_files

    verified_files(
        ROOT, config["published_root"], ["ranking_panel.parquet"], config["pinned_manifests"]
    )
    frame = (
        pl.read_parquet(root / "ranking_panel.parquet")
        .rename(
            {
                "season": "year",
                "through_week": "origin",
                "end_week": "end",
                "player_display_name": "name",
            }
        )
        .with_columns(
            pl.col("horizon").replace({"next4": "next_four", "rest_of_season": "remaining"})
        )
    )
    frame = frame.filter(
        (pl.col("position") == position)
        & (pl.col("horizon") == horizon)
        & pl.col("actual").is_not_null()
    ).sort("year", "player_id")
    cols = sorted(c for c in frame.columns if c.startswith("x_"))
    if view == "original_enriched":
        cols += sorted(c for c in frame.columns if c.startswith("e_"))
    cols += ["scheduled_games"]
    # Preserve original training labels; evaluate against reconciled canonical labels.
    bench = pl.read_parquet(checked(REPORT, "published_reconciled.parquet"))
    frame = (
        frame.rename({"actual": "training_actual"})
        .join(
            bench.select(*KEYS, pl.col("published_actual").alias("actual")),
            on=KEYS,
            how="left",
            validate="1:1",
        )
        .with_columns(pl.lit(1.0).alias("exposure"))
    )
    return frame, frame.select(cols).to_numpy().astype(float), cols, True


DEFAULT_TRIAL = dict(
    position="TE",
    horizon="remaining",
    view="basic",
    engine="hist",
    representation="identity",
    year=2025,
    max_trees=120,
    learning_rate=0.05,
    leaves=15,
    depth=6,
    leaf_samples=30,
    l2=10.0,
    feature_fraction=1.0,
    history_years=0,
    half_life=0,
    seed=20260923,
    objective="mse",
    feature_scope="all",
    target="points",
)


def feature_category(name):
    """Classify dated inputs by football meaning, independent of outcomes."""
    stat = name.split(":")[-1]
    # Team passing opportunity is relevant to receivers and receiving backs.
    if any(s in stat for s in ("team_", "neutral_pass", "vacated_")):
        return "Team context"
    if re.search(
        r"(^|_)(passing|pass|attempts|completions|interceptions|sacks|cpoe|qb)(_|$)", stat
    ):
        return "Player passing"
    if any(
        s in stat
        for s in (
            "receiving",
            "receptions",
            "targets",
            "target_share",
            "air_yard",
            "route",
            "separation",
            "yac",
            "wopr",
        )
    ):
        return "Player receiving"
    if any(s in stat for s in ("rushing", "rush_", "carries", "carry_share", "ryoe")):
        return "Player rushing"
    return "Shared context / production"


def feature_audit(names, position, scope):
    if position not in K or scope not in {"all", "position"}:
        raise ValueError("Choose QB/RB/WR/TE and all/position feature scope")
    # Rushing remains available to WR/TE (end-arounds and hybrid roles).
    allowed = (
        {"Player passing", "Player rushing"}
        if position == "QB"
        else {"Player rushing", "Player receiving"}
    )
    return pl.DataFrame(
        [
            dict(
                feature=n,
                category=feature_category(n),
                included=(
                    scope == "all"
                    or feature_category(n) in allowed
                    or feature_category(n) in {"Team context", "Shared context / production"}
                ),
            )
            for n in names
        ]
    )


def validate_trial(config):
    c = {**DEFAULT_TRIAL, **config}
    if c["position"] not in K or c["feature_scope"] not in {"all", "position"}:
        raise ValueError("Unknown position or feature scope")
    if c["target"] not in ["points", *POSITION_TARGETS[c["position"]]]:
        raise ValueError("Choose a relevant prediction target for this position")
    if c["target"] != "points" and c["view"].startswith("original_"):
        raise ValueError(
            "Original presets predict fantasy points. Choose a new preset for individual stats."
        )
    # Browser JSON represents 1.0 as 1; sklearn distinguishes float feature
    # fractions from integer feature counts. Canonicalize controls before fitting
    # and hashing so equal settings also reuse the same saved trial.
    for key in ["learning_rate", "l2", "feature_fraction"]:
        c[key] = float(c[key])
        if not np.isfinite(c[key]):
            raise ValueError(f"Nonfinite parameter: {key}")
    for key in [
        "year",
        "max_trees",
        "leaves",
        "depth",
        "leaf_samples",
        "history_years",
        "half_life",
        "seed",
    ]:
        if not float(c[key]).is_integer():
            raise ValueError(f"{key} must be an integer")
        c[key] = int(c[key])
    if c["engine"] not in {"hist", "lgb", "xgb", "cat"}:
        raise ValueError("Unknown tree engine")
    if c["representation"] not in {"identity", "selected", "interactions", "leaf"}:
        raise ValueError("Unknown representation")
    if c["objective"] not in {"mse", "mae"} or not 2019 <= c["year"] <= 2025:
        raise ValueError("Use a completed evaluation year and MSE/MAE objective")
    if not 2 <= c["max_trees"] <= 4096 or not 0 < c["learning_rate"] <= 1:
        raise ValueError("Invalid capacity or learning rate")
    if not 0 < c["feature_fraction"] <= 1 or c["leaf_samples"] < 1 or c["l2"] < 0:
        raise ValueError("Invalid regularization")
    if c["depth"] < 0 or c["leaves"] < 2 or c["history_years"] < 0 or c["half_life"] < 0:
        raise ValueError("Invalid tree structure/history")
    if c["engine"] in {"cat", "xgb"} and not 1 <= c["depth"] <= 12:
        raise ValueError("Use depth 1–12 for CatBoost/XGBoost")
    return c


def fit_prefixes(x, target, years, evaluation, c, counts, original=False):
    """Transform on training rows only; return exact prefixes of one fitted booster."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    active = np.flatnonzero(np.any(np.isfinite(x), axis=0))
    if original:
        active = np.array([j for j in active if len(np.unique(x[np.isfinite(x[:, j]), j])) > 1])
    if not len(active):
        raise ValueError("Training rows have no usable features")
    train, arrays = x[:, active], [z[:, active] for z in evaluation]
    weights = (
        np.ones(len(target))
        if not c["half_life"]
        else 2.0 ** ((years - years.max()) / c["half_life"])
    )
    discovered = {}
    if c["representation"] != "identity":
        discovery = Discover(c["representation"], c["seed"]).fit(train, target, weights)
        discovered = {"selected_indices": active[discovery.evidence["selected_indices"]].tolist()}
        train, arrays = discovery.transform(train), [discovery.transform(z) for z in arrays]
    maximum = max(counts)
    if c["engine"] == "hist":
        model = HistGradientBoostingRegressor(
            max_iter=maximum,
            learning_rate=c["learning_rate"],
            max_leaf_nodes=c["leaves"],
            max_depth=c["depth"] or None,
            min_samples_leaf=c["leaf_samples"],
            l2_regularization=c["l2"],
            max_features=c["feature_fraction"],
            max_bins=63,
            early_stopping=False,
            random_state=c["seed"],
        )
    elif c["engine"] == "lgb":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=maximum,
            learning_rate=c["learning_rate"],
            num_leaves=c["leaves"],
            max_depth=c["depth"] or -1,
            min_child_samples=c["leaf_samples"],
            reg_lambda=c["l2"],
            colsample_bytree=c["feature_fraction"],
            max_bin=63,
            n_jobs=1,
            verbosity=-1,
            random_state=c["seed"],
        )
    elif c["engine"] == "xgb":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=maximum,
            learning_rate=c["learning_rate"],
            max_depth=c["depth"],
            min_child_weight=c["leaf_samples"],
            reg_lambda=c["l2"],
            colsample_bytree=c["feature_fraction"],
            subsample=0.8,
            max_bin=63,
            tree_method="hist",
            n_jobs=1,
            random_state=c["seed"],
        )
    else:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=maximum,
            learning_rate=c["learning_rate"],
            depth=c["depth"],
            l2_leaf_reg=c["l2"],
            rsm=c["feature_fraction"],
            border_count=63,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            random_seed=c["seed"],
        )
    model.fit(train, target, sample_weight=weights)
    result = {n: [] for n in counts}
    for z in [train, *arrays]:
        if c["engine"] == "hist":
            values = {i: pred for i, pred in enumerate(model.staged_predict(z), 1) if i in counts}
        else:
            values = {
                n: model.predict(
                    z,
                    **(
                        {"num_iteration": n}
                        if c["engine"] == "lgb"
                        else {"iteration_range": (0, n)}
                        if c["engine"] == "xgb"
                        else {"ntree_end": n}
                    ),
                )
                for n in counts
            }
        for n, pred in values.items():
            nonnegative = c.get("target", "points") not in {
                "points",
                "passing_yards",
                "rushing_yards",
                "receiving_yards",
            }
            result[n].append(np.maximum(pred, 0) if original or nonnegative else pred)
    return result, {"active_columns": len(active), **discovered}


def compute_trial(panel, x, names, c, original=False):
    """Fixed latest-year test; earlier validation chooses capacity, never test labels."""
    c = validate_trial(c)
    audit = feature_audit(names, c["position"], c["feature_scope"])
    keep = np.flatnonzero(audit["included"].to_numpy())
    x, names = x[:, keep], [names[j] for j in keep]
    years = panel["year"].to_numpy()
    y = panel["actual"].to_numpy()
    fit_y = panel["training_actual"].to_numpy() if "training_actual" in panel else y
    exposure = panel["exposure"].to_numpy()
    test, val = years == c["year"], years == c["year"] - 1
    train = years < c["year"] - 1
    if c["history_years"]:
        train &= years >= c["year"] - 1 - c["history_years"]
    if min(train.sum(), val.sum(), test.sum()) == 0:
        raise ValueError("Training, validation and test must each contain rows")
    if not np.isfinite(fit_y[train | val]).all() or not np.isfinite(y[val | test]).all():
        raise ValueError("Unknown outcomes cannot be used as training or evaluation labels")
    counts = sorted(
        set(
            [min(n, c["max_trees"]) for n in [10, 20, 40, 80, 120, 240, 360, 720, 1440, 2880]]
            + [c["max_trees"]]
        )
    )
    stages, discovery = fit_prefixes(
        x[train],
        fit_y[train] / exposure[train],
        years[train],
        [x[val], x[test]],
        c,
        counts,
        original,
    )
    curves = []
    for n in counts:
        for split, mask, pred in zip(
            ["Training", "Validation", "Test"], [train, val, test], stages[n], strict=True
        ):
            estimate = pred * exposure[mask]
            if original:
                estimate = np.where(panel["scheduled_games"].to_numpy()[mask] == 0, 0, estimate)
            truth = fit_y[mask] if split == "Training" else y[mask]
            err = estimate - truth
            curves.append(
                dict(
                    trees=n,
                    split=split,
                    n=int(mask.sum()),
                    mse=float(np.mean(err**2)),
                    rmse=float(np.sqrt(np.mean(err**2))),
                    mae=float(np.mean(abs(err))),
                )
            )
    selected = min(
        (r for r in curves if r["split"] == "Validation"),
        key=lambda r: (r[c["objective"]], r["trees"]),
    )["trees"]
    learning = []
    starts = sorted(
        set(
            [
                int(years[train].min()),
                *[max(int(years[train].min()), c["year"] - 1 - n) for n in [3, 5, 10]],
            ]
        ),
        reverse=True,
    )
    for start in starts:
        take = train & (years >= start)
        pred, _ = fit_prefixes(
            x[take],
            fit_y[take] / exposure[take],
            years[take],
            [x[val], x[test]],
            c,
            [selected],
            original,
        )
        for split, mask, values in zip(
            ["Training", "Validation", "Test"], [take, val, test], pred[selected], strict=True
        ):
            estimate = values * exposure[mask]
            if original:
                estimate = np.where(panel["scheduled_games"].to_numpy()[mask] == 0, 0, estimate)
            truth = fit_y[mask] if split == "Training" else y[mask]
            learning.append(
                dict(
                    train_rows=int(take.sum()),
                    first_year=start,
                    last_train_year=c["year"] - 2,
                    split=split,
                    rmse=float(np.sqrt(np.mean((estimate - truth) ** 2))),
                )
            )
    refit = train | val
    final, refit_discovery = fit_prefixes(
        x[refit], fit_y[refit] / exposure[refit], years[refit], [x[test]], c, [selected], original
    )
    prediction = final[selected][1] * exposure[test]
    if original:
        prediction = np.where(panel["scheduled_games"].to_numpy()[test] == 0, 0, prediction)
    forecast = (
        panel.filter(pl.Series(test))
        .select(
            *KEYS,
            "name",
            "population",
            "actual",
            *(["persistence_prediction"] if "persistence_prediction" in panel else []),
        )
        .with_columns(pl.Series("prediction", prediction))
    )
    return dict(
        features=audit,
        capacity=pl.DataFrame(curves),
        learning=pl.DataFrame(learning),
        forecasts=forecast,
        metadata=dict(
            selected_trees=selected,
            validation_year=c["year"] - 1,
            test_year=c["year"],
            train_first_year=int(years[train].min()),
            train_last_year=c["year"] - 2,
            refit_last_year=c["year"] - 1,
            training_rows=int(train.sum()),
            validation_rows=int(val.sum()),
            test_rows=int(test.sum()),
            discovery=discovery,
            refit_discovery=refit_discovery,
            selected_features=[names[j] for j in refit_discovery.get("selected_indices", [])],
            feature_scope=c["feature_scope"],
            input_features=names,
            excluded_features=audit.filter(~pl.col("included"))["feature"].to_list(),
            target=c["target"],
            units=target_unit(c["target"]),
            target_definition="Original total points"
            if original
            else f"{target_label(c['target'])} per calendar week; evaluated as period totals",
        ),
    )


def trial_key(config):
    provenance = dict(
        config=validate_trial(config),
        data_sha256=digest(DATA / "manifest.json"),
        original_sha256=digest(original_root() / "manifest.json"),
        implementation={
            str(p.relative_to(ROOT)): digest(p)
            for p in [Path(__file__), LAB / "deep/models.py", LAB / "evaluate.py"]
        },
        packages={
            p: version(p)
            for p in ["numpy", "polars", "scikit-learn", "lightgbm", "xgboost", "catboost"]
        },
    )
    return hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[
        :20
    ], provenance


def run_trial(config):
    key, provenance = trial_key(config)
    out = LAB / "runs" / ("notebook_tree_" + key)
    if not out.exists():
        temp = out.with_name(out.name + ".tmp-" + uuid.uuid4().hex)
        temp.mkdir()
        start = time.monotonic()
        try:
            c = provenance["config"]
            panel, x, names, original = sandbox_data(c["position"], c["horizon"], c["view"])
            excluded_labels = {}
            if c["target"] != "points":
                panel = stat_panel(panel, target_weeks(), c["target"])
                known = np.isfinite(panel["actual"].to_numpy())
                excluded_labels = dict(
                    panel.filter(pl.Series(~known)).group_by("year").len().iter_rows()
                )
                panel, x = panel.filter(pl.Series(known)), x[known]
            with threadpool_limits(limits=1):
                result = compute_trial(panel, x, names, c, original)
            for name in ["capacity", "learning", "forecasts", "features"]:
                result[name].write_parquet(temp / (name + ".parquet"))
            metadata = {
                **result["metadata"],
                "seconds": time.monotonic() - start,
                "provenance": provenance,
                "unknown_target_rows_excluded_by_year": excluded_labels,
            }
            (temp / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            (temp / "manifest.json").write_text(
                json.dumps(
                    dict(
                        status="complete",
                        research_only=True,
                        artifacts={p.name: digest(p) for p in temp.iterdir()},
                    ),
                    indent=2,
                )
                + "\n"
            )
            temp.rename(out)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return load_trial(out)


def cached_trial(config):
    key, _ = trial_key(config)
    out = LAB / "runs" / ("notebook_tree_" + key)
    return load_trial(out) if out.exists() else None


def run_target_trials(config, cached=False):
    targets = (
        POSITION_TARGETS[config["position"]]
        if config.get("target") == "all_stats"
        else [config.get("target", "points")]
    )
    trials = {}
    for target in targets:
        trial = (cached_trial if cached else run_trial)({**config, "target": target})
        if trial is not None:
            trials[target] = trial
    return trials


def stat_line(trials):
    frames = [
        t["forecasts"]
        .select(*KEYS, "name", "prediction")
        .with_columns(pl.lit(target).alias("target"))
        for target, t in trials.items()
    ]
    return (
        pl.concat(frames)
        .pivot(on="target", index=KEYS + ["name"], values="prediction")
        .sort("name")
    )


def load_trial(out):
    result = {
        name: pl.read_parquet(checked(out, name + ".parquet"))
        for name in ["capacity", "learning", "forecasts"]
    }
    if "features.parquet" in read_json(out / "manifest.json")["artifacts"]:
        result["features"] = pl.read_parquet(checked(out, "features.parquet"))
    return {**result, "metadata": read_json(checked(out, "metadata.json")), "path": out}


def blend_trial(frame, year, method="convex", shrink=0.1, steps=20):
    names = sorted(frame["model"].unique().to_list())
    wide = frame.pivot(
        on="model", index=KEYS + ["actual", "name", "population"], values="prediction"
    ).sort("year", "player_id")
    p, y, years = wide.select(names).to_numpy(), wide["actual"].to_numpy(), wide["year"].to_numpy()
    train, test = (years < year) & (years >= year - 5), years == year
    if len(set(years[train])) < 3 or not test.any() or not np.isfinite(p).all():
        raise ValueError("Blending needs complete forecasts and three earlier evaluation seasons")
    w = np.array([1 / np.sum(years[train] == n) for n in years[train]])
    w /= w.sum()
    if method == "equal":
        weights = np.ones(len(names)) / len(names)
    elif method == "convex":
        weights = convex_weights(p[train], y[train], w, shrink)
    elif method == "greedy":
        weights = greedy_weights(p[train], y[train], w, [steps])[steps]
    else:
        raise ValueError("Unknown blend method")
    forecast = (
        wide.filter(pl.Series(test))
        .select(*KEYS, "actual", "name", "population")
        .with_columns(pl.Series("prediction", p[test] @ weights))
    )
    return forecast, pl.DataFrame({"model": names, "weight": weights}), sorted(set(years[train]))


def compare_trial(result):
    c = result["metadata"]["provenance"]["config"]
    if c.get("target", "points") != "points":
        f = result["forecasts"]
        rows = []
        for name, column in [
            ("Stat model", "prediction"),
            ("Prior production rate", "persistence_prediction"),
        ]:
            error = f[column].to_numpy() - f["actual"].to_numpy()
            rows.append(
                dict(
                    model=name,
                    target=c["target"],
                    units=target_unit(c["target"]),
                    rmse=float(np.sqrt(np.mean(error**2))),
                    mae=float(np.mean(abs(error))),
                )
            )
        return pl.DataFrame(rows)
    names = ["persistence", "new:top_3"]
    if c["horizon"] in {"next_four", "remaining"}:
        names = ["existing:profile_boost", "published", "combined:top_3"]
    baseline = comparison_frame(c["position"], c["horizon"], names, c["year"], c["year"])
    trial = result["forecasts"].with_columns(pl.lit("sandbox_refit").alias("model"))
    common = baseline.select(KEYS).unique().join(trial.select(KEYS), on=KEYS, validate="1:1")
    columns = KEYS + ["name", "population", "actual", "prediction", "model"]
    frames = [f.join(common, on=KEYS, validate="m:1").select(columns) for f in [baseline, trial]]
    check = (
        frames[0]
        .select(*KEYS, "actual")
        .unique()
        .join(
            frames[1].select(*KEYS, pl.col("actual").alias("trial_actual")), on=KEYS, validate="1:1"
        )
    )
    if not np.allclose(check["actual"], check["trial_actual"], rtol=0, atol=1e-7):
        raise ValueError("Sandbox and historical outcomes do not match")
    return annual_scores(pl.concat(frames))
