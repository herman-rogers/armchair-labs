"""BDB 2020 RB handoff experiment: player profile versus observed play context.

This is a separate at-handoff yardage question, never a season-forecast input.
Local legacy IDs join players within a play only; no canonical identity bridge is assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from patron.data.releases import digest, load_manifest
from patron.data.tracking import load_source

from .run import ROOT, RUNS, json_write, packages

KEY = ["GameId", "PlayId"]
PROFILE_NUMERIC = ["profile_height_inches", "profile_weight", "profile_age"]
PROFILE_CATEGORICAL = ["PlayerCollegeName"]
CONTEXT_CATEGORICAL = [
    "OffenseFormation",
    "OffensePersonnel",
    "DefensePersonnel",
    "PossessionTeam",
    "HomeTeamAbbr",
    "VisitorTeamAbbr",
    "FieldPosition",
    "StadiumType",
    "Turf",
    "GameWeather",
    "WindDirection",
]
CONTEXT_NUMERIC = [
    "Quarter",
    "Down",
    "Distance",
    "YardLine",
    "HomeScoreBeforePlay",
    "VisitorScoreBeforePlay",
    "DefendersInTheBox",
    "Temperature",
    "Humidity",
]
VIEWS = ("profile", "carrier_context", "all_players")
SEED = 20261003


def build_features(source: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Whitelist measurements observed by handoff; labels and IDs never enter x."""
    required = {
        *KEY,
        "NflId",
        "NflIdRusher",
        "Team",
        "Position",
        "Season",
        "Yards",
        "X",
        "Y",
        "S",
        "A",
        "Dis",
        "Dir",
        "Orientation",
        "PlayDirection",
    }
    if missing := required - set(source.columns):
        raise ValueError(f"Missing handoff source columns: {sorted(missing)}")
    if source.select([*KEY, "NflId"]).is_duplicated().any():
        raise ValueError("Duplicate player snapshot within a play")
    quality = source.group_by(KEY).agg(
        pl.len().alias("players"),
        pl.col("Yards").n_unique().alias("labels"),
        (pl.col("NflId") == pl.col("NflIdRusher")).sum().alias("rushers"),
    )
    if quality.filter(
        (pl.col("players") != 22) | (pl.col("labels") != 1) | (pl.col("rushers") != 1)
    ).height:
        raise ValueError("Each play requires 22 players, one rusher, and one consistent label")
    optional_numeric = [*CONTEXT_NUMERIC, "PlayerWeight"]
    optional_strings = [
        *PROFILE_CATEGORICAL,
        *CONTEXT_CATEGORICAL,
        "PlayerHeight",
        "PlayerBirthDate",
        "TimeSnap",
        "TimeHandoff",
        "GameClock",
        "WindSpeed",
    ]
    source = source.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in optional_numeric if c not in source]
        + [pl.lit(None, dtype=pl.String).alias(c) for c in optional_strings if c not in source]
    )
    right = pl.col("PlayDirection").str.to_lowercase() == "right"
    sign = pl.when(right).then(1.0).otherwise(-1.0)
    direction = pl.col("Dir") * np.pi / 180
    orientation = pl.col("Orientation") * np.pi / 180
    source = source.with_columns(
        pl.when(right).then(pl.col("X")).otherwise(120 - pl.col("X")).alias("x_attack"),
        pl.when(right).then(pl.col("Y")).otherwise(160 / 3 - pl.col("Y")).alias("y_attack"),
        (pl.col("S") * direction.sin() * sign).alias("vx"),
        (pl.col("S") * direction.cos() * sign).alias("vy"),
        (direction.sin() * sign).alias("direction_sin"),
        (direction.cos() * sign).alias("direction_cos"),
        (orientation.sin() * sign).alias("orientation_sin"),
        (orientation.cos() * sign).alias("orientation_cos"),
    )
    carrier = source.filter(
        (pl.col("NflId") == pl.col("NflIdRusher")) & (pl.col("Position") == "RB")
    )
    if not carrier.height:
        raise ValueError("No source-labelled RB carriers")
    snap = pl.col("TimeSnap").str.to_datetime(strict=False, time_zone="UTC")
    handoff = pl.col("TimeHandoff").str.to_datetime(strict=False, time_zone="UTC")
    birthday = pl.col("PlayerBirthDate").str.to_date("%m/%d/%Y", strict=False)
    carrier = carrier.with_columns(
        (
            pl.col("PlayerHeight").str.extract(r"^(\d+)-", 1).cast(pl.Float64, strict=False) * 12
            + pl.col("PlayerHeight").str.extract(r"-(\d+)$", 1).cast(pl.Float64, strict=False)
        ).alias("profile_height_inches"),
        pl.col("PlayerWeight").cast(pl.Float64).alias("profile_weight"),
        ((snap.dt.date() - birthday).dt.total_days() / 365.25).alias("profile_age"),
        (
            pl.col("GameClock").str.extract(r"^(\d+):", 1).cast(pl.Float64, strict=False) * 60
            + pl.col("GameClock").str.extract(r"^\d+:(\d+)", 1).cast(pl.Float64, strict=False)
        ).alias("quarter_seconds_left"),
        ((handoff - snap).dt.total_milliseconds() / 1000).alias("snap_to_handoff_seconds"),
        pl.col("WindSpeed")
        .str.extract(r"(\d+(?:\.\d+)?)", 1)
        .cast(pl.Float64, strict=False)
        .alias("wind_speed_first_numeric"),
    ).with_columns(
        pl.when(pl.col("snap_to_handoff_seconds") >= 0)
        .then(pl.col("snap_to_handoff_seconds"))
        .otherwise(None)
        .alias("snap_to_handoff_seconds")
    )
    motion = [
        "S",
        "A",
        "Dis",
        "x_attack",
        "y_attack",
        "vx",
        "vy",
        "direction_sin",
        "direction_cos",
        "orientation_sin",
        "orientation_cos",
    ]
    numeric = [
        *PROFILE_NUMERIC,
        *CONTEXT_NUMERIC,
        "quarter_seconds_left",
        "snap_to_handoff_seconds",
        "wind_speed_first_numeric",
        *[f"carrier_{c}" for c in motion],
    ]
    categories = [*PROFILE_CATEGORICAL, *CONTEXT_CATEGORICAL]
    panel = carrier.select(
        *KEY,
        "Season",
        pl.col("NflId").alias("source_player_id"),
        pl.col("Yards").cast(pl.Float64).alias("actual"),
        *PROFILE_NUMERIC,
        *CONTEXT_NUMERIC,
        "quarter_seconds_left",
        "snap_to_handoff_seconds",
        "wind_speed_first_numeric",
        *categories,
        *[pl.col(c).alias(f"carrier_{c}") for c in motion],
    )
    relative = (
        source.join(
            carrier.select(
                *KEY,
                pl.col("NflId").alias("carrier_id"),
                pl.col("Team").alias("carrier_team"),
                *[pl.col(c).alias(f"rusher_{c}") for c in ["x_attack", "y_attack", "vx", "vy"]],
            ),
            on=KEY,
            how="inner",
            validate="m:1",
        )
        .filter(pl.col("NflId") != pl.col("carrier_id"))
        .with_columns(
            (pl.col("x_attack") - pl.col("rusher_x_attack")).alias("dx"),
            (pl.col("y_attack") - pl.col("rusher_y_attack")).alias("dy"),
            (pl.col("vx") - pl.col("rusher_vx")).alias("relative_vx"),
            (pl.col("vy") - pl.col("rusher_vy")).alias("relative_vy"),
        )
        .with_columns((pl.col("dx").pow(2) + pl.col("dy").pow(2)).sqrt().alias("distance"))
    )
    spatial_names = []
    for label, same in [("teammates", True), ("defenders", False)]:
        players = relative.filter((pl.col("Team") == pl.col("carrier_team")) == same)
        expressions = [pl.len().alias(f"{label}__count")]
        for c in [
            "dx",
            "dy",
            "distance",
            "S",
            "A",
            "Dis",
            "vx",
            "vy",
            "relative_vx",
            "relative_vy",
            "PlayerWeight",
        ]:
            for stat in ("mean", "std", "min", "max"):
                name = f"{label}__{c}__{stat}"
                expressions.append(getattr(pl.col(c), stat)().alias(name))
        for radius in (2, 5, 10):
            expressions.append(
                (pl.col("distance") <= radius).sum().alias(f"{label}__within_{radius}_yards")
            )
        expressions.append((pl.col("dx") > 0).sum().alias(f"{label}__ahead_count"))
        aggregate = players.group_by(KEY).agg(expressions)
        spatial_names += [c for c in aggregate.columns if c not in KEY]
        panel = panel.join(aggregate, on=KEY, how="left", validate="1:1")
    numeric += spatial_names
    panel = panel.with_columns(
        [
            pl.when(pl.col(c).cast(pl.Float64).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
            for c in numeric
        ]
    ).sort("Season", *KEY)
    contract = {
        "numeric": numeric,
        "categorical": categories,
        "views": {
            "profile": {"numeric": PROFILE_NUMERIC, "categorical": PROFILE_CATEGORICAL},
            "carrier_context": {
                "numeric": [c for c in numeric if c not in spatial_names],
                "categorical": categories,
            },
            "all_players": {"numeric": numeric, "categorical": categories},
        },
        "clock": "At handoff, after observed formation and player motion; predicts play yards",
        "cohort": "Exactly one source Position=RB carrier per 22-player play",
        "id_policy": "Legacy NFL identifiers join snapshots only within the same GameId/PlayId",
    }
    return panel, contract


def load_play_dataset(cache_dir=RUNS / "gpu_raw_inputs_001"):
    data = ROOT / "data"
    gold = load_source(data, "bdb_2020")
    base, manifest = load_manifest(data, "enriched", gold.manifest["input"])
    contract = {
        "source": gold.ref,
        "enriched": gold.manifest["input"],
        "implementation_sha256": digest(Path(__file__)),
    }
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cache_dir) / f"bdb2020_plays_{key}"
    if (directory / "dataset.json").exists():
        metadata = json.loads((directory / "dataset.json").read_text())
        if digest(directory / "features.parquet") != metadata["feature_sha256"]:
            raise ValueError("BDB transformed feature cache changed")
        return pl.read_parquet(directory / "features.parquet"), metadata
    source = pl.read_parquet(base / manifest["tables"]["train"]["path"])
    panel, feature_contract = build_features(source)
    directory.mkdir(parents=True, exist_ok=False)
    panel.write_parquet(directory / "features.parquet")
    metadata = {
        "input": contract,
        **feature_contract,
        "cache_directory": str(directory),
        "feature_sha256": digest(directory / "features.parquet"),
        "rows": panel.height,
        "seasons": panel.group_by("Season")
        .agg(
            pl.len().alias("plays"),
            pl.col("GameId").n_unique().alias("games"),
            pl.col("source_player_id").n_unique().alias("players"),
        )
        .sort("Season")
        .to_dicts(),
    }
    json_write(directory / "dataset.json", metadata)
    return panel, metadata


class PlayPreprocessor:
    def __init__(self, view):
        self.numeric, self.categorical = view["numeric"], view["categorical"]
        self.imputer = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)

    def arrays(self, frame):
        numeric = frame.select(self.numeric).to_numpy().astype(np.float32)
        categorical = frame.select(
            [pl.col(c).fill_null("__MISSING__") for c in self.categorical]
        ).to_numpy()
        return numeric, categorical

    def fit(self, frame):
        numeric, categorical = self.arrays(frame)
        self.imputer.fit(numeric)
        self.encoder.fit(categorical)
        return self

    def transform(self, frame):
        numeric, categorical = self.arrays(frame)
        return np.ascontiguousarray(
            np.column_stack([self.imputer.transform(numeric), self.encoder.transform(categorical)]),
            dtype=np.float32,
        )

    def names(self):
        return [
            *self.imputer.get_feature_names_out(self.numeric).tolist(),
            *self.encoder.get_feature_names_out(self.categorical).tolist(),
        ]


def parameter_library():
    return [
        {
            "n_estimators": trees,
            "max_depth": depth,
            "learning_rate": rate,
            "min_child_weight": leaf,
            "reg_lambda": penalty,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "max_bin": 128,
        }
        for trees, depth, rate, leaf, penalty in [
            (100, 2, 0.05, 20, 10),
            (240, 2, 0.03, 20, 20),
            (160, 3, 0.04, 20, 10),
            (300, 3, 0.025, 50, 30),
            (160, 5, 0.035, 30, 20),
            (300, 5, 0.025, 50, 40),
        ]
    ]


def select_recipe(validation):
    return min(validation, key=lambda row: (row["mse"], row["config_index"]))["config_index"]


def residual_scores(actual, prediction, calibration_actual, calibration_prediction):
    """Exact empirical CRPS for all earlier residuals, computed without an N×M matrix."""
    y, pred = np.asarray(actual), np.asarray(prediction)
    residual = np.sort(np.asarray(calibration_actual) - np.asarray(calibration_prediction))
    if not len(residual) or not all(np.isfinite(a).all() for a in [y, pred, residual]):
        raise ValueError("Finite predictions, labels, and nonempty earlier residuals required")
    m = len(residual)
    cumulative = np.r_[0.0, np.cumsum(residual)]
    centered = y - pred
    k = np.searchsorted(residual, centered)
    absolute = (
        centered * k - cumulative[k] + cumulative[-1] - cumulative[k] - centered * (m - k)
    ) / m
    spread = np.sum(residual * (2 * np.arange(1, m + 1) - m - 1)) / m**2
    lo, hi = np.quantile(residual, [0.1, 0.9], method="inverted_cdf")
    return {
        "crps": absolute - spread,
        "lower80": pred + lo,
        "upper80": pred + hi,
        "covered80": ((y >= pred + lo) & (y <= pred + hi)).astype(float),
        "width80": np.full(len(y), hi - lo),
        "distribution_mean": pred + residual.mean(),
    }


def run(out, device="cuda"):
    from experiments.future_player_lab.gpu_backends import build_estimator

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    panel, metadata = load_play_dataset()
    train = panel.filter(pl.col("Season") == 2017)
    validation = panel.filter(pl.col("Season") == 2018)
    test = panel.filter(pl.col("Season") == 2019)
    refit = panel.filter(pl.col("Season") <= 2018)
    params = parameter_library()
    config = {
        "views": VIEWS,
        "parameters": params,
        "device": device,
        "seed": SEED,
        "training_season": 2017,
        "selection_and_calibration_season": 2018,
        "test_season": 2019,
        "refit_seasons": [2017, 2018],
        "input": metadata,
        "packages": packages(),
        "python": platform.python_version(),
    }
    json_write(out / "config.json", config)
    source_files = [
        Path(__file__),
        ROOT / "experiments/future_player_lab/gpu_backends.py",
        Path(__file__).parent / "run.py",
    ]
    implementation = out / "implementation"
    implementation.mkdir()
    for p in source_files:
        (implementation / p.name).write_bytes(p.read_bytes())
    results, selection, predictions, models = [], [], [], []
    y_train, y_val, y_test = [f["actual"].to_numpy() for f in [train, validation, test]]
    for view in VIEWS:
        preprocess = PlayPreprocessor(metadata["views"][view]).fit(train)
        tx, vx = preprocess.transform(train), preprocess.transform(validation)
        val_predictions = {}
        for i, spec in enumerate(params):
            model, backend = build_estimator("xgboost", spec, seed=SEED, device=device, threads=4)
            before = time.monotonic()
            model.fit(tx, y_train)
            pred = model.predict(vx)
            val_predictions[i] = pred
            record = {
                "view": view,
                "config_index": i,
                "season": 2018,
                "mse": float(np.mean((y_val - pred) ** 2)),
                "mae": float(np.mean(np.abs(y_val - pred))),
                "seconds": time.monotonic() - before,
                "features": tx.shape[1],
            }
            results.append(record)
            models.append({**record, "backend": backend})
            print(f"BDB {view} config{i}: validation RMSE={record['mse'] ** 0.5:.3f}", flush=True)
        winner = select_recipe([r for r in results if r["view"] == view])
        # Freeze choice before any test outcome is scored or fed to a fitting routine.
        selection.append({"view": view, "config_index": winner, "selection_max_season": 2018})
        preprocess = PlayPreprocessor(metadata["views"][view]).fit(refit)
        model, backend = build_estimator(
            "xgboost", params[winner], seed=SEED, device=device, threads=4
        )
        model.fit(preprocess.transform(refit), refit["actual"].to_numpy())
        pred = model.predict(preprocess.transform(test))
        scored = residual_scores(y_test, pred, y_val, val_predictions[winner])
        predictions.append(
            test.select(*KEY, "Season", "source_player_id", "actual").with_columns(
                pl.lit(view).alias("model"),
                pl.Series("prediction", pred),
                *[pl.Series(k, v) for k, v in scored.items()],
            )
        )
        np.savez_compressed(
            out / f"{view}_calibration.npz", actual=y_val, prediction=val_predictions[winner]
        )
        joblib.dump(
            {"preprocessor": preprocess, "model": model}, out / f"{view}.joblib", compress=3
        )
        model.estimator.save_model(out / f"{view}.ubj")
        json_write(
            out / f"{view}_feature_importance.json",
            [
                {"feature": n, "importance": float(v)}
                for n, v in sorted(
                    zip(preprocess.names(), model.feature_importances_, strict=True),
                    key=lambda item: -item[1],
                )
            ],
        )
        models.append(
            {"view": view, "config_index": winner, "phase": "final_refit", "backend": backend}
        )
    baseline_val = np.full(len(y_val), y_train.mean())
    baseline_test = np.full(len(y_test), refit["actual"].mean())
    scored = residual_scores(y_test, baseline_test, y_val, baseline_val)
    predictions.append(
        test.select(*KEY, "Season", "source_player_id", "actual").with_columns(
            pl.lit("training_mean").alias("model"),
            pl.Series("prediction", baseline_test),
            *[pl.Series(k, v) for k, v in scored.items()],
        )
    )
    all_predictions = pl.concat(predictions, how="diagonal_relaxed")
    all_predictions.write_parquet(out / "predictions.parquet")
    pl.DataFrame(results).write_parquet(out / "validation.parquet")
    json_write(out / "selection.json", selection)
    json_write(out / "fits.json", models)
    summary = (
        all_predictions.group_by("model")
        .agg(
            pl.len().alias("plays"),
            ((pl.col("actual") - pl.col("prediction")) ** 2).mean().sqrt().alias("rmse"),
            (pl.col("actual") - pl.col("prediction")).abs().mean().alias("mae"),
            pl.col("crps").mean(),
            pl.col("covered80").mean(),
            pl.col("width80").mean(),
        )
        .sort("rmse")
    )
    summary.write_parquet(out / "summary.parquet")
    summary.write_csv(out / "summary.csv")
    report = [
        "# RB handoff context experiment",
        "",
        "Separate play-level development experiment: predict play rushing yards at handoff.",
        "Train 2017, select parameters and calibrate residuals on 2018, refit 2017–2018, "
        "evaluate 2019. Source RB carriers only; no canonical player-ID joins.",
        "",
        "| Model | Plays | RMSE | MAE | CRPS | 80% coverage | 80% width |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.iter_rows(named=True):
        report.append(
            f"| {row['model']} | {row['plays']} | {row['rmse']:.3f} | {row['mae']:.3f} | "
            f"{row['crps']:.3f} | {row['covered80']:.1%} | {row['width80']:.3f} |"
        )
    report += [
        "",
        "Profile uses source height, weight, age and college. Carrier context adds "
        "observed game/formation/weather and carrier motion; all_players adds motion and "
        "spatial summaries of all ten teammates and eleven defenders.",
        "",
        "CRPS evaluates an empirical residual distribution using every 2018 residual. "
        "Selection and calibration share 2018, so calibration can be optimistic. "
        "Repeated players and games are dependent. No prospective claim or season-forecast "
        "improvement follows from this handoff experiment. The 2019 source season is partial. "
        "Permutations/importances would be descriptive, not causal.",
    ]
    (out / "readout.md").write_text("\n".join(report) + "\n")
    json_write(
        out / "manifest.json",
        {
            "status": "complete",
            "seconds": time.monotonic() - started,
            "files": {
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rb_gpu_play_context_20261003_001")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        frame, meta = load_play_dataset()
        print(json.dumps({"rows": frame.height, "metadata": meta}, indent=2))
    else:
        if Path(args.run_id).name != args.run_id:
            raise ValueError("Run ID must be one component")
        run(RUNS / args.run_id, args.device)
