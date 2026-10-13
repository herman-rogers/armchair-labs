"""Run the complete bounded v3 experiment; never overwrite v2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import (
    DIST_METHODS,
    MEAN_METHODS,
    QCOLS,
    fit_distribution,
    fit_mean,
    predict_distribution,
    predict_mean,
    score_predictions,
    select_candidate,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "experiments/future_player_lab/runs/opening_drive_v3_20261011_002"


def json_write(path, value):
    path.write_text(
        json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata():
    return dict(
        mean_methods=MEAN_METHODS,
        primary_mean_methods=[m for m in MEAN_METHODS if m not in ("v2", "lgb_correction_full")],
        distribution_methods=DIST_METHODS,
        features="data.py explicit strictly pregame numeric allowlist",
        mean_parameters=dict(
            trees=160,
            learning_rate=0.045,
            leaves=15,
            depth=5,
            min_child_samples=60,
            l2=15,
            threads=4,
        ),
        xgb_parameters=dict(
            trees=180,
            depth=3,
            min_child_weight=40,
            l2=20,
            learning_rate=0.045,
            device="cpu",
            threads=4,
        ),
        distribution_parameters=dict(
            trees=110,
            player_prior_weeks=24,
            calibration_window_years=5,
            mean_bins=5,
            conditional_shape_holdout="latest previous season",
        ),
        quantiles=[int(c[1:]) / 100 for c in QCOLS],
        history_comparison=dict(
            full="2003 onward", recent="previous 5 seasons", decay="4-season half life"
        ),
        residual_warmup=[2015, 2016, 2017],
        selection_years=[2018, 2019, 2020],
        evaluation_years=list(range(2021, 2026)),
        final_fit_year=2026,
        selection=(
            "Fixed per-position mean using equal-year RMSE, then fixed distribution using "
            "equal-year quantile-integrated CRPS; selected 2018–20, retained for final 2026"
        ),
        limitations=[
            "Development-era retrospective evaluation: these years were used in prior research.",
            "Exact v2 baseline starts in 2021; earlier baseline is a persistence proxy.",
            "Quantile-integrated CRPS uses flat 1%/99% tails and is an approximation.",
            "Quantile outputs sorted; raw crossing diagnostic is measured before sorting.",
            "No external injury/news/odds lookahead; available feature history only.",
            "Standalone player models, NGBoost, explicit hurdle mixtures, and neural models "
            "are outside this bounded comparison.",
        ],
    )


def _score_table(frame, method_col):
    records = []
    for (position, season, method), group in frame.groupby(["position", "season", method_col]):
        records.append(
            dict(position=position, season=int(season), method=method, **score_predictions(group))
        )
    return pd.DataFrame(records)


def _cohort_scores(frame):
    rows = []
    for (position, method), group in frame.groupby(["position", "distribution_method"]):
        cohorts = {
            "all": group,
            "exact_v2": group[group.v2_exact_available.eq(1)],
            "history_under_8": group[group.player_history_count.lt(8)],
            "history_8_to_31": group[group.player_history_count.between(8, 31)],
            "history_32_plus": group[group.player_history_count.ge(32)],
            "projected_10_plus": group[group.prediction.ge(10)],
            "projected_5_to_10": group[group.prediction.between(5, 10, inclusive="left")],
            "projected_under_5": group[group.prediction.lt(5)],
        }
        for cohort, subset in cohorts.items():
            rows.append(
                dict(position=position, method=method, cohort=cohort, **score_predictions(subset))
            )
    return pd.DataFrame(rows)


def _spread_calibration(frame):
    """Check whether predicted wider weeks actually have wider centered errors."""
    rows = []
    for (position, method), group in frame.groupby(["position", "distribution_method"]):
        group = group[group.target_points.notna()].copy()
        group["spread_bin"] = pd.qcut(group["std"], 5, labels=False, duplicates="drop")
        if group.spread_bin.isna().all():
            group["spread_bin"] = 0
        for spread_bin, subset in group.groupby("spread_bin"):
            residual = subset.target_points - subset.prediction
            rows.append(
                dict(
                    position=position,
                    method=method,
                    spread_bin=int(spread_bin),
                    n=len(subset),
                    predicted_std=float(subset["std"].mean()),
                    observed_residual_std=float(residual.std(ddof=0)),
                    observed_bias=float(residual.mean()),
                    **{k: v for k, v in score_predictions(subset).items() if k != "n"},
                )
            )
    return pd.DataFrame(rows)


def run(output_dir=DEFAULT_OUTPUT, *, rebuild_data=False):
    from .data import build_panel

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "folds"
    cache.mkdir(exist_ok=True)
    models = output / "models"
    models.mkdir(exist_ok=True)
    json_write(output / "candidate_metadata.json", metadata())
    start = time.time()
    panel_path = output / "panel.parquet"
    if panel_path.exists() and (output / "dataset.json").exists() and not rebuild_data:
        panel = pd.read_parquet(panel_path)
        dataset = json.loads((output / "dataset.json").read_text())
        features = dataset.get("feature_columns", dataset.get("features"))
        if not features:
            raise ValueError("dataset.json requires features or feature_columns")
    else:
        panel, features, dataset = build_panel(output_dir=output)
    if dataset.get("panel_sha256") != digest(panel_path):
        raise ValueError("Panel checksum differs from dataset manifest")
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("modeling.py"),
        Path(__file__).with_name("data.py"),
    ]
    signature = dict(
        panel_sha256=digest(panel_path),
        dataset_sha256=digest(output / "dataset.json"),
        features=features,
        source_sha256={p.name: digest(p) for p in source_paths},
        candidate_metadata=metadata(),
    )
    signature_path = output / "run_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError(
            "Cached folds have a different data/model contract; use a fresh output directory"
        )
    json_write(signature_path, signature)
    source_snapshot = output / "source_snapshot"
    source_snapshot.mkdir(exist_ok=True)
    for source in source_paths:
        shutil.copy2(source, source_snapshot / source.name)
    forbidden = {"target_points", "season", "week", "player_id"} & set(features)
    if "target_points" in forbidden:
        raise ValueError("Outcome included in feature allowlist")
    panel = panel.copy()
    panel["row_id"] = np.arange(len(panel))
    if "scheduled_game" in panel:
        panel = panel[panel.scheduled_game.eq(1)].copy()
    panel["phase"] = np.select(
        [panel.season.lt(2018), panel.season.le(2020), panel.season.le(2025)],
        ["residual_warmup", "selection", "evaluation"],
        default="current_2026",
    )
    id_cols = [
        c
        for c in [
            "row_id",
            "player_id",
            "player_name",
            "position",
            "season",
            "week",
            "team",
            "opponent",
            "game_id",
            "target_points",
            "v2_weekly_mean",
            "v2_exact_available",
            "baseline_source",
            "player_history_count",
            "phase",
        ]
        if c in panel
    ]
    mean_frames = []
    selection = {}
    for position in ("QB", "RB", "WR", "TE"):
        local = panel[panel.position.eq(position)]
        if local.empty:
            raise ValueError(f"Missing position {position}")
        for year in range(2015, 2027):
            test = local[local.season.eq(year)]
            train = local[local.season.lt(year) & local.target_points.notna()]
            if test.empty:
                continue
            for method in MEAN_METHODS:
                path = cache / f"mean_{position}_{year}_{method}.parquet"
                model_path = models / f"{position}_{method}.pkl"
                if path.exists() and (year != 2026 or model_path.exists()):
                    pred = pd.read_parquet(path)
                else:
                    fitted = fit_mean(train, features, method, year)
                    pred = test[id_cols].copy()
                    pred["mean_method"] = method
                    pred["prediction"] = predict_mean(fitted, test)
                    pred.to_parquet(path, index=False)
                    if year == 2026:
                        with model_path.open("wb") as file:
                            pickle.dump(fitted, file)
                mean_frames.append(pred)
            print(
                f"mean {position} {year}: {len(train):,} train / {len(test):,} forecast rows",
                flush=True,
            )
        combined = pd.concat(
            [f for f in mean_frames if f.position.iloc[0] == position], ignore_index=True
        )
        scores = _score_table(combined, "mean_method")
        # Correction selection would compare proxy-based training against exact
        # v2 deployment. Keep it as an explicitly exploratory challenger.
        chosen = select_candidate(
            scores[~scores.method.isin(["v2", "lgb_correction_full"])], "rmse"
        )
        selection[position] = {"mean_method": chosen}
        print(f"SELECT {position} mean={chosen}", flush=True)
    mean_predictions = pd.concat(mean_frames, ignore_index=True)
    mean_predictions.to_parquet(output / "mean_predictions.parquet", index=False)
    _score_table(mean_predictions, "mean_method").to_csv(output / "mean_scores.csv", index=False)
    json_write(output / "selection.json", selection)
    distribution_frames = []
    for position in ("QB", "RB", "WR", "TE"):
        chosen = selection[position]["mean_method"]
        local_panel = panel[panel.position.eq(position)].set_index("row_id", drop=False)
        for mean_method in dict.fromkeys([chosen, "v2"]):
            preds = mean_predictions[
                (mean_predictions.position.eq(position))
                & mean_predictions.mean_method.eq(mean_method)
            ]
            attached = local_panel.reindex(preds.row_id).reset_index(drop=True).copy()
            attached["prediction"] = preds.prediction.to_numpy()
            methods = DIST_METHODS if mean_method == chosen else ["pooled", "mean_bin"]
            for year in range(2018, 2027):
                test = attached[attached.season.eq(year)]
                calibration = attached[attached.season.lt(year) & attached.target_points.notna()]
                if mean_method == "v2" and calibration.v2_exact_available.eq(1).any():
                    calibration = calibration[calibration.v2_exact_available.eq(1)]
                if test.empty:
                    continue
                for method in methods:
                    label = method if mean_method == chosen else f"v2_{method}"
                    path = cache / f"dist_{position}_{year}_{mean_method}_{method}.parquet"
                    model_path = models / f"{position}_{mean_method}_{method}_distribution.pkl"
                    if path.exists() and (year != 2026 or model_path.exists()):
                        pred = pd.read_parquet(path)
                    else:
                        fitted = fit_distribution(calibration, features, method, year)
                        q, std, crossings = predict_distribution(fitted, test, diagnostics=True)
                        pred = test[id_cols].copy()
                        pred["mean_method"] = mean_method
                        pred["distribution_method"] = label
                        pred["prediction"] = test.prediction.to_numpy()
                        pred["std"] = std
                        pred["raw_quantile_crossing"] = crossings
                        pred[QCOLS] = q
                        pred["calibration_bias"] = fitted["bias"]
                        pred["calibration_n"] = fitted["calibration_n"]
                        pred["calibration_max_year"] = fitted["calibration_max_year"]
                        pred["calibration_exact_v2_fraction"] = float(
                            calibration.v2_exact_available.mean()
                        )
                        pred.to_parquet(path, index=False)
                        if year == 2026:
                            with model_path.open("wb") as file:
                                pickle.dump(fitted, file)
                    distribution_frames.append(pred)
                print(
                    f"distribution {position} {year} mean={mean_method}, methods={len(methods)}",
                    flush=True,
                )
        pos_predictions = pd.concat(
            [f for f in distribution_frames if f.position.iloc[0] == position], ignore_index=True
        )
        scores = _score_table(
            pos_predictions[pos_predictions.mean_method.eq(chosen)], "distribution_method"
        )
        selection[position]["distribution_method"] = select_candidate(scores, "crps_quantile")
        print(
            f"SELECT {position} distribution={selection[position]['distribution_method']}",
            flush=True,
        )
    distributions = pd.concat(distribution_frames, ignore_index=True)
    distributions.to_parquet(output / "distribution_predictions.parquet", index=False)
    _score_table(distributions, "distribution_method").to_csv(
        output / "distribution_scores.csv", index=False
    )
    evaluation = distributions[distributions.phase.eq("evaluation")]
    _cohort_scores(evaluation).to_csv(output / "cohort_scores.csv", index=False)
    _spread_calibration(evaluation).to_csv(output / "spread_calibration.csv", index=False)
    selected = pd.concat(
        [
            distributions[
                distributions.position.eq(pos)
                & distributions.mean_method.eq(spec["mean_method"])
                & distributions.distribution_method.eq(spec["distribution_method"])
            ]
            for pos, spec in selection.items()
        ],
        ignore_index=True,
    )
    selected.to_parquet(output / "selected_predictions.parquet", index=False)
    json_write(output / "selection.json", selection)
    # Replay final selected serialized estimators before accepting the run.
    replay = []
    importance = []
    for pos, spec in selection.items():
        live = panel[(panel.position.eq(pos)) & panel.season.eq(2026)].copy()
        with (models / f"{pos}_{spec['mean_method']}.pkl").open("rb") as file:
            mean_model = pickle.load(file)
        if mean_model["model"] is not None:
            importance.extend(
                dict(position=pos, component="mean", feature=feature, importance=float(value))
                for feature, value in zip(
                    mean_model["features"], mean_model["model"].feature_importances_, strict=True
                )
            )
        live["prediction"] = predict_mean(mean_model, live)
        with (
            models / f"{pos}_{spec['mean_method']}_{spec['distribution_method']}_distribution.pkl"
        ).open("rb") as file:
            distribution_model = pickle.load(file)
        if "model" in distribution_model:
            importance.extend(
                dict(position=pos, component="spread", feature=feature, importance=float(value))
                for feature, value in zip(
                    distribution_model["features"],
                    distribution_model["model"].feature_importances_,
                    strict=True,
                )
            )
        q, std = predict_distribution(distribution_model, live)
        expected = (
            selected[selected.position.eq(pos) & selected.season.eq(2026)]
            .set_index("row_id")
            .loc[live.row_id]
        )
        errors = dict(
            max_quantile_error=float(np.max(np.abs(q - expected[QCOLS].to_numpy()))),
            max_std_error=float(np.max(np.abs(std - expected["std"].to_numpy()))),
            max_mean_error=float(
                np.max(np.abs(live.prediction.to_numpy() - expected.prediction.to_numpy()))
            ),
        )
        if max(errors.values()) > 1e-8:
            raise AssertionError(f"Replay mismatch {pos}: {errors}")
        replay.append(dict(position=pos, n=len(live), **errors))
    json_write(output / "replay.json", replay)
    pd.DataFrame(importance).to_csv(output / "feature_importance.csv", index=False)
    sources = source_paths
    manifest = dict(
        version=3,
        status="research_preview",
        elapsed_seconds=time.time() - start,
        selection=selection,
        features=features,
        candidate_metadata=metadata(),
        packages={
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "pyarrow")
        },
        source_files={str(p.relative_to(ROOT)): digest(p) for p in sources},
        artifact_hashes={
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    )
    json_write(output / "manifest.json", manifest)
    print(f"COMPLETE {output} in {time.time() - start:.1f}s", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, rebuild_data=args.rebuild_data)
