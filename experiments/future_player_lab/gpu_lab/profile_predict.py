"""Frozen chronological benchmark of hard/soft profile-conditioned production means.

This asks whether unsupervised groups carry useful predictive information. It does
not pick a profile algorithm using the evaluation years, and does not calibrate
uncertainty from in-sample residuals. Every saved role geometry was fitted before
the corresponding forecast year and is verified before reuse.
"""

# Embedded standalone HTML keeps its own line layout.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .data import load_dataset
from .run import RUNS, STATS, digest, json_write, packages, stat_arrays

KEY = ["horizon", "year", "origin", "player_id", "target"]
VARIANTS = ("standard_kmeans", "standard_gmm", "balanced_kmeans", "balanced_gmm")


def recipe_library():
    """Twelve prespecified hypotheses, independent of validation/test outcomes."""
    return [
        {
            "id": f"{variant}_{membership}_prior{prior}",
            "variant": variant,
            "membership": membership,
            "prior_strength": prior,
        }
        for variant in VARIANTS
        for membership in (["hard", "soft"] if variant.endswith("gmm") else ["hard"])
        for prior in (0, 20)
    ]


def validate_membership(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or not weights.shape[1] or not np.isfinite(weights).all():
        raise ValueError("Membership must be a finite two-dimensional probability matrix")
    if (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Membership probabilities must be nonnegative and sum to one")
    return weights


class ProfileMeans:
    """Cluster target-rate means with a fixed global-mean pseudo-count prior."""

    def __init__(self, prior_strength=20):
        self.prior_strength = float(prior_strength)
        if not np.isfinite(self.prior_strength) or self.prior_strength < 0:
            raise ValueError("Prior strength must be finite and nonnegative")

    def fit(self, weights, targets):
        weights = validate_membership(weights)
        targets = np.asarray(targets, dtype=float)
        if targets.ndim != 2 or len(targets) != len(weights):
            raise ValueError("Training target matrix must align with training memberships")
        known = np.isfinite(targets)
        if not known.any(axis=0).all():
            raise ValueError("Every target requires at least one observed training label")
        observed = np.where(known, targets, 0)
        self.global_mean_ = observed.sum(axis=0) / known.sum(axis=0)
        self.weighted_count_ = weights.T @ known.astype(float)
        squared_weights = (weights**2).T @ known.astype(float)
        self.effective_count_ = np.divide(
            self.weighted_count_**2,
            squared_weights,
            out=np.zeros_like(squared_weights),
            where=squared_weights > 0,
        )
        numerator = weights.T @ observed + self.prior_strength * self.global_mean_
        denominator = self.weighted_count_ + self.prior_strength
        fallback = np.broadcast_to(self.global_mean_, numerator.shape).copy()
        self.cluster_mean_ = np.divide(numerator, denominator, out=fallback, where=denominator > 0)
        self.k_ = weights.shape[1]
        return self

    def predict(self, weights):
        weights = validate_membership(weights)
        if weights.shape[1] != self.k_:
            raise ValueError("Membership width differs from the fitted profile vocabulary")
        return weights @ self.cluster_mean_


def memberships(role_model, x, mode):
    z = role_model["representation"].transform(x)
    clusterer = role_model["clusterer"]
    if mode == "soft":
        if not hasattr(clusterer, "predict_proba"):
            raise ValueError("Soft memberships require a probabilistic profile model")
        return validate_membership(clusterer.predict_proba(z))
    if mode != "hard":
        raise ValueError("Unknown profile membership rule")
    k = len(clusterer.cluster_centers_)
    return np.eye(k)[clusterer.predict(z)]


def verify_profile_library(root):
    manifest_path = Path(root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("A completed profile library is required")
    for name, expected in manifest["files"].items():
        if digest(Path(root) / name) != expected:
            raise ValueError(f"Profile library artifact changed: {name}")
    return manifest


def validate_fold_artifact(metadata, role_model, dataset, year):
    fold = next((r for r in metadata["chronological_folds"] if r["year"] == year), None)
    training = dataset.panel["year"].to_numpy() < year
    if fold is None or fold["training_max_year"] >= year or fold["training_rows"] != training.sum():
        raise ValueError("Saved profile chronology does not match the requested fold")
    representation = role_model["representation"]
    if representation.n_train_ != training.sum():
        raise ValueError("Saved preprocessing training population differs")
    if representation.feature_names_in_.tolist() != dataset.names:
        raise ValueError("Saved preprocessing feature order differs")
    return training


def forecast_frame(dataset, test, actual, prediction, recipe):
    keys = dataset.panel.filter(pl.Series(test)).select(
        *[
            n
            for n in ["horizon", "year", "origin", "player_id", "player_display_name", "weeks"]
            if n in dataset.panel.columns
        ]
    )
    return pl.concat(
        [
            keys.with_columns(
                pl.lit(stat).alias("target"),
                pl.Series("actual", actual[:, j]),
                pl.Series("prediction", prediction[:, j]),
                pl.lit(recipe).alias("recipe"),
            )
            for j, stat in enumerate(STATS)
        ]
    )


def score_predictions(predictions, seed=20261003):
    valid = predictions.filter(pl.col("actual").is_finite() & pl.col("prediction").is_finite())
    annual = (
        valid.group_by("horizon", "year", "target", "recipe")
        .agg(
            pl.len().alias("n"),
            ((pl.col("prediction") - pl.col("actual")) ** 2).mean().alias("mse"),
            (pl.col("prediction") - pl.col("actual")).abs().mean().alias("mae"),
            (pl.col("prediction") - pl.col("actual")).mean().alias("bias"),
        )
        .with_columns(pl.col("mse").sqrt().alias("rmse"))
    )
    pairs = []
    for control in ("training_mean", "persistence"):
        baseline = valid.filter(pl.col("recipe") == control).select(
            *KEY,
            pl.col("prediction").alias("baseline"),
            pl.col("actual").alias("baseline_actual"),
        )
        joined = valid.join(baseline, on=KEY, how="inner", validate="m:1")
        if joined.filter((pl.col("actual") - pl.col("baseline_actual")).abs() > 1e-8).height:
            raise ValueError("Matched profile and baseline outcomes disagree")
        joined = joined.with_columns(
            ((pl.col("prediction") - pl.col("actual")) ** 2).alias("se"),
            ((pl.col("baseline") - pl.col("actual")) ** 2).alias("baseline_se"),
        )
        groups = ["horizon", "target", "recipe"]
        for keys, frame in joined.sort(groups).group_by(groups, maintain_order=True):
            by_year = (
                frame.group_by("year")
                .agg(
                    pl.col("se").mean(),
                    pl.col("baseline_se").mean(),
                )
                .sort("year")
            )
            mse = float(frame["se"].mean())
            baseline_mse = float(frame["baseline_se"].mean())
            delta = (by_year["se"] - by_year["baseline_se"]).to_numpy()
            group_seed = int(
                hashlib.sha256(repr((seed, control, keys)).encode()).hexdigest()[:8], 16
            )
            rng = np.random.default_rng(group_seed)
            boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
            pairs.append(
                dict(zip(groups, keys, strict=True))
                | {
                    "baseline": control,
                    "n": frame.height,
                    "seasons": len(delta),
                    "rmse": float(np.sqrt(mse)),
                    "baseline_rmse": float(np.sqrt(baseline_mse)),
                    "mse_improvement_pct": 100 * (1 - mse / baseline_mse) if baseline_mse else None,
                    "mean_season_delta_mse": float(delta.mean()),
                    "delta_low95": float(np.quantile(boot, 0.025)),
                    "delta_high95": float(np.quantile(boot, 0.975)),
                }
            )
    return annual.sort("horizon", "year", "target", "recipe"), pl.DataFrame(pairs)


def _report(out, comparisons):
    rows = comparisons.filter(pl.col("target") == "points").sort("horizon", "baseline", "rmse")
    lines = [
        "# Profile-conditioned running back prediction",
        "",
        "Frozen historical benchmark: 2021–2025, preseason and after Week 2 remaining season.",
        "Four earlier-year profile geometries supply fixed k=5 assignments. Twelve recipes compare "
        "hard memberships, GMM soft memberships, and global-mean prior strengths 0 or 20. "
        "No evaluation result selected the recipes or their hyperparameters.",
        "",
        "Every predictor is a profile's mean future production per calendar week, optionally "
        "shrunk toward the global training mean, multiplied by forecast weeks. Test outcomes "
        "never fit profiles or means. Unknown labels are masked separately for each target.",
        "",
        "PPR points = rushing/receiving yards ÷ 10 + receptions + 6 × rushing/receiving TDs. "
        "Other scoring components are excluded. No predictive intervals use in-sample residuals.",
        "",
        "| Horizon | Recipe | Baseline | RMSE | Baseline RMSE | MSE improvement % |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in rows.iter_rows(named=True):
        improvement = "—" if r["mse_improvement_pct"] is None else f"{r['mse_improvement_pct']:.2f}"
        lines.append(
            f"| {r['horizon']} | {r['recipe']} | {r['baseline']} | {r['rmse']:.3f} | "
            f"{r['baseline_rmse']:.3f} | {improvement} |"
        )
    lines += [
        "",
        "RMSE and percent improvement pool matched player rows. Bootstrap intervals in "
        "matched_comparisons.parquet describe equal-season mean MSE differences, with five "
        "season blocks and no adjustment for multiple comparisons. Profile IDs are local "
        "to each year; low membership mass and small effective samples are flagged in "
        "cohort_counts.parquet. Better clustering cohesion need not improve forecasting.",
    ]
    (out / "readout.md").write_text("\n".join(lines) + "\n")
    records = comparisons.to_dicts()
    encoded = json.dumps(records, allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><meta charset="utf-8"><title>RB profile forecasting benchmark</title>
<style>body{font:16px system-ui;margin:30px auto;max-width:1250px;background:#f7f9fc;color:#172839}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #ccd}
select{padding:6px;margin:8px}p{line-height:1.5}</style><h1>Can unsupervised RB profiles predict production?</h1>
<p>Fixed profile algorithms learn only earlier inputs; profile production means learn only earlier labels.
Twelve prespecified pipelines compare hard/soft membership and shrinkage. Evaluation years 2021–2025.
Positive MSE improvement means better predictions than the selected matched baseline.</p>
<div id="controls"></div><table id="table"></table><p><a href="readout.md">Protocol and points readout</a></p>
<script>const rows=__DATA__, fields=['horizon','target','baseline'];
for(const f of fields){const s=document.createElement('select');s.id=f;
[...new Set(rows.map(r=>r[f]))].sort().forEach(v=>s.add(new Option(v,v)));s.onchange=draw;
document.querySelector('#controls').appendChild(s);}document.querySelector('#target').value='points';
function draw(){const data=rows.filter(r=>fields.every(f=>r[f]===document.getElementById(f).value))
.sort((a,b)=>a.rmse-b.rmse);const columns=['recipe','n','rmse','baseline_rmse','mse_improvement_pct'];
document.querySelector('#table').innerHTML='<tr>'+columns.map(c=>'<th>'+c+'</th>').join('')+'</tr>'+
data.map(r=>'<tr>'+columns.map(c=>'<td>'+(r[c]===null?'—':typeof r[c]==='number'?r[c].toFixed(2):r[c])+
'</td>').join('')+'</tr>').join('');}draw();</script>"""
    (out / "index.html").write_text(page.replace("__DATA__", encoded))


def run(out, profile_root, *, years=(2021, 2022, 2023, 2024, 2025), threads=2):
    out, profile_root = Path(out), Path(profile_root)
    manifest = verify_profile_library(profile_root)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    recipes = recipe_library()
    config = {
        "years": list(years),
        "horizons": ["season", "remaining"],
        "targets": STATS,
        "recipes": recipes,
        "threads": threads,
        "profile_root": str(profile_root.resolve()),
        "profile_manifest_sha256": digest(profile_root / "manifest.json"),
        "target_mode": "future totals / forecast calendar weeks",
        "k": 5,
        "selection": "none: all twelve recipes fixed before evaluation",
    }
    json_write(out / "config.json", config)
    json_write(out / "manifest.json", {"status": "running"})
    (out / "implementation").mkdir()
    for name in ["profile_predict.py", "representations.py", "profiles.py", "data.py", "run.py"]:
        shutil.copyfile(Path(__file__).parent / name, out / "implementation" / name)
    predictions, cohort_rows, source_rows = [], [], []
    with threadpool_limits(limits=threads):
        for horizon in config["horizons"]:
            dataset = load_dataset(horizon)
            metadata_path = Path(dataset.provenance["cache_directory"]) / "dataset.json"
            source = next(r for r in manifest["input_sources"] if r["horizon"] == horizon)
            if digest(metadata_path) != source["dataset_manifest_sha256"]:
                raise ValueError("Profile source data differ from current corrected inputs")
            source_rows.append(source)
            targets, persistence = stat_arrays(dataset)
            rates = targets / dataset.exposure[:, None]
            seasons = dataset.panel["year"].to_numpy()
            for year in years:
                test, train = seasons == year, seasons < year
                for baseline, pred in [
                    ("persistence", persistence[test]),
                    (
                        "training_mean",
                        np.nanmean(rates[train], axis=0)[None, :] * dataset.exposure[test, None],
                    ),
                ]:
                    predictions.append(forecast_frame(dataset, test, targets[test], pred, baseline))
                for variant in VARIANTS:
                    folder = profile_root / variant / horizon
                    role_path = folder / "models" / f"roles_before_{year}.joblib"
                    role_model = joblib.load(role_path)
                    metadata = json.loads((folder / "profiles.json").read_text())
                    validate_fold_artifact(metadata, role_model, dataset, year)
                    dest = out / "models" / variant / horizon / str(year)
                    dest.mkdir(parents=True)
                    shutil.copyfile(role_path, dest / "role_model.joblib")
                    for mode in ("hard", "soft") if variant.endswith("gmm") else ("hard",):
                        tw = memberships(role_model, dataset.x[train], mode)
                        vw = memberships(role_model, dataset.x[test], mode)
                        for prior in (0, 20):
                            recipe = f"{variant}_{mode}_prior{prior}"
                            model = ProfileMeans(prior).fit(tw, rates[train])
                            pred = model.predict(vw) * dataset.exposure[test, None]
                            predictions.append(
                                forecast_frame(dataset, test, targets[test], pred, recipe)
                            )
                            joblib.dump(model, dest / f"{mode}_prior{prior}.joblib", compress=3)
                            for cluster in range(model.k_):
                                for j, target in enumerate(STATS):
                                    cohort_rows.append(
                                        {
                                            "horizon": horizon,
                                            "year": year,
                                            "recipe": recipe,
                                            "profile": cluster,
                                            "target": target,
                                            "training_max_year": int(seasons[train].max()),
                                            "weighted_training_count": float(
                                                model.weighted_count_[cluster, j]
                                            ),
                                            "effective_training_count": float(
                                                model.effective_count_[cluster, j]
                                            ),
                                            "test_membership_mass": float(vw[:, cluster].sum()),
                                            "cluster_mean_rate": float(
                                                model.cluster_mean_[cluster, j]
                                            ),
                                            "global_mean_rate": float(model.global_mean_[j]),
                                            "small_cohort": bool(
                                                model.weighted_count_[cluster, j] < 20
                                                or model.effective_count_[cluster, j] < 20
                                            ),
                                        }
                                    )
                    print(f"{horizon} {year} {variant}: profile means scored", flush=True)
    rows = pl.concat(predictions)
    if rows.select(*KEY, "recipe").is_duplicated().any():
        raise ValueError("Duplicate profile forecast keys")
    annual, comparisons = score_predictions(rows)
    rows.write_parquet(out / "predictions.parquet")
    annual.write_parquet(out / "annual_metrics.parquet")
    comparisons.write_parquet(out / "matched_comparisons.parquet")
    counts = pl.DataFrame(cohort_rows)
    counts.write_parquet(out / "cohort_counts.parquet")
    _report(out, comparisons)
    json_write(
        out / "manifest.json",
        {
            "status": "complete",
            "seconds": time.monotonic() - started,
            "packages": packages(),
            "input_sources": source_rows,
            "profile_manifest_sha256": config["profile_manifest_sha256"],
            "small_cohort_target_cells": counts.filter(pl.col("small_cohort")).height,
            "files": {
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    return comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rb_profile_predict_20261003_001")
    parser.add_argument("--profile-root", type=Path, default=RUNS / "rb_gpu_profiles_20261003_002")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(RUNS / args.run_id, args.profile_root, threads=args.threads)
