"""Attach verified descriptive profiles and annual role-conditioned forecast errors.

Run with ``--wait`` while the main campaign is active. Only a completed analysis
manifest authorizes reading forecast artifacts; no training file is modified.
"""

from __future__ import annotations

# Embedded standalone HTML keeps its own line layout.
# ruff: noqa: E501
import argparse
import html
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .data import load_dataset
from .profile_predict import VARIANTS, validate_fold_artifact, verify_profile_library
from .profiles import KEYS, summarize_conditional_errors
from .run import RUNS, digest, json_write

POLICIES = (
    "selected",
    "ensemble_top3",
    "ensemble_diverse",
    "ensemble_convex",
    "persistence_calibrated",
)
FOCUS_TARGETS = ("rushing_yards", "receiving_yards", "receptions", "points")


def analysis_ready(out):
    """A completion manifest, rather than file existence, defines readability."""
    path = Path(out) / "analysis" / "manifest.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    return manifest.get("status") in {"core_complete", "complete"} and all(
        name in manifest.get("files", {})
        for name in ("predictions.parquet", "distributions.parquet")
    )


def copy_profile_library(out, profile_root):
    profile_root = Path(profile_root)
    source = verify_profile_library(profile_root)
    destination = Path(out) / "profiles"
    if destination.exists():
        if digest(destination / "manifest.json") != digest(profile_root / "manifest.json"):
            raise ValueError("The campaign already contains a different profile library")
        for name, expected in source["files"].items():
            if digest(destination / name) != expected:
                raise ValueError(f"Copied profile artifact changed: {name}")
    else:
        shutil.copytree(profile_root, destination)
    return destination, source


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else None


def _finite_median(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else None


def describe_annual_cohorts(dataset, folder, years):
    """Describe annual cluster vocabularies using earlier inputs and no future labels."""
    folder = Path(folder)
    metadata = json.loads((folder / "profiles.json").read_text())
    season = dataset.panel["year"].to_numpy()
    descriptions = []
    for year in years:
        saved = joblib.load(folder / "models" / f"roles_before_{year}.joblib")
        train = validate_fold_artifact(metadata, saved, dataset, year)
        panel = dataset.panel.filter(pl.Series(train))
        labels = np.asarray(saved["clusterer"].labels_)
        if len(labels) != panel.height:
            raise ValueError("Saved training role labels do not match their training population")
        opportunities = (panel["history_carries"] + panel["history_targets"]).to_numpy()
        exposure = panel["history_weeks"].to_numpy()
        usage = np.divide(
            opportunities, exposure, out=np.full(len(panel), np.nan), where=exposure > 0
        )
        share = np.divide(
            panel["history_targets"].to_numpy(),
            opportunities,
            out=np.full(len(panel), np.nan),
            where=opportunities > 0,
        )
        ages = panel["age"].to_numpy() if "age" in panel else np.full(len(panel), np.nan)
        newcomers = (panel["candidate_basis"] == "dated draft / exact ID").to_numpy()
        for label in sorted(np.unique(labels)):
            mask = labels == label
            typical_usage = _finite_median(usage[mask])
            target_share = _finite_median(share[mask])
            newcomer_share = float(np.mean(newcomers[mask]))
            description = (
                f"Prior usage {typical_usage:.1f}/week"
                if typical_usage is not None
                else "Prior usage unknown"
            )
            if target_share is not None:
                description += f"; median receiving opportunity share {target_share:.0%}"
            description += f"; {newcomer_share:.0%} exact-draft entrants"
            descriptions.append(
                {
                    "horizon": metadata["horizon"],
                    "year": int(year),
                    "profile": int(label),
                    "training_rows": int(mask.sum()),
                    "training_max_year": int(season[train].max()),
                    "training_median_prior_usage_per_week": typical_usage,
                    "training_median_receiving_opportunity_share": target_share,
                    "training_newcomer_share": newcomer_share,
                    "training_median_age": _finite_median(ages[mask]),
                    "training_mean_prior_usage_per_week": _finite_mean(usage[mask]),
                    "cohort_description": description,
                }
            )
    return pl.DataFrame(descriptions, infer_schema_length=None)


def aggregate_scoped_cohorts(rows):
    """Summarize distributions of annual cohort diagnostics, never join local IDs."""
    return (
        rows.group_by("variant", "horizon", "target", "recipe")
        .agg(
            pl.len().alias("annual_cohort_cells"),
            (pl.col("n") >= 20).sum().alias("cohort_cells_n20"),
            ((pl.col("n") >= 20) & (pl.col("mse_improvement_pct") > 0))
            .sum()
            .alias("positive_skill_cells_n20"),
            pl.col("n").sum().alias("matched_player_origins"),
            ((pl.col("mse") * pl.col("n")).sum() / pl.col("n").sum()).sqrt().alias("overall_rmse"),
            ((pl.col("baseline_mse") * pl.col("n")).sum() / pl.col("n").sum())
            .sqrt()
            .alias("overall_baseline_rmse"),
            pl.col("mse_improvement_pct")
            .filter(pl.col("n") >= 20)
            .median()
            .alias("median_cohort_skill_n20"),
            pl.col("coverage80")
            .filter(pl.col("interval_rows") >= 20)
            .median()
            .alias("median_cohort_coverage80_n20"),
            pl.col("year").min().alias("first_year"),
            pl.col("year").max().alias("last_year"),
        )
        .sort("variant", "horizon", "target", "recipe")
    )


def _html_report(path, rows, title):
    payload = json.dumps(rows.to_dicts(), allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:15px system-ui;margin:30px auto;max-width:1400px;background:#f7f9fc;color:#172839}
p{line-height:1.5}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #ccd;text-align:left}
select{padding:7px;margin:7px}.small{background:#fff0d7}a{color:#1d4ed8}</style><h1>__TITLE__</h1>
<p>These cohorts use profiles fitted only on earlier years. A profile ID belongs to its method, horizon
and year; matching numbers across years do not identify the same role. Descriptions summarize prior
training inputs. Skill compares each model with persistence on identical players. Orange rows have fewer
than 20 players. Errors and interval coverage are descriptive, with many dependent comparisons.</p>
<p>All methods partition the same forecasts: changing a cohort definition cannot improve overall model error.
The separate profile-mean benchmark tests whether a profile can itself predict production.</p>
<div id="controls"></div><table id="table"></table><script>const rows=__DATA__;
const fields=['variant','horizon','year','target','recipe'];
for(const f of fields){const label=document.createElement('label');label.textContent=f+' ';
const select=document.createElement('select');select.id=f;
[...new Set(rows.map(r=>r[f]))].sort().forEach(v=>select.add(new Option(v,v)));
select.onchange=draw;label.appendChild(select);document.querySelector('#controls').appendChild(label);}
document.getElementById('target').value='points';document.getElementById('recipe').value='ensemble_top3';
function fmt(v){return v===null?'—':typeof v==='number'?v.toFixed(2):String(v);}
function esc(v){return String(v).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function draw(){const selected=rows.filter(r=>fields.every(f=>String(r[f])===document.getElementById(f).value));
const cols=['profile','n','training_rows','cohort_description','rmse','baseline_rmse','mse_improvement_pct','coverage80','width80'];
document.querySelector('#table').innerHTML='<tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr>'+
selected.sort((a,b)=>a.profile-b.profile).map(r=>'<tr class="'+(r.n<20?'small':'')+'">'+cols.map(c=>'<td>'+esc(fmt(r[c]))+'</td>').join('')+'</tr>').join('');}
draw();</script>"""
    path.write_text(page.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload))


def _readout(path, rows, counts):
    focus = rows.filter(
        (pl.col("recipe") == "ensemble_top3") & pl.col("target").is_in(FOCUS_TARGETS)
    )
    lines = [
        "# Which annual RB cohorts were predictable?",
        "",
        "The prespecified top-three ensemble is the focus below; the interactive report includes "
        "all selected policies. Every cohort vocabulary was learned from earlier inputs. "
        "Cohort numbers are local to the method, horizon and year. Prior-usage descriptions use "
        "training inputs, never future outcomes.",
        "",
        "Each table shows the strongest and weakest observed cohort skill among annual cells "
        "with at least 20 matched players. These are retrospective diagnostic extremes chosen "
        "after evaluation, not validated rules for selecting a future model or player. "
        "All cells, including small ones, are retained in the Parquet and interactive reports.",
        "",
        "Intervals are the main model's nominal 80% predictive intervals calibrated from earlier "
        "out-of-season errors. Their measured coverage is not a guarantee; cohort samples are "
        "small, players/teams are dependent, and prior calibration years were also used for model selection.",
        "",
    ]
    for variant in VARIANTS:
        lines += [
            f"## {variant}",
            "",
            "| Horizon / target | Diagnostic | Year / local profile | N | RMSE / persistence | MSE skill % | 80% coverage | Training-input description |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        eligible = focus.filter(
            (pl.col("variant") == variant)
            & (pl.col("n") >= 20)
            & pl.col("mse_improvement_pct").is_finite()
        )
        for key, group in eligible.sort("horizon", "target", "mse_improvement_pct").group_by(
            "horizon",
            "target",
            maintain_order=True,
        ):
            for label, record in [
                ("strongest", group.row(-1, named=True)),
                ("weakest", group.row(0, named=True)),
            ]:
                coverage = "—" if record["coverage80"] is None else f"{record['coverage80']:.0%}"
                lines.append(
                    f"| {key[0]} / {key[1]} | {label} | {record['year']} / {record['profile']} | "
                    f"{record['n']} | {record['rmse']:.2f} / {record['baseline_rmse']:.2f} | "
                    f"{record['mse_improvement_pct']:.1f} | {coverage} | {record['cohort_description']} |"
                )
        lines.append("")
    small = rows.filter(pl.col("n") < 20).height
    lines += [
        f"Across all policy/target/method cells, {small:,} have fewer than 20 matched players. "
        "The cohort-summary file counts positive/negative annual cells without treating similarly "
        "numbered profiles as the same group across years. Overall errors are identical across "
        "profile partitions when all matched rows are included; only their organization changes.",
        "",
        f"Saved {rows.height:,} annual cohort error cells and {counts.height:,} method summaries.",
    ]
    path.write_text("\n".join(lines) + "\n")


def integrate(out, profile_root, *, threads=2):
    out = Path(out)
    if not analysis_ready(out):
        raise ValueError("Main analysis is not complete")
    profile_copy, source = copy_profile_library(out, profile_root)
    dest = profile_copy / "conditional"
    if dest.exists():
        raise ValueError("Conditional integration already exists; preserve its completed artifacts")
    analysis = out / "analysis"
    receipt = json.loads((analysis / "manifest.json").read_text())
    for name in ("predictions.parquet", "distributions.parquet"):
        if digest(analysis / name) != receipt["files"][name]:
            raise ValueError(f"Main analysis artifact changed: {name}")
    config = json.loads((out / "config.json").read_text())
    predictions = pl.read_parquet(analysis / "predictions.parquet").filter(
        pl.col("year").is_in(config["evaluation_years"])
        & pl.col("recipe").is_in([*POLICIES, "persistence"])
    )
    distribution = pl.read_parquet(analysis / "distributions.parquet")
    dest.mkdir()
    json_write(dest / "manifest.json", {"status": "running"})
    all_rows = []
    with threadpool_limits(limits=threads):
        datasets = {
            h: load_dataset(h, tuple(config["origins"]), config["input_cache"])
            for h in config["horizons"]
        }
        for horizon, dataset in datasets.items():
            expected = next(s for s in source["input_sources"] if s["horizon"] == horizon)
            path = Path(dataset.provenance["cache_directory"]) / "dataset.json"
            if digest(path) != expected["dataset_manifest_sha256"]:
                raise ValueError("Main and profile input data differ")
        for variant in VARIANTS:
            tagged, descriptions = [], []
            for horizon, dataset in datasets.items():
                folder = profile_copy / variant / horizon
                tagged.append(pl.read_parquet(folder / "chronological_profile_assignments.parquet"))
                descriptions.append(
                    describe_annual_cohorts(dataset, folder, config["evaluation_years"])
                )
            assignments = pl.concat(tagged).filter(pl.col("year").is_in(config["evaluation_years"]))
            if (
                predictions.select(KEYS)
                .unique()
                .join(assignments.select(KEYS), on=KEYS, how="anti")
                .height
            ):
                raise ValueError("Profiles fail to cover one or more main forecast rows")
            summary = summarize_conditional_errors(
                predictions, assignments, distributions=distribution
            )
            summary = (
                summary.filter(pl.col("recipe").is_in(POLICIES))
                .join(
                    pl.concat(descriptions),
                    on=["horizon", "year", "profile"],
                    how="left",
                    validate="m:1",
                )
                .with_columns(pl.lit(variant).alias("variant"))
            )
            if summary["training_rows"].null_count():
                raise ValueError("An evaluated cluster lacks an earlier training description")
            directory = dest / variant
            directory.mkdir()
            summary.write_parquet(directory / "conditional_errors.parquet")
            _html_report(
                directory / "index.html", summary, f"RB conditional predictability: {variant}"
            )
            all_rows.append(summary)
            print(f"Integrated {variant}: {summary.height} annual cohort cells", flush=True)
    rows = pl.concat(all_rows)
    counts = aggregate_scoped_cohorts(rows)
    rows.write_parquet(dest / "conditional_errors.parquet")
    counts.write_parquet(dest / "cohort_summary.parquet")
    _html_report(dest / "index.html", rows, "Running back conditional predictability")
    _readout(dest / "readout.md", rows, counts)
    shutil.copyfile(__file__, dest / "integration_source.py")
    shutil.copyfile(Path(__file__).parent / "profiles.py", dest / "cohort_helper_source.py")
    json_write(
        dest / "config.json",
        {
            "main_run": str(out.resolve()),
            "profile_library": str(Path(profile_root).resolve()),
            "years": config["evaluation_years"],
            "variants": VARIANTS,
            "policies": POLICIES,
            "focus_policy": "ensemble_top3",
            "highlight_minimum_matched_rows": 20,
            "prediction_sha256": digest(analysis / "predictions.parquet"),
            "distribution_sha256": digest(analysis / "distributions.parquet"),
            "profile_manifest_sha256": digest(profile_copy / "manifest.json"),
        },
    )
    json_write(
        dest / "manifest.json",
        {
            "status": "complete",
            "cohort_cells": rows.height,
            "files": {
                str(p.relative_to(dest)): digest(p)
                for p in sorted(dest.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    return dest


def wait_and_integrate(out, profile_root, *, poll_seconds=30, timeout_seconds=14400, threads=2):
    began = time.monotonic()
    copy_profile_library(out, profile_root)
    print(f"Profile library linked at {Path(out) / 'profiles/index.html'}", flush=True)
    print(
        "Waiting for core_complete/complete main analysis/manifest.json; no training outputs are read early.",
        flush=True,
    )
    while not analysis_ready(out):
        if time.monotonic() - began > timeout_seconds:
            raise TimeoutError("Main analysis did not complete within the requested waiting budget")
        time.sleep(poll_seconds)
    print("Main analysis is complete; integrating annual profile diagnostics.", flush=True)
    return integrate(out, profile_root, threads=threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rb_gpu_20261003_001")
    parser.add_argument("--profile-root", type=Path, default=RUNS / "rb_gpu_profiles_20261003_002")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    args = parser.parse_args()
    if args.wait:
        print(
            wait_and_integrate(
                RUNS / args.run_id,
                args.profile_root,
                threads=args.threads,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        )
    else:
        print(integrate(RUNS / args.run_id, args.profile_root, threads=args.threads))
