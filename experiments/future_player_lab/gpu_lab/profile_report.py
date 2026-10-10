"""Assemble a provenance-pinned comparison of completed descriptive profile fits."""

# Embedded standalone HTML keeps its own line layout.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .data import load_dataset


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize_profiles(root):
    root = Path(root)
    runs = sorted(root.glob("*/*/profiles.json"))
    if not runs:
        raise ValueError("No completed profile metadata found")
    comparisons, optimizer, sources = [], [], []
    for path in runs:
        metadata = json.loads(path.read_text())
        variant = path.parent.parent.name
        for row in metadata["descriptive"]:
            comparisons.append(
                {
                    "horizon": metadata["horizon"],
                    "variant": variant,
                    "representation": metadata["geometry"]["kind"],
                    "method": metadata["method"],
                    "k": row["k"],
                    "rows": metadata["input_rows"],
                    "input_features": metadata["input_features"],
                    "silhouette": row["silhouette"],
                    "mean_seed_ari": row["cross_seed_mean_adjusted_rand"],
                    "bic": row["bic"],
                    "sizes": row["cluster_sizes"],
                    "explained_variance": sum(
                        metadata["full_representation"]["explained_variance_ratio"]
                    ),
                    "link": str(path.parent.relative_to(root) / "index.html"),
                }
            )
            optimizer.append(
                {
                    "variant": variant,
                    "horizon": metadata["horizon"],
                    "fit_scope": "all_inputs_descriptive",
                    "k": row["k"],
                    "converged": row["converged"],
                    "iterations": row["iterations"],
                    "warnings": row.get("optimization_warnings", []),
                }
            )
        for fold in metadata["chronological_folds"]:
            optimizer.append(
                {
                    "variant": variant,
                    "horizon": metadata["horizon"],
                    "fit_scope": "earlier_years_only",
                    "year": fold["year"],
                    "k": fold["k"],
                    "converged": fold["converged"],
                    "iterations": fold["iterations"],
                    "warnings": fold.get("optimization_warnings", []),
                }
            )
    for horizon in sorted({r["horizon"] for r in comparisons}):
        dataset = load_dataset(horizon)
        cache_manifest = Path(dataset.provenance["cache_directory"]) / "dataset.json"
        sources.append(
            {
                "horizon": horizon,
                "dataset_manifest": str(cache_manifest),
                "dataset_manifest_sha256": _digest(cache_manifest),
                "raw_manifest_sha256": dataset.provenance["contract"]["raw_manifest_sha256"],
            }
        )
    pl.DataFrame(comparisons).write_parquet(root / "method_comparison.parquet")
    (root / "optimizer_audit.json").write_text(json.dumps(optimizer, indent=2) + "\n")
    rows = []
    for record in sorted(comparisons, key=lambda r: (r["horizon"], r["k"], r["variant"])):

        def fmt(value):
            return "—" if value is None else f"{value:,.3f}"

        values = [
            html.escape(record["horizon"]),
            f'<a href="{html.escape(record["link"])}">{html.escape(record["variant"])}</a>',
            str(record["k"]),
            fmt(record["silhouette"]),
            fmt(record["mean_seed_ari"]),
            fmt(record["bic"]),
            html.escape(str(record["sizes"])),
        ]
        rows.append(
            '<tr data-horizon="'
            + record["horizon"]
            + '"><td>'
            + "</td><td>".join(values)
            + "</td></tr>"
        )
    page = """<!doctype html><html><head><meta charset="utf-8"><title>RB input profile research</title>
<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;background:#f7f9fc;color:#172839}
p{line-height:1.5;max-width:1100px}table{border-collapse:collapse;width:100%;background:white}
td,th{padding:9px;text-align:left;border-bottom:1px solid #ccd;font-size:14px}
a{color:#1d4ed8}select{padding:8px;margin:12px}h1{font-size:28px}</style></head><body>
<h1>Running back input profiles</h1><p>Four unsupervised pipelines compare two input geometries
and two clustering algorithms. Each starts with all 4,810 source-eligible raw/transformed inputs,
learns imputation and missingness handling, and retains 32 PCA components. Standard PCA uses
median/IQR scaling; balanced PCA standardizes each feature, clips at ±8, and weights each
source-family block by its inverse square root column count. Missing flags have separate
source-family weights.</p>
<p>MiniBatchKMeans uses distance to centers; diagonal Gaussian mixtures allow different
within-component variances. Higher silhouette means more separated groups in that pipeline's
own geometry; it does not prove better prediction and is not directly comparable across geometries.
Higher adjusted Rand agreement means assignments are more stable across three random seeds.
Lower BIC compares mixture counts only within one horizon and input geometry.</p>
<p>These are full-data descriptions of player-seasons, source coverage, and game context.
The same player may appear in many seasons. Roles may reflect data availability or nuisance fields;
cluster numbers are arbitrary. Future stat averages in profile files are associations.
Separate saved annual cohort assignments use only earlier years, with fixed k=5, for forecast-error
analysis. No full-data cluster chooses a historical prediction.</p>
<p>The first exploratory profile pass exposed a data bug: 365 preseason and 131 remaining-season
drafted entrants had franchise names that did not match schedule team abbreviations. These profiles
use corrected team joins; the earlier _001 artifacts remain marked as superseded diagnostic evidence.</p>
<label>Horizon <select id="horizon"><option value="all">All</option><option>season</option>
<option>remaining</option></select></label><table><thead><tr><th>Horizon</th><th>Interactive profiles</th>
<th>k</th><th>Silhouette</th><th>Mean seed ARI</th><th>GMM BIC</th><th>Cluster sizes</th></tr></thead>
<tbody>__ROWS__</tbody></table><p>Machine-readable comparisons: <a href="method_comparison.parquet">Parquet</a>.
Fit warnings and iteration budgets: <a href="optimizer_audit.json">optimizer audit</a>.</p>
<script>document.querySelector('#horizon').onchange=function(){for(const row of document.querySelectorAll('tbody tr'))
row.hidden=this.value!=='all'&&row.dataset.horizon!==this.value;};</script></body></html>"""
    (root / "index.html").write_text(page.replace("__ROWS__", "\n".join(rows)))
    snapshot = root / "implementation"
    snapshot.mkdir(exist_ok=True)
    for name in ["representations.py", "profiles.py", "profile_report.py"]:
        shutil.copyfile(Path(__file__).parent / name, snapshot / name)
    manifest = {
        "status": "complete",
        "completed_utc": datetime.now(UTC).isoformat(),
        "scope": "full-data descriptions and independent earlier-year cohort assignments",
        "input_sources": sources,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "polars", "scikit-learn", "scipy", "joblib"]
        },
        "files": {
            str(path.relative_to(root)): _digest(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path != root / "manifest.json"
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(finalize_profiles(args.root)["status"])
