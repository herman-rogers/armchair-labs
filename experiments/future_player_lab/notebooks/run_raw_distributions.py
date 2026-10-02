"""Reproducible offline campaign for the raw-data marimo notebook."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl

from . import raw_distribution_data as data
from . import raw_distribution_models as models

RUNS = Path(__file__).resolve().parents[1] / "runs"
DEFAULT_RUN = RUNS / "raw_distributions_001"


def write_fold(result, path):
    path.mkdir(parents=True, exist_ok=False)
    for name in ["rows", "summary", "selection", "test"]:
        result[name].write_parquet(path / f"{name}.parquet")
    np.savez_compressed(path / "draws.npz", **result["draws"])
    (path / "metadata.json").write_text(
        json.dumps({k: result[k] for k in ["chronology", "features", "engine", "seed"]}, indent=2)
    )


def load_saved(path=DEFAULT_RUN, verify=True):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if verify:
        for name, sha in manifest["files"].items():
            if data.digest(path / name) != sha:
                raise ValueError(f"Saved result changed: {name}")
    return {
        "rows": pl.read_parquet(path / "rows.parquet"),
        "summary": pl.read_parquet(path / "summary.parquet"),
        "intervals": pl.read_parquet(path / "paired_crps.parquet"),
        "manifest": manifest,
        "path": path,
    }


def campaign(
    out=DEFAULT_RUN,
    horizons=("season", "week", "remaining"),
    years=(2023, 2024, 2025),
    position="RB",
    origins=(2,),
    engine="lightgbm",
    rounds=(40, 100),
    draws=1000,
    ablations=True,
    market=False,
    progress=print,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sources = [Path(data.__file__), Path(models.__file__), Path(__file__)]
    (out / "implementation").mkdir()
    for source in sources:
        shutil.copyfile(source, out / "implementation" / source.name)
    config = {
        "position": position,
        "horizons": list(horizons),
        "years": list(years),
        "origins": list(origins),
        "engine": engine,
        "rounds": list(rounds),
        "draws": draws,
        "ablations": ablations,
        "market": market,
        "seed": 20260925,
        "raw_manifest_sha256": data.digest(data.SNAPSHOT),
        "sources": {str(p.relative_to(data.ROOT)): data.digest(p) for p in sources},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    data.catalog().write_parquet(out / "raw_catalog.parquet")
    results = []
    for horizon in horizons:
        progress(f"Building {position} / {horizon} from raw sources")
        panel, features, audit = data.make_panel(position, horizon, origins)
        panel.write_parquet(out / f"panel_{horizon}.parquet")
        audit.write_parquet(out / f"unknown_targets_{horizon}.parquet")
        data.stability(panel).write_parquet(out / f"stability_{horizon}.parquet")
        progress(f"{panel.height:,} candidate origins, {len(features):,} candidate numeric inputs")
        for year in years:
            result = models.run_fold(
                panel,
                features,
                year,
                engine,
                rounds,
                draws,
                ablations=ablations,
                market=market and horizon == "week",
                progress=progress,
            )
            write_fold(result, out / f"{horizon}_{year}")
            results.append(result["rows"])
    rows = pl.concat(results, how="diagonal_relaxed")
    rows.write_parquet(out / "rows.parquet")
    models.summarize(rows).write_parquet(out / "summary.parquet")
    models.paired_crps_intervals(rows).write_parquet(out / "paired_crps.parquet")
    pooled = (
        rows.group_by("horizon", "target", "model")
        .agg(
            pl.len().alias("n"),
            pl.col("crps").mean(),
            pl.col("negative_log_score").mean(),
            pl.col("covered80").mean(),
            pl.col("squared_error").mean().sqrt().alias("rmse"),
        )
        .sort("horizon", "target", "crps")
    )
    pooled.write_csv(out / "pooled.csv")
    readout = [
        "# Raw-data distribution experiment",
        "",
        "Exploratory historical results; lower CRPS/log score is better.",
        "No Bayesian models. No production rankings were changed.",
        "",
        "Scoring: rushing/receiving yards ÷ 10 + receptions + 6 × rushing/receiving TDs.",
        "This excludes fumbles, passing, bonuses, returns and two-point conversions.",
        "",
        "| Horizon | Target | Model | N | CRPS | Negative log score | 80% coverage | RMSE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in pooled.iter_rows(named=True):
        nll = r["negative_log_score"]
        nll_text = f"{nll:.3f}" if nll is not None and np.isfinite(nll) else "—"
        readout.append(
            f"| {r['horizon']} | {r['target']} | {r['model']} | {r['n']} | "
            f"{r['crps']:.3f} | {nll_text} | {r['covered80']:.1%} | {r['rmse']:.3f} |"
        )
    (out / "readout.md").write_text("\n".join(readout) + "\n")
    packages = {p: version(p) for p in ["numpy", "polars", "scipy", "scikit-learn"]}
    if engine in {"lightgbm", "xgboost"}:
        packages[engine] = version(engine)
    manifest = {
        "config": config,
        "python": platform.python_version(),
        "packages": packages,
        "seconds": time.monotonic() - started,
        "files": {
            str(p.relative_to(out)): data.digest(p) for p in sorted(out.rglob("*")) if p.is_file()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return load_saved(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_RUN)
    p.add_argument(
        "--horizons",
        nargs="+",
        choices=["season", "week", "remaining"],
        default=["season", "week", "remaining"],
    )
    p.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    p.add_argument("--origins", type=int, nargs="+", default=[2])
    p.add_argument("--position", choices=["RB", "WR", "TE"], default="RB")
    p.add_argument("--engine", choices=["lightgbm", "xgboost", "hist"], default="lightgbm")
    p.add_argument("--rounds", type=int, nargs="+", default=[40, 100])
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--no-ablations", action="store_true")
    p.add_argument("--market", action="store_true")
    args = vars(p.parse_args())
    args["ablations"] = not args.pop("no_ablations")
    campaign(**args, progress=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
