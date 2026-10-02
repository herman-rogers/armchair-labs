"""Recombine saved historical model libraries without refitting their base models."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from ..data import digest
from ..run import write_json
from .analyze import chronological, render
from .run import LAB


def history_library(position):
    source = LAB / "runs/history_001"
    audit = json.loads((source / "panel_audit.json").read_text())
    for name, sha in audit["array_sha256"].items():
        if digest(source / name) != sha:
            raise ValueError("Historical feature/label panel changed")
    folder = source / "tasks" / f"{position}_season"
    complete = json.loads((folder / "complete.json").read_text())
    for name, sha in complete["artifacts"].items():
        if digest(folder / name) != sha:
            raise ValueError("Historical model library changed")
    panel = pl.read_parquet(source / "panel.parquet")
    ix = np.flatnonzero(
        (panel["position"].to_numpy() == position) & (panel["horizon"].to_numpy() == "season")
    )
    panel = panel[ix]
    names = ["persistence"] + [
        r["name"] for r in json.loads((source / "candidates.json").read_text())
    ]
    years = panel["year"].to_numpy()
    keep = years >= 2006
    p = np.full((int(keep.sum()), len(names)), np.nan)
    e = panel["exposure"].to_numpy()[keep]
    y = np.load(source / "y.npy", mmap_mode="r")[ix, 0][keep]
    p[:, 0] = np.load(source / "baseline.npy", mmap_mode="r")[ix, 0][keep] * e
    fn = json.loads((source / "features.json").read_text())["names"]
    cols = [fn.index(n) for n in ["context:origin", "static:rookie", "context:history_count"]]
    context = np.load(source / "x.npy", mmap_mode="r")[ix][:, cols][keep]
    panel = panel.filter(pl.Series(keep))
    for year in np.unique(years[keep]):
        for j, name in enumerate(names[1:], 1):
            file = folder / f"{year}_{name}.npz"
            diag = json.loads(file.with_suffix(".json").read_text())
            if diag["train_last_year"] >= year:
                raise ValueError("Invalid historical fold")
            with np.load(file) as saved:
                p[years[keep] == year, j] = saved["prediction"][:, 0]
    # Strong prior representation controls join by identity, date and exact target.
    representation = LAB.parent / "representation_lab/runs/discovery_complete_001"
    manifest = json.loads((representation / "manifest.json").read_text())
    predfile = representation / "predictions.parquet"
    if digest(predfile) != manifest["files"]["predictions.parquet"]:
        raise ValueError("Representation forecasts changed")
    rp = pl.read_parquet(predfile).filter(pl.col("position") == position)
    fixed = [n for n in rp["model"].unique().sort().to_list() if "search" not in n]
    # Representation forecasts begin 2008. Keep a separate expanded library on
    # those years instead of filling unavailable old model predictions.
    ix = panel["year"].to_numpy() >= 2008
    panel, y, p, context = panel.filter(pl.Series(ix)), y[ix], p[ix], context[ix]
    for name in fixed:
        part = rp.filter(pl.col("model") == name).select(
            "player_id",
            pl.col("season").alias("year"),
            "forecast_cutoff_date",
            "actual",
            "prediction",
        )
        joined = panel.join(
            part, on=["player_id", "year", "forecast_cutoff_date"], how="left", validate="1:1"
        )
        if joined["prediction"].null_count() or not np.allclose(
            joined["actual"], y, rtol=0, atol=1e-7
        ):
            raise ValueError("Archived libraries have incompatible populations or labels")
        p = np.column_stack([p, joined["prediction"]])
        names.append("representation:" + name)
    return panel, y, p, context, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("Invalid run id")
    run = LAB / "runs" / args.run_id
    run.mkdir(exist_ok=False)
    config = dict(
        warmup_start=2008,
        ensemble_start=2011,
        evaluation_years=list(range(2014, 2026)),
        inner_seasons=5,
        seed=20260925,
        tree_checkpoints=[40, 120, 360],
    )
    sources = [
        Path(__file__),
        Path(__file__).with_name("analyze.py"),
        Path(__file__).with_name("ensemble.py"),
    ]
    record = dict(
        status="running",
        research_only=True,
        base_refits=0,
        config=config,
        source_hashes={str(p): digest(p) for p in sources},
    )
    write_json(run / "manifest.json", record)
    for position in ["QB", "RB", "WR", "TE"]:
        panel, y, p, context, names = history_library(position)
        with threadpool_limits(limits=1):
            f, m, c, b, d = chronological(panel, y, p, context, names, config)
        folder = run / f"{position}_season"
        folder.mkdir()
        f.write_parquet(folder / "forecasts.parquet")
        pl.DataFrame(m, infer_schema_length=None).write_parquet(folder / "metrics.parquet")
        # No in-season published comparator exists for these preseason tasks.
        f.head(0).write_parquet(folder / "matched.parquet")
        for name, value in [("choices", c), ("budget", b), ("diversity", d), ("library", names)]:
            write_json(folder / f"{name}.json", value)
        write_json(
            folder / "coverage.json",
            dict(
                evaluated_candidates=f.select("player_id", "year").unique().height,
                matched_published_candidates=0,
                published_candidates=0,
            ),
        )
        print(position, len(names), "saved-library ensemble complete", flush=True)
    render(run)
    if any(digest(Path(p)) != sha for p, sha in record["source_hashes"].items()):
        raise ValueError("Ensemble implementation changed")
    record.update(
        status="complete",
        artifacts={
            str(p.relative_to(run)): digest(p)
            for p in run.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        },
    )
    write_json(run / "manifest.json", record)


if __name__ == "__main__":
    main()
