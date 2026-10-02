"""Train each position's individual stat models; preserve verified notebook trials."""

import json
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from . import workbench as w


def train_one(job):
    position, target = job
    trial = w.run_trial(
        {
            **w.DEFAULT_TRIAL,
            "position": position,
            "target": target,
            "view": "all",
            "feature_scope": "position",
        }
    )
    return {
        "position": position,
        "target": target,
        "seconds": trial["metadata"]["seconds"],
        "path": str(trial["path"].relative_to(w.ROOT)),
    }


def main():
    out = w.LAB / "runs/notebook_position_stats_001"
    if out.exists():
        raise FileExistsError("Choose a new run ID to preserve existing results")
    jobs = [(p, t) for p in ["RB", "QB", "WR", "TE"] for t in w.POSITION_TARGETS[p]]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(train_one, jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
    out.mkdir()
    metrics = []
    for position in w.K:
        trials = {
            r["target"]: w.load_trial(w.ROOT / r["path"]) for r in rows if r["position"] == position
        }
        w.stat_line(trials).write_csv(out / f"{position}_predictions.csv")
        for trial in trials.values():
            metrics.extend(
                {**r, "position": position, "rounds": trial["metadata"]["selected_trees"]}
                for r in w.compare_trial(trial).to_dicts()
            )
    pl.DataFrame(metrics).write_csv(out / "metrics.csv")
    (out / "trials.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "research_only": True,
                "evaluation": "2025 remaining-season totals from Week 2, validation 2024",
                "limitation": "Exploratory historical stat models; no live forecast changes",
                "implementation_sha256": w.digest(__file__),
                "artifacts": {p.name: w.digest(p) for p in out.iterdir()},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
