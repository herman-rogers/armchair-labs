"""Train v2 using exactly the v1 estimator, baseline, target and selection methods."""

from __future__ import annotations

import argparse
import json
import re
from functools import partial
from pathlib import Path

from threadpoolctl import threadpool_limits

from ...gpu_lab.run import json_write
from .. import run as training
from . import data, espn, layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="opening_drive_v2_001")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "configs/opening_drive_v2.json"
    )
    parser.add_argument("--sources", type=Path, default=data.SOURCE_DIR)
    parser.add_argument(
        "--positions", nargs="+", default=["QB", "WR", "RB"], choices=list(training.STATS)
    )
    parser.add_argument(
        "--horizons", nargs="+", default=["remaining", "season"], choices=["remaining", "season"]
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    parent = json.loads((data.V1_RUN / "campaign_config.json").read_text())
    for field in (
        "positions",
        "horizons",
        "origin_weeks",
        "targets",
        "scoring",
        "recipes",
        "validation_years",
        "evaluation_years",
        "selection",
    ):
        if config[field] != parent[field]:
            raise ValueError(f"V2 comparison must preserve v1 {field}")
    if not re.fullmatch(r"[a-z][a-z0-9_]+", args.run_id):
        raise ValueError("Unsafe run ID")
    training.CAMPAIGN_ID, training.RECIPES = config["campaign_id"], config["recipes"]
    output = data.ROOT / "experiments/future_player_lab/runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign_config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Campaign config changed")
    json_write(path, config)
    loader = partial(data.load_dataset, source_dir=args.sources)
    sources = [
        Path(__file__),
        Path(data.__file__),
        Path(layer.__file__),
        Path(espn.__file__),
    ]
    with threadpool_limits(limits=4):
        for position in args.positions:
            for horizon in args.horizons:
                training.fit_cell(
                    position,
                    horizon,
                    output,
                    output / "inputs",
                    args.device,
                    dataset_loader=loader,
                    extra_sources=sources,
                    availability_contract=config["availability"],
                )


if __name__ == "__main__":
    main()
