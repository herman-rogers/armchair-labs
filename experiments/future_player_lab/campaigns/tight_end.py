"""Train and export the TE extension of Opening Drive v2 without altering sealed campaigns."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from functools import partial
from pathlib import Path

from threadpoolctl import threadpool_limits

from ..forecasting.data import add_supplements, history_inputs, snapshot_inputs
from ..gpu_lab.run import digest, json_write
from . import data as base
from . import run as training
from .availability import data, espn, layer
from .availability.forecast import export

CONFIG = Path(__file__).parent / "configs/opening_drive_v2_te.json"


def validate_config(config):
    parent = json.loads(CONFIG.with_name("opening_drive_v2.json").read_text())
    if config["positions"] != ["TE"]:
        raise ValueError("This extension supports TE only")
    for key in (
        "horizons",
        "origin_weeks",
        "recipes",
        "validation_years",
        "evaluation_years",
        "selection",
    ):
        if config[key] != parent[key]:
            raise ValueError(f"TE extension must preserve v2 {key}")
    if config["targets"] != {"TE": [*base.STATS["TE"], "points"]}:
        raise ValueError("Unsupported TE target contract")
    if config["scoring"] != {"TE": base.SCORING["TE"]}:
        raise ValueError("Unsupported TE scoring contract")
    for key in (
        "version",
        "cutoff",
        "transaction_clock",
        "roster_clock",
        "eligibility",
        "source_vintage",
    ):
        if config["availability"][key] != parent["availability"][key]:
            raise ValueError(f"TE extension must preserve v2 availability {key}")


def load_dataset(position, horizon, cache, *, source_dir=data.SOURCE_DIR):
    if position != "TE":
        raise ValueError("TE extension requires TE")
    parent = Path(cache).parent / "base"
    base.load_dataset(position, horizon, parent / "inputs")
    return data.load_dataset(
        position,
        horizon,
        cache,
        source_dir=source_dir,
        parent_run=parent,
        population=json.loads(CONFIG.read_text())["availability"]["population"],
    )


def prepare_base_forecasts(run, snapshot, supplements, season=2026):
    """Seal new TE cutoff inputs; there was no TE panel in the original v1 run."""
    parent = Path(run) / "base"
    tables = history_inputs(parent / "inputs/history")
    tables, snapshot_audit = snapshot_inputs(tables, Path(snapshot), season, 2)
    tables, supplement_audit = add_supplements(tables, Path(supplements), season, 2)
    files = {}
    for horizon, origin in (("remaining", 2), ("season", 0)):
        directory = parent / "forecasts" / f"te_{horizon}"
        directory.mkdir(parents=True, exist_ok=True)
        panel = base.feature_rows(tables, "TE", season, origin)
        features = directory / f"{season}_w{origin}_features.parquet"
        receipt = directory / f"{season}_w{origin}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            frozen = data.checked_parquet(directory, features.name, saved["feature_parquet_sha256"])
            if (
                not frozen.equals(panel)
                or saved["snapshot"] != snapshot_audit
                or saved["supplements"] != supplement_audit
            ):
                raise ValueError("Sealed TE base forecast changed")
        else:
            panel.write_parquet(features)
            json_write(
                receipt,
                dict(
                    position="TE",
                    season=season,
                    origin_week=origin,
                    feature_parquet_sha256=digest(features),
                    snapshot=snapshot_audit,
                    supplements=supplement_audit,
                    source_sha256=digest(Path(__file__)),
                ),
            )
        files[str(receipt.relative_to(parent))] = digest(receipt)
    manifest = parent / "manifest.json"
    content = dict(kind="te_base_cutoff_inputs", files=files)
    if manifest.exists() and json.loads(manifest.read_text()) != content:
        raise ValueError("Sealed TE base manifest changed")
    json_write(manifest, content)
    return parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="opening_drive_v2_te_001")
    parser.add_argument("--sources", type=Path, default=data.SOURCE_DIR)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--registry", type=Path, default=base.ROOT / "data/campaigns")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=base.ROOT / "data/raw/snapshots/weekly_20261008_w4_r3/manifest.json",
    )
    parser.add_argument("--supplements", type=Path, default=data.SUPPLEMENTS)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]+", args.run_id):
        raise ValueError("Unsafe run ID")
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    run = base.ROOT / "experiments/future_player_lab/runs" / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / "campaign_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("TE campaign configuration changed; use a fresh run ID")
    json_write(config_path, config)
    training.CAMPAIGN_ID, training.RECIPES = config["campaign_id"], config["recipes"]
    with threadpool_limits(limits=4):
        if not args.export_only:
            # Reuse only the verified raw observation cache, never another position's rows.
            cache = run / "base/inputs/history"
            if not cache.exists():
                history_inputs(data.V1_RUN / "inputs/history")
                shutil.copytree(data.V1_RUN / "inputs/history", cache)
            loader = partial(load_dataset, source_dir=args.sources)
            for horizon in config["horizons"]:
                training.fit_cell(
                    "TE",
                    horizon,
                    run,
                    run / "inputs",
                    args.device,
                    dataset_loader=loader,
                    extra_sources=[
                        Path(__file__),
                        Path(data.__file__),
                        Path(layer.__file__),
                        Path(espn.__file__),
                    ],
                    availability_contract=config["availability"],
                )
        parent = prepare_base_forecasts(run, args.snapshot, args.supplements)
        print(json.dumps(export(run, args.sources, args.registry, parent_run=parent), indent=2))


if __name__ == "__main__":
    main()
