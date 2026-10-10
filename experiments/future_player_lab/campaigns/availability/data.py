"""Sealed availability inputs and additive datasets built from frozen v1 panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ...gpu_lab.run import digest, json_write
from ...notebooks import raw_distribution_data as raw
from . import espn, layer

ROOT = raw.ROOT
V1_RUN = ROOT / "experiments/future_player_lab/runs/opening_drive_v1_001"
SOURCE_DIR = ROOT / "data/research/opening_drive_v2_sources"
SUPPLEMENTS = ROOT / "experiments/future_player_lab/runs/forecast_sources_20261009_003"


def checked_parquet(directory, filename, expected):
    path = Path(directory) / filename
    if digest(path) != expected:
        raise ValueError(f"Availability input checksum mismatch: {path}")
    return pl.read_parquet(path)


def prepare_sources(directory=SOURCE_DIR):
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        load_sources(directory)
        return
    transaction_manifest = directory / "espn/manifest.json"
    manifest = json.loads(transaction_manifest.read_text())
    reports_path = directory / "espn/reports.json"
    if digest(reports_path) != manifest["reports_sha256"]:
        raise ValueError("Transaction archive checksum mismatch")
    roster = raw.read_family("rosters")
    injuries = raw.read_family("injuries")
    identity = raw.read_family("identity")
    schedule = raw.read_family("schedule")
    supplements = json.loads((SUPPLEMENTS / "manifest.json").read_text())
    supplement_receipts = {}
    for name in ("rosters", "injuries"):
        receipt = supplements["sources"][name]
        if receipt["status"] != "available":
            continue
        frame = checked_parquet(SUPPLEMENTS, receipt["path"], receipt["sha256"])
        supplement_receipts[name] = receipt
        if name == "rosters":
            roster = pl.concat([roster, frame], how="diagonal_relaxed")
        else:
            injuries = pl.concat([injuries, frame], how="diagonal_relaxed")
    aliases, _ = layer.name_index(identity, roster)
    transactions, rejected = espn.parse_reports(json.loads(reports_path.read_text()), aliases)
    events, matching = layer.build_events(transactions, roster, identity, schedule)
    matching["unparsed_transaction_clauses"] = rejected
    matching["transaction_coverage"] = manifest["coverage"]
    events.write_parquet(directory / "events.parquet")
    injuries.write_parquet(directory / "injuries.parquet")
    schedule.write_parquet(directory / "schedule.parquet")
    json_write(directory / "matching.json", matching)
    json_write(
        directory / "manifest.json",
        dict(
            schema_version=1,
            transaction_manifest_sha256=digest(transaction_manifest),
            raw_snapshot_sha256=digest(raw.SNAPSHOT),
            historical_assets=raw.catalog()
            .filter(pl.col("family").is_in(["rosters", "injuries", "identity", "schedule"]))
            .to_dicts(),
            supplement_manifest_sha256=digest(SUPPLEMENTS / "manifest.json"),
            supplement_receipts=supplement_receipts,
            source_code={
                str(p.relative_to(ROOT)): digest(p)
                for p in [Path(__file__), Path(layer.__file__), Path(espn.__file__)]
            },
            files={
                name: digest(directory / name)
                for name in [
                    "events.parquet",
                    "injuries.parquet",
                    "schedule.parquet",
                    "matching.json",
                ]
            },
            limitations=[
                "Retrospective event archive; event dates are not archived publication times.",
                "No transaction row is not proof of active status; "
                "unknown identity matches withheld.",
                "Week-level rosters are usable only after the recorded week ends.",
                "Dated absence is not future non-participation; "
                "only unambiguous latest retirement excludes.",
            ],
        ),
    )


def load_sources(directory=SOURCE_DIR):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    frames = {}
    for name, sha in manifest["files"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"Availability source checksum mismatch: {name}")
        if name.endswith(".parquet"):
            frames[Path(name).stem] = pl.read_parquet(directory / name)
    return frames, manifest


def augment(panel, sources):
    return layer.attach_availability(
        panel, sources["events"], sources["injuries"], sources["schedule"]
    )


def load_dataset(
    position,
    horizon,
    cache,
    *,
    source_dir=SOURCE_DIR,
    parent_run=V1_RUN,
    population="V1 candidates, excluding latest as-of RET; no future-appearance filtering",
):
    directory = Path(cache) / f"{position.lower()}_{horizon}"
    base = Path(parent_run) / "inputs" / directory.name
    meta = json.loads((base / "dataset.json").read_text())
    contract = dict(
        parent_dataset_sha256=digest(base / "dataset.json"),
        parent_panel_sha256=meta["panel_sha256"],
        availability_manifest_sha256=digest(Path(source_dir) / "manifest.json"),
        source_sha256=digest(Path(__file__)),
        layer_sha256=digest(Path(layer.__file__)),
        position=position,
        horizon=horizon,
        population=population,
    )
    if (directory / "dataset.json").exists():
        saved = json.loads((directory / "dataset.json").read_text())
        if saved["contract"] != contract:
            raise ValueError("Availability dataset contract changed; start a fresh run")
        panel = checked_parquet(directory, "panel.parquet", saved["panel_sha256"])
        checked_parquet(directory, "candidate_audit.parquet", saved["candidate_audit_sha256"])
        return panel, saved["feature_names"]
    panel = checked_parquet(base, "panel.parquet", meta["panel_sha256"])
    sources, _ = load_sources(source_dir)
    augmented = augment(panel, sources)
    # Prove the full original matrix, outcomes, and row identities survived the join.
    if not augmented.select(panel.columns).equals(panel.sort("year", "player_id")):
        raise ValueError("Availability layer changed frozen base observations")
    names = meta["feature_names"] + [c for c in augmented.columns if c.startswith("availability__")]
    audit = augmented.select(
        "year",
        "origin",
        "player_id",
        "player_display_name",
        "team",
        *[c for c in augmented.columns if c.startswith("availability")],
    )
    eligible = augmented.filter(pl.col("availability_eligible"))
    directory.mkdir(parents=True, exist_ok=False)
    audit.write_parquet(directory / "candidate_audit.parquet")
    eligible.write_parquet(directory / "panel.parquet")
    json_write(
        directory / "dataset.json",
        dict(
            contract=contract,
            feature_names=names,
            rows=eligible.height,
            columns=len(names),
            parent_rows=panel.height,
            excluded_rows=panel.height - eligible.height,
            panel_sha256=digest(directory / "panel.parquet"),
            candidate_audit_sha256=digest(directory / "candidate_audit.parquet"),
            coverage=audit.group_by("year", "availability_status", "availability_eligible")
            .len()
            .sort("year", "availability_status")
            .to_dicts(),
        ),
    )
    print(
        f"availability {position}/{horizon}: {panel.height} candidates, "
        f"{eligible.height} eligible, {len(names) - len(meta['feature_names'])} added features",
        flush=True,
    )
    return eligible, names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=SOURCE_DIR)
    args = parser.parse_args()
    prepare_sources(args.sources)
