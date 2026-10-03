"""Verify captured archives and export a hash-bound offline forecast input."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from engine.data.historical_evidence import (
    BACKFILL_FILE,
    BACKFILL_MANIFEST,
    expand_dated_revisions,
    load_transaction_backfill,
    sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    frame = pl.read_parquet(args.collection / "official_transactions.parquet")
    inventory = json.loads((args.collection / "archive_inventory.json").read_text())
    allowed = {
        r["capture_file"]: r["sha256"] for r in inventory if r["archive_status"] == "captured"
    }
    for row in frame.select("source_capture_file", "source_sha256").unique().to_dicts():
        if allowed.get(row["source_capture_file"]) != row["source_sha256"]:
            raise ValueError("Transaction is not backed by an accepted year archive")
    path = args.data_dir / "static" / BACKFILL_FILE
    frame = expand_dated_revisions(frame)
    frame.sort("transaction_date", "source_team", "description").write_parquet(path)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "collection": args.collection.name,
        "rows": frame.height,
        "sha256": sha256(path.read_bytes()),
        "captures": allowed,
        "source_date_basis": "official_archive_event_date",
        "limitations": [
            "Retrospective web captures, not original publication vintages.",
            "Archive presence does not demonstrate event completeness.",
            "Full-year rows must be filtered at each forecast cutoff.",
        ],
    }
    (path.parent / BACKFILL_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    verified = load_transaction_backfill(args.data_dir)
    assert verified is not None and verified.height == frame.height
    print(f"Verified {frame.height} rows from {len(allowed)} archives: {path}")


if __name__ == "__main__":
    main()
