"""Capture the longest available COMPLETE college-season archive at one Git revision."""

import argparse
import re

from patron.config.settings import get_settings
from patron.data.college import capture_source, discover_source, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--through-season", required=True, type=int)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", args.version):
        parser.error("Invalid version")
    inventory = discover_source()
    start = max(min(years) for years in inventory["coverage"].values())
    root = get_settings().data_dir / "research" / args.version
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "inventory.json", inventory)
    print(f"Capturing {start}–{args.through_season} at {inventory['commit']}", flush=True)
    files = capture_source(root / "raw", inventory, start, args.through_season)
    write_json(
        root / "manifest.json",
        {
            "version": args.version,
            "kind": "college_source",
            "status": "captured",
            "repository": inventory["repository"],
            "commit": inventory["commit"],
            "start_season": start,
            "through_season": args.through_season,
            "files": files,
            "limitation": "Revised provider history, not original publication-time snapshots",
        },
    )
    print(f"Captured {len(files)} verified files: {root}", flush=True)


if __name__ == "__main__":
    main()
