"""Prepare a new immutable expanded feature matrix and published benchmark."""

import argparse
import json
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path

from ..data import digest
from ..run import write_json
from .data import prepare, published_benchmark
from .run import LAB, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("Invalid run id")
    run = LAB / "runs" / args.run_id
    run.mkdir(exist_ok=False)
    config = json.loads(args.config.read_text())
    write_json(run / "config.json", config)
    files = [Path(__file__), Path(__file__).with_name("data.py"), LAB / "data.py"]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in files}
    for p in files:
        dest = run / "source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    manifest = dict(status="running", started=datetime.now(UTC).isoformat(), source=sources)
    write_json(run / "manifest.json", manifest)
    try:
        hashes, meta = prepare(ROOT, config, run)
        frame, more, _ = published_benchmark(ROOT, config)
        hashes.update(more)
        frame.write_parquet(run / "published.parquet")
        if any(digest(ROOT / p) != sha for p, sha in {**sources, **hashes}.items()):
            raise ValueError("Sources changed during preparation")
        manifest.update(
            status="complete",
            inputs=hashes,
            columns=meta["columns"],
            rows=meta["rows"],
            artifacts={
                str(p.relative_to(run)): digest(p)
                for p in run.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        write_json(run / "manifest.json", manifest)
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc())
        write_json(run / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
