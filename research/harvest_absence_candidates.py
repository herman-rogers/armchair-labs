"""Capture candidate announcements for human review; never infer game caps here."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from patron.data.historical_evidence import EvidenceCapture, article_content
from patron.metrics.experimental import _normalized_name, _transaction_aliases

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Review all pre-2012 candidate headlines, including surname-only headlines",
    )
    args = parser.parse_args()
    forecasts = pl.read_parquet(args.forecasts)
    names = _transaction_aliases(pl.read_parquet(args.players))
    identity = defaultdict(set)
    for pid, aliases in names.items():
        for alias in aliases:
            identity[alias].add(pid)
    candidates = defaultdict(set)
    cutoffs = {}
    for row in forecasts.select("player_id", "forecast_season", "forecast_cutoff_date").to_dicts():
        candidates[row["forecast_season"]].add(row["player_id"])
        cutoffs[row["forecast_season"]] = row["forecast_cutoff_date"]
    index = pl.read_parquet(args.collection / "article_index.parquet")
    pattern = re.compile(
        r"suspend|suspension|season.ending|out for (?:the )?season|retir|"
        r"opt(?:s|ed|ing)?[ -]?out|\bpup\b|\bnfi\b",
        re.I,
    )
    queue = []
    for article in index.to_dicts():
        if not pattern.search(article["title"]):
            continue
        year = article["published_on"].year
        if args.legacy and year >= 2012:
            continue
        words = _normalized_name(re.sub(r"['’]s\b", "", article["title"])).split()
        matched = set()
        for size in range(2, 6):
            for start in range(len(words) - size + 1):
                matched.update(identity.get(" ".join(words[start : start + size]), set()))
        matched &= candidates[year]
        if matched or args.legacy:
            queue.append(
                {
                    **article,
                    "player_ids": sorted(matched),
                    "before_forecast_cutoff": article["published_on"] <= cutoffs[year],
                    "forecast_cutoff_date": cutoffs[year],
                    "review_status": "unreviewed",
                }
            )
    capture = EvidenceCapture(ROOT / "data/cache/historical_evidence/v1")

    def fetch(article):
        record, html = capture.fetch(article["url"])
        return {**article, **record, **article_content(html)}

    output = args.collection / (
        "absence_legacy_review.jsonl" if args.legacy else "absence_article_review.jsonl"
    )
    with output.open("w") as file, ThreadPoolExecutor(max_workers=3) as pool:
        for count, record in enumerate(pool.map(fetch, queue), 1):
            file.write(json.dumps(record, default=str) + "\n")
            file.flush()
            if count % 25 == 0:
                print(f"Absence candidates: {count}/{len(queue)}", flush=True)
    print(f"Captured {len(queue)} review candidates at {output}", flush=True)


if __name__ == "__main__":
    main()
