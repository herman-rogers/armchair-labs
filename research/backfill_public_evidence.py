"""Capture the complete official team/year grid and preseason NFL article index.

Network collection is separate from the offline forecast rebuild. Re-running with
the same output resumes saved captures; approved static exports are a separate step.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from engine.data.historical_evidence import ArticleIndexParser, EvidenceCapture, archive_rows
from engine.data.nfl_transactions import _OFFICIAL_TEAM_DOMAINS, _empty

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=2004)
    parser.add_argument("--end", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    capture = EvidenceCapture(ROOT / "data/cache/historical_evidence/v1")
    teams = {**_OFFICIAL_TEAM_DOMAINS, "DAL": "dallascowboys.com"}
    requests = [
        (year, team, domain)
        for year in range(args.start, args.end + 1)
        for team, domain in teams.items()
    ]

    def fetch_archive(item):
        year, team, domain = item
        url = f"https://www.{domain}/team/transactions/{year}"
        record, html = capture.fetch(url)
        rows, status, table_hash = archive_rows(html, record, year, team)
        return {
            **record,
            "year": year,
            "team": team,
            "archive_status": status,
            "table_sha256": table_hash,
            "rows": len(rows),
        }, rows

    pages, batches = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for page, rows in pool.map(fetch_archive, requests):
            pages.append(page)
            batches.append(rows)
            if len(pages) % 32 == 0:
                print(
                    f"Archives: {len(pages)}/{len(requests)}; rows={sum(len(r) for r in batches)}",
                    flush=True,
                )
    # A year route that silently returns another year's table is not dated evidence.
    repeated = defaultdict(list)
    for index, page in enumerate(pages):
        if page["rows"]:
            repeated[page["team"], page["table_sha256"]].append(index)
    for indices in repeated.values():
        if len(indices) > 1:
            for index in indices:
                pages[index]["archive_status"] = "duplicate_year_table"
                batches[index] = []
    rows = [row for batch in batches for row in batch]
    transactions = pl.DataFrame(rows, infer_schema_length=None) if rows else _empty()
    transactions = transactions.with_columns(pl.col("transaction_year").cast(pl.Int32))
    transactions.write_parquet(args.output / "official_transactions.parquet")
    (args.output / "archive_inventory.json").write_text(json.dumps(pages, indent=2) + "\n")

    article_pages, articles = [], []
    requests = [(year, month) for year in range(args.start, args.end + 1) for month in range(1, 9)]

    def fetch_index(item):
        year, month = item
        record, html = capture.fetch(f"https://www.nfl.com/sitemap/html/articles/{year}/{month}")
        index = ArticleIndexParser()
        if record["status_code"] == 200:
            index.feed(html)
        matched = [
            r
            for r in index.records
            if r["published_on"].year == year and r["published_on"].month == month
        ]
        return {**record, "year": year, "month": month, "rows": len(matched)}, matched

    with ThreadPoolExecutor(max_workers=3) as pool:
        for page, rows in pool.map(fetch_index, requests):
            article_pages.append(page)
            articles.extend(rows)
            if len(article_pages) % 8 == 0:
                print(
                    f"Article months: {len(article_pages)}/{len(requests)}; "
                    f"articles={len(articles)}",
                    flush=True,
                )
    pl.DataFrame(articles, infer_schema_length=None).unique("url").write_parquet(
        args.output / "article_index.parquet"
    )
    (args.output / "article_index_inventory.json").write_text(
        json.dumps(article_pages, indent=2) + "\n"
    )
    (args.output / "collection.json").write_text(
        json.dumps(
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "start": args.start,
                "end": args.end,
                "archive_pages": len(pages),
                "transaction_rows": transactions.height,
                "article_pages": len(article_pages),
                "indexed_articles": len(articles),
                "limitations": [
                    "Current retrospective captures, not original publication vintages.",
                    "Archive presence is not proof of complete event coverage.",
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
