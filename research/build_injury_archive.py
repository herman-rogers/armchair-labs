"""Capture a versioned public injury archive; never infer publication from game dates.

Run with .venv/bin/python research/build_injury_archive.py. Resumes immutable URL
captures. News matches are discovery candidates, not approved absence constraints.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from engine.data.historical_evidence import ArticleIndexParser, EvidenceCapture, article_content
from engine.metrics.experimental import _normalized_name, _transaction_aliases

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/research/injury_archive_20260925_r1"
GOLD = ROOT / "data/gold/releases/canonical_targets_20260923_r3"
CACHE = ROOT / "data/cache/historical_evidence/v1"
PATTERN = re.compile(
    r"injur|\bout\b|return|surg|knee|ankle|hamstring|concussion|achilles|acl|mcl|"
    r"\bir\b|reserve|\bpup\b|\bnfi\b|suspend|suspension|fractur|broken|"
    r"healthy|health|practice|questionable|doubtful|ruled|tear|torn|rehab|retir|opt.out",
    re.I,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_open():
    if (OUT / "manifest.json").exists() and json.loads((OUT / "manifest.json").read_text()).get(
        "sealed"
    ):
        raise ValueError("Archive is sealed; create a new version for further collection")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_open()
    manifest = json.loads((GOLD / "manifest.json").read_text())
    if (
        digest(GOLD / "manifest.json")
        != "34f4d34388fa248434014ce76583001178a5cd5c0a87358ea708c2e9d6125dfa"
    ):
        raise ValueError("Gold manifest changed")
    tables = {}
    for name in [
        "nfl_injuries",
        "nfl_transactions",
        "nfl_schedule",
        "players",
        "preseason_features",
    ]:
        entry = manifest["tables"][name]
        path = GOLD / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Gold input changed: {name}")
        tables[name] = pl.read_parquet(path)
    tables["nfl_injuries"].write_parquet(OUT / "injury_reports.parquet")
    tables["nfl_transactions"].write_parquet(OUT / "transactions.parquet")
    capture = EvidenceCapture(CACHE)

    def fetch_index(item):
        year, month = item
        record, html = capture.fetch(f"https://www.nfl.com/sitemap/html/articles/{year}/{month}")
        parser = ArticleIndexParser()
        if record["status_code"] == 200:
            parser.feed(html)
        rows = [
            r for r in parser.records if (r["published_on"].year, r["published_on"].month) == item
        ]
        return {**record, "year": year, "month": month, "rows": len(rows)}, rows

    pages, articles = [], []
    months = [(y, m) for y in range(2001, 2026) for m in range(1, 13)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for page, rows in pool.map(fetch_index, months):
            pages.append(page)
            articles.extend(rows)
            if len(pages) % 12 == 0:
                print(
                    f"Index months {len(pages)}/{len(months)}; articles {len(articles)}", flush=True
                )
    (OUT / "article_index_inventory.json").write_text(json.dumps(pages, indent=2))
    index = (
        pl.DataFrame(articles, infer_schema_length=None).unique("url").sort("published_on", "url")
    )
    index.write_parquet(OUT / "article_index.parquet")
    # Historical RB identity comes from the candidate population, never test outcomes.
    preseason = tables["preseason_features"]
    position_col = "position" if "position" in preseason.columns else "pos"
    rb_ids = set(preseason.filter(pl.col(position_col) == "RB")["player_id"])
    aliases = _transaction_aliases(tables["players"].filter(pl.col("gsis_id").is_in(rb_ids)))
    identity = {}
    for pid, names in aliases.items():
        for name in names:
            identity.setdefault(name, set()).add(pid)
    queue = []
    for article in index.to_dicts():
        if not PATTERN.search(article["title"]):
            continue
        words = _normalized_name(re.sub(r"['’]s\b", "", article["title"])).split()
        matched = set()
        for size in range(2, 6):
            for start in range(len(words) - size + 1):
                matched.update(identity.get(" ".join(words[start : start + size]), set()))
        if matched:
            queue.append({**article, "player_ids": sorted(matched)})
    pl.DataFrame(queue, infer_schema_length=None).write_parquet(OUT / "rb_news_candidates.parquet")

    def fetch_article(article):
        record, html = capture.fetch(article["url"])
        return {**article, **record, **article_content(html), "review_status": "unreviewed"}

    with (OUT / "rb_news.jsonl").open("w") as file, ThreadPoolExecutor(max_workers=4) as pool:
        for count, record in enumerate(pool.map(fetch_article, queue), 1):
            file.write(json.dumps(record, default=str) + "\n")
            file.flush()
            if count % 50 == 0:
                print(f"RB article captures {count}/{len(queue)}", flush=True)
    # Retain existing reviewed decisions, now bind each to a captured original source.
    ledger = json.loads((ROOT / "data/static/historical_absences.json").read_text())
    source_reviews = []
    for event in ledger["events"]:
        record, html = capture.fetch(event["source_url"])
        content = article_content(html)
        source_reviews.append(
            {
                "evidence_id": event["evidence_id"],
                **record,
                "published_at": content["published_at"],
                "modified_at": content["modified_at"],
            }
        )
    (OUT / "absence_source_reviews.json").write_text(json.dumps(source_reviews, indent=2))
    (OUT / "absence_events.json").write_text(json.dumps(ledger, indent=2))
    files = {p.name: digest(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "gold_root": str(GOLD.relative_to(ROOT)),
                "gold_manifest_sha256": digest(GOLD / "manifest.json"),
                "files": files,
                "indexed_articles": index.height,
                "rb_news_candidates": len(queue),
                "limitations": [
                    "Retrospective captures, not original publication vintages.",
                    "News candidates are not automatically approved absence constraints.",
                    "No-report is unknown, not healthy. Event dates are not publication dates.",
                    "Article index coverage does not establish complete reporting coverage.",
                ],
            },
            indent=2,
        )
    )
    print(f"Completed {OUT}", flush=True)


if __name__ == "__main__":
    main()
