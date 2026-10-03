"""Cross-reference undated injury rows with publication-dated NFL reporting.

Never fills original date_modified. Creates separate corroborated observations
with the later publication/modification timestamp, exact player, injury, status,
season and week, plus a retained source capture. Practice fields remain unknown.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from html import unescape

import polars as pl
from build_injury_archive import CACHE, GOLD, OUT, ensure_open

from engine.data.historical_evidence import ArticleIndexParser, EvidenceCapture, article_content
from engine.metrics.experimental import _normalized_name


def text_of(html):
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", html)).replace("\ufeff", "").split())


def corroborate(article, html, reports, schedule):
    """Only status lists in explicitly numbered weekly reports are auto-admitted."""
    content = article_content(html)
    match = re.search(r"\bweek\s+(\d+)\b", article["title"], re.I)
    published, modified = content["published_at"], content["modified_at"]
    if not match or not published or not modified:
        return []
    try:
        pub = datetime.fromisoformat(published.replace("Z", "+00:00"))
        mod = datetime.fromisoformat(modified.replace("Z", "+00:00"))
    except ValueError:
        return []
    known = max(pub, mod)
    season, week = int(article["season"]), int(match[1])
    games = schedule.filter(
        (pl.col("season") == season) & (pl.col("week") == week) & (pl.col("game_type") == "REG")
    )
    if not games.height:
        return []
    first = min(str(d)[:10] for d in games["gameday"])
    last = max(str(d)[:10] for d in games["gameday"])
    if (
        not 0 <= (datetime.fromisoformat(first).date() - pub.date()).days <= 8
        or known.date().isoformat() > last
    ):
        return []
    # Exclude later season-end annotations and unrelated sidebar/navigation text.
    lists = []
    for raw in re.findall(r"<li\b[^>]*>(.*?)</li>", html, re.S | re.I):
        text = text_of(raw)
        status = re.match(r"^(OUT|DOUBTFUL|QUESTIONABLE|PROBABLE):\s*(.*)", text)
        if status:
            lists.append((status[1].lower(), status[2]))
    accepted = []
    rows = reports.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("game_type") == "REG")
        & pl.col("date_modified").is_null()
    )
    for row in rows.to_dicts():
        if not row["gsis_id"] or not row["report_status"] or not row["report_primary_injury"]:
            continue
        name = _normalized_name(row["full_name"])
        injuries = set(re.findall(r"[a-z]+", row["report_primary_injury"].lower())) - {
            "left",
            "right",
        }
        matches = []
        for status, text in lists:
            if status != row["report_status"].lower():
                continue
            for mention in re.finditer(r"(?:^|,\s*)([A-Z]{1,4})\s+([^()]+?)\s*\(([^()]+)\)", text):
                injury_words = set(re.findall(r"[a-z]+", mention[3].lower()))
                if _normalized_name(mention[2]) == name and injuries and injuries <= injury_words:
                    matches.append(mention[0].lstrip(", "))
        if not matches:
            continue
        # A distinct observation, not a manufactured original-feed timestamp.
        accepted.append(
            {
                **{
                    k: row[k]
                    for k in [
                        "season",
                        "week",
                        "game_type",
                        "gsis_id",
                        "position",
                        "full_name",
                        "team",
                        "report_primary_injury",
                        "report_status",
                    ]
                },
                "record_id": f"corroborated:{row['record_id']}:{article['sha256'][:12]}",
                "original_record_id": row["record_id"],
                "date_modified": known,
                "practice_status": None,
                "practice_primary_injury": None,
                "source_url": article["url"],
                "source_sha256": article["sha256"],
                "source_capture_file": article["capture_file"],
                "source_published_at": published,
                "source_modified_at": modified,
                "retrieved_at": article["retrieved_at"],
                "date_basis": "later_of_article_publication_and_modification",
                "match_basis": "weekly_report_exact_name_injury_status",
                "evidence_text": matches[0],
            }
        )
    return accepted


def main():
    ensure_open()
    capture = EvidenceCapture(CACHE)
    index = pl.read_parquet(OUT / "article_index.parquet")
    rec, html = capture.fetch("https://www.nfl.com/sitemap/html/articles/2026/1")
    parser = ArticleIndexParser()
    if rec["status_code"] == 200:
        parser.feed(html)
    january = [r for r in parser.records if str(r["published_on"]).startswith("2026-01")]
    if january:
        index = pl.concat([index, pl.DataFrame(january)]).unique("url")
    index.write_parquet(OUT / "article_index_through_202601.parquet")
    (OUT / "january_2026_index_capture.json").write_text(json.dumps(rec, indent=2))
    queue = []
    for row in index.to_dicts():
        year = row["published_on"].year - (row["published_on"].month == 1)
        if year in {2009, 2010, 2025} and re.search(
            r"week\s+\d+.*injury report|injury report.*week\s+\d+", row["title"], re.I
        ):
            queue.append({**row, "season": year})

    def fetch(row):
        rec, html = capture.fetch(row["url"])
        return {**row, **rec}, html

    reports = pl.read_parquet(OUT / "injury_reports.parquet")
    schedule = pl.read_parquet(GOLD / "tables/nfl_schedule.parquet")
    observations, sources = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for rec, html in pool.map(fetch, queue):
            sources.append({**rec, **article_content(html)})
            if rec["status_code"] == 200:
                observations.extend(corroborate(rec, html, reports, schedule))
    (OUT / "timestamp_recovery_sources.json").write_text(json.dumps(sources, indent=2, default=str))
    if observations:
        pl.DataFrame(observations, infer_schema_length=None).unique("record_id").write_parquet(
            OUT / "corroborated_reports.parquet"
        )
    print(
        json.dumps(
            {"weekly_sources": len(sources), "corroborated_observations": len(observations)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
