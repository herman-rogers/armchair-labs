"""Capture 2001–2008 public weekly archives, retaining missing dates/identities.

Official pages and secondary historical reports are separate from dated model
inputs. A game date is never treated as a report-publication date.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import polars as pl
from build_injury_archive import CACHE, GOLD, OUT, ensure_open
from recover_injury_dates import text_of

from engine.data.historical_evidence import EvidenceCapture
from engine.metrics.experimental import _normalized_name, _transaction_aliases


def parse_official(html, season, week, source, identities):
    if f"{season} NFL Injury Report" not in html or f"Injuries - WEEK {week}<" not in html:
        return []
    result = []
    for unit in re.split(r'<section class="nfl-o-injury-report__unit">', html)[1:]:
        teams = re.findall(r'class="nfl-c-matchup-strip__team-abbreviation">\s*(\w+)\s*<', unit)
        bodies = re.findall(r"<tbody>(.*?)</tbody>", unit, re.S)
        if len(teams) != 2 or len(bodies) != 2:
            continue
        for team, body in zip(teams, bodies, strict=True):
            for raw in re.findall(r"<tr>(.*?)</tr>", body, re.S):
                cells = [text_of(t) for t in re.findall(r"<td\b[^>]*>(.*?)</td>", raw, re.S)]
                if len(cells) != 5:
                    continue
                ids = identities.get(_normalized_name(cells[0]), set())
                result.append(
                    dict(
                        season=season,
                        week=week,
                        game_type="REG",
                        team=team,
                        full_name=cells[0],
                        position=cells[1],
                        report_primary_injury=cells[2] or None,
                        practice_status=cells[3] or None,
                        report_status=cells[4] or None,
                        gsis_id=next(iter(ids)) if len(ids) == 1 else None,
                        identity_status="unique_name" if len(ids) == 1 else "unresolved",
                        date_modified=None,
                        source_published_at=None,
                        **source,
                    )
                )
    return result


def main():
    ensure_open()
    capture = EvidenceCapture(CACHE)
    identities = {}
    for pid, aliases in _transaction_aliases(
        pl.read_parquet(GOLD / "tables/players.parquet")
    ).items():
        for name in aliases:
            identities.setdefault(name, set()).add(pid)

    def fetch_official(item):
        y, w = item
        url = f"https://www.nfl.com/injuries/league/{y}/reg{w}"
        rec, html = capture.fetch(url)
        source = {
            "source_url": url,
            "source_sha256": rec["sha256"],
            "source_capture_file": rec["capture_file"],
            "retrieved_at": rec["retrieved_at"],
        }
        rows = (
            parse_official(html, y, w, source, identities)
            if rec["status_code"] == 200 and rec["final_url"] == url
            else []
        )
        return {**rec, "season": y, "week": w, "rows": len(rows)}, rows

    pages, records = [], []
    # 2001 has an 18-week calendar following the September postponements.
    queue = [(y, w) for y in range(2001, 2009) for w in range(1, 19 if y == 2001 else 18)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rec, rows in pool.map(fetch_official, queue):
            pages.append(rec)
            records.extend(rows)
            if len(pages) % 17 == 0:
                print(
                    f"Early official pages {len(pages)}/{len(queue)}; rows {len(records)}",
                    flush=True,
                )
    (OUT / "early_official_inventory.json").write_text(json.dumps(pages, indent=2))
    if records:
        pl.DataFrame(records, infer_schema_length=None).unique().write_parquet(
            OUT / "early_official_reports.parquet"
        )
    secondary_pages, links = [], set()
    for year in range(2001, 2009):
        url = f"https://www.jt-sw.com/football/pro/index.nsf/Documents/{year}-ir"
        rec, html = capture.fetch(url)
        found = [
            urljoin(url, p)
            for p in re.findall(r'href="([^"]+)"', html, re.I)
            if re.search(rf"/{year}-ir-(?:\d+|ps\d+)$", p)
        ]
        links.update(found)
        secondary_pages.append({**rec, "season": year, "linked_reports": len(found)})

    def fetch_secondary(url):
        rec, html = capture.fetch(url)
        dates = re.findall(r"Last Modifed:.*?(\d{2}/\d{2}/\d{4})", html, re.S)
        return {
            **rec,
            "text": text_of(html),
            "displayed_last_modified": dates[0] if dates else None,
            "source_published_at": None,
            "model_eligible": False,
        }

    with (
        (OUT / "early_secondary_reports.jsonl").open("w") as file,
        ThreadPoolExecutor(max_workers=3) as pool,
    ):
        for count, rec in enumerate(pool.map(fetch_secondary, sorted(links)), 1):
            file.write(json.dumps(rec) + "\n")
            file.flush()
            if count % 25 == 0:
                print(f"Secondary reports {count}/{len(links)}", flush=True)
    (OUT / "early_secondary_inventory.json").write_text(json.dumps(secondary_pages, indent=2))
    print("Early archive collection complete", flush=True)


if __name__ == "__main__":
    main()
