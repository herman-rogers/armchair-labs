"""Immutable source captures; transport failures never mean empty coverage."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests

from engine.data.nfl_transactions import _team
from engine.data.releases import digest, write_json


def capture(root: Path, key: str, url: str) -> dict:
    """Keep failed receipts too; a new snapshot retries unavailable endpoints."""
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / f"{key}.json"
    body = root / f"{key}.bin"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["url"] != url or digest(body) != record["sha256"]:
            raise ValueError(f"Capture changed: {key}")
        return record
    raw, status, error, final_url = b"", 0, None, url
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=45)
            raw, status, final_url = response.content, response.status_code, response.url
            if status < 500 and status != 429:
                break
        except requests.RequestException as exc:
            error = str(exc)
        time.sleep(2**attempt)
    body.write_bytes(raw)
    record = dict(
        key=key,
        url=url,
        final_url=final_url,
        http_status=status,
        error=error,
        collected_at=datetime.now(UTC).isoformat(),
        file=body.name,
        sha256=digest(body),
    )
    write_json(receipt, record)
    return record


def _espn_attempt(root: Path, year: int, family: str, limit: int):
    rows, seen, expected, pages = [], set(), None, []
    page = 1
    while True:
        if family == "site":
            url = (
                "https://site.api.espn.com/apis/site/v2/sports/football/nfl/transactions"
                f"?dates={year}&limit={limit}&page={page}"
            )
        else:
            url = (
                "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/transactions"
                f"?dates={year}0101-{year}1231&limit={limit}&page={page}"
            )
        key = (
            f"espn_{year}_{page}"
            if (family, limit) == ("site", 500)
            else f"espn_{family}_{limit}_{year}_{page}"
        )
        rec = capture(root, key, url)
        pages.append(rec["key"])
        if rec["http_status"] != 200:
            return rows, dict(year=year, source="espn", status="fetch_failed", pages=pages)
        doc = json.loads((root / rec["file"]).read_bytes())
        entries_key = "transactions" if family == "site" else "items"
        if not {"count", "pageCount", "pageIndex", entries_key}.issubset(doc):
            return rows, dict(year=year, source="espn", status="invalid_response", pages=pages)
        if doc["pageIndex"] != page or expected not in (None, doc["count"]):
            raise ValueError(f"Unstable ESPN pagination: {year}/{page}")
        expected = doc["count"]
        entries = doc[entries_key]
        signature = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        if signature in seen:
            raise ValueError(f"Repeated ESPN page: {year}/{page}")
        seen.add(signature)
        for entry in entries:
            if int(entry["date"][:4]) != year:
                raise ValueError(f"Wrong ESPN year: {year}")
            rows.append(
                dict(
                    event_date=entry["date"][:10],
                    description=entry["description"],
                    team=(
                        entry.get("team", {}).get("abbreviation")
                        if family == "site"
                        else _team(entry.get("team"))
                    ),
                    source="espn",
                    source_url=url,
                    capture_key=rec["key"],
                    raw_sha256=rec["sha256"],
                    collected_at=rec["collected_at"],
                    published_at=None,
                )
            )
        if page >= doc["pageCount"]:
            break
        page += 1
    if len(rows) != expected:
        raise ValueError(f"Incomplete ESPN year: {year}")
    return rows, dict(
        year=year,
        source="espn",
        status="all_reported_pages" if rows else "empty",
        reports=len(rows),
        pages=pages,
    )


def espn_year(root: Path, year: int) -> tuple[list[dict], dict]:
    """Retry incomplete archive requests through the alternate ESPN representation.

    Keep failures in receipts, and never combine overlapping partial attempts.
    """
    attempts = []
    for family, limit in (("site", 500), ("core", 1000), ("core", 500)):
        rows, status = _espn_attempt(root, year, family, limit)
        attempts.append(status)
        if status["status"] == "all_reported_pages":
            return rows, {**status, "attempts": attempts}
    return [], dict(year=year, source="espn", status="unavailable", attempts=attempts)


def capture_sources(root: Path, first: int, last: int) -> tuple[list[dict], list[dict]]:
    if first < 2002 or last < first or last > datetime.now(UTC).year:
        raise ValueError("Expected years from 2002 through the current calendar year")
    reports, coverage = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for rows, status in pool.map(lambda y: espn_year(root, y), range(first, last + 1)):
            reports.extend(rows)
            coverage.append(status)
            print(f"ESPN {status['year']}: {status['status']} ({len(rows)} reports)", flush=True)
    jobs = (
        [
            (f"roster_{y}", f"weekly_rosters/roster_weekly_{y}.parquet")
            for y in range(first, last + 1)
        ]
        + [
            (f"injuries_{y}", f"injuries/injuries_{y}.parquet")
            for y in range(max(2009, first), last + 1)
        ]
        + [("players", "players/players.parquet")]
    )

    def download(job):
        key, part = job
        result = capture(
            root, key, "https://github.com/nflverse/nflverse-data/releases/download/" + part
        )
        return dict(source=key, status=result["http_status"], capture_key=key)

    with ThreadPoolExecutor(max_workers=3) as pool:
        for status in pool.map(download, jobs):
            coverage.append(status)
            print(f"nflverse {status['source']}: {status['status']}", flush=True)
    capture(
        root, "schedule", "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    )
    return reports, coverage
