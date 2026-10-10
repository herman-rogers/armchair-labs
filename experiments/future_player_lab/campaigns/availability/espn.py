"""Capture paginated, dated ESPN transaction reports and conservatively parse actions.

Only unique identity matches in recognized action clauses supply state. Unparsed
reports and unmatched clauses remain in the source audit; no absence inference.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

from ...gpu_lab.run import digest, json_write
from .layer import normalize_name

ACTIONS = re.compile(
    r"(?<![/\w-])(Placed|Signed|Re-signed|Resigned|Activated|Reinstated|Released|Waived|Claimed|Traded|Announced|Designated|Promoted|Elevated|Returned)\b",
    re.I,
)


def clauses(description):
    # Do not attach an unrecognized next sentence (e.g. a declined option) to
    # the previous retirement. Initials such as J.J. are not sentence breaks.
    for sentence in re.split(r"(?<=[a-z])\.\s+(?=[A-Z])|;\s*", description):
        matches = list(ACTIONS.finditer(sentence))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(sentence)
            yield match[1].lower(), sentence[match.start() : end].strip()


def state(action, clause):
    text = clause.lower()
    if action in {"activated", "reinstated", "promoted", "elevated"}:
        return "ACT"
    if action in {"released", "waived"}:
        return "CUT"
    if action == "designated":
        # Designated to return is not activated; retain reserve status.
        return "RES" if "return" in text else None
    if action == "returned":
        return "DEV" if "practice squad" in text else None
    if action == "announced":
        # Announcements can be reversed without a corresponding transaction.
        # They are a risk feature, never a hard exclusion in this sparse feed.
        return "RET_ANNOUNCED" if re.search(r"\bretirement\b", text) else None
    if action in {"placed", "signed", "re-signed", "resigned", "claimed", "traded"}:
        if "retir" in text:
            return (
                "RET" if "reserve/retired" in text or "reserve-retired" in text else "RET_ANNOUNCED"
            )
        if "physically unable" in text or "pup" in text:
            return "PUP"
        if "suspend" in text:
            return "SUS"
        if "injured reserve" in text or "reserve/" in text or "reserve list" in text:
            return "RES"
        if "practice squad" in text:
            return "DEV"
        if action != "placed":
            return "ACT"
    return None


def parse_reports(reports, aliases):
    # Longest exact normalized identity names first; no fuzzy matching.
    pattern = re.compile(
        r"(?<![a-z0-9])("
        + "|".join(
            re.escape(x) for x in sorted(aliases, key=len, reverse=True) if len(x.split()) >= 2
        )
        + r")(?![a-z0-9])"
    )
    rows, rejected = [], []
    for report in reports:
        for action, clause in clauses(report["description"]):
            status = state(action, clause)
            names = set(pattern.findall(normalize_name(clause)))
            if status is None or not names:
                rejected.append(dict(date=report["date"], action=action, clause=clause))
                continue
            # A clause containing opposing directions/statuses is not safe to
            # assign to all its names (e.g. a trade also naming the other player).
            if action == "traded" and len(names) > 1:
                rejected.append(dict(date=report["date"], action=action, clause=clause))
                continue
            for name in names:
                rows.append(
                    dict(
                        event_date=report["date"][:10],
                        player_id=aliases[name],
                        player_name=name,
                        status=status,
                        team=report["team"],
                        description=clause,
                        category=action,
                        source_url=report["source_url"],
                    )
                )
    return pl.DataFrame(rows).unique(), rejected


def capture(directory, first_year=2002, last_year=2026):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def year_pages(year):
        reports, receipts = [], []
        page, total = 1, None
        while True:
            url = (
                f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/transactions?dates={year}&limit=1000"
                + (f"&page={page}" if page > 1 else "")
            )
            path = directory / f"{year}_{page}.json"
            receipt_path = directory / f"{year}_{page}.receipt.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if digest(path) != receipt["sha256"]:
                    raise ValueError("Changed ESPN archive capture")
                doc = json.loads(path.read_text())
            else:
                for attempt in range(3):
                    response = requests.get(url, timeout=45)
                    if response.status_code < 500:
                        break
                    time.sleep(2**attempt)
                if response.status_code == 404 and page == 1:
                    doc = {"status": "success", "coverage": "unavailable_http_404"}
                    json_write(path, doc)
                    receipt = dict(
                        url=url,
                        sha256=digest(path),
                        file=path.name,
                        captured_at=datetime.now(UTC).isoformat(),
                        http_status=404,
                    )
                    json_write(receipt_path, receipt)
                    return (
                        [],
                        [receipt],
                        dict(year=year, coverage="unavailable_http_404", reports=0),
                    )
                response.raise_for_status()
                doc = response.json()
                path.write_bytes(response.content)
                receipt = dict(
                    url=url,
                    sha256=digest(path),
                    file=path.name,
                    captured_at=datetime.now(UTC).isoformat(),
                )
                json_write(receipt_path, receipt)
            receipts.append(receipt)
            entries = doc.get("transactions", [])
            if "count" not in doc:
                if doc.get("status") != "success" or entries:
                    raise ValueError(f"Unexpected archive response for {year}")
                # Explicit missing archive, never claimed to be complete coverage.
                return [], receipts, dict(year=year, coverage="unavailable", reports=0)
            if doc["pageIndex"] != page:
                raise ValueError("ESPN pagination did not advance")
            if total is not None and total != doc["count"]:
                raise ValueError("ESPN archive changed during capture")
            total = doc["count"]
            for entry in entries:
                if int(entry["date"][:4]) != year:
                    raise ValueError("Transaction outside requested year")
                reports.append(
                    dict(
                        date=entry["date"],
                        description=entry["description"],
                        team=entry.get("team", {}).get("abbreviation"),
                        source_url=url,
                    )
                )
            if page == doc["pageCount"] or total == 0:
                break
            page += 1
        if len(reports) != total:
            raise ValueError("Incomplete ESPN pagination")
        return (
            reports,
            receipts,
            dict(year=year, coverage="retrieved_all_reported_pages", reports=total),
        )

    reports, pages, coverage = [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows, receipts, year in pool.map(year_pages, range(first_year, last_year + 1)):
            reports.extend(rows)
            pages.extend(receipts)
            coverage.append(year)
            print(f"ESPN transactions {year}", flush=True)
    json_write(directory / "reports.json", reports)
    json_write(
        directory / "manifest.json",
        dict(
            schema_version=1,
            pages=pages,
            coverage=coverage,
            reports_sha256=digest(directory / "reports.json"),
            vintage="Retrospective dated transaction reports; publication-time archive unavailable",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.output)
