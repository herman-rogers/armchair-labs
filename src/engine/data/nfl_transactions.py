"""Dated professional-football transactions for cutoff-safe roster research.

nflverse's weekly rosters do not carry historical publication timestamps. ESPN's
public core API exposes dated team transaction logs back through the full 2004–2026
backtest window. This loader stores the team-level descriptions at the impure data
boundary; player matching and state reconstruction remain pure and unit-testable in
``metrics.experimental``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from html.parser import HTMLParser
from typing import Any

import polars as pl
import requests

BASE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/transactions"

_TEAM_ID = re.compile(r"/teams/(\d+)")
_ESPN_TEAMS = {
    1: "ATL",
    2: "BUF",
    3: "CHI",
    4: "CIN",
    5: "CLE",
    6: "DAL",
    7: "DEN",
    8: "DET",
    9: "GB",
    10: "TEN",
    11: "IND",
    12: "KC",
    13: "LV",
    14: "LA",
    15: "MIA",
    16: "MIN",
    17: "NE",
    18: "NO",
    19: "NYG",
    20: "NYJ",
    21: "PHI",
    22: "ARI",
    23: "PIT",
    24: "LAC",
    25: "SF",
    26: "SEA",
    27: "TB",
    28: "WAS",
    29: "CAR",
    30: "JAX",
    33: "BAL",
    34: "HOU",
}

_OFFICIAL_TEAM_DOMAINS = {
    "ARI": "azcardinals.com",
    "ATL": "atlantafalcons.com",
    "BAL": "baltimoreravens.com",
    "BUF": "buffalobills.com",
    "CAR": "panthers.com",
    "CHI": "chicagobears.com",
    "CIN": "bengals.com",
    "CLE": "clevelandbrowns.com",
    "DEN": "denverbroncos.com",
    "DET": "detroitlions.com",
    "GB": "packers.com",
    "HOU": "houstontexans.com",
    "IND": "colts.com",
    "JAX": "jaguars.com",
    "KC": "chiefs.com",
    "LV": "raiders.com",
    "LAC": "chargers.com",
    "LA": "therams.com",
    "MIA": "miamidolphins.com",
    "MIN": "vikings.com",
    "NE": "patriots.com",
    "NO": "neworleanssaints.com",
    "NYG": "giants.com",
    "NYJ": "newyorkjets.com",
    "PHI": "philadelphiaeagles.com",
    "PIT": "steelers.com",
    "SEA": "seahawks.com",
    "SF": "49ers.com",
    "TB": "buccaneers.com",
    "TEN": "tennesseetitans.com",
    "WAS": "commanders.com",
}
_SHORT_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})$")


class _OfficialTransactionParser(HTMLParser):
    """Extract the date/description cells from the shared NFL club page template."""

    def __init__(self) -> None:
        super().__init__()
        self._in_cell = False
        self._cell_parts: list[str] = []
        self._row: list[str] = []
        self.rows: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag == "td":
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_cell:
            self._row.append(" ".join("".join(self._cell_parts).split()))
            self._in_cell = False
        elif tag == "tr" and len(self._row) >= 2 and _SHORT_DATE.fullmatch(self._row[0]):
            self.rows.append((self._row[0], " ".join(self._row[1:])))


def _team(reference: object) -> str | None:
    if not isinstance(reference, dict):
        return None
    match = _TEAM_ID.search(str(reference.get("$ref") or ""))
    return _ESPN_TEAMS.get(int(match.group(1))) if match else None


def parse_transaction_payload(
    payload: dict[str, Any], *, year: int, source_url: str
) -> list[dict[str, object]]:
    """Normalize one ESPN API page without interpreting its prose."""
    records: list[dict[str, object]] = []
    for item in payload.get("items") or []:
        raw_date = str(item.get("date") or "")
        try:
            transaction_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
        except ValueError:
            continue
        records.append(
            {
                "transaction_date": transaction_date,
                "transaction_year": year,
                "category": "espn",
                "from_team": None,
                "source_team": _team(item.get("team")),
                "to_team": None,
                "player_name": None,
                "description": str(item.get("description") or ""),
                "source_url": source_url,
            }
        )
    return records


def _empty() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "transaction_date": pl.Date,
            "transaction_year": pl.Int32,
            "category": pl.String,
            "source_team": pl.String,
            "from_team": pl.String,
            "to_team": pl.String,
            "player_name": pl.String,
            "description": pl.String,
            "source_url": pl.String,
        }
    )


def load_transactions(
    seasons: Iterable[int],
    cutoff: str = "08-31",
    *,
    today: date | None = None,
) -> pl.DataFrame:
    """Load every transaction from January 1 through each historical cutoff.

    ESPN permits up to 1,000 records per page, so a season needs only one or two API
    calls. For the current year the effective cutoff is capped at today; a prospective
    report can never manufacture tomorrow's roster move.
    """
    cutoff_month, cutoff_day = (int(part) for part in cutoff.split("-"))
    today = today or date.today()
    records: list[dict[str, object]] = []
    for year in sorted(set(seasons)):
        requested = date(year, cutoff_month, cutoff_day)
        effective = min(requested, today) if year == today.year else requested
        date_range = f"{year}0101-{effective:%Y%m%d}"
        page = 1
        while True:
            response = requests.get(
                BASE_URL,
                params=(("limit", "1000"), ("page", str(page)), ("dates", date_range)),
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            records.extend(parse_transaction_payload(payload, year=year, source_url=response.url))
            page_count = int(payload.get("pageCount") or 0)
            if page >= page_count:
                break
            page += 1
    if not records:
        return _empty()
    return (
        pl.DataFrame(records, schema=_empty().schema, orient="row")
        .unique(subset=["transaction_date", "source_team", "description"], keep="first")
        .sort(["transaction_date", "source_team", "description"])
    )


def parse_official_transactions(
    html: str, *, year: int, team: str, source_url: str, cutoff_date: date
) -> list[dict[str, object]]:
    """Parse one official club archive page at its documented calendar dates."""
    parser = _OfficialTransactionParser()
    parser.feed(html)
    records: list[dict[str, object]] = []
    for short_date, description in parser.rows:
        match = _SHORT_DATE.fullmatch(short_date)
        if match is None or not description:
            continue
        transaction_date = date(year, int(match.group(1)), int(match.group(2)))
        if transaction_date > cutoff_date:
            continue
        records.append(
            {
                "transaction_date": transaction_date,
                "transaction_year": year,
                "category": "official",
                "from_team": None,
                "source_team": team,
                "to_team": None,
                "player_name": None,
                "description": description,
                "source_url": source_url,
            }
        )
    return records


def load_official_transactions(
    seasons: Iterable[int],
    cutoff: str = "08-31",
    *,
    today: date | None = None,
) -> pl.DataFrame:
    """Load dated transactions from official NFL club archives when available.

    The ESPN historical archive has documented date errors around roster cutdown.
    Official club pages share one table template and are authoritative for market-era
    point-in-time folds. Dallas does not expose this route and remains an explicit
    ESPN fallback in the pipeline.
    """
    cutoff_month, cutoff_day = (int(part) for part in cutoff.split("-"))
    today = today or date.today()
    records: list[dict[str, object]] = []
    for year in sorted(set(seasons)):
        requested = date(year, cutoff_month, cutoff_day)
        effective = min(requested, today) if year == today.year else requested
        for team, domain in _OFFICIAL_TEAM_DOMAINS.items():
            url = f"https://www.{domain}/team/transactions/{year}"
            try:
                response = requests.get(
                    url, headers={"User-Agent": "Armchair Labs/1.0"}, timeout=30
                )
            except requests.RequestException:
                continue
            if response.status_code != 200:
                continue
            records.extend(
                parse_official_transactions(
                    response.text,
                    year=year,
                    team=team,
                    source_url=response.url,
                    cutoff_date=effective,
                )
            )
    if not records:
        return _empty()
    return (
        pl.DataFrame(records, schema=_empty().schema, orient="row")
        .unique(subset=["transaction_date", "source_team", "description"], keep="first")
        .sort(["transaction_date", "source_team", "description"])
    )
