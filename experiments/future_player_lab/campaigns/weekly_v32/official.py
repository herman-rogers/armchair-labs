"""Collect the official NFL weekly page; refuse silent redirects to another week."""

from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path

import pandas as pd
import requests

from engine.names import normalize

from ..weekly_v31.artifacts import now, publish_directory, seal, write
from .data import DATA, TEAM_ALIASES


def plain(value):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def parse(content, season, week, identity, observed_at):
    title = re.search(r"<title>(.*?)</title>", content, re.S)
    match = re.search(r"Week (\d+) of the (\d+) Season", plain(title[1]) if title else "")
    if not match or (int(match[2]), int(match[1])) != (season, week):
        raise ValueError("Official page does not identify the requested season/week")
    lookup = identity.copy()
    lookup["key"] = lookup.player_name.fillna("").map(normalize)
    rows, teams = [], set()
    units = content.split('<section class="nfl-o-injury-report__unit">')[1:]
    if not units:
        raise ValueError("Official report structure is unavailable")
    for unit in units:
        codes = re.findall(r'nfl-c-matchup-strip__team-abbreviation">\s*([^<]+)', unit)[:2]
        tables = re.findall(r"<tbody>(.*?)</tbody>", unit, re.S)
        if len(codes) != 2 or len(tables) != 2:
            raise ValueError("Official team/report table mismatch")
        for code, table in zip(codes, tables, strict=True):
            code = TEAM_ALIASES.get(code.strip(), code.strip())
            if code in teams:
                raise ValueError("Duplicate team in weekly report")
            teams.add(code)
            for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
                cells = [plain(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
                if len(cells) != 5:
                    raise ValueError("Official injury columns changed")
                name, position, _, practice, status = cells
                candidates = lookup[
                    (lookup.key == normalize(name))
                    & (lookup.team == code)
                    & (lookup.position == position)
                ]
                ids = candidates.player_id.dropna().unique()
                rows.append(
                    dict(
                        player_id=ids[0] if len(ids) == 1 else None,
                        player_name=name,
                        position=position,
                        team=code,
                        season=season,
                        week=week,
                        usable_at=observed_at,
                        report_status=status or None,
                        practice_status=practice or None,
                        source="official_nfl_weekly_page",
                        identity_state="unique_name_team_position"
                        if len(ids) == 1
                        else "unresolved",
                        evidence_kind="official_weekly_report",
                    )
                )
    return pd.DataFrame(rows), sorted(teams)


def capture(season, week, identity, directory=None):
    url = f"https://www.nfl.com/injuries/league/{season}/reg{week}"
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    observed = now()
    teams, frame = [], pd.DataFrame()
    try:
        frame, teams = parse(response.text, season, week, identity, observed.isoformat())
        status, reason = "usable_weekly_report", None
    except ValueError as error:
        status, reason = "unusable_weekly_report", str(error)
    directory = (
        Path(directory)
        if directory
        else (
            DATA
            / "research/opening_drive_v32_official_captures"
            / observed.strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".official-") as tmp:
        staging = Path(tmp) / "capture"
        staging.mkdir()
        (staging / "response.html").write_bytes(response.content)
        identity.to_parquet(staging / "identity.parquet", index=False)
        frame.to_parquet(staging / "reports.parquet", index=False)
        write(
            staging / "summary.json",
            dict(
                status=status,
                reason=reason,
                season=season,
                week=week,
                observed_at=observed.isoformat(),
                teams=teams,
                rows=len(frame),
                source_url=url,
                evidence="Official weekly reports; not official inactives",
            ),
        )
        seal(staging, dict(status=status, observed_at=observed.isoformat(), url=url))
        publish_directory(staging, directory)
    return directory
