"""Collect the official NFL weekly page; refuse silent redirects to another week."""

from __future__ import annotations

import hashlib
import html
import json
import re
import tempfile
from pathlib import Path

import pandas as pd
import requests

from engine.names import normalize

from ..weekly_v31.artifacts import now, publish_directory, seal, timestamp, write
from .data import DATA, TEAM_ALIASES

CAPTURES = DATA / "research/opening_drive_v32_official_captures"
# An unchanged page is re-captured at most this often: often enough to stay well inside
# the 36h admission window of ``live.usable_reports``, without a copy every monitor pass.
UNCHANGED_REFRESH = pd.Timedelta(hours=2)


def content_digest(frame):
    """Digest of report rows, ignoring when they were observed (``usable_at``)."""
    rows = frame.drop(columns=["usable_at"], errors="ignore").astype("string")
    lines = sorted(
        json.dumps(r, sort_keys=True, default=lambda _: None) for r in rows.to_dict("records")
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _content(summary, frame):
    keys = ("status", "reason", "season", "week", "teams")
    return dict({k: summary.get(k) for k in keys}, reports=content_digest(frame))


def unchanged_capture(root, content, observed):
    """Latest capture of the same season/week if its content matches and it is recent."""
    for directory in sorted(Path(root).glob("[!.]*"), reverse=True):
        try:
            summary = json.loads((directory / "summary.json").read_text())
        except (OSError, ValueError):
            continue
        if (summary.get("season"), summary.get("week")) != (content["season"], content["week"]):
            continue
        recent = observed - timestamp(summary["observed_at"]) < UNCHANGED_REFRESH
        frame = pd.read_parquet(directory / "reports.parquet")
        return directory if recent and _content(summary, frame) == content else None
    return None


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
    if directory is None:
        content = dict(status=status, reason=reason, season=season, week=week, teams=teams)
        previous = unchanged_capture(CAPTURES, _content(content, frame), observed)
        if previous is not None:
            return previous
        directory = CAPTURES / observed.strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(directory)
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
