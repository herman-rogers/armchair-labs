"""Immutable all-team provider injury observations; never backdate current statuses."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import pandas as pd
import requests

from ..weekly_v31.artifacts import now, publish_directory, seal, write
from .data import DATA, HISTORY

URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"


def parse(payload, identity, observed_at, week):
    if len({str(t["id"]) for t in payload.get("injuries", [])}) != 32:
        raise ValueError("Capture must enumerate all 32 teams")
    mapping = identity.dropna(subset=["espn_id", "gsis_id"]).copy()
    mapping["espn_id"] = mapping.espn_id.astype(str).str.replace(r"\.0$", "", regex=True)
    unique = mapping.groupby("espn_id").gsis_id.nunique()
    mapping = (
        mapping[mapping.espn_id.isin(unique[unique.eq(1)].index)]
        .drop_duplicates("espn_id")
        .set_index("espn_id")
    )
    rows = []
    for team in payload["injuries"]:
        for record in team.get("injuries", []):
            a = record.get("athlete", {})
            ids = {
                m.group(1)
                for link in a.get("links", [])
                if (m := re.search(r"/player/(?:[^/]+/)?_/id/(\d+)", link.get("href", "")))
            }
            if a.get("id"):
                ids.add(str(a["id"]))
            espn_id = next(iter(ids)) if len(ids) == 1 else None
            player_id = mapping.loc[espn_id, "gsis_id"] if espn_id in mapping.index else None
            rows.append(
                dict(
                    player_id=player_id,
                    espn_id=espn_id,
                    player_name=a.get("displayName"),
                    position=a.get("position", {}).get("abbreviation"),
                    team=a.get("team", {}).get("abbreviation"),
                    season=int(payload["season"]["year"]),
                    forecast_week_context=week,
                    provider_status=record.get("status"),
                    provider_updated_at=record.get("date"),
                    observed_at=observed_at,
                    usable_at=observed_at,
                    evidence_kind="provider_injury_status_not_official_inactives",
                    injury_type=record.get("details", {}).get("type"),
                    source_url=URL,
                    identity_state="exact_espn_id" if player_id else "unresolved",
                )
            )
    return pd.DataFrame(rows)


def capture(week, directory=None):
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    observed = now()
    payload = response.json()
    directory = (
        Path(directory)
        if directory
        else DATA
        / "research/opening_drive_v32_injury_captures"
        / observed.strftime("%Y%m%dT%H%M%SZ")
    )
    identity = pd.read_parquet(HISTORY / "identity.parquet")
    frame = parse(payload, identity, observed.isoformat(), week)
    if directory.exists():
        raise FileExistsError("Use a new immutable capture directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".injury-") as tmp:
        staging = Path(tmp) / "capture"
        staging.mkdir()
        (staging / "response.json").write_bytes(response.content)
        identity.to_parquet(staging / "identity.parquet", index=False)
        frame.to_parquet(staging / "observations.parquet", index=False)
        write(
            staging / "summary.json",
            dict(
                observed_at=observed.isoformat(),
                provider_timestamp=payload.get("timestamp"),
                season=payload["season"],
                week=week,
                teams=32,
                rows=len(frame),
                exact_identity_rows=int(frame.player_id.notna().sum()),
                limitations=[
                    "No row is not proof of health",
                    "Provider status is not an official inactive list",
                    "Usable only from collection time; never from an older story timestamp",
                    "No backfill into historical evaluation",
                    "Forecast week is collection context, not a verified game-week designation",
                ],
            ),
        )
        seal(
            staging,
            dict(status="captured_research_observation", url=URL, observed_at=observed.isoformat()),
        )
        publish_directory(staging, directory)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True, choices=range(1, 19))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(capture(args.week, args.output))
