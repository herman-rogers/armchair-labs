"""Immutable all-team provider injury observations; never backdate current statuses."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import requests

from ..weekly_v31.artifacts import digest, now, publish_directory, seal, write
from .data import DATA, HISTORY

URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ROOT = DATA / "research/opening_drive_v32_injury_captures"
# Shared content-addressed identity snapshots (no summary.json, so never a "capture").
IDENTITY_STORE = "_identity"
# Collection-time columns; everything else defines whether the provider's view changed.
VOLATILE = ("observed_at", "usable_at")
LIMITATIONS = [
    "No row is not proof of health",
    "Provider status is not an official inactive list",
    "Usable only from collection time; never from an older story timestamp",
    "No backfill into historical evaluation",
    "Forecast week is collection context, not a verified game-week designation",
]


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


def observations_digest(frame):
    """Order-insensitive digest of observations, ignoring when they were collected."""
    stable = frame.drop(columns=[c for c in VOLATILE if c in frame.columns])
    rows = sorted(json.dumps(r, sort_keys=True, default=str) for r in stable.to_dict("records"))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def latest(root):
    """Newest capture summary, using the same ordering as engine.data.freshness."""
    summaries = (p for p in Path(root).glob("*/summary.json") if not p.parent.name.startswith("."))
    path = max(summaries, default=None)
    if path is None:
        return None, {}
    summary = json.loads(path.read_text())
    if "observations_digest" not in summary and (path.parent / "observations.parquet").exists():
        # Captures written before digests were recorded.
        observations = pd.read_parquet(path.parent / "observations.parquet")
        summary["observations_digest"] = observations_digest(observations)
    return path.parent, summary


def read_response(directory):
    """Raw provider payload from a capture; gzip since 2026-10, plain JSON before."""
    directory = Path(directory)
    compressed = directory / "response.json.gz"
    if compressed.exists():
        return json.loads(gzip.decompress(compressed.read_bytes()))
    return json.loads((directory / "response.json").read_text())


def store_identity(root):
    """Copy the identity table once per distinct content; return its sha256."""
    source = HISTORY / "identity.parquet"
    sha = digest(source)
    target = Path(root) / IDENTITY_STORE / f"{sha}.parquet"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".", delete=False) as tmp:
            shutil.copyfile(source, tmp.name)
        os.replace(tmp.name, target)
    return sha


def capture(week, directory=None):
    """Capture current provider observations.

    When the observations equal the newest capture's, only a small summary records the
    check (so freshness still advances); the payload and observations are not rewritten.
    """
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    observed = now()
    payload = response.json()
    directory = Path(directory) if directory else ROOT / observed.strftime("%Y%m%dT%H%M%SZ")
    root = directory.parent
    identity = pd.read_parquet(HISTORY / "identity.parquet")
    frame = parse(payload, identity, observed.isoformat(), week)
    if directory.exists():
        raise FileExistsError("Use a new immutable capture directory")
    root.mkdir(parents=True, exist_ok=True)
    observations = observations_digest(frame)
    previous, prior = latest(root)
    unchanged = prior.get("observations_digest") == observations
    summary = dict(
        observed_at=observed.isoformat(),
        provider_timestamp=payload.get("timestamp"),
        season=payload["season"],
        week=week,
        teams=32,
        rows=len(frame),
        exact_identity_rows=int(frame.player_id.notna().sum()),
        observations_digest=observations,
        limitations=LIMITATIONS,
    )
    with tempfile.TemporaryDirectory(dir=root, prefix=".injury-") as tmp:
        staging = Path(tmp) / "capture"
        staging.mkdir()
        if unchanged:
            # Same observations as an earlier full capture: record the check only.
            summary["observations_from"] = prior.get("observations_from", previous.name)
            summary["identity_sha256"] = prior.get("identity_sha256")
            status = "unchanged_research_observation"
        else:
            (staging / "response.json.gz").write_bytes(gzip.compress(response.content, mtime=0))
            frame.to_parquet(staging / "observations.parquet", index=False)
            summary["identity_sha256"] = store_identity(root)
            status = "captured_research_observation"
        write(staging / "summary.json", summary)
        seal(staging, dict(status=status, url=URL, observed_at=observed.isoformat()))
        publish_directory(staging, directory)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True, choices=range(1, 19))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(capture(args.week, args.output))
