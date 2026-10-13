"""External benchmark import adapters. Fetching is explicit and offline from fitting."""

from __future__ import annotations

import io
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, time
from pathlib import Path

import polars as pl
import requests

from engine.chairstats import CampaignStore
from engine.evaluation.benchmarks import (
    QB_SCORING,
    SCORING,
    SKILL_SCORING,
    checked_bytes,
    digest,
    load_calendar,
    publish_source,
    timestamp,
    unit,
)

ESPN_STATS = {
    "attempts": "0",
    "completions": "1",
    "passing_yards": "3",
    "passing_tds": "4",
    "passing_interceptions": "20",
    "carries": "23",
    "rushing_yards": "24",
    "rushing_tds": "25",
    "receptions": "53",
    "receiving_yards": "42",
    "receiving_tds": "43",
    "targets": "58",
}
POSITIONS = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}
POSITION_TARGETS = {
    "QB": [
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ],
    "skill": [
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ],
}


def id_map(frame: pl.DataFrame, column: str) -> tuple[dict, list]:
    """Exact provider IDs only. Ambiguous identities are withheld, never name-matched."""
    mapping = defaultdict(set)
    for row in frame.select(column, "gsis_id").drop_nulls().to_dicts():
        if row[column] and row["gsis_id"]:
            mapping[str(row[column])].add(row["gsis_id"])
    return (
        {k: next(iter(v)) for k, v in mapping.items() if len(v) == 1},
        sorted(k for k, v in mapping.items() if len(v) != 1),
    )


def export_calendar(root: Path, campaign_id: str, output: Path) -> dict:
    """Extract the actual recorded cutoff clock from a dated-availability campaign."""
    if output.exists():
        raise FileExistsError(output)
    store = CampaignStore(root)
    campaign = store.campaign(campaign_id)
    values: dict[tuple[int, int], dict] = {}
    refs = []
    for model in campaign["models"]:
        ref = model.get("artifacts", {}).get("historical_candidate_audit")
        if not ref:
            continue
        f = pl.read_parquet(io.BytesIO(checked_bytes(root / campaign_id, ref)))
        if "availability_as_of" not in f.columns:
            continue
        refs.append(ref)
        for row in f.select("year", "origin", "availability_as_of").unique().to_dicts():
            key = row["year"], row["origin"]
            value = {
                "season": key[0],
                "origin_week": key[1],
                "end_week": 18 if key[0] >= 2021 else 17,
                "as_of": row["availability_as_of"],
            }
            if key in values and values[key] != value:
                raise ValueError("Campaign models disagree on cutoff")
            values[key] = value
    if not values:
        raise ValueError("Campaign has no recorded availability calendar")
    result = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": campaign["manifest_sha256"],
        "artifacts": refs,
        "policy": "Opening Drive regular-season horizon; recorded availability cutoff",
        "rows": [values[k] for k in sorted(values)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def ecr_source(
    archive: Path, crosswalk: Path, calendar: Path, output: Path, seasons: list[int]
) -> Path:
    """Import one latest complete PPR positional snapshot per preseason cutoff.

    Date-only captures are usable at end-of-day, not assumed available at midnight.
    We do not backfill missing players from earlier snapshots or use ROS pages.
    """
    raw = archive.read_bytes()
    cross = crosswalk.read_bytes()
    frame = pl.read_parquet(io.BytesIO(raw))
    mapping, ambiguous = id_map(pl.read_parquet(io.BytesIO(cross)), "fantasypros_id")
    clock = load_calendar(calendar)
    rows, audits = [], []
    for season in seasons:
        if (season, 0) not in clock:
            raise ValueError(f"No preseason calendar for {season}")
        cutoff = timestamp(clock[(season, 0)]["as_of"])
        for pos in POSITIONS.values():
            page = (
                "/nfl/rankings/qb-cheatsheets.php"
                if pos == "QB"
                else f"/nfl/rankings/ppr-{pos.lower()}-cheatsheets.php"
            )
            candidates = frame.filter(
                (pl.col("fp_page") == page)
                & (pl.col("ecr_type") == "rp")
                & (pl.col("pos") == pos)
                & pl.col("scrape_date").str.starts_with(f"{season}-")
            )
            dates = []
            for date in candidates["scrape_date"].unique():
                stamp = datetime.combine(datetime.fromisoformat(date).date(), time.max, UTC)
                if stamp <= cutoff:
                    dates.append((stamp, date))
            if not dates:
                audits.append({"season": season, "position": pos, "reason": "no_snapshot"})
                continue
            stamp, date = max(dates)
            snapshot = candidates.filter(pl.col("scrape_date") == date)
            skipped: Counter[str] = Counter()
            for r in snapshot.to_dicts():
                pid = mapping.get(str(r["id"]))
                if pid is None:
                    skipped["unlinked_or_ambiguous_id"] += 1
                    continue
                if r["ecr"] is None:
                    skipped["missing_rank"] += 1
                    continue
                rows.append(
                    {
                        "season": season,
                        "origin_week": 0,
                        "end_week": clock[(season, 0)]["end_week"],
                        "horizon": "season",
                        "position": pos,
                        "player_id": pid,
                        "target": "points",
                        "prediction": r["ecr"],
                        "unit": "rank",
                        "scoring": "standard_ppr",
                        "vintage": "archived",
                        "available_at": stamp.isoformat(),
                        "issued_at": None,
                        "source_url": "https://www.fantasypros.com" + page,
                        "archive_url": "https://github.com/dynastyprocess/data",
                        "source_player_id": str(r["id"]),
                    }
                )
            audits.append(
                {
                    "season": season,
                    "position": pos,
                    "snapshot_date": date,
                    "snapshot_rows": snapshot.height,
                    "skipped": dict(skipped),
                }
            )
    source = {
        "schema_version": 1,
        "source_id": "fantasypros_ecr",
        "kind": "ranking",
        "name": "FantasyPros preseason positional ECR (DynastyProcess archive)",
        "limitations": [
            "Ranks are not point projections; no MAE or RMSE can be computed from them.",
            "Standard PPR ranking rules can differ from campaign component scoring exclusions.",
            "Date-only archive observations use end-of-day availability; publication time unknown.",
            "Crosswalk identity links are retrospective; ambiguous/unlinked IDs are withheld.",
        ],
        "audit": audits,
        "ambiguous_provider_ids": ambiguous,
        "rows": rows,
    }
    return publish_source(
        output,
        source,
        {
            "raw/ecr.parquet": raw,
            "raw/crosswalk.parquet": cross,
            "raw/calendar.json": calendar.read_bytes(),
        },
    )


def espn_rows(players: list[dict], season: int, mapping: dict) -> tuple[list, dict]:
    """Read only ESPN season projection stats, never actuals or league-scored totals."""
    rows = []
    skipped: Counter[str] = Counter()
    for player in players:
        pos = POSITIONS.get(player.get("defaultPositionId", 0))
        if pos is None:
            continue
        pid = mapping.get(str(player["id"]))
        if pid is None:
            skipped["unlinked_or_ambiguous_id"] += 1
            continue
        matches = [
            s
            for s in player.get("stats", [])
            if s.get("seasonId") == season
            and s.get("statSourceId") == 1
            and s.get("statSplitTypeId") == 0
            and s.get("scoringPeriodId") == 0
        ]
        distinct = {
            json.dumps(s["stats"], sort_keys=True): s["stats"] for s in matches if s.get("stats")
        }
        if len(distinct) > 1:
            skipped["ambiguous_season_projection"] += 1
            continue
        if not distinct:
            skipped["no_projection"] += 1
            continue
        stats = next(iter(distinct.values()))
        targets = POSITION_TARGETS["QB" if pos == "QB" else "skill"]
        values = {
            target: float(stats[ESPN_STATS[target]])
            for target in targets
            if stats.get(ESPN_STATS[target]) is not None
        }
        # Missing components stay unknown. Do not silently turn sparse records into zeros.
        scoring = QB_SCORING if pos == "QB" else SKILL_SCORING
        if set(SCORING[scoring]).issubset(values):
            values["points"] = sum(values[k] * v for k, v in SCORING[scoring].items())
        else:
            skipped["incomplete_points_components"] += 1
        for target, value in values.items():
            rows.append(
                {
                    "season": season,
                    "origin_week": 0,
                    "end_week": 18 if season >= 2021 else 17,
                    "horizon": "season",
                    "position": pos,
                    "player_id": pid,
                    "target": target,
                    "prediction": value,
                    "unit": unit(target),
                    "scoring": scoring if target == "points" else None,
                    "vintage": "reconstructed",
                    "issued_at": None,
                    "available_at": None,
                    "source_player_id": str(player["id"]),
                    "source_url": f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/players",
                }
            )
    return rows, dict(skipped)


def capture_espn(crosswalk: Path, output: Path, seasons: list[int]) -> Path:
    if output.exists():
        raise FileExistsError(output)
    if any(season >= datetime.now(UTC).year for season in seasons):
        raise ValueError("Historical ESPN importer accepts completed seasons only")
    cross = crosswalk.read_bytes()
    mapping, ambiguous = id_map(pl.read_parquet(io.BytesIO(cross)), "espn_id")
    rows, audits = [], []
    raw = {"raw/crosswalk.parquet": cross}
    for season in seasons:
        url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/players"
        response = requests.get(
            url,
            params={"view": "kona_player_info"},
            timeout=45,
            headers={"x-fantasy-filter": json.dumps({"players": {"limit": 10000}})},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) >= 10000:
            raise ValueError("Unexpected or possibly truncated ESPN response")
        ids = [p["id"] for p in payload]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate ESPN player IDs in response")
        raw[f"raw/espn_{season}.json"] = response.content
        parsed, skipped = espn_rows(payload, season, mapping)
        if not parsed:
            raise ValueError(f"No usable ESPN projections for {season}")
        rows.extend(parsed)
        audits.append(
            {
                "season": season,
                "url": response.url,
                "captured_at": datetime.now(UTC).isoformat(),
                "sha256": digest(response.content),
                "players_received": len(payload),
                "normalized_rows": len(parsed),
                "skipped": skipped,
            }
        )
    return publish_source(
        output,
        {
            "schema_version": 1,
            "source_id": "espn_season",
            "kind": "projection",
            "name": "ESPN historical season projections (retrospective API capture)",
            "limitations": [
                "Original projection issue/publication times and later revisions are unknown.",
                "Contextual reconstructed comparison only; not a certified pre-cutoff benchmark.",
                "Missing stat fields and unmatched identities are not zeros.",
                "Points use complete components and exact campaign scoring rules.",
                "Endpoint coverage may differ from the original preseason population.",
            ],
            "scoring_rules": SCORING,
            "stat_ids": ESPN_STATS,
            "audit": audits,
            "ambiguous_provider_ids": ambiguous,
            "rows": rows,
        },
        raw,
    )
