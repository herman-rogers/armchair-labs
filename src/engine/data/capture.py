"""Capture a raw snapshot straight from the providers; no accepted history archive.

A snapshot preserves the provider files the DuckDB enrichment reads: nflverse
player stats, play-by-play, weekly rosters, snap counts, injuries, identities,
schedules, the ffverse ID crosswalk and the FantasyPros ECR archive. Completed
seasons never change, so their files are carried forward from the previous snapshot
by content hash; only the current season and the single-file datasets are fetched
again. The current season's completed weeks are scored with the same DuckDB recipes
as history and preserved as ``current/*.parquet`` for the model scorers.
"""

from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path

import duckdb
import polars as pl
import requests

from engine.config.settings import REPO_ROOT, get_settings
from engine.data.enrich import register_raw, score_weeks
from engine.data.enrich.sources import DATASETS, SEASON_START, RawSources, asset_name
from engine.data.releases import (
    digest,
    identifier,
    inside,
    now,
    preserve_object,
    read_json,
    write_json,
)

STATIC_INPUTS = ("historical_absences.json", "college_identity_overrides.json")
SKILL = ("QB", "RB", "WR", "TE")
TEAM_ALIASES = {"LAR": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS"}


def fetch(url: str, *, attempts: int = 4, timeout: int = 120) -> bytes:
    """Download one provider file; retry transient failures with backoff."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code >= 500:
                raise requests.HTTPError(f"{response.status_code} for {url}")
            response.raise_for_status()
            return response.content
        except (requests.RequestException, OSError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}: {last}")


def latest_snapshot(data: Path) -> dict | None:
    """The most recently registered accepted raw snapshot, for carrying history forward."""
    parent = data / "raw/snapshots"
    candidates = []
    for root in parent.iterdir() if parent.is_dir() else []:
        manifest = root / "manifest.json"
        if manifest.is_file():
            try:
                document = read_json(manifest)
            except (OSError, ValueError):
                continue
            if document.get("status") == "accepted" and document.get("registered_at"):
                candidates.append((document["registered_at"], document))
    return max(candidates)[1] if candidates else None


def completed_cutoff(db: duckdb.DuckDBPyConnection, season: int) -> int:
    """Only consecutive finished weeks with stat coverage for every scheduled team."""
    rows = db.execute(
        """
        WITH games AS (
            SELECT week, home_team, away_team, home_score, away_score
            FROM raw_schedules WHERE season = ? AND game_type = 'REG'
        ),
        teams AS (
            SELECT DISTINCT week, team FROM raw_stats_player_week
            WHERE season = ? AND season_type = 'REG'
        )
        SELECT g.week,
               bool_and(g.home_score IS NOT NULL AND g.away_score IS NOT NULL) AS finished,
               bool_and(h.team IS NOT NULL AND a.team IS NOT NULL) AS covered
        FROM games g
        LEFT JOIN teams h ON h.week = g.week AND h.team = g.home_team
        LEFT JOIN teams a ON a.week = g.week AND a.team = g.away_team
        GROUP BY g.week ORDER BY g.week
        """,
        [season, season],
    ).fetchall()
    status = {int(week): bool(finished and covered) for week, finished, covered in rows}
    cutoff = 0
    for week in range(1, 19):
        if not status.get(week):
            break
        cutoff = week
    return cutoff


def current_tables(
    db: duckdb.DuckDBPyConnection, season: int, cutoff: int
) -> dict[str, pl.DataFrame]:
    """League-scored current weeks, mapped snaps, rookies and schedule through the cutoff."""
    weeks = score_weeks(db, f"season = {int(season)} AND week <= {int(cutoff)}")
    snaps = db.execute(
        """
        SELECT p.gsis_id AS player_id, CAST(s.season AS INTEGER) AS season,
               CAST(s.week AS INTEGER) AS week, s.offense_snaps, s.offense_pct
        FROM raw_snap_counts s
        JOIN (SELECT pfr_id, gsis_id FROM raw_players
              WHERE pfr_id IS NOT NULL AND gsis_id IS NOT NULL
              QUALIFY count(*) OVER (PARTITION BY pfr_id) = 1) p
          ON p.pfr_id = s.pfr_player_id
        WHERE s.season = ? AND s.game_type = 'REG' AND s.week <= ?
        ORDER BY season, week, player_id
        """,
        [season, cutoff],
    ).pl()
    players = db.execute(
        """
        SELECT gsis_id AS player_id, display_name AS player_display_name,
               CASE upper(position) WHEN 'FB' THEN 'RB' WHEN 'HB' THEN 'RB'
                    ELSE upper(position) END AS position,
               CAST(? AS INTEGER) AS season, CAST(draft_pick AS INTEGER) AS draft_pick,
               TRY_CAST(espn_id AS BIGINT) AS espn_id, latest_team AS team
        FROM raw_players
        WHERE coalesce(rookie_season, draft_year) = ? AND gsis_id IS NOT NULL
          AND upper(position) IN ('QB', 'RB', 'WR', 'TE', 'FB', 'HB')
        ORDER BY player_id
        """,
        [season, season],
    ).pl()
    schedule = db.execute(
        "SELECT * REPLACE (CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week) "
        "FROM raw_schedules WHERE season = ? ORDER BY week, gameday, game_id",
        [season],
    ).pl()
    return {"weeks": weeks, "snaps": snaps, "players": players, "schedule": schedule}


def capture_raw(
    data: Path,
    version: str,
    season: int,
    college_source: str,
    *,
    previous: dict | None = None,
    refetch_history: bool = False,
    fetcher=fetch,
) -> Path:
    """Register a complete raw snapshot for ``season`` through its latest completed week."""
    root = data / "raw/snapshots" / identifier(version)
    root.mkdir(parents=True, exist_ok=False)
    previous_assets = (previous or {}).get("assets", {})
    assets: dict[str, dict] = {}
    carried = 0

    def preserve(name: str, path: Path, kind: str, origin: str, expected: str | None = None):
        expected = expected or digest(path)
        assets[name] = {
            "object": preserve_object(data, path, expected),
            "sha256": expected,
            "bytes": path.stat().st_size,
            "origin_path": origin,
            "kind": kind,
        }

    with tempfile.TemporaryDirectory(prefix="engine-capture-") as temporary:
        work = Path(temporary)
        for dataset, (template, _) in DATASETS.items():
            seasons: list[int | None] = (
                list(range(SEASON_START[dataset], season + 1)) if "{season}" in template else [None]
            )
            for year in seasons:
                name = asset_name(dataset, year)
                reusable = year is not None and year < season and not refetch_history
                if reusable and name in previous_assets:
                    assets[name] = dict(previous_assets[name])
                    inside(data, assets[name]["object"])  # must still exist
                    carried += 1
                    continue
                url = template.format(season=year)
                path = work / Path(name).name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(fetcher(url))
                preserve(name, path, "provider_file", url)
                print(f"Captured {name}", flush=True)
        for filename in STATIC_INPUTS:
            path = REPO_ROOT / "data/static" / filename
            preserve(
                f"static/{filename}", path, "reviewed_annotation", str(path.relative_to(REPO_ROOT))
            )
        college = data / "research" / identifier(college_source)
        college_manifest = read_json(college / "manifest.json")
        if college_manifest["status"] != "captured" or college_manifest["kind"] != "college_source":
            raise ValueError("College source is not a completed capture")
        preserve(
            "college/manifest.json", college / "manifest.json", "capture_metadata", str(college)
        )
        for spec in college_manifest["files"]:
            preserve(
                "college/raw/" + spec["filename"],
                inside(college / "raw", spec["filename"]),
                "provider_file",
                str(college / "raw" / spec["filename"]),
                spec["sha256"],
            )
        # Score the current season's completed weeks with the same recipes as history.
        sources = RawSources(data, {"assets": assets})
        with duckdb.connect(config={"threads": 4, "memory_limit": "2GB", "TimeZone": "UTC"}) as db:
            register_raw(db, sources, seasons=[season])
            cutoff = completed_cutoff(db, season)
            if cutoff == 0:
                raise ValueError(f"No completed regular-season week for {season}")
            tables = current_tables(db, season, cutoff)
        current = work / "current"
        current.mkdir()
        for name, frame in tables.items():
            frame.write_parquet(current / f"{name}.parquet")
        saved_at = now()
        report = {
            "season": season,
            "through_week": cutoff,
            "saved_at": saved_at,
            "snapshot_path": ".",
            "current_source": "nflverse weekly statistics, schedules, play-by-play and PFR snaps",
            "source_sha256": {p.name: digest(p) for p in current.glob("*.parquet")},
        }
        write_json(current / "source_report.json", report)
        for path in sorted(current.iterdir()):
            preserve(f"current/{path.name}", path, "scored_current_capture", "derived in capture")
    manifest = {
        "schema_version": 1,
        "layer": "raw",
        "version": version,
        "status": "accepted",
        "registered_at": now(),
        "season": season,
        "college_source": college_source,
        "previous_snapshot": (previous or {}).get("version"),
        "carried_forward_assets": carried,
        "current_observations": {"season": season, "through_week": cutoff, "saved_at": saved_at},
        "assets": assets,
        "files": {a["object"]: a["sha256"] for a in assets.values()},
        "source_kinds": dict(Counter(a["kind"] for a in assets.values())),
        "limitations": [
            "Provider files are the bytes served at capture time, "
            "not original publication vintages.",
            "Completed seasons are carried forward from the previous snapshot by content hash.",
            "Registration time is not publication time.",
        ],
    }
    write_json(root / "manifest.json", manifest)
    print(
        f"Raw snapshot {version}: {len(assets):,} assets ({carried} carried forward); "
        f"{season} complete through week {cutoff}",
        flush=True,
    )
    return root


def completed_week(season: int, *, fetcher=fetch) -> int:
    """Fetch only the schedule and current stats to find the latest completed week."""
    with tempfile.TemporaryDirectory(prefix="engine-cutoff-") as temporary:
        work = Path(temporary)
        paths = {}
        for dataset in ("schedules", "stats_player_week"):
            template = DATASETS[dataset][0]
            path = work / f"{dataset}.parquet"
            path.write_bytes(fetcher(template.format(season=season)))
            paths[dataset] = path
        with duckdb.connect() as db:
            db.execute(
                f"CREATE VIEW raw_schedules AS SELECT * FROM read_parquet('{paths['schedules']}')"
            )
            db.execute(
                "CREATE VIEW raw_stats_player_week AS "
                f"SELECT * FROM read_parquet('{paths['stats_player_week']}')"
            )
            return completed_cutoff(db, season)


def default_season() -> int:
    from engine.config.league import get_league

    return get_league().draft_season


__all__ = [
    "capture_raw",
    "completed_cutoff",
    "completed_week",
    "current_tables",
    "default_season",
    "fetch",
    "latest_snapshot",
]

if __name__ == "__main__":  # pragma: no cover - convenience for manual captures
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--college-source", required=True)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--refetch-history", action="store_true")
    arguments = parser.parse_args()
    data_dir = get_settings().data_dir.resolve()
    capture_raw(
        data_dir,
        arguments.version,
        arguments.season or default_season(),
        arguments.college_source,
        previous=latest_snapshot(data_dir),
        refetch_history=arguments.refetch_history,
    )
