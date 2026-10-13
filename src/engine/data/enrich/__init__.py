"""Enrich a raw snapshot into observation tables with DuckDB.

Every table is one SQL recipe under ``sql/`` run against views over the preserved
provider files. League scoring comes from scoring.yaml, translated to SQL.
The only Python transforms left are college normalization and identity linking,
which parse provider-specific JSON shapes before their results enter DuckDB.
No network is used and nothing here computes model features.
"""

from __future__ import annotations

import json
import shutil
import sys
from importlib.metadata import version as package_version
from pathlib import Path

import duckdb
import polars as pl

from engine.config.league import get_league
from engine.data.enrich.scoring_sql import (
    component_points_sql,
    position_override_sql,
    touchdown_bonus_sql,
)
from engine.data.enrich.sources import RawSources
from engine.data.releases import digest, identifier, load_manifest, now, reference, write_json

SQL_DIR = Path(__file__).with_name("sql")
# Tables written to the release, in build order. Intermediates stay inside DuckDB.
TABLES = (
    "nfl_players",
    "nfl_identity_crosswalk",
    "nfl_player_weeks",
    "nfl_player_seasons",
    "nfl_schedule",
    "nfl_injuries",
    "nfl_snap_counts",
    "market_rankings",
)


def sql(name: str, **params: str) -> str:
    text = (SQL_DIR / f"{name}.sql").read_text()
    return text.format(**params) if params else text


def _file_reader(path: Path, name: str) -> str:
    """Raw objects are content-addressed, so the format comes from the registered name."""
    if name.endswith(".csv"):
        return f"read_csv('{path}', header = true, all_varchar = true)"
    return f"read_parquet('{path}')"


def register_raw(
    db: duckdb.DuckDBPyConnection, sources: RawSources, seasons: list[int] | None = None
) -> dict[str, int]:
    """Views over the preserved provider files; season sets are discovered, not assumed."""
    coverage = {}
    for dataset in ("stats_player_week", "pbp", "roster_weekly", "snap_counts", "injuries"):
        wanted = [s for s in seasons if sources.has(dataset, s)] if seasons else None
        files = sources.nflverse_files(dataset, wanted)
        if not files:
            raise FileNotFoundError(f"Raw snapshot preserves no {dataset} files")
        paths = ", ".join(f"'{p}'" for p in files)
        db.execute(
            f"CREATE VIEW raw_{dataset} AS "
            f"SELECT * FROM read_parquet([{paths}], union_by_name = true)"
        )
        coverage[dataset] = len(files)
    for dataset in ("players", "schedules", "ff_playerids", "ff_rankings"):
        if dataset == "ff_rankings" and not sources.has("ff_rankings"):
            db.execute(
                "CREATE VIEW raw_ff_rankings AS SELECT NULL::VARCHAR AS scrape_date, "
                "NULL::VARCHAR AS page_type, NULL::VARCHAR AS ecr_type, NULL::VARCHAR AS id, "
                "NULL::VARCHAR AS player, NULL::VARCHAR AS pos, NULL::VARCHAR AS team, "
                "NULL::DOUBLE AS ecr, NULL::DOUBLE AS sd, NULL::DOUBLE AS best, "
                "NULL::DOUBLE AS worst, NULL::DOUBLE AS rank_delta WHERE false"
            )
            continue
        path, name = sources.resolve(dataset)
        db.execute(f"CREATE VIEW raw_{dataset} AS SELECT * FROM {_file_reader(path, name)}")
    return coverage


def score_weeks(db: duckdb.DuckDBPyConnection, history_filter: str) -> pl.DataFrame:
    """League-scored, position-repaired player weeks for the rows ``history_filter`` keeps."""
    league = get_league()
    bonus = touchdown_bonus_sql()
    db.execute(
        "CREATE OR REPLACE TABLE touchdown_bonuses AS "
        + sql("touchdown_bonuses", bonus_expression=bonus)
    )
    db.execute("CREATE OR REPLACE TABLE weekly_positions AS " + sql("weekly_positions"))
    overrides = league.position_overrides
    return db.execute(
        sql(
            "nfl_player_weeks",
            position_expression=position_override_sql(overrides, "repaired_position"),
            override_flag=(
                "player_id IN (" + ", ".join(f"'{p}'" for p in overrides) + ")"
                if overrides
                else "false"
            ),
            component_expression=component_points_sql(),
            history_filter=history_filter,
        )
    ).pl()


def build_observations(
    db: duckdb.DuckDBPyConnection, current_season: int, *, market: bool = True
) -> dict:
    """Run the SQL recipes in dependency order; returns the bonus reconciliation."""
    # Completed seasons only; the current season is served by the ``current_*`` tables.
    history_filter = f"season < {int(current_season)}"
    weeks = score_weeks(db, history_filter)
    db.register("scored_weeks", weeks.to_arrow())
    db.execute("CREATE TABLE nfl_player_weeks AS SELECT * FROM scored_weeks")
    audit = db.execute(sql("bonus_audit", bonus_expression=touchdown_bonus_sql())).fetchone()
    assert audit is not None
    duplicated = db.execute(
        "SELECT count(*) FROM (SELECT player_id, season, week FROM nfl_player_weeks "
        "GROUP BY ALL HAVING count(*) > 1)"
    ).fetchone()
    duplicates = duplicated[0] if duplicated else 0
    if duplicates:
        raise ValueError(f"{duplicates} player-weeks are duplicated in the stat feed")
    for name in ("nfl_player_seasons", "nfl_players", "nfl_identity_crosswalk", "nfl_schedule"):
        db.execute(f"CREATE TABLE {name} AS " + sql(name))
    for name in ("nfl_injuries", "nfl_snap_counts"):
        db.execute(f"CREATE TABLE {name} AS " + sql(name, history_filter=history_filter))
    if market:
        db.execute("CREATE TABLE market_rankings AS " + sql("market_rankings"))
    else:
        db.execute(
            "CREATE TABLE market_rankings (scrape_date DATE, page_type VARCHAR, ecr_type VARCHAR, "
            "fantasypros_id VARCHAR, player_id VARCHAR, player_name VARCHAR, position VARCHAR, "
            "team VARCHAR, ecr DOUBLE, sd DOUBLE, best DOUBLE, worst DOUBLE, rank_delta DOUBLE)"
        )
    scoring_plays, expected, credited, dropped = audit
    return {
        "scoring_plays": int(scoring_plays),
        "expected_points": float(expected),
        "credited_points": float(credited),
        "dropped_credits": int(dropped),
        "balanced": abs(float(expected) - float(credited)) < 1e-6,
    }


def build_college(db: duckdb.DuckDBPyConnection, sources: RawSources, tables: Path) -> dict:
    """Normalize the captured college seasons, link identities, and aggregate annually."""
    from engine.data.college import normalize_season
    from engine.data.college_identity import build_links

    manifest = sources.college_manifest()
    seasons, games, audits = [], [], []
    for year in range(manifest["start_season"], manifest["through_season"] + 1):
        frame, game, audit = normalize_season(
            *[
                pl.read_parquet(sources.college(f"raw/{kind}_{year}.parquet"))
                for kind in ("players", "rosters", "schedule")
            ]
        )
        seasons.append(frame)
        games.append(game)
        audits.append(audit)
    college = pl.concat(seasons, how="diagonal_relaxed").sort("college_id", "season", "team_id")
    college.write_parquet(tables / "college_seasons.parquet")
    pl.concat(games, how="diagonal_relaxed").write_parquet(tables / "college_games.parquet")
    links, registry, identity_audit = build_links(
        college,
        db.execute("SELECT * FROM nfl_players").pl(),
        db.execute("SELECT * FROM nfl_identity_crosswalk").pl(),
        json.loads(sources.static("college_identity_overrides.json").read_text()),
    )
    links.write_parquet(tables / "college_identity_links.parquet")
    registry.write_parquet(tables / "college_identity_registry.parquet")
    db.register("college_seasons", college.to_arrow())
    db.execute(
        f"COPY ({sql('college_annual')}) TO '{tables / 'college_annual.parquet'}' (FORMAT PARQUET)"
    )
    return {"coverage": audits, "identity": identity_audit}


def enrich(data: Path, version: str, raw_version: str | None = None) -> Path:
    raw, snapshot = load_manifest(
        data, "raw", reference(data / "raw/snapshots" / identifier(raw_version or version))
    )
    root = data / "enriched/releases" / identifier(version)
    root.mkdir(parents=True, exist_ok=False)
    tables = root / "tables"
    tables.mkdir()
    write_json(root / "build_state.json", {"status": "building", "started_at": now()})
    shutil.copytree(
        Path(__file__).parent,
        root / "implementation/enrich",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    sources = RawSources(data, snapshot)
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB", "TimeZone": "UTC"}) as db:
        coverage = register_raw(db, sources)
        print("Enrichment: reading preserved nflverse files with DuckDB", flush=True)
        bonus_audit = build_observations(
            db, snapshot["current_observations"]["season"], market=sources.has("ff_rankings")
        )
        for name in TABLES:
            db.execute(
                f"COPY (SELECT * FROM {name}) TO '{tables / (name + '.parquet')}' (FORMAT PARQUET)"
            )
        print("Enrichment: normalizing college captures", flush=True)
        college_quality = build_college(db, sources, tables)
    events = json.loads(sources.static("historical_absences.json").read_text())["events"]
    pl.DataFrame(events, infer_schema_length=None).write_parquet(tables / "absence_events.parquet")
    for name in ("weeks", "snaps", "schedule", "players"):
        shutil.copyfile(sources.current(name + ".parquet"), tables / f"current_{name}.parquet")
    write_json(root / "college_quality.json", college_quality)
    write_json(root / "build_state.json", {"status": "complete", "completed_at": now()})
    checks = {
        "offline_build": True,
        "unique_player_weeks": True,
        "bonus_credits_balanced": bonus_audit["balanced"],
    }
    write_json(
        root / "quality.json",
        {
            "checks": checks,
            "bonus_audit": bonus_audit,
            "source_files": coverage,
            "coverage_complete": False,
        },
    )
    manifest = {
        "schema_version": 1,
        "layer": "enriched",
        "version": version,
        "status": "accepted",
        "generated_at": now(),
        "quality_passed": all(checks.values()),
        "input": reference(raw),
        "season": snapshot.get("season", snapshot["current_observations"]["season"]),
        "current_observations": snapshot["current_observations"],
        "engine": "duckdb",
        "environment": {
            "python": sys.version,
            "packages": {name: package_version(name) for name in ("duckdb", "polars", "pyarrow")},
        },
        "files": {
            str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()
        },
        "tables": {p.stem: str(p.relative_to(root)) for p in sorted(tables.glob("*.parquet"))},
    }
    write_json(root / "manifest.json", manifest)
    print(f"Enriched release: {len(manifest['tables'])} tables", flush=True)
    return root
