"""League captures use the same Parquet → catalog → native DuckDB machinery.

Private, frequently refreshed league data has its own local catalog, so an ESPN
poll neither rebuilds the analytical database nor publishes account data to GCS.
All application league observations are reconstructed from these queried tables.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import get_args, get_type_hints

import polars as pl
import yaml

from engine.data.releases import digest
from engine.data.shared import json_bytes, lock, write_new
from engine.tables.build import refresh
from engine.tables.products import PAYLOAD, ROW
from engine.tables.query import connect
from engine.tables.storage import content_hash, load_catalog


def record_schema(cls) -> dict:
    """Stable scalar types, including empty tables and all-null observations."""
    primitives = {int: pl.Int64, float: pl.Float64, str: pl.String, bool: pl.Boolean}
    schema = {}
    for name, annotation in get_type_hints(cls).items():
        choices = [t for t in get_args(annotation) if t is not type(None)]
        scalar = choices[0] if len(choices) == 1 else annotation
        schema[name] = primitives.get(scalar, pl.String)
    return schema


def records_frame(records: list[dict], schema: dict | None = None) -> pl.DataFrame:
    """Typed scalar columns plus lossless JSON for nested and optional fields."""
    rows = []
    for index, record in enumerate(records):
        row = {
            key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
            for key, value in record.items()
        }
        if {ROW, PAYLOAD}.intersection(row):
            raise ValueError("Reserved league record columns")
        rows.append({**row, ROW: index, PAYLOAD: json.dumps(record, allow_nan=False)})
    if not rows:
        return pl.DataFrame(schema={**(schema or {}), ROW: pl.Int64, PAYLOAD: pl.String})
    return pl.from_dicts(rows, infer_schema_length=None, schema_overrides=schema).with_columns(
        pl.col(ROW).cast(pl.Int64)
    )


def capture(
    directory: Path, groups: dict[str, list[dict]], schemas: dict | None = None
) -> tuple[Path, dict]:
    """Seal one complete capture and return its pinned catalog, under one writer lock."""
    key = content_hash({"groups": groups, "implementation": digest(Path(__file__))})
    data = directory / "data"
    with lock(directory / ".capture-lock"):
        receipt = directory / "captures" / key / "catalog.json"
        if receipt.exists():
            return data, load_catalog(data, json.loads(receipt.read_text()))
        source = data / "captures" / key
        source.mkdir(parents=True, exist_ok=True)
        specs = {}
        for name, records in groups.items():
            frame = records_frame(records, (schemas or {}).get(name))
            path = source / f"{name}.parquet"
            # No receipt exists yet; recover safely from an interrupted capture.
            temporary = path.with_suffix(".tmp")
            frame.write_parquet(temporary, compression="zstd")
            temporary.replace(path)
            specs[name] = {
                "description": f"Captured league data: {name}.",
                "grain": "One record within the pinned capture.",
                "primary_key": [ROW],
                "kind": "parquet",
                "min_rows": 0,
                "source": path.relative_to(data).as_posix(),
            }
        registry = source / "registry.yaml"
        write_new(registry, yaml.safe_dump({"schema_version": 1, "tables": specs}).encode())
        result = refresh(data, registry_path=registry)
        ref = {"path": f"tables/catalogs/{result['catalog']}.json", "sha256": result["catalog"]}
        write_new(receipt, json_bytes(ref))
        return data, load_catalog(data, ref)


def query_records(data: Path, catalog: dict, name: str) -> list[dict]:
    if name not in catalog["tables"]:
        raise ValueError(f"Unknown league table: {name}")
    with connect(data, catalog=catalog) as db:
        return [
            json.loads(row[0])
            for row in db.execute(
                f'SELECT "{PAYLOAD}" FROM analytics."{name}" ORDER BY "{ROW}"'
            ).fetchall()
        ]


def query_snapshot(snapshot, data_dir: Path):
    from engine.espn.sync import (
        DraftPick,
        LeagueSnapshot,
        LineupEntry,
        PlayerState,
        ScheduleEntry,
        TeamState,
        TransactionState,
        WeekLineups,
    )

    raw = asdict(snapshot)
    lists = ("teams", "players", "transactions", "draft", "week_lineups")
    groups = {"league_" + name: raw[name] for name in lists}
    groups["league_snapshot"] = [{k: v for k, v in raw.items() if k not in lists}]
    groups["league_schedule"] = [
        {"team_id": team["team_id"], **game} for team in raw["teams"] for game in team["schedule"]
    ]
    groups["league_lineup_entries"] = [
        {"week": game["week"], "team_id": game[side + "_team_id"], **entry}
        for game in raw["week_lineups"]
        for side in ("home", "away")
        for entry in game[side + "_lineup"]
    ]
    directory = data_dir / "league" / str(snapshot.league_id) / str(snapshot.season)
    schemas = {
        "league_" + name: record_schema(cls)
        for name, cls in {
            "players": PlayerState,
            "teams": TeamState,
            "transactions": TransactionState,
            "draft": DraftPick,
            "week_lineups": WeekLineups,
        }.items()
    }
    schemas["league_schedule"] = {"team_id": pl.Int64, **record_schema(ScheduleEntry)}
    schemas["league_lineup_entries"] = {
        "week": pl.Int64,
        "team_id": pl.Int64,
        **record_schema(LineupEntry),
    }
    data, catalog = capture(directory, groups, schemas)
    # Use one pinned native connection for the complete observation snapshot.
    queried = {}
    with connect(data, catalog=catalog) as db:
        for name in ("snapshot", *lists):
            rows = db.execute(
                f'SELECT "{PAYLOAD}" FROM analytics."league_{name}" ORDER BY "{ROW}"'
            ).fetchall()
            queried[name] = [json.loads(row[0]) for row in rows]
    return LeagueSnapshot.from_dict({**queried.pop("snapshot")[0], **queried})


def query_documents(directory: Path, name: str, records: list[dict]) -> list[dict]:
    """Use the same capture/query path for pregame logs and reviewed league news."""
    data, catalog = capture(directory, {name: records})
    return query_records(data, catalog, name)
