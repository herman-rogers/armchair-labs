"""Observation tables straight from DuckDB: players, stats, rookies, and the catalog."""

from typing import Annotated

import duckdb
from fastapi import APIRouter, HTTPException, Query

from engine.tables import observations
from engine.tables.application import catalog_state, connect
from engine.tables.storage import load_table

router = APIRouter(prefix="/api/tables", tags=["tables"])


def _unavailable(exc: Exception) -> HTTPException:
    return HTTPException(409, f"Query tables unavailable: {exc}")


@router.get("/catalog")
def catalog():
    """The pinned table catalog: every table, its rows, and the observation cutoff."""
    try:
        state = catalog_state()
        tables = []
        for name, ref in sorted(state["catalog"]["tables"].items()):
            _, manifest = load_table(state["data"], name, ref)
            tables.append(
                {
                    "name": name,
                    "rows": manifest["rows"],
                    "built_at": manifest["built_at"],
                    "observation_cutoff": manifest.get("observation_cutoff") or {},
                    "description": manifest["recipe"]["description"],
                }
            )
        with connect() as db:
            cutoff = observations.observation_cutoff(db)
        from engine.tables.storage import content_hash

        return {
            "catalog_sha256": content_hash(state["catalog"]),
            "published_at": state["catalog"].get("published_at"),
            "observation_cutoff": cutoff,
            "tables": tables,
        }
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        raise _unavailable(exc) from exc


@router.get("/players")
def players(
    search: str = "",
    position: str = "ALL",
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    try:
        with connect() as db:
            frame = observations.directory(db, search=search, position=position)
            cutoff = observations.observation_cutoff(db)
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        raise _unavailable(exc) from exc
    return {
        "observation_cutoff": cutoff,
        "total": frame.height,
        "players": frame.slice(offset, limit).to_dicts(),
    }


@router.get("/players/{player_id}")
def player(player_id: str, season: int | None = None):
    try:
        with connect() as db:
            identity = observations.identity(db, player_id)
            if identity is None:
                raise HTTPException(404, f"Unknown player: {player_id}")
            return {
                "player": identity,
                "observation_cutoff": observations.observation_cutoff(db),
                "seasons": observations.seasons(db, player_id),
                "weeks": observations.weeks(db, player_id, season),
                "college": observations.college_links(db, player_id),
            }
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        raise _unavailable(exc) from exc


@router.get("/rookies")
def rookies(season: int | None = None):
    try:
        with connect() as db:
            cutoff = observations.observation_cutoff(db)
            frame = observations.rookies(db, season or cutoff["season"])
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        raise _unavailable(exc) from exc
    return {
        "season": season or cutoff["season"],
        "total": frame.height,
        "rookies": frame.to_dicts(),
    }


@router.get("/college/{college_id}")
def college(college_id: str):
    try:
        with connect() as db:
            rows = observations.college_seasons(db, college_id)
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        raise _unavailable(exc) from exc
    if not rows:
        raise HTTPException(404, f"Unknown college player: {college_id}")
    return {"college_id": college_id, "seasons": rows}
