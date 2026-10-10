"""Armchair Labs API: observation tables, ChairStats campaign models, and ESPN league state.

Every model is available at all times; a request names the campaign it wants.
Observations come from the pinned DuckDB table catalog. There is no selected or
promoted product anywhere in this service.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware

from engine.api.chairstats_routes import router as chairstats_router
from engine.api.league_routes import router as league_router
from engine.api.tables_routes import router as tables_router
from engine.api.team_analysis_routes import router as team_analysis_router
from engine.config.league import get_league
from engine.config.settings import get_settings
from engine.data.verification import VerificationChanged, verification_batch
from engine.observability import configure_logging

# Under uvicorn our loggers are otherwise silent, which hides ESPN fetches,
# join health, and stale-data warnings from the server log entirely.
configure_logging()

logger = logging.getLogger(__name__)

# Routes that read the DuckDB catalog run inside one pinned query scope.
TABLE_ROUTES = ("/api/tables", "/api/team-analysis", "/api/league", "/api/status")


def catalog_token(data_dir) -> str | None:
    """Identity of the published table catalog, used to key frontend caches."""
    from engine.tables.storage import content_hash, current

    catalog = current(data_dir)
    return content_hash(catalog) if catalog.get("tables") else None


@asynccontextmanager
async def lifespan(app):
    def warm():
        from engine.tables.query import materialize
        from engine.tables.storage import current
        from engine.tables.team_analysis import team_session

        data = get_settings().data_dir
        catalog = current(data)
        if catalog["tables"]:
            materialize(data, catalog)
        if "team_player_games" in catalog["tables"]:
            with team_session(data):
                pass

    try:
        await run_in_threadpool(warm)
    except (OSError, ValueError, KeyError, TypeError, HTTPException, duckdb.Error):
        logger.exception("Published tables unavailable at startup; table routes will fail closed")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Armchair Labs Analytics Engine",
    description="Observation tables, ChairStats campaign forecasts and league observations.",
    version="0.2.0",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.include_router(chairstats_router)
app.include_router(tables_router)
app.include_router(team_analysis_router)
app.include_router(league_router)


@app.middleware("http")
async def pin_table_catalog(request, call_next):
    """Do not put a new catalog's response into an old frontend query cache."""
    requested = request.headers.get("X-Data-Catalog")
    if requested is None:
        return await call_next(request)

    def stale():
        return JSONResponse(
            {"detail": "The published table catalog changed. Refreshing the data catalog."},
            status_code=409,
            headers={"X-Data-Catalog-Stale": "true"},
        )

    try:
        data = get_settings().data_dir
        if catalog_token(data) != requested:
            return stale()
        response = await call_next(request)
        return response if catalog_token(data) == requested else stale()
    except (OSError, ValueError, KeyError, TypeError):
        return JSONResponse({"detail": "The table catalog is unavailable."}, status_code=409)


@app.middleware("http")
async def verified_request(request, call_next):
    try:
        with verification_batch():
            return await call_next(request)
    except VerificationChanged:
        return JSONResponse(
            {"detail": "Release dependencies changed during request. Refresh the catalog."},
            status_code=409,
        )


@app.middleware("http")
async def table_queries(request, call_next):
    """Observation reads come from one pinned DuckDB catalog per request."""
    from engine.tables.application import query_scope

    if not request.url.path.startswith(TABLE_ROUTES):
        return await call_next(request)
    try:
        with query_scope(get_settings().data_dir):
            response = await call_next(request)
        response.headers["X-Data-Engine"] = "duckdb"
        return response
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        return JSONResponse({"detail": f"Query tables unavailable: {exc}"}, status_code=409)


# The React dev server runs on its own origin. Production serves the built frontend
# as static files from this app, where CORS does not apply.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status() -> dict[str, Any]:
    """What is published: the table catalog, the campaigns, and the league settings."""
    from engine.api.chairstats_routes import store
    from engine.chairstats import views
    from engine.tables.application import catalog_state, connect
    from engine.tables.observations import observation_cutoff

    config = get_league()
    tables: dict[str, Any] = {"available": False}
    try:
        state = catalog_state()
        with connect() as db:
            cutoff = observation_cutoff(db)
        tables = {
            "available": True,
            "catalog_sha256": catalog_token(state["data"]),
            "published_at": state["catalog"].get("published_at"),
            "table_count": len(state["catalog"]["tables"]),
            "observation_cutoff": cutoff,
        }
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        tables["issue"] = str(exc)
    campaigns: list[dict] = []
    issue = None
    try:
        campaigns = [views.summary(m) for m in store().campaigns()]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issue = str(exc)
    return {
        "tables": tables,
        "campaigns": {
            "available": bool(campaigns),
            "count": len(campaigns),
            "campaigns": campaigns,
            "issue": issue,
        },
        "league": {
            "name": config.name,
            "team_count": config.team_count,
            "board_season": config.board_season,
            "draft_season": config.draft_season,
            "seasons": config.seasons,
        },
    }
