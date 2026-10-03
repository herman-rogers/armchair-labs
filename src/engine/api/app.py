"""Verified NextGen analysis and explicitly scoped research archives.

Current measurements and forecasts come from the one published data/policy catalog.
Old boards, model reports and decision experiments require research scope. League
observations use a separately refreshed snapshot without selecting a legacy model.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware

from engine.api.college_routes import router as college_router
from engine.api.league_observation_routes import router as league_observation_router
from engine.api.league_routes import router as league_router
from engine.api.nextgen_routes import router as nextgen_router
from engine.api.profile_routes import router as profile_router
from engine.api.qb_passing_routes import router as qb_passing_router
from engine.api.ranking_routes import router as ranking_router
from engine.api.research_routes import router as research_router
from engine.api.team_analysis_routes import router as team_analysis_router
from engine.artifacts import artifact_status, verify_draft
from engine.config.league import get_league
from engine.config.settings import get_settings
from engine.data.releases import current_catalog
from engine.data.verification import VerificationChanged, verification_batch
from engine.observability import configure_logging

# Under uvicorn our loggers are otherwise silent, which hides ESPN fetches,
# join health, and stale-data warnings from the server log entirely.
configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    def warm():
        from engine.data.serving import generation, validate_sources

        data = get_settings().data_dir
        if generation(data) is not None:
            validate_sources(data, "profile-directory")

    try:
        await run_in_threadpool(warm)
    except (OSError, ValueError, KeyError, TypeError, HTTPException):
        logger.exception("Published release unavailable at startup; API will fail closed")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Armchair Labs Analytics Engine",
    description="Verified player measurements, outcome-specific forecasts and research archives.",
    version="0.1.0",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.include_router(ranking_router)
app.include_router(qb_passing_router)
app.include_router(team_analysis_router)


@app.middleware("http")
async def analysis_boundary(request, call_next):
    """Archived data is never a silent analysis default or fallback."""
    path = request.url.path
    if path.startswith("/api/"):
        try:
            current_catalog(get_settings().data_dir)
        except (OSError, ValueError, KeyError, TypeError):
            return JSONResponse(
                {"detail": "Current catalog is invalid; archive fallback disabled."},
                status_code=409,
            )
        archive = path.startswith(
            (
                "/api/research/",
                "/api/league",
                "/api/board",
                "/api/players",
                "/api/positions",
                "/api/metric-report",
            )
        )
        archive = archive and path != "/api/research/data-catalog"
        research = (
            request.headers.get("X-Analysis-Scope") == "research"
            or request.query_params.get("scope") == "research"
        )
        if path.startswith("/api/profiles") and not research:
            from engine.data.nextgen import load_analysis, read_json

            try:
                root, _ = await run_in_threadpool(load_analysis, get_settings().data_dir)
                if any(r["status"] == "open" for r in read_json(root / "incidents.json")):
                    raise ValueError("Profiles require revalidation after an open data incident")
            except (OSError, ValueError, KeyError, TypeError):
                return JSONResponse(
                    {"detail": "Verified analysis is required for player profiles."},
                    status_code=409,
                )
        if archive:
            if not research:
                return JSONResponse(
                    {
                        "detail": "Archived models/data are excluded from current analysis. "
                        "Use /api/nextgen, or scope=research to inspect the archive.",
                        "disposition": "archived_not_for_analysis",
                    },
                    status_code=409,
                )
            response = await call_next(request)
            response.headers["X-Analysis-Scope"] = "research"
            response.headers["X-Artifact-Disposition"] = "archived_not_for_analysis"
            return response
        if research:
            response = await call_next(request)
            response.headers["X-Analysis-Scope"] = "research"
            response.headers["X-Artifact-Disposition"] = "research_only"
            return response
    return await call_next(request)


@app.middleware("http")
async def pin_data_catalog(request, call_next):
    """Do not put a new release's response into an old frontend query cache."""
    requested = request.headers.get("X-Data-Catalog")
    if requested is None:
        return await call_next(request)

    def token():
        catalog = current_catalog(get_settings().data_dir)
        return f"{catalog['gold']['version']}@{catalog['published_at']}" if catalog else None

    def stale():
        return JSONResponse(
            {"detail": "The current data release changed. Refreshing the data catalog."},
            status_code=409,
            headers={"X-Data-Catalog-Stale": "true"},
        )

    try:
        if token() != requested:
            return stale()
        response = await call_next(request)
        return response if token() == requested else stale()
    except (OSError, ValueError, KeyError, TypeError):
        return JSONResponse(
            {"detail": "The canonical data catalog is unavailable."}, status_code=409
        )


# The React dev server runs on its own origin. Production serves the built frontend
# as static files from this app, where CORS does not apply.
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
async def application_queries(request, call_next):
    """Analysis/league data comes from DuckDB; explicit research keeps its readers."""
    import duckdb

    from engine.tables.application import query_scope

    path = request.url.path
    research = (
        request.headers.get("X-Analysis-Scope") == "research"
        or request.query_params.get("scope") == "research"
        or path.startswith("/api/research/")
    )
    application = path.startswith(("/api/nextgen", "/api/profiles")) or path == "/api/status"
    if research or not application:
        return await call_next(request)
    try:
        with query_scope(get_settings().data_dir):
            response = await call_next(request)
        response.headers["X-Data-Engine"] = "duckdb"
        return response
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        return JSONResponse(
            {"detail": f"Application query tables unavailable: {exc}"}, status_code=409
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Live ESPN state. Its own router so the board endpoints below stay independent
# of anything that can fail because a third-party API is down.
app.include_router(league_router)
app.include_router(research_router)
app.include_router(college_router)
app.include_router(profile_router)
app.include_router(nextgen_router)
app.include_router(league_observation_router)


MetricVersion = Literal["v1", "v2", "adaptive"]
METRIC_VERSIONS: tuple[MetricVersion, ...] = ("v1", "v2", "adaptive")


def board_path(version: MetricVersion = "v1") -> Path:
    """Artifact path for one metric generation, with the legacy v1 alias supported."""
    settings = get_settings()
    static = getattr(settings, "static_dir", None)
    if version == "v1" and static is not None:
        archive = static / f"draft_{get_league().draft_season}"
        if archive.exists():
            return verify_draft(archive)
    outputs = settings.outputs_dir
    explicit = outputs / f"board_{version}.json"
    if version == "v1" and not explicit.exists():
        return outputs / "board.json"
    return explicit


def load_board(version: MetricVersion = "v1") -> list[dict[str, Any]]:
    """Read the built board, or explain how to build it."""
    path = board_path(version)
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No {version} board has been built yet. Run `engine board` to generate "
                f"{path}, then reload."
            ),
        )
    return json.loads(path.read_text())


def metric_report_path() -> Path:
    outputs = get_settings().outputs_dir
    production = outputs / "production_report.json"
    research = outputs / "metric_report.json"
    available = [path for path in (production, research) if path.exists()]
    return max(available, key=lambda path: path.stat().st_mtime) if available else research


def load_metric_report() -> dict[str, Any]:
    """Read the last generated backtest report."""
    path = metric_report_path()
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No metric report has been built yet. Run `engine metric-report` to "
                f"generate {path}, then reload."
            ),
        )
    return json.loads(path.read_text())


@app.get("/api/status")
def status(scope: str = "analysis") -> dict[str, Any]:
    """Whether a board exists, when it was built, and under what league settings."""
    config = get_league()
    data = getattr(get_settings(), "data_dir", None)
    if data and scope != "research":
        from engine.api.nextgen_routes import release
        from engine.tables.application import read_json

        root, manifest = release()
        report = read_json(root / "report.json")
        return {
            "board_available": False,
            "built_at": manifest["generated_at"],
            "player_count": report["players"],
            "metric_versions": {},
            "metric_report": {"available": False, "built_at": None},
            "analysis": {"version": manifest["version"], "default_model": "baseline"},
            "league": {
                "name": config.name,
                "team_count": config.team_count,
                "board_season": config.board_season,
                "draft_season": config.draft_season,
                "seasons": config.seasons,
                "vor_baseline_rank": config.vor_baseline_rank,
            },
            "espn_connected": False,
            "phase": 1,
        }
    paths = {version: board_path(version) for version in METRIC_VERSIONS}
    available = {version: path.exists() for version, path in paths.items()}
    built = any(available.values())
    newest = max(
        (path for path in paths.values() if path.exists()),
        key=lambda path: path.stat().st_mtime,
        default=None,
    )
    boards = {
        version: load_board(version) if available[version] else [] for version in METRIC_VERSIONS
    }
    player_count = len(boards["v2"] if available["v2"] else boards["v1"])
    report_path = metric_report_path()
    return {
        "board_available": built,
        "built_at": (
            datetime.fromtimestamp(newest.stat().st_mtime, tz=UTC).isoformat() if newest else None
        ),
        "player_count": player_count,
        "metric_versions": {
            version: {
                "available": available[version],
                "provenance": artifact_status(paths[version]),
                "player_count": len(boards[version]),
                "built_at": (
                    datetime.fromtimestamp(paths[version].stat().st_mtime, tz=UTC).isoformat()
                    if available[version]
                    else None
                ),
                "forecast_as_of": sorted(
                    {
                        str(row["forecast_as_of"])
                        for row in boards[version]
                        if row.get("forecast_as_of")
                    }
                ),
            }
            for version in METRIC_VERSIONS
        },
        "metric_report": {
            "available": report_path.exists(),
            "built_at": (
                datetime.fromtimestamp(report_path.stat().st_mtime, tz=UTC).isoformat()
                if report_path.exists()
                else None
            ),
        },
        "league": {
            "name": config.name,
            "team_count": config.team_count,
            "board_season": config.board_season,
            "draft_season": config.draft_season,
            "seasons": config.seasons,
            "vor_baseline_rank": config.vor_baseline_rank,
        },
        # Phase 2 lights these up; the frontend uses them to show what is not built yet
        # rather than pretending the features are missing.
        "espn_connected": False,
        "phase": 1,
    }


@app.get("/api/metric-report")
def metric_report() -> dict[str, Any]:
    """Latest configurable v2 metric catalog and rolling backtest results."""
    return load_metric_report()


@app.get("/api/board")
def board(
    version: Annotated[MetricVersion, Query(description="Metric generation to return.")] = "v1",
    position: str | None = Query(None, description="Filter to one position."),
    flag: str | None = Query(None, description="Filter to rows carrying this flag."),
    search: str | None = Query(None, description="Case-insensitive name substring."),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """The ranked board, with optional filters."""
    rows = load_board(version)

    if position:
        wanted = position.upper()
        rows = [row for row in rows if row["position"] == wanted]
    if flag:
        rows = [row for row in rows if flag.lower() in (row.get("flags") or "").lower()]
    if search:
        needle = search.lower()
        rows = [row for row in rows if needle in row["player_display_name"].lower()]

    return {
        "total": len(rows),
        "version": version,
        "offset": offset,
        "limit": limit,
        "players": rows[offset : offset + limit],
    }


@app.get("/api/players/{player_id}")
def player(
    player_id: str,
    version: Annotated[MetricVersion, Query(description="Metric generation to search.")] = "v1",
) -> dict[str, Any]:
    """One player's full metric line."""
    for row in load_board(version):
        if row.get("player_id") == player_id:
            return row
    raise HTTPException(status_code=404, detail=f"No player with id {player_id} on the board.")


@app.get("/api/positions")
def positions(
    version: Annotated[MetricVersion, Query(description="Metric generation to summarize.")] = "v1",
) -> dict[str, Any]:
    """Per-position counts and replacement level, for the frontend's filter chips."""
    rows = load_board(version)
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = summary.setdefault(
            row["position"],
            {
                "count": 0,
                "replacement_ppg": (
                    row.get("proj_repl_ppg") if version != "v1" else row.get("repl_ppg")
                ),
            },
        )
        entry["count"] += 1
    return {"version": version, "positions": summary}
