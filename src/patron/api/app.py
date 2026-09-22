"""Read-only HTTP API over the built board.

Phase 1 serves what the pipeline has already produced rather than computing on
request. A board build pulls several seasons of play-by-play; that belongs on a
schedule, not on an HTTP handler. The API reads `data/outputs/board.json` and reports
plainly when it is missing, which is the state before the first `patron board` run.

Phase 3 replaces the file with a live store and adds the ESPN-derived endpoints
(the ranked wire, roster health, the alerts feed).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from patron.api.league_routes import router as league_router
from patron.config.league import get_league
from patron.config.settings import get_settings
from patron.observability import configure_logging

# Under uvicorn our loggers are otherwise silent, which hides ESPN fetches,
# join health, and stale-data warnings from the server log entirely.
configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Patron Saints Analytics Engine",
    description="League-exact fantasy valuation for Sweaty Plays.",
    version="0.1.0",
)

# The React dev server runs on its own origin. Production serves the built frontend
# as static files from this app, where CORS does not apply.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Live ESPN state. Its own router so the board endpoints below stay independent
# of anything that can fail because a third-party API is down.
app.include_router(league_router)


MetricVersion = Literal["v1", "v2", "adaptive"]
METRIC_VERSIONS: tuple[MetricVersion, ...] = ("v1", "v2", "adaptive")


def board_path(version: MetricVersion = "v1") -> Path:
    """Artifact path for one metric generation, with the legacy v1 alias supported."""
    outputs = get_settings().outputs_dir
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
                f"No {version} board has been built yet. Run `patron board` to generate "
                f"{path}, then reload."
            ),
        )
    return json.loads(path.read_text())


def metric_report_path() -> Path:
    return get_settings().outputs_dir / "metric_report.json"


def load_metric_report() -> dict[str, Any]:
    """Read the last generated backtest report."""
    path = metric_report_path()
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No metric report has been built yet. Run `patron metric-report` to "
                f"generate {path}, then reload."
            ),
        )
    return json.loads(path.read_text())


@app.get("/api/status")
def status() -> dict[str, Any]:
    """Whether a board exists, when it was built, and under what league settings."""
    paths = {version: board_path(version) for version in METRIC_VERSIONS}
    config = get_league()
    available = {version: path.exists() for version, path in paths.items()}
    built = any(available.values())
    newest = max(
        (path for path in paths.values() if path.exists()),
        key=lambda path: path.stat().st_mtime,
        default=None,
    )
    player_count = (
        len(load_board("v2"))
        if available["v2"]
        else (len(load_board("v1")) if available["v1"] else 0)
    )

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
                "player_count": len(load_board(version)) if available[version] else 0,
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
