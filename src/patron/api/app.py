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
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from patron.config.league import get_league
from patron.config.settings import get_settings

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


def board_path() -> Path:
    return get_settings().outputs_dir / "board.json"


def load_board() -> list[dict[str, Any]]:
    """Read the built board, or explain how to build it."""
    path = board_path()
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No board has been built yet. Run `patron board` to generate {path}, then reload."
            ),
        )
    return json.loads(path.read_text())


@app.get("/api/status")
def status() -> dict[str, Any]:
    """Whether a board exists, when it was built, and under what league settings."""
    path = board_path()
    config = get_league()
    built = path.exists()

    return {
        "board_available": built,
        "built_at": (
            datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat() if built else None
        ),
        "player_count": len(load_board()) if built else 0,
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


@app.get("/api/board")
def board(
    position: str | None = Query(None, description="Filter to one position."),
    flag: str | None = Query(None, description="Filter to rows carrying this flag."),
    search: str | None = Query(None, description="Case-insensitive name substring."),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """The ranked board, with optional filters."""
    rows = load_board()

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
        "offset": offset,
        "limit": limit,
        "players": rows[offset : offset + limit],
    }


@app.get("/api/players/{player_id}")
def player(player_id: str) -> dict[str, Any]:
    """One player's full metric line."""
    for row in load_board():
        if row.get("player_id") == player_id:
            return row
    raise HTTPException(status_code=404, detail=f"No player with id {player_id} on the board.")


@app.get("/api/positions")
def positions() -> dict[str, Any]:
    """Per-position counts and replacement level, for the frontend's filter chips."""
    rows = load_board()
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = summary.setdefault(
            row["position"],
            {"count": 0, "replacement_ppg": row.get("repl_ppg")},
        )
        entry["count"] += 1
    return {"positions": summary}
