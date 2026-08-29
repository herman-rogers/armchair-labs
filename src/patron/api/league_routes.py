"""League endpoints — everything that needs live ESPN state.

Kept in its own router so `app.py` only ever registers it, and so the board endpoints
(pure metrics, no network) stay independent of anything that can fail because a
third-party API is down.

Every handler is a plain `def`, not `async def`. The ESPN client is blocking, and
FastAPI runs sync handlers in a threadpool — which is what we want here. Declaring
them async would block the event loop for the length of an ESPN fetch and stall every
other request in the process.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Query

from patron.config.league import get_league
from patron.espn import crosswalk, reports
from patron.espn.crosswalk import (
    AVAILABILITY,
    IS_FREE_AGENT,
    IS_MINE,
    OWNER_TEAM_ID,
    OWNER_TEAM_NAME,
)
from patron.espn.service import LeagueService, LeagueState, NotAuthenticatedError, get_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/league", tags=["league"])

ServiceDep = Annotated[LeagueService, Depends(get_service)]


def _rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """DataFrame to JSON-safe records.

    Polars yields NaN and infinity for undefined float results; JSON has no way to
    spell either, and `json.dumps` emits bare `NaN`, which is invalid JSON that
    browsers reject. They become null here rather than at every call site.
    """
    records = frame.to_dicts()
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                record[key] = None
    return records


def _state(service: LeagueService, version: str, force: bool = False) -> LeagueState:
    try:
        return service.get(version=version, force=force)
    except NotAuthenticatedError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001 - unofficial upstream
        logger.exception("league state unavailable")
        raise HTTPException(status_code=502, detail=f"Could not reach ESPN: {error}") from error


def _envelope(state: LeagueState) -> dict[str, Any]:
    """Freshness metadata attached to every league response.

    The UI has to be able to say "this is nine minutes old" or "ESPN is down, this is
    what we last saw". Data without its age is how a stale wire gets acted on.
    """
    return {
        "age_seconds": round(state.age_seconds),
        "stale": state.stale,
        "week": state.snapshot.week,
        "season": state.snapshot.season,
    }


@router.get("/status")
def league_status(service: ServiceDep) -> dict[str, Any]:
    """Freshness, authentication, and join health. Never fetches."""
    return service.status()


@router.post("/refresh")
def refresh(service: ServiceDep, version: str = Query("v1")) -> dict[str, Any]:
    """Force a pull from ESPN, ignoring the TTL."""
    _state(service, version, force=True)
    return service.status()


@router.get("")
def league(service: ServiceDep, version: str = Query("v1")) -> dict[str, Any]:
    """Teams, standings, and who I am."""
    state = _state(service, version)
    snapshot = state.snapshot

    return {
        **_envelope(state),
        "league_id": snapshot.league_id,
        "league_name": snapshot.league_name,
        "my_team_id": snapshot.my_team_id,
        "teams": [
            {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "owner": team.owner,
                "wins": team.wins,
                "losses": team.losses,
                "faab_remaining": team.faab_remaining,
                "is_mine": team.team_id == snapshot.my_team_id,
            }
            for team in snapshot.teams
        ],
    }


@router.get("/players")
def players(
    service: ServiceDep,
    version: str = Query("v1", description="Metric generation."),
    position: str | None = Query(None),
    availability: str | None = Query(None, description="rostered | free_agent | unknown"),
    owner: int | None = Query(None, description="Filter to one team id."),
    search: str | None = Query(None),
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """The full board, cross-referenced with who owns whom.

    This is the players view: every ranked player, with metrics and league status on
    the same row, so "who is the best available tight end" and "who is worth more than
    what I am starting" are the same query.
    """
    state = _state(service, version)
    frame = state.tagged_board

    if position:
        frame = frame.filter(pl.col("position") == position.upper())
    if availability:
        frame = frame.filter(pl.col(AVAILABILITY) == availability)
    if owner is not None:
        frame = frame.filter(pl.col(OWNER_TEAM_ID) == owner)
    if search:
        frame = frame.filter(
            pl.col("player_display_name").str.to_lowercase().str.contains(search.lower())
        )

    total = frame.height
    page = frame.slice(offset, limit)
    return {
        **_envelope(state),
        "total": total,
        "offset": offset,
        "limit": limit,
        "players": _rows(page),
    }


@router.get("/roster/{team_id}")
def roster(service: ServiceDep, team_id: int, version: str = Query("v1")) -> dict[str, Any]:
    """One team's roster, board-joined, decisions first."""
    state = _state(service, version)
    frame = state.tagged_board.filter(pl.col(OWNER_TEAM_ID) == team_id)
    if frame.height == 0:
        raise HTTPException(status_code=404, detail=f"No roster for team {team_id}.")

    health = reports.roster_health(frame.with_columns(pl.lit(True).alias(IS_MINE)))
    team_name = frame[OWNER_TEAM_NAME][0]
    return {
        **_envelope(state),
        "team_id": team_id,
        "team_name": team_name,
        "players": _rows(health),
    }


@router.get("/wire")
def wire(
    service: ServiceDep,
    version: str = Query("v1"),
    position: str | None = Query(None),
    min_vor: float | None = Query(None),
    healthy_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """Free agents ranked against what else is actually free."""
    state = _state(service, version)
    config = get_league()

    frame = reports.ranked_wire(
        state.tagged_board,
        config.vor_baseline_rank,
        min_vor=min_vor,
        limit=None,
        healthy_only=healthy_only,
    )
    if position:
        frame = frame.filter(pl.col("position") == position.upper())

    levels = reports.wire_replacement_levels(state.tagged_board, config.vor_baseline_rank)

    return {
        **_envelope(state),
        # Surfaced so the UI can explain why the numbers differ from the draft board:
        # replacement level is measured against the pool that is actually free.
        "replacement_levels": {k: round(v, 2) for k, v in levels.items()},
        "total": frame.height,
        "players": _rows(frame.head(limit)),
    }


@router.get("/unrankable")
def unrankable(service: ServiceDep, version: str = Query("v1")) -> dict[str, Any]:
    """ESPN players the board cannot price — rookies, and anyone with no prior tape.

    Its own endpoint because it is a real answer, not an error. A wire that silently
    omits a hyped rookie is worse than one that says it cannot price him.
    """
    state = _state(service, version)
    board_ids, board_names = crosswalk.board_lookups(state.tagged_board)
    resolved, _ = crosswalk.resolve_player_ids(state.snapshot.to_frame(), board_ids, board_names)
    listing = reports.unrankable_players(resolved)
    return {**_envelope(state), "total": listing.height, "players": _rows(listing)}


@router.get("/opponents")
def opponents(service: ServiceDep, version: str = Query("v1")) -> dict[str, Any]:
    """Every team's positional depth — where a trade or a blocking claim lands."""
    state = _state(service, version)
    config = get_league()
    weaknesses = reports.opponent_weaknesses(state.tagged_board, config.vor_baseline_rank)
    return {**_envelope(state), "teams": _rows(weaknesses)}


@router.get("/transactions")
def transactions(service: ServiceDep, limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    """Recent adds, drops, and waiver claims — the league's revealed price signal."""
    state = _state(service, "v1")
    entries = [
        {
            "date": entry.date,
            "kind": entry.kind,
            "team_name": entry.team_name,
            "player_name": entry.player_name,
            "bid_amount": entry.bid_amount,
        }
        for entry in state.snapshot.transactions[:limit]
    ]
    return {**_envelope(state), "total": len(state.snapshot.transactions), "transactions": entries}


@router.get("/free-agent-counts")
def free_agent_counts(service: ServiceDep, version: str = Query("v1")) -> dict[str, Any]:
    """How many claimable players exist at each position, for the UI's filter chips."""
    state = _state(service, version)
    counts = (
        state.tagged_board.filter(pl.col(IS_FREE_AGENT))
        .group_by("position")
        .agg(pl.len().alias("count"))
    )
    return {**_envelope(state), "counts": {row[0]: row[1] for row in counts.iter_rows()}}
