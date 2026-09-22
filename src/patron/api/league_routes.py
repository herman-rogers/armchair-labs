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
from patron.espn import lineup, reports, sync
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
    config = get_league()
    strengths = reports.team_strengths(
        state.tagged_board,
        snapshot.to_frame(),
        (team.team_id for team in snapshot.teams),
        season_games=config.metrics.projection_season_games,
        fallback_availability=config.metrics.projection_availability_prior,
        columns=reports.RosterProjectionColumns(
            mean=tuple(config.metrics.roster_mean_columns),
            games=tuple(config.metrics.roster_games_columns),
            availability=tuple(config.metrics.roster_availability_columns),
            volatility=tuple(config.metrics.roster_volatility_columns),
        ),
    )

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
                "division_id": team.division_id,
                "division_name": team.division_name,
                **strengths.get(team.team_id, {}),
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

    frame = reports.wire_lineup_improvements(frame, state.tagged_board, state.snapshot.roster_slots)
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
    listing = reports.unrankable_players(state.tagged_board)
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


@router.get("/draft")
def draft(service: ServiceDep, version: str = Query("v2")) -> dict[str, Any]:
    """The draft recap scored against the board: every pick, exact best-available, grades."""
    state = _state(service, version)
    snapshot = state.snapshot
    analysis = reports.draft_analysis(
        state.tagged_board, snapshot.draft, my_team_id=snapshot.my_team_id
    )
    return {**_envelope(state), **analysis, "pick_count": len(snapshot.draft)}


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


def _projected_row(slot: lineup.LineupSlot) -> dict[str, Any]:
    """One modelled slot, in the same shape an actual week's row uses.

    The two paths share a renderer, so they have to share a shape. The fields that
    only a played week can fill — what ESPN projected, the pro opponent — are null
    here rather than absent, which is the difference between the UI showing a blank
    and the UI showing `undefined`.
    """
    return {
        "slot": slot.slot,
        "player_id": slot.player_id,
        "espn_id": None,
        "player_display_name": slot.player_name,
        "position": slot.position,
        "value": round(slot.value, 2),
        "projected": None,
        "opponent": None,
        "on_bye": False,
    }


def _strength(state: LeagueState, team_id: int, team_name: str) -> dict[str, Any]:
    """One team's best fieldable lineup, as JSON."""
    roster = state.tagged_board.filter(pl.col(OWNER_TEAM_ID) == team_id)
    column = lineup.value_column_for(state.tagged_board)

    # ESPN-only rows are present so managers can see them, but their model value stays
    # null. Preserve that uncertainty in lineup comparisons rather than treating an
    # included fallback row as a scored player.
    unranked = roster.filter(pl.col(column).is_null()).height
    built = lineup.best_lineup(
        roster,
        state.snapshot.roster_slots,
        column,
        team_id,
        team_name,
        unranked_count=unranked,
    )

    return {
        "team_id": team_id,
        "team_name": team_name,
        # A model answer to "who has the better team", not a record of anything that
        # happened. The UI must say which of the two it is showing.
        "source": "projected",
        "metric": column,
        "total": round(built.total, 2),
        "projected_total": None,
        "by_position": {k: round(v, 2) for k, v in built.by_position.items()},
        "unranked_starters": built.unranked_starters,
        "starters": [_projected_row(slot) for slot in built.starters],
        "bench": [_projected_row(slot) for slot in built.bench],
    }


#: The order a manager reads their own lineup in. ESPN returns roster entries in
#: whatever order it stores them, which interleaves bench players with starters.
_SLOT_PRIORITY = ("QB", "RB", "WR", "TE")


def _slot_sort_key(slot: str) -> tuple[int, int, str]:
    """Dedicated skill slots, then flex, then kicker and defense."""
    if lineup.FLEX_SEPARATOR in slot:
        return (1, 0, slot)
    if slot in _SLOT_PRIORITY:
        return (0, _SLOT_PRIORITY.index(slot), slot)
    return (2, 0, slot)


def _lineup_row(entry: sync.LineupEntry) -> dict[str, Any]:
    return {
        "slot": entry.slot,
        # Board ids do not reach here: this is ESPN's own record of the week, and
        # joining it to the board would reintroduce exactly the gap being fixed.
        "player_id": None,
        "espn_id": entry.espn_id,
        "player_display_name": entry.player_display_name,
        "position": entry.position,
        "value": round(entry.points, 2),
        "projected": round(entry.projected_points, 2)
        if entry.projected_points is not None
        else None,
        "opponent": entry.pro_opponent,
        "on_bye": entry.on_bye,
    }


def _actual_side(week: sync.WeekLineups, team_id: int, team_name: str) -> dict[str, Any]:
    """One team's week as they actually played it.

    Deliberately not the best lineup available: the point of a played week is the
    lineup that was set, bench mistakes and all. Starters keep ESPN's slot; everyone
    else is bench, ordered by what they scored, because the question a manager asks of
    a finished week is what they left there.
    """
    entries = week.lineup_for(team_id)
    starters = sorted(
        (entry for entry in entries if entry.started),
        key=lambda entry: (_slot_sort_key(entry.slot), -entry.points),
    )
    bench = sorted(
        (entry for entry in entries if not entry.started),
        key=lambda entry: -entry.points,
    )

    by_position: dict[str, float] = {}
    for entry in starters:
        if entry.position:
            by_position[entry.position] = by_position.get(entry.position, 0.0) + entry.points

    is_home = team_id == week.home_team_id
    projected = week.home_projected if is_home else week.away_projected
    return {
        "team_id": team_id,
        "team_name": team_name,
        "source": "actual",
        # Fantasy points, as scored. Named so the UI labels it as points rather than
        # reaching for a board metric's name.
        "metric": "points",
        "total": round(week.home_score if is_home else week.away_score, 2),
        "projected_total": round(projected, 2) if projected is not None else None,
        "by_position": {k: round(v, 2) for k, v in by_position.items()},
        # Every player here has a real score; nothing is understated for want of a
        # board value, so the projected view's caveat does not apply.
        "unranked_starters": 0,
        "starters": [_lineup_row(entry) for entry in starters],
        "bench": [_lineup_row(entry) for entry in bench],
    }


def _team_names(state: LeagueState) -> dict[int, str]:
    return {team.team_id: team.team_name for team in state.snapshot.teams}


@router.get("/schedule")
def schedule(service: ServiceDep, version: str = Query("v1")) -> dict[str, Any]:
    """Every team's full season schedule, with results where they exist."""
    state = _state(service, version)
    names = _team_names(state)

    return {
        **_envelope(state),
        "regular_season_weeks": state.snapshot.regular_season_weeks,
        "my_team_id": state.snapshot.my_team_id,
        "teams": [
            {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "schedule": [
                    {
                        "week": entry.week,
                        "opponent_team_id": entry.opponent_team_id,
                        "opponent_team_name": names.get(entry.opponent_team_id),
                        "score": entry.score,
                        "outcome": entry.outcome,
                        "played": entry.played,
                    }
                    for entry in team.schedule
                ],
            }
            for team in state.snapshot.teams
        ],
    }


@router.get("/matchups")
def matchups(
    service: ServiceDep,
    week: int | None = Query(None, ge=1, le=20, description="Defaults to the current week."),
    version: str = Query("v1"),
) -> dict[str, Any]:
    """Every matchup in a week, each side's lineup shown as starters and bench.

    Two different questions share this endpoint, and the response says which one it
    answered:

    `source: "actual"` — a week that has been played or is being played. ESPN's own
    record of the lineup each manager set: real slots, real bench, real points. A
    player benched that week appears on the bench, however highly the board rates him.

    `source: "projected"` — a week that has not happened. There is no lineup to report,
    so this falls back to each team's best fieldable lineup by board value. That is a
    season-long strength comparison, not a weekly forecast: it knows nothing about
    byes, the injury report, or opponent.
    """
    state = _state(service, version)
    target = week or state.snapshot.week
    names = _team_names(state)
    mine = state.snapshot.my_team_id

    played = [entry for entry in state.snapshot.week_lineups if entry.week == target]
    if played:
        # The week's own box scores are authoritative for who played whom — the
        # schedule is the plan, this is the record.
        played.sort(key=lambda entry: (mine not in (entry.home_team_id, entry.away_team_id),))
        games: list[dict[str, Any]] = []
        for entry in played:
            home = _actual_side(entry, entry.home_team_id, names.get(entry.home_team_id, ""))
            away = _actual_side(entry, entry.away_team_id, names.get(entry.away_team_id, ""))
            games.append(
                {
                    "home": home,
                    "away": away,
                    "involves_me": mine in (entry.home_team_id, entry.away_team_id),
                    "margin": round(home["total"] - away["total"], 2),
                }
            )
        return {
            **_envelope(state),
            "requested_week": target,
            "current_week": state.snapshot.week,
            "regular_season_weeks": state.snapshot.regular_season_weeks,
            "my_team_id": mine,
            "source": "actual",
            "matchups": games,
        }

    # A future week: no lineup exists yet, so compare fieldable strength instead.
    #
    # A matchup appears on both teams' schedules; keep one copy, ordered so the
    # viewer's own game is first.
    seen: set[frozenset[int]] = set()
    pairs: list[tuple[int, int]] = []
    for team in state.snapshot.teams:
        scheduled = next((e for e in team.schedule if e.week == target), None)
        if scheduled is None:
            continue
        key = frozenset({team.team_id, scheduled.opponent_team_id})
        if key in seen:
            continue
        seen.add(key)
        pairs.append((team.team_id, scheduled.opponent_team_id))

    pairs.sort(key=lambda pair: (mine not in pair, pair[0]))

    strengths: dict[int, dict[str, Any]] = {}
    for team_id in {team_id for pair in pairs for team_id in pair}:
        strengths[team_id] = _strength(state, team_id, names.get(team_id, str(team_id)))

    return {
        **_envelope(state),
        "requested_week": target,
        "current_week": state.snapshot.week,
        "regular_season_weeks": state.snapshot.regular_season_weeks,
        "my_team_id": mine,
        "source": "projected",
        "matchups": [
            {
                "home": strengths[home],
                "away": strengths[away],
                "involves_me": mine in (home, away),
                "margin": round(strengths[home]["total"] - strengths[away]["total"], 2),
            }
            for home, away in pairs
        ],
    }


@router.get("/compare")
def compare(
    service: ServiceDep,
    left: int = Query(..., description="Team id on the left."),
    right: int = Query(..., description="Team id on the right."),
    version: str = Query("v1"),
) -> dict[str, Any]:
    """Two teams side by side, by best fieldable lineup.

    The same shape as one matchup, but for any pair — which is what a trade
    conversation needs and a schedule does not provide.
    """
    state = _state(service, version)
    names = _team_names(state)
    for team_id in (left, right):
        if team_id not in names:
            raise HTTPException(status_code=404, detail=f"No team {team_id} in this league.")

    left_strength = _strength(state, left, names[left])
    right_strength = _strength(state, right, names[right])
    return {
        **_envelope(state),
        "left": left_strength,
        "right": right_strength,
        "margin": round(left_strength["total"] - right_strength["total"], 2),
    }
