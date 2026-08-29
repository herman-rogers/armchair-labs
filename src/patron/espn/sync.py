"""Pulling live league state from ESPN.

One snapshot per poll: every roster, the free-agent pool, this week's matchups, and the
recent transaction log. Snapshots are written to disk so successive polls can be diffed
— an add, a drop, a new injury tag — which is what Phase 3's alert tripwires will fire
on.

ESPN's fantasy API is unofficial, undocumented, and free. The posture is a quiet
houseguest: one request cycle per poll, everything needed taken in that cycle, and no
retry storm when something fails.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from patron.espn import teams
from patron.espn.credentials import EspnCredentials

logger = logging.getLogger(__name__)

#: Positions the board ranks. Kickers and defenses are tiered separately.
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

#: How deep to read the free-agent pool. The wire below this is noise, and every extra
#: page is another request against a community-run service.
DEFAULT_FREE_AGENT_DEPTH = 300


@dataclass
class PlayerState:
    """What ESPN currently believes about one player."""

    espn_id: int
    player_display_name: str
    position: str
    espn_team: str | None
    injury_status: str | None
    percent_owned: float | None
    percent_started: float | None
    projected_points: float | None
    owner_team_id: int | None
    owner_team_name: str | None
    lineup_slot: str | None


@dataclass
class TeamState:
    team_id: int
    team_name: str
    owner: str | None
    wins: int
    losses: int
    faab_remaining: int | None


@dataclass
class TransactionState:
    """One entry from the transaction log — the league's revealed price signal."""

    date: str | None
    kind: str | None
    team_name: str | None
    player_name: str | None
    bid_amount: int | None


@dataclass
class LeagueSnapshot:
    """Everything one poll captured."""

    captured_at: str
    league_id: int
    league_name: str
    season: int
    week: int
    my_team_id: int | None
    teams: list[TeamState] = field(default_factory=list)
    players: list[PlayerState] = field(default_factory=list)
    transactions: list[TransactionState] = field(default_factory=list)

    def to_frame(self) -> pl.DataFrame:
        """Player states as a frame, ready to join against the board."""
        schema = {
            "espn_id": pl.Int64,
            "player_display_name": pl.String,
            "position": pl.String,
            "espn_team": pl.String,
            "injury_status": pl.String,
            "percent_owned": pl.Float64,
            "percent_started": pl.Float64,
            "projected_points": pl.Float64,
            "owner_team_id": pl.Int64,
            "owner_team_name": pl.String,
            "lineup_slot": pl.String,
        }
        # An explicit schema, not inference: a column that happens to be entirely null
        # in one snapshot would otherwise come back as Null dtype and fail to join.
        return pl.DataFrame([asdict(player) for player in self.players], schema=schema)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path

    @classmethod
    def read(cls, path: Path) -> LeagueSnapshot:
        raw = json.loads(path.read_text())
        return cls(
            **{
                **raw,
                "teams": [TeamState(**team) for team in raw.get("teams", [])],
                "players": [PlayerState(**player) for player in raw.get("players", [])],
                "transactions": [
                    TransactionState(**entry) for entry in raw.get("transactions", [])
                ],
            }
        )


def _attribute(source: Any, *names: str, default: Any = None) -> Any:
    """First present attribute among `names`.

    espn-api's model fields drift between versions and across player types; reading
    defensively here is cheaper than a hard failure mid-poll.
    """
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return value
    return default


def _text(value: Any) -> str | None:
    """A scalar string, or None.

    espn-api's field types are not uniform across player kinds — a defense's
    `injuryStatus` comes back as an empty list rather than a string or None, which is
    enough to make an otherwise well-typed column build fail. Coercing at the boundary
    keeps that irregularity out of everything downstream.
    """
    if value is None or isinstance(value, list | dict | tuple):
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any) -> float | None:
    """A float, or None. Keeps int/float columns from being inferred inconsistently."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _player_state(player: Any, team: Any = None) -> PlayerState:
    return PlayerState(
        espn_id=int(_attribute(player, "playerId", default=0) or 0),
        player_display_name=str(_attribute(player, "name", default="") or ""),
        position=str(_attribute(player, "position", default="") or ""),
        # ESPN writes the literal string "None" in proTeam for an unsigned player.
        # Left alone it renders as a team called "None" and, worse, compares unequal to
        # every real team code.
        espn_team=teams.to_nflverse(_text(_attribute(player, "proTeam"))),
        injury_status=_text(_attribute(player, "injuryStatus")),
        percent_owned=_number(_attribute(player, "percent_owned")),
        percent_started=_number(_attribute(player, "percent_started")),
        projected_points=_number(_attribute(player, "projected_total_points", "projected_points")),
        owner_team_id=int(team.team_id) if team is not None else None,
        owner_team_name=_text(getattr(team, "team_name", None)) if team is not None else None,
        lineup_slot=_text(_attribute(player, "lineupSlot")),
    )


def fetch_snapshot(
    credentials: EspnCredentials,
    season: int,
    free_agent_depth: int = DEFAULT_FREE_AGENT_DEPTH,
    transaction_size: int = 100,
) -> LeagueSnapshot:
    """Pull one complete picture of the league.

    Args:
        credentials: Cookies plus the league id.
        season: Season to read.
        free_agent_depth: How many free agents to pull.
        transaction_size: How many recent transactions to read.

    Raises:
        ValueError: if no league id is configured.
    """
    from espn_api.football import League

    if credentials.league_id is None:
        raise ValueError(
            "No ESPN league id configured. Run `patron auth login`, or set ESPN_LEAGUE_ID in .env."
        )

    league = League(
        league_id=credentials.league_id,
        year=season,
        espn_s2=credentials.espn_s2,
        swid=credentials.swid,
    )

    teams = [
        TeamState(
            team_id=int(team.team_id),
            team_name=str(team.team_name),
            owner=_owner_name(team),
            wins=int(_attribute(team, "wins", default=0)),
            losses=int(_attribute(team, "losses", default=0)),
            faab_remaining=_remaining_faab(league, team),
        )
        for team in league.teams
    ]

    players: list[PlayerState] = []
    for team in league.teams:
        players.extend(_player_state(player, team) for player in team.roster)

    free_agents = league.free_agents(size=free_agent_depth)
    players.extend(_player_state(player) for player in free_agents)

    transactions = _read_transactions(league, transaction_size)

    snapshot = LeagueSnapshot(
        captured_at=datetime.now(tz=UTC).isoformat(),
        league_id=int(credentials.league_id),
        league_name=str(league.settings.name),
        season=season,
        week=int(league.current_week),
        my_team_id=credentials.team_id,
        teams=teams,
        players=players,
        transactions=transactions,
    )
    logger.info(
        "snapshot: %s teams, %s rostered + %s free agents, %s transactions",
        len(teams),
        len(players) - len(free_agents),
        len(free_agents),
        len(transactions),
    )
    return snapshot


def _owner_name(team: Any) -> str | None:
    owners = _attribute(team, "owners", default=None)
    if isinstance(owners, list) and owners:
        first = owners[0]
        if isinstance(first, dict):
            return first.get("displayName") or first.get("firstName")
        return str(first)
    return None


def _remaining_faab(league: Any, team: Any) -> int | None:
    """FAAB left, if the league runs one. Budget minus spend."""
    budget = _attribute(league.settings, "acquisition_budget")
    spent = _attribute(team, "acquisition_budget_spent")
    if budget is None or spent is None:
        return None
    try:
        return int(budget) - int(spent)
    except (TypeError, ValueError):
        return None


def _epoch_ms_to_iso(value: Any) -> str | None:
    """ESPN timestamps to an ISO string.

    The activity feed reports times as epoch milliseconds. Passed through as a string
    it renders as "1787929226212" — technically the data, and useless to read. Doing
    the conversion here rather than in the UI keeps every consumer, including a future
    alerts feed, from having to know ESPN's encoding.
    """
    if value is None:
        return None
    try:
        milliseconds = float(value)
    except (TypeError, ValueError):
        # Already a formatted string from some other code path; pass it through.
        return str(value) or None

    # Values below this are implausible as milliseconds and are almost certainly
    # already seconds — guard rather than produce a date in 1970.
    seconds = milliseconds / 1000 if milliseconds > 1e11 else milliseconds
    try:
        return datetime.fromtimestamp(seconds, tz=UTC).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _read_transactions(league: Any, size: int) -> list[TransactionState]:
    """Recent adds, drops, and waiver claims.

    Best-effort: the transaction endpoint is the flakiest part of an already unofficial
    API, and a snapshot without it is still worth having.
    """
    try:
        activity = league.recent_activity(size=size)
    except Exception as error:  # noqa: BLE001 - unofficial API, unbounded failure modes
        logger.warning("could not read the transaction log: %s", error)
        return []

    entries: list[TransactionState] = []
    for item in activity:
        date = _epoch_ms_to_iso(_attribute(item, "date"))
        for action in _attribute(item, "actions", default=[]) or []:
            team, kind, player, bid = (list(action) + [None] * 4)[:4]
            entries.append(
                TransactionState(
                    date=date,
                    kind=str(kind) if kind else None,
                    team_name=str(getattr(team, "team_name", team)) if team else None,
                    player_name=str(getattr(player, "name", player)) if player else None,
                    bid_amount=int(bid) if isinstance(bid, int | float) else None,
                )
            )
    return entries
