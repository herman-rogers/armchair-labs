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

from engine.espn import teams
from engine.espn.credentials import EspnCredentials

logger = logging.getLogger(__name__)

#: Positions the board ranks. Kickers and defenses are tiered separately.
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

#: How deep to read the free-agent pool. The wire below this is noise, and every extra
#: page is another request against a community-run service.
DEFAULT_FREE_AGENT_DEPTH = 300

#: Slots that are not part of a starting lineup. A player in one of these scored for
#: nobody that week, which is the whole distinction the matchup view turns on.
_NON_STARTING_SLOTS = frozenset({"BE", "IR", "ER", ""})


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
    # ESPN's PPR draft-room ordering. This is market fallback evidence, not a model
    # projection; optional defaults keep pre-feature snapshots readable.
    espn_draft_rank: int | None = None
    espn_position_rank: int | None = None
    espn_adp: float | None = None


@dataclass
class ScheduleEntry:
    """One week of a team's season."""

    week: int
    opponent_team_id: int
    #: Points scored, once the week has been played. 0.0 for a future week.
    score: float
    #: ESPN's outcome code: W, L, T, or U for unplayed.
    outcome: str

    @property
    def played(self) -> bool:
        return self.outcome in {"W", "L", "T"}


@dataclass
class TeamState:
    team_id: int
    team_name: str
    owner: str | None
    wins: int
    losses: int
    faab_remaining: int | None
    # ESPN assigns both on the Team object. Optional defaults keep snapshots written
    # before division ingestion backward-compatible.
    division_id: int | None = None
    division_name: str | None = None
    schedule: list[ScheduleEntry] = field(default_factory=list)


@dataclass
class TransactionState:
    """One entry from the transaction log — the league's revealed price signal."""

    date: str | None
    kind: str | None
    team_name: str | None
    player_name: str | None
    bid_amount: int | None


@dataclass
class DraftPick:
    """One selection from the league's draft recap."""

    overall: int
    round: int
    round_pick: int
    team_id: int
    team_name: str
    espn_id: int
    player_display_name: str
    bid_amount: int | None = None
    keeper: bool = False


@dataclass
class LineupEntry:
    """One rostered player in one week, in the slot they actually occupied.

    This is history, not a plan: the slot is what the manager set before kickoff and
    `points` is what ESPN finally credited. A player on the bench here scored for
    nobody that week, however highly the board rates him.
    """

    espn_id: int
    player_display_name: str
    position: str | None
    #: ESPN's slot for that week: a starting slot like "QB" or "RB/WR/TE", or a
    #: non-starting one — "BE", "IR", "ER".
    slot: str
    points: float
    #: What ESPN projected before the week. Kept because the interesting bench
    #: question is whether a start was defensible at the time, not only in hindsight.
    projected_points: float | None
    pro_opponent: str | None
    on_bye: bool

    @property
    def started(self) -> bool:
        return self.slot not in _NON_STARTING_SLOTS


@dataclass
class WeekLineups:
    """One head-to-head, as the two managers actually fielded it."""

    week: int
    home_team_id: int
    away_team_id: int
    home_score: float
    away_score: float
    #: ESPN's own pre-week projection for each side, when it publishes one.
    home_projected: float | None = None
    away_projected: float | None = None
    home_lineup: list[LineupEntry] = field(default_factory=list)
    away_lineup: list[LineupEntry] = field(default_factory=list)

    def lineup_for(self, team_id: int) -> list[LineupEntry]:
        if team_id == self.home_team_id:
            return self.home_lineup
        if team_id == self.away_team_id:
            return self.away_lineup
        return []


@dataclass
class LeagueSnapshot:
    """Everything one poll captured."""

    captured_at: str
    league_id: int
    league_name: str
    season: int
    week: int
    my_team_id: int | None
    #: Weeks in the regular season, so the UI knows how far the schedule runs.
    regular_season_weeks: int = 0
    #: Starting lineup shape, e.g. {"QB": 1, "RB": 2, "WR": 3, "RB/WR/TE": 1}. Read
    #: from ESPN rather than configured: league settings are authoritative, and a
    #: commissioner adding a flex should not require a code change here.
    roster_slots: dict[str, int] = field(default_factory=dict)
    teams: list[TeamState] = field(default_factory=list)
    players: list[PlayerState] = field(default_factory=list)
    transactions: list[TransactionState] = field(default_factory=list)
    draft: list[DraftPick] = field(default_factory=list)
    #: Every played and in-progress week's real lineups, oldest first. `players` above
    #: is the roster as it stands *now*; this is the only record of who started when,
    #: and the two disagree the moment anyone benches a player or works the wire.
    week_lineups: list[WeekLineups] = field(default_factory=list)

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
            "espn_draft_rank": pl.Int64,
            "espn_position_rank": pl.Int64,
            "espn_adp": pl.Float64,
        }
        # An explicit schema, not inference: a column that happens to be entirely null
        # in one snapshot would otherwise come back as Null dtype and fail to join.
        return pl.DataFrame([asdict(player) for player in self.players], schema=schema)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path

    def write_adp_snapshot(self, directory: Path) -> Path | None:
        """Persist one dated observed-ADP panel for future historical evaluation.

        ESPN exposes current ADP but no archive.  Keeping one file per UTC date means
        future reports can use genuine observed draft price without pretending the
        existing FantasyPros ECR archive is ADP. Repeated polls on a date replace that
        date's panel instead of creating an unbounded stream of files.
        """
        frame = self.to_frame().filter(
            pl.col("espn_adp").is_not_null() & pl.col("position").is_in(SKILL_POSITIONS)
        )
        if frame.height == 0:
            return None
        captured = datetime.fromisoformat(self.captured_at.replace("Z", "+00:00"))
        path = directory / f"espn_adp_{self.season}_{captured.date().isoformat()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.select(
            pl.lit(self.season).cast(pl.Int32).alias("forecast_season"),
            pl.lit(self.captured_at).alias("captured_at"),
            "espn_id",
            "player_display_name",
            "position",
            "espn_team",
            "espn_draft_rank",
            "espn_position_rank",
            "espn_adp",
        ).write_parquet(path)
        return path

    @classmethod
    def read(cls, path: Path) -> LeagueSnapshot:
        raw = json.loads(path.read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> LeagueSnapshot:
        """Decode either a raw capture or queried league table records."""
        return cls(
            **{
                **raw,
                "teams": [
                    TeamState(
                        **{
                            **team,
                            "schedule": [
                                ScheduleEntry(**entry) for entry in team.get("schedule", [])
                            ],
                        }
                    )
                    for team in raw.get("teams", [])
                ],
                "players": [PlayerState(**player) for player in raw.get("players", [])],
                "transactions": [
                    TransactionState(**entry) for entry in raw.get("transactions", [])
                ],
                "draft": [DraftPick(**entry) for entry in raw.get("draft", [])],
                # Absent from snapshots written before per-week lineups existed, which
                # must stay readable rather than crash a server on restart.
                "week_lineups": [
                    WeekLineups(
                        **{
                            **week,
                            "home_lineup": [
                                LineupEntry(**entry) for entry in week.get("home_lineup", [])
                            ],
                            "away_lineup": [
                                LineupEntry(**entry) for entry in week.get("away_lineup", [])
                            ],
                        }
                    )
                    for week in raw.get("week_lineups", [])
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


def _player_state(
    player: Any,
    team: Any = None,
    *,
    draft_rank: int | None = None,
    position_rank: int | None = None,
    adp: float | None = None,
) -> PlayerState:
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
        espn_draft_rank=draft_rank,
        espn_position_rank=position_rank,
        espn_adp=adp,
    )


def _fetch_draft_market(league: Any, size: int = 1000) -> dict[int, tuple[int, int, float | None]]:
    """Current ESPN PPR draft order as player id → overall/position rank and ADP.

    ``espn-api`` requests this ordering for free agents but discards both the sorted
    ordinal and ``ownership.averageDraftPosition``. One bounded request retains that
    evidence for every status. Skill-position ordinals exclude K/DST because Armchair Labs's
    board does too.
    """
    filters = {
        "players": {
            "filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]},
            "limit": size,
            "sortDraftRanks": {
                "sortPriority": 1,
                "sortAsc": True,
                "value": "PPR",
            },
        }
    }
    try:
        data = league.espn_request.league_get(
            params={"view": "kona_player_info", "scoringPeriodId": league.current_week},
            headers={"x-fantasy-filter": json.dumps(filters)},
        )
    except Exception as error:  # noqa: BLE001 - optional evidence from unofficial API
        logger.warning("ESPN draft ranking unavailable: %s", error)
        return {}

    skill_position_ids = {1, 2, 3, 4}
    overall_rank = 0
    position_ranks: dict[int, int] = {position_id: 0 for position_id in skill_position_ids}
    market: dict[int, tuple[int, int, float | None]] = {}
    for entry in data.get("players") or []:
        raw = (entry.get("playerPoolEntry") or {}).get("player") or entry.get("player") or {}
        position_id = raw.get("defaultPositionId")
        player_id = raw.get("id")
        if position_id not in skill_position_ids or player_id is None:
            continue
        overall_rank += 1
        position_ranks[position_id] += 1
        ownership = raw.get("ownership") or {}
        market[int(player_id)] = (
            overall_rank,
            position_ranks[position_id],
            _number(ownership.get("averageDraftPosition")),
        )
    return market


def _read_draft(league: Any, team_count: int) -> list[DraftPick]:
    """The draft recap in pick order; empty when the league has not drafted."""
    picks: list[DraftPick] = []
    try:
        raw = list(getattr(league, "draft", None) or [])
    except Exception as error:  # noqa: BLE001 - optional evidence from unofficial API
        logger.warning("ESPN draft recap unavailable: %s", error)
        return picks
    for entry in raw:
        team = _attribute(entry, "team")
        round_num = int(_attribute(entry, "round_num", default=0) or 0)
        round_pick = int(_attribute(entry, "round_pick", default=0) or 0)
        overall = (round_num - 1) * max(team_count, 1) + round_pick if round_num else len(picks) + 1
        picks.append(
            DraftPick(
                overall=overall,
                round=round_num,
                round_pick=round_pick,
                team_id=int(getattr(team, "team_id", 0) or 0),
                team_name=str(getattr(team, "team_name", "") or ""),
                espn_id=int(_attribute(entry, "playerId", default=0) or 0),
                player_display_name=str(_attribute(entry, "playerName", default="") or ""),
                bid_amount=(
                    int(bid) if (bid := _attribute(entry, "bid_amount")) is not None else None
                ),
                keeper=bool(_attribute(entry, "keeper_status", default=False)),
            )
        )
    picks.sort(key=lambda pick: pick.overall)
    return picks


def _team_id(side: Any) -> int | None:
    """The team id on one side of a box score.

    espn-api swaps the raw id for a `Team` once it can match one, leaves the bare int
    when it cannot, and uses None for a bye. All three arrive here.
    """
    if side is None:
        return None
    value = getattr(side, "team_id", side)
    try:
        # ESPN numbers teams from 1; a zero is its sentinel for "no opponent".
        return int(value) or None
    except (TypeError, ValueError):
        return None


def _projected(value: Any) -> float | None:
    """ESPN's projected total, with its -1 "not published" sentinel as None."""
    number = _number(value)
    return None if number is None or number < 0 else number


def _lineup_entry(player: Any) -> LineupEntry:
    return LineupEntry(
        espn_id=int(_attribute(player, "playerId", default=0) or 0),
        player_display_name=str(_attribute(player, "name", default="") or ""),
        position=_text(_attribute(player, "position")),
        slot=str(_attribute(player, "slot_position", default="") or ""),
        points=float(_number(_attribute(player, "points")) or 0.0),
        projected_points=_number(_attribute(player, "projected_points")),
        pro_opponent=teams.to_nflverse(_text(_attribute(player, "pro_opponent"))),
        on_bye=bool(_attribute(player, "on_bye_week", default=False)),
    )


def _read_week_lineups(
    league: Any,
    current_week: int,
    previous: list[WeekLineups] | None = None,
) -> list[WeekLineups]:
    """Who actually started, week by week, up to and including the current one.

    The roster on the snapshot is the roster *now*. It cannot answer "who did I start
    in week 2" — a bench move or a waiver claim since then makes it wrong, and the
    best-lineup solver makes it wrong in a way that looks plausible, quietly promoting
    a benched player into a starting slot. This is the record that settles it.

    Complete weeks already captured are reused rather than refetched. A played week
    is settled history, and each refetch is three more requests against an unofficial API for an
    answer that cannot have changed. Only the current week — still scoring, still being
    edited — is always pulled fresh, so the ongoing cost of this is one week per poll
    rather than one per week of the season.
    """
    settled: dict[int, list[WeekLineups]] = {}
    for game in previous or []:
        if game.week < current_week:
            settled.setdefault(game.week, []).append(game)

    # Older snapshots accidentally kept just the last matchup in each week. Use
    # the schedule to detect those incomplete weeks and fetch their missing games.
    scheduled: dict[int, set[frozenset[int]]] = {}
    for team in getattr(league, "teams", []):
        for entry in _read_schedule(team):
            if entry.opponent_team_id <= 0 or entry.opponent_team_id == team.team_id:
                continue
            scheduled.setdefault(entry.week, set()).add(
                frozenset({team.team_id, entry.opponent_team_id})
            )

    weeks: list[WeekLineups] = []
    # Shared across weeks so a player on bye resolves to the pro team he was on at the
    # time rather than whoever has traded for him since.
    player_team_cache: dict[int, int] = {}

    for week in range(1, max(current_week, 0) + 1):
        cached = settled.get(week, [])
        cached_pairs = {frozenset({game.home_team_id, game.away_team_id}) for game in cached}
        if (
            cached
            and all(game.home_lineup and game.away_lineup for game in cached)
            and (week not in scheduled or cached_pairs == scheduled[week])
        ):
            weeks.extend(cached)
            continue
        try:
            box_scores = league.box_scores(week, player_team_cache=player_team_cache)
        except Exception as error:  # noqa: BLE001 - unofficial API, unbounded failures
            # Best-effort, week by week: one unreadable week should cost that week's
            # detail, not the whole snapshot.
            logger.warning("ESPN lineups for week %s unavailable: %s", week, error)
            continue

        for box in box_scores:
            home_id = _team_id(getattr(box, "home_team", None))
            away_id = _team_id(getattr(box, "away_team", None))
            if home_id is None or away_id is None:
                # A bye in an odd-sized league: there is no head-to-head to show.
                continue
            weeks.append(
                WeekLineups(
                    week=week,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    home_score=float(_number(getattr(box, "home_score", None)) or 0.0),
                    away_score=float(_number(getattr(box, "away_score", None)) or 0.0),
                    home_projected=_projected(getattr(box, "home_projected", None)),
                    away_projected=_projected(getattr(box, "away_projected", None)),
                    home_lineup=[_lineup_entry(p) for p in getattr(box, "home_lineup", [])],
                    away_lineup=[_lineup_entry(p) for p in getattr(box, "away_lineup", [])],
                )
            )

    weeks.sort(key=lambda entry: (entry.week, entry.home_team_id))
    return weeks


def fetch_snapshot(
    credentials: EspnCredentials,
    season: int,
    free_agent_depth: int = DEFAULT_FREE_AGENT_DEPTH,
    transaction_size: int = 100,
    previous: LeagueSnapshot | None = None,
) -> LeagueSnapshot:
    """Pull one complete picture of the league.

    Args:
        credentials: Cookies plus the league id.
        season: Season to read.
        free_agent_depth: How many free agents to pull.
        transaction_size: How many recent transactions to read.
        previous: The last snapshot, if there is one. Its settled week lineups are
            carried forward instead of refetched — see `_read_week_lineups`.

    Raises:
        ValueError: if no league id is configured.
    """
    from espn_api.football import League

    if credentials.league_id is None:
        raise ValueError(
            "No ESPN league id configured. Run `engine auth login`, or set ESPN_LEAGUE_ID in .env."
        )

    league = League(
        league_id=credentials.league_id,
        year=season,
        espn_s2=credentials.espn_s2,
        swid=credentials.swid,
    )
    draft_market = _fetch_draft_market(league)

    def state(player: Any, team: Any = None) -> PlayerState:
        player_id = int(_attribute(player, "playerId", default=0) or 0)
        rank, position_rank, adp = draft_market.get(player_id, (None, None, None))
        return _player_state(
            player,
            team,
            draft_rank=rank,
            position_rank=position_rank,
            adp=adp,
        )

    teams = [
        TeamState(
            team_id=int(team.team_id),
            team_name=str(team.team_name),
            owner=_owner_name(team),
            wins=int(_attribute(team, "wins", default=0)),
            losses=int(_attribute(team, "losses", default=0)),
            faab_remaining=_remaining_faab(league, team),
            schedule=_read_schedule(team),
            division_id=(
                int(division_id)
                if (division_id := _attribute(team, "division_id")) is not None
                else None
            ),
            division_name=_text(_attribute(team, "division_name")),
        )
        for team in league.teams
    ]

    players: list[PlayerState] = []
    for team in league.teams:
        players.extend(state(player, team) for player in team.roster)

    free_agents = league.free_agents(size=free_agent_depth)
    players.extend(state(player) for player in free_agents)

    transactions = _read_transactions(league, transaction_size)
    draft = _read_draft(league, len(teams))
    current_week = int(league.current_week)
    week_lineups = _read_week_lineups(
        league,
        current_week,
        previous.week_lineups if previous is not None and previous.season == season else None,
    )

    snapshot = LeagueSnapshot(
        captured_at=datetime.now(tz=UTC).isoformat(),
        league_id=int(credentials.league_id),
        league_name=str(league.settings.name),
        season=season,
        week=current_week,
        my_team_id=credentials.team_id,
        regular_season_weeks=int(_attribute(league.settings, "reg_season_count", default=0) or 0),
        roster_slots=_starting_slots(league),
        teams=teams,
        players=players,
        transactions=transactions,
        draft=draft,
        week_lineups=week_lineups,
    )
    logger.info(
        "snapshot: %s teams, %s rostered + %s free agents, %s transactions, %s weeks of lineups",
        len(teams),
        len(players) - len(free_agents),
        len(free_agents),
        len(transactions),
        len({week.week for week in week_lineups}),
    )
    return snapshot


def _starting_slots(league: Any) -> dict[str, int]:
    """The league's starting lineup shape, bench and IR excluded."""
    counts = _attribute(league.settings, "position_slot_counts", default={}) or {}
    return {
        slot: int(count)
        for slot, count in counts.items()
        if slot not in _NON_STARTING_SLOTS and int(count) > 0
    }


def _read_schedule(team: Any) -> list[ScheduleEntry]:
    """A team's season, week by week.

    espn-api exposes the schedule as three parallel lists — opponents, scores, and
    outcomes — so they are zipped back into one row per week here. A future week comes
    through with a zero score and outcome "U", which is how the UI tells a played week
    from a scheduled one.
    """
    opponents = _attribute(team, "schedule", default=[]) or []
    scores = _attribute(team, "scores", default=[]) or []
    outcomes = _attribute(team, "outcomes", default=[]) or []

    entries: list[ScheduleEntry] = []
    for index, opponent in enumerate(opponents):
        opponent_id = getattr(opponent, "team_id", None)
        if opponent_id is None:
            continue
        entries.append(
            ScheduleEntry(
                week=index + 1,
                opponent_team_id=int(opponent_id),
                score=float(scores[index]) if index < len(scores) else 0.0,
                outcome=str(outcomes[index]) if index < len(outcomes) else "U",
            )
        )
    return entries


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
