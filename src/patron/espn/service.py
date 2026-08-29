"""Server-owned league state.

ESPN pulling belongs behind the API, not in a CLI command someone has to remember to
run. The server holds one snapshot, refreshes it when it goes stale, and serves
everything else from that.

Three properties make it safe to put a third-party fetch on a request path:

**One fetch at a time.** A lock, not a plain TTL check. Ten concurrent requests
arriving on a cold cache must produce one ESPN fetch and nine waits, not ten fetches —
this is an unofficial, free, community-run API and a request stampede is exactly the
behaviour that gets an integration blocked.

**Stale-if-error.** A refresh that fails serves the last good snapshot with its real
age attached rather than surfacing an error. The wire being ten minutes old is almost
always better than the wire being unavailable.

**Warm across restarts.** The snapshot persists to disk, so a server restart does not
mean an immediate refetch.

This is deliberately pull-driven: the server refreshes because someone asked for data
that had gone stale, never on a timer of its own. The autonomous poll is Phase 3.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from patron.config.league import LeagueConfig, get_league
from patron.config.settings import Settings, get_settings
from patron.espn import credentials as creds
from patron.espn import crosswalk, sync
from patron.espn.crosswalk import JoinReport

logger = logging.getLogger(__name__)

SNAPSHOT_NAME = "league_snapshot.json"

#: How long a snapshot is served before a request triggers a refresh. Matches the
#: plan's 10-minute league cadence — rosters, injury tags and the wire move intraday,
#: but not faster than this.
DEFAULT_TTL_SECONDS = 600

#: How long to wait behind another request's in-flight fetch before giving up and
#: serving what we have.
FETCH_WAIT_SECONDS = 45


class NotAuthenticatedError(RuntimeError):
    """Raised when no ESPN credentials are stored."""


@dataclass
class LeagueState:
    """A snapshot plus everything derived from it."""

    snapshot: sync.LeagueSnapshot
    tagged_board: pl.DataFrame
    join_report: JoinReport
    fetched_at: float
    #: True when the last refresh failed and this is the previous good snapshot.
    stale: bool = False

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.fetched_at


class LeagueService:
    """Holds league state for the API. Thread-safe, TTL-refreshed, disk-backed."""

    def __init__(
        self,
        settings: Settings | None = None,
        config: LeagueConfig | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._settings = settings or get_settings()
        self._config = config or get_league()
        self._ttl = ttl_seconds
        self._state: LeagueState | None = None
        self._lock = threading.Lock()
        self._board_cache: tuple[float, pl.DataFrame] | None = None

    @property
    def snapshot_path(self) -> Path:
        return self._settings.outputs_dir / SNAPSHOT_NAME

    def board_path(self, version: str = "v1") -> Path:
        """Where the built board lives, preferring the versioned file."""
        versioned = self._settings.outputs_dir / f"board_{version}.json"
        return versioned if versioned.exists() else self._settings.outputs_dir / "board.json"

    def is_authenticated(self) -> bool:
        stored = creds.read(self._settings.env_path)
        return stored is not None and stored.league_id is not None

    def _credentials(self) -> creds.EspnCredentials:
        stored = creds.read(self._settings.env_path)
        if stored is None or stored.league_id is None:
            raise NotAuthenticatedError(
                "No ESPN credentials stored. Run `patron auth login` to sign in."
            )
        return stored

    def _load_board(self, version: str) -> pl.DataFrame:
        """The built board, re-read when the file on disk changes.

        Keyed by mtime rather than held forever, so a `patron board` run while the
        server is up is picked up without a restart.
        """
        path = self.board_path(version)
        if not path.exists():
            raise FileNotFoundError(f"No board at {path}. Run `patron board` to build one.")

        mtime = path.stat().st_mtime
        if self._board_cache and self._board_cache[0] == mtime:
            return self._board_cache[1]

        board = pl.read_json(path)
        self._board_cache = (mtime, board)
        return board

    def _build_state(self, snapshot: sync.LeagueSnapshot, version: str) -> LeagueState:
        board = self._load_board(version)
        ids, names = crosswalk.board_lookups(board)
        players, report = crosswalk.resolve_player_ids(snapshot.to_frame(), ids, names)
        tagged = crosswalk.attach_ownership(board, players, snapshot.my_team_id)

        # §8's instruction: log unmatched loudly. The API surfaces this too, so a
        # degraded join is visible in the UI rather than only in a log nobody reads.
        logger.info(report.summary())
        return LeagueState(
            snapshot=snapshot,
            tagged_board=tagged,
            join_report=report,
            fetched_at=time.monotonic(),
        )

    def _restore_from_disk(self, version: str) -> LeagueState | None:
        if not self.snapshot_path.exists():
            return None
        try:
            snapshot = sync.LeagueSnapshot.read(self.snapshot_path)
        except (OSError, ValueError, TypeError):
            logger.warning("could not read the persisted snapshot", exc_info=True)
            return None

        state = self._build_state(snapshot, version)
        # Age it by the file's real age so a restart does not pretend to be fresh.
        file_age = time.time() - self.snapshot_path.stat().st_mtime
        state.fetched_at = time.monotonic() - file_age
        logger.info("restored snapshot from disk, %.0fs old", file_age)
        return state

    def get(self, version: str = "v1", force: bool = False) -> LeagueState:
        """Current league state, refreshing it if stale.

        Args:
            version: Metric generation whose board to join against.
            force: Refresh regardless of age.

        Raises:
            NotAuthenticatedError: if no credentials are stored and nothing is cached.
        """
        if self._state is None:
            self._state = self._restore_from_disk(version)

        fresh_enough = self._state is not None and not force and self._state.age_seconds < self._ttl
        if fresh_enough:
            return self._state  # type: ignore[return-value]

        # Recorded before queueing on the lock. A refresh that completed after this
        # point is newer than the request, and satisfies it.
        requested_at = time.monotonic()

        acquired = self._lock.acquire(timeout=FETCH_WAIT_SECONDS)
        if not acquired:
            # Another request is already fetching and is taking a long time. Serving
            # what we have beats piling a second request onto ESPN.
            if self._state is not None:
                logger.warning("refresh still in flight; serving the cached snapshot")
                return self._state
            raise TimeoutError("Timed out waiting for the in-flight ESPN refresh.")

        try:
            # Re-check under the lock: whoever held it may have just refreshed.
            #
            # `force` means "do not serve me data older than my request" — it does not
            # mean "always issue a network call". Without this second clause, ten
            # concurrent refreshes produced ten ESPN fetches: they serialised on the
            # lock but every one of them still went out. Someone leaning on a Refresh
            # button would hammer an unofficial, free API.
            if self._state is not None and self._state.fetched_at >= requested_at:
                logger.debug("a concurrent refresh already satisfied this request")
                return self._state
            if self._state is not None and not force and self._state.age_seconds < self._ttl:
                return self._state

            try:
                snapshot = sync.fetch_snapshot(self._credentials(), self._config.draft_season)
                snapshot.write(self.snapshot_path)
                self._state = self._build_state(snapshot, version)
            except NotAuthenticatedError:
                raise
            except Exception as error:  # noqa: BLE001 - unofficial API, many failure modes
                if self._state is None:
                    raise
                # Stale-if-error: an old wire beats no wire.
                logger.warning("ESPN refresh failed, serving stale state: %s", error)
                self._state.stale = True
                return self._state

            return self._state
        finally:
            self._lock.release()

    def status(self) -> dict[str, object]:
        """What the UI needs to describe freshness and join health honestly."""
        authenticated = self.is_authenticated()
        if self._state is None:
            return {
                "authenticated": authenticated,
                "synced": False,
                "age_seconds": None,
                "stale": False,
                "ttl_seconds": self._ttl,
            }

        state = self._state
        return {
            "authenticated": authenticated,
            "synced": True,
            "age_seconds": round(state.age_seconds),
            "stale": state.stale,
            "ttl_seconds": self._ttl,
            "league_name": state.snapshot.league_name,
            "season": state.snapshot.season,
            "week": state.snapshot.week,
            "my_team_id": state.snapshot.my_team_id,
            "join": {
                "matched": state.join_report.matched_by_id + state.join_report.matched_by_name,
                "total": state.join_report.total,
                "match_rate": round(state.join_report.match_rate, 3),
                "unmatched": len(state.join_report.unmatched),
                "not_ranked": state.join_report.not_ranked,
            },
        }


_service: LeagueService | None = None
_service_lock = threading.Lock()


def get_service() -> LeagueService:
    """Process-wide service instance, for FastAPI dependency injection."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = LeagueService()
    return _service


def reset_service() -> None:
    """Drop the cached instance. For tests."""
    global _service
    with _service_lock:
        _service = None
