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
        self._snapshot: sync.LeagueSnapshot | None = None
        self._fetched_at: float = 0.0
        self._stale = False
        self._lock = threading.Lock()
        # Derived state is per metric version: the snapshot is ESPN data and version
        # independent, but the board it joins against is not. Keyed by (version,
        # board mtime) so a `patron board` run is picked up without a restart.
        self._states: dict[str, LeagueState] = {}
        self._board_cache: dict[tuple[str, float], pl.DataFrame] = {}

    @property
    def snapshot_path(self) -> Path:
        return self._settings.outputs_dir / SNAPSHOT_NAME

    def board_path(self, version: str = "v1") -> Path:
        """Where the built board lives, preferring the versioned file."""
        if version == "v1":
            from patron.artifacts import verify_draft

            archive = self._settings.static_dir / f"draft_{self._config.draft_season}"
            if archive.exists():
                return verify_draft(archive)
        versioned = self._settings.outputs_dir / f"board_{version}.json"
        if version == "v1" and not versioned.exists():
            return self._settings.outputs_dir / "board.json"
        return versioned

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

        key = (version, path.stat().st_mtime)
        cached = self._board_cache.get(key)
        if cached is not None:
            return cached

        board = pl.read_json(path)
        # Bounded: one entry per version per board build, and old mtimes are dropped.
        self._board_cache = {k: v for k, v in self._board_cache.items() if k[0] != version}
        self._board_cache[key] = board
        return board

    def _build_state(self, snapshot: sync.LeagueSnapshot, version: str) -> LeagueState:
        board = self._load_board(version)
        ids, names = crosswalk.board_lookups(board)
        players, report = crosswalk.resolve_player_ids(snapshot.to_frame(), ids, names)
        tagged = crosswalk.attach_ownership(board, players, snapshot.my_team_id)
        tagged = crosswalk.append_espn_fallbacks(tagged, players, snapshot.my_team_id)

        # §8's instruction: log unmatched loudly. The API surfaces this too, so a
        # degraded join is visible in the UI rather than only in a log nobody reads.
        logger.info(report.summary())
        return LeagueState(
            snapshot=snapshot,
            tagged_board=tagged,
            join_report=report,
            fetched_at=time.monotonic(),
        )

    def _restore_from_disk(self) -> bool:
        """Load the persisted snapshot, aged by the file's real mtime.

        Aging it matters: resetting the clock would let a day-old snapshot masquerade
        as fresh and never trigger a refresh.
        """
        if not self.snapshot_path.exists():
            return False
        try:
            snapshot = sync.LeagueSnapshot.read(self.snapshot_path)
        except (OSError, ValueError, TypeError):
            logger.warning("could not read the persisted snapshot", exc_info=True)
            return False

        file_age = time.time() - self.snapshot_path.stat().st_mtime
        self._snapshot = snapshot
        self._fetched_at = time.monotonic() - file_age
        self._states.clear()
        logger.info("restored snapshot from disk, %.0fs old", file_age)
        return True

    def _derive(self, version: str) -> LeagueState:
        """Join the current snapshot to `version`'s board, caching per version."""
        assert self._snapshot is not None
        cached = self._states.get(version)
        if cached is not None and cached.fetched_at == self._fetched_at:
            return cached

        state = self._build_state(self._snapshot, version)
        state.fetched_at = self._fetched_at
        state.stale = self._stale
        self._states[version] = state
        return state

    def get(self, version: str = "v1", force: bool = False) -> LeagueState:
        """Archived model state, using the shared snapshot refresh lifecycle."""
        self._refresh(force)
        return self._derive(version)

    def observations(self, *, force: bool = False) -> tuple[sync.LeagueSnapshot, bool, float]:
        """Current league observations without reading or calculating any model board."""
        self._refresh(force=force)
        assert self._snapshot is not None
        return self._snapshot, self._stale, time.monotonic() - self._fetched_at

    def _refresh(self, force: bool = False) -> None:
        """Refresh the common snapshot; consumers choose their own derived data."""
        if self._snapshot is None:
            self._restore_from_disk()

        age = time.monotonic() - self._fetched_at if self._snapshot is not None else None
        if self._snapshot is not None and not force and age is not None and age < self._ttl:
            return

        # Recorded before queueing on the lock. A refresh that completed after this
        # point is newer than the request, and satisfies it.
        requested_at = time.monotonic()

        acquired = self._lock.acquire(timeout=FETCH_WAIT_SECONDS)
        if not acquired:
            # Another request is already fetching and is taking a long time. Serving
            # what we have beats piling a second request onto ESPN.
            if self._snapshot is not None:
                logger.warning("refresh still in flight; serving the cached snapshot")
                return
            raise TimeoutError("Timed out waiting for the in-flight ESPN refresh.")

        try:
            # Re-check under the lock.
            #
            # `force` means "do not serve me data older than my request" — it does not
            # mean "always issue a network call". Without this second clause, ten
            # concurrent refreshes produced ten ESPN fetches: they serialised on the
            # lock but every one of them still went out. Someone leaning on a Refresh
            # button would hammer an unofficial, free API.
            if self._snapshot is not None and self._fetched_at >= requested_at:
                logger.debug("a concurrent refresh already satisfied this request")
                return
            age = time.monotonic() - self._fetched_at if self._snapshot is not None else None
            if self._snapshot is not None and not force and age is not None and age < self._ttl:
                return

            try:
                snapshot = sync.fetch_snapshot(
                    self._credentials(),
                    self._config.draft_season,
                    # Settled week lineups carry forward, so a poll costs one week of
                    # box scores rather than one per week played so far.
                    previous=self._snapshot,
                )
                snapshot.write(self.snapshot_path)
                snapshot.write_adp_snapshot(self._settings.outputs_dir / "market_snapshots")
                self._snapshot = snapshot
                self._fetched_at = time.monotonic()
                self._stale = False
                self._states.clear()
            except NotAuthenticatedError:
                raise
            except Exception as error:  # noqa: BLE001 - unofficial API, many failure modes
                if self._snapshot is None:
                    raise
                # Stale-if-error: an old wire beats no wire.
                logger.warning("ESPN refresh failed, serving stale state: %s", error)
                self._stale = True
                for state in self._states.values():
                    state.stale = True

            return
        finally:
            self._lock.release()

    def status(self) -> dict[str, object]:
        """What the UI needs to describe freshness and join health honestly."""
        authenticated = self.is_authenticated()
        if self._snapshot is None:
            return {
                "authenticated": authenticated,
                "synced": False,
                "age_seconds": None,
                "stale": False,
                "ttl_seconds": self._ttl,
            }

        snapshot = self._snapshot
        report = next(iter(self._states.values()), None)
        payload: dict[str, object] = {
            "authenticated": authenticated,
            "synced": True,
            "age_seconds": round(time.monotonic() - self._fetched_at),
            "stale": self._stale,
            "ttl_seconds": self._ttl,
            "league_name": snapshot.league_name,
            "season": snapshot.season,
            "week": snapshot.week,
            "my_team_id": snapshot.my_team_id,
        }
        if report is not None:
            join = report.join_report
            payload["join"] = {
                "matched": join.matched_by_id + join.matched_by_name,
                "total": join.total,
                "match_rate": round(join.match_rate, 3),
                "unmatched": len(join.unmatched),
                "not_ranked": join.not_ranked,
            }
        return payload


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
