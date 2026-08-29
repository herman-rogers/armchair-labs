"""Unit tests for the server-owned league service.

The service puts a third-party fetch on a request path, so the properties that matter
are the ones that keep that safe: one fetch at a time, stale-if-error, and a `force`
that means "not older than my request" rather than "always hit the network".
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import polars as pl
import pytest

from patron.espn import service as service_module
from patron.espn.credentials import EspnCredentials
from patron.espn.service import LeagueService, NotAuthenticatedError
from patron.espn.sync import LeagueSnapshot, PlayerState, TeamState


def make_snapshot(week: int = 1) -> LeagueSnapshot:
    return LeagueSnapshot(
        captured_at="2026-08-29T00:00:00+00:00",
        league_id=244615326,
        league_name="Sweaty Plays",
        season=2026,
        week=week,
        my_team_id=3,
        teams=[TeamState(3, "Patron Saints", "me", 0, 0, 145)],
        players=[
            PlayerState(1, "Owned Guy", "RB", "SF", "ACTIVE", 99.0, 80.0, 12.0, 3, "Mine", "RB"),
            PlayerState(2, "Free Guy", "WR", "SEA", "ACTIVE", 10.0, 2.0, 8.0, None, None, None),
        ],
    )


def write_board(path: Path) -> None:
    pl.DataFrame(
        {
            "player_id": ["g1", "g2"],
            "player_display_name": ["Owned Guy", "Free Guy"],
            "position": ["RB", "WR"],
            "team": ["SF", "SEA"],
            "ppg": [18.0, 9.0],
            "adj_vor": [6.0, -1.0],
            "flags": ["", ""],
        }
    ).write_json(path)


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LeagueService:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    write_board(outputs / "board_v1.json")

    from patron.config.settings import Settings

    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(
        service_module.creds,
        "read",
        lambda _: EspnCredentials(espn_s2="s2", swid="{X}", league_id=1, team_id=3),
    )
    return LeagueService(settings=settings, ttl_seconds=600)


class TestFetching:
    def test_first_call_fetches_and_joins(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot())
        state = service.get()

        assert state.snapshot.league_name == "Sweaty Plays"
        assert state.tagged_board.height == 2
        assert state.join_report.matched_by_name == 2

    def test_a_fresh_snapshot_is_not_refetched(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            service_module.sync,
            "fetch_snapshot",
            lambda *a, **k: (calls.append(1), make_snapshot())[1],
        )
        service.get()
        service.get()
        service.get()

        assert len(calls) == 1

    def test_an_expired_snapshot_is_refetched(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            service_module.sync,
            "fetch_snapshot",
            lambda *a, **k: (calls.append(1), make_snapshot())[1],
        )
        service.get()
        service._state.fetched_at = time.monotonic() - 10_000
        service.get()

        assert len(calls) == 2


class TestConcurrency:
    def test_a_cold_stampede_produces_one_fetch(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ten simultaneous requests on a cold cache must make one call, not ten.
        This is an unofficial, free, community-run API."""
        calls = []

        def slow_fetch(*args, **kwargs):
            calls.append(1)
            time.sleep(0.2)
            return make_snapshot()

        monkeypatch.setattr(service_module.sync, "fetch_snapshot", slow_fetch)

        threads = [threading.Thread(target=service.get) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(calls) == 1

    def test_concurrent_forced_refreshes_coalesce(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bug this guards: `force` skipped the under-lock recheck, so ten
        concurrent refreshes serialised on the lock and every one still went out to
        ESPN. Force means "nothing older than my request", not "always hit the
        network"."""
        calls = []

        def slow_fetch(*args, **kwargs):
            calls.append(1)
            time.sleep(0.2)
            return make_snapshot()

        monkeypatch.setattr(service_module.sync, "fetch_snapshot", slow_fetch)
        service.get()  # warm
        calls.clear()

        threads = [threading.Thread(target=lambda: service.get(force=True)) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(calls) == 1, f"expected coalescing, got {len(calls)} fetches"

    def test_a_later_force_still_refetches(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Coalescing must not turn force into a no-op for a genuinely later request."""
        calls = []
        monkeypatch.setattr(
            service_module.sync,
            "fetch_snapshot",
            lambda *a, **k: (calls.append(1), make_snapshot())[1],
        )
        service.get()
        time.sleep(0.01)
        service.get(force=True)

        assert len(calls) == 2


class TestResilience:
    def test_a_failed_refresh_serves_the_last_good_snapshot(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An old wire beats no wire. But it must announce itself as old."""
        monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot())
        service.get()

        def boom(*args, **kwargs):
            raise RuntimeError("ESPN is down")

        monkeypatch.setattr(service_module.sync, "fetch_snapshot", boom)
        state = service.get(force=True)

        assert state.stale is True
        assert state.snapshot.league_name == "Sweaty Plays"
        assert service.status()["stale"] is True

    def test_a_failure_with_no_cache_raises(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("ESPN is down")

        monkeypatch.setattr(service_module.sync, "fetch_snapshot", boom)
        with pytest.raises(RuntimeError, match="ESPN is down"):
            service.get()

    def test_missing_credentials_raise_a_named_error(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service_module.creds, "read", lambda _: None)
        with pytest.raises(NotAuthenticatedError, match="patron auth login"):
            service.get()

    def test_a_missing_board_is_reported_clearly(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (service._settings.outputs_dir / "board_v1.json").unlink()
        monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot())

        with pytest.raises(FileNotFoundError, match="patron board"):
            service.get()


class TestPersistence:
    def test_a_snapshot_survives_a_restart(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A server restart should not mean an immediate refetch."""
        calls = []
        monkeypatch.setattr(
            service_module.sync,
            "fetch_snapshot",
            lambda *a, **k: (calls.append(1), make_snapshot())[1],
        )
        service.get()
        assert service.snapshot_path.exists()

        restarted = LeagueService(settings=service._settings, ttl_seconds=600)
        monkeypatch.setattr(
            service_module.creds,
            "read",
            lambda _: EspnCredentials(espn_s2="s2", swid="{X}", league_id=1, team_id=3),
        )
        state = restarted.get()

        assert len(calls) == 1, "restoring from disk must not refetch"
        assert state.snapshot.league_name == "Sweaty Plays"

    def test_a_restored_snapshot_reports_its_real_age(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Restoring must not reset the clock, or a day-old snapshot would look fresh
        and never refresh."""
        import os

        monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot())
        service.get()

        old = time.time() - 3600
        os.utime(service.snapshot_path, (old, old))

        restarted = LeagueService(settings=service._settings, ttl_seconds=600)
        restored = restarted._restore_from_disk("v1")

        assert restored is not None
        assert restored.age_seconds > 3000


class TestStatus:
    def test_status_before_any_sync_never_fetches(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            service_module.sync,
            "fetch_snapshot",
            lambda *a, **k: (calls.append(1), make_snapshot())[1],
        )
        status = service.status()

        assert calls == []
        assert status["synced"] is False
        assert status["authenticated"] is True

    def test_status_reports_join_health(
        self, service: LeagueService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Degraded joins must be visible in the UI, not only in a log nobody reads."""
        monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot())
        service.get()
        join = service.status()["join"]

        assert join["matched"] == 2
        assert join["match_rate"] == pytest.approx(1.0)
