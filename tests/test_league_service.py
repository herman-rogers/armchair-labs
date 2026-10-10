"""Unit tests for the server-owned league service.

The service puts a third-party fetch on a request path, so the properties that matter
are the ones that keep that safe: one fetch at a time, stale-if-error, and a `force`
that means "not older than my request" rather than "always hit the network".
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from engine.espn import service as service_module
from engine.espn.credentials import EspnCredentials
from engine.espn.service import LeagueService, NotAuthenticatedError
from engine.espn.sync import LeagueSnapshot, PlayerState, TeamState


def make_snapshot(week: int = 1) -> LeagueSnapshot:
    return LeagueSnapshot(
        captured_at="2026-08-29T00:00:00+00:00",
        league_id=244615326,
        league_name="Sweaty Plays",
        season=2026,
        week=week,
        my_team_id=3,
        teams=[TeamState(3, "Armchair Labs", "me", 0, 0, 145)],
        players=[
            PlayerState(1, "Owned Guy", "RB", "SF", "ACTIVE", 99.0, 80.0, 12.0, 3, "Mine", "RB"),
            PlayerState(2, "Free Guy", "WR", "SEA", "ACTIVE", 10.0, 2.0, 8.0, None, None, None),
        ],
    )


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LeagueService:
    (tmp_path / "outputs").mkdir()
    from engine.config.settings import Settings

    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(
        service_module.creds,
        "read",
        lambda _: EspnCredentials(espn_s2="s2", swid="{X}", league_id=1, team_id=3),
    )
    return LeagueService(settings=settings, ttl_seconds=600)


def counting_fetch(monkeypatch, calls, week=1, delay=0.0):
    def fetch(*args, **kwargs):
        calls.append(1)
        if delay:
            time.sleep(delay)
        return make_snapshot(week)

    monkeypatch.setattr(service_module.sync, "fetch_snapshot", fetch)


class TestFetching:
    def test_first_call_fetches(self, service, monkeypatch) -> None:
        monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot())
        snapshot, stale, age = service.observations()
        assert snapshot.league_name == "Sweaty Plays"
        assert not stale
        assert age >= 0

    def test_a_fresh_snapshot_is_not_refetched(self, service, monkeypatch) -> None:
        calls: list[int] = []
        counting_fetch(monkeypatch, calls)
        for _ in range(3):
            service.observations()
        assert len(calls) == 1

    def test_an_expired_snapshot_is_refetched(self, service, monkeypatch) -> None:
        calls: list[int] = []
        counting_fetch(monkeypatch, calls)
        service.observations()
        service._fetched_at = time.monotonic() - 10_000
        service.observations()
        assert len(calls) == 2


class TestConcurrency:
    def test_a_cold_stampede_produces_one_fetch(self, service, monkeypatch) -> None:
        """Ten simultaneous requests on a cold cache must make one call, not ten."""
        calls: list[int] = []
        counting_fetch(monkeypatch, calls, delay=0.2)
        threads = [threading.Thread(target=service.observations) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(calls) == 1

    def test_concurrent_forced_refreshes_coalesce(self, service, monkeypatch) -> None:
        """Force means "nothing older than my request", not "always hit the network"."""
        calls: list[int] = []
        counting_fetch(monkeypatch, calls, delay=0.2)
        service.observations()
        calls.clear()
        threads = [
            threading.Thread(target=lambda: service.observations(force=True)) for _ in range(10)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(calls) == 1, f"expected coalescing, got {len(calls)} fetches"

    def test_a_later_force_still_refetches(self, service, monkeypatch) -> None:
        calls: list[int] = []
        counting_fetch(monkeypatch, calls)
        service.observations()
        time.sleep(0.01)
        service.observations(force=True)
        assert len(calls) == 2


class TestResilience:
    def test_a_failed_refresh_serves_the_last_good_snapshot(self, service, monkeypatch) -> None:
        """An old wire beats no wire. But it must announce itself as old."""
        monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot())
        service.observations()

        def boom(*args, **kwargs):
            raise RuntimeError("ESPN is down")

        monkeypatch.setattr(service_module.sync, "fetch_snapshot", boom)
        snapshot, stale, _ = service.observations(force=True)
        assert stale is True
        assert snapshot.league_name == "Sweaty Plays"
        assert service.status()["stale"] is True

    def test_a_failure_with_no_cache_raises(self, service, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("ESPN is down")

        monkeypatch.setattr(service_module.sync, "fetch_snapshot", boom)
        with pytest.raises(RuntimeError, match="ESPN is down"):
            service.observations()

    def test_missing_credentials_raise_a_named_error(self, service, monkeypatch) -> None:
        monkeypatch.setattr(service_module.creds, "read", lambda _: None)
        with pytest.raises(NotAuthenticatedError, match="engine auth login"):
            service.observations()


class TestPersistence:
    def test_a_snapshot_survives_a_restart(self, service, monkeypatch) -> None:
        """A server restart should not mean an immediate refetch."""
        calls: list[int] = []
        counting_fetch(monkeypatch, calls)
        service.observations()
        assert service.snapshot_path.exists()
        restarted = LeagueService(settings=service._settings, ttl_seconds=600)
        snapshot, _, _ = restarted.observations()
        assert len(calls) == 1, "restoring from disk must not refetch"
        assert snapshot.league_name == "Sweaty Plays"

    def test_a_restored_snapshot_reports_its_real_age(self, service, monkeypatch) -> None:
        import os

        monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot())
        service.observations()
        old = time.time() - 3600
        os.utime(service.snapshot_path, (old, old))
        restarted = LeagueService(settings=service._settings, ttl_seconds=600)
        assert restarted._restore_from_disk() is True
        assert time.monotonic() - restarted._fetched_at > 3000


class TestStatus:
    def test_status_before_any_sync_never_fetches(self, service, monkeypatch) -> None:
        calls: list[int] = []
        counting_fetch(monkeypatch, calls)
        status = service.status()
        assert calls == []
        assert status["synced"] is False
        assert status["authenticated"] is True

    def test_status_after_sync_describes_the_snapshot(self, service, monkeypatch) -> None:
        monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot(4))
        service.observations()
        status = service.status()
        assert status["synced"] is True
        assert status["week"] == 4
        assert "join" not in status


def test_a_refresh_replaces_the_snapshot_for_every_reader(service, monkeypatch):
    monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot(1))
    service.observations()
    monkeypatch.setattr(service_module.sync, "fetch_snapshot", lambda *a, **k: make_snapshot(7))
    snapshot, _, _ = service.observations(force=True)
    assert snapshot.week == 7
    assert service.observations()[0].week == 7
