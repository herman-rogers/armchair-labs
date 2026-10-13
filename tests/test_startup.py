"""Startup table fetch, offline reuse, and dependency installation behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from engine import startup


def test_restart_refreshes_provider_data_before_loading_espn(tmp_path, monkeypatch):
    from contextlib import nullcontext

    events = []
    monkeypatch.setattr(startup.shutil, "which", lambda name: name)
    monkeypatch.setattr(startup, "get_settings", lambda: SimpleNamespace(runtime_dir=tmp_path))
    monkeypatch.setattr(startup.shared, "lock", lambda path: nullcontext())
    monkeypatch.setattr(startup, "prepare_frontend", lambda root: events.append("frontend"))
    monkeypatch.setattr(startup.subprocess, "run", lambda *a, **kw: events.append("tables"))
    monkeypatch.setattr(startup, "refresh_data", lambda **kw: events.append(("refresh", kw)))
    monkeypatch.setattr(startup, "prepare_auth", lambda: events.append("espn"))
    startup.prepare()
    assert events == ["frontend", "tables", ("refresh", {"rescore": True}), "espn"]


def test_verification_distinguishes_network_outage_from_invalid_credentials(monkeypatch):
    from espn_api import football
    from espn_api.requests.espn_requests import ESPNAccessDenied

    from engine.espn import credentials, discovery

    stored = credentials.EspnCredentials("test", "test")
    monkeypatch.setattr(football, "League", Mock(side_effect=ConnectionError("offline")))
    with pytest.raises(discovery.DiscoveryError, match="offline"):
        discovery.verify(stored, 1, 2026, raise_unavailable=True)
    monkeypatch.setattr(football, "League", Mock(side_effect=ESPNAccessDenied))
    ok, message = discovery.verify(stored, 1, 2026, raise_unavailable=True)
    assert not ok
    assert "denied access" in message


@pytest.fixture
def prepared_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(startup, "catalog_state", lambda: {})
    connection = MagicMock()
    execute = connection.__enter__.return_value.execute.return_value
    execute.fetchone.side_effect = [(10,), (2026, 4)]
    monkeypatch.setattr(startup, "connect", lambda data: connection)
    from engine import chairstats

    monkeypatch.setattr(
        chairstats, "CampaignStore", lambda root: SimpleNamespace(campaigns=lambda: [])
    )
    return data


def test_valid_local_tables_need_no_network(prepared_data, monkeypatch):
    monkeypatch.setattr(startup, "current", lambda data: {"tables": {"players": {}}})
    monkeypatch.setattr(
        startup.transport, "fetch", Mock(side_effect=AssertionError("Unexpected fetch"))
    )
    monkeypatch.setattr(
        startup, "cloud_login", Mock(side_effect=AssertionError("Unexpected login"))
    )
    startup.prepare_data(prepared_data)


def test_missing_tables_are_fetched_offline_first(prepared_data, monkeypatch):
    catalogs = iter([{"tables": {}}, {"tables": {"players": {}}}])
    monkeypatch.setattr(startup, "current", lambda data: next(catalogs))
    fetch = Mock()
    monkeypatch.setattr(startup.transport, "fetch", fetch)
    monkeypatch.setattr(
        startup, "cloud_login", Mock(side_effect=AssertionError("Unexpected login"))
    )
    startup.prepare_data(prepared_data)
    assert fetch.call_args.kwargs == {"offline": True}


def test_corrupt_tables_are_not_overwritten(prepared_data, monkeypatch):
    monkeypatch.setattr(startup, "current", lambda data: {"tables": {"players": {}}})
    monkeypatch.setattr(startup, "catalog_state", Mock(side_effect=ValueError("checksum")))
    with pytest.raises(ValueError, match="checksum"):
        startup.prepare_data(prepared_data)


def test_frontend_install_is_reused_until_lock_changes(tmp_path, monkeypatch):
    web = tmp_path / "web"
    (web / "node_modules/.bin").mkdir(parents=True)
    (web / "package-lock.json").write_text("first")
    (web / "node_modules/.bin/vite").touch()
    monkeypatch.setattr(startup.shutil, "which", lambda name: name)
    run = Mock()
    monkeypatch.setattr(startup.subprocess, "run", run)
    startup.prepare_frontend(tmp_path)
    startup.prepare_frontend(tmp_path)
    assert run.call_count == 1
    (web / "package-lock.json").write_text("updated")
    startup.prepare_frontend(tmp_path)
    assert run.call_count == 2


def test_espn_network_outage_does_not_trigger_browser_login(tmp_path, monkeypatch):
    from engine import cli_auth
    from engine.espn import credentials, discovery, service
    from engine.tables import league

    monkeypatch.setattr(
        startup,
        "get_settings",
        lambda: SimpleNamespace(env_path=tmp_path / ".env", data_dir=tmp_path),
    )
    monkeypatch.setattr(credentials, "read", lambda path: SimpleNamespace(league_id=1))
    monkeypatch.setattr(
        discovery, "verify", Mock(side_effect=discovery.DiscoveryError("DNS failed"))
    )
    login = Mock(side_effect=AssertionError("Network failure is not an expired login"))
    monkeypatch.setattr(cli_auth, "login", login)
    snapshot = SimpleNamespace(season=2026, week=5)
    monkeypatch.setattr(
        service,
        "LeagueService",
        lambda **kwargs: SimpleNamespace(observations=lambda: (snapshot, True, 100)),
    )
    monkeypatch.setattr(league, "query_snapshot", lambda *args: snapshot)
    startup.prepare_auth()
    login.assert_not_called()
