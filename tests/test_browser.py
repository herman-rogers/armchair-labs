"""Browser discovery and nonblocking Linux startup regressions."""

from unittest.mock import Mock

import pytest

from engine.espn import browser


def test_flatpak_chrome_gets_access_only_to_login_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(browser, "_chrome_binary", lambda: None)
    monkeypatch.setattr(browser.shutil, "which", lambda name: "/usr/bin/flatpak")
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(browser.subprocess, "run", run)
    assert browser._chrome_command(tmp_path) == [
        "/usr/bin/flatpak",
        "run",
        f"--filesystem={tmp_path}",
        "com.google.Chrome",
    ]
    assert run.call_args.args[0] == ["/usr/bin/flatpak", "info", "com.google.Chrome"]


def test_missing_flatpak_apps_report_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(browser, "_chrome_binary", lambda: None)
    monkeypatch.setattr(browser.shutil, "which", lambda name: "/usr/bin/flatpak")
    monkeypatch.setattr(browser.subprocess, "run", Mock(return_value=Mock(returncode=1)))
    with pytest.raises(browser.BrowserUnavailableError, match="No Chrome"):
        browser._chrome_command(tmp_path)


def test_linux_waits_for_debugger_without_waiting_for_browser_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(browser.platform, "system", lambda: "Linux")
    monkeypatch.setattr(browser, "_chrome_command", lambda profile: ["chromium"])
    monkeypatch.setattr(browser, "_probe", lambda port: {"Browser": "Chromium"})
    monkeypatch.setattr(browser, "_find_pid", lambda profile: None)
    process = Mock(pid=1234)
    popen = Mock(return_value=process)
    monkeypatch.setattr(browser.subprocess, "Popen", popen)
    session = browser.launch(tmp_path, port=9223)
    assert session.pid == 1234
    assert session.cdp_url == "http://127.0.0.1:9223"
    assert "--remote-debugging-port=9223" in popen.call_args.args[0]
    process.wait.assert_not_called()


def test_successful_launcher_exit_does_not_preempt_debugger_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(browser.platform, "system", lambda: "Linux")
    monkeypatch.setattr(browser, "_chrome_command", lambda profile: ["chromium"])
    monkeypatch.setattr(browser, "_probe", Mock(side_effect=[None, {"Browser": "Chromium"}]))
    monkeypatch.setattr(browser, "_find_pid", lambda profile: 4567)
    monkeypatch.setattr(browser.time, "sleep", lambda seconds: None)
    process = Mock(pid=1234, returncode=0)
    process.poll.return_value = 0
    monkeypatch.setattr(browser.subprocess, "Popen", Mock(return_value=process))
    assert browser.launch(tmp_path, port=9223).pid == 4567
