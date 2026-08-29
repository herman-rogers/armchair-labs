"""Project-scoped development process supervision."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

from patron import dev


def test_state_round_trip(tmp_path: Path) -> None:
    state = dev.DevState(
        token="abc",
        supervisor_pid=10,
        repo_root="/repo",
        api_port=8000,
        web_port=5173,
        children=[dev.ChildRecord("api", 11, 11, "patron.api.app:app")],
    )
    path = tmp_path / "runtime" / "dev.json"

    state.write(path)

    assert dev.DevState.read(path) == state
    assert not list(path.parent.glob("*.tmp")), "atomic-write temporary file leaked"


def test_corrupt_state_is_treated_as_stale(tmp_path: Path) -> None:
    path = tmp_path / "dev.json"
    path.write_text("not json")

    assert dev.DevState.read(path) is None


def test_legacy_cleanup_only_accepts_a_validated_project_listener() -> None:
    root = str(dev.REPO_ROOT)
    processes = {
        100: (1, 90, "uv run patron serve --port 8000"),
        101: (100, 90, f"{root}/.venv/bin/patron serve --port 8000"),
        102: (101, 90, "python worker"),
        200: (1, 200, "/usr/bin/python -m http.server 8000"),
    }

    assert dev._legacy_family(101, processes) == {100, 101, 102}
    assert dev._legacy_family(200, processes) == set()
    assert dev._orphaned_legacy_families(processes) == [{100, 101, 102}]


def test_active_project_vite_owned_by_a_terminal_is_not_called_an_orphan() -> None:
    root = str(dev.REPO_ROOT)
    processes = {
        300: (250, 300, "npm run dev"),
        301: (300, 300, f"node {root}/web/node_modules/.bin/vite"),
        250: (10, 250, "interactive development shell"),
    }

    assert dev._orphaned_legacy_families(processes) == []


def test_recorded_process_group_is_terminated(monkeypatch) -> None:
    monkeypatch.setattr(dev, "SHUTDOWN_GRACE_SECONDS", 1.0)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "patron-dev-test"],
        start_new_session=True,
    )
    record = dev.ChildRecord(
        name="test",
        pid=process.pid,
        pgid=os.getpgid(process.pid),
        marker="patron-dev-test",
    )

    try:
        dev._terminate_groups([record], output=io.StringIO())
        assert process.wait(timeout=2) < 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
