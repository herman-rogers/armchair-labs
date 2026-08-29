"""A project-scoped supervisor for the API and frontend development servers.

The old shell recipe relied on ``trap 'kill 0'``. That is fragile around nested
launchers (uv, uvicorn reload, npm, Vite): after the shell exits, a grandchild can be
adopted by PID 1 and keep its port forever. This supervisor gives each service its own
process group, persists the exact group IDs, and owns cleanup on INT, TERM, HUP,
normal exit, child failure, and the next restart after an unclean SIGKILL.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from types import FrameType
from typing import Any, TextIO

from patron.config.settings import REPO_ROOT, get_settings

SHUTDOWN_GRACE_SECONDS = 5.0
POLL_SECONDS = 0.1
STATE_NAME = "dev.json"


class DevAlreadyRunningError(RuntimeError):
    """Raised when a second supervisor would collide with the active one."""


@dataclass(frozen=True)
class ChildRecord:
    name: str
    pid: int
    pgid: int
    marker: str


@dataclass(frozen=True)
class DevState:
    token: str
    supervisor_pid: int
    repo_root: str
    api_port: int
    web_port: int
    children: list[ChildRecord]

    @classmethod
    def read(cls, path: Path) -> DevState | None:
        try:
            raw = json.loads(path.read_text())
            return cls(
                token=str(raw["token"]),
                supervisor_pid=int(raw["supervisor_pid"]),
                repo_root=str(raw["repo_root"]),
                api_port=int(raw["api_port"]),
                web_port=int(raw["web_port"]),
                children=[ChildRecord(**child) for child in raw.get("children", [])],
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2))
        os.replace(temporary, path)


def state_path() -> Path:
    return get_settings().runtime_dir / STATE_NAME


def _process_table() -> dict[int, tuple[int, int, str]]:
    """PID -> (parent PID, process group ID, command), for owned-process checks."""
    try:
        result = subprocess.run(
            ["ps", "-ax", "-o", "pid=,ppid=,pgid=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    processes: dict[int, tuple[int, int, str]] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4 or not all(value.isdigit() for value in fields[:3]):
            continue
        pid, parent, group = (int(value) for value in fields[:3])
        processes[pid] = (parent, group, fields[3])
    return processes


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _group_matches(record: ChildRecord, processes: dict[int, tuple[int, int, str]]) -> bool:
    """Guard against signaling a recycled process group from an old state file."""
    return any(
        group == record.pgid and record.marker in command
        for _, group, command in processes.values()
    )


def _group_alive(pgid: int) -> bool:
    """Whether a process group has at least one non-zombie member."""
    try:
        result = subprocess.run(
            ["ps", "-ax", "-o", "pgid=,stat="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            fields = line.strip().split(maxsplit=1)
            if (
                len(fields) == 2
                and fields[0].isdigit()
                and int(fields[0]) == pgid
                and not fields[1].startswith("Z")
            ):
                return True
        return False
    except (OSError, subprocess.SubprocessError):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def _terminate_groups(records: list[ChildRecord], output: TextIO = sys.stdout) -> None:
    processes = _process_table()
    owned = [record for record in records if _group_matches(record, processes)]
    for record in owned:
        with contextlib.suppress(OSError):
            os.killpg(record.pgid, signal.SIGTERM)

    deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS
    while time.monotonic() < deadline and any(_group_alive(record.pgid) for record in owned):
        time.sleep(POLL_SECONDS)

    for record in owned:
        if _group_alive(record.pgid):
            print(
                f"{record.name} ignored SIGTERM; forcing process group {record.pgid}",
                file=output,
            )
            with contextlib.suppress(OSError):
                os.killpg(record.pgid, signal.SIGKILL)


def _listening_pids(port: int) -> list[int]:
    lsof = shutil.which("lsof")
    if lsof is None:
        return []
    try:
        result = subprocess.run(
            [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(value) for value in result.stdout.split() if value.isdigit()]


def _descendants(root: int, processes: dict[int, tuple[int, int, str]]) -> set[int]:
    family = {root}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _, _) in processes.items():
            if parent in family and pid not in family:
                family.add(pid)
                changed = True
    return family


def _legacy_family(listener: int, processes: dict[int, tuple[int, int, str]]) -> set[int]:
    """A validated project listener plus its orphaned launcher and descendants."""
    row = processes.get(listener)
    if row is None:
        return set()
    command = row[2]
    root_text = str(REPO_ROOT)
    expected = root_text in command and (
        "patron serve" in command or "node_modules/.bin/vite" in command
    )
    if not expected:
        return set()

    root = listener
    parent = row[0]
    # Include launcher wrappers such as `uv run` and `npm run dev`, but never walk
    # through PID 1 or into an interactive shell that owns other user processes.
    while parent > 1:
        parent_row = processes.get(parent)
        if parent_row is None:
            break
        parent_command = parent_row[2]
        if root_text not in parent_command and not (
            parent_command.startswith("uv run patron serve")
            or parent_command.startswith("npm run dev")
        ):
            break
        root = parent
        parent = parent_row[0]
    return _descendants(root, processes)


def _orphaned_legacy_families(
    processes: dict[int, tuple[int, int, str]],
) -> list[set[int]]:
    """Validated project launch trees whose top-level wrapper was adopted by PID 1."""
    families: list[set[int]] = []
    root_text = str(REPO_ROOT)
    for pid, (_, _, command) in processes.items():
        if root_text not in command or not (
            "patron serve" in command or "node_modules/.bin/vite" in command
        ):
            continue
        family = _legacy_family(pid, processes)
        roots = [member for member in family if processes[member][0] not in family]
        if family and roots and processes[roots[0]][0] == 1:
            families.append(family)
    return families


def _terminate_legacy_ports(ports: tuple[int, int], output: TextIO = sys.stdout) -> int:
    """Reclaim validated port conflicts and old project trees adopted by PID 1."""
    processes = _process_table()
    families = _orphaned_legacy_families(processes)
    for port in ports:
        for listener in _listening_pids(port):
            family = _legacy_family(listener, processes)
            if not family:
                command = processes.get(listener, (0, 0, "unknown"))[2]
                raise DevAlreadyRunningError(
                    f"Port {port} is occupied by a process this project does not own: "
                    f"PID {listener} ({command})."
                )
            families.append(family)

    targets = set().union(*families) if families else set()
    if targets:
        print(f"Reclaiming {len(targets)} orphaned project process(es).", file=output)
    for pid in sorted(targets, reverse=True):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)

    deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS
    while time.monotonic() < deadline and any(_pid_alive(pid) for pid in targets):
        time.sleep(POLL_SECONDS)
    for pid in targets:
        if _pid_alive(pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    return len(targets)


def stop(*, include_legacy: bool = True, output: TextIO = sys.stdout) -> bool:
    """Stop the recorded supervisor and children; optionally reclaim legacy listeners."""
    path = state_path()
    state = DevState.read(path)
    stopped = False
    if state is not None and state.repo_root == str(REPO_ROOT):
        supervisor_signaled = False
        if state.supervisor_pid != os.getpid() and _pid_alive(state.supervisor_pid):
            command = _process_table().get(state.supervisor_pid, (0, 0, ""))[2]
            if "patron dev" in command:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(state.supervisor_pid, signal.SIGTERM)
                    supervisor_signaled = True
                    stopped = True
        if supervisor_signaled:
            # Let the owner shut its children down and exit cleanly. The group pass
            # below remains a fallback if the supervisor is wedged or killed midway.
            deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS
            while time.monotonic() < deadline and _pid_alive(state.supervisor_pid):
                time.sleep(POLL_SECONDS)
        _terminate_groups(state.children, output=output)
        stopped = stopped or bool(state.children)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()

    if include_legacy:
        stopped = _terminate_legacy_ports((8000, 5173), output=output) > 0 or stopped
    return stopped


def status(output: TextIO = sys.stdout) -> bool:
    state = DevState.read(state_path())
    if state is None:
        print("Patron dev services are not recorded as running.", file=output)
        return False
    processes = _process_table()
    running = [record.name for record in state.children if _group_matches(record, processes)]
    if not running:
        print("Patron dev state is stale; run `just restart`.", file=output)
        return False
    print(
        f"Patron dev supervisor PID {state.supervisor_pid}; running: {', '.join(running)} "
        f"(API :{state.api_port}, web :{state.web_port}).",
        file=output,
    )
    return True


def run(
    *,
    restart: bool = False,
    api_host: str = "127.0.0.1",
    api_port: int = 8000,
    web_host: str = "127.0.0.1",
    web_port: int = 5173,
    reload: bool = True,
    output: TextIO = sys.stdout,
) -> int:
    """Run API and Vite in the foreground until interrupted or either child fails."""
    path = state_path()
    existing = DevState.read(path)
    if restart:
        stop(include_legacy=True, output=output)
    elif existing is not None:
        processes = _process_table()
        if any(_group_matches(record, processes) for record in existing.children):
            raise DevAlreadyRunningError(
                f"Patron dev services are already running under PID {existing.supervisor_pid}. "
                "Use `just restart` to replace them."
            )
        stop(include_legacy=False, output=output)

    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is not installed or not on PATH.")

    token = uuid.uuid4().hex
    environment = {**os.environ, "PATRON_DEV_TOKEN": token, "PYTHONUNBUFFERED": "1"}
    api_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "patron.api.app:app",
        "--host",
        api_host,
        "--port",
        str(api_port),
    ]
    if reload:
        api_command.append("--reload")
    web_command = [
        npm,
        "run",
        "dev",
        "--",
        "--host",
        web_host,
        "--port",
        str(web_port),
        "--strictPort",
    ]

    children: list[tuple[str, subprocess.Popen[bytes], str]] = []
    stop_requested = Event()
    handler_type = Callable[[int, FrameType | None], Any]
    previous_handlers: dict[signal.Signals, int | handler_type | None] = {}

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        stop_requested.set()

    handled = [signal.SIGINT, signal.SIGTERM]
    for optional in (getattr(signal, "SIGHUP", None), getattr(signal, "SIGQUIT", None)):
        if optional is not None:
            handled.append(optional)
    for handled_signal in handled:
        previous_handlers[handled_signal] = signal.getsignal(handled_signal)
        signal.signal(handled_signal, request_stop)

    try:
        api = subprocess.Popen(
            api_command,
            cwd=REPO_ROOT,
            env=environment,
            start_new_session=True,
        )
        children.append(("api", api, "patron.api.app:app"))
        web = subprocess.Popen(
            web_command,
            cwd=REPO_ROOT / "web",
            env=environment,
            start_new_session=True,
        )
        children.append(("web", web, "npm run dev"))

        records = [
            ChildRecord(name=name, pid=process.pid, pgid=os.getpgid(process.pid), marker=marker)
            for name, process, marker in children
        ]
        DevState(
            token=token,
            supervisor_pid=os.getpid(),
            repo_root=str(REPO_ROOT),
            api_port=api_port,
            web_port=web_port,
            children=records,
        ).write(path)
        print(
            f"Patron dev ready: API http://{api_host}:{api_port} · "
            f"web http://{web_host}:{web_port} · Ctrl+C stops both",
            file=output,
        )

        exit_code = 0
        while not stop_requested.wait(POLL_SECONDS):
            failed = next(
                (
                    (name, process.returncode)
                    for name, process, _ in children
                    if process.poll() is not None
                ),
                None,
            )
            if failed is not None:
                name, child_code = failed
                print(
                    f"{name} exited unexpectedly ({child_code}); stopping both services.",
                    file=output,
                )
                exit_code = child_code or 1
                break
        return exit_code
    finally:
        records = [
            ChildRecord(name=name, pid=process.pid, pgid=process.pid, marker=marker)
            for name, process, marker in children
        ]
        _terminate_groups(records, output=output)
        for _, process, _ in children:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=0.5)
        current = DevState.read(path)
        if current is not None and current.token == token:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        for handled_signal, previous in previous_handlers.items():
            signal.signal(handled_signal, previous)
