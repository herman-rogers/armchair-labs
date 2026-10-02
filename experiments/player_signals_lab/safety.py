"""An accident barrier for this offline lab, not an OS security sandbox."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(paths: list[Path], repo: Path, *, missing_ok=False) -> dict:
    return {
        str(p.relative_to(repo)): None if missing_ok and not p.exists() else sha256(p)
        for p in sorted(set(paths))
    }


def verify_source_pins(repo: Path, pins: dict[str, str]) -> None:
    required = {
        "src/patron/__init__.py",
        "src/patron/data/__init__.py",
        "src/patron/metrics/__init__.py",
        "src/patron/data/releases.py",
        "src/patron/metrics/nextgen.py",
    }
    if set(pins) != required:
        raise ValueError("Config must pin the exact set of imported Patron source modules")
    for name, expected in pins.items():
        if sha256(repo / name) != expected:
            raise ValueError(
                f"Pinned source changed: {name}; review it before creating a new protocol"
            )


def new_run(lab: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Run ID must contain only letters, digits, underscores, or hyphens")
    lab = lab.resolve()
    root = lab / "runs"
    if root.is_symlink():
        raise ValueError("The runs directory must not be a symlink")
    root.mkdir(exist_ok=True)
    destination = root / name
    # Never overwrite an earlier trial, including failed trials.
    destination.mkdir(exist_ok=False)
    return destination


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def install_write_guard(destination: Path) -> None:
    """Guard Python writes, links, subprocesses, and network calls for this process.

    Native extension file IO is not universally audited by Python. Our Parquet
    writers receive only paths under destination; protected inputs are hashed
    again at completion. This hook is irreversible within the runner process.
    """
    destination = destination.resolve()

    def check(path) -> None:
        if isinstance(path, int):
            # Permit stdout/stderr only; no arbitrary inherited writable FDs.
            if path not in (1, 2):
                raise PermissionError("Lab forbids writes through inherited file descriptors")
            return
        resolved = Path(os.fsdecode(path)).resolve()
        if not resolved.is_relative_to(destination):
            raise PermissionError(f"Lab write escaped this run: {resolved}")
        if resolved.exists() and resolved.is_file() and resolved.stat().st_nlink != 1:
            raise PermissionError("Lab forbids writing hard-linked files")

    def audit(event, args) -> None:
        if event == "open":
            path, mode, flags = args
            if (mode and any(c in mode for c in "wax+")) or (
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            ):
                check(path)
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime", "os.truncate"}:
            check(args[0])
            # Relative paths with a directory FD can bypass normal resolution.
            if event in {"os.mkdir", "os.chmod"}:
                directory_fd = args[2] if len(args) > 2 else -1
            elif event in {"os.remove", "os.rmdir"}:
                directory_fd = args[1] if len(args) > 1 else -1
            else:
                directory_fd = -1
            if directory_fd not in (None, -1):
                raise PermissionError("Lab forbids directory-FD mutations")
        elif event == "os.rename":
            check(args[0])
            check(args[1])
            if any(fd not in (None, -1) for fd in args[2:]):
                raise PermissionError("Lab forbids directory-FD mutations")
        elif event in {"os.link", "os.symlink"}:
            raise PermissionError("Lab forbids creating links")
        elif event in {
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
            "os.fork",
            "os.exec",
            "socket.connect",
            "socket.bind",
            "socket.sendto",
        }:
            raise PermissionError(f"Offline lab forbids {event}")

    sys.addaudithook(audit)
