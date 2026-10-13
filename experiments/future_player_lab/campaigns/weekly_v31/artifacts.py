"""Checksummed, atomically published research artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC)


def timestamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp requires an explicit timezone")
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seal(directory, metadata):
    directory = Path(directory)
    manifest = dict(
        metadata,
        artifacts={
            str(p.relative_to(directory)): digest(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    )
    write(directory / "manifest.json", manifest)
    (directory / "manifest.sha256").write_text(digest(directory / "manifest.json") + "\n")
    return manifest


def verify(directory):
    directory = Path(directory)
    if digest(directory / "manifest.json") != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p.name not in ("manifest.json", "manifest.sha256")
    }
    if actual != set(manifest["artifacts"]):
        raise ValueError("Sealed artifact inventory changed")
    for name, expected in manifest["artifacts"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return manifest


def publish_directory(staging, destination):
    """Exclusive lock prevents concurrent writers from replacing a sealed origin."""
    destination = Path(destination)
    lock = destination.with_name(destination.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        os.rename(staging, destination)
    finally:
        os.close(descriptor)
        lock.unlink()
