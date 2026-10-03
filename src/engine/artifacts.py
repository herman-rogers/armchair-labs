"""Immutable draft reference and content-addressed publication provenance."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write(path: Path, payload: str | bytes) -> None:
    """Readers see one complete publication even while a replacement is being written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload.encode() if isinstance(payload, str) else payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_draft(directory: Path) -> Path:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Frozen draft dependency changed: {name}")
    return directory / "board_v1.json"


def publish_draft(directory: Path, outputs: Path) -> None:
    payload = verify_draft(directory).read_bytes()
    outputs.mkdir(parents=True, exist_ok=True)
    for name in ("board.json", "board_v1.json"):
        atomic_write(outputs / name, payload)


def publish_provenance(board: Path, dependencies: list[Path]) -> None:
    files = {str(path.resolve()): digest(path) for path in dependencies if path.exists()}
    rows = json.loads(board.read_text())
    unavailable = sum(r.get("forecast_status") == "unavailable" for r in rows)
    record: dict[str, Any] = {
        "schema_version": 1,
        "unavailable_forecasts": unavailable,
        "board_sha256": digest(board),
        "dependencies": files,
    }
    record["artifact_id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    atomic_write(board.with_suffix(".manifest.json"), json.dumps(record, indent=2))


def artifact_status(board: Path) -> dict[str, Any]:
    archive_manifest = board.parent / "manifest.json"
    if archive_manifest.exists() and board.name == "board_v1.json":
        archive = json.loads(archive_manifest.read_text())
        if archive.get("role") == "immutable_draft_reference":
            verify_draft(board.parent)
            return {"state": "frozen_reference", "artifact_id": digest(archive_manifest)}
    manifest = board.with_suffix(".manifest.json")
    if not manifest.exists():
        return {"state": "unverified", "artifact_id": None}
    record = json.loads(manifest.read_text())
    valid = board.exists() and digest(board) == record["board_sha256"]
    valid = valid and all(
        Path(p).exists() and digest(Path(p)) == sha for p, sha in record["dependencies"].items()
    )
    return {
        "state": ("degraded" if record.get("unavailable_forecasts") else "current")
        if valid
        else "stale",
        "artifact_id": record["artifact_id"],
    }
