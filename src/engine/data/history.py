"""Locate the accepted historical research archive that seeds a raw snapshot.

The archive is the preserved nflverse cache plus audited replay references. It is
an ingestion input only; nothing in the API reads it.
"""

from __future__ import annotations

import re
from pathlib import Path

from engine.config.settings import get_settings
from engine.data.releases import digest, inside
from engine.data.verification import read_json


def accepted_version(dataset: str) -> tuple[Path, dict, dict]:
    """Fail closed; never substitute live data for a broken selected archive."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", dataset):
        raise ValueError("Invalid research dataset identifier")
    root = get_settings().data_dir / "research"
    version = inside(root, dataset)
    acceptance = read_json(version / "acceptance.json")
    manifest = read_json(version / "manifest.json")
    if (
        acceptance.get("status") != "accepted_for_research_not_promoted"
        or acceptance.get("version") != dataset
        or manifest.get("version") != dataset
        or manifest.get("status") != "rebuilt_pending_audit"
        or not manifest.get("protected_unchanged")
    ):
        raise ValueError("Dataset has not completed research acceptance")
    provenance = acceptance["audited_input"]
    if digest(version / "manifest.json") != provenance["manifest_sha256"]:
        raise ValueError("Build manifest changed after acceptance")
    audits = [
        path
        for path in (version / "outputs").glob("data_integrity_audit_*.json")
        if digest(path) == provenance["audit_sha256"]
    ]
    if len(audits) != 1:
        raise ValueError("Accepted integrity audit is missing or changed")
    audit = read_json(audits[0])
    checks = audit.get("acceptance_checks")
    if (
        audit.get("accepted") is not True
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("Integrity audit did not pass")
    if acceptance.get("acceptance_checks") != checks:
        raise ValueError("Acceptance and integrity checks disagree")
    if not manifest["source_sha256"]:
        raise ValueError("Accepted source fingerprints are incomplete")
    for name, expected in {
        **manifest["source_sha256"],
        **manifest["output_sha256"],
        **acceptance.get("research_output_sha256", {}),
    }.items():
        if digest(inside(version, name)) != expected:
            raise ValueError(f"Accepted artifact changed: {name}")
    return version, acceptance, manifest
