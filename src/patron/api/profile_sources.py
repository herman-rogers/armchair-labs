"""Verified access to the pinned player-profile release."""

import re
from pathlib import Path

from fastapi import HTTPException

from patron.api.research_sources import _inside, accepted_version, digest
from patron.config.settings import get_settings
from patron.data.releases import product_reference, verify_product_gold
from patron.data.verification import observe, read_json, verified

REQUIRED = {
    "players.parquet",
    "nfl_weeks.parquet",
    "profile_features.parquet",
    "injuries.parquet",
    "report.json",
}


def load_profiles(*, reference_override: dict | None = None) -> tuple[Path, dict]:
    settings = get_settings()
    key = tuple(sorted(reference_override.items())) if reference_override else None
    return verified(
        ("profiles", str(settings.data_dir.absolute()), str(settings.outputs_dir.absolute()), key),
        lambda: _load_profiles(reference_override=reference_override),
    )


def _load_profiles(*, reference_override: dict | None = None) -> tuple[Path, dict]:
    settings = get_settings()
    pointer = settings.outputs_dir / "player_profiles.json"
    observe(pointer)
    observe(settings.data_dir / "current.json")
    if (
        not pointer.exists()
        and reference_override is None
        and not (settings.data_dir / "current.json").exists()
    ):
        raise HTTPException(404, "Player profiles have not been built yet")
    try:
        reference = reference_override or product_reference(settings.data_dir, "profiles", pointer)
        name = reference["version"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name):
            raise ValueError("Invalid profile version")
        root = _inside(settings.data_dir / "research", name)
        if digest(root / "manifest.json") != reference["manifest_sha256"]:
            raise ValueError("Profile manifest changed")
        manifest = read_json(root / "manifest.json")
        verify_product_gold(settings.data_dir, manifest)
        if (
            manifest["version"] != name
            or manifest["kind"] != "player_profiles"
            or manifest["status"] != "complete"
            or manifest["protected_artifacts_unchanged"] is not True
        ):
            raise ValueError("Profile release did not complete")
        if not set(manifest["output_sha256"]) >= REQUIRED or not manifest["source_sha256"]:
            raise ValueError("Incomplete profile provenance")
        if (
            manifest.get("schema_version") == 2
            and "tracking_seasons.parquet" not in manifest["output_sha256"]
        ):
            raise ValueError("Missing profile tracking summaries")
        for file, expected in manifest["output_sha256"].items():
            if digest(_inside(root, file)) != expected:
                raise ValueError(f"Profile artifact changed: {file}")
        for file, expected in manifest["source_sha256"].items():
            if digest(_inside(settings.data_dir, file)) != expected:
                raise ValueError(f"Profile source changed: {file}")
        for file, expected in manifest["implementation_sha256"].items():
            if digest(_inside(root / "implementation", file)) != expected:
                raise ValueError("Profile implementation snapshot changed")
        _, acceptance, _ = accepted_version(manifest["history"]["version"])
        if acceptance["audited_input"] != manifest["history"]:
            raise ValueError("Profile historical acceptance changed")
        report = read_json(root / "report.json")
        if (
            report["version"] != name
            or report["history"] != manifest["history"]
            or report["schema_version"] not in {1, 2}
            or report["schema_version"] != manifest.get("schema_version", 1)
            or report.get("gold") != manifest.get("gold")
        ):
            raise ValueError("Profile report and manifest disagree")
        return root, report
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"Player profiles unavailable: {exc}") from exc
