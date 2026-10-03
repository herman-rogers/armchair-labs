"""Verified college artifacts shared by pathways and player profiles."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException

from engine.api.research_sources import _inside, accepted_version, digest
from engine.config.settings import get_settings
from engine.data.releases import product_reference, verify_product_gold
from engine.data.verification import observe, read_json, verified

OUTPUTS = {
    "report.json",
    "college_seasons.parquet",
    "college_games.parquet",
    "college_annual.parquet",
    "identity_links.parquet",
    "identity_registry.parquet",
    "nfl_cohort.parquet",
    "nfl_predictions.parquet",
    "college_predictions.parquet",
}
INPUTS = {
    f"inputs/{name}"
    for name in [
        "runner.py",
        "college.py",
        "college_identity.py",
        "college_translation.py",
        "identity_overrides.json",
        "college_source_manifest.json",
        "nfl_identities.parquet",
        "nfl_crosswalk.parquet",
        "nfl_candidates.parquet",
        "nfl_seasons.parquet",
    ]
}


def load_college(*, reference_override: dict | None = None) -> tuple[Path, dict]:
    settings = get_settings()
    key = tuple(sorted(reference_override.items())) if reference_override else None
    return verified(
        ("college", str(settings.data_dir.absolute()), str(settings.outputs_dir.absolute()), key),
        lambda: _load_college(reference_override=reference_override),
    )


def _load_college(*, reference_override: dict | None = None) -> tuple[Path, dict]:
    settings = get_settings()
    pointer = settings.outputs_dir / "college_nfl.json"
    observe(pointer)
    observe(settings.data_dir / "current.json")
    if (
        not pointer.exists()
        and reference_override is None
        and not (settings.data_dir / "current.json").exists()
    ):
        raise HTTPException(404, "College pathways have not been built yet")
    try:
        ref = reference_override or product_reference(settings.data_dir, "college", pointer)
        name = ref["version"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name):
            raise ValueError("Invalid college version")
        root = _inside(settings.data_dir / "research", name)
        manifest_path = root / "manifest.json"
        if digest(manifest_path) != ref["manifest_sha256"]:
            raise ValueError("College manifest changed")
        manifest = read_json(manifest_path)
        verify_product_gold(settings.data_dir, manifest)
        if (
            manifest["version"] != name
            or manifest["kind"] != "college_nfl_research"
            or manifest["status"] != "complete"
            or manifest["protected_artifacts_unchanged"] is not True
        ):
            raise ValueError("College release did not complete")
        if (
            not set(manifest["output_sha256"]) >= OUTPUTS
            or not set(manifest["source_sha256"]) >= INPUTS
        ):
            raise ValueError("College provenance incomplete")
        for path, expected in {**manifest["source_sha256"], **manifest["output_sha256"]}.items():
            if digest(_inside(root, path)) != expected:
                raise ValueError(f"College artifact changed: {path}")
        source_root = _inside(settings.data_dir / "research", manifest["source_version"])
        if digest(source_root / "manifest.json") != manifest["source_manifest_sha256"]:
            raise ValueError("College source manifest changed")
        source = read_json(source_root / "manifest.json")
        if source["version"] != manifest["source_version"] or source["status"] != "captured":
            raise ValueError("College source is incomplete")
        for spec in source["files"]:
            if digest(_inside(source_root / "raw", spec["filename"])) != spec["sha256"]:
                raise ValueError(f"College raw source changed: {spec['filename']}")
        _, acceptance, _ = accepted_version(manifest["history"]["version"])
        if acceptance["audited_input"] != manifest["history"]:
            raise ValueError("College NFL history no longer matches accepted history")
        report = read_json(root / "report.json")
        if (
            report["version"] != name
            or report["status"] != "experimental_not_promoted"
            or report["history"] != manifest["history"]
            or report.get("gold") != manifest.get("gold")
        ):
            raise ValueError("College report policy/provenance mismatch")
        return root, report
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"College pathways unavailable: {exc}") from exc
