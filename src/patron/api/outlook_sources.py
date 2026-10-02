"""Fail-closed, read-only access to separately versioned player outlooks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import HTTPException

from patron.api.research_sources import _inside, accepted_version, digest
from patron.config.settings import get_settings
from patron.data.releases import product_reference, verify_product_gold


def load_outlook_release(
    season: int, *, reference_override: dict | None = None
) -> tuple[Path, dict]:
    settings = get_settings()
    pointer_path = settings.outputs_dir / f"player_outlook_{season}.json"
    if (
        not pointer_path.exists()
        and reference_override is None
        and not (settings.data_dir / "current.json").exists()
    ):
        raise HTTPException(404, "Player outlook has not been built for this season yet.")
    try:
        pointer = reference_override or product_reference(
            settings.data_dir, f"outlook_{season}", pointer_path
        )
        name = pointer["version"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", name):
            raise ValueError("Invalid outlook version")
        version = _inside(settings.data_dir / "research", name)
        manifest_path = version / "manifest.json"
        if digest(manifest_path) != pointer["manifest_sha256"]:
            raise ValueError("Outlook manifest changed")
        manifest = json.loads(manifest_path.read_text())
        verify_product_gold(settings.data_dir, manifest)
        if (
            manifest["version"] != name
            or manifest["status"] != "complete"
            or manifest["protected_artifacts_unchanged"] is not True
        ):
            raise ValueError("Outlook build did not complete")
        required_inputs = {
            f"inputs/current_{key}.parquet" for key in ("weeks", "snaps", "schedule", "players")
        }
        required_inputs |= {"inputs/runner.py", "inputs/model.py"}
        required_outputs = {
            "outlook.json",
            "historical_features.parquet",
            "validation_predictions.parquet",
            "current_research_predictions.parquet",
        }
        if not required_inputs <= set(manifest["source_sha256"]) or not required_outputs <= set(
            manifest["output_sha256"]
        ):
            raise ValueError("Outlook provenance is incomplete")
        for file, expected in {**manifest["source_sha256"], **manifest["output_sha256"]}.items():
            if digest(_inside(version, file)) != expected:
                raise ValueError(f"Outlook artifact changed: {file}")
        history_path, acceptance, _ = accepted_version(manifest["history"]["version"])
        if (
            acceptance["audited_input"] != manifest["history"]
            or _inside(settings.data_dir, manifest["history_path"]) != history_path.resolve()
        ):
            raise ValueError("Outlook does not match accepted history")
        report = json.loads((version / "outlook.json").read_text())
        if (
            report["version"] != name
            or report["season"] != season
            or report["history"] != manifest["history"]
            or report["confidence_scores_published"] is not False
            or report.get("gold") != manifest.get("gold")
        ):
            raise ValueError("Outlook report disagrees with its manifest/policy")
        if any(r.get("confidence_score") is not None for r in report["players"]):
            raise ValueError("Unapproved confidence score in research payload")
        return version, report
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(409, f"Player outlook is unavailable: {exc}") from exc


def load_outlook(season: int) -> dict:
    return load_outlook_release(season)[1]
