"""Read-only, verified dataset selection for the Intelligence workspace."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from patron.config.settings import get_settings
from patron.data.releases import current_catalog, load_gold
from patron.data.verification import observe, observe_listing, read_json, verified


def _json(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an artifact object: {path.name}")
    return value


@lru_cache(maxsize=2048)
def _hash_at_revision(path: str, size: int, modified: int, changed: int) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(path: Path) -> str:
    observe(path)
    stat = path.stat()
    return _hash_at_revision(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _inside(root: Path, name: str) -> Path:
    observe(root / name)
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its dataset")
    return path


def accepted_version(dataset: str) -> tuple[Path, dict, dict]:
    return verified(
        ("history", str(get_settings().data_dir.absolute()), dataset),
        lambda: _accepted_version(dataset),
    )


def _accepted_version(dataset: str) -> tuple[Path, dict, dict]:
    """Fail closed; never substitute live/legacy data for a broken selected version."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", dataset):
        raise HTTPException(422, "Invalid research dataset identifier")
    root = get_settings().data_dir / "research"
    try:
        version = _inside(root, dataset)
        acceptance = _json(version / "acceptance.json")
        manifest = _json(version / "manifest.json")
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
        observe_listing(version / "outputs")
        audits = [
            path
            for path in (version / "outputs").glob("data_integrity_audit_*.json")
            if digest(path) == provenance["audit_sha256"]
        ]
        if len(audits) != 1:
            raise ValueError("Accepted integrity audit is missing or changed")
        audit = _json(audits[0])
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
        required_outputs = {
            "outputs/metric_backtest_predictions.parquet",
            "outputs/metric_report.json",
            "outputs/research_audited_weekly_points.parquet",
        }
        if not manifest["source_sha256"] or not required_outputs <= set(manifest["output_sha256"]):
            raise ValueError("Accepted source/output fingerprints are incomplete")
        for name, expected in {
            **manifest["source_sha256"],
            **manifest["output_sha256"],
            **acceptance.get("research_output_sha256", {}),
            "outputs/metric_backtest_predictions.parquet": provenance["prediction_sha256"],
            "outputs/research_audited_weekly_points.parquet": audit["weekly_points_sha256"],
        }.items():
            if digest(_inside(version, name)) != expected:
                raise ValueError(f"Accepted artifact changed: {name}")
        if audit["prediction_sha256"] != provenance["prediction_sha256"]:
            raise ValueError("Forecast provenance disagrees with integrity audit")
        return version, acceptance, manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"Research dataset {dataset} is unavailable: {exc}") from exc


def dataset_info(dataset: str) -> dict[str, Any]:
    if dataset == "legacy":
        outputs = get_settings().outputs_dir
        return {
            "id": dataset,
            "label": "Archived legacy research (before repairs)",
            "accepted": False,
            "accepted_at": None,
            "checks_passed": None,
            "forecast_rows": None,
            "report_available": (outputs / "metric_report.json").exists(),
            "description": "Original saved research; not the repaired historical rebuild. "
            "Selecting it does not change the production board.",
        }
    version, acceptance, manifest = accepted_version(dataset)
    return {
        "id": dataset,
        "label": f"Archived historical research · {dataset}",
        "accepted": True,
        "accepted_at": acceptance["generated_at"],
        "checks_passed": len(acceptance["acceptance_checks"]),
        "forecast_rows": acceptance["forecast_rows"],
        "report_available": (version / "outputs/metric_report.json").exists(),
        "description": "Audited historical reconstruction, accepted for research only. "
        "Unknown historical context stays unknown; provider publication vintages are incomplete. "
        "2026 forecasts are pending and do not replace V1, production, or the frozen experiment.",
        "model_specs": manifest["report_config"]["fit"]["models"],
    }


def public_info(info: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in info.items() if key != "model_specs"}


def list_datasets() -> dict[str, Any]:
    root = get_settings().data_dir / "research"
    datasets, unavailable = [], []
    for path in sorted(root.glob("*/acceptance.json")):
        try:
            datasets.append(public_info(dataset_info(path.parent.name)))
        except HTTPException as exc:
            unavailable.append({"id": path.parent.name, "reason": exc.detail})
    datasets.sort(key=lambda row: (row["accepted_at"], row["id"]), reverse=True)
    if (get_settings().outputs_dir / "metric_backtest_predictions.parquet").exists():
        datasets.append(dataset_info("legacy"))
    default_id = datasets[0]["id"] if datasets else None
    try:
        catalog = current_catalog(get_settings().data_dir)
        if catalog:
            gold = load_gold(get_settings().data_dir)
            default_id = gold.manifest["history"]["version"]
            if not any(row["id"] == default_id and row["accepted"] for row in datasets):
                raise ValueError("Canonical historical research reference is unavailable")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"Canonical data catalog is unavailable: {exc}") from exc
    return {
        "datasets": datasets,
        "default_id": default_id,
        "gold": catalog["gold"] if catalog else None,
        "unavailable": unavailable,
    }


def output_path(dataset: str, filename: str) -> Path:
    if dataset == "legacy":
        outputs = get_settings().outputs_dir
    else:
        version, _, _ = accepted_version(dataset)
        outputs = version / "outputs"
    path = outputs / filename
    if not path.is_file():
        raise HTTPException(404, f"No {filename} is available for research dataset {dataset}")
    return path
