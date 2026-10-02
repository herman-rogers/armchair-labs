"""Fail-closed dependency checks for research on rebuilt historical artifacts."""

import json
from pathlib import Path

from data_integrity_audit import digest


def require_audited_version(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Research requires an explicit versioned rebuild, not legacy live outputs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "rebuilt_pending_audit" or not manifest.get("protected_unchanged"):
        raise ValueError("Rebuild did not complete with protected artifacts unchanged")
    audit_path = data_dir / "outputs/data_integrity_audit_2026-09-22.json"
    audit = json.loads(audit_path.read_text())
    if (
        not audit.get("accepted")
        or not audit.get("acceptance_checks")
        or not all(audit["acceptance_checks"].values())
    ):
        raise ValueError("Integrity audit has not accepted this rebuild")
    prediction_path = data_dir / "outputs/metric_backtest_predictions.parquet"
    if digest(prediction_path) != audit["prediction_sha256"]:
        raise ValueError("Forecasts changed after the integrity audit")
    if (
        digest(data_dir / "outputs/research_audited_weekly_points.parquet")
        != audit["weekly_points_sha256"]
    ):
        raise ValueError("Weekly outcomes changed after the integrity audit")
    for name, expected in manifest["source_sha256"].items():
        if digest(data_dir / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    return {
        "version": manifest["version"],
        "manifest_sha256": digest(manifest_path),
        "audit_sha256": digest(audit_path),
        "prediction_sha256": digest(prediction_path),
    }
