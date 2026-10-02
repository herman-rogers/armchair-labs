"""Verified NextGen releases and the shared, fail-closed serving policy."""

from __future__ import annotations

from pathlib import Path

from patron.data.releases import current_catalog, digest, identifier, inside, load_gold
from patron.data.verification import read_json, verified

REQUIRED = {
    "registry.json",
    "evaluations.json",
    "players.parquet",
    "forecasts.parquet",
    "predictions.parquet",
    "incidents.json",
    "report.json",
    "protocol.md",
}


def verify_evidence(data: Path, ref: dict) -> tuple[Path, dict]:
    root = data / "research" / identifier(ref["version"])
    if digest(root / "manifest.json") != ref["manifest_sha256"]:
        raise ValueError("Evidence manifest changed")
    manifest = read_json(root / "manifest.json")
    if manifest.get("status") != "completed_research_only":
        raise ValueError("Evidence run is not complete")
    for name, expected in manifest["files"].items():
        if digest(inside(root, name)) != expected:
            raise ValueError(f"Evidence changed: {name}")
    return root, manifest


def load_analysis(data: Path, ref: dict | None = None) -> tuple[Path, dict]:
    key = tuple(sorted(ref.items())) if ref else None
    return verified(("analysis", str(data.absolute()), key), lambda: _load_analysis(data, ref))


def _load_analysis(data: Path, ref: dict | None = None) -> tuple[Path, dict]:
    catalog = current_catalog(data) if ref is None else None
    if ref is None:
        if not catalog or "analysis" not in catalog["products"]:
            raise ValueError("NextGen has not been published")
        ref = catalog["products"]["analysis"]
    root = data / "research" / identifier(ref["version"])
    if digest(root / "manifest.json") != ref["manifest_sha256"]:
        raise ValueError("NextGen manifest changed")
    manifest = read_json(root / "manifest.json")
    if manifest.get("kind") != "nextgen_analysis" or manifest.get("status") != "complete":
        raise ValueError("NextGen release is incomplete")
    if not set(manifest["files"]) >= REQUIRED:
        raise ValueError("NextGen release is missing required artifacts")
    if manifest.get("ranking_schema_version") not in (None, 1):
        raise ValueError("Unsupported NextGen ranking schema")
    if manifest.get("ranking_schema_version") == 1 and not set(manifest["files"]) >= {
        "rankings.parquet",
        "ranking_predictions.parquet",
        "ranking_panel.parquet",
        "ranking_evaluations.json",
        "ranking_folds.json",
        "ranking_report.json",
        "ranking_availability.json",
    }:
        raise ValueError("NextGen ranking release is missing required artifacts")
    if manifest.get("qb_passing_schema_version") not in (None, 1):
        raise ValueError("Unsupported QB passing schema")
    if manifest.get("qb_passing_schema_version") == 1 and not set(manifest["files"]) >= {
        "qb_passing/current.parquet",
        "qb_passing/evaluations.json",
        "qb_passing/report.json",
        "qb_passing/predictions.parquet",
        "qb_passing/panel.parquet",
        "qb_passing/protocol.md",
        "qb_passing/events.json",
        "qb_passing/news.json",
        "qb_passing/folds.json",
    }:
        raise ValueError("QB passing release is missing required artifacts")
    if catalog and manifest["gold"] != catalog["gold"]:
        raise ValueError("NextGen policy and player data use different releases")
    if manifest.get("qb_variations_schema_version") not in (None, 1):
        raise ValueError("Unsupported QB variation schema")
    if manifest.get("qb_variations_schema_version") == 1 and not set(manifest["files"]) >= {
        "qb_variations/manifest.json",
        "qb_variations/protocol.md",
        "qb_variations/report.json",
        "qb_variations/decisions.json",
        "qb_variations/evaluations.json",
        "qb_variations/predictions.parquet",
        "qb_variations/ranking_predictions.parquet",
        "qb_variations/folds.json",
    }:
        raise ValueError("QB variation release is missing required artifacts")
    if catalog and manifest["profiles"] != catalog["products"]["profiles"]:
        raise ValueError("NextGen policy and profiles use different releases")
    profile_root = data / "research" / identifier(manifest["profiles"]["version"])
    if digest(profile_root / "manifest.json") != manifest["profiles"]["manifest_sha256"]:
        raise ValueError("NextGen profile dependency changed")
    profile_manifest = read_json(profile_root / "manifest.json")
    for key, base in (
        ("output_sha256", profile_root),
        ("source_sha256", data),
        ("implementation_sha256", profile_root / "implementation"),
    ):
        for name, expected in profile_manifest.get(key, {}).items():
            if digest(inside(base, name)) != expected:
                raise ValueError(f"NextGen profile dependency changed: {name}")
    gold = load_gold(data, manifest["gold"]["version"])
    if gold.ref != manifest["gold"] or "target_quality.json" not in gold.manifest["files"]:
        raise ValueError("NextGen requires the exact corrected gold release")
    if manifest.get("qb_variations_schema_version") == 1:
        experiment = read_json(root / "qb_variations/manifest.json")
        experiment_report = read_json(root / "qb_variations/report.json")
        observations = gold.manifest["current_observations"]
        if experiment["gold"] != gold.ref or any(
            experiment_report[k] != observations[k] for k in ("season", "through_week")
        ):
            raise ValueError("QB variation policies require the current gold and week cutoff")
        dependency = data / "research" / identifier(experiment["source_analysis"]["version"])
        if digest(dependency / "manifest.json") != experiment["source_analysis"]["manifest_sha256"]:
            raise ValueError("QB variation source analysis changed")
        if read_json(dependency / "manifest.json")["profiles"] != manifest["profiles"]:
            raise ValueError("QB variation policies and profiles use different releases")
    for name, expected in manifest["files"].items():
        if digest(inside(root, name)) != expected:
            raise ValueError(f"NextGen artifact changed: {name}")
    verify_evidence(data, manifest["evidence"])
    return root, manifest


def eligible(
    entry: dict,
    *,
    use: str,
    target: str | None = None,
    position: str | None = None,
    population: str | None = None,
    horizon: str | None = None,
    incidents: list[dict] = (),
) -> tuple[bool, str]:
    """Same decision for selectors, direct requests, exports and calculations."""
    if entry.get("validity") != "verified":
        return False, entry.get("reason", "Input validity requires revalidation")
    if entry.get("serving") not in {"approved", "baseline"}:
        return False, "Excluded from analysis: " + entry.get("serving", "unregistered")
    if use not in entry.get("allowed_uses", []):
        return False, entry.get("reason", "This use has not earned inclusion")
    if target is not None and target != entry.get("target"):
        return False, "Evidence is for a different outcome"
    if position and position != "ALL" and position not in entry.get("positions", []):
        return False, "Evidence does not cover this position"
    if population and population != "all" and population not in entry.get("populations", []):
        return False, "Evidence does not cover this population"
    if horizon is not None and horizon != entry.get("horizon"):
        return False, "Evidence is for a different horizon"
    for incident in incidents:
        if incident.get("status") == "open" and (
            set(entry.get("dependencies", [])) & set(incident.get("dependencies", []))
        ):
            return False, "Suspended: " + incident["reason"]
    return True, entry.get("reason", "Allowed for this declared use")


def resolve(root: Path, item: str, **scope) -> dict:
    entries = read_json(root / "registry.json")
    entry = next((r for r in entries if r["id"] == item), None)
    if entry is None:
        raise ValueError("Unknown analysis item")
    allowed, reason = eligible(entry, incidents=read_json(root / "incidents.json"), **scope)
    if not allowed:
        raise ValueError(reason)
    return entry


def published(data: Path) -> bool:
    catalog = current_catalog(data)
    return bool(catalog and "analysis" in catalog.get("products", {}))
