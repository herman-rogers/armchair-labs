"""Publish one consistent data/product catalog after all artifacts pass validation."""

from __future__ import annotations

import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path

from patron.data.releases import (
    atomic_json,
    current_catalog,
    digest,
    identifier,
    load_gold,
    reference,
)


def publish_catalog(
    data_dir: Path,
    gold_version: str,
    products: dict[str, str],
    *,
    expected_catalog_sha256: str | None = None,
) -> dict:
    (data_dir / ".runtime").mkdir(parents=True, exist_ok=True)
    with (data_dir / ".runtime/catalog_publish.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if expected_catalog_sha256 is not None and (
            digest(data_dir / "current.json") != expected_catalog_sha256
        ):
            raise ValueError("Catalog changed during rebuild; staged results were not published")
        return _publish_catalog(data_dir, gold_version, products)


def _publish_catalog(data_dir: Path, gold_version: str, products: dict[str, str]) -> dict:
    from patron.api.college_sources import load_college
    from patron.api.outlook_sources import load_outlook_release
    from patron.api.profile_sources import load_profiles

    gold = load_gold(data_dir, gold_version)
    season = gold.manifest["current_observations"]["season"]
    required = {"college", "profiles", f"outlook_{season}"}
    existing = current_catalog(data_dir)
    if "analysis" in products:
        required.add("analysis")
    elif existing and "analysis" in existing.get("products", {}):
        raise ValueError("A NextGen catalog cannot silently drop its analysis policy")
    if set(products) != required:
        raise ValueError(f"Publication requires exactly {sorted(required)}")
    refs = {}
    for key, version in products.items():
        root = data_dir / "research" / identifier(version)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("gold") != gold.ref or manifest["history"] != gold.manifest["history"]:
            raise ValueError(f"Mixed data releases: {key}")
        refs[key] = reference(root)
    _, college = load_college(reference_override=refs["college"])
    _, outlook = load_outlook_release(season, reference_override=refs[f"outlook_{season}"])
    _, profiles = load_profiles(reference_override=refs["profiles"])
    if "analysis" in refs:
        from patron.data.nextgen import load_analysis

        _, analysis = load_analysis(data_dir, refs["analysis"])
        if analysis["gold"] != gold.ref or analysis["profiles"] != refs["profiles"]:
            raise ValueError("NextGen policy and profiles use different dependencies")
        if existing and "analysis" in existing.get("products", {}):
            prior_ref = existing["products"]["analysis"]
            prior_manifest_path = (
                data_dir / "research" / identifier(prior_ref["version"]) / "manifest.json"
            )
            if digest(prior_manifest_path) != prior_ref["manifest_sha256"]:
                raise ValueError("Previously published analysis manifest changed")
            prior_analysis = json.loads(prior_manifest_path.read_text())
            if prior_analysis.get("ranking_schema_version") and not analysis.get(
                "ranking_schema_version"
            ):
                raise ValueError("A ranked NextGen catalog cannot silently drop its rankings")
            if prior_analysis.get("qb_passing_schema_version") and not analysis.get(
                "qb_passing_schema_version"
            ):
                raise ValueError("A NextGen catalog cannot silently drop its QB passing forecasts")
            if prior_analysis.get("qb_variations_schema_version") and not analysis.get(
                "qb_variations_schema_version"
            ):
                raise ValueError("A NextGen catalog cannot silently drop its QB variation policies")
    if (
        profiles["college_version"] != products["college"]
        or profiles["outlook_version"] != products[f"outlook_{season}"]
        or profiles["through_week"] != outlook["through_week"]
        or outlook["through_week"] != gold.manifest["current_observations"]["through_week"]
        or any(report.get("gold") != gold.ref for report in (college, outlook, profiles))
    ):
        raise ValueError("Product reports disagree on data dependencies or observation cutoff")
    catalog = {
        "schema_version": 1,
        "published_at": datetime.now(UTC).isoformat(),
        "gold": gold.ref,
        "historical_research_reference": gold.manifest["history"],
        "products": refs,
        "previous_catalog_sha256": (
            digest(data_dir / "current.json") if (data_dir / "current.json").exists() else None
        ),
    }
    # Keep the prior pointer itself for reproducible rollbacks. Publication updates
    # exactly one file, and happens only after all data/product checks above pass.
    if (data_dir / "current.json").exists():
        archive = data_dir / "catalog_history" / (catalog["previous_catalog_sha256"] + ".json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_bytes((data_dir / "current.json").read_bytes())
    atomic_json(data_dir / "current.json", catalog)
    return catalog


def catalog_info(data_dir: Path) -> dict:
    catalog = current_catalog(data_dir)
    if catalog is None:
        return {"available": False, "gold": None}
    gold = load_gold(data_dir)
    if "analysis" in catalog.get("products", {}):
        from patron.api.profile_sources import load_profiles
        from patron.data.nextgen import load_analysis

        load_analysis(data_dir)
        load_profiles()
    return {
        "available": True,
        "gold": catalog["gold"],
        "published_at": catalog["published_at"],
        "layers": ["raw", "enriched", "gold"],
        "products": catalog["products"],
        "historical_research_reference": catalog["historical_research_reference"],
        "current_observations": gold.manifest["current_observations"],
        "tables": [
            {"name": name, **{k: spec[k] for k in ("rows", "primary_key", "role", "description")}}
            for name, spec in gold.manifest["tables"].items()
        ],
        "coverage": json.loads((gold.root / "coverage.json").read_text()),
        "quality": json.loads((gold.root / "quality.json").read_text()),
        "limitations": gold.manifest["limitations"],
    }
