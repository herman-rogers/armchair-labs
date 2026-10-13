"""Organize method definitions, published campaign bundles, and local research inventory.

This is an index, not a serving registry: every campaign bundle is queryable through
the API, and research-run completion is never a selection.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path

from engine.data.shared import sha256 as digest

FAMILIES = Path(__file__).with_name("families.json")
CAMPAIGNS = Path(__file__).with_name("campaigns.json")
LAYERS = {
    "observations",
    "description",
    "measurement_research",
    "development",
    "execution",
    "opportunity",
    "context",
    "production",
    "fantasy_application",
    "benchmark",
    "ability",
}


def read_json(path: Path):
    return json.loads(path.read_text())


def within(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes catalog root: {value}")
    return path


def load_families(path: Path = FAMILIES) -> list[dict]:
    document = read_json(path)
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported model catalog schema")
    rows = document["families"]
    ids = set()
    required = {
        "id",
        "name",
        "layer",
        "method",
        "targets",
        "horizons",
        "sources",
        "assessment",
        "limits",
        "registry_prefixes",
    }
    for row in rows:
        if not required <= row.keys() or row["layer"] not in LAYERS:
            raise ValueError(f"Invalid model family: {row.get('id')}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate model family: {row['id']}")
        ids.add(row["id"])
        # Curated descriptions must not become a parallel hardcoded approval registry.
        if "serving" in row or "approved" in row:
            raise ValueError("Serving status must come from current artifacts")
    return rows


def load_campaigns(path: Path = CAMPAIGNS) -> list[dict]:
    """Descriptive lineage stays separate from executable bundle availability."""
    from engine.chairstats.store import safe_id

    document = read_json(path)
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported campaign provenance schema")
    ids = set()
    for row in document["campaigns"]:
        identity = safe_id(row["campaign_id"])
        if identity in ids:
            raise ValueError("Duplicate campaign provenance identity")
        ids.add(identity)
        if "approved" in row or "serving" in row or "query_ready" in row:
            raise ValueError("Campaign provenance cannot declare serving availability or approval")
        if not row.get("campaign_name"):
            raise ValueError("Campaign provenance requires a display name")
    by_id = {row["campaign_id"]: row for row in document["campaigns"]}
    for row in document["campaigns"]:
        kind = row.get("entity_kind", "campaign")
        if kind not in {"campaign", "source_study"}:
            raise ValueError("Unknown campaign provenance entity kind")
        parent = row.get("parent_campaign_id")
        if kind == "source_study":
            if parent not in by_id or by_id[parent].get("entity_kind", "campaign") != "campaign":
                raise ValueError("A source study requires an existing parent campaign")
        elif parent is not None:
            raise ValueError("Top-level campaign cannot declare a parent campaign")
    return document["campaigns"]


def _current(data: Path, issues: list[dict]) -> dict:
    """Every published campaign bundle is available; there is no selected product."""
    from engine.chairstats import CampaignStore, views

    result = {"campaigns": [], "status": "unavailable"}
    try:
        bundles = CampaignStore(data / "campaigns").campaigns()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append({"severity": "error", "path": str(data / "campaigns"), "message": str(exc)})
        return result
    if not bundles:
        issues.append(
            {
                "severity": "info",
                "path": str(data / "campaigns"),
                "message": "No campaign bundles; research inventory remains usable.",
            }
        )
        return result
    result.update(campaigns=[views.summary(m) for m in bundles], status="bundles_verified")
    return result


def _runs(root: Path, data: Path, issues: list[dict]) -> list[dict]:
    directories = set((data / "research").glob("*"))
    directories.update(
        path for path in (data / "campaigns").glob("*") if not path.name.startswith(".")
    )
    directories.update(root.glob("experiments/*/runs/*"))
    rows = []
    for directory in sorted(d for d in directories if d.is_dir()):
        relative = (
            str(directory.relative_to(root)) if directory.is_relative_to(root) else str(directory)
        )
        path = directory / "manifest.json"
        row = {
            "id": directory.name,
            "path": relative,
            "status": "unindexed",
            "kind": None,
            "has_manifest": path.exists(),
            "published_campaign": False,
            "serving": "not_inferred",
            "files": [],
            "artifact_count": 0,
        }
        for name in (
            "manifest.json",
            "report.json",
            "report.md",
            "readout.md",
            "index.html",
            "config.json",
            "decisions.json",
            "metrics.json",
            "README.md",
        ):
            if (directory / name).exists():
                row["files"].append(name)
        if path.exists():
            try:
                m = read_json(path)
                row.update(
                    status=m.get("status", "unspecified"),
                    kind=m.get("kind"),
                    manifest_sha256=digest(path),
                    generated_at=m.get("generated_at", m.get("saved_at")),
                    season=m.get("season"),
                    through_week=m.get("through_week"),
                    artifact_count=len(m.get("files", m.get("output_sha256", {}))),
                )
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                row["status"] = "unreadable"
                issues.append({"severity": "warning", "path": str(path), "message": str(exc)})
        rows.append(row)
    return rows


def _configs(root: Path, issues: list[dict]) -> list[dict]:
    """Index checked-in experiment configurations and saved-model contracts, not secrets."""
    rows = []
    for lab in (root / "experiments").glob("*_lab"):
        for path in sorted(lab.rglob("*.json")):
            parts = path.relative_to(lab).parts
            if any(x in {"runs", "__pycache__", ".test-tmp", "__marimo__"} for x in parts):
                continue
            if path.stat().st_size > 2_000_000:
                continue
            try:
                value = read_json(path)
                if not isinstance(value, dict):
                    continue
                is_saved = "model_directory" in value and "model_manifest_sha256" in value
                fields = {
                    k: value[k]
                    for k in (
                        "id",
                        "campaign_id",
                        "campaign_name",
                        "positions",
                        "position",
                        "targets",
                        "target",
                        "horizons",
                        "horizon",
                        "origins",
                        "supported_origins",
                        "families",
                        "evaluation_years",
                        "recipes",
                        "scoring",
                        "policy",
                        "policies",
                        "model_directory",
                        "model_manifest_sha256",
                    )
                    if k in value
                }
                if not fields:
                    continue
                row = {
                    "path": str(path.relative_to(root)),
                    "kind": "saved_model" if is_saved else "configuration",
                    "sha256": digest(path),
                    "contract": fields,
                }
                if is_saved:
                    model_root = within(root, value["model_directory"])
                    # Existing contracts use paths relative to the future-player lab.
                    if not model_root.exists():
                        model_root = within(lab, value["model_directory"])
                    mp = model_root / "manifest.json"
                    row["artifact_state"] = (
                        "missing"
                        if not mp.exists()
                        else "manifest_matches"
                        if digest(mp) == value["model_manifest_sha256"]
                        else "manifest_mismatch"
                    )
                    if row["artifact_state"] == "manifest_matches":
                        manifest = read_json(mp)
                        retired = set(read_json(Path(__file__).with_name("retired_features.json")))
                        row["contract"]["input_audit"] = {
                            horizon: {
                                "input_count": len(provenance.get("names", [])),
                                "retired_input_matches": [
                                    name
                                    for name in provenance.get("names", [])
                                    if set(name.split("__")) & retired
                                ],
                                "scope": "Recorded names only; weights and data not executed",
                            }
                            for horizon, provenance in manifest.get("input_provenance", {}).items()
                        }
                rows.append(row)
            except (OSError, ValueError, TypeError) as exc:
                issues.append({"severity": "warning", "path": str(path), "message": str(exc)})
    return rows


def build_catalog(
    root: Path,
    data: Path | None = None,
    definitions: Path = FAMILIES,
    campaign_definitions: Path = CAMPAIGNS,
) -> dict:
    root = root.resolve()
    data = (data or root / "data").resolve()
    issues = []
    families = load_families(definitions)
    campaigns = load_campaigns(campaign_definitions)
    current = _current(data, issues)
    for family in families:
        family["source_state"] = [
            {"path": p, "exists": within(root, p).exists()} for p in family["sources"]
        ]
        for source in family["source_state"]:
            if not source["exists"]:
                issues.append(
                    {
                        "severity": "warning",
                        "path": source["path"],
                        "message": f"Missing family source: {family['id']}",
                    }
                )
    runs = _runs(root, data, issues)
    bundles = {c["campaign_id"] for c in current["campaigns"]}
    for run in runs:
        run["published_campaign"] = run["id"] in bundles and run["path"].startswith(
            ("data/campaigns/", str(data / "campaigns"))
        )
    configs = _configs(root, issues)
    campaign_by_id = {row["campaign_id"]: row for row in campaigns}
    for campaign in campaigns:
        tag = {key: campaign[key] for key in ("campaign_id", "campaign_name")}
        tag["entity_kind"] = campaign.get("entity_kind", "campaign")
        tags = [tag]
        if parent_id := campaign.get("parent_campaign_id"):
            parent = campaign_by_id[parent_id]
            campaign["parent_campaign_name"] = parent["campaign_name"]
            tag.update(parent_campaign_id=parent_id, parent_campaign_name=parent["campaign_name"])
            tags.append(
                {
                    "campaign_id": parent_id,
                    "campaign_name": parent["campaign_name"],
                    "entity_kind": parent.get("entity_kind", "campaign"),
                }
            )

        def associate(row, tags=tags):
            present = {item["campaign_id"] for item in row.get("campaigns", [])}
            row.setdefault("campaigns", []).extend(
                item for item in tags if item["campaign_id"] not in present
            )

        for family in families:
            if family["id"] in campaign.get("family_ids", []):
                associate(family)
        for run in runs:
            run_path = (root / run["path"]).resolve()
            identity_path = (
                "data/" + str(run_path.relative_to(data))
                if run_path.is_relative_to(data)
                else run["path"]
            )
            if identity_path == campaign.get("comparison_directory") or any(
                fnmatchcase(identity_path, pattern) for pattern in campaign.get("run_patterns", [])
            ):
                associate(run)
        for config in configs:
            if config["path"] in campaign.get("config_paths", []):
                associate(config)
        reference = campaign["saved_model_manifest"]
        path = (
            within(data, reference.removeprefix("data/"))
            if reference.startswith("data/")
            else within(root, reference)
        )
        campaign["local_manifest_present"] = path.exists()
        expected = campaign.get("saved_model_manifest_sha256")
        campaign["manifest_state"] = (
            "missing"
            if not path.exists()
            else "manifest_matches"
            if expected and digest(path) == expected
            else "manifest_mismatch"
            if expected
            else "present_not_verified_by_inventory"
        )
    # All active implementation modules get an index row, even if not yet curated.
    sources = set()
    for pattern in (
        "src/engine/metrics/*.py",
        "research/*.py",
        "experiments/*_lab/*.py",
        "experiments/*_lab/deep/*.py",
        "experiments/*_lab/gpu_lab/*.py",
        "experiments/*_lab/forecasting/*.py",
        "experiments/*_lab/campaigns/*.py",
        "src/engine/chairstats/*.py",
    ):
        sources.update(root.glob(pattern))
    groups = read_json(Path(__file__).with_name("source_groups.json"))
    implementations = []
    for p in sorted(sources):
        if p.name.startswith(("test_", "__")):
            continue
        path = str(p.relative_to(root))
        direct = [f["id"] for f in families if path in f["sources"]]
        # Match globs component by component so '*' never crosses a directory boundary.
        group = next(
            (
                g
                for g in groups
                if any(
                    len(Path(pattern).parts) == len(Path(path).parts)
                    and all(
                        fnmatchcase(part, glob)
                        for part, glob in zip(Path(path).parts, Path(pattern).parts, strict=True)
                    )
                    for pattern in g["patterns"]
                )
            ),
            None,
        )
        owned = direct or ([group["family"]] if group and group["family"] else [])
        role = (
            "curated_entrypoint"
            if direct
            else (group.get("role", "family_support") if group else "unclassified")
        )
        implementations.append({"path": path, "families": owned, "role": role})
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "data_root": str(data),
        "purpose": "Research inventory, not serving authorization or player-quality validation",
        "direction": {
            "ability": "Separate ability ratings; aggregation formula undecided.",
            "team_context": "Separate team rankings with opportunity and teammate interactions.",
            "boundary": "Production and fantasy outcomes are applications, not ability labels.",
        },
        "families": families,
        "campaigns": campaigns,
        "current": current,
        "runs": runs,
        "configurations": configs,
        "claims": read_json(Path(__file__).with_name("claims.json")),
        "implementations": implementations,
        "issues": issues,
        "summary": {
            "families": len(families),
            "runs": len(runs),
            "configurations": len(configs),
            "saved_models": sum(c["kind"] == "saved_model" for c in configs),
            "campaign_bundles": len(current["campaigns"]),
            "implementations": len(implementations),
            "unclassified_implementations": sum(
                x["role"] == "unclassified" for x in implementations
            ),
        },
        "limitations": [
            "Family assessments are curated interpretations with source links, not approval rules.",
            "Displayed current metadata is checksum matched. This index does "
            "not revalidate the full dependency graph, model weights, or statistical findings.",
            "Run completion is self-reported metadata, not approval.",
            "Configurations describe possible recipes; they are not counts of trained estimators.",
            "Unclassified source files and unindexed run directories are explicit review backlog.",
        ],
    }
