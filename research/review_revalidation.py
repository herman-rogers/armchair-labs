"""Resolve the published revalidation queue in a new, immutable analysis release.

This is a retirement and dependency review, not a model promotion experiment.
Existing forecast evidence and serving permissions are preserved exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from patron.api.college_sources import load_college
from patron.api.outlook_sources import load_outlook_release
from patron.data.catalog import publish_catalog
from patron.data.nextgen import load_analysis, read_json, verify_evidence
from patron.data.releases import digest, identifier, load_gold, reference, write_json

ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "preseason_features",
    "season_outcomes",
    "nfl_player_weeks",
    "nfl_player_seasons",
    "nfl_weekly_usage",
    "nfl_snap_counts",
    "players",
    "college_annual",
    "nfl_injuries",
)
NOTES = (
    "nextgen_system_implementation_2026-09-23.md",
    "nextgen_rankings_2026-09-23.md",
    "individual_stats_2026-09-23.md",
    "nextgen_signal_consolidation_2026-09-23.md",
    "qb_role_transition_2026-09-23.md",
    "rb_residual_correction_2026-09-23.md",
    "qb_passing_forecasts_review_2026-09-24.md",
)
STAT_REOPEN = {
    "constant": "Reconsider only when corrected historical inputs contain finite variation.",
    "unavailable_in_gold": "Reconsider after dated, valid input coverage is added to gold.",
    "metadata": (
        "Retain as metadata; any predictor use needs a separately declared cutoff-safe definition."
    ),
    "outcome_only": (
        "Retain as an evaluation label; never admit the held-out outcome as its predictor."
    ),
    "unsafe_or_fitted_output": (
        "Requires nested out-of-fold reconstruction and a cutoff/leakage audit."
    ),
    "unreconstructed": (
        "Requires an explicit corrected-gold reconstruction and a new matched evaluation."
    ),
}


def stat_decision(entry, evidence, features):
    name = entry["id"].removeprefix("stat:")
    row = evidence[name]
    status = row["status"]
    if status not in STAT_REOPEN or entry["reason"] != row["reason"]:
        raise ValueError(f"Unreviewed stat disposition: {name}")
    values = (
        features[name].cast(pl.Float64, strict=False)
        if name in features.columns
        else pl.Series([], dtype=pl.Float64)
    )
    finite = values.filter(values.is_finite().fill_null(False))
    if len(finite) != row["finite_rows"] or features.height != row["total_rows"]:
        raise ValueError(f"Stat coverage changed: {name}")
    if status == "constant" and finite.n_unique() > 1:
        raise ValueError(f"Stat is no longer constant: {name}")
    if status == "unavailable_in_gold" and len(finite):
        raise ValueError(f"Stat is now observed: {name}")
    return dict(
        category="stat_" + status,
        reason=row["reason"]
        + " Predictor candidate retired for this data vintage; descriptive uses are unaffected.",
        reconsideration=STAT_REOPEN[status],
        evidence_details=dict(
            source_status=status,
            finite_rows=len(finite),
            total_rows=features.height,
            distinct_finite_values=finite.n_unique(),
            definition_ref=row["definition_ref"],
        ),
    )


def legacy_decision(entry, specs):
    name = entry["id"].removeprefix("legacy:")
    spec = specs[name]
    # apply_live belongs to the archived board builder, not the NextGen policy.
    # Preserve it for reproduction; apply_decisions enforces current exclusion.
    if spec.get("kind") in {"product", "coalesce", "adaptive_select"}:
        category = "legacy_composite"
        detail = (
            "Depends on retired fitted recipes; combining or selecting them does not "
            "repair their input vintage."
        )
    elif "market" in name:
        category = "legacy_market"
        detail = (
            "Requires cutoff-certified market coverage and a corrected, matched "
            "incremental evaluation."
        )
    elif any(s in name for s in ("games", "return_prob", "roster", "status", "security")):
        category = "legacy_participation"
        detail = (
            "Old participation/roster proxies do not validate starting status or "
            "medical availability."
        )
    elif any(
        s in name for s in ("career", "athletic", "rich_weekly", "research", "core_plus", "soph")
    ):
        category = "legacy_feature_stack"
        detail = (
            "Corrected individual-stat and profile experiments now cover these "
            "information families; their scoped results do not approve this old fitted "
            "recipe."
        )
    else:
        category = "legacy_baseline"
        detail = (
            "Superseded by explicit outcome-specific references and separately "
            "evaluated current-season rankings."
        )
    return dict(
        category=category,
        reason=f"Retired recipe {name}. {detail} No current corrected-input serving approval.",
        reconsideration=(
            "A new recipe/version needs corrected cutoff-safe inputs and "
            "target/population/horizon-matched evaluation; do not restore this saved "
            "fit."
        ),
        evidence_details=dict(configured_recipe=spec),
    )


def artifact_decision(entry, verified_dependencies):
    name = entry["id"].split(":", 1)[1]
    if entry["id"] in verified_dependencies:
        return dict(
            category="verified_dependency",
            validity="verified",
            reason=verified_dependencies[entry["id"]],
            reconsideration=(
                "Recheck exact dependencies on an input change. Predictive serving "
                "approval requires a separate scoped evaluation."
            ),
        )
    if entry["kind"] == "data_release":
        reason = (
            "Superseded corrected-gold snapshot; the completed evidence pins r3 and the "
            "catalog pins canonical NextGen gold."
        )
    elif name.startswith(
        (
            "canonical_products_",
            "nextgen_products_",
            "college_nfl_",
            "player_profiles_",
            "nextgen_profiles_",
            "player_outlook_",
        )
    ):
        reason = (
            "Superseded product build; use the catalog-pinned corrected profiles, "
            "college and outlook products. Old model claims are not transferred."
        )
    elif name.startswith(("historical_", "evidence_backfill_")):
        reason = (
            "Retired historical reconstruction/model run. Any accepted observations "
            "remain preserved through the current gold dependency chain; old forecasts "
            "are not approved."
        )
    elif name.startswith(("individual_stats_", "nextgen_signals_")):
        reason = (
            "Superseded screening run; corrected individual_stats_20260923_r4 retains "
            "scoped findings and negative results. No old aggregate model is "
            "reinstated."
        )
    elif name.startswith("nextgen_system_"):
        reason = (
            "Superseded analysis build; its useful measurements and scoped evidence are "
            "retained in the published analysis."
        )
    elif name.startswith(("qb_role_transition_", "preseason_role_workload_")):
        reason = (
            "Retired role/workload recipe: better opening-job recognition did not "
            "establish market-relative acquisition value or reliable conditional "
            "workload. The dated-evidence idea remains a research lead."
        )
    elif name == "rb_residual_20260923_r1":
        reason = (
            "Retired negative experiment: all five advancement checks failed, point MAE "
            "worsened, and small-prior-sample selection lost points."
        )
    elif name == "baseline_incremental_20260922_r1":
        reason = (
            "Exploratory incremental-information screen, superseded for serving by the "
            "corrected stat audit and ranking evaluation; no decision model earned "
            "promotion."
        )
    elif name.startswith(("profile_roster_review_", "roster_review_")):
        reason = (
            "Dated roster cross-reference/advice snapshot, not a reusable validated "
            "forecast; current league observations and profiles supersede its "
            "operational use."
        )
    else:
        raise ValueError(f"Artifact needs a specific review: {name}")
    return dict(
        category="retired_artifact",
        reason=reason + " Original files remain available for reproduction.",
        reconsideration=(
            "New scoped research may reuse verified observations, but must not inherit "
            "this artifact's predictive claims."
        ),
    )


def apply_decisions(registry, decisions, version, date):
    """Require exhaustive, one-to-one review; never expand serving permissions."""
    ids = [r["id"] for r in registry]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate registry ids")
    pending = {r["id"] for r in registry if r["validity"] == "revalidation_required"}
    if set(decisions) != pending:
        raise ValueError("Review must cover exactly the pending registry entries")
    result = deepcopy(registry)
    for entry in result:
        if entry["id"] not in pending:
            continue
        decision = decisions[entry["id"]]
        validity = decision.get("validity", "archived")
        if validity not in {"archived", "verified"}:
            raise ValueError("Unsupported review disposition")
        if validity == "verified" and (
            entry["kind"] not in {"study", "data_release"}
            or decision["category"] != "verified_dependency"
        ):
            raise ValueError("This review cannot promote a predictive model")
        if entry["serving"] != "archive" or entry["allowed_uses"]:
            raise ValueError("Pending entry unexpectedly has serving permissions")
        entry.update(
            validity=validity,
            reason=decision["reason"],
            reconsideration=decision["reconsideration"],
            evidence="verified_dependency" if validity == "verified" else entry["evidence"],
            decision_date=date,
            analysis_version=version,
            review_category=decision["category"],
            review_artifact="revalidation_review.json",
        )
    return result


def build(version: str, *, publish: bool = False):
    data = ROOT / "data"
    catalog = read_json(data / "current.json")
    catalog_hash = digest(data / "current.json")
    source, manifest = load_analysis(data)
    registry = read_json(source / "registry.json")
    if not any(r["validity"] == "revalidation_required" for r in registry):
        raise ValueError("The published registry has no pending revalidation entries")
    destination = data / "research" / identifier(version)
    if destination.exists():
        raise ValueError("Review releases are create-only")
    gold = load_gold(data)
    evidence_root, evidence_manifest = verify_evidence(data, manifest["evidence"])
    evidence_gold = load_gold(data, evidence_manifest["gold"]["version"])
    if evidence_gold.ref != evidence_manifest["gold"]:
        raise ValueError("Evidence gold reference changed")
    for table in TABLES:
        keys = gold.manifest["tables"][table]["primary_key"]
        if not gold.read(table).sort(keys).equals(evidence_gold.read(table).sort(keys)):
            raise ValueError(f"Evidence no longer matches current data: {table}")
    college_root, _ = load_college(reference_override=catalog["products"]["college"])
    season = gold.manifest["current_observations"]["season"]
    outlook_root, _ = load_outlook_release(
        season, reference_override=catalog["products"][f"outlook_{season}"]
    )
    college_manifest = read_json(college_root / "manifest.json")
    verified = {
        "study:" + college_root.name: (
            "Revalidated exact current college product, source hashes, accepted history "
            "and corrected gold. Valid as a profile/research dependency; college "
            "forecast promotion remains unapproved."
        ),
        "study:" + outlook_root.name: (
            "Revalidated exact current outlook product, source/output hashes, accepted "
            "history and corrected gold. Retained as a profile/research dependency; old "
            "four-week predictions remain archived."
        ),
        "study:" + college_manifest["source_version"]: (
            "Revalidated captured college source manifest and every raw file against "
            "the current college product. Valid source observations, not a predictive "
            "model or certified historical publication vintage."
        ),
        "data_release:" + evidence_gold.root.name: (
            "Revalidated the exact gold dependency of the completed individual-stat "
            "study, including equality of all nine consumed tables with current "
            "corrected gold. Valid for evidence reproduction, not the current serving "
            "pointer."
        ),
    }
    stat_registry = {r["stat"]: r for r in read_json(evidence_root / "registry.json")}
    features = pl.read_parquet(evidence_root / "features.parquet")
    config_path = ROOT / "src/patron/config/metric_report.yaml"
    specs = {s["name"]: s for s in yaml.safe_load(config_path.read_text())["fit"]["models"]}
    decisions = {}
    for entry in registry:
        if entry["validity"] != "revalidation_required":
            continue
        kind = entry["kind"]
        if kind == "research_stat":
            decision = stat_decision(entry, stat_registry, features)
        elif kind == "legacy_model":
            decision = legacy_decision(entry, specs)
        elif kind in {"study", "data_release"}:
            decision = artifact_decision(entry, verified)
            artifact = data / entry["path"] / "manifest.json"
            if (
                entry.get("source_manifest_sha256")
                and digest(artifact) != entry["source_manifest_sha256"]
            ):
                raise ValueError(f"Archived manifest changed: {entry['id']}")
        elif kind == "forecast" and entry["id"].startswith("unavailable:"):
            decision = dict(
                category="unimplemented_claim",
                reason=(
                    "Retired placeholder: no implemented, target-matched validated "
                    "model exists for this claim. Keep it unavailable; descriptive "
                    "measurements and separate research leads remain useful."
                ),
                reconsideration=(
                    "Requires a new target definition, dated source coverage, "
                    "cutoff-safe model and matched evaluation before any serving "
                    "review."
                ),
            )
        else:
            raise ValueError(f"Entry needs a specific review: {entry['id']}")
        decisions[entry["id"]] = dict(previous_entry=entry, **decision)
    now = datetime.now(UTC).isoformat()
    updated = apply_decisions(registry, decisions, version, now[:10])
    # Copy bytes, never hard-link immutable historical artifacts to a writable release.
    shutil.copytree(source, destination)
    (destination / "manifest.json").unlink()
    write_json(destination / "registry.json", updated)
    summary = dict(
        reviewed=len(decisions),
        revalidated=sum(d.get("validity") == "verified" for d in decisions.values()),
        archived=sum(d.get("validity", "archived") == "archived" for d in decisions.values()),
        pending=0,
        model_promotions=0,
        by_kind=dict(Counter(d["previous_entry"]["kind"] for d in decisions.values())),
        by_category=dict(Counter(d["category"] for d in decisions.values())),
    )
    review = dict(
        version=version,
        reviewed_at=now,
        source=reference(source),
        source_catalog_sha256=catalog_hash,
        summary=summary,
        checks=dict(
            exact_evidence_table_equivalence=list(TABLES),
            stat_coverage_recomputed=True,
            current_products_verified=True,
        ),
        scope=(
            "Retirement and dependency validation; no new model fits, independent "
            "holdout or predictive promotion."
        ),
        decisions=decisions,
    )
    write_json(destination / "revalidation_review.json", review)
    report = read_json(destination / "report.json")
    report.update(version=version, revalidation_review=summary)
    write_json(destination / "report.json", report)
    inputs = destination / "review_inputs"
    inputs.mkdir()
    for path in [Path(__file__), config_path, *(ROOT / "docs" / n for n in NOTES)]:
        shutil.copy2(path, inputs / path.name)
    write_json(inputs / "source_catalog.json", catalog)
    new_manifest = dict(
        manifest, version=version, generated_at=now, review_source=reference(source)
    )
    new_manifest["files"] = {
        str(p.relative_to(destination)): digest(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    write_json(destination / "manifest.json", new_manifest)
    load_analysis(data, reference(destination))
    # All original evidence/prediction artifacts must be byte-identical.
    for name, expected in manifest["files"].items():
        if name not in {"registry.json", "report.json"} and digest(destination / name) != expected:
            raise ValueError(f"Review altered forecast evidence: {name}")
    load_analysis(data, reference(source))
    if digest(data / "current.json") != catalog_hash:
        raise ValueError("Catalog changed during review; review the new source before publication")
    if publish:
        products = {key: ref["version"] for key, ref in catalog["products"].items()}
        products["analysis"] = version
        publish_catalog(data, catalog["gold"]["version"], products)
    print(json.dumps(dict(version=version, published=publish, **summary), indent=2))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    build(args.version, publish=args.publish)
