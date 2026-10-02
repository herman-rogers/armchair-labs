"""Package a complete QB passing experiment into the atomic NextGen catalog."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from patron.data.catalog import publish_catalog
from patron.data.nextgen import load_analysis
from patron.data.releases import current_catalog, digest, identifier, inside, reference, write_json
from patron.metrics.qb_passing import CHALLENGERS, HORIZONS, summarize

ROOT = Path(__file__).resolve().parents[1]


def verify_run(source: Path, base_ref: dict):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("kind") != "qb_passing_research" or manifest.get("status") != "complete":
        raise ValueError("QB passing research is incomplete")
    if manifest["source_analysis"] != base_ref:
        raise ValueError("QB passing was fitted against a different published analysis")
    for name, expected in manifest["files"].items():
        if digest(inside(source, name)) != expected:
            raise ValueError(f"QB passing artifact changed: {name}")
    current = pl.read_parquet(source / "current.parquet")
    if (
        current.height == 0
        or current.select(pl.struct("player_id", "horizon").n_unique()).item() != current.height
    ):
        raise ValueError("Missing or duplicate current QB forecasts")
    if any(c in current.columns for c in CHALLENGERS) or "actual" in current.columns:
        raise ValueError("Research-only scores or target labels leaked into current forecasts")
    if current.filter(
        pl.col("prediction").is_null()
        | ~pl.col("prediction").is_finite()
        | (pl.col("prediction") < 0)
    ).height:
        raise ValueError("Invalid current QB prediction")
    folds = json.loads((source / "folds.json").read_text())
    if any(f["train_last_year"] >= f["season"] for f in folds):
        raise ValueError("Nonchronological training fold")
    return manifest


def package(data, source_version, version, publish=False, analysis_version=None):
    base_ref = (
        reference(data / "research" / identifier(analysis_version)) if analysis_version else None
    )
    base, base_manifest = load_analysis(data, base_ref)
    source = data / "research" / identifier(source_version)
    source_manifest = verify_run(source, reference(base))
    if source_manifest["gold"] != base_manifest["gold"]:
        raise ValueError("Mixed QB passing and gold releases")
    root = data / "research" / identifier(version)
    if root.exists():
        raise ValueError("Delivery releases are create-only")
    shutil.copytree(base, root)
    (root / "manifest.json").unlink()
    if (root / "qb_passing").exists():
        # This is the new private copy; the previous published release is retained.
        shutil.rmtree(root / "qb_passing")
    shutil.copytree(source, root / "qb_passing")
    # Bind the captured current-news source bytes alongside the reviewed claims.
    news = json.loads((root / "qb_passing/news.json").read_text())["observations"]
    for note in news:
        sha = note.get("source_sha256")
        if not sha:
            raise ValueError("Current QB news is missing its source hash")
        raw = data / "raw/objects" / sha[:2] / sha
        if digest(raw) != sha:
            raise ValueError("Current QB news source changed")
        destination = root / "qb_passing/news_sources" / sha
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(raw, destination)
    now = datetime.now(UTC).isoformat()
    registry = json.loads((root / "registry.json").read_text())
    registry = [r for r in registry if not r["id"].startswith("qb_passing:")]
    dependencies = [
        "nfl_player_weeks.passing_yards",
        "nfl_player_weeks.attempts",
        "nfl_transactions",
        "qb_passing.events",
    ]
    for horizon in HORIZONS:
        registry.append(
            dict(
                id="qb_passing:reference:" + horizon,
                label="QB passing · " + horizon.replace("_", " "),
                kind="qb_forecast",
                target="passing_yards",
                horizon=horizon,
                unit="passing yards",
                validity="verified",
                evidence="chronological_reference",
                serving="baseline",
                allowed_uses=["forecast"],
                positions=["QB"],
                populations=["all"],
                dependencies=dependencies,
                reason="Career and current production reference; selection uses earlier seasons. "
                "Availability news adjusts opportunity; no validated-challenger "
                "or fantasy-ranking claim.",
            )
        )
        for model in CHALLENGERS:
            registry.append(
                dict(
                    id=f"qb_passing:{model}:{horizon}",
                    label=f"QB passing {model} · {horizon}",
                    kind="qb_forecast",
                    target="passing_yards",
                    horizon=horizon,
                    unit="passing yards",
                    validity="verified",
                    evidence="retrospective_research",
                    serving="shadow",
                    allowed_uses=[],
                    positions=["QB"],
                    populations=["all"],
                    dependencies=dependencies,
                    reason="Retained challenger research; prospective review is pending. "
                    "Excluded from current analysis.",
                )
            )
    for prior in sorted((data / "research").glob("qb_passing_*/manifest.json")):
        name = prior.parent.name
        if name == version:
            continue
        prototype = name in {"qb_passing_20260924_r1", "qb_passing_20260924_r2"}
        reason = (
            "Superseded QB prototype: population fallback, changed-position outcomes "
            "or duplicate partial-absence adjustment required correction. "
            "Revision 3 supersedes these outputs; excluded from analysis."
            if prototype
            else "Preserved QB research/audit. Only scoped reference entries in the "
            "current delivery release may serve forecasts."
        )
        registry.append(
            dict(
                id="study:" + name,
                label=name,
                kind="study",
                target="passing_yards",
                horizon="mixed",
                unit="yards",
                validity="quarantined" if prototype else "verified",
                evidence="preserved_research",
                serving="archive",
                allowed_uses=[],
                positions=["QB"],
                populations=["all"],
                dependencies=[],
                path=str(prior.parent.relative_to(data)),
                source_manifest_sha256=digest(prior),
                reason=reason,
            )
        )
    current_ref = (current_catalog(data) or {}).get("products", {}).get("analysis")
    if current_ref and current_ref["version"] != base_manifest["version"]:
        registry.append(
            dict(
                id="study:" + current_ref["version"],
                label=current_ref["version"],
                kind="study",
                target="passing_yards",
                horizon="mixed",
                unit="yards",
                validity="verified",
                evidence="superseded_delivery",
                serving="archive",
                allowed_uses=[],
                positions=["QB"],
                populations=["all"],
                dependencies=[],
                path="research/" + current_ref["version"],
                source_manifest_sha256=current_ref["manifest_sha256"],
                reason="Superseded delivery preserved for reproduction; "
                "excluded from current analysis.",
            )
        )
    for entry in registry:
        if entry["kind"] == "ranking":
            entry["analysis_version"] = version
    registry = list({entry["id"]: entry for entry in registry}.values())
    write_json(root / "registry.json", registry)
    report = json.loads((root / "report.json").read_text())
    passing_report = json.loads((root / "qb_passing/report.json").read_text())
    # Audit the exact legacy cohort independently of the new weekly panel.
    old = (
        pl.read_parquet(base / "predictions.parquet")
        .filter(
            (pl.col("target") == "passing_yards")
            & (pl.col("position") == "QB")
            & (pl.col("model") == "boost")
            & pl.col("complete")
        )
        .select(
            "player_id",
            "season",
            pl.col("actual").alias("saved_actual"),
            pl.col("prediction").alias("saved_boost"),
            pl.col("baseline").alias("saved_baseline"),
        )
    )
    new = pl.read_parquet(source / "predictions.parquet").filter(
        (pl.col("through_week") == 0)
        & (pl.col("horizon") == "rest_of_season")
        & pl.col("actual").is_not_null()
    )
    matched = new.join(old, on=["player_id", "season"], validate="1:1")
    if (
        matched.height != old.height
        or matched.filter(pl.col("actual") != pl.col("saved_actual")).height
    ):
        raise ValueError("Preseason outcome audit does not match the saved QB cohort")
    audit = dict(
        matched_preseason_outcomes=matched.height,
        outcome_mismatches=0,
        note="Original saved forecasts are unchanged. The new run has additional dated "
        "constraints; comparisons against the original booster therefore mix input "
        "and model changes.",
        comparisons=[
            dict(
                window=window,
                **summarize(
                    matched.filter(pl.col("season") >= first).to_dicts(), model, "saved_baseline"
                ),
            )
            for window, first in [("all_history", 2007), ("modern", 2019)]
            for model in ["saved_baseline", "saved_boost", "reference", "conditional_boost"]
        ],
    )
    write_json(root / "qb_passing/delivery_audit.json", audit)
    old_examples = {(r["player_id"], r["season"]): r for r in old.to_dicts()}
    for example in passing_report["examples"]:
        saved = old_examples.get((example["player_id"], example["season"]), {})
        example.update({k: saved.get(k) for k in ["saved_baseline", "saved_boost"]})
    passing_report["delivery_audit"] = "delivery_audit.json"
    write_json(root / "qb_passing/report.json", passing_report)
    report.update(
        version=version,
        registry_entries=len(registry),
        qb_passing=dict(
            source_version=source_version,
            serving="reference",
            published_at=now,
            season=passing_report["season"],
            through_week=passing_report["through_week"],
            candidates=passing_report["candidates"],
            challenger_status="prospective_shadow",
        ),
    )
    write_json(root / "report.json", report)
    for name in [
        "research/publish_qb_passing.py",
        "research/build_qb_passing.py",
        "research/score_qb_passing.py",
        "research/qb_passing_run_protocol.md",
        "src/patron/api/qb_passing_routes.py",
        "src/patron/data/nextgen.py",
        "src/patron/data/catalog.py",
        "src/patron/metrics/transaction_events.py",
        "src/patron/metrics/experimental.py",
        "src/patron/metrics/current_rankings.py",
        "src/patron/metrics/nextgen.py",
        "web/src/components/QBPassing.tsx",
        "web/src/api/qbPassing.ts",
        "web/src/components/NextGenView.tsx",
        "web/src/components/PlayerProfile.tsx",
        "docs/qb_passing_system_2026-09-24.md",
        "tests/test_qb_passing.py",
        "tests/test_qb_passing_routes.py",
        "web/tests/qb_passing_smoke.py",
    ]:
        destination = root / "delivery" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    manifest = dict(
        base_manifest,
        version=version,
        generated_at=now,
        qb_passing_schema_version=1,
        qb_passing_research=reference(source),
        files={str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
    )
    write_json(root / "manifest.json", manifest)
    load_analysis(data, reference(root))
    if publish:
        profile = json.loads(
            (data / "research" / base_manifest["profiles"]["version"] / "report.json").read_text()
        )
        products = dict(
            college=profile["college_version"],
            profiles=base_manifest["profiles"]["version"],
            analysis=version,
        )
        products[f"outlook_{passing_report['season']}"] = profile["outlook_version"]
        publish_catalog(data, manifest["gold"]["version"], products)
    print(json.dumps(dict(version=version, published=publish), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--source", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--analysis-version", help="Verified staged analysis for a new data release"
    )
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    package(args.data, args.source, args.version, args.publish, args.analysis_version)
