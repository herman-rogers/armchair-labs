"""Package a verified ranking run for delivery without altering its research evidence."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from patron.data.catalog import publish_catalog
from patron.data.nextgen import load_analysis
from patron.data.releases import digest, identifier, load_gold, reference, write_json

ROOT = Path(__file__).resolve().parents[1]


def package(source: str, version: str, *, publish: bool = False):
    data = ROOT / "data"
    source_root, source_manifest = load_analysis(
        data, reference(data / "research" / identifier(source))
    )
    if source_manifest.get("ranking_schema_version") != 1:
        raise ValueError("Source has no verified current-season rankings")
    root = data / "research" / identifier(version)
    if root.exists():
        raise ValueError("Delivery releases are create-only")
    shutil.copytree(source_root, root)
    (root / "manifest.json").unlink()
    now = datetime.now(UTC).isoformat()
    registry = json.loads((root / "registry.json").read_text())
    for item in registry:
        if item["kind"] == "ranking":
            item["analysis_version"] = version
    for prior in sorted((data / "research").glob("nextgen_rankings_*/manifest.json")):
        manifest = json.loads(prior.read_text())
        registry = [r for r in registry if r["id"] != "study:" + prior.parent.name]
        registry.append(
            dict(
                id="study:" + prior.parent.name,
                label=prior.parent.name,
                kind="study",
                target="league_points",
                horizon="current_season",
                validity="verified",
                evidence="retrospective_research",
                serving="archive",
                allowed_uses=[],
                positions=["QB", "RB", "WR", "TE"],
                populations=["all"],
                dependencies=[],
                path=str(prior.parent.relative_to(data)),
                source_manifest_sha256=digest(prior),
                original_status=manifest["status"],
                reason=(
                    "Preserved ranking research run. Only the new delivery release's "
                    "scoped ranking entries may serve predictions."
                ),
            )
        )
    write_json(root / "registry.json", registry)
    ranking_report = json.loads((root / "ranking_report.json").read_text())
    ranking_report["published_at"] = now
    write_json(root / "ranking_report.json", ranking_report)
    report = json.loads((root / "report.json").read_text())
    report.update(
        version=version,
        registry_entries=len(registry),
        rankings=ranking_report,
        ranking_models_promoted=report["rankings"]["validated_scopes"],
    )
    report["limitations"] = [
        text.replace(
            "Multiple outcome comparisons are exploratory; no challenger is promoted.",
            "Preseason multi-outcome challengers remain research-only. "
            "Current-season ranking scopes have separate publication decisions.",
        )
        for text in report["limitations"]
    ]
    write_json(root / "report.json", report)
    # Preserve exactly the fit code in its research snapshot; bind delivery code separately.
    for source_path in [
        Path(__file__),
        ROOT / "src/patron/api/ranking_routes.py",
        ROOT / "src/patron/data/nextgen.py",
        ROOT / "src/patron/data/catalog.py",
        ROOT / "web/src/components/NextGenRankings.tsx",
        ROOT / "web/src/components/PlayerRanking.tsx",
        ROOT / "web/src/components/PlayerProfile.tsx",
        ROOT / "web/src/components/NextGenView.tsx",
        ROOT / "web/src/components/LeagueWorkspace.tsx",
        ROOT / "web/src/api/nextgen.ts",
    ]:
        destination = root / "delivery" / source_path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    manifest = dict(
        source_manifest,
        version=version,
        generated_at=now,
        ranking_research=reference(source_root),
        files={str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
    )
    write_json(root / "manifest.json", manifest)
    load_analysis(data, reference(root))
    if publish:
        gold = load_gold(data, manifest["gold"]["version"])
        profile = json.loads(
            (data / "research" / manifest["profiles"]["version"] / "report.json").read_text()
        )
        products = {
            "college": profile["college_version"],
            "profiles": manifest["profiles"]["version"],
            f"outlook_{gold.manifest['current_observations']['season']}": profile[
                "outlook_version"
            ],
            "analysis": version,
        }
        publish_catalog(data, gold.manifest["version"], products)
    print(
        json.dumps(
            dict(
                version=version,
                published=publish,
                ranking_scopes=report["rankings"]["validated_scopes"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    package(args.source, args.version, publish=args.publish)
