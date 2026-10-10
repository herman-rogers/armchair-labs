"""Import preserved pre-Opening-Drive RB research into the common campaign store.

The sealed scout_team_v1 bundle also holds QB passing forecasts imported from the
retired NextGen products; that importer no longer exists, so a rebuild is RB only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from engine.chairstats import CampaignStore, publish_campaign

CAMPAIGN_ID = "scout_team_v1"
ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(data_root: Path, output: Path) -> dict:
    from .scout_rb import prepare_rb

    if (data_root / "campaigns" / CAMPAIGN_ID).exists():
        raise FileExistsError("Scout Team v1 already exists; inspect it with engine campaigns show")
    protected = [
        data_root / "current.json",
        data_root / "campaigns/opening_drive_v1/manifest.json",
        data_root / "campaigns/opening_drive_v1/manifest.sha256",
    ]
    before = {str(p): digest(p) for p in protected if p.exists()}
    output.mkdir(parents=True, exist_ok=False)
    models = prepare_rb(data_root, output / "rb")
    metadata = {
        "campaign_id": CAMPAIGN_ID,
        "name": "Scout Team v1",
        "kind": "model_campaign",
        "status": "complete",
        "positions": ["RB", "QB"],
        "generated_at": datetime.now(UTC).isoformat(),
        "description": "Preserved RB and QB research preceding Opening Drive",
        "organization": (
            "Retrospective grouping of separate original studies; no shared-training claim"
        ),
        "bundle_replay_supported": False,
        "replay_note": "Use the original RB adapter for saved-weight inference. QB estimators were "
        "not retained; the whole campaign cannot replay weights with the Opening Drive runner.",
        "source_studies": [
            {"study_id": "rb_exploratory_research", "name": "RB Exploratory Research"},
            {"study_id": "qb_passing_research", "name": "QB Passing / Variations Research"},
        ],
        "limitations": [
            "Original RB weights are preserved; this import does not refit models.",
            "QB fitted estimators were not saved by the original workflow; only saved forecasts, "
            "evaluation evidence and source code are available.",
            "QB passing targets and RB PPR targets are different contracts; "
            "no invented QB fantasy points.",
            "Campaign organization does not change current application serving selections.",
        ],
        "import_source": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": digest(Path(__file__)),
        },
        "scoring": {
            "RB": "Original research PPR target",
            "QB": "Passing statistics; no fantasy-point target",
        },
    }
    # Check protected selections before making the new bundle visible.
    if any(digest(Path(p)) != checksum for p, checksum in before.items()):
        raise ValueError(
            "A protected current selection or Opening Drive manifest changed during import"
        )
    manifest = publish_campaign(data_root / "campaigns", metadata, models)
    result = CampaignStore(data_root / "campaigns").check(CAMPAIGN_ID)
    result.update(
        {
            "manifest": str(manifest.resolve()),
            "protected_before": before,
            "models": [
                {
                    "model_id": m["model_id"],
                    "targets": m["targets"],
                    "origins": [f["origin_week"] for f in m["forecasts"]],
                }
                for m in models
            ],
        }
    )
    (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "campaign_import",
                "status": "complete",
                "campaign_id": CAMPAIGN_ID,
                "campaign_manifest_sha256": result["manifest_sha256"],
                "files": {
                    str(path.relative_to(output)): digest(path)
                    for path in sorted(output.rglob("*"))
                    if path.is_file()
                },
            },
            indent=2,
        )
        + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True, help="New import receipt directory")
    args = parser.parse_args()
    print(json.dumps(publish(args.data_root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
