"""Consolidate the application registry into Opening Drive v2, retaining archives."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from engine.chairstats.application import LATEST
from engine.chairstats.store import (
    CampaignStore,
    _verify_bundle,
    publish_campaign,
    replace_campaign,
)

TE_SOURCE = "opening_drive_v2_te"
RETIRED = ("opening_drive_v1", "scout_team_v1", TE_SOURCE)


def _merge_te(root: Path, archive: Path) -> dict | None:
    store = CampaignStore(root)
    current = store.campaign(LATEST)
    extension = store.campaign(TE_SOURCE)
    provenance = current.get("te_integration", {})
    if provenance.get("manifest_sha256") == extension["manifest_sha256"]:
        return None
    if any(m.get("position") == "TE" for m in current["models"]):
        raise ValueError("Opening Drive already has TE models from a different source revision")
    expected = {"te_season", "te_remaining"}
    if {m["model_id"] for m in extension["models"]} != expected or any(
        m.get("position") != "TE" for m in extension["models"]
    ):
        raise ValueError("TE integration requires both TE season and remaining models")
    store.check(LATEST)
    store.check(TE_SOURCE)
    with tempfile.TemporaryDirectory(prefix=".consolidate-", dir=root.parent) as temporary:
        staging = Path(temporary)
        models = []
        for source in (current, extension):
            source_id = source["campaign_id"]
            bundle = root / source_id
            for model in source["models"]:
                entry = dict(model)
                entry["metrics"] = bundle / model["metrics"]["path"]
                entry["artifacts"] = {
                    name: bundle / ref["path"] for name, ref in model.get("artifacts", {}).items()
                }
                entry["forecasts"] = []
                for ref in model["forecasts"]:
                    path = bundle / ref["path"]
                    if source_id == TE_SOURCE:
                        payload = json.loads(path.read_text())
                        if isinstance(payload, dict):
                            payload["campaign_id"] = LATEST
                            payload["source_campaign_id"] = TE_SOURCE
                        path = staging / model["model_id"] / Path(ref["path"]).name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
                    entry["forecasts"].append({**ref, "path": path})
                if source_id == TE_SOURCE:
                    # The fitted model receipts and weights retain their original provenance.
                    entry["source_campaign_id"] = TE_SOURCE
                    entry["source_manifest_sha256"] = extension["manifest_sha256"]
                    entry["availability"] = extension.get("availability")
                models.append(entry)
        metadata = {k: v for k, v in current.items() if k not in {"models", "manifest_sha256"}}
        metadata.update(
            positions=["QB", "RB", "WR", "TE"],
            description="Opening Drive v2 forecasts for QB, RB, WR and TE with dated availability",
            te_integration={
                "campaign_id": TE_SOURCE,
                "manifest_sha256": extension["manifest_sha256"],
                "parent_manifest_sha256": current["manifest_sha256"],
                "integrated_at": datetime.now(UTC).isoformat(),
                "source_metadata": {
                    k: v for k, v in extension.items() if k not in {"models", "manifest_sha256"}
                },
            },
        )
        for key in ("targets", "scoring"):
            metadata[key] = {**current.get(key, {}), **extension.get(key, {})}
        metadata["limitations"] = list(
            dict.fromkeys(current.get("limitations", []) + extension.get("limitations", []))
        )
        metadata["rolling_origin_rescores"] = current.get(
            "rolling_origin_rescores", []
        ) + extension.get("rolling_origin_rescores", [])
        candidates = staging / "campaigns"
        publish_campaign(candidates, metadata, models)
        CampaignStore(candidates).check(LATEST)
        return replace_campaign(root, candidates, LATEST, current["manifest_sha256"], archive)


def consolidate_opening_drive(root: Path, archive: Path) -> dict:
    """Idempotent migration; move obsolete bundles only after v2 is complete and verified.

    Empty registries are left alone. A partial installation fails before archiving
    anything, so missing TE artifacts cannot silently remove existing coverage.
    """
    import fcntl

    root, archive = Path(root).resolve(), Path(archive).resolve()
    if archive.is_relative_to(root) or root.is_relative_to(archive):
        raise ValueError("Campaign archive must be separate from the application registry")
    if not root.exists() or not any(
        p.is_dir() and not p.name.startswith(".") for p in root.iterdir()
    ):
        return {"status": "empty", "archived": []}
    with (root / ".consolidation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store = CampaignStore(root)
        manifest = store.campaign(LATEST)
        has_te = {m["model_id"] for m in manifest["models"]} >= {"te_season", "te_remaining"}
        if not has_te and not (root / TE_SOURCE).exists():
            raise ValueError("Opening Drive v2 requires the TE source bundle before consolidation")
        revision = _merge_te(root, archive) if (root / TE_SOURCE).exists() else None
        store.check(LATEST)
        retired = []
        for campaign_id in RETIRED:
            source = root / campaign_id
            if not source.exists():
                continue
            checked = store.check(campaign_id)
            destination = archive / campaign_id / checked["manifest_sha256"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                _verify_bundle(destination, campaign_id, checked["manifest_sha256"])
                shutil.rmtree(source)
            else:
                source.rename(destination)
            retired.append({"campaign_id": campaign_id, "archive": str(destination)})
        return {
            "status": "consolidated",
            "campaign_id": LATEST,
            "revision": revision,
            "archived": retired,
        }
