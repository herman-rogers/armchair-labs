"""Publish the fixed-weight v4 comparison with all horizons in one replacement."""

import json
import tempfile
from pathlib import Path

import pandas as pd

from engine.chairstats import CampaignStore, publish_campaign, replace_campaign

from ..weekly_v3.data import SCORING
from ..weekly_v31.artifacts import digest, now, verify, write
from .artifacts import BASE, CAMPAIGN, check


def publish(registry, integration, captures):
    registry, integration = Path(registry), Path(integration)
    contract = check(integration)
    captures = [Path(p) for p in captures]
    receipts = [verify(p) for p in captures]
    if not captures or receipts[0]["forecast_horizon"] != "early_week":
        raise ValueError("V4 publication requires its locked early capture")
    if any(r["integration_sha256"] != digest(integration / "manifest.json") for r in receipts):
        raise ValueError("Cannot mix v4 integrations")
    season, week = receipts[0]["season"], receipts[0]["week"]
    if any((r["season"], r["week"]) != (season, week) for r in receipts):
        raise ValueError("Cannot mix v4 forecast origins")
    hashes = {str(p): digest(p / "manifest.json") for p in captures}
    store = CampaignStore(registry)
    old = store.campaign(CAMPAIGN) if (registry / CAMPAIGN).exists() else None
    if old and old.get("latest_capture_hashes") == hashes:
        store.check(CAMPAIGN)
        return dict(status="current", manifest_sha256=old["manifest_sha256"])
    if old and any(
        (f["season"], f["origin_week"]) > (season, week - 1)
        for m in old["models"]
        for f in m["forecasts"]
    ):
        raise ValueError("Cannot publish an older v4 origin over a newer one")
    combined = {h: {} for h in ("weekly", "remaining", "season")}
    for capture in captures:
        projection = json.loads((capture / "projections.json").read_text())
        for horizon in combined:
            for row in projection[horizon]:
                if capture != captures[0] and row["player_id"] not in combined[horizon]:
                    raise ValueError("Late capture changes the locked eligible population")
                combined[horizon][row["player_id"]] = row
    horizons = Path(contract["dependencies"]["horizons"]["path"])
    uncertainty = Path(contract["dependencies"]["uncertainty"]["path"])
    baseline = Path(contract["dependencies"]["baseline"]["path"])
    calibration = json.loads((baseline / "dependence.json").read_text())["early_week"]
    calibration.update(
        campaign_id=CAMPAIGN,
        campaign_name="Opening Drive v4 candidate",
        mixed_horizons=len(captures) > 1,
        range_reliability_note="V4 candidate: v3.2 dependence retained; "
        "K/DST and mixed-timing lineup ranges withheld; prospective validation pending",
    )
    registry.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=registry.parent, prefix=".v4-publish-") as temporary:
        tmp = Path(temporary)
        cal_path = tmp / "lineup_calibration.json"
        write(cal_path, calibration)
        models = []
        for horizon, mapping in combined.items():
            for position in ("QB", "RB", "WR", "TE"):
                model_id = f"{position.lower()}_{horizon}"
                rows = sorted(
                    (r for r in mapping.values() if r["position"] == position),
                    key=lambda r: (-r["predictions"]["points"], r["player_id"]),
                )
                if not rows:
                    raise ValueError("V4 needs complete position/horizon coverage")
                for rank, row in enumerate(rows, 1):
                    row["rank"] = rank
                path = tmp / f"{model_id}.json"
                write(
                    path,
                    dict(
                        campaign_id=CAMPAIGN,
                        model_id=model_id,
                        season=season,
                        origin_week=week - 1,
                        horizon=horizon,
                        forecast_week=week,
                        training_through_season=2025,
                        generated_at=now().isoformat(),
                        forecast_kind="prospective_v4"
                        if horizon == "weekly"
                        else "coherent_schedule_projection",
                        retrospective=False,
                        base_campaign=BASE,
                        model_lineage="v3 -> v3.2 -> v4; learned v3-based future residuals",
                        capture_hashes=hashes,
                        rows=rows,
                        integration_manifest_sha256=digest(integration / "manifest.json"),
                        lineup_calibration_artifact=(
                            f"forecast_lineup_calibration_{season}_{week - 1}"
                        ),
                    ),
                )
                previous = (
                    next((m for m in old["models"] if m["model_id"] == model_id), None)
                    if old
                    else None
                )
                forecasts, historical_calibrations = [], {}
                if previous:
                    forecasts = [
                        dict(f, path=registry / CAMPAIGN / f["path"])
                        for f in previous["forecasts"]
                        if (f["season"], f["origin_week"]) != (season, week - 1)
                    ]
                    historical_calibrations = {
                        n: registry / CAMPAIGN / r["path"]
                        for n, r in previous["artifacts"].items()
                        if n.startswith("forecast_lineup_calibration_")
                    }
                elif horizon == "season":
                    original = store.forecast("opening_drive_v3", model_id, season, 0)
                    preserved = dict(
                        original["metadata"],
                        rows=original["rows"],
                        campaign_id=CAMPAIGN,
                        model_id=model_id,
                        horizon=horizon,
                        training_through_season=2025,
                        source_campaign="opening_drive_v3",
                        forecast_kind="preserved_preseason_comparison",
                    )
                    preseason_path = tmp / f"{model_id}_preseason.json"
                    write(preseason_path, preserved)
                    forecasts.append(dict(season=season, origin_week=0, path=preseason_path))
                forecasts.append(
                    dict(season=season, origin_week=week - 1, rolling_origin=True, path=path)
                )
                metric_path = tmp / f"{model_id}_metrics.json"
                evidence = pd.read_csv(
                    (uncertainty if horizon == "weekly" else horizons) / "scores.csv"
                )
                if "position" in evidence:
                    evidence = evidence[evidence.position.eq(position)]
                write(
                    metric_path,
                    dict(
                        scores=json.loads(evidence.to_json(orient="records")),
                        prospective="Pending future outcomes; historical years reused",
                    ),
                )
                models.append(
                    dict(
                        model_id=model_id,
                        position=position,
                        horizon=horizon,
                        origin_week=0,
                        targets=list(rows[0]["predictions"]),
                        training_through_season=2025,
                        scoring=SCORING["QB" if position == "QB" else "RB/WR/TE"],
                        metrics=metric_path,
                        forecasts=forecasts,
                        artifacts={
                            **historical_calibrations,
                            f"forecast_lineup_calibration_{season}_{week - 1}": cal_path,
                            "lineup_calibration": cal_path,
                            "projection_policy": horizons / "policy.json",
                            "projection_evaluation": horizons / "scores.csv",
                            "uncertainty_policy": uncertainty / "uncertainty.json",
                            "horizon_uncertainty_policy": integration / "horizon_uncertainty.json",
                            "horizon_uncertainty_evaluation": integration
                            / "horizon_uncertainty_scores.csv",
                            "integration_manifest": integration / "manifest.json",
                            "release_policy": integration / "release_policy.json",
                            "functionality": integration / "functionality.json",
                        },
                    )
                )
        metadata = dict(
            campaign_id=CAMPAIGN,
            name="Opening Drive v4 candidate",
            description="V3.2 weekly forecasts with evaluated v3-based horizons, "
            "component forecasts and distribution calibration",
            kind="weekly_distribution",
            status="prospective_candidate",
            base_campaign=BASE,
            default_changed=False,
            generated_at=now().isoformat(),
            latest_capture_hashes=hashes,
            horizon_contract=dict(version=1, components_required=True, base_campaign=BASE),
            limitations=json.loads((integration / "release_policy.json").read_text())[
                "limitations"
            ],
        )
        if old is None:
            publish_campaign(registry, metadata, models)
        else:
            publish_campaign(tmp / "candidate", metadata, models)
            replace_campaign(
                registry,
                tmp / "candidate",
                CAMPAIGN,
                old["manifest_sha256"],
                registry.parent / "campaign_revisions",
            )
    return dict(status="published", verification=store.check(CAMPAIGN))
