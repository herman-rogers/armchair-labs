"""Publish v4.1 weekly/remaining/season forecasts as the `opening_drive_v41` campaign.

Dev posture (2026-10-27): no promotion gates. Each publish keeps every earlier
forecast origin already in the bundle; republishing the same origin replaces it.
Release coherence checks only warn for this campaign (`horizons.ENFORCED`).
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from engine.chairstats import CampaignStore, publish_campaign, replace_campaign

from ..weekly_v3.data import ROOT, SCORING
from ..weekly_v31.artifacts import digest, now

CAMPAIGN = "opening_drive_v41"
NAME = "Opening Drive v4.1"
RANGE_NOTE = "V4.1: K/DST lineup ranges withheld"
POSITIONS = ("QB", "RB", "WR", "TE")
HORIZONS = ("weekly", "remaining", "season")
TRAINING_THROUGH_SEASON = 2025
OUTPUTS = ROOT / "data/outputs/opening_drive_v41"
ARCHIVE = ROOT / "data/campaign_revisions"
LIMITATIONS = [
    "V4.1 is a development model; earlier versions remain available as comparisons.",
    "2021-25 evaluation years were seen during earlier development; results are "
    "retrospective, not an untouched test.",
    "Historical early-week market lines are closing lines; prospective weeks are the honest test.",
    "Lineup correlations come from v4.1 out-of-time weekly errors; K/DST ranges are withheld.",
]


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NaT:
        return None
    return value


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")


def save_outputs(players, projection, horizon, root=OUTPUTS):
    """Latest capture for one origin/timing; overwritten on recompute."""
    first = projection["weekly"][0]
    week = int(first["origin_week"]) + 1
    destination = Path(root) / "forecasts" / f"{int(first['season'])}_{week:02d}" / horizon
    destination.mkdir(parents=True, exist_ok=True)
    players.to_parquet(destination / "players.parquet", index=False)
    _write(destination / "projections.json", projection)
    return destination


def lineup_calibration(research_run, cache=OUTPUTS / "lineup_calibration.json"):
    """Weekly lineup dependence from v4.1's own early-week out-of-time residuals.

    Same estimator as v3.2 (`weekly_v32.uncertainty.dependence`). Falls back to
    independence (no correlations) when the run lacks the inputs.
    """
    research_run = Path(research_run)
    key = (
        digest(research_run / "manifest.json")
        if (research_run / "manifest.json").exists()
        else None
    )
    cache = Path(cache) if cache else None
    if cache and key and cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("research_manifest_sha256") == key:
            # Labels are presentation, not calibration: refresh them on a cache hit.
            saved.update(campaign_name=NAME, range_reliability_note=RANGE_NOTE)
            return saved
    try:
        from ..weekly_v32.uncertainty import dependence

        frame = pd.read_parquet(research_run / "predictions.parquet")
        frame = frame[frame.method.eq("selected") & frame.horizon.eq("early_week")]
        if "opponent" not in frame:
            panel = pd.read_parquet(research_run / "panel.parquet", columns=["row_id", "opponent"])
            frame = frame.merge(panel.drop_duplicates("row_id"), on="row_id", how="left")
        calibration = dependence(frame, 2026)
        calibration["note"] = "V4.1 out-of-time early-week residuals; K/DST uncertainty withheld"
    except (OSError, KeyError, ValueError) as error:
        calibration = dict(
            covariance_before_season=2026,
            correlations=[],
            special_teams={},
            note=f"Independent players: v4.1 dependence unavailable ({error})",
        )
    calibration.update(
        campaign_id=CAMPAIGN,
        campaign_name=NAME,
        research_manifest_sha256=key,
        range_reliability_note=RANGE_NOTE,
    )
    if cache and key:
        _write(cache, calibration)
    return calibration


def _metrics(research_run, horizon, position):
    name = "uncertainty_scores.csv" if horizon == "weekly" else "horizon_scores.csv"
    path = Path(research_run) / name
    scores = []
    if path.exists():
        frame = pd.read_csv(path)
        if "position" in frame:
            frame = frame[frame.position.eq(position)]
        scores = json.loads(frame.to_json(orient="records"))
    return dict(scores=scores, source=str(path), prospective="Pending future outcomes")


def _policy_artifacts(research_run):
    run = Path(research_run)
    names = {
        "projection_policy": "protocol.json",
        "projection_evaluation": "judging_horizons.csv",
        "weekly_evaluation": "judging_weekly.csv",
        "horizon_scores": "horizon_scores.csv",
        "uncertainty_scores": "uncertainty_scores.csv",
        "stage1_selection": "stage1_selection.json",
        "selection": "selection.json",
        "research_manifest": "manifest.json",
    }
    return {k: run / v for k, v in names.items() if (run / v).exists()}


def publish(
    registry,
    projection,
    players,
    research_run,
    snapshot,
    archive=ARCHIVE,
    outputs=OUTPUTS,
    retrospective=False,
):
    """Publish (first time) or replace the v4.1 bundle with this origin's forecasts.

    ``retrospective`` marks a backfilled earlier origin; it never replaces the bundle's
    latest capture receipt.
    """
    registry, research_run, snapshot = Path(registry), Path(research_run), Path(snapshot)
    weekly_rows = projection["weekly"]
    if not weekly_rows:
        raise ValueError("V4.1 projection has no weekly rows")
    season, origin = int(weekly_rows[0]["season"]), int(weekly_rows[0]["origin_week"])
    timing = sorted({str(h) for h in players.get("forecast_horizon", pd.Series(dtype=str))})
    capture_hashes = dict(
        snapshot_sha256=digest(snapshot),
        research_manifest_sha256=(
            digest(research_run / "manifest.json")
            if (research_run / "manifest.json").exists()
            else None
        ),
        timing=",".join(timing),
    )
    store = CampaignStore(registry)
    old = store.campaign(CAMPAIGN) if (registry / CAMPAIGN).exists() else None
    generated = now().isoformat()
    registry.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=registry.parent, prefix=".v41-publish-") as temporary:
        tmp = Path(temporary)
        calibration = tmp / "lineup_calibration.json"
        _write(
            calibration,
            lineup_calibration(research_run, Path(outputs) / "lineup_calibration.json"),
        )
        policies = _policy_artifacts(research_run)
        models = []
        for horizon in HORIZONS:
            for position in POSITIONS:
                model_id = f"{position.lower()}_{horizon}"
                rows = sorted(
                    (r for r in projection[horizon] if r["position"] == position),
                    key=lambda r: (-r["predictions"]["points"], r["player_id"]),
                )
                if not rows:
                    raise ValueError(f"V4.1 projection lacks {position} {horizon} rows")
                for rank, row in enumerate(rows, 1):
                    row["rank"] = rank
                path = tmp / f"{model_id}.json"
                _write(
                    path,
                    dict(
                        campaign_id=CAMPAIGN,
                        model_id=model_id,
                        season=season,
                        origin_week=origin,
                        horizon=horizon,
                        forecast_week=origin + 1,
                        training_through_season=TRAINING_THROUGH_SEASON,
                        generated_at=generated,
                        forecast_kind="opening_drive_v41_weekly"
                        if horizon == "weekly"
                        else "coherent_schedule_projection",
                        retrospective=retrospective,
                        base_campaign=CAMPAIGN,
                        model_lineage=projection.get("provenance", {}).get("lineage"),
                        provenance=projection.get("provenance"),
                        capture_hashes=capture_hashes,
                        rows=rows,
                    ),
                )
                previous = (
                    next((m for m in old["models"] if m["model_id"] == model_id), None)
                    if old
                    else None
                )
                forecasts = [
                    dict(f, path=registry / CAMPAIGN / f["path"])
                    for f in (previous["forecasts"] if previous else [])
                    if (f["season"], f["origin_week"]) != (season, origin)
                ]
                forecasts.append(
                    dict(season=season, origin_week=origin, rolling_origin=True, path=path)
                )
                forecasts.sort(key=lambda f: (f["season"], f["origin_week"]))
                metrics = tmp / f"{model_id}_metrics.json"
                _write(metrics, _metrics(research_run, horizon, position))
                models.append(
                    dict(
                        model_id=model_id,
                        position=position,
                        horizon=horizon,
                        origin_week=0,
                        targets=list(rows[0]["predictions"]),
                        training_through_season=TRAINING_THROUGH_SEASON,
                        scoring=SCORING["QB" if position == "QB" else "RB/WR/TE"],
                        metrics=metrics,
                        forecasts=forecasts,
                        artifacts={"lineup_calibration": calibration, **policies},
                    )
                )
        metadata = dict(
            campaign_id=CAMPAIGN,
            name=NAME,
            description="Full v4.1 retrain: v3 weekly method with market lines, v3.2 "
            "injury/workload layer and v4 horizon/component method, all fitted in one run",
            kind="weekly_distribution",
            status="active",
            base_campaign=CAMPAIGN,
            default_changed=False,
            generated_at=generated,
            research_run=str(research_run),
            latest_capture_hashes=(
                old["latest_capture_hashes"]
                if retrospective and old and old.get("latest_capture_hashes")
                else capture_hashes
            ),
            horizon_contract=dict(version=1, components_required=True, base_campaign=CAMPAIGN),
            limitations=LIMITATIONS,
        )
        if old is None:
            publish_campaign(registry, metadata, models)
        else:
            publish_campaign(tmp / "candidate", metadata, models)
            replace_campaign(registry, tmp / "candidate", CAMPAIGN, old["manifest_sha256"], archive)
    return dict(
        status="published",
        season=season,
        origin_week=origin,
        timing=timing,
        verification=store.check(CAMPAIGN),
    )
