"""Append a weekly forecast using sealed v3 weights; never fit or replace an origin."""

from __future__ import annotations

import importlib.metadata
import json
import pickle
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine.chairstats import CampaignStore, replace_campaign

from . import data, modeling
from .modeling import QCOLS, QUANTILES, predict_distribution, predict_mean


def due_models(manifest, season, origin):
    return [
        m
        for m in manifest["models"]
        if m["horizon"] == "weekly"
        and not any(
            f["season"] == season and f["origin_week"] == origin for f in m.get("forecasts", [])
        )
    ]


def validate_snapshot(snapshot, season, origin):
    document = json.loads(Path(snapshot).read_text())
    current = document.get("current_observations", {})
    if document.get("status") != "accepted":
        raise ValueError("Weekly refresh requires an accepted raw snapshot")
    if season != 2026:
        raise ValueError("V3 saved weekly models support 2026; a new season requires retraining")
    if current.get("season") != season or current.get("through_week") != origin:
        raise ValueError("Weekly origin must equal the accepted snapshot's completed week")
    if not 0 <= origin < 18:
        raise ValueError("No regular-season weekly forecast follows this origin")
    return document


def build_live_panel(snapshot, sources, registry, output_dir):
    """Reuse sealed feature code with dated runtime inputs; restore module defaults."""
    previous = data.SNAPSHOT, data.AVAILABILITY, data.CAMPAIGN
    try:
        data.SNAPSHOT = Path(snapshot)
        if sources is not None:
            data.AVAILABILITY = Path(sources)
        data.CAMPAIGN = Path(registry) / "opening_drive_v2"
        panel, features, receipt = data.build_panel(
            output_dir=output_dir, start_year=2026, end_year=2026
        )
    finally:
        data.SNAPSHOT, data.AVAILABILITY, data.CAMPAIGN = previous
    week = receipt["current_prediction_week"]
    panel = panel[panel.season.eq(2026) & panel.week.eq(week)].copy()
    if panel.empty or panel.target_points.notna().any():
        raise ValueError("Live weekly feature matrix must be nonempty with unobserved outcomes")
    if panel.duplicated(["position", "player_id"]).any():
        raise ValueError("Duplicate player identities in live weekly feature matrix")
    path = Path(output_dir) / "panel.parquet"
    panel.to_parquet(path, index=False)
    receipt.update(rows=len(panel), panel_sha256=data.digest(path), features=features)
    receipt["baseline"] = (
        f"Matching v2 forecast at origin {week - 1} for live week {week} where available; "
        "otherwise earlier-only persistence fallback, identified by v2_exact_available."
    )
    receipt["live_exact_v2_fraction"] = float(panel.v2_exact_available.mean())
    (Path(output_dir) / "dataset.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return panel, receipt


def _verified_content(bundle, entry, name):
    ref = entry["artifacts"][name]
    path = bundle / ref["path"]
    if data.digest(path) != ref["sha256"]:
        raise ValueError(f"Sealed weekly artifact checksum changed: {name}")
    return path.read_bytes()


def _saved_models(bundle, entry):
    for name, path in (
        ("source_modeling", Path(modeling.__file__)),
        ("source_data", Path(data.__file__)),
    ):
        if _verified_content(bundle, entry, name) != path.read_bytes():
            raise ValueError(f"Weekly refresh source differs from sealed contract: {name}")
    contract = json.loads(_verified_content(bundle, entry, "run_manifest"))
    for package, version in contract["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Weekly refresh package differs from sealed contract: {package}")
    # Checksummed, locally generated trusted pickle artifacts, never external input.
    return (
        pickle.loads(_verified_content(bundle, entry, "mean_model")),
        pickle.loads(_verified_content(bundle, entry, "distribution_model")),
    )


def _rows(panel, entry, mean, spread, season, origin):
    frame = panel[panel.position.eq(entry["position"])].copy()
    if frame.duplicated("player_id").any():
        raise ValueError("Duplicate player identities in weekly forecast")
    required = set(mean["features"]) | (set(spread["features"]) - {"prediction"})
    if required - set(frame.columns):
        raise ValueError(
            f"Missing saved weekly feature columns: {sorted(required - set(frame.columns))}"
        )
    scheduled = frame.scheduled_game.eq(1)
    byes = frame.is_bye.eq(1)
    frame = frame[scheduled | byes].copy()
    scheduled = frame.scheduled_game.eq(1)
    frame["prediction"], frame["std"] = 0.0, 0.0
    frame[QCOLS] = 0.0
    if scheduled.any():
        live = frame.loc[scheduled].copy()
        live["prediction"] = predict_mean(mean, live)
        q, std = predict_distribution(spread, live)
        frame.loc[scheduled, "prediction"] = live.prediction.to_numpy()
        frame.loc[scheduled, "std"] = std
        frame.loc[scheduled, QCOLS] = q
    if not np.isfinite(frame[["prediction", "std", *QCOLS]].to_numpy()).all():
        raise ValueError("Nonfinite weekly refresh forecast")
    rows = []

    def optional(value):
        return None if pd.isna(value) else value

    for rank, (_, row) in enumerate(frame.sort_values("prediction", ascending=False).iterrows(), 1):
        q = row[QCOLS].to_numpy(dtype=float)
        bye = bool(row.is_bye)
        distribution = {c: float(row[c]) for c in QCOLS}
        distribution.update(
            std=float(row["std"]),
            samples=np.interp((np.arange(256) + 0.5) / 256, QUANTILES, q).tolist(),
            mean_method="scheduled_bye" if bye else mean["method"],
            distribution_method="deterministic_zero" if bye else spread["method"],
        )
        rows.append(
            dict(
                player_id=row.player_id,
                player_name=optional(row.get("player_name")),
                position=entry["position"],
                season=season,
                origin_week=origin,
                team=optional(row.get("team")),
                opponent=optional(row.get("opponent")),
                predictions={"points": float(row.prediction)},
                scheduled_games=int(row.scheduled_game),
                calendar_weeks=1,
                points_per_scheduled_game=float(row.prediction),
                rank=rank,
                distribution=distribution,
                candidate_basis="Earlier-only weekly eligibility; known byes zero",
            )
        )
    return rows


def rescore_weekly(store, manifest, *, season, origin, snapshot, sources, registry, archive):
    """Archive the previous revision and append missing weekly forecast origins."""
    validate_snapshot(snapshot, season, origin)
    targets = due_models(manifest, season, origin)
    if not targets:
        return {"campaign_id": manifest["campaign_id"], "status": "current"}
    campaign_id = manifest["campaign_id"]
    registry = Path(registry)
    bundle = registry / campaign_id
    store.check(campaign_id)
    saved = {entry["model_id"]: _saved_models(bundle, entry) for entry in targets}
    with tempfile.TemporaryDirectory(prefix=".weekly-rescore-", dir=registry.parent) as temporary:
        candidate_root = Path(temporary)
        candidate = candidate_root / campaign_id
        shutil.copytree(bundle, candidate)
        inputs_relative = f"rolling_inputs/{season}_{origin}"
        inputs = candidate / inputs_relative
        if inputs.exists():
            raise ValueError("Refusing to overwrite an existing weekly feature matrix")
        inputs.mkdir(parents=True)
        panel, dataset = build_live_panel(snapshot, sources, registry, inputs)
        if not (panel.season.eq(season) & panel.week.eq(origin + 1)).all():
            raise ValueError("Weekly feature matrix crosses the requested forecast origin")
        if panel.target_points.notna().any():
            raise ValueError("Weekly forecast outcome is already observed in feature matrix")
        revised = json.loads((candidate / "manifest.json").read_text())
        added = []
        matrix_key = f"forecast_matrix_{season}_{origin}"
        now = datetime.now(UTC).isoformat()
        for entry in targets:
            model_id = entry["model_id"]
            mean, spread = saved[model_id]
            rows = _rows(panel, entry, mean, spread, season, origin)
            relative = f"{model_id}/forecasts/{season}_{origin}.json"
            path = candidate / relative
            if path.exists():
                raise ValueError("Refusing to overwrite an existing weekly forecast")
            payload = dict(
                campaign_id=campaign_id,
                model_id=model_id,
                season=season,
                origin_week=origin,
                forecast_week=origin + 1,
                horizon="weekly",
                training_through_season=entry["training_through_season"],
                generated_at=now,
                forecast_kind="saved_weekly_model_rescore",
                feature_matrix_artifact=matrix_key,
                snapshot_sha256=data.digest(snapshot),
                baseline_exact_fraction=float(
                    panel.loc[panel.position.eq(entry["position"]), "v2_exact_available"].mean()
                ),
                baseline_policy=(
                    "Exact matching v2 origin where available; earlier-only persistence "
                    "fallback otherwise. Refresh v2 first for the latest baseline."
                ),
                rows=rows,
                calibration_policy="Original sealed calibration retained; no new errors consumed",
                retrospective=True,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            revised_entry = next(m for m in revised["models"] if m["model_id"] == model_id)
            revised_entry["forecasts"].append(
                dict(
                    season=season,
                    origin_week=origin,
                    rolling_origin=True,
                    path=relative,
                    sha256=data.digest(path),
                    feature_matrix_artifact=matrix_key,
                )
            )
            for key, name in (
                (matrix_key, "panel.parquet"),
                (f"forecast_dataset_{season}_{origin}", "dataset.json"),
            ):
                revised_entry["artifacts"][key] = dict(
                    path=f"{inputs_relative}/{name}", sha256=data.digest(inputs / name)
                )
            added.append(dict(model_id=model_id, rows=len(rows)))
        revised.setdefault("weekly_rescores", []).append(
            dict(
                season=season,
                origin_week=origin,
                generated_at=now,
                models=added,
                snapshot_sha256=data.digest(snapshot),
                panel_sha256=dataset["panel_sha256"],
                refit=False,
                calibration="Original sealed calibration retained; no new K/DST errors consumed",
            )
        )
        (candidate / "manifest.json").write_text(
            json.dumps(revised, indent=2, allow_nan=False) + "\n"
        )
        (candidate / "manifest.sha256").write_text(data.digest(candidate / "manifest.json") + "\n")
        receipt = replace_campaign(
            registry, candidate_root, campaign_id, manifest["manifest_sha256"], Path(archive)
        )
    CampaignStore(registry).check(campaign_id)
    return dict(campaign_id=campaign_id, status="rescored", models=added, receipt=receipt)
