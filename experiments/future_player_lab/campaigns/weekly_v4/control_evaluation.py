"""Freeze the matching v3.2 horizon control before the prospective deadline.

Controls are separate immutable artifacts. They retain the baseline's campaign
identity and source receipts, and never reconstruct a missed historical forecast.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from engine.chairstats import CampaignStore

from ..weekly_v31.artifacts import digest, now, publish_directory, seal, timestamp, verify, write
from .horizon_evaluation import score_projection

BASE = "opening_drive_v32"


def capture_controls(directory, registry, log):
    directory, registry, log = map(Path, (directory, registry, log))
    receipt = verify(directory)
    if receipt.get("status") != "prospective" or not (
        timestamp(receipt["evidence_cutoff"])
        <= timestamp(receipt["generated_at"])
        < timestamp(receipt["deadline"])
    ):
        raise ValueError("Control pairing requires a valid prospective v4 capture")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", directory.name):
        raise ValueError("Invalid prospective capture name")
    season, week = int(receipt["season"]), int(receipt["week"])
    relative = Path(f"{season}_{week:02d}") / directory.name
    destination = log / "controls" / relative
    status_path = log / "control_status" / relative / "status.json"
    paired_hash = digest(directory / "manifest.json")

    def unavailable(reason, detail=None):
        result = dict(
            status="unavailable",
            reason=reason,
            detail=detail,
            paired_capture=str(directory),
            paired_capture_sha256=paired_hash,
            deadline=receipt["deadline"],
            checked_at=now().isoformat(),
        )
        write(status_path, result)
        return result

    if destination.exists():
        saved = verify(destination)
        if saved.get("paired_capture_sha256") != paired_hash:
            raise ValueError("Existing control belongs to a different v4 capture")
        return dict(status="current", path=str(destination), paired_capture_sha256=paired_hash)
    deadline = timestamp(receipt["deadline"])
    if now() >= deadline:
        return unavailable("missed_prospective_deadline")
    source = pd.read_parquet(directory / "baseline/players.parquet")
    if (
        source.player_id.duplicated().any()
        or not source.season.eq(season).all()
        or not source.week.eq(week).all()
    ):
        raise ValueError("Paired baseline contains inconsistent forecast identities")
    projection = json.loads((directory / "projections.json").read_text())
    eligible = set(source.player_id)
    if any(
        {r["player_id"] for r in projection[h]} != eligible
        for h in ("weekly", "remaining", "season")
    ):
        raise ValueError("Paired v4 horizon populations disagree")
    store = CampaignStore(registry)
    try:
        manifest = store.campaign(BASE)
        baseline = source.set_index("player_id")
        collected, references = {h: [] for h in ("weekly", "remaining", "season")}, []
        for horizon in collected:
            for position in sorted(source.position.unique()):
                model_id = f"{position.lower()}_{horizon}"
                forecast = store.forecast(BASE, model_id, season, week - 1)
                metadata = forecast["metadata"]
                if (
                    receipt["base_capture_sha256"]
                    not in metadata.get("capture_hashes", {}).values()
                ):
                    return unavailable("baseline_capture_mismatch", model_id)
                if (metadata.get("season"), metadata.get("origin_week")) != (season, week - 1):
                    return unavailable("baseline_origin_mismatch", model_id)
                mapping = {r["player_id"]: r for r in forecast["rows"]}
                ids = set(source.loc[source.position.eq(position), "player_id"])
                if not ids <= set(mapping):
                    return unavailable("baseline_population_missing", model_id)
                rows = [mapping[pid] for pid in sorted(ids)]
                if horizon == "weekly":
                    for row in rows:
                        base = baseline.loc[row["player_id"]]
                        draws = row.get("distribution", {}).get("samples", [])
                        if (
                            not np.isclose(
                                row["predictions"]["points"], base.prediction, rtol=0, atol=1e-10
                            )
                            or np.asarray(draws).shape != np.asarray(base.samples).shape
                            or not np.allclose(draws, base.samples, rtol=0, atol=1e-10)
                        ):
                            return unavailable(
                                "baseline_weekly_forecast_replaced", row["player_id"]
                            )
                collected[horizon].extend(rows)
                model = next(m for m in manifest["models"] if m["model_id"] == model_id)
                ref = next(
                    f
                    for f in model["forecasts"]
                    if (f["season"], f["origin_week"]) == (season, week - 1)
                )
                path = (registry / BASE / ref["path"]).resolve()
                if (
                    not path.is_relative_to((registry / BASE).resolve())
                    or digest(path) != ref["sha256"]
                ):
                    return unavailable("baseline_payload_changed", model_id)
                references.append(
                    dict(model_id=model_id, path=str(path), sha256=ref["sha256"], metadata=metadata)
                )
        if digest(registry / BASE / "manifest.json") != manifest["manifest_sha256"]:
            return unavailable("baseline_registry_changed_during_capture")
    except (LookupError, FileNotFoundError, ValueError, StopIteration) as exc:
        return unavailable("matching_baseline_unavailable", str(exc))
    if now() >= deadline:
        return unavailable("deadline_passed_during_capture")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".v32-control-") as tmp:
        staging = Path(tmp) / "control"
        staging.mkdir()
        write(staging / "projections.json", collected)
        write(staging / "baseline_sources.json", references)
        for reference in references:
            target = staging / "sources" / f"{reference['model_id']}.json"
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(reference["path"], target)
            if digest(target) != reference["sha256"]:
                return unavailable("baseline_payload_changed_during_copy", reference["model_id"])
        shutil.copy2(directory / "manifest.json", staging / "paired_capture_receipt.json")
        generated = now()
        if generated >= deadline:
            return unavailable("deadline_passed_during_capture")
        seal(
            staging,
            dict(
                status="prospective",
                kind="matched_v32_horizon_control",
                source_campaign=BASE,
                paired_campaign="opening_drive_v4",
                paired_capture=str(directory),
                paired_capture_sha256=paired_hash,
                baseline_manifest_sha256=manifest["manifest_sha256"],
                base_capture_sha256=receipt["base_capture_sha256"],
                season=season,
                week=week,
                forecast_horizon=receipt["forecast_horizon"],
                evidence_cutoff=receipt["evidence_cutoff"],
                generated_at=generated.isoformat(),
                deadline=receipt["deadline"],
                source_policy="Exact baseline payloads copied before deadline; "
                "no refit, recalculation or historical backfill",
                helper_source_sha256=digest(__file__),
            ),
        )
        if now() >= deadline:
            return unavailable("deadline_passed_before_publication")
        publish_directory(staging, destination)
    result = dict(status="captured", path=str(destination), paired_capture_sha256=paired_hash)
    write(status_path, dict(result, checked_at=now().isoformat()))
    return result


def score_controls(directory, snapshot):
    directory = Path(directory)
    receipt = verify(directory)
    if (
        receipt.get("kind") != "matched_v32_horizon_control"
        or receipt.get("source_campaign") != BASE
    ):
        raise ValueError("Expected an immutable matched v3.2 control")
    frame = score_projection(directory, snapshot)
    frame["method"] = "v32"
    frame["paired_capture"] = receipt["paired_capture"]
    frame["paired_capture_sha256"] = receipt["paired_capture_sha256"]
    return frame
