"""Add rolling-origin forecasts for the latest completed week to published campaigns.

The sealed models were trained to forecast from one cutoff (preseason, or after
Week 2). A rolling re-score builds the same feature rows at a later completed week
and applies the saved weights. That is a deliberate contract extension: the weights
never saw later cutoffs, so every re-scored forecast is labelled ``rolling_origin``
and carries that limitation. The campaign bundle is replaced in place through
``engine.chairstats.replace_campaign``; the previous revision is archived.

Nothing in the sealed training modules is modified. Features come from the same
functions the campaigns used (``feature_rows``, ``snapshot_inputs``,
``add_supplements``, ``attach_availability``); only the origin guard in ``predict``
is relaxed here, on purpose.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from engine.chairstats import CampaignStore, replace_campaign
from engine.chairstats.consolidate import LATEST, consolidate_opening_drive

from ..forecasting.data import add_supplements, history_inputs, snapshot_inputs
from ..gpu_lab.run import digest, json_write
from ..notebooks import raw_distribution_data as raw
from .availability import data as availability_data
from .availability import espn, layer
from .data import ROOT, feature_rows, points
from .qb_features import augment_qb
from .run import clamp, matrix
from .runtime import verify_environment

REGISTRY = ROOT / "data/campaigns"
ARCHIVE = ROOT / "data/campaign_revisions"
RUNS = ROOT / "experiments/future_player_lab/runs"
HISTORY_CACHE = availability_data.V1_RUN / "inputs/history"
SOURCES_ROOT = ROOT / "data/research"


# ----------------------------------------------------------------- inputs


def latest_raw_snapshot(data_root: Path = ROOT / "data") -> Path:
    candidates = []
    for manifest in (data_root / "raw/snapshots").glob("*/manifest.json"):
        document = json.loads(manifest.read_text())
        if document.get("status") == "accepted" and "current_observations" in document:
            candidates.append((document.get("registered_at", ""), manifest))
    if not candidates:
        raise FileNotFoundError("No accepted raw snapshot with current observations")
    return max(candidates)[1]


def latest_supplements(season: int, saved_at: str) -> Path | None:
    """The newest complete supplement capture for ``season`` taken after the snapshot."""
    best = None
    for manifest in RUNS.glob("forecast_sources_*/manifest.json"):
        document = json.loads(manifest.read_text())
        if document.get("status") != "complete" or document.get("season") != season:
            continue
        if document.get("captured_at", "") < saved_at:
            continue
        if best is None or document["captured_at"] > best[0]:
            best = (document["captured_at"], manifest.parent)
    return best[1] if best else None


def capture_supplements(season: int) -> Path:
    from ..forecasting.capture import capture

    output = RUNS / f"forecast_sources_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    print(f"Capturing current-season supplements into {output.name}", flush=True)
    return capture(output, season)


def current_schedule(snapshot: Path) -> pl.DataFrame:
    manifest = json.loads(snapshot.read_text())
    asset = manifest["assets"]["current/schedule.parquet"]
    return pl.read_parquet(ROOT / "data" / asset["object"])


def refresh_availability_sources(season: int, supplements: Path, snapshot: Path) -> Path:
    """Rebuild dated availability events through the latest capture, reusing old pages.

    Mirrors ``availability.data.prepare_sources`` but with a fresh ESPN transaction
    capture for the current year and current-season rosters/injuries from the
    newest supplement run. Earlier years' pages and receipts are copied, so only the
    current year is downloaded again.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    directory = SOURCES_ROOT / f"opening_drive_v2_sources_rolling_{stamp}"
    pages = directory / "espn"
    pages.mkdir(parents=True)
    previous = availability_data.SOURCE_DIR / "espn"
    for path in previous.glob("*.json"):
        year = int(path.name.split("_")[0]) if path.name[:4].isdigit() else None
        if year is not None and year < season:
            shutil.copyfile(path, pages / path.name)
    espn.capture(pages, 2002, season)
    manifest = json.loads((pages / "manifest.json").read_text())
    roster = raw.read_family("rosters")
    injuries = raw.read_family("injuries")
    identity = raw.read_family("identity")
    schedule = pl.concat(
        [raw.read_family("schedule"), current_schedule(snapshot)], how="diagonal_relaxed"
    ).unique(["game_id"], keep="last")
    supplement_manifest = json.loads((supplements / "manifest.json").read_text())
    for name in ("rosters", "injuries"):
        receipt = supplement_manifest["sources"][name]
        if receipt["status"] != "available":
            continue
        frame = availability_data.checked_parquet(supplements, receipt["path"], receipt["sha256"])
        if name == "rosters":
            roster = pl.concat([roster, frame], how="diagonal_relaxed")
        else:
            injuries = pl.concat([injuries, frame], how="diagonal_relaxed")
    aliases, _ = layer.name_index(identity, roster)
    transactions, rejected = espn.parse_reports(
        json.loads((pages / "reports.json").read_text()), aliases
    )
    events, matching = layer.build_events(transactions, roster, identity, schedule)
    matching["unparsed_transaction_clauses"] = rejected
    matching["transaction_coverage"] = manifest["coverage"]
    events.write_parquet(directory / "events.parquet")
    injuries.write_parquet(directory / "injuries.parquet")
    schedule.write_parquet(directory / "schedule.parquet")
    json_write(directory / "matching.json", matching)
    json_write(
        directory / "manifest.json",
        dict(
            schema_version=1,
            kind="availability_sources_rolling",
            season=season,
            transaction_manifest_sha256=digest(pages / "manifest.json"),
            supplement_manifest_sha256=digest(supplements / "manifest.json"),
            snapshot_sha256=digest(snapshot),
            files={
                name: digest(directory / name)
                for name in [
                    "events.parquet",
                    "injuries.parquet",
                    "schedule.parquet",
                    "matching.json",
                ]
            },
            limitations=[
                "Retrospective event archive; event dates are not archived publication times.",
                "Rebuilt for a rolling re-score; the training-time sources are unchanged.",
            ],
        ),
    )
    return directory


# ----------------------------------------------------------------- scoring


def predict_rolling(model: dict, panel: pl.DataFrame) -> dict[str, np.ndarray]:
    """``run.predict`` without the origin guard: the saved weights see a later cutoff."""
    if set(panel["horizon"].to_list()) != {model["horizon"]}:
        raise ValueError("Horizon does not match trained model contract")
    if set(panel["position"].to_list()) != {model["position"]}:
        raise ValueError("Position does not match trained model contract")
    if panel["year"].min() <= model["training_through_season"]:
        raise ValueError("Prediction season overlaps final training data")
    x = matrix(panel, model["feature_names"])
    exposure = panel["weeks"].to_numpy()
    outputs = {}
    for target, members in model["models"].items():
        outputs[target] = np.mean(
            [clamp(m["model"].predict(x[:, m["indices"]]) * exposure, target) for m in members],
            axis=0,
        )
    outputs["component_points"] = points(model["position"], outputs)
    return outputs


def forecast_rows(model_entry: dict, panel: pl.DataFrame, outputs: dict, season: int, origin: int):
    ranks = np.empty(panel.height, dtype=int)
    ranks[np.argsort(-outputs["points"], kind="stable")] = np.arange(1, panel.height + 1)
    rows = []
    for index, row in enumerate(panel.iter_rows(named=True)):
        games = int(row["scheduled_games"]) if row["scheduled_games"] is not None else None
        entry = {
            "player_id": row["player_id"],
            "player_name": row["player_display_name"],
            "team": row["team"],
            "position": model_entry["position"],
            "season": season,
            "origin_week": origin,
            "horizon": model_entry["horizon"],
            "rank": int(ranks[index]),
            "scheduled_games": games,
            "calendar_weeks": int(row["weeks"]),
            "candidate_basis": row["candidate_basis"],
            "predictions": {t: float(outputs[t][index]) for t in model_entry["targets"]},
            "component_points": float(outputs["component_points"][index]),
            "points_per_scheduled_game": float(outputs["points"][index] / games) if games else None,
        }
        if "availability_status" in row:
            entry["availability"] = {
                key: row.get("availability_" + key)
                for key in ("status", "team", "observed_at", "as_of", "evidence")
            }
        rows.append(entry)
    rows.sort(key=lambda r: r["rank"])
    return rows


def rescore_campaign(
    store: CampaignStore,
    manifest: dict,
    *,
    season: int,
    origin: int,
    snapshot: Path,
    supplements: Path,
    sources: Path | None,
    registry: Path,
    archive: Path,
) -> dict:
    campaign_id = manifest["campaign_id"]
    bundle = registry / campaign_id
    targets = [
        m
        for m in manifest["models"]
        if m["horizon"] == "remaining"
        and "weights" in m.get("artifacts", {})
        and not any(
            f["season"] == season and f["origin_week"] == origin for f in m.get("forecasts", [])
        )
    ]
    if not targets:
        return {"campaign_id": campaign_id, "status": "current"}
    tables = history_inputs(HISTORY_CACHE)
    tables, snapshot_audit = snapshot_inputs(tables, snapshot, season, origin)
    tables, supplement_audit = add_supplements(tables, supplements, season, origin)
    availability = None
    if manifest.get("availability"):
        if sources is None:
            raise ValueError(f"{campaign_id} needs rebuilt availability sources")
        availability, _ = availability_data.load_sources(sources)
    with tempfile.TemporaryDirectory(prefix=".rescore-", dir=registry.parent) as temporary:
        candidate_root = Path(temporary)
        candidate = candidate_root / campaign_id
        shutil.copytree(bundle, candidate)
        revised = json.loads((candidate / "manifest.json").read_text())
        added = []
        for entry in targets:
            revised_entry = next(m for m in revised["models"] if m["model_id"] == entry["model_id"])
            saved = json.loads((bundle / entry["artifacts"]["model_manifest"]["path"]).read_text())
            verify_environment(saved, check_sources=False)
            model = joblib.load(bundle / entry["artifacts"]["weights"]["path"])
            panel = feature_rows(tables, entry["position"], season, origin)
            if any(name.startswith("qb__") for name in model["feature_names"]):
                # Revised QB cells add explicit efficiency/opportunity features.
                panel = augment_qb(panel, tables)
            if availability is not None:
                audit = availability_data.augment(panel, availability)
                panel = audit.filter(pl.col("availability_eligible"))
            missing = [name for name in model["feature_names"] if name not in panel.columns]
            if missing:
                raise ValueError(
                    f"{entry['model_id']} lacks features at week {origin}: {missing[:5]}"
                )
            outputs = predict_rolling(model, panel)
            if not all(np.isfinite(values).all() for values in outputs.values()):
                raise ValueError(f"Nonfinite rolling forecast in {entry['model_id']}")
            rows = forecast_rows(entry, panel, outputs, season, origin)
            relative = f"{entry['model_id']}/forecasts/{season}_{origin}.json"
            receipt = {
                "schema_version": 1,
                "campaign_id": campaign_id,
                "model_id": entry["model_id"],
                "season": season,
                "origin_week": origin,
                "horizon": entry["horizon"],
                "generated_at": datetime.now(UTC).isoformat(),
                "disposition": manifest.get("disposition", "research_preview"),
                "forecast_kind": "rolling_origin_rescore",
                "training_origin_week": entry["origin_week"],
                "scoring": entry.get("scoring"),
                "training_through_season": entry["training_through_season"],
                "model_manifest_sha256": entry["artifacts"]["model_manifest"]["sha256"],
                "snapshot": snapshot_audit,
                "supplements": supplement_audit,
                "availability_sources": str(sources) if sources else None,
                "notes": [
                    f"Weights were trained from week {entry['origin_week']} cutoffs; this forecast "
                    f"applies them at week {origin}. Later cutoffs were never seen in training.",
                    "Features use observations through the origin week only.",
                    "Per-game points is a flat average over the remaining scheduled games.",
                ],
                "rows": rows,
            }
            path = candidate / relative
            json_write(path, receipt)
            revised_entry.setdefault("forecasts", []).append(
                {
                    "season": season,
                    "origin_week": origin,
                    "rolling_origin": True,
                    "path": relative,
                    "sha256": digest(path),
                }
            )
            added.append({"model_id": entry["model_id"], "rows": len(rows)})
            print(
                f"re-scored {campaign_id}/{entry['model_id']} at week {origin}: "
                f"{len(rows)} players",
                flush=True,
            )
        revised.setdefault("rolling_origin_rescores", []).append(
            {
                "season": season,
                "origin_week": origin,
                "generated_at": datetime.now(UTC).isoformat(),
                "models": [a["model_id"] for a in added],
            }
        )
        limitation = (
            "Rolling re-scores apply weights trained at an earlier cutoff; they are labelled "
            "per forecast and are not a separately validated weekly model."
        )
        if limitation not in revised.get("limitations", []):
            revised.setdefault("limitations", []).append(limitation)
        content = (json.dumps(revised, indent=2, allow_nan=False) + "\n").encode()
        (candidate / "manifest.json").write_bytes(content)
        (candidate / "manifest.sha256").write_text(digest(candidate / "manifest.json") + "\n")
        receipt = replace_campaign(
            registry, candidate_root, campaign_id, manifest["manifest_sha256"], archive
        )
    store.check(campaign_id)
    return {"campaign_id": campaign_id, "status": "rescored", "models": added, "receipt": receipt}


# ----------------------------------------------------------------- driver


def _track_v31(args, snapshot):
    """The separately sealed challenger follows refreshes, including no-op v3 refreshes."""
    from .weekly_v31.prospective import track
    from .weekly_v31.run import DEFAULT_LOG, DEFAULT_RUN

    if args.campaign and not {"opening_drive_v3", "opening_drive_v31"}.intersection(args.campaign):
        return None
    # Alternate registries are isolated unless the caller explicitly supplies a challenger.
    if args.v31_run is None and args.registry.resolve() != REGISTRY.resolve():
        return None
    run = args.v31_run or DEFAULT_RUN
    if args.v31_run and not run.exists():
        raise FileNotFoundError(f"V3.1 challenger run not found: {run}")
    if not run.exists():
        return None
    return track(snapshot, args.registry, run, args.v31_log or DEFAULT_LOG)


def main(argv: list[str] | None = None) -> int:
    from .weekly_v3.rescore import due_models, rescore_weekly, validate_snapshot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--snapshot", type=Path, help="Raw snapshot manifest; default: latest")
    parser.add_argument("--origin-week", type=int, help="Default: the snapshot's completed week")
    parser.add_argument("--campaign", action="append", help="Limit to these campaign ids")
    parser.add_argument(
        "--due", action="store_true", help="Only re-score models that lack the current week"
    )
    parser.add_argument("--no-capture", action="store_true", help="Reuse existing supplements")
    parser.add_argument(
        "--v31-run", type=Path, help="Sealed interval challenger; default: local v3.1"
    )
    parser.add_argument("--v31-log", type=Path, help="Prospective archive destination")
    args = parser.parse_args(argv)
    consolidate_opening_drive(args.registry, args.archive)
    snapshot = args.snapshot or latest_raw_snapshot()
    current = json.loads(snapshot.read_text())["current_observations"]
    season = current["season"]
    origin = current["through_week"] if args.origin_week is None else args.origin_week
    store = CampaignStore(args.registry)
    campaigns = store.campaigns()
    if origin == 18:
        tracked = _track_v31(args, snapshot)
        print(json.dumps({"status": "regular_season_complete", "v31": tracked}, indent=2))
        return 0
    manifests = [
        m
        for m in campaigns
        if m["campaign_id"] == LATEST
        and (
            not args.campaign
            or m["campaign_id"] in args.campaign
            or "opening_drive_v31" in args.campaign
        )
        and any(
            x["horizon"] == "remaining" and "weights" in x.get("artifacts", {}) for x in m["models"]
        )
    ]
    due = [
        m
        for m in manifests
        if any(
            x["horizon"] == "remaining"
            and "weights" in x.get("artifacts", {})
            and not any(
                f["season"] == season and f["origin_week"] == origin for f in x["forecasts"]
            )
            for x in m["models"]
        )
    ]
    weekly_due = [
        m
        for m in campaigns
        if m["campaign_id"] == "opening_drive_v3"
        and (
            not args.campaign
            or m["campaign_id"] in args.campaign
            or "opening_drive_v31" in args.campaign
        )
        and due_models(m, season, origin)
    ]
    if weekly_due:
        validate_snapshot(snapshot, season, origin)
    if args.due and not due and not weekly_due:
        tracked = _track_v31(args, snapshot)
        if tracked:
            print(json.dumps({"v31": tracked}, indent=2))
        print(f"Campaign forecasts already cover {season} week {origin}.")
        return 0
    if not due and not weekly_due:
        tracked = _track_v31(args, snapshot)
        if tracked:
            print(json.dumps({"v31": tracked}, indent=2))
        print("Nothing to re-score.")
        return 0
    supplements = None if args.no_capture else latest_supplements(season, current["saved_at"])
    if supplements is None:
        supplements = (
            capture_supplements(season) if not args.no_capture else latest_supplements(season, "")
        )
    if supplements is None:
        raise FileNotFoundError("No supplement capture is available for this season")
    sources = None
    if weekly_due or any(m.get("availability") for m in due):
        sources = refresh_availability_sources(season, supplements, snapshot)
    results = []
    with threadpool_limits(limits=4):
        for manifest in due:
            results.append(
                rescore_campaign(
                    store,
                    manifest,
                    season=season,
                    origin=origin,
                    snapshot=snapshot,
                    supplements=supplements,
                    sources=sources,
                    registry=args.registry,
                    archive=args.archive,
                )
            )
        for manifest in weekly_due:
            results.append(
                rescore_weekly(
                    store,
                    manifest,
                    season=season,
                    origin=origin,
                    snapshot=snapshot,
                    sources=sources,
                    registry=args.registry,
                    archive=args.archive,
                )
            )
    tracked = _track_v31(args, snapshot)
    if tracked:
        results.append(dict(campaign_id="opening_drive_v31", **tracked))
    print(
        json.dumps(
            {"season": season, "origin_week": origin, "campaigns": results}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.SubprocessError as error:  # pragma: no cover - external tooling
        raise SystemExit(str(error)) from error
