"""Stage and verify a same-ID Opening Drive v2 update before explicit activation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from engine.chairstats import CampaignStore, publish_campaign

from .data import ROOT
from .replay import replay_campaign
from .run import predict

CAMPAIGN = "opening_drive_v2"
PARENT_SHA = "43ecaf34fe32b4defdeeb7ada0cf29bcabfd0ce44ab4c62e8b05556dcdec6f2c"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_candidate(run: Path, registry: Path, *, data_root=ROOT / "data"):
    from .revision_data import current_panel

    run, registry, data_root = Path(run), Path(registry), Path(data_root)
    store = CampaignStore(data_root / "campaigns")
    parent = store.campaign(CAMPAIGN)
    if parent["manifest_sha256"] != PARENT_SHA:
        raise ValueError("Active v2 changed since experiment started; refusing stale revision")
    store.check(CAMPAIGN)
    old_root = data_root / "campaigns" / CAMPAIGN
    destination = run / "export"
    destination.mkdir(parents=True, exist_ok=False)
    exports, coverage = [], []
    for entry in parent["models"]:
        model_id, position, horizon = entry["model_id"], entry["position"], entry["horizon"]
        directory = destination / model_id
        directory.mkdir()
        artifacts = {key: old_root / ref["path"] for key, ref in entry["artifacts"].items()}
        artifacts["parent_metrics"] = old_root / entry["metrics"]["path"]
        artifacts["parent_evaluation_predictions"] = artifacts["evaluation_predictions"]
        artifacts["parent_forecast_candidate_audit"] = artifacts.pop("forecast_candidate_audit")
        revised = copy.deepcopy(entry)
        revised["model_card"]["limitations"] = [
            item
            for item in revised["model_card"]["limitations"]
            if not item.startswith("V1 candidate universe and team-context features are frozen")
        ] + [
            "Current candidate repair admits exact-ID drafted rookies; missing current weekly "
            "records are unknown rather than verified zero appearances."
        ]
        original = store.forecast(CAMPAIGN, model_id, 2026, entry["origin_week"])
        panel, names, candidate_audit = current_panel(position, horizon)
        if position == "QB":
            model_dir = run / "qb_challenge" / model_id
            saved = json.loads((model_dir / "manifest.json").read_text())
            for name, checksum in saved["files"].items():
                if digest(model_dir / name) != checksum:
                    raise ValueError(f"Revised QB artifact changed: {name}")
            model = joblib.load(model_dir / "models.joblib")
            panel = pl.read_parquet(model_dir / "current_features.parquet")
            names = model["feature_names"]
            for key, filename in {
                "weights": "models.joblib",
                "model_manifest": "manifest.json",
                "feature_contract": "feature_names.json",
                "fit_contract": "fit_contract.json",
                "evaluation_predictions": "evaluation_predictions.parquet",
            }.items():
                artifacts[key] = model_dir / filename
            for filename in saved["files"]:
                if filename not in {
                    "models.joblib",
                    "feature_names.json",
                    "fit_contract.json",
                    "evaluation_predictions.parquet",
                    "current_features.parquet",
                    "metrics.json",
                }:
                    artifacts["qb_evidence_" + Path(filename).stem] = model_dir / filename
            for index, (relative, checksum) in enumerate(saved.get("source_files", {}).items()):
                path = ROOT / relative
                if digest(path) != checksum:
                    raise ValueError(f"Revised QB source changed: {relative}")
                artifacts[f"revision_source_{index:02d}"] = path
            revised["metrics"] = model_dir / "metrics.json"
            revised["feature_count"] = len(names)
            revised["final_members"] = saved.get("final_members", entry["final_members"])
            revised["model_card"]["qb_passing_revision"] = saved["qb_revision"]
            revised["model_card"]["qb_passing_revision"]["baseline_evidence"] = (
                "This bounded comparison uses the original OD v2 ensemble as control. "
                "Original affine/persistence comparisons remain in parent_metrics and "
                "parent_evaluation_predictions artifacts."
            )
            artifacts["qb_preservation_review"] = run / "qb_challenge" / f"review_{horizon}.json"
        else:
            model = joblib.load(artifacts["weights"])
            revised["metrics"] = old_root / entry["metrics"]["path"]
        if names != model["feature_names"]:
            raise ValueError(f"Feature names differ for {model_id}")
        values = predict(model, panel)
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError("Nonfinite revised forecast")
        ranks = np.empty(panel.height, dtype=int)
        ranks[np.argsort(-values["points"], kind="stable")] = np.arange(1, panel.height + 1)
        rows = []
        for index, row in enumerate(panel.iter_rows(named=True)):
            games = int(row["scheduled_games"]) if row["scheduled_games"] is not None else None
            rows.append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_display_name"],
                    "team": row["team"],
                    "position": position,
                    "season": 2026,
                    "origin_week": entry["origin_week"],
                    "horizon": horizon,
                    "rank": int(ranks[index]),
                    "scheduled_games": games,
                    "calendar_weeks": int(row["weeks"]),
                    "candidate_basis": row["candidate_basis"],
                    "availability": {
                        key: row.get("availability_" + field)
                        for key, field in [
                            ("status", "status"),
                            ("team", "team"),
                            ("observed_at", "observed_at"),
                            ("as_of", "as_of"),
                            ("evidence", "evidence"),
                        ]
                    },
                    "predictions": {
                        target: float(values[target][index]) for target in entry["targets"]
                    },
                    "component_points": float(values["component_points"][index]),
                    "points_per_scheduled_game": float(values["points"][index] / games)
                    if games
                    else None,
                }
            )
        rows.sort(key=lambda row: row["rank"])
        by_id = {row["player_id"]: row for row in rows}
        preserved = [
            t
            for t in entry["targets"]
            if position != "QB" or t not in {"attempts", "passing_yards"}
        ]
        for row in original["rows"]:
            if row["player_id"] not in by_id:
                raise ValueError("Candidate fix unexpectedly removed an existing eligible player")
            for target in preserved:
                np.testing.assert_allclose(
                    by_id[row["player_id"]]["predictions"][target],
                    row["predictions"][target],
                    rtol=1e-7,
                    atol=1e-7,
                )
        features = directory / "features.parquet"
        panel.write_parquet(features)
        audit_path = directory / "candidate_revision.json"
        write_json(audit_path, candidate_audit)
        artifacts.update(
            forecast_features=features,
            candidate_revision=audit_path,
            candidate_revision_source=Path(__file__).with_name("revision_data.py"),
            candidate_admission_source=Path(__file__).with_name("data.py"),
            revision_export_source=Path(__file__),
            added_candidate_audit=run
            / "current_candidates"
            / model_id
            / "added_candidate_audit.parquet",
            historical_candidate_revision=run / "inputs_candidate" / model_id / "dataset.json",
            candidate_revision_summary=run / "candidate_revision_summary.json",
        )
        row_receipt = dict(original["metadata"])
        row_receipt.update(
            generated_at=datetime.now(UTC).isoformat(),
            rows=rows,
            model_manifest_sha256=digest(artifacts["model_manifest"]),
            feature_parquet_sha256=digest(features),
            candidate_revision_sha256=digest(audit_path),
            replaces_forecast_sha256=original["artifact_sha256"],
            model_card=revised["model_card"],
        )
        # The old audit describes the old candidate population; retain it as lineage only.
        if "candidate_audit_sha256" in row_receipt:
            row_receipt["parent_candidate_audit_sha256"] = row_receipt.pop("candidate_audit_sha256")
        forecast = directory / "forecast.json"
        write_json(forecast, row_receipt)
        revised.update(
            artifacts=artifacts,
            forecasts=[
                {
                    "season": 2026,
                    "origin_week": entry["origin_week"],
                    "path": forecast,
                }
            ],
        )
        revised["candidate_revision"] = candidate_audit
        exports.append(revised)
        added = [
            r for r in rows if r["player_id"] not in {o["player_id"] for o in original["rows"]}
        ]
        coverage.append(
            {
                "model_id": model_id,
                "before": len(original["rows"]),
                "after": len(rows),
                "added": [{k: r[k] for k in ["player_id", "player_name", "team"]} for r in added],
            }
        )
    metadata = {k: v for k, v in parent.items() if k not in {"models", "manifest_sha256"}}
    metadata.update(
        generated_at=datetime.now(UTC).isoformat(),
        description=(
            "Dated availability, corrected drafted-rookie admission "
            "and evaluated QB passing policies"
        ),
        revision={
            "parent_manifest_sha256": PARENT_SHA,
            "campaign_id_unchanged": True,
            "scope": "2026 candidate admission plus QB attempts/passing-yards experiment",
            "run": str(run.relative_to(ROOT)),
            "source_sha256": digest(__file__),
        },
    )
    metadata["availability"]["population"] = (
        "Original historical population; current candidates also admit exactly identified drafted "
        "combine entrants using draft-year facts in the identity source"
    )
    metadata["selection"] = (
        "Original target policies; QB attempts/yardage revision selected on prior-year evidence"
    )
    metadata["limitations"] = [
        item
        for item in metadata["limitations"]
        if item
        != "V1 base population and team context retained for a controlled additive experiment."
    ] + [
        "Current rookie admission uses retrospective immutable draft facts; absent weekly records "
        "remain unavailable rather than proof of zero appearances.",
        "QB revision evaluates attempts and passing yards only; seven other targets retain "
        "their original weights and evidence.",
    ]
    metadata["lineage"] += "; exact-ID rookie repair and bounded QB passing experiment"
    metadata["availability"]["features"] = (
        "Frozen base numeric matrix plus dated availability; QB adds explicit past-only "
        "efficiency and opportunity fields"
    )
    if CampaignStore(data_root / "campaigns").campaign(CAMPAIGN)["manifest_sha256"] != PARENT_SHA:
        raise ValueError("Active v2 changed during export; refusing stale revision")
    store.check(CAMPAIGN)
    publish_campaign(registry, metadata, exports)
    replay = replay_campaign(registry, CAMPAIGN)
    result = {
        "coverage": coverage,
        "replay": replay,
        "candidate_manifest_sha256": CampaignStore(registry).campaign(CAMPAIGN)["manifest_sha256"],
    }
    write_json(destination / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        print(json.dumps(build_candidate(args.run.resolve(), args.registry.resolve()), indent=2))


if __name__ == "__main__":
    main()
