"""Score v2 from frozen v1 cutoff matrices plus dated availability; seal a local bundle."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from engine.chairstats import CampaignStore, publish_campaign

from ...gpu_lab.run import digest, json_write
from ..forecast import model_card
from ..run import load_model, predict
from ..runtime import source_artifacts, verify_environment
from .data import ROOT, SOURCE_DIR, V1_RUN, augment, checked_parquet, load_sources


def export(
    run, source_dir=SOURCE_DIR, registry=ROOT / "data/campaigns", season=2026, *, parent_run=V1_RUN
):
    run = Path(run)
    parent_run = Path(parent_run)
    config = json.loads((run / "campaign_config.json").read_text())
    sources, source_manifest = load_sources(source_dir)
    source_sha = digest(Path(source_dir) / "manifest.json")
    exports = []
    for position in config["positions"]:
        for horizon in config["horizons"]:
            model_id = f"{position.lower()}_{horizon}"
            directory = run / model_id
            saved = json.loads((directory / "manifest.json").read_text())
            verify_environment(saved)
            dataset = json.loads((run / "inputs" / model_id / "dataset.json").read_text())
            if dataset["contract"]["availability_manifest_sha256"] != source_sha:
                raise ValueError("Inference availability source differs from training")
            origin = saved["origin_week"]
            base_dir = parent_run / "forecasts" / model_id
            base_receipt_path = base_dir / f"{season}_w{origin}.json"
            parent_manifest = json.loads((parent_run / "manifest.json").read_text())
            relative = str(base_receipt_path.relative_to(parent_run))
            if digest(base_receipt_path) != parent_manifest["files"][relative]:
                raise ValueError("Frozen v1 forecast receipt changed")
            base_receipt = json.loads(base_receipt_path.read_text())
            base = checked_parquet(
                base_dir,
                f"{season}_w{origin}_features.parquet",
                base_receipt["feature_parquet_sha256"],
            )
            audit = augment(base, sources)
            panel = audit.filter(pl.col("availability_eligible"))
            output = predict(load_model(directory), panel)
            if not all(np.isfinite(x).all() for x in output.values()):
                raise ValueError("Nonfinite v2 forecast")
            ranks = np.empty(panel.height, dtype=int)
            ranks[np.argsort(-output["points"], kind="stable")] = np.arange(1, panel.height + 1)
            rows = []
            for index, row in enumerate(panel.iter_rows(named=True)):
                games = int(row["scheduled_games"]) if row["scheduled_games"] is not None else None
                rows.append(
                    dict(
                        player_id=row["player_id"],
                        player_name=row["player_display_name"],
                        team=row["team"],
                        position=position,
                        season=season,
                        origin_week=origin,
                        horizon=horizon,
                        rank=int(ranks[index]),
                        scheduled_games=games,
                        calendar_weeks=int(row["weeks"]),
                        candidate_basis=row["candidate_basis"],
                        availability=dict(
                            status=row["availability_status"],
                            team=row["availability_team"],
                            observed_at=row["availability_observed_at"],
                            as_of=row["availability_as_of"],
                            evidence=row["availability_evidence"],
                        ),
                        predictions={
                            target: float(output[target][index]) for target in saved["targets"]
                        },
                        component_points=float(output["component_points"][index]),
                        points_per_scheduled_game=float(output["points"][index] / games)
                        if games
                        else None,
                    )
                )
            rows.sort(key=lambda r: r["rank"])
            destination = run / "forecasts" / model_id
            destination.mkdir(parents=True, exist_ok=True)
            forecast_path = destination / f"{season}_w{origin}.json"
            features = destination / f"{season}_w{origin}_features.parquet"
            audit_path = destination / f"{season}_w{origin}_candidate_audit.parquet"
            card = model_card(run, directory, saved)
            if forecast_path.exists():
                previous = json.loads(forecast_path.read_text())
                if (
                    previous["rows"] != rows
                    or previous["availability_manifest_sha256"] != source_sha
                    or previous["model_manifest_sha256"] != digest(directory / "manifest.json")
                ):
                    raise ValueError("Immutable v2 forecast differs")
                frozen = checked_parquet(
                    destination, features.name, previous["feature_parquet_sha256"]
                )
                if not frozen.equals(panel):
                    raise ValueError("Immutable v2 feature matrix differs")
                checked_parquet(destination, audit_path.name, previous["candidate_audit_sha256"])
            else:
                panel.write_parquet(features)
                audit.select(
                    "year",
                    "origin",
                    "player_id",
                    "player_display_name",
                    *[c for c in audit.columns if c.startswith("availability")],
                ).write_parquet(audit_path)
                json_write(
                    forecast_path,
                    dict(
                        schema_version=1,
                        campaign_id=config["campaign_id"],
                        model_id=model_id,
                        season=season,
                        origin_week=origin,
                        horizon=horizon,
                        generated_at=datetime.now(UTC).isoformat(),
                        disposition="research_preview",
                        forecast_kind="reconstructed_cutoff_from_later_capture",
                        scoring=saved["scoring"],
                        training_through_season=saved["training_through_season"],
                        model_manifest_sha256=digest(directory / "manifest.json"),
                        feature_parquet_sha256=digest(features),
                        candidate_audit_sha256=digest(audit_path),
                        availability_manifest_sha256=source_sha,
                        parent_forecast_sha256=digest(base_receipt_path),
                        snapshot=base_receipt["snapshot"],
                        supplements=base_receipt["supplements"],
                        model_card=card,
                        notes=card["limitations"],
                        rows=rows,
                        exclusions=audit.filter(~pl.col("availability_eligible"))
                        .select(
                            "player_id",
                            "player_display_name",
                            "availability_status",
                            "availability_evidence",
                        )
                        .to_dicts(),
                    ),
                )
            exports.append(
                dict(
                    **{
                        k: saved[k]
                        for k in [
                            "model_id",
                            "position",
                            "horizon",
                            "origin_week",
                            "targets",
                            "scoring",
                            "training_through_season",
                            "feature_count",
                            "final_members",
                        ]
                    },
                    model_card=card,
                    metrics=directory / "metrics.json",
                    forecasts=[dict(season=season, origin_week=origin, path=forecast_path)],
                    artifacts={
                        **source_artifacts(saved),
                        "weights": directory / "models.joblib",
                        "model_manifest": directory / "manifest.json",
                        "feature_contract": directory / "feature_names.json",
                        "fit_contract": directory / "fit_contract.json",
                        "evaluation_predictions": directory / "evaluation_predictions.parquet",
                        "forecast_features": features,
                        "forecast_candidate_audit": audit_path,
                        "historical_candidate_audit": run
                        / "inputs"
                        / model_id
                        / "candidate_audit.parquet",
                        "availability_manifest": Path(source_dir) / "manifest.json",
                        "forecast_source": Path(__file__),
                    },
                )
            )
            print(f"v2 forecast {model_id}: {panel.height}/{base.height} eligible", flush=True)
    metadata = dict(
        **config,
        description=(
            "Opening Drive models with dated availability features "
            "and formal retirement eligibility"
        ),
        generated_at=datetime.now(UTC).isoformat(),
        status="complete",
        kind="model_campaign",
        config_sha256=digest(run / "campaign_config.json"),
        availability_manifest_sha256=source_sha,
        limitations=source_manifest["limitations"]
        + [
            "Historical development comparison; no pristine holdout.",
            config["availability"]["population"],
        ],
    )
    published = publish_campaign(registry, metadata, exports)
    check = CampaignStore(registry).check(config["campaign_id"])
    json_write(
        run / "manifest.json",
        dict(
            schema_version=1,
            campaign_id=config["campaign_id"],
            name=config["name"],
            kind="model_campaign",
            status="complete",
            disposition="research_preview",
            model_count=len(exports),
            completed_at=datetime.now(UTC).isoformat(),
            published_manifest=str(published),
            published_manifest_sha256=digest(published),
            files={
                str(p.relative_to(run)): digest(p)
                for p in [
                    run / "campaign_config.json",
                    *[run / m["model_id"] / "manifest.json" for m in exports],
                ]
            },
        ),
    )
    return check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "experiments/future_player_lab/runs/opening_drive_v2_001"
    )
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        print(json.dumps(export(args.run), indent=2))
