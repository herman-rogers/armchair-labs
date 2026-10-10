"""Score saved campaign models locally, then publish immutable API-readable JSON."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from engine.chairstats import CampaignStore, publish_campaign

from ..forecasting.data import add_supplements, history_inputs, snapshot_inputs
from ..gpu_lab.run import digest, json_write
from .data import ROOT, feature_rows
from .run import load_model, predict
from .runtime import source_artifacts, verify_environment


def model_card(run, directory, saved):
    """Derive user-relevant limitations from the sealed historical evaluation."""
    report = json.loads((directory / "metrics.json").read_text())
    points = pl.read_parquet(directory / "evaluation_predictions.parquet").filter(
        pl.col("target") == "points"
    )
    controls = {
        row["recipe"]: row
        for row in report["rows"]
        if row["target"] == "points"
        and row["cohort"] == "all_candidates"
        and row["year"] is None
        and row["recipe"] in {"ensemble", "affine"}
    }
    dataset = run / "inputs" / saved["model_id"]
    meta_path = dataset / "dataset.json"
    meta = json.loads(meta_path.read_text())
    if (
        digest(meta_path) != saved["dataset_manifest_sha256"]
        or digest(dataset / "panel.parquet") != meta["panel_sha256"]
    ):
        raise ValueError("Model-card input provenance mismatch")
    usage = "history_attempts" if saved["position"] == "QB" else "history_targets"
    panel = pl.read_parquet(
        dataset / "panel.parquet",
        columns=["year", "player_id", usage, "history_carries", "history_weeks"],
    ).with_columns(
        ((pl.col(usage) + pl.col("history_carries")) / pl.col("history_weeks")).alias("known_usage")
    )
    pairs = (
        points.filter(pl.col("recipe") == "ensemble")
        .select("year", "player_id", "actual", pl.col("prediction").alias("ensemble"))
        .join(
            points.filter(pl.col("recipe") == "affine").select(
                "year", "player_id", pl.col("prediction").alias("affine")
            ),
            on=["year", "player_id"],
            how="inner",
            validate="1:1",
        )
    )
    pieces, ranking = [], []
    k = 12 if saved["position"] in {"QB", "TE"} else 24
    for year in report["development_evaluation_years"]:
        prior = panel.filter((pl.col("year") < year) & (pl.col("known_usage") > 0))
        threshold = float(np.quantile(prior["known_usage"].drop_nulls().to_numpy(), 0.75))
        pieces.append(
            pairs.filter(pl.col("year") == year).join(
                panel.filter(
                    (pl.col("year") == year) & (pl.col("known_usage") >= threshold)
                ).select("year", "player_id"),
                on=["year", "player_id"],
                how="inner",
                validate="1:1",
            )
        )
        current = points.filter(pl.col("year") == year)
        actuals = {}
        for (recipe,), rows in current.partition_by("recipe", as_dict=True).items():
            actuals[recipe] = float(
                rows.sort(["prediction", "player_id"], descending=[True, False])
                .head(k)["actual"]
                .sum()
            )
        ranking.append(
            {
                "year": year,
                "ensemble_minus_affine_actual_points": actuals["ensemble"] - actuals["affine"],
                "all_tree_recipes_below_affine": all(
                    actuals[r] < actuals["affine"] for r in saved["recipes"]
                ),
            }
        )
    high = pl.concat(pieces)
    finite = high.filter(pl.all_horizontal(pl.col("actual", "ensemble", "affine").is_finite()))
    losses = {
        r: float(np.mean((finite[r].to_numpy() - finite["actual"].to_numpy()) ** 2))
        for r in ("ensemble", "affine")
    }
    delta = 100 * (losses["ensemble"] / losses["affine"] - 1) if losses["affine"] else None
    limitations = [
        "Historical development evaluation; these years were not a pristine holdout.",
        "Aggregate production and explicit fantasy scoring, not general player ability.",
        "Per-game averages are not individual weekly or matchup predictions.",
    ]
    if delta is not None and delta > 0:
        limitations.append(
            f"High-prior-usage points MSE was {delta:.2f}% higher than affine persistence."
        )
    below = sum(row["ensemble_minus_affine_actual_points"] < 0 for row in ranking)
    if below:
        limitations.append(
            f"Selected top{k} realized points trailed affine persistence "
            f"in {below}/{len(ranking)} evaluation years."
        )
    if saved.get("availability"):
        limitations.extend(
            [
                "Dated availability is reconstructed from retrospective transaction and "
                "weekly roster archives; coverage is incomplete.",
                "Formal retired status excludes; retirement announcements, injuries, "
                "reserve status, and missing evidence remain forecast risks.",
                saved["availability"]["population"],
            ]
        )
    elif saved["position"] == "QB" and saved["horizon"] == "season":
        limitations.append(
            "No dated offseason retirement/availability event feed; "
            "retired prior-season participants can remain candidates."
        )
    if all(row["all_tree_recipes_below_affine"] for row in ranking):
        limitations.append(
            "Every fixed tree candidate also trailed affine for selected top-K "
            "realized points in every evaluation year."
        )
    return {
        "disposition": "research_preview",
        "scoring": saved["scoring"],
        "evaluation_years": report["development_evaluation_years"],
        "all_candidate_points": controls,
        "high_prior_usage": {
            "definition": (
                "At least the upper quartile of positive prior usage rates, "
                "learned from earlier training seasons"
            ),
            "n": finite.height,
            "ensemble_mse": losses["ensemble"],
            "affine_mse": losses["affine"],
            "mse_percent_change_vs_affine": delta,
        },
        "ranking_top_k": k,
        "ranking_by_year": ranking,
        "limitations": limitations,
        "readout": (
            "docs/research/opening_drive_v2_te_2026-10-10.md"
            if saved["position"] == "TE"
            else "docs/research/opening_drive_v2_2026-10-10.md"
            if saved.get("availability")
            else "docs/research/opening_drive_2026-10-09.md"
        ),
    }


def export_forecasts(run, snapshot, supplements, season=2026):
    run = Path(run)
    config = json.loads((run / "campaign_config.json").read_text())
    history = history_inputs(run / "inputs/history")
    tables, snapshot_audit = snapshot_inputs(history, Path(snapshot), season, 2)
    tables, supplement_audit = add_supplements(tables, Path(supplements), season, 2)
    forecast_sources = {
        str(path.relative_to(ROOT)): digest(path)
        for path in (Path(__file__), Path(__file__).with_name("runtime.py"))
    }
    exports = []
    for position in config["positions"]:
        for horizon in config["horizons"]:
            model_id = f"{position.lower()}_{horizon}"
            directory = run / model_id
            saved = json.loads((directory / "manifest.json").read_text())
            verify_environment(saved)
            model = load_model(directory)
            card = model_card(run, directory, saved)
            origin = saved["origin_week"]
            panel = feature_rows(tables, position, season, origin)
            outputs = predict(model, panel)
            if not all(np.isfinite(values).all() for values in outputs.values()):
                raise ValueError(f"Nonfinite forecast in {model_id}")
            ranks = np.empty(panel.height, dtype=int)
            ranks[np.argsort(-outputs["points"], kind="stable")] = np.arange(1, panel.height + 1)
            rows = []
            for index, row in enumerate(panel.iter_rows(named=True)):
                scheduled_games = (
                    int(row["scheduled_games"]) if row["scheduled_games"] is not None else None
                )
                rows.append(
                    {
                        "player_id": row["player_id"],
                        "player_name": row["player_display_name"],
                        "team": row["team"],
                        "position": position,
                        "season": season,
                        "origin_week": origin,
                        "horizon": horizon,
                        "rank": int(ranks[index]),
                        "scheduled_games": scheduled_games,
                        "calendar_weeks": int(row["weeks"]),
                        "candidate_basis": row["candidate_basis"],
                        "predictions": {
                            target: float(outputs[target][index]) for target in saved["targets"]
                        },
                        "component_points": float(outputs["component_points"][index]),
                        "points_per_scheduled_game": float(
                            outputs["points"][index] / scheduled_games
                        )
                        if scheduled_games
                        else None,
                    }
                )
            rows.sort(key=lambda r: r["rank"])
            destination = run / "forecasts" / model_id
            forecast_path = destination / f"{season}_w{origin}.json"
            feature_path = destination / f"{season}_w{origin}_features.parquet"
            if forecast_path.exists():
                existing = json.loads(forecast_path.read_text())
                if (
                    existing["rows"] != rows
                    or existing["model_manifest_sha256"] != digest(directory / "manifest.json")
                    or existing["snapshot"] != snapshot_audit
                    or existing["supplements"] != supplement_audit
                    or existing["forecast_source_files"] != forecast_sources
                    or existing["feature_parquet_sha256"] != digest(feature_path)
                    or not pl.read_parquet(feature_path).equals(panel)
                ):
                    raise ValueError(f"Immutable forecast replay differs: {forecast_path}")
            else:
                destination.mkdir(parents=True, exist_ok=True)
                panel.write_parquet(feature_path)
            receipt = {
                "schema_version": 1,
                "campaign_id": config["campaign_id"],
                "model_id": model_id,
                "season": season,
                "origin_week": origin,
                "horizon": horizon,
                "generated_at": datetime.now(UTC).isoformat(),
                "disposition": "research_preview",
                "forecast_kind": "reconstructed_cutoff_from_later_capture",
                "forecast_source_files": forecast_sources,
                "scoring": saved["scoring"],
                "training_through_season": 2025,
                "model_manifest_sha256": digest(directory / "manifest.json"),
                "model_card": card,
                "snapshot": snapshot_audit,
                "supplements": supplement_audit,
                "feature_parquet_sha256": digest(
                    destination / f"{season}_w{origin}_features.parquet"
                ),
                "notes": [
                    "Features use observations through origin; capture was obtained later.",
                    "Preseason forecasts ignore all current-season observed outcomes.",
                    "Per-game points is a flat average, not a weekly or matchup forecast.",
                    "Independent direct fantasy and component-derived points can differ.",
                    "Team is last observed and can lag roster moves.",
                    *card["limitations"],
                ],
                "rows": rows,
            }
            if not forecast_path.exists():
                json_write(forecast_path, receipt)
            exports.append(
                {
                    "model_id": model_id,
                    "position": position,
                    "horizon": horizon,
                    "origin_week": origin,
                    "targets": saved["targets"],
                    "scoring": saved["scoring"],
                    "model_card": card,
                    "training_through_season": 2025,
                    "metrics": directory / "metrics.json",
                    "feature_count": saved["feature_count"],
                    "final_members": saved["final_members"],
                    "forecasts": [{"season": season, "origin_week": origin, "path": forecast_path}],
                    "artifacts": {
                        **source_artifacts(saved),
                        "weights": directory / "models.joblib",
                        "model_manifest": directory / "manifest.json",
                        "feature_contract": directory / "feature_names.json",
                        "fit_contract": directory / "fit_contract.json",
                        "evaluation_predictions": directory / "evaluation_predictions.parquet",
                        "forecast_features": destination / f"{season}_w{origin}_features.parquet",
                    },
                }
            )
            print(f"scored {model_id}: {len(rows)} players", flush=True)
    return config, exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="opening_drive_v1_001")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "data/raw/snapshots/weekly_20261008_w4_r3/manifest.json",
    )
    parser.add_argument(
        "--supplements",
        type=Path,
        default=ROOT / "experiments/future_player_lab/runs/forecast_sources_20261009_003",
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--registry", type=Path, default=ROOT / "data/campaigns")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]+", args.run_id):
        raise ValueError("Unsafe run ID")
    run = ROOT / "experiments/future_player_lab/runs" / args.run_id
    with threadpool_limits(limits=4):
        config, exports = export_forecasts(run, args.snapshot, args.supplements, args.season)
    lineage = "experiments/future_player_lab/runs/rb_gpu_20261003_001/final_models/manifest.json"
    metadata = {
        **config,
        "description": "Position-specific boosted-tree production and fantasy forecasts",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "complete",
        "kind": "model_campaign",
        "config_sha256": digest(run / "campaign_config.json"),
        "lineage_artifacts": [
            {
                "path": lineage,
                "sha256": digest(ROOT / lineage),
            }
        ],
        "limitations": [
            "Research preview; no production promotion or general-ability claim.",
            "Chronological historical development evaluation; no pristine holdout.",
            "Current forecasts reconstructed from later raw captures.",
        ],
    }
    manifest = publish_campaign(args.registry, metadata, exports)
    check = CampaignStore(args.registry).check(config["campaign_id"])
    parent_manifest = run / "manifest.json"
    if parent_manifest.exists():
        raise FileExistsError(f"Run already sealed: {parent_manifest}")
    files = {"campaign_config.json": digest(run / "campaign_config.json")}
    for model in exports:
        cell = run / model["model_id"] / "manifest.json"
        files[str(cell.relative_to(run))] = digest(cell)
        for forecast in model["forecasts"]:
            path = forecast["path"]
            files[str(path.relative_to(run))] = digest(path)
    json_write(
        parent_manifest,
        {
            "schema_version": 1,
            "kind": "model_campaign",
            "status": "complete",
            "campaign_id": config["campaign_id"],
            "name": config["name"],
            "disposition": "research_preview",
            "model_count": len(exports),
            "completed_at": datetime.now(UTC).isoformat(),
            "published_manifest": str(manifest),
            "published_manifest_sha256": digest(manifest),
            "files": files,
        },
    )
    print(json.dumps(check, indent=2))
    print(manifest)


if __name__ == "__main__":
    main()
