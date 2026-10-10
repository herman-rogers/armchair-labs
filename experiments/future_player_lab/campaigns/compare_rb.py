"""Read-only comparison of exploratory RB research and Opening Drive v1."""

# Long generated report rows and literal readout sentences preserve their layout.
# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from engine.chairstats import CampaignStore

from ..gpu_lab.run import digest, json_write
from .data import ROOT

RUNS = ROOT / "experiments/future_player_lab/runs"
TARGETS = [
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "points",
]
KEYS = ["horizon", "year", "origin", "player_id", "target"]
CAMPAIGNS = {
    "original": {
        "campaign_id": "rb_exploratory_research",
        "name": "RB Exploratory Research",
        "run_id": "rb_gpu_20261003_001",
        "policy": "ensemble_top3",
        "candidate_recipes": 52,
        "members_per_target": 3,
        "feature_count": 4810,
        "explicit_draft_features": False,
    },
    "opening_drive": {
        "campaign_id": "opening_drive_v1",
        "name": "Opening Drive v1",
        "run_id": "opening_drive_v1_001",
        "policy": "ensemble",
        "candidate_recipes": 4,
        "members_per_target": 2,
        "feature_count": 4813,
        "explicit_draft_features": True,
    },
}


def verify_files(directory, receipt="manifest.json", *, require_status=True):
    manifest_path = Path(directory) / receipt
    manifest = json.loads(manifest_path.read_text())
    if require_status and manifest.get("status") != "complete":
        raise ValueError(f"Incomplete artifact set: {manifest_path}")
    for name, expected in manifest["files"].items():
        if digest(Path(directory) / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {manifest_path}/{name}")
    return manifest


def paired_rows(original, newer):
    """Require identical keys/outcome availability; compare the same player outcomes."""
    for name, frame in (("original", original), ("opening_drive", newer)):
        if frame.select(KEYS).is_duplicated().any():
            raise ValueError(f"Duplicate {name} historical keys")
    missing_new = original.select(KEYS).join(newer.select(KEYS), on=KEYS, how="anti")
    missing_old = newer.select(KEYS).join(original.select(KEYS), on=KEYS, how="anti")
    if missing_new.height or missing_old.height:
        raise ValueError("Historical candidate populations differ")
    paired = original.select(
        *KEYS, "player_display_name", "actual", pl.col("prediction").alias("original")
    )
    paired = paired.join(
        newer.select(
            *KEYS, pl.col("actual").alias("new_actual"), pl.col("prediction").alias("opening_drive")
        ),
        on=KEYS,
        how="inner",
        validate="1:1",
    )
    a, b = paired["actual"].to_numpy(), paired["new_actual"].to_numpy()
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise ValueError("Historical outcome availability differs")
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-10, equal_nan=True)
    return paired


def compare_errors(block):
    actual, older, newer = [block[c].to_numpy() for c in ("actual", "original", "opening_drive")]
    valid = np.isfinite(actual) & np.isfinite(older) & np.isfinite(newer)
    actual, older, newer = actual[valid], older[valid], newer[valid]
    if not len(actual):
        raise ValueError("No comparable finite outcomes")
    old_mse, new_mse = float(np.mean((actual - older) ** 2)), float(np.mean((actual - newer) ** 2))
    return {
        "n": len(actual),
        "original_rmse": float(np.sqrt(old_mse)),
        "opening_drive_rmse": float(np.sqrt(new_mse)),
        "original_mse": old_mse,
        "opening_drive_mse": new_mse,
        "original_mae": float(np.mean(abs(actual - older))),
        "opening_drive_mae": float(np.mean(abs(actual - newer))),
        "mse_improvement_percent": 100 * (1 - new_mse / old_mse) if old_mse else None,
        "rmse_improvement_percent": 100 * (1 - np.sqrt(new_mse / old_mse)) if old_mse else None,
    }


def comparison(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Immutable comparison exists: {output}")
    old = RUNS / "rb_gpu_20261003_001"
    new = RUNS / "opening_drive_v1_001"
    preview = RUNS / "rb_ppr_2026_w2_preview_002"
    research = json.loads((old / "research_manifest.json").read_text())
    for name in (
        "manifest.json",
        "analysis/manifest.json",
        "analysis/predictions.parquet",
        "final_models/manifest.json",
        "config.json",
    ):
        if digest(old / name) != research["inputs"][name]:
            raise ValueError(f"Original research input changed: {name}")
    analysis = verify_files(old / "analysis")
    final = verify_files(old / "final_models")
    old_forecast = verify_files(preview)
    if old_forecast["model_manifest_sha256"] != digest(old / "final_models/manifest.json"):
        raise ValueError("Original forecast model binding differs")
    old_summary = json.loads((preview / "summary.json").read_text())
    if (old_summary["season"], old_summary["origin"], old_summary["primary_policy"]) != (
        2026,
        2,
        "ensemble_top3",
    ):
        raise ValueError("Original forecast cutoff/policy differs")
    store = CampaignStore(ROOT / "data/campaigns")
    bundle_check = store.check("opening_drive_v1")
    original = pl.read_parquet(old / "analysis/predictions.parquet").filter(
        (pl.col("recipe") == "ensemble_top3")
        & pl.col("year").is_between(2021, 2025)
        & pl.col("target").is_in(TARGETS)
    )
    new_parts = []
    for horizon in ("season", "remaining"):
        cell = new / f"rb_{horizon}"
        verify_files(cell, require_status=False)
        new_parts.append(
            pl.read_parquet(cell / "evaluation_predictions.parquet")
            .filter(pl.col("recipe") == "ensemble")
            .with_columns(pl.lit(horizon).alias("horizon"))
        )
    paired = paired_rows(original, pl.concat(new_parts))
    historical = []
    for (horizon, target), block in paired.partition_by(
        ["horizon", "target"], as_dict=True
    ).items():
        for year, group in [
            (None, block),
            *[
                (int(key[0]), part)
                for key, part in block.partition_by("year", as_dict=True).items()
            ],
        ]:
            historical.append(
                {"horizon": horizon, "target": target, "year": year, **compare_errors(group)}
            )
    old_current = pl.read_parquet(preview / "rankings.parquet")
    new_current = store.forecast("opening_drive_v1", "rb_remaining", 2026, 2)
    old_by_id = {row["player_id"]: row for row in old_current.to_dicts()}
    new_by_id = {row["player_id"]: row for row in new_current["rows"]}
    if len(old_by_id) != old_current.height or set(old_by_id) != set(new_by_id):
        raise ValueError("Current player populations differ or contain duplicate IDs")
    current = []
    for player_id, row in old_by_id.items():
        newer = new_by_id[player_id]
        if row["scheduled_games"] != newer["scheduled_games"]:
            raise ValueError("Scheduled-game exposure differs")
        current.append(
            {
                "player_id": player_id,
                "player_name": row["player_display_name"],
                "original_rank": row["rank"],
                "opening_drive_rank": newer["rank"],
                "rank_improvement": row["rank"] - newer["rank"],
                "original_points": row["ppr_ensemble_top3"],
                "opening_drive_points": newer["predictions"]["points"],
                "point_change": newer["predictions"]["points"] - row["ppr_ensemble_top3"],
                "original_points_per_game": row["ppr_per_scheduled_game"],
                "opening_drive_points_per_game": newer["points_per_scheduled_game"],
                "scheduled_games": row["scheduled_games"],
            }
        )
    current.sort(key=lambda row: row["opening_drive_rank"])
    entry = store.model("opening_drive_v1", "rb_remaining")
    feature_ref = entry["artifacts"]["forecast_features"]
    old_features = pl.read_parquet(preview / "features.parquet").sort("player_id")
    new_features = pl.read_parquet(
        ROOT / "data/campaigns/opening_drive_v1" / feature_ref["path"]
    ).sort("player_id")
    names = json.loads((preview / "feature_names.json").read_text())
    if not old_features["player_id"].equals(new_features["player_id"]):
        raise ValueError("Current input row order/population differs")
    a, b = [
        frame.select(names).to_numpy().astype(np.float32) for frame in (old_features, new_features)
    ]
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
    different = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    points = [row for row in historical if row["target"] == "points" and row["year"] is None]
    notes = [
        "Both policies use chronological selection from previous three validation years; compare 2021–2025 development seasons, not a new untouched holdout.",
        "Original explored 52 recipes and averages its selected top 3; Opening Drive uses 4 raw-preserving recipes and top 2, with explicit draft inputs.",
        "This compares complete model builds, not a controlled draft-only ablation; gains cannot all be attributed to draft features.",
        "Current projections share the Week 2 cutoff and 170 exact player IDs, but captures differ; all 4,810 shared feature values agree within 2.4e-7 at model precision.",
        "Current projections are reconstructions from later captures, not genuinely issued before the games; differences are not accuracy evidence.",
        "PPR excludes passing, fumbles, returns, conversions and bonuses; per-game numbers are flat averages, not weekly matchup predictions.",
    ]
    report = {
        "schema_version": 1,
        "kind": "cross_campaign_comparison",
        "campaigns": CAMPAIGNS,
        "evaluation_years": list(range(2021, 2026)),
        "primary_target": "points",
        "notes": notes,
        "historical": {
            "headline": points,
            "rows": historical,
            "paired_target_rows": paired.height,
            "max_outcome_difference": float(
                np.nanmax(abs(paired["actual"].to_numpy() - paired["new_actual"].to_numpy()))
            ),
        },
        "current": {
            "season": 2026,
            "origin_week": 2,
            "horizon": "remaining",
            "rows": current,
            "matched_players": len(current),
            "rank_spearman": float(
                spearmanr(
                    [r["original_rank"] for r in current],
                    [r["opening_drive_rank"] for r in current],
                ).statistic
            ),
            "top12_overlap": len(
                {r["player_id"] for r in current if r["original_rank"] <= 12}
                & {r["player_id"] for r in current if r["opening_drive_rank"] <= 12}
            ),
            "top24_overlap": len(
                {r["player_id"] for r in current if r["original_rank"] <= 24}
                & {r["player_id"] for r in current if r["opening_drive_rank"] <= 24}
            ),
            "walker": next(r for r in current if r["player_id"] == "00-0038134"),
            "largest_point_changes": sorted(
                current, key=lambda r: abs(r["point_change"]), reverse=True
            )[:10],
            "largest_rank_changes": sorted(
                current, key=lambda r: abs(r["rank_improvement"]), reverse=True
            )[:10],
            "largest_relevant_rank_changes": sorted(
                [r for r in current if min(r["original_rank"], r["opening_drive_rank"]) <= 24],
                key=lambda r: abs(r["rank_improvement"]),
                reverse=True,
            )[:10],
            "feature_comparison": {
                "shared_feature_count": len(names),
                "exactly_different_float32_cells": int(different.sum()),
                "max_absolute_difference": float(np.nanmax(abs(a - b))),
                "original_snapshot": old_summary["raw_snapshot"],
                "opening_drive_snapshot": new_current["metadata"]["snapshot"],
            },
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    paired.write_parquet(output / "historical_pairs.parquet")
    pl.DataFrame(current).write_csv(output / "current_pairs.csv")
    json_write(output / "comparison.json", report)
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    (output / "readout.md").write_text(render(report))
    sources = [
        old / "research_manifest.json",
        old / "analysis/manifest.json",
        old / "analysis/predictions.parquet",
        old / "final_models/manifest.json",
        preview / "manifest.json",
        preview / "rankings.parquet",
        preview / "features.parquet",
        ROOT / "data/campaigns/opening_drive_v1/manifest.json",
    ]
    sources.extend(new / f"rb_{h}/manifest.json" for h in ("season", "remaining"))
    json_write(
        output / "manifest.json",
        {
            "schema_version": 1,
            "kind": "cross_campaign_comparison",
            "status": "complete",
            "campaign_ids": [c["campaign_id"] for c in CAMPAIGNS.values()],
            "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "verification": {
                "original_analysis_files": len(analysis["files"]),
                "original_fitted_files": len(final["files"]),
                "original_forecast_files": len(old_forecast["files"]),
                "opening_drive_bundle": bundle_check,
            },
            "files": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        },
    )
    return report


def render(report):
    lines = [
        "# RB campaign comparison",
        "",
        "**RB Exploratory Research** (`rb_exploratory_research`) versus **Opening Drive v1** (`opening_drive_v1`).",
        "",
        "Same players, targets and 2021–2025 outcomes; lower error is better. Improvement percentages use MSE, not RMSE.",
        "",
        "| Horizon | Target | Players/years | Original RMSE | Opening Drive RMSE | MSE improvement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["historical"]["rows"]:
        if row["year"] is None:
            lines.append(
                f"| {row['horizon']} | {row['target']} | {row['n']} | {row['original_rmse']:.3f} | {row['opening_drive_rmse']:.3f} | {row['mse_improvement_percent']:+.2f}% |"
            )
    w = report["current"]["walker"]
    lines += [
        "",
        "## Current 2026 Week 2 comparison",
        "",
        f"{report['current']['matched_players']} matched RBs; rank correlation {report['current']['rank_spearman']:.3f}. Top 12 overlap {report['current']['top12_overlap']}/12; top 24 overlap {report['current']['top24_overlap']}/24.",
        "",
        f"Kenneth Walker: original rank {w['original_rank']}, {w['original_points']:.2f} remaining PPR ({w['original_points_per_game']:.2f}/game); Opening Drive rank {w['opening_drive_rank']}, {w['opening_drive_points']:.2f} PPR ({w['opening_drive_points_per_game']:.2f}/game).",
        "",
        "| Player | Original rank | Opening Drive rank | Original PPR | Opening Drive PPR | Change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["current"]["largest_point_changes"]:
        lines.append(
            f"| {row['player_name']} | {row['original_rank']} | {row['opening_drive_rank']} | {row['original_points']:.2f} | {row['opening_drive_points']:.2f} | {row['point_change']:+.2f} |"
        )
    lines += ["", "## Interpretation", "", *[f"- {note}" for note in report["notes"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUNS / "rb_campaign_comparison_20261010_001")
    args = parser.parse_args()
    report = comparison(args.output)
    print(
        json.dumps(
            {"headline": report["historical"]["headline"], "walker": report["current"]["walker"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
