"""Attach a dated expert comparison to an immutable saved forecast, without rescoring."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

from ..gpu_lab.run import digest, json_write
from ..notebooks.raw_distribution_data import TEAM_CODES
from .report import write_report


def normalized_name(name):
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name.strip().lower())
    return re.sub(r"[^a-z0-9]", "", name)


def compare_rankings(rankings, experts, season, position):
    """Unique name/position joins; retain team changes and missing names explicitly."""
    if (
        str(experts["year"]),
        experts["position_id"],
        experts["scoring"],
        experts["ranking_type_name"],
    ) != (str(season), position, "PPR", "ros"):
        raise ValueError("Expert rankings must match the season, position, PPR and ROS horizon")
    players = experts["players"]
    if any(p["player_position_id"] != position for p in players):
        raise ValueError("Mixed expert positions")
    if len({p["player_id"] for p in players}) != len(players):
        raise ValueError("Duplicate expert IDs")
    for p in players:
        rank = p["rank_ecr"]
        if not isinstance(rank, (int, float)) or not np.isfinite(rank) or rank <= 0:
            raise ValueError("Expert ranks must be finite positive numbers")
    if rankings["player_id"].n_unique() != rankings.height:
        raise ValueError("Duplicate model IDs")
    expert_counts = Counter(normalized_name(p["player_name"]) for p in players)
    model_counts = Counter(normalized_name(p["player_display_name"]) for p in rankings.to_dicts())
    by_name = {normalized_name(p["player_name"]): p for p in players}
    rows, matched_ids = [], set()
    for row in rankings.iter_rows(named=True):
        key = normalized_name(row["player_display_name"])
        candidate = by_name.get(key)
        ambiguous = expert_counts[key] > 1 or model_counts[key] > 1
        expert = candidate if not ambiguous else None
        if expert:
            matched_ids.add(expert["player_id"])
        model_team = TEAM_CODES.get(row["team"], row["team"])
        expert_team = expert["player_team_id"] if expert else None
        different_team = bool(expert and model_team != TEAM_CODES.get(expert_team, expert_team))
        rows.append(
            {
                **row,
                "expert_rank": float(expert["rank_ecr"]) if expert else None,
                "expert_player_id": str(expert["player_id"]) if expert else None,
                "expert_name": expert["player_name"] if expert else None,
                "expert_team": expert_team,
                "expert_team_differs": different_team,
                "expert_match": "unique_name_position"
                if expert
                else "ambiguous_name"
                if ambiguous
                else "not_listed",
                "rank_gap": float(expert["rank_ecr"] - row["rank"]) if expert else None,
            }
        )
    matched = [r for r in rows if r["expert_rank"] is not None]
    if len(matched) < 2:
        raise ValueError("At least two matched players are needed for comparison")
    # Comparable-population ranks avoid counting unlisted candidates as disagreements.
    # Model point ties and expert rank ties receive average ranks for these metrics.
    model_ranks = rankdata([-r["predicted_ppr"] for r in matched], method="average")
    expert_ranks = rankdata([r["expert_rank"] for r in matched], method="average")
    for row, mr, er in zip(matched, model_ranks, expert_ranks, strict=True):
        row.update(
            model_matched_rank=float(mr),
            expert_matched_rank=float(er),
            matched_rank_gap=float(er - mr),
        )
    rho = float(spearmanr(model_ranks, expert_ranks).statistic)
    overlaps = {}
    for k in (12, 24):
        model_top = {
            r["expert_player_id"]
            for r in rows
            if r["rank"] <= k and r["expert_player_id"] is not None
        }
        expert_top = {str(p["player_id"]) for p in players if p["rank_ecr"] <= k}
        overlaps[str(k)] = {
            "shared": len(model_top & expert_top),
            "model_count": sum(r["rank"] <= k for r in rows),
            "expert_count": len(expert_top),
            "model_names_without_expert_rank": [
                r["player_display_name"]
                for r in rows
                if r["rank"] <= k and r["expert_rank"] is None
            ],
        }
    for row in rows:
        for name in ("model_matched_rank", "expert_matched_rank", "matched_rank_gap"):
            row.setdefault(name, None)
    metrics = {
        "matched": len(matched),
        "model_count": len(rows),
        "expert_count": len(players),
        "spearman": rho if np.isfinite(rho) else None,
        "mean_absolute_matched_rank_gap": float(np.mean(np.abs(model_ranks - expert_ranks))),
        "median_absolute_matched_rank_gap": float(np.median(np.abs(model_ranks - expert_ranks))),
        "within_five_matched_ranks_pct": float(
            100 * np.mean(np.abs(model_ranks - expert_ranks) <= 5)
        ),
        "top_overlap": overlaps,
        "unmatched_experts": [
            {"name": p["player_name"], "rank": p["rank_ecr"], "team": p["player_team_id"]}
            for p in players
            if p["player_id"] not in matched_ids
        ],
        "team_differences": [r["player_display_name"] for r in rows if r["expert_team_differs"]],
        "gap_definition": "expert_rank - model_rank: positive means model ranks player higher",
        "metric_population": "Only uniquely matched players; both lists reranked with average ties",
        "overlap_population": "Original full-list top 12/24, including unmatched players",
    }
    return pl.DataFrame(rows, infer_schema_length=None), metrics


def build_comparison(forecast_dir, expert_path, output):
    forecast_dir, expert_path, output = map(Path, (forecast_dir, expert_path, output))
    source_manifest = json.loads((forecast_dir / "manifest.json").read_text())
    if source_manifest.get("status") != "complete":
        raise ValueError("Forecast is incomplete")
    for name, expected in source_manifest["files"].items():
        if digest(forecast_dir / name) != expected:
            raise ValueError(f"Forecast artifact changed: {name}")
    experts = json.loads(expert_path.read_text())
    expert_manifest = expert_path.with_name("manifest.json")
    if expert_manifest.exists():
        parent = json.loads(expert_manifest.read_text())
        expected = parent.get("artifact_sha256", {}).get(expert_path.name)
        if expected is not None and digest(expert_path) != expected:
            raise ValueError("Expert source checksum mismatch")
    summary = json.loads((forecast_dir / "summary.json").read_text())
    rankings = pl.read_parquet(forecast_dir / "rankings.parquet")
    combined, metrics = compare_rankings(rankings, experts, summary["season"], summary["position"])
    comparison = {
        **metrics,
        "provider": "FantasyPros",
        "scoring": experts["scoring"],
        "type": experts["type"],
        "total_experts": experts["total_experts"],
        "source_date": f"{experts['year']}-{experts['last_updated'].replace('/', '-')}",
        "source_accessed": experts.get("accessed"),
        "source_url": experts.get("source_url"),
        "source_sha256": digest(expert_path),
        "source_file": str(expert_path.resolve()),
        "generated_at": datetime.now(UTC).isoformat(),
        "date_warning": (
            f"Model observations stop after Week {summary['origin']}; the expert snapshot is "
            f"dated {experts['last_updated']}/{experts['year']} and can reflect later information. "
            "This measures agreement, not forecast accuracy or an equal-cutoff comparison."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in ("coverage.csv", "model.json"):
        shutil.copy2(forecast_dir / name, output / name)
    shutil.copy2(expert_path, output / "expert_source.json")
    combined.write_csv(output / "rankings.csv")
    combined.write_parquet(output / "rankings.parquet")
    json_write(output / "rankings.json", combined.to_dicts())
    summary = {**summary, "expert_comparison": comparison}
    json_write(output / "summary.json", summary)
    write_report(output, combined, summary)
    json_write(
        output / "manifest.json",
        {
            "status": "complete",
            "type": "expert_comparison",
            "forecast": str(forecast_dir.resolve()),
            "forecast_manifest_sha256": digest(forecast_dir / "manifest.json"),
            "expert_source_sha256": digest(expert_path),
            "sources": {
                str(p): digest(p) for p in (Path(__file__), Path(write_report.__code__.co_filename))
            },
            "files": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        },
    )
    print(json.dumps(comparison, indent=2))
    print(f"Report: {output / 'index.html'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--experts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_comparison(args.forecast, args.experts, args.output)


if __name__ == "__main__":
    main()
