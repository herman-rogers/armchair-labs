"""Exploratory recorded-Out comparisons, not medical injury incidence.

Run: .venv/bin/python research/injury_position_comparison.py --output-dir PATH
Uses only a pinned local gold release; never changes the current catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import chi2, norm

POSITIONS = ["QB", "RB", "TE", "WR"]
# Explicit illness/non-injury/ambiguous primary labels excluded from the narrow outcome.
EXCLUDE = (
    r"(?i)illness|covid|not injury|personal|inactive|^other$|"
    r"migrain|appendi|infection|hernia"
)


def fit_comparison(frame: pl.DataFrame, outcome: str) -> dict:
    """Season fixed effects LPM, CR1 player-cluster covariance, Wald inference.

    This estimates differences in recorded season-level probabilities, not hazards.
    Cluster asymptotic inference handles repeated seasons for the same player.
    """
    position = frame["position"].to_numpy()
    season = frame["season"].to_numpy()
    x = np.column_stack(
        [
            np.ones(frame.height),
            *[(position == p).astype(float) for p in POSITIONS[1:]],
            *[(season == y).astype(float) for y in sorted(set(season))[1:]],
        ]
    )
    y = frame[outcome].to_numpy().astype(float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    _, clusters = np.unique(frame["player_id"].to_numpy(), return_inverse=True)
    g = clusters.max() + 1
    scores = np.zeros((g, x.shape[1]))
    np.add.at(scores, clusters, x * (y - x @ beta)[:, None])
    bread = np.linalg.inv(x.T @ x)
    n, k = x.shape
    covariance = bread @ (scores.T @ scores) @ bread * g / (g - 1) * (n - 1) / (n - k)
    omnibus = float(beta[1:4] @ np.linalg.solve(covariance[1:4, 1:4], beta[1:4]))
    pairs = []
    for a, b in itertools.combinations(POSITIONS, 2):
        contrast = np.zeros(k)
        if a != "QB":
            contrast[POSITIONS.index(a)] = 1
        if b != "QB":
            contrast[POSITIONS.index(b)] -= 1
        difference = float(contrast @ beta)
        se = float(np.sqrt(contrast @ covariance @ contrast))
        pairs.append(
            {
                "a": a,
                "b": b,
                "adjusted_difference_pp": difference * 100,
                "ci95_pointwise_pp": [
                    (difference - 1.96 * se) * 100,
                    (difference + 1.96 * se) * 100,
                ],
                "p": float(2 * norm.sf(abs(difference / se))),
            }
        )
    previous = 0.0
    for rank, index in enumerate(sorted(range(6), key=lambda j: pairs[j]["p"])):
        previous = max(previous, min(1.0, (6 - rank) * pairs[index]["p"]))
        pairs[index]["p_holm"] = previous
    return {
        "player_seasons": n,
        "unique_players": int(g),
        "outcome": outcome,
        "omnibus_chi2": omnibus,
        "omnibus_df": 3,
        "omnibus_p": float(chi2.sf(omnibus, 3)),
        "pairs": pairs,
        "rates": frame.group_by("position")
        .agg(
            pl.len().alias("player_seasons"),
            pl.col(outcome).sum().alias("affected_seasons"),
            (pl.col(outcome).mean() * 100).alias("percent"),
        )
        .sort("percent", descending=True)
        .to_dicts(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-version", default="canonical_nextgen_20260923_r1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    gold = root / "data/gold/releases" / args.gold_version
    names = ["nfl_injuries", "nfl_player_seasons"]
    paths = [gold / "tables" / f"{name}.parquet" for name in names]
    injuries, seasons = [pl.read_parquet(path) for path in paths]
    cohort = seasons.filter(
        pl.col("season").is_between(2009, 2025) & pl.col("position").is_in(POSITIONS)
    ).select("player_id", "season", "position", "games")
    assert cohort.height == cohort.unique(["player_id", "season"]).height
    assert cohort["games"].min() >= 1
    regular = injuries.filter(pl.col("game_type") == "REG")
    latest = (
        regular.sort("date_modified", nulls_last=False)
        .unique(["gsis_id", "season", "week", "team"], keep="last", maintain_order=True)
        .with_columns(
            pl.col("season").cast(pl.Int32),
            pl.col("gsis_id").alias("player_id"),
            (pl.col("report_status") == "Out").fill_null(False).alias("any_out"),
        )
        .with_columns(
            (
                pl.col("any_out")
                & pl.col("report_primary_injury").is_not_null()
                & ~pl.col("report_primary_injury").str.contains(EXCLUDE).fill_null(True)
            ).alias("injury_out")
        )
    )
    flags = latest.group_by("player_id", "season").agg(
        pl.col("any_out").any(),
        pl.col("injury_out").any(),
    )
    cohort = (
        cohort.join(flags, on=["player_id", "season"], how="left")
        .with_columns(pl.col("any_out", "injury_out").fill_null(False))
        .sort("player_id", "season")
    )
    assert cohort.filter(pl.col("injury_out") & ~pl.col("any_out")).is_empty()
    results = {
        "primary_2009_2025": fit_comparison(cohort, "injury_out"),
        "all_out_reasons_2009_2025": fit_comparison(cohort, "any_out"),
        "recent_2016_2025": fit_comparison(cohort.filter(pl.col("season") >= 2016), "injury_out"),
        "through_2024": fit_comparison(cohort.filter(pl.col("season") <= 2024), "injury_out"),
    }
    unmatched = latest.filter(pl.col("injury_out") & pl.col("position").is_in(POSITIONS)).join(
        cohort.select("player_id", "season"), on=["player_id", "season"], how="anti"
    )
    audit = {
        "gold_version": args.gold_version,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*paths, gold / "manifest.json", Path(__file__)]
        },
        "regular_report_rows": regular.height,
        "duplicate_player_team_weeks_removed": regular.height - latest.height,
        "injury_out_player_seasons_outside_cohort": unmatched.select("player_id", "season")
        .unique()
        .height,
        "excluded_primary_label_pattern": EXCLUDE,
        "source_season_coverage": regular.group_by("season")
        .agg(
            pl.len().alias("report_rows"),
            pl.col("team").n_unique().alias("teams"),
            pl.col("week").min().alias("first_week"),
            pl.col("week").max().alias("last_week"),
        )
        .sort("season")
        .to_dicts(),
        "limitations": [
            "Observed gold player-season cohort with at least one statistical appearance; "
            "not all rostered players.",
            "No recorded Out means no captured qualifying designation, not confirmed health.",
            "Out designations are not new injury events or a complete count of missed games; "
            "IR/PUP may be absent.",
            "Narrow outcome uses a heuristic primary-reason filter; "
            "secondary and practice reasons are not used.",
            "Season adjustment and player clustering do not adjust age, workload, role, "
            "team reporting, or selection bias.",
            "Pointwise confidence intervals are unadjusted; "
            "six pairwise p-values use Holm within each analysis.",
            "Sensitivity analyses are exploratory, not additional independent confirmations.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "implementation.py").write_bytes(Path(__file__).read_bytes())
    cohort.write_parquet(args.output_dir / "cohort.parquet")
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print("Unmatched injury-Out player-seasons:", audit["injury_out_player_seasons_outside_cohort"])


if __name__ == "__main__":
    main()
