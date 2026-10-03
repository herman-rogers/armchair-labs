"""Read corrected gold and construct explicit preseason-only predictor blocks."""

from __future__ import annotations

import math

import polars as pl

from engine.metrics.nextgen import build_panel

RATE_PAIRS = {
    "passing": ("passing_yards", "attempts"),
    "rushing": ("rushing_yards", "carries"),
    "receiving": ("receiving_yards", "targets"),
}
FAMILIES = {
    "QB": ("passing", "rushing"),
    "RB": ("rushing", "receiving"),
    "WR": ("receiving",),
    "TE": ("receiving",),
}
MARKET = ["m_log_ecr", "m_ecr_sd", "m_log_overall_ecr", "m_observed"]
PROXY = [
    "p_participation",
    "p_target_earning",
    "p_participation_coverage",
    "p_target_earning_coverage",
]


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def role_state(games: int) -> int:
    """Observed >=15-attempt games: none, 1–7, or 8+. Not an opening-starter label."""
    return 0 if games == 0 else 1 if games < 8 else 2


def make_panel(features, outcomes, weeks) -> list[dict]:
    rows = build_panel(features, outcomes, weeks)
    keys = ["player_id", "forecast_season", "forecast_cutoff_date"]
    source_columns = [
        "source_season",
        "market_ecr",
        "market_ecr_sd",
        "market_overall_ecr",
        "rich_s0_route_participation_mean",
        "rich_s0_route_participation_coverage",
        "rich_s0_targets_per_route_mean",
        "rich_s0_targets_per_route_coverage",
    ]
    originals = {
        (r["player_id"], r["forecast_season"], str(r["forecast_cutoff_date"])): r
        for r in features.select(keys + source_columns).to_dicts()
    }
    regular = weeks.filter(pl.col("season_type") == "REG")
    role_counts = {
        (r["player_id"], r["season"]): r["role_games"]
        for r in regular.group_by("player_id", "season")
        .agg((pl.col("attempts") >= 15).sum().alias("role_games"))
        .to_dicts()
    }
    for row in rows:
        key = tuple(row[k] for k in keys)
        original = originals[key]
        year = row["forecast_season"]
        if original["source_season"] >= year:
            raise ValueError("Source season must precede the forecast season")
        ecr, overall = (finite(original[c]) for c in ("market_ecr", "market_overall_ecr"))
        row.update(
            m_log_ecr=math.log(ecr) if ecr and ecr > 0 else None,
            m_ecr_sd=finite(original["market_ecr_sd"]),
            m_log_overall_ecr=math.log(overall) if overall and overall > 0 else None,
            m_observed=float(ecr is not None and ecr > 0),
        )
        for label, field in (
            ("participation", "route_participation"),
            ("target_earning", "targets_per_route"),
        ):
            coverage = finite(original[f"rich_s0_{field}_coverage"])
            row[f"p_{label}_coverage"] = coverage
            row[f"p_{label}"] = (
                finite(original[f"rich_s0_{field}_mean"])
                if coverage is not None and coverage > 0
                else None
            )
        row["proxy_observed"] = all(
            row[f"p_{s}"] is not None for s in ("participation", "target_earning")
        )
        row["proxy_half_season"] = all(
            (row[f"p_{s}_coverage"] or 0) >= 0.5 for s in ("participation", "target_earning")
        )
        # Historical role duration is a predictor. Same-season role is LABEL ONLY.
        row["r_prior_role_games"] = role_counts.get((row["player_id"], year - 1), 0)
        row["y_role_state"] = (
            role_state(role_counts.get((row["player_id"], year), 0))
            if row["outcome_complete"]
            else None
        )
        for family, (numerator, denominator) in RATE_PAIRS.items():
            n = finite(row[f"x_prior_{denominator}"])
            total = finite(row[f"x_prior_{numerator}"])
            row[f"rate_{family}"] = total / n if total is not None and n and n > 0 else None
            row[f"exposure_{family}"] = n if row[f"rate_{family}"] is not None else None
    return rows
