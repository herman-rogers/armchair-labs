"""Cutoff-safe adapters; no outcome-dependent input screening."""

from __future__ import annotations

import numpy as np
import polars as pl

WEEK_FIELDS = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "league_points",
)
STATIC = (
    "age_at_season",
    "player_experience",
    "draft_pick",
    "draft_round",
    "was_drafted",
    "combine_weight",
    "depth_chart_rank",
    "cutoff_state_observed",
    "cutoff_preseason_rostered",
    "cutoff_preseason_reserve",
    "cutoff_injured_reserve",
    "cutoff_pup_nfi",
    "cutoff_suspended",
    "cutoff_reserve_other",
    "cutoff_transaction_count",
    "cutoff_transaction_recency_days",
    "cutoff_team_observed",
    "known_absence_games",
    "known_suspension_games",
    "known_available_games_cap",
    "team_changed",
    "college_linked",
    "college_partial",
    "college_left_truncated",
    "college_latest_carries",
    "college_latest_passing_ints",
    "college_latest_passing_tds",
    "college_latest_passing_yards",
    "college_latest_receiving_tds",
    "college_latest_receiving_yards",
    "college_latest_receptions",
    "college_latest_rushing_tds",
    "college_latest_rushing_yards",
)


def numeric(frame, columns):
    x = frame.select(pl.col(c).cast(pl.Float64) for c in columns).to_numpy().copy()
    x[~np.isfinite(x)] = np.nan
    return x


def inventory_columns(registry, *, market=False):
    # Do not consume screen results, global aliases, coverage years or correlations.
    names = [
        r["stat"]
        for r in registry
        if r["status"] == "admitted_research_only" and (market or not r["market_input"])
    ]
    if any(c.startswith(("actual_", "fitted_", "adaptive_", "week1_proxy_")) for c in names):
        raise ValueError("Unsafe inventory column")
    return names


def raw_history(frame, weeks):
    """Three calendar seasons of weekly values, plus all earlier annual observations.

    Annual totals are generic sums, unknown if any recorded week is unknown. No
    record stays missing (not a made-up zero). Missing weeks are not absence labels.
    The 25-season layout is fixed in the protocol, independent of the test data.
    """
    weeks = weeks.filter((pl.col("season_type") == "REG") & pl.col("week").is_between(1, 18))
    if weeks.select("player_id", "season", "week").is_duplicated().any():
        raise ValueError("Duplicate weekly key")
    if frame.filter(pl.col("source_season") >= pl.col("forecast_season")).height:
        raise ValueError("Source season crosses forecast cutoff")
    names = list(STATIC)
    groups = ["static:" + c for c in STATIC]
    for lag in range(1, 4):
        for week in range(1, 19):
            for field in WEEK_FIELDS:
                names.append(f"week_lag{lag}_w{week:02d}_{field}")
                groups.append(f"lag{lag}:{field}")
    for lag in range(4, 26):
        for field in WEEK_FIELDS:
            names.append(f"annual_lag{lag}_{field}")
            groups.append("older:" + field)
    matrix = np.full((frame.height, len(names)), np.nan)
    matrix[:, : len(STATIC)] = numeric(frame, STATIC)
    by_key = {}
    annual = {}
    for key, block in weeks.partition_by(["player_id", "season"], as_dict=True).items():
        values = numeric(block, WEEK_FIELDS)
        by_key[key] = {w: v for w, v in zip(block["week"], values, strict=True)}
        annual[key] = values.sum(axis=0)  # NaN propagates for incomplete totals.
    for i, row in enumerate(frame.select("player_id", "forecast_season").iter_rows(named=True)):
        offset = len(STATIC)
        for lag in range(1, 4):
            history = by_key.get((row["player_id"], row["forecast_season"] - lag), {})
            for week in range(1, 19):
                if week in history:
                    matrix[i, offset : offset + len(WEEK_FIELDS)] = history[week]
                offset += len(WEEK_FIELDS)
        for lag in range(4, 26):
            key = (row["player_id"], row["forecast_season"] - lag)
            if key in annual:
                matrix[i, offset : offset + len(WEEK_FIELDS)] = annual[key]
            offset += len(WEEK_FIELDS)
    return matrix, names, groups
