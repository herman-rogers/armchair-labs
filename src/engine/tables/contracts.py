"""Table contracts: observations, cutoff-safe features, and labels have separate tables."""

from __future__ import annotations

import polars as pl

FORECAST_KEY = ["player_id", "forecast_season", "forecast_cutoff_date"]
RICH_SERIES = (
    "points",
    "opportunities",
    "target_share",
    "carry_share",
    "air_yards_share",
    "snap_pct",
    "route_participation",
    "targets_per_route",
    "team_dropbacks",
    "team_carries",
    "team_pass_rate",
    "injury_severity",
    "practice_limit",
    "roster_score",
    "red_zone_opportunities",
    "passing_epa_rate",
    "rushing_epa_rate",
    "receiving_epa_rate",
)
RICH_SUMMARIES = (
    "coverage",
    "mean",
    "std",
    "slope",
    "last4",
    "last4_delta",
    "late_delta",
    "max",
    "q75",
    "zero_rate",
    "lag1",
    "entropy",
    "peak_week",
    "max_jump4",
    "min_jump4",
    "end_streak",
)
RICH_FEATURES = {
    f"rich_s{window}_{series}_{summary}"
    for window in range(3)
    for series in RICH_SERIES
    for summary in RICH_SUMMARIES
} | {
    f"rich_{series}_{summary}"
    for series in RICH_SERIES
    for summary in ("mean_yoy", "last4_yoy", "three_year_trend", "year_stability")
}
LABELS = {
    "actual_ppg",
    "actual_vor",
    "actual_games",
    "actual_season_points",
    "actual_availability_value",
    "actual_matched",
    "outcome_complete",
}
# Explicitly admitted non-model features. Unknown additions stay in enriched data
# until their temporal meaning has been reviewed. Current identity metadata never
# supplies historical team, position, health, or career experience.
FEATURES = set(
    [
        "player_id",
        "forecast_season",
        "forecast_cutoff_date",
        "source_season",
        "season",
        "player_display_name",
        "position",
        "player_population",
        "returning_indicator",
        "rookie_indicator",
        "team",
        "rushing_tds",
        "receiving_tds",
        "passing_tds",
        "attempts",
        "passing_interceptions",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "base_pts",
        "bonus_pts",
        "season_pts",
        "games",
        "total_pts",
        "ppg",
        "floor",
        "volatility",
        "carries",
        "targets",
        "receptions",
        "target_share",
        "air_yards_share",
        "active_games",
        "route_opportunities",
        "route_participation",
        "targets_per_route_opportunity",
        "red_zone_carries",
        "goal_line_carries",
        "goal_line_rushing_tds",
        "designed_carries",
        "scramble_carries",
        "red_zone_targets",
        "end_zone_targets",
        "end_zone_receiving_tds",
        "team_games",
        "team_pass_attempts",
        "team_dropbacks",
        "official_team_carries",
        "injury_report_weeks",
        "out_report_weeks",
        "doubtful_report_weeks",
        "questionable_report_weeks",
        "passing_epa_per_attempt",
        "passing_cpoe",
        "passing_first_down_rate",
        "rushing_first_down_rate",
        "receiving_first_down_rate",
        "ngs_cpoe",
        "ngs_separation",
        "ngs_yac_oe",
        "ngs_ryoe_per_att",
        "neutral_plays",
        "neutral_games",
        "neutral_pass_rate",
        "neutral_epa_per_play",
        "neutral_pass_oe",
        "neutral_plays_pg",
        "injury_data_available",
        "participation_data_available",
        "ppg_denominator_games",
        "birth_date",
        "age_at_season",
        "depth_chart_rank",
        "depth_chart_position",
        "depth_chart_position_group",
        "depth_chart_date",
        "rookie_draft_round",
        "rookie_draft_pick",
        "rookie_was_drafted",
        "rookie_draft_capital_score",
        "rookie_round_one",
        "rookie_day_two",
        "college_major_conference",
        "rookie_age",
        "combine_weight",
        "combine_speed_score",
        "combine_burst_score",
        "combine_agility_score",
        "market_position",
        "market_ecr",
        "market_ecr_score",
        "market_ecr_sd",
        "market_snapshot",
        "market_overall_ecr",
        "market_overall_ecr_score",
        "market_overall_ecr_sd",
        "market_overall_snapshot",
        "market_price_source",
        "cutoff_preseason_team",
        "cutoff_preseason_status",
        "cutoff_preseason_status_score",
        "cutoff_state_resolution",
        "cutoff_state_observed",
        "cutoff_source_url",
        "cutoff_evidence_clause",
        "cutoff_evidence",
        "cutoff_preseason_rostered",
        "cutoff_preseason_reserve",
        "cutoff_availability_class",
        "cutoff_injured_reserve",
        "cutoff_pup_nfi",
        "cutoff_suspended",
        "cutoff_reserve_other",
        "cutoff_transaction_recency_days",
        "cutoff_recent_event_score",
        "cutoff_transaction_matched",
        "cutoff_transaction_count",
        "cutoff_last_transaction_date",
        "team_changed",
        "team_vacated_target_share",
        "team_vacated_carry_share",
        "known_vacated_target_share",
        "known_vacated_carry_share",
        "team_context_observed_share",
        "source_offense_snap_pct",
        "late_offense_snap_pct",
        "source_offense_snaps_pg",
        "offense_snap_pct_trend",
        "draft_round",
        "draft_pick",
        "was_drafted",
        "draft_capital_score",
        "career_seasons_observed",
        "career_games",
        "career_opportunities",
        "career_pass_attempts",
        "player_experience",
        "contract_point_in_time_verified",
        "contract_ambiguous",
        "contract_signing_date",
        "contract_available_date",
        "contract_apy_cap_pct",
        "contract_guaranteed_log",
        "contract_years_remaining",
        "entering_sophomore",
        "source_opportunities_pg",
        "cutoff_team_observed",
        "availability_evidence_status",
        "known_absence_games",
        "known_absence_game_numbers",
        "known_available_games_cap",
        "known_suspension_games",
        "availability_evidence_ids",
        "availability_source_urls",
        "availability_latest_known_on",
    ]
)


def unique(frame: pl.DataFrame, keys: list[str], name: str) -> None:
    if not set(keys) <= set(frame.columns):
        raise ValueError(f"{name}: missing key columns {keys}")
    if frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        raise ValueError(f"{name}: null primary key")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{name}: duplicate primary key {keys}")


def clean_floats(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Undefined source ratios are unknown, never zero, NaN, or infinity in accepted tables."""
    columns = [c for c, dtype in frame.schema.items() if dtype in (pl.Float32, pl.Float64)]
    if not columns:
        return frame, {}
    counts = frame.select(
        (pl.col(c).is_not_null() & ~pl.col(c).is_finite()).sum().alias(c) for c in columns
    ).row(0, named=True)
    changed = {c: n for c, n in counts.items() if n}
    return frame.with_columns(
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in changed
    ), changed


def split_forecasts(inputs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    unique(inputs, FORECAST_KEY, "historical inputs")
    # Explicit pattern for the existing three historical-season windows. Modeled
    # expected-opportunity quantities lack publication vintages and stay quarantined.
    rich = [c for c in inputs.columns if c in RICH_FEATURES]
    columns = [c for c in inputs.columns if c in FEATURES or c in rich]
    features = inputs.select(columns).rename({"team": "source_team", "season": "prior_season"})
    late_market = pl.col("market_snapshot").str.to_date() > pl.col("forecast_cutoff_date")
    late_market_count = features.filter(late_market).height
    unsupported_candidate = (
        (pl.col("player_population") == "market_only")
        & late_market.fill_null(False)
        & pl.col("market_overall_ecr").is_null()
    )
    quarantine = features.filter(unsupported_candidate).select(
        *FORECAST_KEY,
        "player_display_name",
        "player_population",
        pl.lit("Only candidate-admission market evidence is after the cutoff").alias("reason"),
    )
    features = features.with_columns(
        pl.when(late_market).then(None).otherwise(pl.col(name)).alias(name)
        for name in (
            "market_snapshot",
            "market_ecr",
            "market_ecr_score",
            "market_ecr_sd",
            "market_position",
        )
        if name in features.columns
    )
    labels = inputs.select(*FORECAST_KEY, *sorted(LABELS & set(inputs.columns)))
    labels = labels.with_columns(
        pl.when(pl.col("outcome_complete")).then(pl.col(name)).otherwise(None).alias(name)
        for name in LABELS - {"outcome_complete"}
        if name in labels.columns
    )
    features = features.join(quarantine.select(FORECAST_KEY), on=FORECAST_KEY, how="anti")
    labels = labels.join(quarantine.select(FORECAST_KEY), on=FORECAST_KEY, how="anti")
    validate_forecasts(features, labels)
    excluded = [c for c in inputs.columns if c not in columns and c not in LABELS]
    return (
        features.sort(FORECAST_KEY),
        labels.sort(FORECAST_KEY),
        {
            "excluded_from_gold_features": excluded,
            "late_positional_market_rows_quarantined": late_market_count,
            "quarantined_candidates": quarantine.to_dicts(),
            "quarantined_candidate_rows": quarantine.height,
            "pending_outcome_rows_masked": labels.filter(~pl.col("outcome_complete")).height,
            "policy": "Explicit feature allowlist; no labels, Week 1 roster proxies, "
            "undated contract proxies, modeled xFP, or model predictions. Remaining columns "
            "are retained in enriched/historical_inputs for legacy model reproducibility.",
        },
    )


def validate_forecasts(features: pl.DataFrame, labels: pl.DataFrame) -> None:
    unique(features, FORECAST_KEY, "preseason_features")
    unique(labels, FORECAST_KEY, "season_outcomes")
    if (
        not features.select(FORECAST_KEY)
        .sort(FORECAST_KEY)
        .equals(labels.select(FORECAST_KEY).sort(FORECAST_KEY))
    ):
        raise ValueError("Feature and outcome populations differ")
    if any(
        c in LABELS or c.startswith(("actual_", "fitted_", "week1_proxy_"))
        for c in features.columns
    ):
        raise ValueError("Outcome, model output, or future roster proxy in features")
    if features.filter(pl.col("source_season") >= pl.col("forecast_season")).height:
        raise ValueError("Feature source season is not earlier than the forecast")
    cutoff = pl.col("forecast_cutoff_date").cast(pl.Date)
    if features.filter(cutoff.dt.year() != pl.col("forecast_season")).height:
        raise ValueError("Forecast cutoff is in a different season")
    for name in (
        "cutoff_last_transaction_date",
        "availability_latest_known_on",
        "depth_chart_date",
        "contract_signing_date",
        "contract_available_date",
        "market_snapshot",
        "market_overall_snapshot",
    ):
        if name in features.columns:
            evidence_date = pl.col(name)
            if features.schema[name] == pl.String:
                evidence_date = evidence_date.str.slice(0, 10).str.to_date()
            if features.filter(evidence_date.cast(pl.Date) > cutoff).height:
                raise ValueError(f"Evidence after forecast cutoff: {name}")
    for name in LABELS - {"outcome_complete"}:
        if (
            name in labels.columns
            and labels.filter(~pl.col("outcome_complete") & pl.col(name).is_not_null()).height
        ):
            raise ValueError(f"Incomplete-season outcome is populated: {name}")
    if "known_available_games_cap" in features.columns:
        schedule = pl.when(pl.col("forecast_season") >= 2021).then(17).otherwise(16)
        if features.filter(~pl.col("known_available_games_cap").is_between(0, schedule)).height:
            raise ValueError("Known availability exceeds the season schedule")
        if features.filter(
            pl.col("known_absence_games").is_not_null()
            & (pl.col("known_available_games_cap") != schedule - pl.col("known_absence_games"))
        ).height:
            raise ValueError("Known absence and available-game cap disagree")


def coverage(features: pl.DataFrame) -> dict:
    def count(expr: pl.Expr) -> int:
        return features.filter(expr).height

    fields = [
        "cutoff_preseason_team",
        "cutoff_state_observed",
        "depth_chart_rank",
        "contract_available_date",
        "contract_apy_cap_pct",
        "contract_years_remaining",
        "team_vacated_target_share",
        "team_vacated_carry_share",
        "known_vacated_target_share",
        "known_available_games_cap",
        "market_overall_ecr",
    ]
    return {
        "forecast_rows": features.height,
        "coverage_complete": False,
        "observed_roster_state": count(pl.col("cutoff_state_observed") == 1),
        "inferred_prior_team": count(pl.col("cutoff_state_resolution") == "inferred_prior_team"),
        "known_absence": count(pl.col("known_absence_games").fill_null(0) > 0),
        "missing_names": count(pl.col("player_display_name").is_null()),
        "by_season": features.group_by("forecast_season")
        .agg(
            pl.len().alias("rows"),
            (pl.col("cutoff_state_observed") == 1).sum().alias("observed_roster_state"),
            (pl.col("known_absence_games").fill_null(0) > 0).sum().alias("known_absence"),
        )
        .sort("forecast_season")
        .to_dicts(),
        "nonnull_fields": {
            c: features[c].len() - features[c].null_count() for c in fields if c in features.columns
        },
        "interpretation": "Null is unknown. No absence announcement is not proof of availability. "
        "Integrity acceptance is independent of evidence coverage.",
    }


def table_spec(frame: pl.DataFrame, keys: list[str], role: str, description: str) -> dict:
    unique(frame, keys, description)
    # Counts are diagnostics, never an instruction to impute missing values.
    return {
        "rows": frame.height,
        "primary_key": keys,
        "role": role,
        "description": description,
        "schema": {c: str(t) for c, t in frame.schema.items()},
        "null_counts": frame.null_count().row(0, named=True),
    }
