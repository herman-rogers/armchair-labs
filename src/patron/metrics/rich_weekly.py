"""Rich, cutoff-safe player-week panel for automated feature discovery.

The ordinary discovery experiment sees weekly box-score sequences.  This module adds
the role and context that a zero-point week cannot explain by itself: offensive snaps,
dropback participation, target rate per route opportunity, injury/practice status,
weekly roster state, expected fantasy opportunity, team volume, and red-zone usage.

Every output row summarizes only seasons before ``forecast_season`` (three by
default). Source coverage is represented explicitly, so a feed unavailable in an
older era is null rather than fabricated zero evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl

from patron.metrics.positions import canonical_positions
from patron.scoring.columns import require_columns

_POSITIONS = ("QB", "RB", "WR", "TE")
_ROUTE_POSITIONS = frozenset({"RB", "FB", "HB", "WR", "TE"})
_SKILL_POSITIONS = frozenset({"QB", "RB", "FB", "HB", "WR", "TE"})

RICH_SIGNALS: tuple[tuple[str, str | None], ...] = (
    ("points", None),
    ("opportunities", None),
    ("target_share", None),
    ("carry_share", None),
    ("air_yards_share", None),
    ("snap_pct", "snap_data_available"),
    ("route_participation", "route_data_available"),
    ("targets_per_route", "route_data_available"),
    ("xfp", "xfp_data_available"),
    ("expected_td", "xfp_data_available"),
    ("team_dropbacks", None),
    ("team_carries", None),
    ("team_pass_rate", None),
    ("injury_severity", "injury_data_available"),
    ("practice_limit", "injury_data_available"),
    ("roster_score", "roster_data_available"),
    ("red_zone_opportunities", "play_data_available"),
    ("passing_epa_rate", None),
    ("rushing_epa_rate", None),
    ("receiving_epa_rate", None),
)


def _split(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


def build_weekly_route_panel(
    player_weeks: pl.DataFrame,
    plays: pl.DataFrame,
    participation: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate play participation and high-value touches to player-week rows."""
    require_columns(
        player_weeks.columns,
        ("player_id", "season", "position"),
        "rich weekly player position input",
    )
    require_columns(
        plays.columns,
        (
            "game_id",
            "play_id",
            "season",
            "week",
            "season_type",
            "posteam",
            "qb_dropback",
            "rush_attempt",
            "qb_scramble",
            "qb_kneel",
            "pass_attempt",
            "receiver_player_id",
            "rusher_player_id",
            "yardline_100",
            "two_point_attempt",
        ),
        "rich weekly play input",
    )
    require_columns(
        participation.columns,
        (
            "nflverse_game_id",
            "play_id",
            "possession_team",
            "offense_players",
            "offense_positions",
        ),
        "rich weekly participation input",
    )
    positions = {
        (str(row["player_id"]), int(row["season"])): str(row["position"] or "").upper()
        for row in player_weeks.select("player_id", "season", "position")
        .filter(pl.col("player_id").is_not_null())
        .unique(subset=["player_id", "season"], keep="last")
        .iter_rows(named=True)
    }
    valid: dict[tuple[str, float], tuple[int, int, str, bool]] = {}
    valuable: dict[tuple[str, int, int], dict[str, float]] = defaultdict(
        lambda: {"red_zone_opportunities": 0.0, "designed_carries": 0.0, "scrambles": 0.0}
    )
    for row in plays.unique(subset=["game_id", "play_id"]).iter_rows(named=True):
        if row.get("season_type") != "REG" or _number(row.get("two_point_attempt")) == 1:
            continue
        if row.get("play_type") == "no_play":
            continue
        week = int(row.get("week") or 0)
        if not 1 <= week <= 18:
            continue
        game_id = str(row.get("game_id") or "")
        team = str(row.get("posteam") or "")
        if not game_id or not team or row.get("play_id") is None:
            continue
        season = int(row["season"])
        dropback = _number(row.get("qb_dropback")) == 1
        valid[(game_id, float(row["play_id"]))] = (season, week, team, dropback)
        yardline = row.get("yardline_100")
        receiver = row.get("receiver_player_id")
        if (
            receiver
            and _number(row.get("pass_attempt")) == 1
            and yardline is not None
            and float(yardline) <= 20
        ):
            valuable[(str(receiver), season, week)]["red_zone_opportunities"] += 1
        rusher = row.get("rusher_player_id")
        if rusher and _number(row.get("rush_attempt")) == 1 and _number(row.get("qb_kneel")) != 1:
            usage = valuable[(str(rusher), season, week)]
            if yardline is not None and float(yardline) <= 20:
                usage["red_zone_opportunities"] += 1
            if _number(row.get("qb_scramble")) == 1:
                usage["scrambles"] += 1
            else:
                usage["designed_carries"] += 1

    role: dict[tuple[str, int, int], dict[str, object]] = defaultdict(
        lambda: {"route_opportunities": 0.0, "offensive_plays": 0.0, "team": None}
    )
    expected_dropbacks: dict[tuple[int, int, str], int] = defaultdict(int)
    observed_dropbacks: dict[tuple[int, int, str], int] = defaultdict(int)
    for season, week, team, dropback in valid.values():
        expected_dropbacks[(season, week, team)] += int(dropback)
    # Identical duplicate feed rows must not count a second play. Conflicting
    # versions are excluded, not arbitrarily selected from input order.
    participation = (
        participation.unique()
        .with_columns(pl.len().over("nflverse_game_id", "play_id").alias("_versions"))
        .filter(pl.col("_versions") == 1)
    )
    for row in participation.iter_rows(named=True):
        game_id = str(row.get("nflverse_game_id") or "")
        play_id = row.get("play_id")
        if not game_id or play_id is None:
            continue
        play = valid.get((game_id, float(play_id)))
        if play is None:
            continue
        season, week, play_team, dropback = play
        ids = _split(row.get("offense_players"))
        supplied = _split(row.get("offense_positions"))
        team = str(row.get("possession_team") or play_team)
        team = {"OAK": "LV", "STL": "LA", "SD": "LAC"}.get(team, team)
        if not ids or team != play_team:
            continue
        observed_dropbacks[(season, week, team)] += int(dropback)
        seen_players: set[str] = set()
        for index, player_id in enumerate(ids):
            if player_id in seen_players:
                continue
            seen_players.add(player_id)
            position = supplied[index].upper() if index < len(supplied) else ""
            position = position or positions.get((player_id, season), "")
            if position not in _SKILL_POSITIONS:
                continue
            role_usage = role[(player_id, season, week)]
            role_usage["team"] = team
            role_usage["offensive_plays"] = _number(role_usage["offensive_plays"]) + 1
            if dropback and position in _ROUTE_POSITIONS:
                role_usage["route_opportunities"] = _number(role_usage["route_opportunities"]) + 1

    keys = set(role) | set(valuable)
    rows: list[dict[str, object]] = []
    for player_id, season, week in sorted(keys):
        weekly_role = role.get((player_id, season, week), {})
        team_key = (season, week, str(weekly_role.get("team") or ""))
        rows.append(
            {
                "player_id": player_id,
                "season": season,
                "week": week,
                "route_team": weekly_role.get("team"),
                "route_opportunities": weekly_role.get("route_opportunities"),
                "offensive_plays": weekly_role.get("offensive_plays"),
                "route_team_dropbacks": observed_dropbacks.get(team_key),
                "route_expected_dropbacks": expected_dropbacks.get(team_key),
                **valuable[(player_id, season, week)],
            }
        )
    if not rows:
        return pl.DataFrame()
    # Team can remain null for a long prefix when participation is sparse. Infer over
    # every row so a later string team code is not forced into a null-typed builder.
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)
    )


def _canonical_stats(player_weeks: pl.DataFrame) -> pl.DataFrame:
    required = (
        "player_id",
        "season",
        "week",
        "position",
        "team",
        "fantasy_points_ppr",
        "attempts",
        "passing_epa",
        "carries",
        "rushing_epa",
        "targets",
        "receiving_epa",
        "target_share",
        "air_yards_share",
    )
    require_columns(player_weeks.columns, required, "rich weekly stat input")
    return (
        canonical_positions(player_weeks)
        .filter(pl.col("position").is_in(_POSITIONS) & pl.col("week").is_between(1, 18))
        .group_by("player_id", "season", "week")
        .agg(
            pl.col("position").last(),
            pl.col("team").last().alias("stat_team"),
            pl.col("fantasy_points_ppr").fill_null(0).sum().alias("points"),
            (
                pl.col("attempts").fill_null(0).sum()
                + pl.col("carries").fill_null(0).sum()
                + pl.col("targets").fill_null(0).sum()
            ).alias("opportunities"),
            pl.col("attempts").fill_null(0).sum(),
            pl.col("carries").fill_null(0).sum(),
            pl.col("targets").fill_null(0).sum(),
            pl.col("target_share").mean().alias("target_share_source"),
            pl.col("air_yards_share").mean(),
            pl.col("passing_epa").fill_null(0).sum(),
            pl.col("rushing_epa").fill_null(0).sum(),
            pl.col("receiving_epa").fill_null(0).sum(),
        )
        .with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    )


def _canonical_snaps(snap_counts: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    require_columns(
        snap_counts.columns,
        (
            "season",
            "week",
            "game_type",
            "pfr_player_id",
            "position",
            "team",
            "offense_snaps",
            "offense_pct",
        ),
        "rich weekly snap input",
    )
    require_columns(players.columns, ("pfr_id", "gsis_id"), "rich weekly player crosswalk")
    identity = (
        players.select(
            pl.col("pfr_id").alias("pfr_player_id"), pl.col("gsis_id").alias("player_id")
        )
        .drop_nulls()
        .filter((pl.col("pfr_player_id") != "") & (pl.col("player_id") != ""))
        .unique(subset="pfr_player_id", keep="first")
    )
    return (
        snap_counts.filter(
            (pl.col("game_type") == "REG")
            & pl.col("week").is_between(1, 18)
            & pl.col("position").str.to_uppercase().is_in(_SKILL_POSITIONS)
        )
        .join(identity, on="pfr_player_id", how="inner")
        .group_by("player_id", "season", "week")
        .agg(
            pl.col("team").last().alias("snap_team"),
            pl.col("offense_snaps").cast(pl.Float64, strict=False).sum(),
            pl.col("offense_pct").cast(pl.Float64, strict=False).mean().alias("snap_pct"),
        )
        .with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    )


def _canonical_rosters(rosters: pl.DataFrame) -> pl.DataFrame:
    require_columns(
        rosters.columns,
        ("season", "week", "game_type", "gsis_id", "position", "team", "status"),
        "rich weekly roster input",
    )
    status = pl.col("status").fill_null("").str.to_uppercase()
    return (
        rosters.filter(
            (pl.col("game_type") == "REG")
            & pl.col("week").is_between(1, 18)
            & pl.col("position").is_in(_POSITIONS)
            & pl.col("gsis_id").is_not_null()
        )
        .with_columns(
            pl.when(status == "ACT")
            .then(1.0)
            .when(status == "INA")
            .then(0.35)
            .when(status == "RES")
            .then(0.1)
            .otherwise(0.0)
            .alias("roster_score")
        )
        .group_by(pl.col("gsis_id").alias("player_id"), "season", "week")
        .agg(
            pl.col("position").last().alias("roster_position"),
            pl.col("team").last().alias("roster_team"),
            pl.col("roster_score").max(),
        )
        .with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    )


def _canonical_injuries(injuries: pl.DataFrame) -> pl.DataFrame:
    require_columns(
        injuries.columns,
        ("season", "week", "gsis_id", "team", "report_status", "practice_status"),
        "rich weekly injury input",
    )
    report = pl.col("report_status").fill_null("").str.to_uppercase()
    practice = pl.col("practice_status").fill_null("").str.to_uppercase()
    return (
        injuries.filter(pl.col("gsis_id").is_not_null() & pl.col("week").is_between(1, 18))
        .with_columns(
            pl.when(report == "OUT")
            .then(1.0)
            .when(report == "DOUBTFUL")
            .then(0.75)
            .when(report == "QUESTIONABLE")
            .then(0.4)
            .otherwise(0.0)
            .alias("injury_severity"),
            pl.when(practice.str.contains("DID NOT"))
            .then(1.0)
            .when(practice.str.contains("LIMITED"))
            .then(0.5)
            .otherwise(0.0)
            .alias("practice_limit"),
        )
        .group_by(pl.col("gsis_id").alias("player_id"), "season", "week")
        .agg(
            pl.col("team").last().alias("injury_team"),
            pl.col("injury_severity").max(),
            pl.col("practice_limit").max(),
        )
        .with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    )


def _canonical_expected(expected: pl.DataFrame) -> pl.DataFrame:
    require_columns(
        expected.columns,
        (
            "season",
            "week",
            "player_id",
            "posteam",
            "total_fantasy_points_exp",
            "total_touchdown_exp",
        ),
        "rich weekly expected-opportunity input",
    )
    return (
        expected.filter(pl.col("player_id").is_not_null() & pl.col("week").is_between(1, 18))
        .with_columns(
            pl.col("season").cast(pl.Int32, strict=False),
            pl.col("week").cast(pl.Int32, strict=False),
        )
        .group_by("player_id", "season", "week")
        .agg(
            pl.col("posteam").last().alias("xfp_team"),
            pl.col("total_fantasy_points_exp").fill_null(0).sum().alias("xfp"),
            pl.col("total_touchdown_exp").fill_null(0).sum().alias("expected_td"),
        )
    )


def build_rich_weekly_panel(
    player_weeks: pl.DataFrame,
    team_weeks: pl.DataFrame,
    snap_counts: pl.DataFrame,
    players: pl.DataFrame,
    injuries: pl.DataFrame,
    rosters: pl.DataFrame,
    expected: pl.DataFrame,
    route_panel: pl.DataFrame,
) -> pl.DataFrame:
    """Join every weekly source while retaining source-coverage indicators."""
    keys = ["player_id", "season", "week"]
    player_weeks = canonical_positions(player_weeks)
    rosters = canonical_positions(rosters)
    stats = _canonical_stats(player_weeks)
    snaps = _canonical_snaps(snap_counts, players)
    roster = _canonical_rosters(rosters)
    injury = _canonical_injuries(injuries)
    xfp = _canonical_expected(expected)
    panel = stats.join(roster, on=keys, how="full", coalesce=True)
    for source in (snaps, injury, xfp, route_panel):
        if source.height:
            panel = panel.join(source, on=keys, how="full", coalesce=True)
    # Old cached route panels do not have a compatible denominator. Fail closed.
    panel = panel.with_columns(
        *(
            pl.lit(None, dtype=pl.Float64).alias(name)
            for name in ("route_team_dropbacks", "route_expected_dropbacks", "route_opportunities")
            if name not in panel.columns
        )
    )
    panel = panel.with_columns(
        pl.coalesce(
            "position",
            "roster_position",
        ).alias("position"),
        pl.coalesce(
            "roster_team", "stat_team", "snap_team", "route_team", "injury_team", "xfp_team"
        ).alias("team"),
    )
    require_columns(
        team_weeks.columns,
        ("season", "week", "team", "attempts", "sacks_suffered", "carries"),
        "rich weekly team input",
    )
    team = team_weeks.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "team",
        (pl.col("attempts").fill_null(0) + pl.col("sacks_suffered").fill_null(0)).alias(
            "team_dropbacks"
        ),
        pl.col("carries").fill_null(0).alias("team_carries"),
    )
    panel = panel.join(team, on=["season", "week", "team"], how="left")
    source_seasons = {
        "snap_data_available": set(snaps["season"].unique().to_list()),
        "route_data_available": set(
            route_panel.filter(pl.col("offensive_plays").is_not_null())["season"].unique().to_list()
        )
        if route_panel.height
        else set(),
        "play_data_available": set(route_panel["season"].unique().to_list())
        if route_panel.height
        else set(),
        "injury_data_available": set(injury["season"].unique().to_list()),
        "roster_data_available": set(roster["season"].unique().to_list()),
        "xfp_data_available": set(xfp["season"].unique().to_list()),
    }
    panel = panel.with_columns(
        *(pl.col("season").is_in(values).alias(name) for name, values in source_seasons.items())
    )
    panel = panel.with_columns(
        (pl.col("route_team_dropbacks") / pl.col("route_expected_dropbacks")).alias(
            "route_play_coverage"
        ),
        (
            pl.col("route_team_dropbacks").is_not_null()
            & (pl.col("route_team_dropbacks") == pl.col("route_expected_dropbacks"))
            & pl.col("route_opportunities").is_not_null()
        )
        .fill_null(False)
        .alias("route_data_available"),
    )
    attempts = pl.col("attempts").fill_null(0.0)
    carries = pl.col("carries").fill_null(0.0)
    targets = pl.col("targets").fill_null(0.0)
    route_opportunities = pl.col("route_opportunities")
    return (
        panel.with_columns(
            pl.col("points").fill_null(0.0),
            pl.col("opportunities").fill_null(0.0),
            pl.when(pl.col("team_dropbacks") > 0)
            .then(targets / pl.col("team_dropbacks"))
            .otherwise(pl.col("target_share_source"))
            .alias("target_share"),
            pl.when(pl.col("team_carries") > 0)
            .then(carries / pl.col("team_carries"))
            .otherwise(None)
            .alias("carry_share"),
            pl.when(pl.col("team_dropbacks") + pl.col("team_carries") > 0)
            .then(pl.col("team_dropbacks") / (pl.col("team_dropbacks") + pl.col("team_carries")))
            .otherwise(None)
            .alias("team_pass_rate"),
            pl.when(pl.col("route_data_available") & (pl.col("route_team_dropbacks") > 0))
            .then(route_opportunities / pl.col("route_team_dropbacks"))
            .otherwise(None)
            .alias("route_participation"),
            pl.when(pl.col("route_data_available") & (route_opportunities > 0))
            .then(targets / route_opportunities)
            .otherwise(None)
            .alias("targets_per_route"),
            pl.when(attempts > 0)
            .then(pl.col("passing_epa") / attempts)
            .otherwise(None)
            .alias("passing_epa_rate"),
            pl.when(carries > 0)
            .then(pl.col("rushing_epa") / carries)
            .otherwise(None)
            .alias("rushing_epa_rate"),
            pl.when(targets > 0)
            .then(pl.col("receiving_epa") / targets)
            .otherwise(None)
            .alias("receiving_epa_rate"),
        )
        .with_columns(
            *(
                pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
                for name, _ in RICH_SIGNALS
            )
        )
        .select(
            "player_id",
            "season",
            "week",
            "position",
            "team",
            "route_team_dropbacks",
            "route_expected_dropbacks",
            "route_play_coverage",
            *(name for name, _ in RICH_SIGNALS),
            *dict.fromkeys(name for _, name in RICH_SIGNALS if name is not None),
        )
        .sort("season", "week", "player_id")
    )


def _descriptor(values: np.ndarray) -> dict[str, float]:
    present = np.isfinite(values)
    coverage = float(present.mean())
    if not np.any(present):
        return {
            name: math.nan
            for name in (
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
        }
    observed = values[present]
    weeks = np.arange(1.0, 19.0)[present]
    mean = float(observed.mean())
    std = float(observed.std())
    slope = float(np.polyfit(weeks, observed, 1)[0]) if len(observed) > 1 else 0.0
    adjacent = present[:-1] & present[1:]
    if np.any(adjacent):
        left = values[:-1][adjacent]
        right = values[1:][adjacent]
        denominator = float(np.sqrt(np.sum((left - mean) ** 2) * np.sum((right - mean) ** 2)))
        lag1 = float(np.sum((left - mean) * (right - mean)) / denominator) if denominator else 0.0
    else:
        lag1 = 0.0
    absolute = np.abs(observed)
    probabilities = absolute / absolute.sum() if absolute.sum() else np.zeros_like(absolute)
    positive = probabilities[probabilities > 0]
    entropy = (
        float(-np.sum(positive * np.log(positive)) / math.log(len(values)))
        if len(positive)
        else 0.0
    )

    def observed_mean(window: np.ndarray) -> float:
        finite = window[np.isfinite(window)]
        return float(finite.mean()) if len(finite) else math.nan

    rolling = np.array([observed_mean(values[start : start + 4]) for start in range(15)])
    jumps = np.diff(rolling)
    jumps = jumps[np.isfinite(jumps)]
    streak = 0
    for value in values[::-1]:
        if not math.isfinite(value) or value <= 0:
            break
        streak += 1
    return {
        "coverage": coverage,
        "mean": mean,
        "std": std,
        "slope": slope,
        "last4": observed_mean(values[-4:]),
        "last4_delta": observed_mean(values[-4:]) - observed_mean(values[-8:-4]),
        "late_delta": observed_mean(values[9:]) - observed_mean(values[:9]),
        "max": float(observed.max()),
        "q75": float(np.quantile(observed, 0.75)),
        "zero_rate": float(np.mean(observed == 0)),
        "lag1": lag1,
        "entropy": entropy,
        "peak_week": float(np.nanargmax(values) + 1) / 18.0,
        "max_jump4": float(jumps.max()) if len(jumps) else math.nan,
        "min_jump4": float(jumps.min()) if len(jumps) else math.nan,
        "end_streak": float(streak) / 18.0 if present[-1] else math.nan,
    }


def build_rich_weekly_features(
    panel: pl.DataFrame,
    forecast_rows: pl.DataFrame,
    *,
    history_seasons: int = 3,
) -> pl.DataFrame:
    """Summarize the requested history, keeping named three-year trends to three years."""
    if history_seasons < 1:
        raise ValueError("history_seasons must be positive")
    require_columns(
        forecast_rows.columns,
        ("forecast_season", "player_id"),
        "rich weekly forecast rows",
    )
    by_player_season: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    season_coverage: dict[int, dict[str, bool]] = defaultdict(dict)
    for row in panel.iter_rows(named=True):
        season = int(row["season"])
        by_player_season[(str(row["player_id"]), season)].append(row)
        for _, availability in RICH_SIGNALS:
            if availability is not None and row.get(availability):
                season_coverage[season][availability] = True
    output: list[dict[str, object]] = []
    forecast_keys = forecast_rows.select("forecast_season", "player_id").unique()
    for forecast in forecast_keys.iter_rows(named=True):
        forecast_season = int(forecast["forecast_season"])
        player_id = str(forecast["player_id"])
        result: dict[str, object] = {
            "forecast_season": forecast_season,
            "player_id": player_id,
        }
        summaries: dict[str, list[dict[str, float]]] = defaultdict(list)
        for lag in range(history_seasons):
            season = forecast_season - 1 - lag
            rows = by_player_season.get((player_id, season), [])
            for signal, availability in RICH_SIGNALS:
                source_available = availability is None or season_coverage[season].get(
                    availability, False
                )
                missing_is_unknown = signal in {
                    "route_participation",
                    "targets_per_route",
                    "snap_pct",
                    "xfp",
                    "expected_td",
                }
                values = np.full(
                    18, 0.0 if source_available and not missing_is_unknown else np.nan, dtype=float
                )
                if source_available:
                    for row in rows:
                        week = int(row["week"]) - 1
                        value = row.get(signal)
                        if value is not None and isinstance(value, (int, float)):
                            values[week] = float(value)
                summary = _descriptor(values)
                summaries[signal].append(summary)
                for name, value in summary.items():
                    result[f"rich_s{lag}_{signal}_{name}"] = value if math.isfinite(value) else None
        for signal, seasons in summaries.items():
            source = seasons[0]
            previous = seasons[1] if len(seasons) > 1 else {"mean": math.nan, "last4": math.nan}
            mean_yoy = source["mean"] - previous["mean"]
            last4_yoy = source["last4"] - previous["last4"]
            result[f"rich_{signal}_mean_yoy"] = mean_yoy if math.isfinite(mean_yoy) else None
            result[f"rich_{signal}_last4_yoy"] = last4_yoy if math.isfinite(last4_yoy) else None
            means = np.array([summary["mean"] for summary in reversed(seasons[:3])])
            finite = np.isfinite(means)
            three_year_trend = (
                float(np.polyfit(np.arange(len(means))[finite], means[finite], 1)[0])
                if finite.sum() > 1
                else math.nan
            )
            year_stability = float(np.std(means[finite])) if finite.any() else math.nan
            result[f"rich_{signal}_three_year_trend"] = (
                three_year_trend if math.isfinite(three_year_trend) else None
            )
            result[f"rich_{signal}_year_stability"] = (
                year_stability if math.isfinite(year_stability) else None
            )
        output.append(result)
    return pl.DataFrame(output, infer_schema_length=None).with_columns(
        pl.col("forecast_season").cast(pl.Int32)
    )


def rich_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(name for name in columns if name.startswith("rich_"))
