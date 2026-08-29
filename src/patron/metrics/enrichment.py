"""nflverse-first projection inputs beyond the weekly fantasy stat line.

The participation feed identifies the offensive players present on each play, but it
does not chart a route for every eligible receiver.  We therefore call an eligible
player's presence on a dropback a ``route opportunity``.  It is a reproducible,
useful proxy for routes run, and is deliberately exported under that honest name.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import polars as pl

from patron.scoring.columns import require_columns

_SKILL_POSITIONS = frozenset({"QB", "RB", "FB", "HB", "WR", "TE"})
_ROUTE_POSITIONS = frozenset({"RB", "FB", "HB", "WR", "TE"})


def _number(value: Any) -> float:
    return float(value or 0.0)


def _split(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def build_team_volume(team_weeks: pl.DataFrame) -> pl.DataFrame:
    """Aggregate official attempts, carries, and games to team-season rows."""
    require_columns(
        team_weeks.columns,
        ("season", "team", "week", "attempts", "sacks_suffered", "carries"),
        "team volume input",
    )
    return (
        team_weeks.group_by(["season", "team"])
        .agg(
            pl.col("week").n_unique().alias("team_games"),
            pl.col("attempts").fill_null(0).sum().alias("team_pass_attempts"),
            (
                pl.col("attempts").fill_null(0).sum() + pl.col("sacks_suffered").fill_null(0).sum()
            ).alias("team_dropbacks"),
            pl.col("carries").fill_null(0).sum().alias("official_team_carries"),
        )
        .sort(["season", "team"])
    )


def build_player_usage(
    player_weeks: pl.DataFrame,
    pbp: pl.DataFrame,
    participation: pl.DataFrame,
) -> pl.DataFrame:
    """Derive route opportunities, active games, and high-value touches.

    A route opportunity is an RB/FB/WR/TE appearing in nflverse's offensive
    participation list on an official quarterback dropback. Tight ends and backs can
    remain in protection, so this is intentionally not presented as a charted route.
    """
    require_columns(
        pbp.columns,
        (
            "game_id",
            "play_id",
            "season",
            "season_type",
            "posteam",
            "pass_attempt",
            "rush_attempt",
            "qb_dropback",
            "qb_scramble",
            "qb_kneel",
            "yardline_100",
            "air_yards",
            "pass_touchdown",
            "rush_touchdown",
            "receiver_player_id",
            "rusher_player_id",
            "two_point_attempt",
        ),
        "projection play-by-play input",
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
        "participation input",
    )
    require_columns(
        player_weeks.columns,
        ("player_id", "season", "targets"),
        "player usage weekly input",
    )

    valid_plays: dict[tuple[str, float], dict[str, Any]] = {}
    dropbacks_by_game_team: dict[tuple[str, str], int] = defaultdict(int)
    high_value: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {
            "red_zone_carries": 0.0,
            "goal_line_carries": 0.0,
            "goal_line_rushing_tds": 0.0,
            "designed_carries": 0.0,
            "red_zone_targets": 0.0,
            "end_zone_targets": 0.0,
            "end_zone_receiving_tds": 0.0,
        }
    )

    for row in pbp.iter_rows(named=True):
        if row.get("season_type") != "REG" or _number(row.get("two_point_attempt")) == 1:
            continue
        game_id = str(row.get("game_id") or "")
        team = str(row.get("posteam") or "")
        if not game_id or not team:
            continue
        key = (game_id, float(row["play_id"]))
        valid_plays[key] = row
        season = int(row["season"])
        yardline = row.get("yardline_100")

        if _number(row.get("qb_dropback")) == 1:
            dropbacks_by_game_team[(game_id, team)] += 1

        rusher = row.get("rusher_player_id")
        if rusher and _number(row.get("rush_attempt")) == 1 and _number(row.get("qb_kneel")) != 1:
            usage = high_value[(str(rusher), season)]
            if yardline is not None and float(yardline) <= 20:
                usage["red_zone_carries"] += 1
            if yardline is not None and float(yardline) <= 5:
                usage["goal_line_carries"] += 1
                if _number(row.get("rush_touchdown")) == 1:
                    usage["goal_line_rushing_tds"] += 1
            if _number(row.get("qb_scramble")) != 1:
                usage["designed_carries"] += 1

        receiver = row.get("receiver_player_id")
        if receiver and _number(row.get("pass_attempt")) == 1:
            usage = high_value[(str(receiver), season)]
            if yardline is not None and float(yardline) <= 20:
                usage["red_zone_targets"] += 1
            air_yards = row.get("air_yards")
            if (
                yardline is not None
                and air_yards is not None
                and float(air_yards) >= float(yardline)
            ):
                usage["end_zone_targets"] += 1
                if _number(row.get("pass_touchdown")) == 1:
                    usage["end_zone_receiving_tds"] += 1

    route_opportunities: dict[tuple[str, int], int] = defaultdict(int)
    active_games: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for row in participation.iter_rows(named=True):
        game_id = str(row.get("nflverse_game_id") or "")
        play_id = row.get("play_id")
        if not game_id or play_id is None:
            continue
        play = valid_plays.get((game_id, float(play_id)))
        if play is None:
            continue
        ids = _split(row.get("offense_players"))
        positions = _split(row.get("offense_positions"))
        team = str(row.get("possession_team") or play.get("posteam") or "")
        season = int(play["season"])
        is_dropback = _number(play.get("qb_dropback")) == 1
        for player_id, position in zip(ids, positions, strict=False):
            normalized_position = position.upper()
            if normalized_position not in _SKILL_POSITIONS:
                continue
            player_key = (player_id, season)
            active_games[player_key].add((game_id, team))
            if is_dropback and normalized_position in _ROUTE_POSITIONS:
                route_opportunities[player_key] += 1

    target_totals = {
        (str(row["player_id"]), int(row["season"])): _number(row.get("targets"))
        for row in player_weeks.group_by(["player_id", "season"])
        .agg(pl.col("targets").fill_null(0).sum().alias("targets"))
        .iter_rows(named=True)
    }
    keys = set(target_totals) | set(active_games) | set(route_opportunities) | set(high_value)
    rows: list[dict[str, Any]] = []
    for player_id, season in sorted(keys):
        games = active_games.get((player_id, season), set())
        active_dropbacks = sum(dropbacks_by_game_team.get(game, 0) for game in games)
        route_opp = route_opportunities.get((player_id, season), 0)
        targets = target_totals.get((player_id, season), 0.0)
        valuable = high_value[(player_id, season)]
        rows.append(
            {
                "player_id": player_id,
                "season": season,
                "active_games": len(games),
                "active_game_dropbacks": active_dropbacks,
                "route_opportunities": route_opp,
                "route_participation": route_opp / active_dropbacks if active_dropbacks else None,
                "targets_per_route_opportunity": targets / route_opp if route_opp else None,
                **valuable,
            }
        )

    return pl.DataFrame(rows, infer_schema_length=None)


def build_injury_history(injuries: pl.DataFrame) -> pl.DataFrame:
    """Summarise official injury-report burden by player and season."""
    require_columns(
        injuries.columns,
        (
            "season",
            "week",
            "gsis_id",
            "report_primary_injury",
            "report_status",
            "practice_primary_injury",
            "practice_status",
        ),
        "injury history input",
    )
    valid = injuries.filter(pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != ""))
    status = pl.col("report_status").fill_null("").str.to_uppercase()
    injury_text = pl.concat_str(
        [
            pl.col("report_primary_injury").fill_null(""),
            pl.col("practice_primary_injury").fill_null(""),
        ],
        separator=" ",
    ).str.to_lowercase()
    has_injury = (injury_text.str.len_chars() > 1) & ~injury_text.str.contains("not injury related")
    return (
        valid.group_by(["gsis_id", "season"])
        .agg(
            pl.col("week").filter(has_injury).n_unique().alias("injury_report_weeks"),
            pl.col("week").filter(status == "OUT").n_unique().alias("out_report_weeks"),
            pl.col("week").filter(status == "DOUBTFUL").n_unique().alias("doubtful_report_weeks"),
            pl.col("week")
            .filter(status == "QUESTIONABLE")
            .n_unique()
            .alias("questionable_report_weeks"),
        )
        .rename({"gsis_id": "player_id"})
        .sort(["player_id", "season"])
    )


def current_depth_chart(depth_charts: pl.DataFrame) -> pl.DataFrame:
    """Select the latest nflverse offensive depth entry for each GSIS player."""
    require_columns(
        depth_charts.columns,
        ("dt", "team", "gsis_id", "pos_abb", "pos_name", "pos_rank"),
        "depth chart input",
    )
    position = pl.col("pos_abb").fill_null("").str.to_uppercase()
    return (
        depth_charts.filter(
            pl.col("gsis_id").is_not_null()
            & (pl.col("gsis_id") != "")
            & position.is_in(["QB", "RB", "FB", "HB", "WR", "LWR", "RWR", "SWR", "TE"])
        )
        .sort(["dt", "pos_rank"], descending=[True, False])
        .unique(subset=["gsis_id"], keep="first", maintain_order=True)
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("team").alias("current_team"),
            pl.when(position.is_in(["RB", "FB", "HB"]))
            .then(pl.lit("RB"))
            .when(position.str.contains("WR"))
            .then(pl.lit("WR"))
            .otherwise(position)
            .alias("depth_chart_position_group"),
            pl.col("pos_rank").cast(pl.Int32, strict=False).alias("depth_chart_rank"),
            pl.col("pos_name").alias("depth_chart_position"),
            pl.col("dt").alias("depth_chart_date"),
        )
    )


def enrich_player_seasons(
    player_seasons: pl.DataFrame,
    usage: pl.DataFrame,
    team_volume: pl.DataFrame,
    injury_history: pl.DataFrame,
) -> pl.DataFrame:
    """Attach optional v2-only nflverse evidence to the historical season rows."""
    frame = (
        player_seasons.join(usage, on=["player_id", "season"], how="left")
        .join(team_volume, on=["season", "team"], how="left")
        .join(injury_history, on=["player_id", "season"], how="left")
    )
    count_columns = [
        "active_games",
        "active_game_dropbacks",
        "route_opportunities",
        "red_zone_carries",
        "goal_line_carries",
        "goal_line_rushing_tds",
        "designed_carries",
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
    ]
    return frame.with_columns(pl.col(column).fill_null(0) for column in count_columns)
