"""nflverse-first projection inputs beyond the weekly fantasy stat line.

The participation feed identifies the offensive players present on each play, but it
does not chart a route for every eligible receiver.  We therefore call an eligible
player's presence on a dropback a ``route opportunity``.  It is a reproducible,
useful proxy for routes run, and is deliberately exported under that honest name.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

import polars as pl

from engine.scoring.columns import require_columns

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


def build_player_efficiency(player_weeks: pl.DataFrame) -> pl.DataFrame:
    """Aggregate stable rate-stat inputs that the fantasy box score omits.

    EPA and first downs are summed before division. CPOE is attempt-weighted instead
    of averaging weekly averages, and every denominator is retained implicitly by the
    ordinary season columns already on the player row.
    """
    columns = (
        "player_id",
        "season",
        "attempts",
        "passing_epa",
        "passing_cpoe",
        "passing_first_downs",
        "carries",
        "rushing_epa",
        "rushing_first_downs",
        "targets",
        "receiving_epa",
        "receiving_first_downs",
    )
    require_columns(player_weeks.columns, columns, "player efficiency input")
    attempts = pl.col("attempts").fill_null(0).sum()
    carries = pl.col("carries").fill_null(0).sum()
    targets = pl.col("targets").fill_null(0).sum()
    return (
        player_weeks.filter(pl.col("player_id").is_not_null())
        .group_by(["player_id", "season"])
        .agg(
            pl.when(attempts > 0)
            .then(pl.col("passing_epa").fill_null(0).sum() / attempts)
            .otherwise(None)
            .alias("passing_epa_per_attempt"),
            pl.when(attempts > 0)
            .then(
                (pl.col("passing_cpoe").fill_null(0) * pl.col("attempts").fill_null(0)).sum()
                / attempts
            )
            .otherwise(None)
            .alias("passing_cpoe"),
            pl.when(attempts > 0)
            .then(pl.col("passing_first_downs").fill_null(0).sum() / attempts)
            .otherwise(None)
            .alias("passing_first_down_rate"),
            pl.when(carries > 0)
            .then(pl.col("rushing_first_downs").fill_null(0).sum() / carries)
            .otherwise(None)
            .alias("rushing_first_down_rate"),
            pl.when(targets > 0)
            .then(pl.col("receiving_first_downs").fill_null(0).sum() / targets)
            .otherwise(None)
            .alias("receiving_first_down_rate"),
        )
        .sort(["player_id", "season"])
    )


def build_team_tendencies(pbp: pl.DataFrame) -> pl.DataFrame:
    """Situation-neutral offensive volume and pass tendency by team-season.

    Neutral means quarters 1–3 with the score within eight points. Kneels and
    non-scrimmage plays are removed. This avoids treating fourth-quarter comeback or
    clock-killing volume as the offense's preferred baseline behavior.
    """
    required = (
        "game_id",
        "season",
        "season_type",
        "posteam",
        "qtr",
        "down",
        "score_differential",
        "no_huddle",
        "qb_dropback",
        "rush_attempt",
        "qb_kneel",
        "epa",
        "pass_oe",
    )
    require_columns(pbp.columns, required, "team tendency play-by-play input")
    neutral = pbp.filter(
        (pl.col("season_type") == "REG")
        & pl.col("posteam").is_not_null()
        & pl.col("qtr").is_between(1, 3)
        & pl.col("down").is_between(1, 4)
        & (pl.col("score_differential").abs() <= 8)
        & ((pl.col("qb_dropback") == 1) | (pl.col("rush_attempt") == 1))
        & (pl.col("qb_kneel").fill_null(0) != 1)
    )
    return (
        neutral.group_by(["season", pl.col("posteam").alias("team")])
        .agg(
            pl.len().alias("neutral_plays"),
            pl.col("game_id").n_unique().alias("neutral_games"),
            pl.col("qb_dropback").fill_null(0).mean().alias("neutral_pass_rate"),
            pl.col("epa").mean().alias("neutral_epa_per_play"),
            pl.col("pass_oe").mean().alias("neutral_pass_oe"),
        )
        .with_columns((pl.col("neutral_plays") / pl.col("neutral_games")).alias("neutral_plays_pg"))
        .sort(["season", "team"])
    )


def build_expected_opportunity(
    opportunity: pl.DataFrame,
    player_weeks: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate ffopportunity xFP over official regular-season game ids."""
    required = (
        "season",
        "game_id",
        "player_id",
        "total_fantasy_points_exp",
        "total_yards_gained_exp",
        "total_touchdown_exp",
        "total_first_down_exp",
    )
    require_columns(opportunity.columns, required, "expected opportunity input")
    require_columns(player_weeks.columns, ("season", "game_id"), "regular player-week input")
    regular_games = player_weeks.select("season", "game_id").drop_nulls().unique()
    rows = (
        opportunity.filter(pl.col("player_id").is_not_null())
        .with_columns(pl.col("season").cast(pl.Int32))
        .join(
            regular_games.with_columns(pl.col("season").cast(pl.Int32)),
            on=["season", "game_id"],
            how="semi",
        )
    )
    return (
        rows.group_by(["player_id", "season"])
        .agg(
            pl.col("game_id").n_unique().alias("xfp_games"),
            pl.col("total_fantasy_points_exp").fill_null(0).sum().alias("xfp_total"),
            pl.col("total_first_down_exp").fill_null(0).sum().alias("expected_first_downs_total"),
        )
        .with_columns(
            (pl.col("xfp_total") / pl.col("xfp_games")).alias("xfp_pg"),
            (pl.col("expected_first_downs_total") / pl.col("xfp_games")).alias(
                "expected_first_downs_pg"
            ),
        )
        .sort(["player_id", "season"])
    )


def build_nextgen_features(
    passing: pl.DataFrame,
    receiving: pl.DataFrame,
    rushing: pl.DataFrame,
) -> pl.DataFrame:
    """Canonical player-season NGS residual and process metrics."""

    def season_rows(frame: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
        require_columns(
            frame.columns,
            ("season", "season_type", "week", "player_gsis_id", *columns),
            "Next Gen Stats input",
        )
        return frame.filter((pl.col("season_type") == "REG") & (pl.col("week") == 0)).select(
            pl.col("player_gsis_id").alias("player_id"), "season", *columns
        )

    pass_rows = season_rows(
        passing,
        ("completion_percentage_above_expectation",),
    ).rename({"completion_percentage_above_expectation": "ngs_cpoe"})
    rec_rows = season_rows(
        receiving,
        ("avg_separation", "avg_yac_above_expectation"),
    ).rename(
        {
            "avg_separation": "ngs_separation",
            "avg_yac_above_expectation": "ngs_yac_oe",
        }
    )
    rush_rows = season_rows(
        rushing,
        ("rush_yards_over_expected_per_att",),
    ).rename({"rush_yards_over_expected_per_att": "ngs_ryoe_per_att"})
    return (
        pass_rows.join(rec_rows, on=["player_id", "season"], how="full", coalesce=True)
        .join(rush_rows, on=["player_id", "season"], how="full", coalesce=True)
        .sort(["player_id", "season"])
    )


def build_market_rankings(
    rankings: pl.DataFrame,
    id_crosswalk: pl.DataFrame,
    cutoff: str = "08-31",
) -> pl.DataFrame:
    """Latest pre-cutoff FantasyPros price snapshots per season and GSIS id.

    Positional ECR remains the apples-to-apples baseline for position boards.  The
    overall redraft page is retained separately because it is the dated market-price
    ordering needed to measure two-round disagreements across positions.  ECR is not
    called ADP here: it is an expert-consensus price proxy, not an observed draft.
    """
    require_columns(
        rankings.columns,
        ("page_type", "id", "pos", "ecr", "sd", "scrape_date"),
        "FantasyPros rankings input",
    )
    require_columns(
        id_crosswalk.columns,
        ("fantasypros_id", "gsis_id"),
        "fantasy ID crosswalk",
    )
    month, day = (int(part) for part in cutoff.split("-", maxsplit=1))
    dated = rankings.with_columns(
        pl.col("scrape_date").cast(pl.String).str.to_date(strict=False).alias("_date"),
        pl.col("id").cast(pl.Int64, strict=False).alias("_fantasypros_id"),
    ).with_columns(pl.col("_date").dt.year().alias("forecast_season"))
    positional = dated.filter(
        pl.col("page_type").is_in(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te"])
        & pl.col("ecr").is_not_null()
        & (pl.col("_date") <= pl.date(pl.col("forecast_season"), month, day))
    )
    overall = dated.filter(
        (pl.col("page_type") == "redraft-overall")
        & pl.col("ecr").is_not_null()
        & (pl.col("_date") <= pl.date(pl.col("forecast_season"), month, day))
    )
    crosswalk = id_crosswalk.select(
        pl.col("fantasypros_id").cast(pl.Int64, strict=False).alias("_fantasypros_id"),
        pl.col("gsis_id").alias("player_id"),
    ).drop_nulls()

    # Latest snapshot per page, not per season: archive scrapes cover every page the
    # same day, but the Wayback backfill can carry different capture dates per page,
    # and a season-wide max would silently drop every page but the newest.
    positional_latest = positional.group_by(["forecast_season", "page_type"]).agg(
        pl.col("_date").max().alias("_date")
    )
    positional_rows = (
        positional.join(
            positional_latest, on=["forecast_season", "page_type", "_date"], how="inner"
        )
        .join(crosswalk, on="_fantasypros_id", how="inner")
        .select(
            "forecast_season",
            "player_id",
            pl.col("pos").alias("market_position"),
            pl.col("ecr").alias("market_ecr"),
            (-pl.col("ecr")).alias("market_ecr_score"),
            pl.col("sd").alias("market_ecr_sd"),
            pl.col("_date").cast(pl.String).alias("market_snapshot"),
        )
        .unique(subset=["forecast_season", "player_id"], keep="first")
    )
    if overall.height == 0:
        return positional_rows.sort(["forecast_season", "market_position", "market_ecr"])

    overall_latest = overall.group_by("forecast_season").agg(
        pl.col("_date").max().alias("_date")
    )
    overall_rows = (
        overall.join(overall_latest, on=["forecast_season", "_date"], how="inner")
        .join(crosswalk, on="_fantasypros_id", how="inner")
        .select(
            "forecast_season",
            "player_id",
            pl.col("ecr").alias("market_overall_ecr"),
            (-pl.col("ecr")).alias("market_overall_ecr_score"),
            pl.col("sd").alias("market_overall_ecr_sd"),
            pl.col("_date").cast(pl.String).alias("market_overall_snapshot"),
        )
        .unique(subset=["forecast_season", "player_id"], keep="first")
    )
    return (
        positional_rows.join(
            overall_rows,
            on=["forecast_season", "player_id"],
            how="full",
            coalesce=True,
        )
        .with_columns(pl.lit("fantasypros_ecr").alias("market_price_source"))
        .sort(["forecast_season", "market_overall_ecr", "market_position", "market_ecr"])
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

    position_by_player_season: dict[tuple[str, int], str] = {}
    if "position" in player_weeks.columns:
        for row in (
            player_weeks.select("player_id", "season", "position")
            .filter(pl.col("position").is_not_null())
            .unique(subset=["player_id", "season"], keep="last")
            .iter_rows(named=True)
        ):
            position_by_player_season[(str(row["player_id"]), int(row["season"]))] = str(
                row["position"]
            ).upper()

    valid_plays: dict[tuple[str, float], dict[str, Any]] = {}
    dropbacks_by_game_team: dict[tuple[str, str], int] = defaultdict(int)
    high_value: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {
            "red_zone_carries": 0.0,
            "goal_line_carries": 0.0,
            "goal_line_rushing_tds": 0.0,
            "designed_carries": 0.0,
            "scramble_carries": 0.0,
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
            else:
                usage["scramble_carries"] += 1

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
        for index, player_id in enumerate(ids):
            supplied_position = positions[index] if index < len(positions) else ""
            normalized_position = supplied_position.upper() or position_by_player_season.get(
                (player_id, season), ""
            )
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
                "route_opportunities": route_opp,
                "route_participation": route_opp / active_dropbacks if active_dropbacks else None,
                "targets_per_route_opportunity": targets / route_opp if route_opp else None,
                **valuable,
            }
        )

    return pl.DataFrame(rows, infer_schema_length=None)


def validate_participation_usage(usage: pl.DataFrame, seasons: list[int]) -> None:
    """Fail when a requested participation season silently produced no player usage."""
    missing: list[int] = []
    for season in seasons:
        season_rows = usage.filter(pl.col("season") == season)
        if season_rows.height == 0 or float(season_rows["route_opportunities"].sum() or 0) <= 0:
            missing.append(season)
    if missing:
        listing = ", ".join(map(str, missing))
        raise ValueError(
            f"participation produced no route opportunities for season(s): {listing}. "
            "Check player-position fallback and the nflverse participation schema."
        )


def normalize_ppg_for_active_games(player_seasons: pl.DataFrame) -> pl.DataFrame:
    """Use participation-observed active games as the v2 PPG denominator when present.

    nflverse weekly stats omit dressed players who record no box-score event. ``games``
    therefore remains the number of production rows for v1 compatibility, while v2's
    PPG denominator includes every game observed in offensive participation.
    """
    denominator = pl.max_horizontal(
        pl.col("games").cast(pl.Float64), pl.col("active_games").cast(pl.Float64)
    )
    return player_seasons.with_columns(denominator.alias("ppg_denominator_games")).with_columns(
        pl.when(pl.col("ppg_denominator_games") > 0)
        .then(pl.col("season_pts") / pl.col("ppg_denominator_games"))
        .otherwise(pl.col("ppg"))
        .alias("ppg")
    )


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
    # Older nflverse injury partitions have encoded season as Float64. Normalize the
    # join key at the source boundary so multi-era histories join player stats safely.
    valid = injuries.filter(
        pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != "")
    ).with_columns(pl.col("season").cast(pl.Int32, strict=True))
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


def depth_chart_as_of(
    depth_charts: pl.DataFrame | None,
    season: int,
    cutoff: str,
) -> pl.DataFrame | None:
    """The latest depth entry per player published on or before ``cutoff`` of ``season``.

    This is the single selection used by both the rolling backtest (per forecast
    season) and the live board (for the draft season), so the live board is scored on
    exactly the depth-chart evidence the pending backtest fold was scored on.
    ``cutoff`` is ``MM-DD``.

    Only entries published after September 1 of the previous season count.  A player
    whose last chart entry is years old is not on anyone's current depth chart, and
    letting him through would put him back on a team's backfield or target tree when
    teammate availability is computed.  (Legacy week-one proxies are dated August 31
    of their own season, so the previous season's proxy is excluded too.)
    """
    if depth_charts is None or depth_charts.height == 0:
        return None
    month, day = (int(part) for part in cutoff.split("-", maxsplit=1))
    limit = date(season, month, day)
    floor = date(season - 1, 9, 1)
    date_expression = (
        pl.col("dt").str.slice(0, 10).str.to_date(strict=False)
        if depth_charts.schema["dt"] == pl.String
        else pl.col("dt").cast(pl.Date, strict=False)
    )
    dated = depth_charts.with_columns(date_expression.alias("_date"))
    eligible = dated.filter(
        pl.col("_date").is_not_null() & (pl.col("_date") <= limit) & (pl.col("_date") > floor)
    ).drop("_date")
    return current_depth_chart(eligible) if eligible.height else None


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
    player_efficiency: pl.DataFrame | None = None,
    expected_opportunity: pl.DataFrame | None = None,
    nextgen_features: pl.DataFrame | None = None,
    team_tendencies: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach optional v2-only nflverse evidence to the historical season rows."""
    frame = (
        player_seasons.join(usage, on=["player_id", "season"], how="left")
        .join(team_volume, on=["season", "team"], how="left")
        .join(injury_history, on=["player_id", "season"], how="left")
    )
    for research in (player_efficiency, expected_opportunity, nextgen_features):
        if research is not None and research.height:
            frame = frame.join(research, on=["player_id", "season"], how="left")
    if team_tendencies is not None and team_tendencies.height:
        frame = frame.join(team_tendencies, on=["season", "team"], how="left")
    count_columns = [
        "active_games",
        "route_opportunities",
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
        "xfp_games",
        "neutral_plays",
        "neutral_games",
    ]
    return frame.with_columns(
        pl.col(column).fill_null(0) for column in count_columns if column in frame.columns
    )
