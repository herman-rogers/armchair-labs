"""SQL over the observation tables that the API joins with ESPN and campaigns.

Every function takes an open DuckDB connection on the pinned catalog. Nothing here
reads release artifacts or model outputs.
"""

from __future__ import annotations

import polars as pl

SKILL = ("QB", "RB", "WR", "TE")


def observation_cutoff(db) -> dict:
    """The latest captured regular-season week in the current observations."""
    season, week = db.execute(
        "SELECT max(season), max(week) FILTER (WHERE season = (SELECT max(season) "
        "FROM analytics.current_weeks)) FROM analytics.current_weeks"
    ).fetchone()
    return {"season": season, "through_week": week}


def scoring_weeks(db, season: int) -> list[dict]:
    """Current capture wins over historical overlap; preserve unknown stats as null.

    This SQL projection refreshes with the existing weekly observation tables. It
    deliberately does not mix model outputs into the observation catalog.
    """
    return db.execute(
        """
        WITH source AS (
            SELECT player_id, player_display_name, position, team, week, league_points,
                   passing_yards, passing_tds, passing_interceptions, rushing_yards,
                   rushing_tds, receptions, receiving_yards, receiving_tds, 0 AS priority
            FROM analytics.current_weeks WHERE season = ? AND season_type = 'REG'
            UNION ALL
            SELECT player_id, player_display_name, position, team, week, league_points,
                   passing_yards, passing_tds, passing_interceptions, rushing_yards,
                   rushing_tds, receptions, receiving_yards, receiving_tds, 1 AS priority
            FROM analytics.nfl_player_weeks WHERE season = ? AND season_type = 'REG'
        ), deduped AS (
            SELECT * FROM source QUALIFY row_number() OVER (
                PARTITION BY player_id, week ORDER BY priority
            ) = 1
        )
        SELECT player_id, player_display_name, position, team, week, league_points,
            CAST(CASE WHEN position = 'QB' THEN
                .04 * passing_yards + 4 * passing_tds - 2 * passing_interceptions
                + .1 * rushing_yards + 6 * rushing_tds
            WHEN position IN ('RB', 'WR', 'TE') THEN
                .1 * (rushing_yards + receiving_yards) + receptions
                + 6 * (rushing_tds + receiving_tds)
            ELSE NULL END AS DOUBLE) AS model_points
        FROM deduped WHERE position IN ('QB', 'RB', 'WR', 'TE')
        ORDER BY week, player_id
        """, [season, season],
    ).pl().to_dicts()


def candidates(db, season: int) -> pl.DataFrame:
    """Latest known name, position and team for every player seen since ``season``."""
    return db.execute(
        """
        WITH observed AS (
            SELECT player_id, player_display_name, position, team, season, week
            FROM analytics.current_weeks
            UNION ALL
            SELECT player_id, player_display_name, position, team, season, week
            FROM analytics.nfl_player_weeks WHERE season >= ?
            UNION ALL
            SELECT player_id, player_display_name, position, team, season, 0 AS week
            FROM analytics.current_players
        ),
        latest AS (
            SELECT *, row_number() OVER (
                PARTITION BY player_id ORDER BY season DESC, week DESC
            ) AS rn
            FROM observed WHERE player_id IS NOT NULL
        )
        SELECT player_id, player_display_name, position, team
        FROM latest WHERE rn = 1
        """,
        [season - 1],
    ).pl()


def espn_ids(db) -> dict[int, str]:
    """ESPN id → gsis id from the identity table plus current-season rookies."""
    rows = db.execute(
        """
        SELECT espn_id, gsis_id FROM analytics.players
        WHERE espn_id IS NOT NULL AND gsis_id IS NOT NULL
        UNION
        SELECT CAST(espn_id AS VARCHAR), player_id FROM analytics.current_players
        WHERE espn_id IS NOT NULL
        """
    ).fetchall()
    result: dict[int, str] = {}
    for espn_id, gsis_id in rows:
        try:
            result[int(float(espn_id))] = gsis_id
        except (TypeError, ValueError):
            continue
    return result


def schedule(db, season: int) -> list[dict]:
    return (
        db.execute(
            "SELECT game_id, season, week, game_type, gameday, home_team, away_team "
            "FROM analytics.current_schedule WHERE season = ? "
            "UNION SELECT game_id, season, week, game_type, gameday, home_team, away_team "
            "FROM analytics.nfl_schedule WHERE season = ? ORDER BY week, gameday, game_id",
            [season, season],
        )
        .pl()
        .to_dicts()
    )


def directory(db, *, search: str = "", position: str = "ALL", limit: int = 5000) -> pl.DataFrame:
    frame = candidates(db, observation_cutoff(db)["season"] or 2000)
    frame = frame.filter(pl.col("position").is_in(SKILL))
    if position != "ALL":
        frame = frame.filter(pl.col("position") == position)
    if search:
        frame = frame.filter(
            pl.col("player_display_name")
            .str.to_lowercase()
            .str.contains(search.lower(), literal=True)
        )
    return frame.sort("player_display_name").head(limit)


def identity(db, player_id: str) -> dict | None:
    row = db.execute(
        "SELECT gsis_id AS player_id, display_name, birth_date, espn_id, pfr_id, rookie_season, "
        "draft_year, draft_round, draft_pick, draft_team FROM analytics.players WHERE gsis_id = ?",
        [player_id],
    ).pl()
    if row.height == 0:
        latest = candidates(db, 2000).filter(pl.col("player_id") == player_id)
        if latest.height == 0:
            return None
        return {"player_id": player_id, "display_name": latest["player_display_name"][0]}
    return row.to_dicts()[0]


def seasons(db, player_id: str) -> list[dict]:
    return (
        db.execute(
            "SELECT * FROM analytics.nfl_player_seasons WHERE player_id = ? ORDER BY season",
            [player_id],
        )
        .pl()
        .to_dicts()
    )


def weeks(db, player_id: str, season: int | None = None) -> list[dict]:
    columns = (
        "player_id, season, week, team, opponent_team, position, "
        "completions, attempts, passing_yards, passing_tds, passing_interceptions, "
        "carries, rushing_yards, rushing_tds, targets, receptions, receiving_yards, "
        "receiving_tds, league_points"
    )
    sql = (
        f"SELECT {columns} FROM analytics.nfl_player_weeks WHERE player_id = ? "
        f"UNION ALL SELECT {columns} FROM analytics.current_weeks WHERE player_id = ?"
    )
    params: list = [player_id, player_id]
    if season is not None:
        sql = f"SELECT * FROM ({sql}) WHERE season = ?"
        params.append(season)
    return db.execute(sql + " ORDER BY season, week", params).pl().to_dicts()


def rookies(db, season: int) -> pl.DataFrame:
    """This season's rookies with draft capital and linked college careers."""
    return db.execute(
        """
        SELECT c.player_id, c.player_display_name, c.position, c.team, c.draft_pick,
               p.draft_round, p.draft_team, p.birth_date,
               l.college_id, l.college_name, l.first_college_season, l.last_college_season,
               l.status AS college_link_status
        FROM analytics.current_players c
        LEFT JOIN analytics.players p ON p.gsis_id = c.player_id
        LEFT JOIN analytics.college_identity_links l
               ON l.player_id = c.player_id AND l.status = 'linked'
        WHERE c.season = ?
        ORDER BY coalesce(c.draft_pick, 10000), c.player_display_name
        """,
        [season],
    ).pl()


def college_links(db, player_id: str) -> list[dict]:
    return (
        db.execute(
            "SELECT college_id, college_name, first_college_season, last_college_season, status, "
            "method, evidence FROM analytics.college_identity_links WHERE player_id = ? "
            "ORDER BY status, college_id",
            [player_id],
        )
        .pl()
        .to_dicts()
    )


def college_seasons(db, college_id: str) -> list[dict]:
    return (
        db.execute(
            "SELECT * FROM analytics.college_annual WHERE college_id = ? ORDER BY season",
            [college_id],
        )
        .pl()
        .to_dicts()
    )


def market_consensus(db, positions: tuple[str, ...] = SKILL) -> dict[str, dict]:
    """Latest FantasyPros overall and positional ECR per NFL identity."""
    if not db.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'analytics' AND table_name = 'market_rankings'"
    ).fetchone()[0]:
        return {}
    rows = db.execute(
        """
        WITH latest AS (
            SELECT page_type, max(scrape_date) AS scrape_date FROM analytics.market_rankings
            WHERE page_type LIKE 'redraft-%' GROUP BY page_type
        )
        SELECT m.player_id, m.page_type, m.scrape_date, m.ecr, m.sd, m.position
        FROM analytics.market_rankings m JOIN latest USING (page_type, scrape_date)
        WHERE m.player_id IS NOT NULL
        """
    ).fetchall()
    result: dict[str, dict] = {}
    for player_id, page, scrape_date, ecr, sd, position in rows:
        entry = result.setdefault(player_id, {})
        if page == "redraft-overall":
            entry.update(ecr_overall=ecr, ecr_overall_sd=sd, ecr_overall_date=str(scrape_date))
        elif page == f"redraft-{str(position).lower()}" and position in positions:
            entry.update(ecr_position=ecr, ecr_position_sd=sd, ecr_position_date=str(scrape_date))
    return result
