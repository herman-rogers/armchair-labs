"""Historical same-NFL-team starter connections; never inputs to forecasts.

Identity and shared-game selection use the pinned DuckDB table catalog. Historical
pairs retain their own samples and are never combined into a roster covariance
matrix. Current-season attribution uses only reconciled, completed team scores.
"""

from itertools import combinations

import numpy as np

from engine.metrics.team_correlations import correlation, covariance, risk_intervals
from engine.metrics.team_strength import RESERVE

POSITIONS = {"QB", "WR", "TE"}
ALIASES = {"LAR": "LA", "STL": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC"}


def historical_summary(rows):
    """Center within season, on exactly the same games for both players."""
    n = len(rows)
    seasons = np.array([r[0] for r in rows])
    values = np.array([[r[1], r[2]] for r in rows], dtype=float).reshape(-1, 2)
    cov, df = covariance(values, seasons, "season")
    result = dict(
        n=n,
        seasons=sorted(set(seasons.tolist())),
        status="insufficient_overlap",
        r=None,
        interval=None,
        variance=None,
        independent_variance=None,
        covariance_effect=None,
        variance_share_pct=None,
        variance_change_pct=None,
        sd=None,
        independent_sd=None,
        variance_change_interval=None,
    )
    if cov is None:
        return result
    independent = float(np.trace(cov))
    total = max(0.0, float(cov.sum()))
    effect = float(2 * cov[0, 1])
    r = correlation(cov)
    interval = None
    if r is not None and df > 2 and abs(r) < 1:
        interval = np.tanh(np.arctanh(r) + np.array([-1, 1]) * 1.96 / np.sqrt(df - 2)).tolist()
    result.update(
        status="available",
        r=r,
        interval=interval,
        variance=total,
        independent_variance=independent,
        covariance_effect=effect,
        variance_share_pct=100 * effect / total if total > 1e-12 else None,
        variance_change_pct=100 * effect / independent if independent > 1e-12 else None,
        sd=total**0.5,
        independent_sd=independent**0.5,
        variance_change_interval=risk_intervals(values, seasons, "season", df)[
            "variance_change_interval"
        ],
    )
    return result


def observed_pair(history, a, b):
    rows = []
    for week in history:
        contributions = week.get("contributions")
        players = {p["espn_id"]: p["points"] for p in contributions or []}
        if a in players and b in players:
            rows.append([players[a], players[b], week["score"]])
    result = dict(
        n=len(rows),
        status="insufficient_overlap",
        r=None,
        covariance_effect=None,
        team_variance_share_pct=None,
    )
    # A subset's covariance cannot explain the full-season team's variance.
    if len(rows) != len(history):
        result["status"] = "incomplete_lineup_history"
        return result
    if len(rows) < 3:
        return result
    cov = np.cov(np.array(rows), rowvar=False, ddof=1)
    effect = float(2 * cov[0, 1])
    result.update(
        status="available",
        r=correlation(cov[:2, :2]),
        covariance_effect=effect,
        team_variance_share_pct=100 * effect / cov[2, 2] if cov[2, 2] > 1e-12 else None,
    )
    return result


def attach_roster_correlations(result, snapshot, db, report):
    """Attach per-fantasy-team diagnostics without changing any existing statistic."""
    end = min(snapshot.season, report["season"]) - 1
    start = max(end - 4, report["earliest_season"])
    # Ambiguous/missing identities remain unavailable; never fuzzy-match names.
    identities = dict(
        db.execute("""
        SELECT try_cast(espn_id AS BIGINT), min(gsis_id)
        FROM analytics.players WHERE try_cast(espn_id AS BIGINT) IS NOT NULL
        GROUP BY 1 HAVING count(DISTINCT gsis_id) = 1
    """).fetchall()
    )
    cache = {}
    for team in result["teams"]:
        starters = [
            p
            for p in snapshot.players
            if p.owner_team_id == team["team_id"]
            and p.lineup_slot not in RESERVE
            and p.position in POSITIONS
        ]
        connections = []
        for a, b in combinations(starters, 2):
            club = ALIASES.get(a.espn_team, a.espn_team)
            if not club or club != ALIASES.get(b.espn_team, b.espn_team):
                continue
            if a.position == b.position == "QB":
                continue
            aid, bid = identities.get(a.espn_id), identities.get(b.espn_id)
            key = (club, aid, bid, start, end)
            if key not in cache:
                rows = (
                    db.execute(
                        """
                    SELECT a.season, a.league_points, b.league_points
                    FROM analytics.team_player_games a
                    JOIN analytics.team_player_games b
                      ON a.game_id = b.game_id AND a.team = b.team
                    WHERE a.player_id = ? AND b.player_id = ? AND a.team = ?
                      AND a.season BETWEEN ? AND ? AND a.starter_game AND b.starter_game
                    ORDER BY a.season, a.week
                """,
                        [aid, bid, club, start, end],
                    ).fetchall()
                    if aid and bid
                    else []
                )
                historical = historical_summary(rows)
                historical["by_season"] = [
                    dict(season=year, **historical_summary([r for r in rows if r[0] == year]))
                    for year in sorted({r[0] for r in rows})
                ]
                if not aid or not bid:
                    historical["status"] = "unmapped_identity"
                cache[key] = historical
            connections.append(
                dict(
                    id=f"{a.espn_id}-{b.espn_id}",
                    nfl_team=club,
                    a=dict(espn_id=a.espn_id, name=a.player_display_name, position=a.position),
                    b=dict(espn_id=b.espn_id, name=b.player_display_name, position=b.position),
                    historical=cache[key],
                    observed=observed_pair(team["history"], a.espn_id, b.espn_id),
                )
            )
        complete = all(p["observed"]["status"] == "available" for p in connections)
        effect = sum(p["observed"]["covariance_effect"] for p in connections) if complete else None
        sd = team["results"]["volatility"]
        team["correlation_risk"] = dict(
            status="available",
            history_start=start,
            history_end=end,
            source=report,
            connections=connections,
            unmapped_starters=sum(p.espn_id not in identities for p in starters),
            unknown_team_starters=sum(not p.espn_team for p in starters),
            observed=dict(
                n=len(team["history"]),
                covariance_effect=effect,
                team_variance_share_pct=100 * effect / sd**2
                if effect is not None and sd is not None and sd > 0
                else None,
                covered_connections=sum(
                    p["observed"]["status"] == "available" for p in connections
                ),
            ),
        )
    return result
