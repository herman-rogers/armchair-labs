"""Weekly lineup forecasts from a campaign's horizon forecast, spread per scheduled game."""

from collections import defaultdict
from math import isfinite

from engine.espn.observations import weekly_matchups
from engine.names import normalize


def team_code(value):
    return {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV"}.get(value, value)


def number(value):
    return isinstance(value, (int, float)) and isfinite(value)


def weekly_forecast(snapshot, rankings, identities, schedule):
    """Assign each ESPN starter a per-game point estimate from campaign rankings.

    ``rankings`` is a ChairStats ranking response: rows with ``player_id``,
    ``position``, ``team``, ``points_per_game`` and ``season``. K and D/ST use up to
    four prior recorded ESPN scores because no campaign forecasts them.
    """
    week = snapshot.week
    rows = rankings["rankings"]
    seasons = {r["season"] for r in rows}
    aligned = bool(rows) and seasons == {snapshot.season}
    ranks = {r["player_id"]: r for r in rows}
    names = defaultdict(list)
    for row in rows:
        names[normalize(row["player_display_name"] or ""), row["position"]].append(row)
    games = defaultdict(set)
    for row in schedule:
        if row["season"] == snapshot.season and row["game_type"] == "REG":
            for team in (row["home_team"], row["away_team"]):
                games[team_code(team)].add(row["week"])
    earlier = defaultdict(dict)
    for game in snapshot.week_lineups:
        if game.week < week:
            for player in game.home_lineup + game.away_lineup:
                if player.position in ("K", "DST", "D/ST") and number(player.points):
                    earlier[player.espn_id][game.week] = player.points

    def predict(player):
        value, reason, source = None, None, None
        if not aligned:
            reason = "The campaign has no forecast for the current season."
        elif player["on_bye"]:
            value, source = 0.0, "Scheduled bye"
        elif player["position"] in ("K", "DST", "D/ST"):
            scores = [score for _, score in sorted(earlier[player["espn_id"]].items())[-4:]]
            if scores:
                value, source = (
                    sum(scores) / len(scores),
                    f"Prior {len(scores)} recorded games · K/DST reference",
                )
            else:
                reason = "No earlier recorded K/DST scores."
        else:
            row = ranks.get(identities.get(player["espn_id"]))
            if row is None:
                matches = names[normalize(player["player_display_name"]), player["position"]]
                row = matches[0] if len(matches) == 1 else None
            if row is None or row["position"] != player["position"]:
                reason = "No campaign forecast matched this player."
            elif not games.get(team_code(row["team"])):
                reason = "NFL schedule unavailable."
            elif week not in games[team_code(row["team"])]:
                value, source = 0.0, "Scheduled bye"
            elif number(row["points_per_game"]):
                value = row["points_per_game"]
                source = (
                    f"{rankings['campaign_name']} {row['horizon']} points per scheduled game "
                    f"(as of week {row['origin_week']})"
                )
            else:
                reason = "No usable point forecast."
        return dict(
            espn_id=player["espn_id"],
            player_display_name=player["player_display_name"],
            position=player["position"],
            slot=player["slot"],
            started=player["started"],
            prediction=value,
            basis=source,
            unavailable_reason=reason,
        )

    result = weekly_matchups(snapshot, week)
    result["source"] = "campaign_per_game"
    for game in result["matchups"]:
        for side in (game["home"], game["away"]):
            side["forecasts"] = [predict(p) for p in side["lineup"]]
            starters = [p for p in side["forecasts"] if p["started"]]
            covered = [p for p in starters if p["prediction"] is not None]
            # Empty starting slots contribute zero; a missing lineup is not zero.
            side["covered_starters"] = len(covered)
            side["starter_count"] = len(starters)
            side["model_projection"] = (
                sum(p["prediction"] for p in covered)
                if starters and len(covered) == len(starters)
                else None
            )
            side.pop("lineup", None)
        home, away = game["home"]["model_projection"], game["away"]["model_projection"]
        game["margin"] = home - away if home is not None and away is not None else None
        game["favorite_team_id"] = (
            None
            if game["margin"] in (None, 0)
            else game["home"]["team_id"]
            if game["margin"] > 0
            else game["away"]["team_id"]
        )
    dates = [
        str(r["gameday"])[:10]
        for r in schedule
        if r["season"] == snapshot.season
        and r["week"] == week
        and r["game_type"] == "REG"
        and r.get("gameday")
    ]
    origins = sorted({(r["season"], r["origin_week"]) for r in rows})
    return dict(
        **result,
        campaign_id=rankings["campaign_id"],
        campaign_name=rankings["campaign_name"],
        manifest_sha256=rankings["manifest_sha256"],
        horizon=rankings["horizon"],
        forecast_origins=[{"season": s, "origin_week": w} for s, w in origins],
        pregame_deadline=f"{min(dates)}T00:00:00+00:00" if dates else None,
        limitations=[
            "Horizon points allocated evenly across scheduled games; "
            "no weekly opponent adjustment or calibrated win probability.",
            "K/DST use up to four prior recorded scores; "
            "no ESPN projections enter any point estimate.",
            "Starting lineups are observed ESPN selections. "
            "Missing starter forecasts withhold team totals and winner picks.",
            *rankings.get("limitations", []),
        ],
    )
