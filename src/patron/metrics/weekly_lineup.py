"""Dedicated weekly points, explicit reference scopes, and no provider fallback."""

from collections import defaultdict
from math import isfinite

from patron.espn.observations import weekly_matchups
from patron.metrics.current_rankings import team_code
from patron.names import normalize

RECIPE = "nextgen-weekly-reference-v1"


def number(value):
    return isinstance(value, (int, float)) and isfinite(value)


def weekly_forecast(snapshot, rankings, identities, schedule, weekly=None):
    week = snapshot.week
    report = rankings["report"]
    aligned = (
        rankings["horizon"] == "next4"
        and report["season"] == snapshot.season
        and report["through_week"] == week - 1
    )
    ranks = {r["player_id"]: r for r in rankings["rankings"]}
    learned = (
        {
            r["player_id"]: r
            for r in weekly["predictions"]
            if r["season"] == snapshot.season
            and r["target_week"] == week
            and r["through_week"] == week - 1
        }
        if weekly
        else {}
    )
    if weekly:
        aligned = (
            aligned
            and weekly["manifest"]["season"] == snapshot.season
            and weekly["manifest"]["through_week"] == report["through_week"]
            and weekly["manifest"]["analysis"]["version"] == rankings["version"]
        )
    names = defaultdict(list)
    for row in (learned if weekly else ranks).values():
        names[normalize(row["player_display_name"]), row["position"]].append(row)
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
            reason = "A NextGen release through the previous week is required."
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
            row = (learned if weekly else ranks).get(identities.get(player["espn_id"]))
            if row is None:
                matches = names[normalize(player["player_display_name"]), player["position"]]
                row = matches[0] if len(matches) == 1 else None
            if row is None or row["position"] != player["position"]:
                reason = "No published NextGen forecast matched this player."
            elif weekly:
                value = row["prediction"] if number(row["prediction"]) else None
                source = f"Dedicated weekly {row['recipe']} · {row['evidence_status']}"
                constraint = ranks.get(row["player_id"])
                if constraint and constraint.get("constraint") and constraint["prediction"] == 0:
                    value, source = 0.0, "Dated published availability constraint"
            elif not row["schedule_known"] or not games.get(team_code(row["team"])):
                reason = "NFL schedule unavailable."
            elif week not in games[team_code(row["team"])]:
                value, source = 0.0, "Scheduled bye"
            elif number(row["prediction"]) and row["scheduled_games"] > 0:
                value = row["prediction"] / row["scheduled_games"]
                source = "Published NextGen next-four points per scheduled game"
            else:
                reason = "No usable published next-four point forecast."
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
    result["source"] = "nextgen_weekly_model" if weekly else "nextgen_weekly_reference"
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
    return dict(
        **result,
        version=rankings["version"],
        recipe="nextgen-weekly-model-v1" if weekly else RECIPE,
        weekly_version=weekly["manifest"]["version"] if weekly else None,
        evidence_status="weekly_model_with_reference_scopes"
        if any(r["evidence_status"] != "reference" for r in learned.values())
        else "reference",
        through_week=report["through_week"],
        published_at=max(
            report.get("published_at", report["generated_at"]), weekly["manifest"]["generated_at"]
        )
        if weekly
        else report.get("published_at", report["generated_at"]),
        pregame_deadline=f"{min(dates)}T00:00:00+00:00" if dates else None,
        limitations=[
            (
                "Dedicated one-week offensive forecasts trained on earlier seasons. "
                "Failed promotion scopes use dedicated weekly references; "
                "no calibrated win probabilities."
                if weekly
                else "Weekly reference: next-four points allocated evenly across scheduled games; "
                "no weekly opponent adjustment or calibrated win probability."
            ),
            "K/DST use up to four prior recorded scores; "
            "no ESPN projections enter any point estimate.",
            "Starting lineups are observed ESPN selections. "
            "Missing starter forecasts withhold team totals and winner picks.",
        ],
    )
