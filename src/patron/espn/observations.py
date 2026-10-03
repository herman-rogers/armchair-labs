"""League schedule and actual lineups, independent of every player-value model."""

from patron.espn.sync import LeagueSnapshot, LineupEntry


def lineup_rows(entries: list[LineupEntry]) -> list[dict]:
    # This API supplies league facts only. Provider projections stay in the raw
    # snapshot for other consumers, never in the overview's observation contract.
    return [
        dict(
            espn_id=entry.espn_id,
            player_display_name=entry.player_display_name,
            position=entry.position,
            slot=entry.slot,
            points=entry.points,
            pro_opponent=entry.pro_opponent,
            on_bye=entry.on_bye,
            started=entry.started,
        )
        for entry in entries
    ]


def weekly_matchups(snapshot: LeagueSnapshot, week: int) -> dict:
    teams = {team.team_id: team for team in snapshot.teams}
    records = {
        frozenset((r.home_team_id, r.away_team_id)): r
        for r in snapshot.week_lineups
        if r.week == week
    }
    pairs = {key: (r.home_team_id, r.away_team_id) for key, r in records.items()}
    for team in snapshot.teams:
        for entry in team.schedule:
            if entry.week == week and entry.opponent_team_id in teams:
                pairs.setdefault(
                    frozenset((team.team_id, entry.opponent_team_id)),
                    (team.team_id, entry.opponent_team_id),
                )

    def side(team_id, record):
        team = teams.get(team_id)
        scheduled = next((e for e in team.schedule if e.week == week), None) if team else None
        home = record and record.home_team_id == team_id
        return {
            "team_id": team_id,
            "team_name": team.team_name if team else str(team_id),
            "score": (record.home_score if home else record.away_score)
            if record
            else scheduled.score
            if scheduled and scheduled.played
            else None,
            "lineup": lineup_rows(record.lineup_for(team_id)) if record else [],
            "lineup_available": bool(record and record.lineup_for(team_id)),
        }

    games = []
    for key, (home, away) in pairs.items():
        record = records.get(key)
        games.append(
            {
                "home": side(home, record),
                "away": side(away, record),
                "status": "completed"
                if week < snapshot.week
                else "current"
                if week == snapshot.week
                else "scheduled",
                "involves_me": snapshot.my_team_id in key,
            }
        )
    games.sort(key=lambda r: (not r["involves_me"], r["home"]["team_id"]))
    weeks = sorted(
        set(range(1, snapshot.regular_season_weeks + 1))
        | {e.week for t in snapshot.teams for e in t.schedule}
        | {e.week for e in snapshot.week_lineups}
    )
    return {
        "season": snapshot.season,
        "current_week": snapshot.week,
        "requested_week": week,
        "available_weeks": weeks,
        "my_team_id": snapshot.my_team_id,
        "captured_at": snapshot.captured_at,
        "matchups": games,
        "source": "espn_observations",
    }
