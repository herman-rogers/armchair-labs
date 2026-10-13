"""Full-season scoring series from pinned observations and saved forecast origins.

Historical estimates are reconstructions, not verified pregame forecast captures.
No weights are fitted here and no current forecast is backfilled into earlier weeks.
"""

from collections import Counter, defaultdict
from statistics import mean

from engine.chairstats.views import _row
from engine.espn.sync import LineupEntry
from engine.espn.teams import to_nflverse
from engine.metrics.team_strength import RESERVE, UNAVAILABLE, assumed_lineup, position
from engine.metrics.weekly_lineup import number, team_code
from engine.names import normalize

SKILL = {"QB", "RB", "WR", "TE"}
SPECIAL = {"K", "DST", "D/ST"}


def code(team):
    return team_code(to_nflverse(team))


class ForecastTimeline:
    """Choose each position's latest origin strictly before a week, capped at known data."""

    def __init__(self, store, campaign, season, through_week, schedule):
        self.store, self.campaign = store, campaign
        self.season, self.through_week = season, through_week
        self.manifest = store.campaign(campaign)
        self.cells = defaultdict(list)
        self.games = defaultdict(set)
        self.opponents = {}
        self.last_week = 0
        for game in schedule:
            if game["season"] != season or game["game_type"] != "REG":
                continue
            week = game["week"]
            self.last_week = max(self.last_week, week)
            home, away = code(game["home_team"]), code(game["away_team"])
            for team, opponent in ((home, away), (away, home)):
                self.games[team].add(week)
                self.opponents[team, week] = opponent
        if not self.last_week:
            raise ValueError("No regular-season schedule is available")
        for model in self.manifest["models"]:
            if model.get("position") not in SKILL or "points" not in model["targets"]:
                continue
            for ref in model["forecasts"]:
                if ref["season"] != season or ref["origin_week"] > through_week:
                    continue
                forecast = store.forecast(campaign, model["model_id"], season, ref["origin_week"])
                rows = [_row(model, ref, r, self.manifest) for r in forecast["rows"]]
                names = defaultdict(list)
                for row in rows:
                    names[normalize(row["player_display_name"] or "")].append(row)
                self.cells[model["position"]].append(
                    dict(
                        origin=ref["origin_week"],
                        horizon=model["horizon"],
                        model_id=model["model_id"],
                        rows={r["player_id"]: r for r in rows},
                        names=names,
                        generated_at=forecast["metadata"].get("generated_at"),
                    )
                )

    def cell(self, position, week):
        eligible = [
            c
            for c in self.cells[position]
            if c["origin"] < week and (c["horizon"] != "weekly" or c["origin"] == week - 1)
        ]
        return max(
            eligible,
            key=lambda c: (c["horizon"] == "weekly", c["origin"], c["horizon"] == "remaining"),
            default=None,
        )

    def row(self, position, player_id, week, name=""):
        cell = self.cell(position, week)
        if cell is None:
            return None
        row = cell["rows"].get(player_id)
        if row is None and name:
            matches = cell["names"].get(normalize(name), [])
            row = matches[0] if len(matches) == 1 else None
        return row

    def prediction(self, row, week, *, team=None):
        if row is None:
            return None
        team = code(team or row["team"])
        if team not in self.games:
            return None
        return 0.0 if week not in self.games[team] else row["points_per_game"]

    def metadata(self):
        if (
            self.store.campaign(self.campaign)["manifest_sha256"]
            != self.manifest["manifest_sha256"]
        ):
            raise ValueError("Campaign changed while building scoring trends")
        return dict(
            campaign_id=self.campaign,
            campaign_name=self.manifest["name"],
            manifest_sha256=self.manifest["manifest_sha256"],
            season=self.season,
            through_week=self.through_week,
            last_week=self.last_week,
            forecast_basis="reconstructed",
            forecast_vintages=[
                dict(
                    position=position,
                    model_id=c["model_id"],
                    origin_week=c["origin"],
                    generated_at=c["generated_at"],
                )
                for position, cells in self.cells.items()
                for c in cells
            ],
            method="Latest saved position forecast with an observation cutoff before each "
            "week. Future weeks use the latest available cutoff. Horizon points are divided "
            "by scheduled games; these are scoring paces, not matchup-specific predictions.",
        )


def player_trends(timeline, actuals, *, position="ALL", search="", offset=0, limit=12):
    """Missing stat rows imply zero only for known teams in the complete-week feed."""
    observed = {(r["player_id"], r["week"]): r for r in actuals}
    if len(observed) != len(actuals):
        raise ValueError("Duplicate player-week observations")
    directory = {}
    for cells in timeline.cells.values():
        for cell in sorted(cells, key=lambda c: c["origin"]):
            directory.update(cell["rows"])
    # Observed players without a forecast remain discoverable; predictions stay null.
    for r in sorted(actuals, key=lambda r: r["week"]):
        if r["position"] in SKILL and r["player_id"] not in directory:
            directory[r["player_id"]] = dict(r, points_per_game=None)
    for pid, identity in directory.items():
        latest = timeline.row(identity["position"], pid, timeline.through_week + 1)
        # Keep historical players searchable, without presenting an obsolete rate
        # as their current forecast after a later candidate list excludes them.
        directory[pid] = dict(
            identity, points_per_game=latest["points_per_game"] if latest else None
        )
    candidates = [
        r
        for r in directory.values()
        if (position == "ALL" or r["position"] == position)
        and search.lower() in (r["player_display_name"] or "").lower()
    ]
    candidates.sort(key=lambda r: (-(r.get("points_per_game") or 0), r["player_id"]))
    players = []
    for identity in candidates[offset : offset + limit]:
        pid, pos = identity["player_id"], identity["position"]
        weeks, prior_team = [], None
        for week in range(1, timeline.last_week + 1):
            row = timeline.row(pos, pid, week)
            observation = observed.get((pid, week)) if week <= timeline.through_week else None
            # Actual observations establish actual team, without borrowing a later team.
            if observation:
                prior_team = observation["team"]
            actual_team = code(
                observation["team"] if observation else prior_team or (row or {}).get("team")
            )
            known_schedule = actual_team in timeline.games
            bye = known_schedule and week not in timeline.games[actual_team]
            actual, league_actual, status = None, None, "Upcoming"
            if week <= timeline.through_week:
                if observation:
                    actual, league_actual = (
                        observation["model_points"],
                        observation["league_points"],
                    )
                    status = "Completed"
                elif known_schedule:
                    actual, league_actual = 0.0, 0.0
                    status = "Bye" if bye else "No recorded production"
                else:
                    status = "Observation unavailable"
            future_team = (row or {}).get("team")
            if week > timeline.through_week:
                bye = (
                    code(future_team) in timeline.games
                    and week not in timeline.games[code(future_team)]
                )
                if bye:
                    status = "Bye"
            weeks.append(
                dict(
                    week=week,
                    actual=actual,
                    league_actual=league_actual,
                    predicted=timeline.prediction(row, week),
                    bye=bool(bye),
                    status=status,
                    origin_week=row["origin_week"] if row else None,
                    opponent=timeline.opponents.get(
                        (actual_team if observation else code(future_team), week)
                    ),
                )
            )
        complete = [w for w in weeks if w["week"] <= timeline.through_week and not w["bye"]]
        scored = [w["actual"] for w in complete if w["actual"] is not None]
        league_scored = [w["league_actual"] for w in complete if w["league_actual"] is not None]
        players.append(
            dict(
                player_id=pid,
                name=identity["player_display_name"],
                position=pos,
                team=identity["team"],
                forecast_rate=identity.get("points_per_game"),
                average=mean(scored) if scored else None,
                league_average=mean(league_scored) if league_scored else None,
                average_games=len(scored),
                league_average_games=len(league_scored),
                weeks=weeks,
            )
        )
    return dict(
        **timeline.metadata(),
        players=players,
        total=len(candidates),
        offset=offset,
        limit=limit,
        scoring="Opening Drive scoring: PPR offense; QB 4-point passing TD. "
        "Excludes fumbles, bonuses and cross-position production. Season average uses "
        "completed scheduled games, including zero production but excluding byes.",
    )


def _cover_byes(players, roster_slots, roster, team_id):
    """Retain non-bye starters and jointly assign owned bench cover, including FLEX moves."""
    selected = [p for p in players if p["started"]]
    byes = [p for p in selected if p["bye"]]
    if not byes:
        return selected, [], []
    slots = [p["slot"] for p in selected]
    counts = Counter(slots)
    if any(count > roster_slots.get(slot, 0) for slot, count in counts.items()):
        return (
            selected,
            ["Bye replacements unavailable: starting slots do not match league rules"],
            [],
        )
    candidates = []
    for p in players:
        if p["bye"]:
            continue
        current = roster.get(p["espn_id"])
        if p["started"] or (
            current
            and current.owner_team_id == team_id
            and current.lineup_slot in {"BE", "BN"}
            and p["slot"] in {"BE", "BN"}
            and current.injury_status not in UNAVAILABLE
            and number(p["prediction"])
        ):
            candidates.append(p)
    # The solver keeps all available starters, maximizes filled slots, then points;
    # one player can occupy only one slot even when multiple FLEX slots overlap.
    lineup = assumed_lineup(candidates, slots)
    original_slots = {p["espn_id"]: p["slot"] for p in players}
    adjustments = [f"{p['name']}: bye" for p in byes]
    adjustments.extend(
        f"{p['name']}: {original_slots[p['espn_id']]} → {p['slot']}"
        for p in lineup
        if original_slots[p["espn_id"]] != p["slot"]
    )
    unfilled = list((counts - Counter(p["slot"] for p in lineup)).elements())
    for slot in unfilled:
        lineup.append(
            dict(
                espn_id=None,
                name=f"Unfilled {slot}",
                position=None,
                slot=slot,
                prediction=0.0,
                basis="No eligible bench replacement",
                origin_week=None,
                bye=False,
                started=False,
            )
        )
    return lineup, adjustments, unfilled


def league_trends(timeline, snapshot, identities):
    """Recorded lineups through this week; future byes get legal owned-bench cover."""
    lineups = {}
    history = defaultdict(dict)
    espn_players = {p.espn_id: p for p in snapshot.players}
    for game in snapshot.week_lineups:
        for team_id, lineup in (
            (game.home_team_id, game.home_lineup),
            (game.away_team_id, game.away_lineup),
        ):
            key = team_id, game.week
            if key in lineups:
                raise ValueError("Duplicate fantasy team-week lineup")
            lineups[key] = lineup
            if game.week <= timeline.through_week and game.week < snapshot.week:
                for p in lineup:
                    if p.position in SPECIAL and number(p.points) and not p.on_bye:
                        history[p.espn_id][game.week] = p.points
    teams = []
    for team in snapshot.teams:
        entries = {e.week: e for e in team.schedule}
        weeks = []
        for week in range(1, timeline.last_week + 1):
            future = week > snapshot.week
            lineup = lineups.get((team.team_id, snapshot.week if future else week), [])
            starters = [p for p in lineup if p.started]
            if future:
                captured_ids = {p.espn_id for p in lineup}
                # Newly acquired bench players can have a model forecast before a
                # weekly box score includes them. Never use their season ESPN total
                # as a weekly K/DST fallback.
                lineup = [
                    *lineup,
                    *(
                        LineupEntry(
                            p.espn_id,
                            p.player_display_name,
                            p.position,
                            p.lineup_slot,
                            0.0,
                            None,
                            None,
                            False,
                        )
                        for p in snapshot.players
                        if p.owner_team_id == team.team_id
                        and p.lineup_slot in {"BE", "BN"}
                        and p.espn_id not in captured_ids
                    ),
                ]
            entry = entries.get(week)
            completed = bool(
                entry and entry.played and week < snapshot.week and week <= timeline.through_week
            )
            actual = entry.score if completed and number(entry.score) else None
            details = []
            for p in lineup if future else starters:
                value, basis = None, "Forecast unavailable"
                origin = None
                current = espn_players.get(p.espn_id)
                future_team = code(current.espn_team) if current else None
                bye = p.on_bye if not future else False
                if not future and p.on_bye:
                    value, basis = 0.0, "Scheduled bye"
                elif p.position in SPECIAL:
                    prior = [score for w, score in sorted(history[p.espn_id].items()) if w < week][
                        -4:
                    ]
                    if future:
                        if future_team not in timeline.games:
                            basis = "Future K/DST schedule unavailable"
                        elif week not in timeline.games[future_team]:
                            value, basis = 0.0, "Scheduled bye"
                            bye = True
                        elif prior:
                            value, basis = mean(prior), "Recent ESPN scoring average"
                        elif number(p.projected_points):
                            value, basis = p.projected_points, "Current ESPN estimate held constant"
                    elif number(p.projected_points):
                        value, basis = p.projected_points, f"ESPN Week {week} estimate"
                    elif prior:
                        value, basis = mean(prior), "Prior ESPN scoring average"
                elif p.position in SKILL:
                    row = timeline.row(
                        p.position, identities.get(p.espn_id), week, p.player_display_name
                    )
                    value = timeline.prediction(row, week, team=future_team if future else None)
                    if row:
                        origin = row["origin_week"]
                        basis = f"Opening Drive · after Week {row['origin_week']}"
                        forecast_team = (
                            code(future_team or row["team"]) if future else code(row["team"])
                        )
                        bye = (
                            forecast_team in timeline.games
                            and week not in timeline.games[forecast_team]
                        )
                        if bye:
                            basis = "Scheduled bye"
                # Known future byes remain identifiable even without a model row.
                if (
                    future
                    and future_team in timeline.games
                    and week not in timeline.games[future_team]
                ):
                    value, basis, bye = 0.0, "Scheduled bye", True
                details.append(
                    dict(
                        espn_id=p.espn_id,
                        name=p.player_display_name,
                        position=position(p.position),
                        slot=position(p.slot),
                        prediction=value,
                        basis=basis,
                        origin_week=origin,
                        bye=bye,
                        started=p.slot not in RESERVE,
                    )
                )
            adjustments, unfilled = [], []
            if future:
                details, adjustments, unfilled = _cover_byes(
                    details,
                    {position(slot): count for slot, count in snapshot.roster_slots.items()},
                    espn_players,
                    team.team_id,
                )
            predictions = [p["prediction"] for p in details]
            origins = {p["origin_week"] for p in details if p["origin_week"] is not None}
            predicted = (
                sum(predictions) if predictions and all(number(v) for v in predictions) else None
            )
            status = (
                "Completed"
                if completed
                else "Projected lineup · bye cover"
                if future
                else "Pending"
            )
            if unfilled:
                status += " · unfilled: " + ", ".join(unfilled)
            if not starters:
                status += " · lineup unavailable"
            # ESPN scores are authoritative; disclose an unreconciled breakdown.
            reconciliation = bool(
                completed
                and starters
                and all(number(p.points) for p in starters)
                and abs(sum(p.points for p in starters) - actual) < 0.01
            )
            weeks.append(
                dict(
                    week=week,
                    actual=actual,
                    predicted=predicted,
                    status=status,
                    origin_week=max(origins) if origins else None,
                    origins=sorted(origins),
                    bye=False,
                    starters=details,
                    lineup_basis="bye_adjusted" if future else "recorded",
                    adjustments=adjustments,
                    unfilled_slots=unfilled,
                    reconciled=reconciliation if completed else None,
                )
            )
        complete = [w for w in weeks if w["actual"] is not None]
        pairs = [w for w in complete if w["predicted"] is not None]
        teams.append(
            dict(
                team_id=team.team_id,
                name=team.team_name,
                average=mean(w["actual"] for w in complete) if complete else None,
                average_games=len(complete),
                weeks=weeks,
                mae=mean(abs(w["actual"] - w["predicted"]) for w in pairs) if pairs else None,
            )
        )
    return dict(
        **timeline.metadata(),
        captured_at=snapshot.captured_at,
        current_week=snapshot.week,
        regular_season_weeks=snapshot.regular_season_weeks,
        teams=teams,
        scoring="Full ESPN team scores versus Opening Drive offense + ESPN K/DST estimates. "
        "Offensive model scoring differs from league bonuses and other rules.",
        future_basis="After the current week, keep non-bye starters and fill bye slots with "
        "the highest-projected eligible bench combination under the league's position and FLEX "
        "rules. Only currently rostered bench players are used; reserve, out, doubtful and "
        "suspended players are excluded from replacements. Unfilled slots score zero. "
        "K/DST use recent completed scores, or the current ESPN weekly "
        "estimate if no history exists. Beyond the fantasy regular season, this is a "
        "lineup scoring scenario, not a scheduled matchup.",
    )
