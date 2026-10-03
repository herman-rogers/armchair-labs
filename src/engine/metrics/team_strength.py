"""Team diagnostics: observed results, published weekly points and legal depth.

Concentration describes the allocation of points, not a calibrated floor or win
probability. All current estimates come from our weekly forecast, never ESPN.
"""

from collections import defaultdict
from math import isclose, isfinite
from statistics import mean, median, stdev

SKILLS = {"QB", "RB", "WR", "TE"}
RESERVE = {"BE", "BN", "IR", "ER", "", None}
UNAVAILABLE = {"OUT", "INJURY_RESERVE", "IR", "SUSPENSION", "SUSPENDED", "DOUBTFUL"}
POSITIONS = ("QB", "RB", "WR", "TE", "K", "D/ST")


def number(value):
    return isinstance(value, (int, float)) and isfinite(value)


def average(values):
    return mean(values) if values else None


def position(value):
    return "D/ST" if value == "DST" else value


def scoring_shape(values):
    if not values:
        return dict(
            total=None,
            top2=None,
            other_starters=None,
            middle=None,
            weakest=None,
            median=None,
            top2_share=None,
            effective_contributors=None,
        )
    values = sorted(values, reverse=True)
    positive = [max(0, v) for v in values]
    total = sum(positive)
    shares = [v / total for v in positive] if total else []
    return dict(
        total=sum(values),
        top2=sum(values[:2]),
        other_starters=sum(values[2:]),
        middle=sum(values[2:6]),
        weakest=sum(values[-3:]),
        median=median(values),
        top2_share=sum(shares[:2]) if shares else None,
        effective_contributors=1 / sum(v * v for v in shares) if shares else None,
    )


def legal_lineup(players, slots):
    """Small exact assignment without adding an optimizer dependency."""
    states = {0: (0.0, [])}
    for player in players:
        if not number(player["prediction"]):
            continue
        next_states = dict(states)
        for mask, (score, lineup) in states.items():
            for i, slot in enumerate(slots):
                if mask & (1 << i) or player["position"] not in slot.split("/"):
                    continue
                updated = mask | (1 << i)
                candidate = score + player["prediction"]
                if updated not in next_states or candidate > next_states[updated][0]:
                    next_states[updated] = (candidate, [*lineup, (player, slot)])
        states = next_states
    return states.get((1 << len(slots)) - 1)


def assumed_lineup(players, slots):
    """Keep selected starters, filling gaps jointly from the eligible bench.

    Prioritize retaining starters, then filling slots, then projected points.
    Unknown starter forecasts remain unknown; they do not justify benching a
    healthy player. Each player can be assigned once, including across FLEX.
    """
    states = {0: ((0, 0.0, 0), [])}
    for player in players:
        next_states = dict(states)
        for mask, (score, lineup) in states.items():
            for i, slot in enumerate(slots):
                eligible = player["position"] == slot or player["position"] in slot.split("/")
                if mask & (1 << i) or not eligible:
                    continue
                updated = mask | (1 << i)
                candidate = (
                    score[0] + player["started"],
                    score[1] + (player["prediction"] if number(player["prediction"]) else 0),
                    score[2] + (player["slot"] == slot),
                )
                if updated not in next_states or candidate > next_states[updated][0]:
                    next_states[updated] = (candidate, [*lineup, (player, slot)])
        states = next_states
    mask = max(states, key=lambda m: (states[m][0][0], m.bit_count(), *states[m][0][1:]))
    return [dict(player, slot=slot) for player, slot in states[mask][1]]


def team_strength(snapshot, forecasts=None):
    through = min(snapshot.week - 1, snapshot.regular_season_weeks)
    slots = [
        slot
        for slot, count in snapshot.roster_slots.items()
        if set(slot.split("/")) <= SKILLS
        for _ in range(count)
    ]
    aligned = bool(
        forecasts
        and forecasts.get("source") in ("nextgen_weekly_model", "nextgen_weekly_reference")
        and forecasts.get("season") == snapshot.season
        and forecasts.get("requested_week") == snapshot.week
        and forecasts.get("current_week") == snapshot.week
        and forecasts.get("through_week") == snapshot.week - 1
        and forecasts.get("captured_at") == snapshot.captured_at
    )
    predicted = (
        {
            s["team_id"]: {p["espn_id"]: p for p in s["forecasts"]}
            for g in forecasts["matchups"]
            for s in (g["home"], g["away"])
        }
        if aligned
        else {}
    )
    lineup_records = {
        (g.week, tid): g.lineup_for(tid)
        for g in snapshot.week_lineups
        for tid in (g.home_team_id, g.away_team_id)
    }
    actuals = {
        (t.team_id, g.week): g.score
        for t in snapshot.teams
        for g in t.schedule
        if 1 <= g.week <= through and g.outcome in ("W", "L", "T") and number(g.score)
    }
    teams = []
    for team in snapshot.teams:
        history, positional, contributions = [], defaultdict(list), defaultdict(float)
        breakdowns, top_players = [], set()
        for game in sorted(team.schedule, key=lambda g: g.week):
            score = actuals.get((team.team_id, game.week))
            if score is None:
                continue
            opponent = actuals.get((game.opponent_team_id, game.week))
            peers = [v for (tid, w), v in actuals.items() if w == game.week and tid != team.team_id]
            all_play = (
                sum((score > v) + 0.5 * (score == v) for v in peers) / len(peers)
                if len(peers) == len(snapshot.teams) - 1 and peers
                else None
            )
            starters = [
                p
                for p in lineup_records.get((game.week, team.team_id), [])
                if p.slot not in RESERVE
            ]
            complete = bool(
                starters
                and all(number(p.points) for p in starters)
                and isclose(sum(p.points for p in starters), score, abs_tol=0.05)
            )
            shape = scoring_shape([])
            if complete:
                skill = [p for p in starters if p.position in SKILLS]
                if len(skill) <= len(slots):
                    shape = scoring_shape(
                        [p.points for p in skill] + [0] * (len(slots) - len(skill))
                    )
                    breakdowns.append(shape)
                    top_players.update(
                        p.espn_id for p in sorted(skill, key=lambda p: -p.points)[:2]
                    )
                for pos in POSITIONS:
                    positional[pos].append(
                        sum(p.points for p in starters if position(p.position) == pos)
                    )
                for p in skill:
                    contributions[p.espn_id] += p.points
            history.append(
                dict(
                    week=game.week,
                    score=score,
                    opponent_score=opponent,
                    opponent_team_id=game.opponent_team_id,
                    outcome=game.outcome,
                    margin=score - opponent if opponent is not None else None,
                    all_play=all_play,
                    scoring_rank=1 + sum(v > score for v in peers)
                    if all_play is not None
                    else None,
                    top2_share=shape["top2_share"],
                    # Keep recorded ownership and negative scores; missing or
                    # unreconciled lineups must not become zero-point charts.
                    contributions=[
                        dict(
                            espn_id=p.espn_id,
                            name=p.player_display_name,
                            position=p.position,
                            points=p.points,
                        )
                        for p in starters
                        if p.position in SKILLS
                    ]
                    if complete
                    else None,
                )
            )
        scores = [r["score"] for r in history]
        against = [r["opponent_score"] for r in history if r["opponent_score"] is not None]
        positive = sorted([max(v, 0) for v in contributions.values()], reverse=True)
        ppg = average(scores)
        results = dict(
            weeks=len(scores),
            through_week=through,
            record="–".join(str(sum(r["outcome"] == v for r in history)) for v in ("W", "L", "T")),
            points=sum(scores) if scores else None,
            ppg=ppg,
            ppg_change=ppg - mean(scores[:-1]) if len(scores) > 1 else None,
            points_against=average(against) if len(against) == len(scores) else None,
            average_margin=average([r["margin"] for r in history])
            if history and all(r["margin"] is not None for r in history)
            else None,
            all_play=average([r["all_play"] for r in history])
            if history and all(r["all_play"] is not None for r in history)
            else None,
            high=max(scores) if scores else None,
            low=min(scores) if scores else None,
            volatility=stdev(scores) if len(scores) > 1 else None,
            season_pace=ppg * snapshot.regular_season_weeks
            if len(scores) == through and scores
            else None,
            breakdown_weeks=len(breakdowns),
            middle_ppg=average([r["middle"] for r in breakdowns]),
            weakest_ppg=average([r["weakest"] for r in breakdowns]),
            top2_share=average(
                [r["top2_share"] for r in breakdowns if r["top2_share"] is not None]
            ),
            cumulative_top2_share=sum(positive[:2]) / sum(positive) if sum(positive) > 0 else None,
            different_top_scorers=len(top_players),
        )
        current_lineup = {
            p.espn_id: p for p in lineup_records.get((snapshot.week, team.team_id), [])
        }
        roster = []
        unavailable_ids = set()
        for player in snapshot.players:
            if player.owner_team_id != team.team_id:
                continue
            forecast = predicted.get(team.team_id, {}).get(player.espn_id)
            observed = current_lineup.get(player.espn_id)
            value = forecast.get("prediction") if forecast else None
            reasons = []
            if player.lineup_slot in ("IR", "ER"):
                reasons.append("Reserve slot")
            if player.injury_status in UNAVAILABLE:
                reasons.append(player.injury_status.replace("_", " ").title())
            if observed is None:
                reasons.append("Weekly lineup not captured")
            elif observed.on_bye:
                reasons.append("Bye")
            if (
                player.lineup_slot in ("IR", "ER")
                or player.injury_status in UNAVAILABLE
                or (observed is not None and observed.on_bye)
            ):
                unavailable_ids.add(player.espn_id)
            if forecast and (forecast["slot"] != player.lineup_slot):
                reasons.append("Lineup changed since forecast")
                value = None
            if not number(value):
                value = None
                reasons.append("Weekly forecast unavailable")
            roster.append(
                dict(
                    espn_id=player.espn_id,
                    name=player.player_display_name,
                    position=position(player.position),
                    slot=player.lineup_slot,
                    prediction=value,
                    basis=forecast.get("basis") if forecast else None,
                    status=player.injury_status,
                    started=player.lineup_slot not in RESERVE,
                    usable=not reasons,
                    reason=" · ".join(reasons) or None,
                )
            )
        all_slots = [
            slot
            for slot, count in snapshot.roster_slots.items()
            if slot not in RESERVE
            for _ in range(count)
        ]
        lineup = assumed_lineup(
            [
                p
                for p in roster
                if p["espn_id"] not in unavailable_ids and (p["started"] or p["usable"])
            ],
            all_slots,
        )
        lineup_slots = {p["espn_id"]: p["slot"] for p in lineup}
        lineup_ids = set(lineup_slots)
        adjustments = [
            f"{p['name']}: {p['slot']} → {lineup_slots[p['espn_id']]}"
            for p in roster
            if p["espn_id"] in lineup_ids and p["slot"] != lineup_slots[p["espn_id"]]
        ]
        omitted = [p for p in roster if p["started"] and p["espn_id"] not in lineup_ids]
        selected = [p for p in lineup if p["position"] in SKILLS]
        ready = [p for p in roster if p["usable"] and p["position"] in SKILLS]
        legal = legal_lineup(selected, slots) if slots and len(selected) == len(slots) else None
        complete = bool(legal and all(p["usable"] for p in selected))
        current_shape = (
            scoring_shape([p["prediction"] for p in selected]) if complete else scoring_shape([])
        )
        best = legal_lineup(ready, slots) if slots else None
        bench = [p for p in ready if p["espn_id"] not in lineup_ids]
        depth = []
        for starter in selected:
            best_replacement, conservative = None, None
            if complete:
                remaining = [p for p in selected if p["espn_id"] != starter["espn_id"]]
                for backup in bench:
                    option = legal_lineup([*remaining, backup], slots)
                    if option and (best_replacement is None or option[0] > best_replacement[0]):
                        best_replacement = (*option, backup)
                    if option and backup["status"] != "QUESTIONABLE":
                        conservative = (
                            max(conservative, option[0]) if conservative is not None else option[0]
                        )
            backup = best_replacement[2] if best_replacement else None
            moves = (
                [
                    f"{p['name']}: {p['slot']} → {slot}"
                    for p, slot in best_replacement[1]
                    if p["slot"] != slot
                ]
                if best_replacement
                else []
            )
            depth.append(
                dict(
                    starter=starter["name"],
                    espn_id=starter["espn_id"],
                    position=starter["position"],
                    starter_points=starter["prediction"],
                    replacement=backup["name"] if backup else None,
                    replacement_points=backup["prediction"] if backup else None,
                    replacement_status=backup["status"] if backup else None,
                    moves=moves,
                    drop=current_shape["total"] - best_replacement[0] if best_replacement else None,
                    conservative_drop=current_shape["total"] - conservative
                    if conservative is not None
                    else None,
                    reason=None
                    if backup
                    else "Current lineup incomplete"
                    if not complete
                    else "No eligible bench forecast",
                )
            )
        drops = [d["drop"] for d in depth if d["drop"] is not None]
        positions = []
        for pos in POSITIONS:
            pool = [p for p in roster if p["position"] == pos]
            starters = [p for p in lineup if p["position"] == pos]
            available = [p for p in pool if p["usable"] and p["espn_id"] not in lineup_ids]
            positions.append(
                dict(
                    position=pos,
                    actual_ppg=average(positional[pos]),
                    historical_weeks=len(positional[pos]),
                    starters=len(starters),
                    weekly_points=sum(p["prediction"] for p in starters)
                    if all(p["usable"] for p in starters)
                    and aligned
                    and len(starters) >= snapshot.roster_slots.get(pos, 0)
                    else None,
                    usable_backups=len(available),
                    best_backup=max(available, key=lambda p: p["prediction"])["name"]
                    if available
                    else None,
                    unavailable=sum(not p["usable"] for p in pool),
                )
            )
        total_starters = lineup
        weekly_total = (
            sum(p["prediction"] for p in total_starters)
            if complete
            and total_starters
            and len(lineup) == len(all_slots)
            and all(p["usable"] for p in total_starters)
            else None
        )
        teams.append(
            dict(
                team_id=team.team_id,
                team_name=team.team_name,
                results=results,
                history=history,
                positions=positions,
                players=roster,
                lineup=lineup,
                faab_remaining=team.faab_remaining,
                current=dict(
                    **current_shape,
                    adjustments=adjustments,
                    omitted=[dict(name=p["name"], reason=p["reason"]) for p in omitted],
                    weekly_total=weekly_total,
                    complete=complete,
                    covered=sum(p["usable"] for p in selected),
                    required=len(slots),
                    best_legal=best[0] if best else None,
                    usable_bench=len(bench),
                    replaceable=len(drops),
                    mean_drop=mean(drops) if complete and len(drops) == len(slots) else None,
                    worst_drop=max(drops) if drops else None,
                ),
                depth=depth,
            )
        )
    # Relative ranks are withheld unless every team has the same coverage.
    for team in teams:
        for key in ("ppg", "all_play"):
            comparable = all(
                t["results"]["weeks"] == through and t["results"][key] is not None for t in teams
            )
            team["results"][key + "_rank"] = (
                (1 + sum(t["results"][key] > team["results"][key] for t in teams))
                if comparable
                else None
            )
        for row in team["positions"]:
            peers = [
                next(p for p in t["positions"] if p["position"] == row["position"]) for t in teams
            ]
            comparable = all(
                p["historical_weeks"] == through and p["actual_ppg"] is not None for p in peers
            )
            row["rank"] = (
                1 + sum(p["actual_ppg"] > row["actual_ppg"] for p in peers) if comparable else None
            )
    return dict(
        season=snapshot.season,
        week=snapshot.week,
        captured_at=snapshot.captured_at,
        version=forecasts.get("version") if aligned else None,
        forecast_available=aligned,
        forecast_label="NextGen weekly estimates",
        teams=teams,
    )
