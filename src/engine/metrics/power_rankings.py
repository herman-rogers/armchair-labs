"""Model power rankings: rest-of-season lineup strength and a schedule simulation.

Each team's strength is the average of its best legal offensive lineup (QB/RB/WR/TE)
for every remaining regular-season week, using the campaign's week-by-week player
curves. Players on bye score zero; players without a curve stay out rather than
being treated as zero-point starters. K and D/ST have no campaign curve, so model
totals exclude them.

The simulation plays the remaining league schedule with those weekly means and the
league's observed within-team weekly spread from ESPN results. It describes
schedule and roster strength under the model; it is not a calibrated playoff model.
"""

from collections import defaultdict
from math import erf, sqrt

import numpy as np

from engine.metrics.team_strength import SKILLS, number
from engine.names import normalize

COMPLETE = ("W", "L", "T")
UNAVAILABLE_NOW = {"OUT", "INJURY_RESERVE", "IR", "SUSPENSION", "SUSPENDED", "DOUBTFUL"}
LONG_TERM = {"INJURY_RESERVE", "IR", "SUSPENSION", "SUSPENDED"}
#: NFL injured reserve keeps a player out at least four games.
LONG_TERM_WEEKS = 4
SIMULATIONS = 10_000


def best_lineup(players, slots):
    """Highest-scoring assignment of players to slots; slots may stay empty."""
    states = {0: (0.0, ())}
    for player in players:
        next_states = dict(states)
        for mask, (score, lineup) in states.items():
            for i, slot in enumerate(slots):
                if mask & (1 << i) or player["position"] not in slot.split("/"):
                    continue
                updated = mask | (1 << i)
                candidate = score + player["points"]
                if updated not in next_states or candidate > next_states[updated][0]:
                    next_states[updated] = (candidate, (*lineup, player["espn_id"]))
        states = next_states
    mask = max(states, key=lambda m: (states[m][0], m.bit_count()))
    return states[mask][0], states[mask][1], mask.bit_count()


def win_probability(mean_a, mean_b, sd):
    """Normal approximation for independent team scores with a shared spread."""
    return 0.5 * (1 + erf((mean_a - mean_b) / (2 * sd)))


def weekly_spread(snapshot):
    """Pooled within-team standard deviation of completed ESPN weekly scores."""
    deviations, teams = [], 0
    for team in snapshot.teams:
        scores = [g.score for g in team.schedule if g.outcome in COMPLETE and number(g.score)]
        if len(scores) >= 2:
            teams += 1
            centre = sum(scores) / len(scores)
            deviations += [(s - centre) ** 2 for s in scores]
    dof = len(deviations) - teams
    return (sqrt(sum(deviations) / dof), len(deviations)) if dof > 0 else (None, len(deviations))


def power_rankings(snapshot, rankings, identities, sims=SIMULATIONS, seed=0):
    rows = [r for r in rankings["rankings"] if r.get("season") == snapshot.season]
    base = dict(
        campaign_id=rankings.get("campaign_id"),
        campaign_name=rankings.get("campaign_name"),
        horizon=rankings.get("horizon"),
        season=snapshot.season,
        week=snapshot.week,
        captured_at=snapshot.captured_at,
        playoff_teams=snapshot.playoff_teams or None,
    )
    curved = [r for r in rows if (r.get("projection") or {}).get("future_weeks") is not None]
    if not curved:
        return dict(
            base,
            status="unavailable",
            issue="This campaign has no week-by-week remaining-season forecasts this season.",
            teams=[],
        )
    origins = sorted({r["origin_week"] for r in curved})
    by_id = {r["player_id"]: r for r in curved}
    by_name = defaultdict(list)
    for row in curved:
        by_name[normalize(row["player_display_name"] or ""), row["position"]].append(row)

    def curve(player):
        row = by_id.get(identities.get(player.espn_id))
        if row is None:
            matches = by_name[normalize(player.player_display_name), player.position]
            row = matches[0] if len(matches) == 1 else None
        if row is None or row["position"] != player.position:
            return None
        return {g["week"]: g["points"] for g in row["projection"]["future_weeks"]}

    slots = [
        slot
        for slot, count in snapshot.roster_slots.items()
        if set(slot.split("/")) <= SKILLS
        for _ in range(count)
    ]
    remaining = {}
    for team in snapshot.teams:
        for g in team.schedule:
            if (
                g.outcome not in COMPLETE
                and snapshot.week <= g.week <= snapshot.regular_season_weeks
            ):
                remaining[g.week, *sorted((team.team_id, g.opponent_team_id))] = None
    games = sorted(remaining)
    weeks = sorted({w for w, _, _ in games})

    teams = []
    for team in snapshot.teams:
        roster = [
            p for p in snapshot.players if p.owner_team_id == team.team_id and p.position in SKILLS
        ]
        curves = {p.espn_id: curve(p) for p in roster}
        unmatched = [p.player_display_name for p in roster if curves[p.espn_id] is None]
        weekly = {}
        for week in weeks:
            out = week - snapshot.week
            pool = [
                dict(
                    espn_id=p.espn_id, position=p.position, points=curves[p.espn_id].get(week, 0.0)
                )
                for p in roster
                if curves[p.espn_id] is not None
                and not (out == 0 and (p.injury_status or "").upper() in UNAVAILABLE_NOW)
                and not (out < LONG_TERM_WEEKS and (p.injury_status or "").upper() in LONG_TERM)
            ]
            points, starters, filled = best_lineup(pool, slots)
            weekly[week] = dict(week=week, points=points, filled=filled, starters=list(starters))
        completed = [g for g in team.schedule if g.outcome in COMPLETE]
        teams.append(
            dict(
                team_id=team.team_id,
                team_name=team.team_name,
                wins=sum(g.outcome == "W" for g in completed),
                losses=sum(g.outcome == "L" for g in completed),
                ties=sum(g.outcome == "T" for g in completed),
                points_for=sum(g.score for g in completed if number(g.score)),
                rating=sum(w["points"] for w in weekly.values()) / len(weekly) if weekly else None,
                this_week=weekly[snapshot.week]["points"] if snapshot.week in weekly else None,
                unfilled_weeks=sum(w["filled"] < len(slots) for w in weekly.values()),
                unmatched_players=unmatched,
                weekly=weekly,
            )
        )
    ratings = sorted((t["rating"] for t in teams if t["rating"] is not None), reverse=True)
    for t in teams:
        t["rating_rank"] = ratings.index(t["rating"]) + 1 if t["rating"] is not None else None

    sd, sd_weeks = weekly_spread(snapshot)
    index = {t["team_id"]: i for i, t in enumerate(teams)}
    limitations = [
        "Model totals cover QB/RB/WR/TE starters only; K and D/ST have no campaign curve.",
        "Players listed OUT or DOUBTFUL miss this week; IR and suspended players miss four weeks.",
        "Later-week curves decay toward each player's per-game anchor; "
        "they do not anticipate trades, waivers or new injuries.",
    ]
    simulated = dict(simulations=0, score_sd=sd, score_sd_samples=sd_weeks)
    if sd is not None and games:
        rng = np.random.default_rng(seed)
        means = np.array(
            [[t["weekly"][w]["points"] for w in weeks] for t in teams], dtype=float
        )  # teams x weeks
        scores = means[None] + rng.normal(0, sd, (sims, *means.shape))
        wins = np.tile(
            np.array([t["wins"] + 0.5 * t["ties"] for t in teams], dtype=float), (sims, 1)
        )
        points = np.tile(np.array([t["points_for"] for t in teams], dtype=float), (sims, 1))
        points += scores.sum(axis=2)
        column = {w: i for i, w in enumerate(weeks)}
        for week, a, b in games:
            if a not in index or b not in index:
                continue
            i, j, k = index[a], index[b], column[week]
            a_wins = scores[:, i, k] > scores[:, j, k]
            wins[:, i] += a_wins
            wins[:, j] += ~a_wins
        # Seeds: wins first, points for as the tiebreak (divisions are not modelled).
        order = np.lexsort((-points, -wins), axis=1)
        seeds = np.empty_like(order)
        np.put_along_axis(seeds, order, np.arange(1, len(teams) + 1)[None].repeat(sims, 0), axis=1)
        for t in teams:
            i = index[t["team_id"]]
            t["projected_wins"] = float(wins[:, i].mean())
            played = t["wins"] + t["losses"] + t["ties"]
            t["projected_losses"] = (
                played + sum(t["team_id"] in (a, b) for _, a, b in games) - t["projected_wins"]
            )
            t["average_seed"] = float(seeds[:, i].mean())
            t["first_seed_odds"] = float((seeds[:, i] == 1).mean())
            t["playoff_odds"] = (
                float((seeds[:, i] <= snapshot.playoff_teams).mean())
                if snapshot.playoff_teams
                else None
            )
        simulated["simulations"] = sims
        limitations.append(
            f"Simulated weekly scores vary by {sd:.1f} points, the league's observed within-team "
            "spread; playoff seeding uses record then points for and ignores divisions."
        )
        if not snapshot.playoff_teams:
            limitations.append(
                "Playoff spots are unknown for this snapshot; refresh data to capture them."
            )
    for t in teams:
        for week in t["weekly"].values():
            opponent = next(
                (
                    b if a == t["team_id"] else a
                    for w, a, b in games
                    if w == week["week"] and t["team_id"] in (a, b)
                ),
                None,
            )
            week["opponent_team_id"] = opponent
            week["win_probability"] = (
                win_probability(
                    week["points"], teams[index[opponent]]["weekly"][week["week"]]["points"], sd
                )
                if opponent in index and sd
                else None
            )
    for t in teams:
        t["weekly"] = list(t["weekly"].values())
    return dict(
        base,
        status="ok",
        issue=None,
        origin_weeks=origins,
        weeks=weeks,
        slots=slots,
        limitations=limitations,
        **simulated,
        teams=teams,
    )
