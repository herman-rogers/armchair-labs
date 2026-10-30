import pytest

from engine.espn.sync import LeagueSnapshot, PlayerState, ScheduleEntry, TeamState
from engine.metrics.power_rankings import best_lineup, power_rankings, win_probability


def fixture(playoff_teams=2):
    """Four teams, weeks 1-2 played, weeks 3-4 left; one QB and one FLEX slot."""
    snapshot = LeagueSnapshot(
        "2026-10-01T12:00:00+00:00",
        1,
        "Test",
        2026,
        3,
        1,
        regular_season_weeks=4,
        playoff_teams=playoff_teams,
        roster_slots={"QB": 1, "RB/WR/TE": 1, "K": 1},
    )
    pairs = {1: 2, 2: 1, 3: 4, 4: 3}
    played = {1: (120, 110), 2: (90, 100), 3: (100, 105), 4: (95, 80)}
    for i in (1, 2, 3, 4):
        opponent = pairs[i]
        schedule = [
            ScheduleEntry(
                w,
                opponent,
                played[i][w - 1],
                "W" if played[i][w - 1] > played[opponent][w - 1] else "L",
            )
            for w in (1, 2)
        ]
        schedule += [ScheduleEntry(w, 5 - i if w == 3 else pairs[i], 0, "U") for w in (3, 4)]
        snapshot.teams.append(TeamState(i, f"Team {i}", None, 0, 0, 100, schedule=schedule))
        for j, pos in enumerate(("QB", "WR", "RB")):
            pid = i * 10 + j
            snapshot.players.append(
                PlayerState(pid, f"P{pid}", pos, "BUF", "ACTIVE", 0, 0, None, i, f"Team {i}", "BE")
            )
    curves = {}
    for i in (1, 2, 3, 4):
        curves[i * 10] = {3: 10.0 * i, 4: 10.0 * i}
        curves[i * 10 + 1] = {3: 5.0, 4: 5.0}
        curves[i * 10 + 2] = {3: 4.0, 4: 4.0}
    rankings = dict(
        campaign_id="fixture",
        campaign_name="Fixture",
        horizon="remaining",
        rankings=[
            dict(
                player_id=f"id{pid}",
                player_display_name=f"P{pid}",
                position=next(p.position for p in snapshot.players if p.espn_id == pid),
                team="BUF",
                season=2026,
                origin_week=2,
                projection=dict(future_weeks=[dict(week=w, points=v) for w, v in curve.items()]),
            )
            for pid, curve in curves.items()
        ],
    )
    identities = {pid: f"id{pid}" for pid in curves}
    return snapshot, rankings, identities


def team(result, team_id):
    return next(t for t in result["teams"] if t["team_id"] == team_id)


def test_rating_is_mean_best_lineup_and_ranks_order_by_it():
    result = power_rankings(*fixture(), sims=500)
    assert result["status"] == "ok" and result["weeks"] == [3, 4]
    assert team(result, 4)["rating"] == pytest.approx(45.0)
    assert [team(result, i)["rating_rank"] for i in (4, 3, 2, 1)] == [1, 2, 3, 4]


def test_bye_counts_zero_but_missing_curve_is_excluded_not_zero():
    snapshot, rankings, identities = fixture()
    rankings["rankings"][0]["projection"]["future_weeks"] = [
        dict(week=4, points=10.0)
    ]  # P10 bye in week 3
    del identities[11]
    rankings["rankings"] = [r for r in rankings["rankings"] if r["player_id"] != "id11"]
    t = team(power_rankings(snapshot, rankings, identities, sims=100), 1)
    assert t["unmatched_players"] == ["P11"]
    assert t["weekly"][0]["points"] == pytest.approx(4.0)  # QB bye scores 0, RB fills FLEX
    assert t["weekly"][1]["points"] == pytest.approx(14.0)


@pytest.mark.parametrize("status,weeks_out", [("OUT", 1), ("INJURY_RESERVE", 2)])
def test_unavailable_players_are_benched_for_the_right_number_of_weeks(status, weeks_out):
    snapshot, rankings, identities = fixture()
    next(p for p in snapshot.players if p.espn_id == 41).injury_status = status
    weekly = team(power_rankings(snapshot, rankings, identities, sims=100), 4)["weekly"]
    assert [w["points"] for w in weekly][:weeks_out] == [44.0] * weeks_out
    assert [w["points"] for w in weekly][weeks_out:] == [45.0] * (2 - weeks_out)


def test_simulation_is_deterministic_and_awards_exactly_the_playoff_spots():
    a = power_rankings(*fixture(), sims=2000)
    b = power_rankings(*fixture(), sims=2000)
    assert [t["playoff_odds"] for t in a["teams"]] == [t["playoff_odds"] for t in b["teams"]]
    assert sum(t["playoff_odds"] for t in a["teams"]) == pytest.approx(2.0)
    assert sum(t["first_seed_odds"] for t in a["teams"]) == pytest.approx(1.0)
    for t in a["teams"]:
        assert t["projected_wins"] + t["projected_losses"] == pytest.approx(4)


def test_unknown_playoff_spots_skip_odds_but_still_project_wins():
    result = power_rankings(*fixture(playoff_teams=0), sims=200)
    assert all(
        t["playoff_odds"] is None and t["projected_wins"] is not None for t in result["teams"]
    )
    assert any("Playoff spots are unknown" in note for note in result["limitations"])


def test_no_curves_is_unavailable():
    snapshot, rankings, identities = fixture()
    for row in rankings["rankings"]:
        row["projection"] = None
    result = power_rankings(snapshot, rankings, identities)
    assert result["status"] == "unavailable" and result["teams"] == []


def test_win_probability_is_symmetric_and_lineup_leaves_slots_empty_when_short():
    assert win_probability(110, 100, 20) == pytest.approx(1 - win_probability(100, 110, 20))
    assert win_probability(100, 100, 20) == pytest.approx(0.5)
    points, starters, filled = best_lineup(
        [dict(espn_id=1, position="QB", points=9.0)], ["QB", "RB"]
    )
    assert (points, starters, filled) == (9.0, (1,), 1)
