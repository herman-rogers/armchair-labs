"""The matchup view must report the week that happened, not the week that should have.

The bug these guard: `/matchups` ignored its `week` argument when building lineups. It
solved a best-available lineup from *today's* roster and labelled the result week 2,
so a quarterback who spent week 2 on the bench appeared as that week's starter and a
player acquired since appeared in a week he had not been rostered for. Every number on
the page was a model output wearing a scoreboard's clothes.
"""

from __future__ import annotations

import time

import polars as pl
import pytest

from patron.api import league_routes
from patron.espn.crosswalk import IS_MINE, OWNER_TEAM_ID, OWNER_TEAM_NAME, JoinReport
from patron.espn.service import LeagueState
from patron.espn.sync import LeagueSnapshot, LineupEntry, ScheduleEntry, TeamState, WeekLineups

SLOTS = {"QB": 1, "RB": 1, "WR": 1, "RB/WR/TE": 1}


def entry(name: str, position: str, slot: str, points: float, espn_id: int) -> LineupEntry:
    return LineupEntry(
        espn_id=espn_id,
        player_display_name=name,
        position=position,
        slot=slot,
        points=points,
        projected_points=points + 1.0,
        pro_opponent="SF",
        on_bye=False,
    )


def board() -> pl.DataFrame:
    """A board that rates the benched quarterback above the one who actually started."""
    return pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4"],
            "player_display_name": ["Started QB", "Benched QB", "Mine RB", "Theirs RB"],
            "position": ["QB", "QB", "RB", "RB"],
            "adj_vor": [1.0, 9.0, 5.0, 4.0],
            OWNER_TEAM_ID: [3, 3, 3, 4],
            OWNER_TEAM_NAME: ["Mine", "Mine", "Mine", "Theirs"],
            IS_MINE: [True, True, True, False],
        }
    )


def snapshot(week: int = 3, *, lineups: list[WeekLineups] | None = None) -> LeagueSnapshot:
    schedule = [
        ScheduleEntry(week=w, opponent_team_id=4, score=0.0, outcome="U") for w in range(1, 6)
    ]
    return LeagueSnapshot(
        captured_at="2026-09-22T00:00:00+00:00",
        league_id=1,
        league_name="Sweaty Plays",
        season=2026,
        week=week,
        my_team_id=3,
        regular_season_weeks=14,
        roster_slots=SLOTS,
        teams=[
            TeamState(3, "Mine", "me", 1, 1, 100, schedule=schedule),
            TeamState(
                4,
                "Theirs",
                "them",
                1,
                1,
                100,
                schedule=[
                    ScheduleEntry(week=w, opponent_team_id=3, score=0.0, outcome="U")
                    for w in range(1, 6)
                ],
            ),
        ],
        week_lineups=lineups or [],
    )


def played_week(week: int = 2) -> WeekLineups:
    return WeekLineups(
        week=week,
        home_team_id=3,
        away_team_id=4,
        home_score=42.0,
        away_score=40.0,
        home_projected=38.0,
        away_projected=39.0,
        home_lineup=[
            entry("Started QB", "QB", "QB", 27.0, 1),
            entry("Mine RB", "RB", "RB", 15.0, 3),
            entry("Benched QB", "QB", "BE", 31.0, 2),
        ],
        away_lineup=[entry("Theirs RB", "RB", "RB", 40.0, 4)],
    )


def state(snap: LeagueSnapshot) -> LeagueState:
    return LeagueState(
        snapshot=snap,
        tagged_board=board(),
        join_report=JoinReport(),
        fetched_at=time.monotonic(),
    )


@pytest.fixture
def call(monkeypatch: pytest.MonkeyPatch):
    def run(snap: LeagueSnapshot, week: int | None = None):
        monkeypatch.setattr(league_routes, "_state", lambda *a, **k: state(snap))
        return league_routes.matchups(service=None, week=week, version="v1")  # type: ignore[arg-type]

    return run


def mine(response: dict) -> dict:
    game = next(m for m in response["matchups"] if m["involves_me"])
    return next(side for side in (game["home"], game["away"]) if side["team_id"] == 3)


class TestAPlayedWeek:
    def test_the_lineup_that_was_set_is_the_lineup_shown(self, call) -> None:
        team = mine(call(snapshot(lineups=[played_week(2)]), week=2))

        assert [s["player_display_name"] for s in team["starters"]] == [
            "Started QB",
            "Mine RB",
        ]
        assert [b["player_display_name"] for b in team["bench"]] == ["Benched QB"]

    def test_a_higher_rated_bench_player_is_not_promoted(self, call) -> None:
        """The regression itself. The board rates Benched QB 9x the starter; a
        best-available lineup put him under centre in a week he never played."""
        team = mine(call(snapshot(lineups=[played_week(2)]), week=2))
        assert all(s["player_display_name"] != "Benched QB" for s in team["starters"])

    def test_the_total_is_what_was_scored(self, call) -> None:
        response = call(snapshot(lineups=[played_week(2)]), week=2)
        team = mine(response)
        assert team["total"] == pytest.approx(42.0)
        assert team["metric"] == "points"
        assert team["source"] == "actual"
        assert response["source"] == "actual"

    def test_espn_s_pre_week_projection_is_kept(self, call) -> None:
        assert mine(call(snapshot(lineups=[played_week(2)]), week=2))["projected_total"] == 38.0

    def test_starters_read_in_lineup_order_not_espn_storage_order(self, call) -> None:
        week = played_week(2)
        week.home_lineup = [
            entry("Flex Guy", "WR", "RB/WR/TE", 5.0, 9),
            entry("Mine RB", "RB", "RB", 15.0, 3),
            entry("Started QB", "QB", "QB", 27.0, 1),
        ]
        team = mine(call(snapshot(lineups=[week]), week=2))
        assert [s["slot"] for s in team["starters"]] == ["QB", "RB", "RB/WR/TE"]

    def test_the_bench_leads_with_what_was_left_on_it(self, call) -> None:
        week = played_week(2)
        week.home_lineup += [entry("Quiet Bench", "WR", "BE", 2.0, 8)]
        team = mine(call(snapshot(lineups=[week]), week=2))
        assert [b["player_display_name"] for b in team["bench"]] == ["Benched QB", "Quiet Bench"]

    def test_a_played_week_claims_no_unrankable_caveat(self, call) -> None:
        """Every player in a played week scored what they scored, so the projected
        view's "understated total" warning would be a lie here."""
        assert mine(call(snapshot(lineups=[played_week(2)]), week=2))["unranked_starters"] == 0


class TestAFutureWeek:
    def test_falls_back_to_fieldable_strength(self, call) -> None:
        response = call(snapshot(lineups=[played_week(2)]), week=5)
        assert response["source"] == "projected"
        assert mine(response)["source"] == "projected"

    def test_the_best_available_lineup_is_used(self, call) -> None:
        """With no lineup to report, promoting the better player is the right answer."""
        team = mine(call(snapshot(lineups=[played_week(2)]), week=5))
        assert [s["player_display_name"] for s in team["starters"] if s["slot"] == "QB"] == [
            "Benched QB"
        ]

    def test_both_paths_emit_the_same_row_shape(self, call) -> None:
        """One renderer draws both, so a field present in one and absent in the other
        surfaces as `undefined` on screen rather than a blank."""
        snap = snapshot(lineups=[played_week(2)])
        actual = mine(call(snap, week=2))
        projected = mine(call(snap, week=5))

        for key in ("source", "metric", "total", "projected_total", "starters", "bench"):
            assert key in actual and key in projected
        assert set(actual["starters"][0]) == set(projected["starters"][0])
        assert set(actual["bench"][0]) == set(projected["bench"][0])


class TestWeekSelection:
    def test_the_current_week_is_the_default(self, call) -> None:
        response = call(snapshot(week=2, lineups=[played_week(2)]))
        assert response["requested_week"] == 2
        assert response["source"] == "actual"

    @pytest.mark.parametrize("week", [1, 2, 3])
    def test_your_own_game_comes_first_with_all_other_games_present(self, call, week: int) -> None:
        other = WeekLineups(
            week=week, home_team_id=5, away_team_id=6, home_score=1.0, away_score=2.0
        )
        response = call(snapshot(lineups=[other, played_week(week)]), week=week)
        assert len(response["matchups"]) == 2
        assert response["matchups"][0]["involves_me"] is True
        assert response["matchups"][1]["involves_me"] is False

    def test_the_margin_is_between_the_two_sides(self, call) -> None:
        response = call(snapshot(lineups=[played_week(2)]), week=2)
        game = next(m for m in response["matchups"] if m["involves_me"])
        assert game["margin"] == pytest.approx(game["home"]["total"] - game["away"]["total"])
