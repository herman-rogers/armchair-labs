"""Chronology, missingness and lineup assumptions behind the season charts."""

from contextlib import nullcontext
from copy import deepcopy

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine.espn.sync import (
    LeagueSnapshot,
    LineupEntry,
    PlayerState,
    ScheduleEntry,
    TeamState,
    WeekLineups,
)
from engine.metrics.scoring_trends import (
    ForecastTimeline,
    _cover_byes,
    league_trends,
    player_trends,
)
from engine.tables.observations import scoring_weeks


class Store:
    def __init__(self):
        self.token = "pinned"

    def campaign(self, _):
        return dict(
            name="Test campaign",
            manifest_sha256=self.token,
            campaign_id="test",
            models=[
                dict(
                    model_id="qb",
                    origin_week=0,
                    position="QB",
                    horizon="season",
                    targets=["points"],
                    forecasts=[dict(season=2026, origin_week=w) for w in (0, 2, 4)],
                )
            ],
        )

    def forecast(self, campaign, model, season, origin):
        # The Week 4 forecast is intentionally huge to expose accidental leakage.
        rate = {0: 20, 2: 30, 4: 999}[origin]
        return dict(
            metadata={"generated_at": "2026-10-11T00:00:00Z"},
            rows=[
                dict(
                    player_id="p1",
                    player_name="Passer",
                    position="QB",
                    team="BUF",
                    predictions={"points": rate * 3},
                    scheduled_games=3,
                ),
                dict(
                    player_id="p2",
                    player_name="Unknown Team",
                    position="QB",
                    team=None,
                    predictions={"points": 10},
                    scheduled_games=3,
                ),
            ],
        )


def schedule():
    return [
        dict(
            season=2026,
            week=w,
            game_type="REG",
            home_team="BUF" if w != 3 else "LA",
            away_team="NYJ" if w != 3 else "SF",
        )
        for w in range(1, 6)
    ]


def timeline(through=2, store=None):
    return ForecastTimeline(store or Store(), "test", 2026, through, schedule())


def actual(week=1, points=10):
    return dict(
        player_id="p1",
        player_display_name="Passer",
        position="QB",
        team="BUF",
        week=week,
        model_points=points,
        league_points=points + 2,
    )


def test_rates_change_only_after_origin_and_future_origins_do_not_leak():
    result = player_trends(timeline(), [actual()])
    player = next(p for p in result["players"] if p["player_id"] == "p1")
    assert [w["predicted"] for w in player["weeks"]] == [20, 20, 0, 30, 30]
    assert [w["origin_week"] for w in player["weeks"]] == [0, 0, 2, 2, 2]
    assert [w["actual"] for w in player["weeks"]] == [10, 0, None, None, None]
    assert player["average"] == 5  # Missed scheduled game is included, future is not.
    newer = player_trends(timeline(4), [actual()])["players"][0]
    assert newer["weeks"][3]["predicted"] == 30
    assert newer["weeks"][4]["predicted"] == 999


def test_byes_excluded_from_average_unknown_coverage_not_zero_and_negative_kept():
    result = player_trends(timeline(3), [actual(points=-4)])
    passer = next(p for p in result["players"] if p["player_id"] == "p1")
    unknown = next(p for p in result["players"] if p["player_id"] == "p2")
    assert passer["average"] == -2
    assert passer["average_games"] == 2
    assert passer["weeks"][2]["actual"] == 0 and passer["weeks"][2]["bye"]
    assert unknown["average"] is None
    assert all(w["actual"] is None and w["predicted"] is None for w in unknown["weeks"])


def test_unknown_recorded_scoring_remains_missing_and_observed_only_players_are_visible():
    missing = dict(actual(), model_points=None, league_points=None)
    observed_only = dict(actual(), player_id="new", player_display_name="New player")
    players = player_trends(timeline(), [missing, observed_only])["players"]
    passer = next(p for p in players if p["player_id"] == "p1")
    assert passer["weeks"][0]["actual"] is None
    newcomer = next(p for p in players if p["player_id"] == "new")
    assert newcomer["weeks"][0]["actual"] == 10
    assert all(w["predicted"] is None for w in newcomer["weeks"])


def test_filtering_pagination_and_revision_pin():
    assert player_trends(timeline(), [], search="pass", limit=1)["total"] == 1
    assert player_trends(timeline(), [], position="RB")["players"] == []
    assert player_trends(timeline(), [], offset=1, limit=1)["players"][0]["player_id"] == "p2"
    store = Store()
    pinned = timeline(store=store)
    store.token = "changed"
    with pytest.raises(ValueError, match="Campaign changed"):
        player_trends(pinned, [])


def lineup(points, projected=7, bye=False):
    return [
        LineupEntry(1, "Passer", "QB", "QB", points, 500, None, bye),
        LineupEntry(3, "Kicker", "K", "K", points, projected, None, bye),
    ]


def league_snapshot():
    return LeagueSnapshot(
        captured_at="2026-10-11T00:00:00Z",
        league_id=1,
        league_name="Test",
        season=2026,
        week=3,
        my_team_id=1,
        regular_season_weeks=4,
        roster_slots={"QB": 1, "K": 1},
        teams=[
            TeamState(
                1,
                "One",
                None,
                1,
                0,
                100,
                schedule=[ScheduleEntry(1, 2, 16, "W"), ScheduleEntry(2, 2, 24, "U")],
            ),
            TeamState(2, "Two", None, 0, 1, 100),
        ],
        players=[
            PlayerState(3, "Kicker", "K", "BUF", None, None, None, None, 1, "One", "K"),
            PlayerState(1, "Passer", "QB", "BUF", None, None, None, None, 1, "One", "QB"),
        ],
        week_lineups=[
            WeekLineups(1, 1, 2, 16, 0, home_lineup=lineup(8)),
            WeekLineups(2, 1, 2, 24, 0, home_lineup=lineup(12)),
            WeekLineups(3, 1, 2, 999, 0, home_lineup=lineup(999, bye=True)),
        ],
    )


def test_league_uses_recorded_lineups_espn_specials_and_only_settled_actuals():
    result = league_trends(timeline(), league_snapshot(), {1: "p1"})
    team = result["teams"][0]
    assert team["weeks"][0]["predicted"] == 27  # ESPN 7, not same-week actual 8.
    assert team["weeks"][0]["actual"] == 16
    assert team["weeks"][1]["actual"] is None  # Outcome U is not completed.
    assert team["weeks"][2]["actual"] is None
    assert team["weeks"][3]["predicted"] == 40  # OD 30 + prior K scores mean(8,12).
    assert team["weeks"][3]["lineup_basis"] == "bye_adjusted"
    assert team["average"] == 16
    assert all(w["predicted"] is None for w in result["teams"][1]["weeks"])


def test_current_actual_mutation_cannot_change_future_predictions():
    original = league_snapshot()
    changed = deepcopy(original)
    changed.week_lineups[-1].home_lineup[1].points = -9999
    a = league_trends(timeline(), original, {1: "p1"})
    b = league_trends(timeline(), changed, {1: "p1"})
    assert a == b


def test_missing_starter_withholds_total_and_future_bye_flags_are_recomputed():
    snapshot = league_snapshot()
    snapshot.week_lineups[-1].home_lineup[0].player_display_name = "No match"
    snapshot.week_lineups[-1].home_lineup[0].espn_id = 99
    result = league_trends(timeline(), snapshot, {1: "p1"})
    assert result["teams"][0]["weeks"][3]["predicted"] is None
    assert result["teams"][0]["weeks"][3]["starters"][1]["prediction"] == 10


def cover_player(pid, pos, slot, points, *, bye=False):
    return dict(
        espn_id=pid,
        name=f"Player {pid}",
        position=pos,
        slot=slot,
        prediction=points,
        started=slot not in {"BE", "BN", "IR", "ER"},
        bye=bye,
        origin_week=2,
        basis="Test forecast",
    )


def cover_roster(players):
    return {
        p["espn_id"]: PlayerState(
            p["espn_id"],
            p["name"],
            p["position"],
            "LA",
            None,
            None,
            None,
            None,
            1,
            "One",
            p["slot"],
        )
        for p in players
    }


def test_bye_cover_moves_flex_starter_to_unlock_best_legal_bench_combination():
    players = [
        cover_player(1, "RB", "RB", 0, bye=True),
        cover_player(2, "RB", "RB/WR/TE", 12),
        cover_player(3, "WR", "WR", 0, bye=True),
        cover_player(4, "WR", "BE", 20),
        cover_player(5, "TE", "BE", 19),
        cover_player(6, "RB", "BE", 1),
        cover_player(7, "QB", "BE", 100),
    ]
    result, changes, unfilled = _cover_byes(
        players, {"RB": 1, "WR": 1, "RB/WR/TE": 1}, cover_roster(players), 1
    )
    assert {(p["espn_id"], p["slot"]) for p in result} == {(2, "RB"), (4, "WR"), (5, "RB/WR/TE")}
    assert sum(p["prediction"] for p in result) == 51
    assert "Player 2: RB/WR/TE → RB" in changes
    assert not unfilled


@pytest.mark.parametrize(
    "exclusion",
    ["bye", "IR", "ER", "OUT", "DOUBTFUL", "SUSPENDED", "other_team", "free_agent", "missing"],
)
def test_bye_cover_rejects_unavailable_or_unowned_bench_players(exclusion):
    players = [cover_player(1, "QB", "QB", 0, bye=True), cover_player(2, "QB", "BE", 99)]
    roster = cover_roster(players)
    if exclusion == "bye":
        players[1]["bye"] = True
    elif exclusion in {"IR", "ER"}:
        roster[2].lineup_slot = exclusion
    elif exclusion in {"OUT", "DOUBTFUL", "SUSPENDED"}:
        roster[2].injury_status = exclusion
    elif exclusion in {"other_team", "free_agent"}:
        roster[2].owner_team_id = 2 if exclusion == "other_team" else None
    else:
        players[1]["prediction"] = None
    result, _, unfilled = _cover_byes(players, {"QB": 1}, roster, 1)
    assert unfilled == ["QB"]
    assert result[0]["prediction"] == 0
    assert result[0]["espn_id"] is None


def test_bye_cover_never_reuses_a_player_or_benches_a_non_bye_starter():
    players = [
        cover_player(1, "WR", "WR", None),
        cover_player(2, "WR", "WR", 0, bye=True),
        cover_player(3, "TE", "RB/WR/TE", 0, bye=True),
        cover_player(4, "WR", "BE", 30),
    ]
    result, _, unfilled = _cover_byes(players, {"WR": 2, "RB/WR/TE": 1}, cover_roster(players), 1)
    assert len(unfilled) == 1
    assert sorted(p["espn_id"] for p in result if p["espn_id"] is not None) == [1, 4]
    assert next(p for p in result if p["espn_id"] == 1)["prediction"] is None


def test_bye_cover_respects_superflex_and_defense_slots():
    players = [
        cover_player(1, "QB", "QB/RB/WR/TE", 0, bye=True),
        cover_player(2, "D/ST", "D/ST", 0, bye=True),
        cover_player(3, "QB", "BE", 30),
        cover_player(4, "D/ST", "BE", 9),
        cover_player(5, "K", "BE", 100),
    ]
    result, _, unfilled = _cover_byes(
        players, {"QB/RB/WR/TE": 1, "D/ST": 1}, cover_roster(players), 1
    )
    assert {(p["espn_id"], p["slot"]) for p in result} == {(3, "QB/RB/WR/TE"), (4, "D/ST")}
    assert not unfilled


def test_bye_cover_requires_matching_league_rules_and_leaves_non_bye_lineups_alone():
    players = [cover_player(1, "QB", "QB", 0, bye=True), cover_player(2, "QB", "BE", 30)]
    result, changes, _ = _cover_byes(players, {"RB": 1}, cover_roster(players), 1)
    assert result == [players[0]]
    assert "do not match league rules" in changes[0]
    players[0].update(bye=False, prediction=10)
    result, changes, _ = _cover_byes(players, {"QB": 1}, cover_roster(players), 1)
    assert result == [players[0]]
    assert not changes


def test_future_bye_cover_uses_weekly_schedule_and_special_team_history_only():
    class BenchStore(Store):
        def forecast(self, *args):
            result = super().forecast(*args)
            result["rows"].append(
                dict(
                    player_id="bench",
                    player_name="Backup",
                    position="QB",
                    team="LA",
                    predictions={"points": 120},
                    scheduled_games=3,
                )
            )
            return result

    snapshot = league_snapshot()
    snapshot.week = 2
    snapshot.week_lineups = snapshot.week_lineups[:2]
    snapshot.players.extend(
        [
            PlayerState(10, "Backup", "QB", "LA", None, None, None, None, 1, "One", "BE"),
            PlayerState(11, "Backup K", "K", "SF", None, None, None, 999, 1, "One", "BE"),
        ]
    )
    for game in snapshot.week_lineups:
        game.home_lineup.append(LineupEntry(11, "Backup K", "K", "BE", 6, None, None, False))
    result = league_trends(timeline(store=BenchStore()), snapshot, {1: "p1", 10: "bench"})
    weeks = result["teams"][0]["weeks"]
    assert weeks[0]["predicted"] == 27 and weeks[0]["actual"] == 16
    assert weeks[1]["predicted"] == 27  # The current lineup remains the observed selection.
    assert weeks[2]["predicted"] == 46  # New bench QB 40 + prior K score 6, not season total 999.
    assert {p["espn_id"] for p in weeks[2]["starters"]} == {10, 11}
    assert weeks[2]["origin_week"] == 2
    assert not weeks[2]["unfilled_slots"]
    assert weeks[3]["predicted"] == 38  # Original starters return after their bye.
    assert not weeks[3]["adjustments"]


def test_sql_prefers_current_capture_and_preserves_unknown_component_values():
    with duckdb.connect() as db:
        db.execute("CREATE SCHEMA analytics")
        schema = """player_id VARCHAR, player_display_name VARCHAR, position VARCHAR,
            team VARCHAR, week INTEGER, season INTEGER, season_type VARCHAR, league_points DOUBLE,
            passing_yards INTEGER, passing_tds INTEGER, passing_interceptions INTEGER,
            rushing_yards INTEGER, rushing_tds INTEGER, receptions INTEGER,
            receiving_yards INTEGER, receiving_tds INTEGER"""
        for table in ("current_weeks", "nfl_player_weeks"):
            db.execute(f"CREATE TABLE analytics.{table} ({schema})")
            db.execute(
                f"INSERT INTO analytics.{table} VALUES "
                "('p1','Passer','QB','BUF',1,2026,'REG',99,100,1,1,10,1,0,0,0)"
            )
        db.execute("UPDATE analytics.current_weeks SET passing_yards=200")
        db.execute(
            "INSERT INTO analytics.current_weeks VALUES "
            "('p2','Receiver','WR','BUF',1,2026,'REG',NULL,0,0,0,0,0,1,NULL,0)"
        )
        rows = scoring_weeks(db, 2026)
        assert len(rows) == 2
        assert rows[0]["model_points"] == 17
        assert isinstance(rows[0]["model_points"], float)
        assert rows[1]["model_points"] is None


def test_api_refreshes_from_new_observation_cutoff_and_leaves_future_actuals_blank(monkeypatch):
    from engine.api import scoring_routes

    cutoff = dict(season=2026, through_week=2)
    monkeypatch.setattr(scoring_routes, "connect", lambda: nullcontext(None))
    monkeypatch.setattr(scoring_routes, "store", Store)
    monkeypatch.setattr(scoring_routes.observations, "observation_cutoff", lambda db: cutoff)
    monkeypatch.setattr(scoring_routes.observations, "schedule", lambda db, season: schedule())
    monkeypatch.setattr(scoring_routes.observations, "scoring_weeks", lambda db, season: [actual()])
    monkeypatch.setattr(scoring_routes.observations, "espn_ids", lambda db: {1: "p1"})
    monkeypatch.setattr(
        scoring_routes, "snapshot_observations", lambda service: (league_snapshot(), False, 0)
    )
    app = FastAPI()
    app.include_router(scoring_routes.router)
    with TestClient(app) as client:
        first = client.get("/api/scoring/players")
        assert first.status_code == 200
        assert first.headers["X-Campaign-Manifest"] == "pinned"
        assert first.json()["players"][0]["weeks"][-1]["predicted"] == 30
        assert client.get("/api/scoring/players?limit=100").status_code == 422
        assert client.get("/api/scoring/league").json()["teams"][0]["weeks"][3]["predicted"] == 40
        cutoff["through_week"] = 4
        second = client.get("/api/scoring/players").json()
        assert second["through_week"] == 4
        assert second["players"][0]["weeks"][-1]["predicted"] == 999
        assert second["players"][0]["weeks"][-1]["actual"] is None
