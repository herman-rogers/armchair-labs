import pytest

from engine.espn.sync import (
    LeagueSnapshot,
    LineupEntry,
    PlayerState,
    ScheduleEntry,
    TeamState,
    WeekLineups,
)
from engine.metrics.team_strength import legal_lineup, team_strength


def fixture():
    snapshot = LeagueSnapshot(
        "2026-10-01T12:00:00+00:00",
        1,
        "Test",
        2026,
        3,
        1,
        roster_slots={"QB": 1, "TE": 1, "RB/WR/TE": 1},
        regular_season_weeks=14,
    )
    snapshot.teams = [
        TeamState(
            i,
            f"Team {i}",
            None,
            1,
            1,
            100,
            schedule=[
                ScheduleEntry(1, 3 - i, 30 if i == 1 else 40, "L" if i == 1 else "W"),
                ScheduleEntry(2, 3 - i, 60 if i == 1 else 40, "W" if i == 1 else "L"),
            ],
        )
        for i in (1, 2)
    ]
    for w in (1, 2, 3):
        entries = [
            [
                LineupEntry(
                    i * 10 + j,
                    f"P{i * 10 + j}",
                    pos,
                    slot,
                    (10 if w == 1 else 20) if i == 1 else 40 / 3,
                    999999,
                    "BUF",
                    False,
                )
                for j, (pos, slot) in enumerate((("QB", "QB"), ("TE", "TE"), ("TE", "RB/WR/TE")))
            ]
            for i in (1, 2)
        ]
        snapshot.week_lineups.append(
            WeekLineups(
                w, 1, 2, 30 if w == 1 else 60, 40, home_lineup=entries[0], away_lineup=entries[1]
            )
        )
    sides = []
    for i in (1, 2):
        predictions = []
        for j, (pos, slot, value) in enumerate(
            (
                ("QB", "QB", 20),
                ("TE", "TE", 15),
                ("TE", "RB/WR/TE", 10),
                ("WR", "BE", 8),
                ("QB", "BE", 12),
            )
        ):
            pid = i * 10 + j
            snapshot.players.append(
                PlayerState(
                    pid, f"P{pid}", pos, "BUF", "ACTIVE", 0, 0, 999999, i, f"Team {i}", slot
                )
            )
            if slot == "BE":
                getattr(
                    snapshot.week_lineups[-1], "home_lineup" if i == 1 else "away_lineup"
                ).append(LineupEntry(pid, f"P{pid}", pos, slot, 0, 999999, "BUF", False))
            predictions.append(
                dict(espn_id=pid, slot=slot, prediction=value, basis="Our weekly model")
            )
        sides.append(dict(team_id=i, forecasts=predictions))
    forecast = dict(
        source="nextgen_weekly_model",
        version="test",
        season=2026,
        requested_week=3,
        current_week=3,
        through_week=2,
        captured_at=snapshot.captured_at,
        matchups=[dict(home=sides[0], away=sides[1])],
    )
    return snapshot, forecast


def test_completed_results_and_flex_replacement():
    snapshot, forecast = fixture()
    team = team_strength(snapshot, forecast)["teams"][0]
    assert team["results"]["ppg"] == 45
    assert team["results"]["ppg_change"] == 15
    assert team["results"]["all_play"] == 0.5
    assert team["current"]["total"] == 45
    assert team["current"]["top2_share"] == pytest.approx(35 / 45)
    te = next(d for d in team["depth"] if d["espn_id"] == 11)
    assert te["replacement"] == "P13"
    assert te["drop"] == 7
    assert "P12: RB/WR/TE → TE" in te["moves"]
    assert team["current"]["replaceable"] == 3
    # Each player can fill only one slot in an optimized lineup.
    assert legal_lineup([dict(position="TE", prediction=100)], ["TE", "RB/WR/TE"]) is None


def test_no_provider_or_partial_week_leakage_and_missing_forecast_no_fallback():
    snapshot, forecast = fixture()
    before = team_strength(snapshot, forecast)
    snapshot.week_lineups[-1].home_score = 999999
    snapshot.week_lineups[-1].home_projected = 999999
    snapshot.week_lineups[-1].home_lineup[0].points = 999999
    snapshot.players[0].projected_points = -999999
    assert team_strength(snapshot, forecast) == before
    forecast["matchups"][0]["home"]["forecasts"][0]["prediction"] = None
    result = team_strength(snapshot, forecast)["teams"][0]
    assert result["current"]["total"] is None
    assert result["current"]["mean_drop"] is None
    assert result["results"]["ppg"] == 45


@pytest.mark.parametrize(
    "key,value",
    [
        ("captured_at", "old"),
        ("season", 2025),
        ("through_week", 1),
        ("source", "espn_observations"),
    ],
)
def test_mismatched_forecasts_are_withheld(key, value):
    snapshot, forecast = fixture()
    forecast[key] = value
    result = team_strength(snapshot, forecast)
    assert not result["forecast_available"]
    assert result["teams"][0]["current"]["total"] is None


def test_zero_and_unavailable_depth_and_questionable_sensitivity():
    snapshot, forecast = fixture()
    snapshot.players[4].injury_status = "OUT"
    snapshot.players[3].injury_status = "QUESTIONABLE"
    result = team_strength(snapshot, forecast)["teams"][0]
    assert next(d for d in result["depth"] if d["position"] == "QB")["replacement"] is None
    assert result["current"]["mean_drop"] is None
    te = next(d for d in result["depth"] if d["espn_id"] == 11)
    assert te["drop"] == 7 and te["conservative_drop"] is None
    for p in forecast["matchups"][0]["home"]["forecasts"]:
        p["prediction"] = 0
    result = team_strength(snapshot, forecast)["teams"][0]
    assert result["current"]["total"] == 0
    assert result["current"]["top2_share"] is None


def test_missing_history_and_lineup_reconciliation_do_not_invent_rank():
    snapshot, forecast = fixture()
    snapshot.teams[1].schedule.pop()
    result = team_strength(snapshot, forecast)
    assert all(t["results"]["ppg_rank"] is None for t in result["teams"])
    assert result["teams"][0]["results"]["all_play"] is None
    snapshot, forecast = fixture()
    snapshot.week_lineups[0].home_lineup[0].points = 9999
    result = team_strength(snapshot, forecast)["teams"][0]
    assert result["results"]["ppg"] == 45
    assert result["results"]["breakdown_weeks"] == 1
    assert all(p["rank"] is None for p in result["positions"])
    assert result["history"][0]["contributions"] is None


def test_chart_contributions_use_recorded_starters_and_preserve_negative_points():
    snapshot, forecast = fixture()
    recorded = snapshot.week_lineups[0].home_lineup
    recorded[0].points = -2
    recorded[1].points = 22
    recorded[0].player_display_name = "Former starter"
    recorded.append(LineupEntry(999, "Bench scorer", "WR", "BE", 100, 999999, "BUF", False))
    snapshot.players[0].player_display_name = "Renamed current player"
    team = team_strength(snapshot, forecast)["teams"][0]
    history = team["history"]
    assert [w["week"] for w in history] == [1, 2]
    assert history[0]["contributions"][0] == dict(
        espn_id=10, name="Former starter", position="QB", points=-2
    )
    assert sum(p["points"] for p in history[0]["contributions"]) == 30
    assert all(p["espn_id"] != 999 for p in history[0]["contributions"])
    snapshot.week_lineups[0].home_lineup[0].points = None
    assert team_strength(snapshot, forecast)["teams"][0]["history"][0]["contributions"] is None


@pytest.mark.parametrize("status", ["OUT", "DOUBTFUL", "INJURY_RESERVE", "SUSPENDED", "BYE"])
def test_unavailable_starter_uses_bench_and_flex_for_every_current_metric(status):
    snapshot, forecast = fixture()
    if status == "BYE":
        snapshot.week_lineups[-1].home_lineup[1].on_bye = True
    else:
        snapshot.players[1].injury_status = status
    team = team_strength(snapshot, forecast)["teams"][0]
    assert team["current"]["complete"]
    assert team["current"]["total"] == 38
    assert team["current"]["weekly_total"] == 38
    assert team["current"]["covered"] == 3
    assert team["current"]["usable_bench"] == 1
    assert {p["espn_id"]: p["slot"] for p in team["lineup"]} == {
        10: "QB",
        12: "TE",
        13: "RB/WR/TE",
    }
    assert "P12: RB/WR/TE → TE" in team["current"]["adjustments"]
    assert "P13: BE → RB/WR/TE" in team["current"]["adjustments"]
    assert team["current"]["omitted"][0]["name"] == "P11"
    assert team["current"]["top2_share"] == pytest.approx(30 / 38)
    assert next(p for p in team["positions"] if p["position"] == "WR")["weekly_points"] == 8
    qb = next(d for d in team["depth"] if d["position"] == "QB")
    assert qb["replacement"] == "P14" and qb["drop"] == 8
    assert all(d["replacement"] != "P13" for d in team["depth"])
    assert team["results"]["ppg"] == 45
    # Observed ownership and slots are preserved separately from assumptions.
    assert next(p for p in team["players"] if p["espn_id"] == 13)["slot"] == "BE"
    assert snapshot.players[1].lineup_slot == "TE"


def test_simultaneous_absences_do_not_reuse_backup_or_invent_complete_lineup():
    snapshot, forecast = fixture()
    snapshot.players[0].injury_status = "OUT"
    snapshot.players[1].injury_status = "OUT"
    team = team_strength(snapshot, forecast)["teams"][0]
    assert team["current"]["total"] == 30
    assert team["current"]["usable_bench"] == 0
    assert len({p["espn_id"] for p in team["lineup"]}) == 3
    snapshot.players[2].injury_status = "OUT"
    team = team_strength(snapshot, forecast)["teams"][0]
    assert not team["current"]["complete"]
    assert team["current"]["total"] is None
    assert team["current"]["weekly_total"] is None
    assert len(team["lineup"]) == 2


def test_healthy_selected_starters_stay_even_with_higher_projected_bench():
    snapshot, forecast = fixture()
    forecast["matchups"][0]["home"]["forecasts"][4]["prediction"] = 100
    team = team_strength(snapshot, forecast)["teams"][0]
    assert team["current"]["total"] == 45
    assert team["current"]["best_legal"] == 125
    assert team["current"]["adjustments"] == []


def test_empty_slot_is_filled_and_unavailable_bench_is_skipped():
    snapshot, forecast = fixture()
    snapshot.players[0].lineup_slot = "BE"
    snapshot.players[0].injury_status = "OUT"
    forecast["matchups"][0]["home"]["forecasts"][0]["slot"] = "BE"
    team = team_strength(snapshot, forecast)["teams"][0]
    assert team["current"]["total"] == 37
    assert "P14: BE → QB" in team["current"]["adjustments"]
    assert team["current"]["omitted"] == []
    snapshot.players[4].injury_status = "OUT"
    team = team_strength(snapshot, forecast)["teams"][0]
    assert team["current"]["total"] is None
    assert next(p for p in team["positions"] if p["position"] == "QB")["weekly_points"] is None
    assert next(p for p in team["positions"] if p["position"] == "TE")["weekly_points"] == 25


@pytest.mark.parametrize("pos", ["K", "D/ST"])
def test_specialist_backup_is_included_in_weekly_total(pos):
    snapshot, forecast = fixture()
    snapshot.roster_slots[pos] = 1
    for pid, slot, status, prediction in [(90, pos, "OUT", 10), (91, "BE", "ACTIVE", 7)]:
        snapshot.players.append(
            PlayerState(pid, f"P{pid}", pos, "BUF", status, 0, 0, 999999, 1, "Team 1", slot)
        )
        snapshot.week_lineups[-1].home_lineup.append(
            LineupEntry(pid, f"P{pid}", pos, slot, 0, 999999, "BUF", False)
        )
        forecast["matchups"][0]["home"]["forecasts"].append(
            dict(espn_id=pid, slot=slot, prediction=prediction, basis="Our weekly model")
        )
    team = team_strength(snapshot, forecast)["teams"][0]
    assert team["current"]["total"] == 45
    assert team["current"]["weekly_total"] == 52
    assert next(p for p in team["positions"] if p["position"] == pos)["weekly_points"] == 7
