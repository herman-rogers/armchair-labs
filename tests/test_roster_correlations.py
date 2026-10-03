from copy import deepcopy
from types import SimpleNamespace

import duckdb
import pytest

from engine.metrics.roster_correlations import (
    attach_roster_correlations,
    historical_summary,
    observed_pair,
)


def history():
    return [
        dict(score=score, contributions=[dict(espn_id=1, points=a), dict(espn_id=2, points=b)])
        for score, a, b in [(104, 4, 5), (147, 27, 39), (150, 22, 20)]
    ]


def test_exact_recorded_variance_attribution():
    result = observed_pair(history(), 1, 2)
    assert result["covariance_effect"] == pytest.approx(382.333333333)
    assert result["team_variance_share_pct"] == pytest.approx(57.72521389)
    assert result["r"] == pytest.approx(0.9274533721)


def test_subset_cannot_be_attributed_to_full_season_and_missing_is_not_zero():
    rows = history()
    rows[1]["contributions"] = None
    result = observed_pair(rows, 1, 2)
    assert result["status"] == "incomplete_lineup_history"
    assert result["covariance_effect"] is None
    assert result["team_variance_share_pct"] is None
    assert observed_pair(history()[:2], 1, 2)["covariance_effect"] is None


def test_season_centering_and_exact_variance_identity():
    rows = [
        (2024, 1, 5),
        (2024, 2, 4),
        (2024, 3, 3),
        (2025, 101, 105),
        (2025, 102, 104),
        (2025, 103, 103),
    ]
    result = historical_summary(rows)
    assert result["r"] == pytest.approx(-1)
    assert result["covariance_effect"] == pytest.approx(-2)
    assert result["variance"] == pytest.approx(0)
    assert result["variance_share_pct"] is None  # zero denominator, not 0%
    assert result["variance_change_pct"] == pytest.approx(-100)
    assert historical_summary([(2025, 0, 1), (2025, 1, 2)])["sd"] is None


def fixture():
    db = duckdb.connect(config={"threads": 1})
    db.execute("create schema analytics")
    db.execute("create table analytics.players (espn_id varchar, gsis_id varchar)")
    db.execute("insert into analytics.players values ('1','qb'),('2','wr'),('3','bench')")
    db.execute("""create table analytics.team_player_games
        (game_id varchar, season int, week int, team varchar, player_id varchar,
         league_points double, starter_game boolean)""")
    rows = []
    for season, club in [(2020, "LA"), (2025, "LA"), (2025, "LV"), (2026, "LA")]:
        for week in range(1, 5):
            for pid, points in [("qb", week), ("wr", 2 * week)]:
                rows.append(
                    (
                        f"{season}_{club}_{week}",
                        season,
                        week,
                        club,
                        pid,
                        points if club == "LA" else -points,
                        True,
                    )
                )
    db.executemany("insert into analytics.team_player_games values (?,?,?,?,?,?,?)", rows)
    players = [
        SimpleNamespace(
            espn_id=pid,
            player_display_name=name,
            position=pos,
            espn_team=club,
            lineup_slot=slot,
            owner_team_id=1,
        )
        for pid, name, pos, club, slot in [
            (1, "Q", "QB", "LAR", "QB"),
            (2, "W", "WR", "LA", "WR"),
            (3, "Bench", "WR", "LA", "BE"),
        ]
    ]
    snapshot = SimpleNamespace(season=2026, players=players)
    result = {
        "teams": [
            dict(
                team_id=1,
                results={"volatility": (662 + 1 / 3) ** 0.5},
                history=history(),
                current={"total": 999999},
            )
        ]
    }
    report = dict(season=2026, earliest_season=2013)
    return db, snapshot, result, report


def test_duckdb_identity_join_cutoff_club_and_no_forecast_mutation():
    db, snapshot, result, report = fixture()
    original = deepcopy(result)
    try:
        attach_roster_correlations(result, snapshot, db, report)
        risk = result["teams"][0].pop("correlation_risk")
        assert result == original
        assert len(risk["connections"]) == 1  # bench excluded
        h = risk["connections"][0]["historical"]
        assert h["n"] == 4  # excludes 2020, 2026 and different-club records
        assert h["seasons"] == [2025]
        assert h["variance"] == pytest.approx(15)
        assert h["covariance_effect"] == pytest.approx(20 / 3)
        assert h["variance_share_pct"] == pytest.approx(100 * 4 / 9)
        # Poison future outcomes and verify the historical result is unchanged.
        db.execute("update analytics.team_player_games set league_points=999999 where season=2026")
        attach_roster_correlations(result, snapshot, db, report)
        assert result["teams"][0]["correlation_risk"] == risk
    finally:
        db.close()


def test_ambiguous_identity_is_not_guessed_but_observed_accounting_survives():
    db, snapshot, result, report = fixture()
    try:
        db.execute("insert into analytics.players values ('2','different_player')")
        attach_roster_correlations(result, snapshot, db, report)
        risk = result["teams"][0]["correlation_risk"]
        assert risk["unmapped_starters"] == 1
        assert risk["connections"][0]["historical"]["status"] == "unmapped_identity"
        assert risk["connections"][0]["historical"]["r"] is None
        assert risk["observed"]["covariance_effect"] == pytest.approx(382 + 1 / 3)
    finally:
        db.close()


def test_three_connected_players_observed_terms_are_unique():
    db, snapshot, result, report = fixture()
    try:
        snapshot.players[2].lineup_slot = "TE"
        snapshot.players[2].position = "TE"
        for row, value in zip(result["teams"][0]["history"], [1, 2, 3], strict=True):
            row["contributions"].append(dict(espn_id=3, points=value))
        attach_roster_correlations(result, snapshot, db, report)
        risk = result["teams"][0]["correlation_risk"]
        assert len(risk["connections"]) == 3
        assert risk["observed"]["covariance_effect"] == pytest.approx(415 + 1 / 3)
        # No pooling pairwise historical samples into a historical roster metric.
        assert "historical_variance" not in risk
    finally:
        db.close()
