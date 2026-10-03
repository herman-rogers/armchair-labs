from copy import deepcopy

from engine.data.releases import write_json
from engine.espn.forecast_log import accuracy
from engine.espn.sync import LeagueSnapshot, LineupEntry, ScheduleEntry, TeamState, WeekLineups
from engine.metrics.weekly_lineup import weekly_forecast


def player(pid, position="QB", points=10, projected=999, slot=None):
    return LineupEntry(
        pid, f"Player {pid}", position, slot or position, points, projected, "BUF", False
    )


def fixture():
    snap = LeagueSnapshot(
        "2026-09-22T12:00:00+00:00",
        1,
        "Fixture",
        2026,
        3,
        1,
        regular_season_weeks=14,
        teams=[
            TeamState(
                i,
                f"Team {i}",
                "Owner",
                1,
                1,
                100,
                schedule=[
                    ScheduleEntry(1, 3 - i, 100 if i == 1 else 90, "W" if i == 1 else "L"),
                    ScheduleEntry(2, 3 - i, 100 if i == 1 else 90, "W" if i == 1 else "L"),
                ],
            )
            for i in (1, 2)
        ],
        week_lineups=[
            WeekLineups(1, 1, 2, 100, 90, home_lineup=[player(3, "K", 4)], away_lineup=[]),
            WeekLineups(2, 1, 2, 100, 90, home_lineup=[player(3, "K", 8)], away_lineup=[]),
            WeekLineups(
                3,
                1,
                2,
                999,
                0,
                home_projected=999,
                away_projected=1,
                home_lineup=[player(1), player(3, "K", 1000)],
                away_lineup=[player(2)],
            ),
        ],
    )
    ranks = dict(
        version="test",
        horizon="next4",
        report=dict(season=2026, through_week=2, generated_at="2026-09-22T10:00:00+00:00"),
        rankings=[
            dict(
                player_id=str(i),
                player_display_name=f"Player {i}",
                position="QB",
                team="BUF",
                prediction=points,
                scheduled_games=4,
                schedule_known=True,
            )
            for i, points in [(1, 80), (2, 120)]
        ],
    )
    schedule = [
        dict(
            season=2026,
            week=w,
            game_type="REG",
            home_team="BUF",
            away_team="NYJ",
            gameday="2026-09-24",
        )
        for w in range(3, 7)
    ]
    return snap, ranks, {1: "1", 2: "2"}, schedule


def test_model_points_and_prior_only_special_teams_ignore_espn_predictions():
    args = fixture()
    result = weekly_forecast(*args)
    game = result["matchups"][0]
    assert game["home"]["model_projection"] == 26  # 80/4 + mean(4,8)
    assert game["away"]["model_projection"] == 30
    assert game["favorite_team_id"] == 2  # ESPN says the opposite.
    assert "espn_projection" not in game["home"]
    assert "lineup" not in game["home"]
    assert all("projected_points" not in p for p in game["home"]["forecasts"])
    args[0].week_lineups[-1].home_projected = -99999
    args[0].week_lineups[-1].away_projected = 99999
    for entry in args[0].week_lineups[-1].home_lineup:
        entry.projected_points = 99999
    args[0].week_lineups[-1].home_score = 50000
    args[0].week_lineups[-1].home_lineup[-1].points = -50000
    changed = weekly_forecast(*args)["matchups"][0]
    assert changed["home"]["model_projection"] == 26
    assert changed["favorite_team_id"] == 2


def test_missing_player_and_stale_model_withhold_totals():
    args = fixture()
    args[1]["rankings"].pop()
    game = weekly_forecast(*args)["matchups"][0]
    assert game["away"]["model_projection"] is None
    assert game["favorite_team_id"] is None
    args[1]["report"]["through_week"] = 1
    game = weekly_forecast(*args)["matchups"][0]
    assert game["home"]["model_projection"] is None


def test_dedicated_weekly_release_mismatch_withholds_points():
    product = weekly_product()
    product["manifest"]["analysis"]["version"] = "different-release"
    game = weekly_forecast(*fixture(), weekly=product)["matchups"][0]
    assert game["home"]["model_projection"] is None
    assert game["away"]["model_projection"] is None
    assert game["favorite_team_id"] is None


def test_bench_excluded_zero_preserved_and_bye_zero():
    args = fixture()
    args[0].week_lineups[-1].home_lineup.append(player(99, points=10000, slot="BE"))
    args[1]["rankings"][0]["prediction"] = 0
    game = weekly_forecast(*args)["matchups"][0]
    assert game["home"]["model_projection"] == 6
    args[0].week_lineups[-1].home_lineup[-2].on_bye = True
    assert weekly_forecast(*args)["matchups"][0]["home"]["model_projection"] == 0


def test_accuracy_excludes_late_generated_picks_and_never_backfills_week_one(tmp_path):
    snap, ranks, ids, schedule = fixture()
    forecast = weekly_forecast(snap, ranks, ids, schedule)
    # Fixture with a verified pregame capture for Week 2, generated before the deadline.
    forecast.update(
        league_id=1,
        requested_week=2,
        through_week=1,
        pregame_deadline="2026-09-17T00:00:00+00:00",
        captured_at="2026-09-16T10:00:00+00:00",
        published_at="2026-09-16T09:00:00+00:00",
        generated_at="2026-09-16T11:00:00+00:00",
    )
    root = tmp_path / "1" / "2026"
    root.mkdir(parents=True)
    late = deepcopy(forecast)
    late["requested_week"] = 1
    late["through_week"] = 0
    late["generated_at"] = "2026-09-25T00:00:00+00:00"
    write_json(root / "late.json", late)
    report = accuracy(snap, tmp_path)
    assert report["evaluated"] == 0 and report["accuracy"] is None
    write_json(root / "valid.json", forecast)
    report = accuracy(snap, tmp_path)
    assert report["point_mae"] == 67
    assert report["point_teams"] == 2
    assert report["evaluated"] == 1 and report["correct"] == 0  # Away favored; home won.
    assert report["games"][0]["status"] == "No verified pregame forecast"
    forecast["source"] = "nextgen_weekly_model"
    write_json(root / "valid.json", forecast)
    assert accuracy(snap, tmp_path)["evaluated"] == 1
    forecast["source"] = "espn_observations"
    write_json(root / "valid.json", forecast)
    assert accuracy(snap, tmp_path)["evaluated"] == 0
    forecast["source"] = "nextgen_weekly_reference"
    forecast["published_at"] = "2026-09-18T00:00:00+00:00"
    write_json(root / "valid.json", forecast)
    assert accuracy(snap, tmp_path)["evaluated"] == 0


def weekly_product():
    return dict(
        manifest=dict(
            version="weekly",
            season=2026,
            through_week=2,
            generated_at="2026-09-22T11:00:00+00:00",
            analysis={"version": "test"},
        ),
        predictions=[
            dict(
                player_id=str(i),
                player_display_name=f"Player {i}",
                position="QB",
                season=2026,
                target_week=w,
                through_week=w - 1,
                prediction=points,
                recipe="current",
                evidence_status="reference",
            )
            for w in (1, 2, 3)
            for i, points in [(1, 35), (2, 20)]
        ],
    )


def test_dedicated_weekly_points_ignore_next_four_and_wrong_origin():
    args = fixture()
    product = weekly_product()
    result = weekly_forecast(*args, weekly=product)
    assert result["source"] == "nextgen_weekly_model"
    assert result["evidence_status"] == "reference"
    assert result["matchups"][0]["home"]["model_projection"] == 41
    assert result["matchups"][0]["favorite_team_id"] == 1
    args[1]["rankings"][0]["prediction"] = 99999
    assert weekly_forecast(*args, weekly=product)["matchups"] == result["matchups"]
    product["predictions"][-1]["through_week"] = 3
    game = weekly_forecast(*args, weekly=product)["matchups"][0]
    assert game["away"]["model_projection"] is None
    assert game["favorite_team_id"] is None


def test_replay_does_not_use_future_scores_or_call_it_pregame():
    from engine.metrics.weekly_replay import season_replay

    snap, _, ids, schedule = fixture()
    # Include actual skill-player lineup selections in both past weeks.
    for game in snap.week_lineups[:2]:
        game.home_lineup.append(player(1))
        game.away_lineup.append(player(2))
    product = weekly_product()
    result = season_replay(snap, product, ids, schedule)
    assert result["total"] == 2 and result["evaluated"] == 1
    assert result["games"][0]["projected_home"] is None  # No Week 1 K history.
    assert result["games"][1]["projected_home"] == 39
    assert "Not a record of pregame published picks" in result["method"]
    snap.week_lineups[-1].home_lineup[-1].points = 99999
    product["predictions"][-1]["prediction"] = 99999
    assert season_replay(snap, product, ids, schedule) == result
