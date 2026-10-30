import copy
import json

import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.horizon_projection.curve import (
    durable_signal,
    future_curve,
)
from experiments.future_player_lab.campaigns.horizon_projection.publication import (
    actual_points,
    projected_rows,
)
from experiments.future_player_lab.campaigns.horizon_projection.research import select

# Publication only warns; the coherence checks themselves still raise.
from engine.chairstats.horizons import _validate as validate_opening_drive_release


def test_earned_points_use_campaign_scoring_and_exclude_future_and_postseason():
    rows = pd.DataFrame(
        dict(
            player_id=["a"] * 4,
            position="QB",
            season=2026,
            week=[1, 2, 3, 2],
            season_type=["REG", "REG", "REG", "POST"],
            passing_yards=[100, 200, 99999, 99999],
            passing_tds=1,
            passing_interceptions=1,
            rushing_yards=20,
            rushing_tds=0,
            receiving_yards=0,
            receptions=0,
            receiving_tds=0,
        )
    )
    assert actual_points(rows, 2026, 2) == {"a": 20}
    rows.loc[rows.week.eq(1), "passing_yards"] = float("nan")
    with pytest.raises(ValueError, match="Missing scoring components"):
        actual_points(rows, 2026, 2)


def test_curve_keeps_current_week_and_decays_across_calendar_bye():
    curve = future_curve([3, 4, 6], 3, 0, 10, 18, 2)
    assert [g["week"] for g in curve] == [3, 4, 6]
    assert curve[0]["points"] == 0
    assert curve[1]["points"] == pytest.approx(10 + 8 * 2**-0.5)
    assert curve[2]["points"] == pytest.approx(10 + 8 * 2**-1.5)
    assert future_curve([3, 4], 3, 20, 10, 18, 0)[1]["points"] == 10
    with pytest.raises(ValueError, match="past weeks"):
        future_curve([2, 3], 3, 20, 10, 18, 2)


def test_same_game_injury_and_teammate_effects_do_not_extend_to_tail():
    early = dict(prediction=0, report_known=1)
    assert durable_signal(early, 12) == 12
    assert durable_signal(dict(prediction=24, room_out_usage=5), 12) == 12
    assert durable_signal(dict(prediction=24, report_known=0), 12) == 24


def test_selector_never_uses_current_or_future_season_outcomes():
    rows = []
    for season in (2021, 2022, 2023):
        for h in (0, 2, 4, 8, 10000):
            rows.append(
                dict(
                    season=season,
                    position="RB",
                    method=f"v32_decay_{h}",
                    actual=10,
                    prediction=10 if h == 2 else 20,
                )
            )
    frame = pd.DataFrame(rows)
    assert select(frame, 2023, "RB") == 2
    frame.loc[frame.season.eq(2023), "actual"] = 1000000
    assert select(frame, 2023, "RB") == 2
    with pytest.raises(ValueError, match="strictly earlier"):
        select(frame, 2021, "RB")


def fixture():
    positions = ["QB", "RB", "WR", "TE"]
    players = pd.DataFrame(
        [
            dict(
                player_id=p,
                player_name=p,
                position=p,
                season=2026,
                week=3,
                team="B" if p == "TE" else "A",
                is_bye=int(p == "TE"),
                prediction=0 if p == "TE" else 20,
            )
            for p in positions
        ]
    )
    context = dict(
        season=2026,
        origin=2,
        schedule=pd.DataFrame(
            [
                dict(week=w, home_team=t, away_team="OTHER")
                for t, weeks in [("A", [1, 2, 3, 4, 6]), ("B", [1, 2, 4, 6])]
                for w in weeks
            ]
        ),
        anchors={
            p: dict(
                predictions={"points": 20 if p == "TE" else 30},
                scheduled_games=2 if p == "TE" else 3,
            )
            for p in positions
        },
        early={
            p: dict(prediction=0 if p == "TE" else 20, base_prediction=12, report_known=0)
            for p in positions
        },
        panel={p: dict(v2_weekly_mean=10) for p in positions},
        actuals={p: 15 for p in positions},
        policy=dict(half_lives={p: 2 for p in positions}),
    )
    return players, context


def test_late_change_flows_once_into_totals_without_changing_future_signal():
    players, context = fixture()
    early, _ = projected_rows(players, context)
    players.loc[players.player_id.eq("RB"), "prediction"] = 0
    late, _ = projected_rows(players, context)
    for horizon in early:
        a = next(r for r in early[horizon] if r["player_id"] == "RB")
        b = next(r for r in late[horizon] if r["player_id"] == "RB")
        assert a["predictions"]["points"] - b["predictions"]["points"] == pytest.approx(20)
        assert a["projection"]["future_weeks"][1:] == b["projection"]["future_weeks"][1:]
        assert b["points_per_scheduled_game"] == b["predictions"]["points"] / b["scheduled_games"]
    r = next(r for r in late["remaining"] if r["player_id"] == "TE")
    assert r["predictions"]["points"] == 20  # Bye does not turn future games into zeros.
    s = next(r for r in late["season"] if r["player_id"] == "TE")
    assert s["predictions"]["points"] == 35
    assert s["scheduled_games"] == 4


def release(tmp_path):
    players, context = fixture()
    projected, _ = projected_rows(players, context)
    models = []
    for position in ("QB", "RB", "WR", "TE"):
        for horizon in ("weekly", "remaining", "season"):
            model_id = f"{position.lower()}_{horizon}"
            path = tmp_path / f"{model_id}.json"
            if horizon == "weekly":
                rows = [
                    dict(
                        player_id=position,
                        scheduled_games=0 if position == "TE" else 1,
                        predictions={"points": 0 if position == "TE" else 20},
                    )
                ]
            else:
                rows = [r for r in projected[horizon] if r["position"] == position]
            path.write_text(
                json.dumps(
                    dict(
                        rows=rows,
                        capture_hashes={"capture": "abc"},
                        forecast_kind="coherent_schedule_projection",
                    )
                )
            )
            models.append(
                dict(
                    model_id=model_id,
                    position=position,
                    horizon=horizon,
                    forecasts=[dict(season=2026, origin_week=2, path=path.name)],
                    artifacts=dict(projection_policy={}, projection_evaluation={}),
                )
            )
    return dict(campaign_id="opening_drive_v32", horizon_contract=dict(version=1), models=models)


def test_release_rejects_missing_horizons_and_incoherent_totals(tmp_path):
    manifest = release(tmp_path)
    validate_opening_drive_release(tmp_path, manifest)
    incomplete = copy.deepcopy(manifest)
    incomplete["models"].pop()
    with pytest.raises(ValueError, match="all three horizons"):
        validate_opening_drive_release(tmp_path, incomplete)
    path = tmp_path / "rb_remaining.json"
    payload = json.loads(path.read_text())
    payload["rows"][0]["predictions"]["points"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Remaining total"):
        validate_opening_drive_release(tmp_path, manifest)


@pytest.mark.parametrize("failure", ["stale", "inherited", "average", "weekly", "capture"])
def test_release_rejects_stale_or_disconnected_views(tmp_path, failure):
    manifest = release(tmp_path)
    model = next(m for m in manifest["models"] if m["model_id"] == "qb_remaining")
    path = tmp_path / "qb_remaining.json"
    payload = json.loads(path.read_text())
    if failure == "stale":
        model["forecasts"][0]["origin_week"] = 1
    elif failure == "inherited":
        payload["forecast_kind"] = "inherited"
    elif failure == "average":
        payload["rows"][0]["points_per_scheduled_game"] += 1
    elif failure == "weekly":
        path = tmp_path / "qb_weekly.json"
        payload = json.loads(path.read_text())
        payload["rows"][0]["predictions"]["points"] += 1
    elif failure == "capture":
        payload["capture_hashes"] = {"capture": "different"}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_opening_drive_release(tmp_path, manifest)


def test_publication_warns_but_never_blocks(tmp_path):
    from engine.chairstats.horizons import validate_opening_drive_release as release_check

    manifest = dict(campaign_id="any_new_version", horizon_contract={"version": 1}, models=[])
    with pytest.warns(UserWarning, match="all three horizons"):
        release_check(tmp_path, manifest)
    release_check(tmp_path, dict(campaign_id="no_contract", models=[]))  # not checked
