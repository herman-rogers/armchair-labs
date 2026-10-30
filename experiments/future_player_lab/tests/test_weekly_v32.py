import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v32 import data, modeling
from experiments.future_player_lab.campaigns.weekly_v32.capture import parse


def reports():
    return pd.DataFrame(
        dict(
            player_id=["a", "a", "b"],
            season=[2025] * 3,
            week=[1, 2, 2],
            team=["ABC"] * 3,
            usable_at=pd.to_datetime(
                ["2025-09-01T12:00Z", "2025-09-10T18:00Z", "2025-09-10T18:00Z"]
            ),
            report_status=["Out", "Questionable", "Out"],
            practice_status=["Did Not Participate"] * 3,
            source=["test"] * 3,
        )
    )


def test_report_join_requires_current_week_team_and_timestamp():
    frame = pd.DataFrame(
        dict(
            player_id=["a"] * 3,
            season=2025,
            week=2,
            team=["ABC", "ABC", "XYZ"],
            cutoff=pd.to_datetime(["2025-09-10T12:00Z", "2025-09-10T19:00Z", "2025-09-10T19:00Z"]),
        )
    )
    out = data.attach_reports(frame, reports(), "cutoff")
    assert out.report_known.tolist() == [0, 1, 0]
    assert np.isnan(out.report_out.iloc[0])
    assert out.report_questionable.iloc[1] == 1
    assert (pd.to_datetime(out.report_usable_at, utc=True).dropna() <= frame.cutoff.iloc[1]).all()


def test_workload_lags_do_not_read_this_game_or_future_game():
    f = pd.DataFrame(dict(player_id=["a"] * 3, row_id=range(3), season=2025, week=[1, 2, 3]))
    for c in ["snaps", "snap_share", "played", *data.OPPORTUNITIES]:
        f[f"label_{c}"] = [1.0, 2.0, 999.0]
    a = data.lag_workloads(f)
    f.loc[2, [c for c in f if c.startswith("label_")]] = 0
    b = data.lag_workloads(f)
    pd.testing.assert_frame_equal(a[list(data.WORK_FEATURES)], b[list(data.WORK_FEATURES)])
    assert a.prior4_snaps.iloc[2] == 1.5
    assert np.isnan(a.prior4_snaps.iloc[0])


def test_teammate_context_excludes_own_report_and_current_labels():
    f = pd.DataFrame(
        dict(
            season=2025,
            week=2,
            team=["ABC", "ABC"],
            report_out=[1.0, 0.0],
            report_questionable=[0.0, 0.0],
            practice_dnp=[1.0, 0.0],
            prior4_attempts=0.0,
            prior4_carries=[10.0, 2.0],
            prior4_targets=[2.0, 1.0],
        )
    )
    a = data.teammate_features(f)
    assert a.room_out_usage.tolist() == [0, 12]
    assert a.room_out_players.tolist() == [0, 1]
    assert a.room_other_usage.tolist() == [3, 12]


@pytest.mark.parametrize("fraction", [0, 0.5, 1])
def test_redistribution_conserves_budget_and_does_not_double_apply_probability(fraction):
    f = pd.DataFrame(dict(season=2025, week=1, team=["ABC"] * 3))
    before = np.array([[0, 10, 2], [0, 4, 1], [0, 2, 1]], float)
    after = before.copy()
    after[0] = 0
    p = np.array([0.0, 0.8, 0.2])
    result, receipts = modeling.redistribute(f, before, after, p, fraction)
    assert (result >= after).all()
    np.testing.assert_array_equal(result[0], 0)
    np.testing.assert_array_equal(result[2], after[2])
    np.testing.assert_allclose(
        result.sum(axis=0), after.sum(axis=0) + fraction * (before - after).sum(axis=0)
    )
    active = np.divide(result, p[:, None], out=np.zeros_like(result), where=p[:, None] > 0)
    np.testing.assert_allclose(active * p[:, None], result)
    assert all(r["unassigned"] >= -1e-8 for r in receipts)


def sample(year, n=240):
    rng = np.random.default_rng(year)
    f = pd.DataFrame(
        dict(
            season=year,
            position="RB",
            base_prediction=rng.uniform(1, 10, n),
            target_points=rng.uniform(0, 15, n),
            label_played=np.arange(n) % 2,
            report_out=0.0,
            recent4_points_mean=5.0,
            player_history_mean=5.0,
        )
    )
    for c in [*data.WORK_FEATURES, *data.INJURY_FEATURES, *data.TEAM_FEATURES]:
        if c not in f:
            f[c] = rng.uniform(0, 1, n)
    for c in data.OPPORTUNITIES:
        f[f"label_{c}"] = rng.uniform(0, 10, n)
    return f


def test_models_and_selection_reject_future_labels_and_inference_is_label_free():
    with pytest.raises(ValueError, match="precede"):
        modeling.fit(sample(2021), 2021, ["base_prediction"])
    bundle = modeling.fit(sample(2020), 2021, ["base_prediction"])
    f = sample(2021)
    a, _ = modeling.predict(bundle, f)
    f = f.drop(columns=[c for c in f if c.startswith("label_") or c == "target_points"])
    b, _ = modeling.predict(bundle, f)
    for m in a:
        np.testing.assert_array_equal(a[m], b[m])
    f.loc[0, "report_out"] = 1
    c, d = modeling.predict(bundle, f)
    assert c["mixture_team"][0] == c["workload_team"][0] == c["mixture_team_blend"][0] == 0
    assert d["team"]["probability"][0] == 0
    with pytest.raises(ValueError, match="Outcome"):
        modeling.matrix(sample(2021), ["label_played"])
    h = pd.DataFrame(dict(season=[2021], method=["v3"], prediction=[1.0], target_points=[0.0]))
    with pytest.raises(ValueError, match="precede"):
        modeling.choose(h, 2021)


def test_current_provider_capture_never_backdates_or_infers_health():
    teams = [dict(id=str(i), injuries=[]) for i in range(32)]
    teams[0]["injuries"] = [
        dict(
            status="Out",
            date="2026-09-01T00:00Z",
            athlete=dict(
                displayName="Example",
                links=[dict(href="https://www.espn.com/nfl/player/_/id/123/example")],
                team={"abbreviation": "ABC"},
                position={"abbreviation": "RB"},
            ),
        )
    ]
    identity = pd.DataFrame(dict(espn_id=["123"], gsis_id=["player"]))
    out = parse(
        dict(season={"year": 2026}, injuries=teams), identity, "2026-10-13T20:00:00+00:00", 6
    )
    assert len(out) == 1
    assert out.player_id.iloc[0] == "player"
    assert out.usable_at.iloc[0] == "2026-10-13T20:00:00+00:00"
    assert out.provider_updated_at.iloc[0] == "2026-09-01T00:00Z"
    with pytest.raises(ValueError, match="32"):
        parse(dict(season={"year": 2026}, injuries=teams[:-1]), identity, "2026-10-13T20:00Z", 6)


def test_v31_automatic_tracking_is_paused_without_mutating_captures(monkeypatch):
    from experiments.future_player_lab.campaigns import rescore
    from experiments.future_player_lab.campaigns.weekly_v31 import prospective

    configs = {
        cid: config
        for cid, config in rescore.campaign_configs().items()
        if cid == "opening_drive_v31"
    }
    assert configs["opening_drive_v31"]["status"] == "paused"
    monkeypatch.setattr(
        prospective, "track", lambda *a, **k: pytest.fail("Paused tracking executed")
    )
    results = rescore.run_trackers(
        "refresh", dict(snapshot=None, registry=rescore.REGISTRY), configs=configs
    )
    assert results["opening_drive_v31"]["status"] == "paused"
