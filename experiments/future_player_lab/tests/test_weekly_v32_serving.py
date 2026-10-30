from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v31.artifacts import seal, write
from experiments.future_player_lab.campaigns.weekly_v32 import live, official, shadow, uncertainty


def test_official_page_rejects_wrong_week_even_when_request_url_is_right():
    identity = pd.DataFrame(
        dict(player_id=["a"], player_name=["Joe Smith"], team=["DAL"], position=["RB"])
    )
    html = """<title>Official NFL Injury Report for Players - Week 5 of the 2026 Season</title>
<section class="nfl-o-injury-report__unit">
<span class="nfl-c-matchup-strip__team-abbreviation"> DAL </span>
<span class="nfl-c-matchup-strip__team-abbreviation"> TB </span>
<tbody><tr><td>Joe Smith</td><td>RB</td><td>Knee</td>
<td>Did Not Participate</td><td>Out</td></tr></tbody><tbody></tbody>"""
    with pytest.raises(ValueError, match="requested season/week"):
        official.parse(html, 2026, 6, identity, "2026-10-13T12:00:00Z")
    frame, teams = official.parse(html, 2026, 5, identity, "2026-10-13T12:00:00Z")
    assert teams == ["DAL", "TB"]
    assert frame.player_id.tolist() == ["a"]
    assert frame.report_status.tolist() == ["Out"]
    ambiguous = pd.concat([identity, identity.assign(player_id="b")])
    assert (
        official.parse(html, 2026, 5, ambiguous, "2026-10-13T12:00:00Z")[0].player_id.isna().all()
    )


def report_capture(
    path, observed="2026-10-13T12:00:00Z", status="usable_weekly_report", usable=None
):
    path.mkdir()
    write(
        path / "summary.json",
        dict(observed_at=observed, status=status, season=2026, week=6, teams=["DAL"]),
    )
    pd.DataFrame(
        dict(
            player_id=["a"],
            season=2026,
            week=6,
            team="DAL",
            usable_at=usable or observed,
            report_status="Out",
            practice_status=None,
            source="official_nfl_weekly_page",
            evidence_kind="official_weekly_report",
        )
    ).to_parquet(path / "reports.parquet", index=False)
    seal(path, dict(status=status))
    return path


@pytest.mark.parametrize(
    "at,known",
    [
        ("2026-10-13T11:59:00Z", False),
        ("2026-10-13T12:00:00Z", True),
        ("2026-10-15T01:00:00Z", False),
    ],
)
def test_official_evidence_must_be_observed_and_fresh(tmp_path, at, known):
    path = report_capture(tmp_path / "capture")
    result, audit = live.usable_reports([path], 2026, 6, pd.Timestamp(at).to_pydatetime())
    assert bool(len(result)) == known
    assert audit[0]["admitted"] == known


def test_provider_status_does_not_qualify_as_official_and_backdating_is_rejected(tmp_path):
    path = report_capture(tmp_path / "provider", status="captured_research_observation")
    now = datetime(2026, 10, 13, 13, tzinfo=UTC)
    result, _ = live.usable_reports([path], 2026, 6, now)
    assert result.empty
    path = report_capture(tmp_path / "backdated", usable="2026-10-12T12:00:00Z")
    with pytest.raises(ValueError, match="backdated"):
        live.usable_reports([path], 2026, 6, now)


def test_workload_features_use_prior_scheduled_rows_and_keep_unknowns():
    history = pd.DataFrame(dict(player_id=["a"] * 5, season=2026, week=range(1, 6)))
    for c in ("snaps", "snap_share", "played", *live.data.OPPORTUNITIES):
        history[f"label_{c}"] = [99, 2, np.nan, 0, 6]
    panel = pd.DataFrame(dict(player_id=["a", "b"], season=2026, week=6))
    out = live.attach_workloads(panel, history)
    assert out.prior4_snaps.iloc[0] == pytest.approx(8 / 3)
    assert np.isnan(out.prior4_snaps.iloc[1])
    with pytest.raises(ValueError, match="future outcomes"):
        live.attach_workloads(panel, history.assign(week=6))


def test_uncertainty_uses_earlier_errors_and_preserves_means():
    f = pd.DataFrame(
        dict(
            season=2024,
            position="RB",
            prediction=np.arange(1000) / 50,
            target_points=np.arange(1000) / 40,
        )
    )
    with pytest.raises(ValueError, match="strictly earlier"):
        uncertainty.fit(f, 2024)
    cal = uncertainty.fit(f, 2025)
    t = pd.DataFrame(dict(season=2025, position="RB", prediction=[4.0, 0.0], report_out=[0, 1]))
    draws = uncertainty.samples(t, cal)
    np.testing.assert_allclose(draws.mean(axis=1), t.prediction)
    assert (np.diff(draws, axis=1) >= 0).all()
    assert (draws[1] == 0).all()
    with pytest.raises(ValueError, match="future outcomes"):
        uncertainty.samples(t.assign(season=2024), cal)


def test_pregame_window_never_admits_post_cutoff_or_too_early():
    schedule = pd.DataFrame(
        dict(
            game_id=["early", "due", "late"],
            kickoff=pd.to_datetime(["2026-10-18T17:31Z", "2026-10-18T17:15Z", "2026-10-18T16:59Z"]),
        )
    )
    result = shadow.game_window(schedule, pd.Timestamp("2026-10-18T16:00Z"))
    assert result.game_id.tolist() == ["due"]


def test_late_missing_evidence_requires_locked_fallback():
    frame = pd.DataFrame(dict(player_id=["a"], season=2026, week=6, team="DAL", opponent="TB"))
    for c in live.data.OPPORTUNITIES:
        frame[f"prior4_{c}"] = 1
    empty, _ = live.usable_reports([], 2026, 6, datetime.now(UTC))
    with pytest.raises(ValueError, match="reuse the locked early"):
        shadow.make_predictions(
            Path("unused"), frame, empty, datetime.now(UTC), "pregame_60m", {"DAL"}
        )


def test_trackers_are_isolated_from_alternate_registries(monkeypatch):
    from experiments.future_player_lab.campaigns import rescore

    monkeypatch.setattr(shadow, "track", lambda *a, **k: pytest.fail("Tracked alternate registry"))
    context = dict(snapshot=None, registry=Path("elsewhere"), allow_network=False)
    assert rescore.run_trackers("refresh", context, isolated=True) == {}


def test_tracker_failure_does_not_block_other_trackers(monkeypatch, tmp_path):
    from experiments.future_player_lab.campaigns import rescore
    from experiments.future_player_lab.campaigns.weekly_v4 import prospective as v4
    from experiments.future_player_lab.campaigns.weekly_v41 import track as v41

    def boom(*a, **k):
        raise RuntimeError("v3.2 failed")

    monkeypatch.setattr(shadow, "track", boom)
    monkeypatch.setattr(v4, "track", lambda *a, **k: pytest.fail("Child ran after failure"))
    monkeypatch.setattr(v41, "track", lambda *a, **k: {"status": "ok", "registry": k["registry"]})
    configs = rescore.campaign_configs()
    for campaign_id in ("opening_drive_v32", "opening_drive_v4"):
        configs[campaign_id]["tracker"]["requires"] = []
    context = dict(snapshot=None, registry=tmp_path, allow_network=False)
    results = rescore.run_trackers("refresh", context, configs=configs)
    assert list(results)[-2:] == ["opening_drive_v41", "opening_drive_v32"]
    assert results["opening_drive_v32"]["status"] == "failed"
    assert results["opening_drive_v41"] == {"status": "ok", "registry": tmp_path}
    assert (
        rescore.run_trackers("refresh", context, configs=configs, filters=["opening_drive_v1"])
        == {}
    )


def test_v41_is_the_default_campaign():
    from engine.chairstats.application import DEFAULT_CAMPAIGN

    assert DEFAULT_CAMPAIGN == "opening_drive_v41"


def test_late_update_retains_fine_timestamp_and_original_population():
    early = pd.DataFrame(
        dict(
            player_id=["a", "b"],
            prediction=[1.0, 2.0],
            report_usable_at=pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ms]"),
        )
    )
    stamp = pd.Timestamp("2026-10-18T16:00:00.123456Z")
    late = pd.DataFrame(dict(player_id=["a"], prediction=[0.0], report_usable_at=[stamp]))
    out = shadow.merge_game_update(early, late)
    assert out.player_id.tolist() == ["a", "b"]
    assert out.prediction.tolist() == [0.0, 2.0]
    assert out.report_usable_at.iloc[0] == stamp
    with pytest.raises(ValueError, match="locked population"):
        shadow.merge_game_update(early, late.assign(player_id="new"))
