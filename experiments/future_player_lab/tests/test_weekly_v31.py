import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns import rescore as driver
from experiments.future_player_lab.campaigns.weekly_v3 import data, modeling
from experiments.future_player_lab.campaigns.weekly_v3.rescore import _rows
from experiments.future_player_lab.campaigns.weekly_v31 import calibration, prospective
from experiments.future_player_lab.campaigns.weekly_v31.artifacts import (
    digest,
    publish_directory,
    seal,
    timestamp,
    verify,
    write,
)

from engine.chairstats.store import publish_campaign
from engine.metrics import weekly_distribution


def history(n=1000):
    return pd.DataFrame(
        dict(
            season=2025,
            position="QB",
            prediction=7.0,
            target_points=np.linspace(-5, 19, n),
            q05=-1.0,
            q10=2.0,
            q90=12.0,
            q95=15.0,
        )
    )


def test_calibration_improves_known_undercoverage_preserves_mean_and_quantiles():
    past = history()
    bundle = calibration.fit(past, 2026)
    future = history().assign(season=2026)
    corrected = calibration.apply(future, bundle)
    assert calibration.score(corrected, adjusted=True)["coverage_80"] >= 0.8
    assert calibration.score(corrected)["coverage_80"] < 0.5
    pd.testing.assert_frame_equal(corrected[future.columns], future)
    assert (corrected.lower_90 <= corrected.lower_80).all()
    assert (corrected.upper_90 >= corrected.upper_80).all()
    assert set(corrected.group_80) == {"QB/5_to_10"}


def test_past_only_window_fallback_byes_and_finite_sample_rank():
    old = history().assign(season=2010, target_points=10000)
    past = pd.concat([history(), old], ignore_index=True)
    bundle = calibration.fit(past, 2026)
    assert bundle["calibration_rows"] == 1000
    assert bundle["groups"]["QB"]["80"]["rank"] == 801
    future = history(3).assign(season=2026, prediction=20)
    future["is_bye"] = [0, 1, 0]
    result = calibration.apply(future, bundle)
    assert list(result.group_80) == ["QB", "scheduled_bye", "QB"]
    assert result.iloc[1].lower_80 == result.iloc[1].upper_90 == 0
    with pytest.raises(ValueError, match="earlier seasons"):
        calibration.fit(future, 2026)
    with pytest.raises(ValueError, match="out-of-time"):
        calibration.fit(past.assign(phase="in_sample"), 2026)
    with pytest.raises(ValueError, match="match forecast season"):
        calibration.apply(past, bundle)
    unsupported = calibration.fit(history(5), 2026)
    assert calibration.apply(future, unsupported).iloc[0].group_80.startswith("uncalibrated")


def test_timing_rejects_late_naive_or_observed_sources():
    document = dict(
        current_observations=dict(season=2026, through_week=5, saved_at="2026-10-13T12:00:00Z"),
        registered_at="2026-10-13T12:01:00Z",
    )
    schedule = pd.DataFrame(
        [
            dict(
                season=2026,
                week=6,
                game_type="REG",
                game_id="game",
                gameday="2026-10-15",
                home_score=None,
                away_score=None,
            )
        ]
    )
    assert prospective.check_timing(document, schedule, timestamp("2026-10-14T12:00:00Z"))
    with pytest.raises(prospective.DeadlinePassed):
        prospective.check_timing(document, schedule, timestamp("2026-10-15T00:00:00Z"))
    with pytest.raises(ValueError, match="timezone"):
        timestamp("2026-10-13T12:00:00")
    with pytest.raises(ValueError, match="outcomes"):
        prospective.check_timing(
            document, schedule.assign(home_score=0), timestamp("2026-10-14T12:00:00Z")
        )
    with pytest.raises(ValueError, match="after capture"):
        prospective.check_timing(document, schedule, timestamp("2026-10-12T12:00:00Z"))


@pytest.fixture
def paired(tmp_path, monkeypatch):
    monkeypatch.setattr(prospective, "now", lambda: timestamp("2026-10-13T14:00:00Z"))
    monkeypatch.setattr(data, "ROOT", tmp_path)
    inputs = tmp_path / "input"
    inputs.mkdir()
    schedule = pd.DataFrame(
        [
            dict(
                season=2026,
                week=6,
                game_type="REG",
                game_id="game",
                gameday="2026-10-15",
                home_score=None,
                away_score=None,
            )
        ]
    )
    (tmp_path / "data").mkdir()
    schedule.to_parquet(tmp_path / "data/schedule.parquet", index=False)
    snapshot = tmp_path / "snapshot.json"
    write(
        snapshot,
        dict(
            status="accepted",
            registered_at="2026-10-13T12:01:00Z",
            current_observations=dict(season=2026, through_week=5, saved_at="2026-10-13T12:00:00Z"),
            assets={
                "current/schedule.parquet": dict(
                    object="schedule.parquet", sha256=digest(tmp_path / "data/schedule.parquet")
                )
            },
        ),
    )
    panel = pd.DataFrame(
        [
            dict(
                player_id=f"{pos}{i:02d}",
                player_name=f"{pos}{i}",
                position=pos,
                season=2026,
                week=6,
                team="BUF",
                opponent="NE",
                scheduled_game=1,
                is_bye=0,
                target_points=np.nan,
                v2_weekly_mean=12.0 + i / 100,
                v2_exact_available=1.0,
                recent4_points_mean=12.0 + i / 100,
                player_history_mean=10.0,
                player_history_count=20.0,
            )
            for pos, count in (("QB", 10), ("RB", 30), ("WR", 40), ("TE", 10))
            for i in range(count)
        ]
    )
    panel.to_parquet(inputs / "panel.parquet", index=False)
    write(
        inputs / "dataset.json",
        dict(
            panel_sha256=digest(inputs / "panel.parquet"),
            current_through_week=5,
            current_season=2026,
        ),
    )
    mean = dict(
        method="recent",
        features=["recent4_points_mean", "player_history_mean", "player_history_count"],
    )
    spread = modeling.fit_distribution(history(), [], "pooled", 2026)
    (inputs / "mean.pkl").write_bytes(pickle.dumps(mean))
    (inputs / "spread.pkl").write_bytes(pickle.dumps(spread))
    write(inputs / "run.json", {"packages": {}})
    write(inputs / "metrics.json", {})
    write(inputs / "lineup.json", dict(covariance_before_season=2026, correlations=[]))
    entries = []
    for pos in ("QB", "RB", "WR", "TE"):
        rows = _rows(panel, {"position": pos}, mean, spread, 2026, 5)
        path = inputs / f"{pos}.json"
        write(
            path,
            dict(
                rows=rows,
                generated_at="2026-10-13T13:00:00Z",
                snapshot_sha256=digest(snapshot),
                feature_matrix_artifact="forecast_matrix_2026_5",
            ),
        )
        artifacts = dict(
            mean_model=inputs / "mean.pkl",
            distribution_model=inputs / "spread.pkl",
            source_modeling=Path(modeling.__file__),
            source_data=Path(data.__file__),
            source_lineup_runtime=Path(weekly_distribution.__file__),
            source_lineup_research=Path(prospective.v3_lineups.__file__),
            source_rescore=Path(prospective.v3_rescore.__file__),
            run_manifest=inputs / "run.json",
            lineup_calibration=inputs / "lineup.json",
            forecast_matrix_2026_5=inputs / "panel.parquet",
            forecast_dataset_2026_5=inputs / "dataset.json",
        )
        entries.append(
            dict(
                model_id=f"{pos.lower()}_weekly",
                position=pos,
                horizon="weekly",
                origin_week=5,
                training_through_season=2025,
                targets=["points"],
                metrics=inputs / "metrics.json",
                artifacts=artifacts,
                forecasts=[
                    dict(
                        season=2026,
                        origin_week=5,
                        path=path,
                        feature_matrix_artifact="forecast_matrix_2026_5",
                    )
                ],
            )
        )
    registry = tmp_path / "registry"
    publish_campaign(registry, dict(campaign_id="opening_drive_v3", name="V3 test"), entries)
    run = tmp_path / "run"
    run.mkdir()
    baseline = {}
    for pos in ("QB", "RB", "WR", "TE"):
        baseline[pos] = {}
        for name, source in entries[0]["artifacts"].items():
            if name.startswith("forecast_"):
                continue
            relative = f"baseline/{pos}/{name}{source.suffix}"
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, path)
            baseline[pos][name] = dict(path=relative, sha256=digest(path))
    write(run / "baseline_models.json", baseline)
    past = pd.concat([history().assign(position=p) for p in baseline], ignore_index=True)
    write(
        run / "calibrators.json",
        dict(
            players=calibration.fit(past, 2026),
            lineups=calibration.fit(
                history().assign(position="LINEUP"), 2026, policy="position", levels=(80,)
            ),
        ),
    )
    shutil.copytree(
        Path(prospective.__file__).parent,
        run / "source",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    seal(run, dict(forecast_season=2026, generated_at="2026-10-13T13:30:00Z"))
    return run, registry, snapshot, tmp_path / "log", panel


def test_capture_replay_immutable_idempotent_and_missing_outcomes(paired):
    run, registry, snapshot, log, panel = paired
    before = {str(p): digest(p) for p in registry.rglob("*") if p.is_file()}
    captured = prospective.capture(run, registry, snapshot, log)
    directory = Path(captured["path"])
    assert captured["players"] == 90 and captured["lineups"] == 20
    assert prospective.replay(directory)["status"] == "passed"
    assert before == {str(p): digest(p) for p in registry.rglob("*") if p.is_file()}
    archived = {str(p): digest(p) for p in directory.rglob("*") if p.is_file()}
    assert prospective.capture(run, registry, snapshot, log)["status"] == "already_locked"
    assert archived == {str(p): digest(p) for p in directory.rglob("*") if p.is_file()}
    scores, support = prospective.score_captures(log, panel)
    assert support[0]["observed_players"] == support[0]["observed_lineups"] == 0
    assert scores.n.eq(0).all()
    outcomes = panel.assign(target_points=12.0)
    # The missing QB prevents precisely one lineup per selection design from settling.
    outcomes.loc[outcomes.player_id.eq("QB00"), "target_points"] = np.nan
    scores, support = prospective.score_captures(log, outcomes)
    assert support[0]["observed_players"] == 89
    assert support[0]["observed_lineups"] == 18
    all_scores = scores[scores.cohort.eq("all") & scores.season.eq("all")]
    for _, rows in all_scores.groupby(["kind", "position"]):
        assert rows.mae.nunique() == rows.rmse.nunique() == 1
    (directory / "calibrators.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        prospective.replay(directory)


def test_capture_cannot_backfill_or_replace_frozen_run(paired, monkeypatch):
    run, registry, snapshot, log, _ = paired
    monkeypatch.setattr(prospective, "now", lambda: timestamp("2026-10-16T12:00:00Z"))
    with pytest.raises(prospective.DeadlinePassed):
        prospective.capture(run, registry, snapshot, log)
    assert not (log / "forecasts").exists()
    with pytest.raises(FileExistsError):
        publish_directory(registry, run)
    verify(run)


def test_driver_tracks_when_no_models_are_due(paired, monkeypatch):
    run, registry, snapshot, log, _ = paired
    monkeypatch.setattr(driver, "consolidate_opening_drive", lambda *args: None)
    calls = []
    monkeypatch.setattr(prospective, "track", lambda *args: calls.append(args) or {"status": "ok"})
    monkeypatch.setattr(
        driver, "capture_supplements", lambda *args: pytest.fail("Unneeded capture")
    )
    assert (
        driver.main(
            [
                "--registry",
                str(registry),
                "--snapshot",
                str(snapshot),
                "--due",
                "--v31-run",
                str(run),
                "--v31-log",
                str(log),
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0] == (snapshot, registry, run, log)


def test_evaluation_outcomes_cannot_change_same_or_earlier_forecast_bounds():
    from experiments.future_player_lab.campaigns.weekly_v31.run import evaluate

    frame = pd.concat([history(250).assign(season=y) for y in range(2018, 2026)], ignore_index=True)
    before, _, receipts = evaluate(frame)
    frame.loc[frame.season.eq(2025), "target_points"] = 10000
    after, _, _ = evaluate(frame)
    pd.testing.assert_frame_equal(
        before[["lower_80", "upper_80", "lower_90", "upper_90"]],
        after[["lower_80", "upper_80", "lower_90", "upper_90"]],
    )
    assert all(r["max_season"] < r["before_season"] for r in receipts)


def test_actual_league_rosters_are_frozen_and_incomplete_teams_are_disclosed(paired):
    run, _, _, log, panel = paired
    rows = prospective.baseline_rows(run, panel, 2026, 5)
    players = prospective.player_frame(rows)
    ids = ["QB00", "RB00", "RB01", "WR00", "WR01", "WR02", "TE00", "RB02"]
    starters = [
        dict(espn_id=i, player_display_name=pid, position=pid[:2], slot=pid[:2])
        for i, pid in enumerate(ids)
    ]
    snapshot = dict(
        season=2026,
        week=6,
        league_id=1,
        week_lineups=[
            dict(
                week=6,
                home_team_id=1,
                away_team_id=2,
                home_lineup=starters,
                away_lineup=starters[:-1],
            )
        ],
    )
    write(log / "league_inputs.json", dict(snapshot=snapshot, identities=dict(enumerate(ids))))
    bundle = calibration.fit(
        history().assign(position="LINEUP"), 2026, policy="position", levels=(80,)
    )
    frame, missing = prospective.league.predict(log, players, {"correlations": []}, bundle)
    assert len(frame) == 1 and frame.iloc[0].player_ids == ",".join(ids)
    assert missing[0]["team_id"] == 2
    assert frame.iloc[0].prediction == pytest.approx(
        players.set_index("player_id").loc[ids].prediction.sum()
    )


def test_challenger_source_drift_and_duplicate_outcomes_fail_closed(paired):
    run, registry, snapshot, log, panel = paired
    with pytest.raises(ValueError, match="Duplicate outcome"):
        prospective.score_captures(log, pd.concat([panel, panel]))
    (run / "source/calibration.py").write_text("# changed")
    with pytest.raises(ValueError, match="checksum"):
        prospective.capture(run, registry, snapshot, log)


def test_driver_scores_final_week_without_requesting_week_nineteen(paired, monkeypatch):
    run, registry, snapshot, log, _ = paired
    document = json.loads(snapshot.read_text())
    document["current_observations"]["through_week"] = 18
    write(snapshot, document)
    monkeypatch.setattr(driver, "consolidate_opening_drive", lambda *args: None)
    calls = []
    monkeypatch.setattr(prospective, "track", lambda *args: calls.append(args) or {"status": "ok"})
    monkeypatch.setattr(driver, "capture_supplements", lambda *args: pytest.fail("Season over"))
    assert (
        driver.main(
            [
                "--registry",
                str(registry),
                "--snapshot",
                str(snapshot),
                "--due",
                "--v31-run",
                str(run),
                "--v31-log",
                str(log),
            ]
        )
        == 0
    )
    assert len(calls) == 1
