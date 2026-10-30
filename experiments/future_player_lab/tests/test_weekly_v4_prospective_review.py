from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v4 import (
    lineup_evaluation,
    prospective,
)
from experiments.future_player_lab.campaigns.weekly_v31.artifacts import seal, write


def capture_fixture(root, week, status="prospective"):
    path = root / "forecasts" / f"2026_{week:02d}" / "early_week"
    path.mkdir(parents=True)
    players = pd.DataFrame(
        [
            dict(
                player_id="known",
                season=2026,
                week=week,
                is_bye=0,
                prediction=5,
                target_points=999,
                forecast_horizon="early_week",
            ),
            dict(
                player_id="unknown",
                season=2026,
                week=week,
                is_bye=0,
                prediction=7,
                target_points=999,
                forecast_horizon="early_week",
            ),
            dict(
                player_id="bye",
                season=2026,
                week=week,
                is_bye=1,
                prediction=0,
                target_points=999,
                forecast_horizon="early_week",
            ),
        ]
    )
    players.to_parquet(path / "players.parquet", index=False)
    seal(
        path,
        dict(
            season=2026,
            week=week,
            status=status,
            generated_at="2026-10-01T12:00:00+00:00",
            deadline="2026-10-02T12:00:00+00:00",
        ),
    )
    return path


def test_completed_training_uses_only_earlier_genuine_known_labels(tmp_path, monkeypatch):
    capture_fixture(tmp_path, 5)
    capture_fixture(tmp_path, 6)
    labels = pd.DataFrame(
        [
            dict(player_id="known", season=2026, week=5, target_points=11),
            dict(player_id="unknown", season=2026, week=5, target_points=np.nan),
            dict(player_id="bye", season=2026, week=5, target_points=0),
            dict(player_id="known", season=2026, week=6, target_points=500),
        ]
    )
    monkeypatch.setattr(prospective, "outcomes", lambda *_: labels)
    result = prospective.completed_training(tmp_path, "snapshot", "registry", 2026, 6)
    assert result.player_id.tolist() == ["known"]
    assert result.target_points.tolist() == [11]
    assert result.week.tolist() == [5]
    assert result.training_capture_sha256.str.len().eq(64).all()


def test_completed_training_rejects_reconstructed_forecasts(tmp_path, monkeypatch):
    capture_fixture(tmp_path, 5, status="retrospective")
    monkeypatch.setattr(prospective, "outcomes", lambda *_: pytest.fail("Must fail before labels"))
    with pytest.raises(ValueError, match="genuine earlier"):
        prospective.completed_training(tmp_path, "snapshot", "registry", 2026, 6)


def test_v4_capture_cannot_be_backfilled_after_deadline(tmp_path, monkeypatch):
    receipt = dict(
        status="prospective", integration_sha256="base", deadline="2026-10-01T12:00:00+00:00"
    )
    monkeypatch.setattr(prospective, "verify", lambda _: receipt)
    monkeypatch.setattr(
        prospective, "check", lambda _: {"dependencies": {"baseline": {"sha256": "base"}}}
    )
    monkeypatch.setattr(prospective, "now", lambda: datetime(2026, 10, 2, tzinfo=UTC))
    monkeypatch.setattr(prospective, "predictions", lambda *_: pytest.fail("Cannot infer late"))
    with pytest.raises(ValueError, match="backfill"):
        prospective.capture(
            tmp_path / "source", tmp_path / "output", tmp_path / "integration", pd.DataFrame()
        )
    assert not (tmp_path / "output").exists()


def test_lineup_score_replays_frozen_distribution_and_keeps_missing_pending(tmp_path, monkeypatch):
    base = tmp_path / "base"
    write(base / "dependence.json", {"early_week": {"correlations": []}})
    capture = tmp_path / "capture"
    (capture / "baseline").mkdir(parents=True)
    player = pd.DataFrame([dict(player_id="p", prediction=5, samples=[4, 6], position="QB")])
    for path in ("baseline/players", "players", "retrained"):
        player.to_parquet(capture / f"{path}.parquet", index=False)
    frozen = pd.DataFrame(
        [
            dict(
                player_ids="p",
                method="v4",
                season=2026,
                week=8,
                team_id=1,
                prediction=5,
                q10=3,
                q50=5,
                q90=7,
                std=2,
                crps=None,
                independent_crps=None,
            )
        ]
    )
    frozen.to_parquet(capture / "lineups.parquet", index=False)
    monkeypatch.setattr(lineup_evaluation, "verify", lambda _: {"forecast_horizon": "early_week"})
    monkeypatch.setattr(
        lineup_evaluation, "check", lambda _: {"dependencies": {"baseline": {"path": str(base)}}}
    )
    monkeypatch.setattr(
        lineup_evaluation,
        "aggregate",
        lambda players, dep, seed, actual: dict(
            prediction=5, q10=3, q50=5, q90=7, std=2, crps=abs(actual - 5), independent_crps=2
        ),
    )
    scored = lineup_evaluation.score_lineups(capture, pd.Series({"p": 8}), "integration")
    assert scored.crps.iloc[0] == 3
    assert scored.interval_score80.iloc[0] == 14
    assert scored.coverage80.iloc[0] == 0
    assert scored.squared_error.iloc[0] == 9
    pending = lineup_evaluation.score_lineups(capture, pd.Series({"p": np.nan}), "integration")
    assert not pending.scored.iloc[0]
    assert pending.pending_players.iloc[0] == 1
