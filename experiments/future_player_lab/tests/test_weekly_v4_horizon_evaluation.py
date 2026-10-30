import json

import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v4 import horizon_evaluation as evaluation
from experiments.future_player_lab.campaigns.weekly_v31.artifacts import digest, seal, write


def fixture(tmp_path, monkeypatch, cutoff=4, missing=False, absent=False, duplicate=False):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for week in (1, 2, 3, 4):
        teams = ("A", "B") if week < 4 else ("B", "C")
        for team in teams:
            rows.append(
                dict(
                    player_id=f"filler_{team}",
                    season=2026,
                    week=week,
                    season_type="REG",
                    team=team,
                    position="QB",
                    passing_yards=0,
                    passing_tds=0,
                    passing_interceptions=0,
                    rushing_yards=0,
                    rushing_tds=0,
                    receiving_yards=0,
                    receiving_tds=0,
                    receptions=0,
                    completions=0,
                    attempts=0,
                    targets=0,
                )
            )
        if not absent:
            rows.append(
                dict(
                    player_id="p",
                    season=2026,
                    week=week,
                    season_type="REG",
                    team="A" if week <= 2 else "B",
                    position="QB",
                    passing_yards=np.nan if missing and week == 3 else 100,
                    passing_tds=0,
                    passing_interceptions=0,
                    rushing_yards=0,
                    rushing_tds=0,
                    receiving_yards=0,
                    receiving_tds=0,
                    receptions=0,
                    completions=10,
                    attempts=20,
                    targets=0,
                )
            )
    if duplicate:
        rows.append(dict(rows[5]))  # duplicate p's Week 2 identity
    pd.DataFrame(rows).to_parquet(data / "weeks.parquet", index=False)
    schedule = pd.DataFrame(
        [
            dict(
                season=2026,
                week=w,
                game_type="REG",
                home_team="A" if w < 4 else "C",
                away_team="B",
                home_score=10 if w <= cutoff else np.nan,
                away_score=10 if w <= cutoff else np.nan,
            )
            for w in (1, 2, 3, 4)
        ]
    )
    schedule.to_parquet(data / "schedule.parquet", index=False)
    snapshot = tmp_path / "snapshot.json"
    write(
        snapshot,
        dict(
            status="accepted",
            current_observations=dict(season=2026, through_week=cutoff),
            assets={
                f"current/{n}.parquet": dict(
                    object=f"{n}.parquet", sha256=digest(data / f"{n}.parquet")
                )
                for n in ("weeks", "schedule")
            },
        ),
    )
    capture = tmp_path / "capture"
    capture.mkdir()
    projection = {}
    for horizon in ("weekly", "remaining", "season"):
        projection[horizon] = [
            dict(
                player_id="p",
                position="QB",
                season=2026,
                origin_week=1,
                predictions=dict(points=8, passing_yards=200, completions=20, attempts=40),
                distribution=dict(samples=[6, 10]),
                projection=dict(future_weeks=[dict(week=2), dict(week=3)]),
            )
        ]
    write(capture / "projections.json", projection)
    seal(
        capture,
        dict(
            status="prospective",
            evidence_cutoff="2026-09-01T10:00:00+00:00",
            generated_at="2026-09-01T11:00:00+00:00",
            deadline="2026-09-02T10:00:00+00:00",
        ),
    )
    return capture, snapshot


def test_whole_horizons_wait_for_league_completion_and_follow_transfers(tmp_path, monkeypatch):
    capture, snapshot = fixture(tmp_path, monkeypatch, cutoff=3)
    rows = evaluation.score_projection(capture, snapshot)
    weekly = rows[rows.horizon.eq("weekly")]
    assert weekly.status.eq("scored").all()
    assert weekly.loc[weekly.target.eq("passing_yards"), "actual"].iloc[0] == 100
    assert rows[~rows.horizon.eq("weekly")].pending_reason.eq("horizon_not_complete").all()
    schedule = pd.read_parquet(tmp_path / "data/schedule.parquet")
    schedule[["home_score", "away_score"]] = 10
    schedule.to_parquet(tmp_path / "data/schedule.parquet", index=False)
    doc = json.loads(snapshot.read_text())
    doc["current_observations"]["through_week"] = 4
    doc["assets"]["current/schedule.parquet"]["sha256"] = digest(tmp_path / "data/schedule.parquet")
    write(snapshot, doc)
    rows = evaluation.score_projection(capture, snapshot)
    assert rows.status.eq("scored").all()
    yards = rows[rows.target.eq("passing_yards")].set_index("horizon").actual
    assert yards.to_dict() == dict(weekly=100, remaining=300, season=400)
    point = rows[rows.horizon.eq("remaining") & rows.target.eq("points")].iloc[0]
    assert point.actual == 12
    assert point.crps == 3
    assert point.coverage80 == 0


def test_missing_component_stays_pending_independently(tmp_path, monkeypatch):
    capture, snapshot = fixture(tmp_path, monkeypatch, missing=True)
    rows = evaluation.score_projection(capture, snapshot)
    long = rows[rows.horizon.eq("remaining")].set_index("target")
    assert long.loc["passing_yards", "pending_reason"] == "missing_component_label"
    assert long.loc["points", "pending_reason"] == "missing_component_label"
    assert long.loc["completions", "status"] == "scored"
    assert long.loc["completions", "actual"] == 30


def test_supported_absent_player_is_zero_and_duplicates_are_unknown(tmp_path, monkeypatch):
    first = tmp_path / "absent"
    first.mkdir()
    capture, snapshot = fixture(first, monkeypatch, absent=True)
    rows = evaluation.score_projection(capture, snapshot)
    assert rows.status.eq("scored").all()
    assert rows.actual.eq(0).all()
    second = tmp_path / "duplicate"
    second.mkdir()
    capture, snapshot = fixture(second, monkeypatch, duplicate=True)
    rows = evaluation.score_projection(capture, snapshot)
    assert rows.status.eq("pending").all()
    assert rows.pending_reason.eq("ambiguous_player_week").all()


def test_rejects_modified_accepted_assets(tmp_path, monkeypatch):
    capture, snapshot = fixture(tmp_path, monkeypatch)
    (tmp_path / "data/weeks.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset changed"):
        evaluation.score_projection(capture, snapshot)


def test_missing_team_capture_cannot_turn_absence_into_zero(tmp_path, monkeypatch):
    capture, snapshot = fixture(tmp_path, monkeypatch, absent=True)
    path = tmp_path / "data/weeks.parquet"
    box = pd.read_parquet(path)
    box = box[~(box.week.eq(2) & box.team.eq("B"))]
    box.to_parquet(path, index=False)
    document = json.loads(snapshot.read_text())
    document["assets"]["current/weeks.parquet"]["sha256"] = digest(path)
    write(snapshot, document)
    rows = evaluation.score_projection(capture, snapshot)
    assert rows.status.eq("pending").all()
    assert rows.pending_reason.eq("incomplete_source_week").all()
