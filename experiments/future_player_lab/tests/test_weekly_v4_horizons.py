from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v4 import horizons as h


def test_fixed_origin_features_do_not_consume_future_labels():
    panel = pd.DataFrame(
        [
            dict(
                player_id="p",
                season=2024,
                week=3,
                origin_week=2,
                team="A",
                position="QB",
                horizon="early_week",
                base_prediction=10,
                recent4_points_mean=9,
            )
        ]
    )
    box = pd.DataFrame(
        [
            dict(
                player_id="p",
                season=2024,
                week=w,
                season_type="REG",
                position="QB",
                **{c: float(w) for c in h.COMPONENTS},
            )
            for w in (3, 4, 5)
        ]
    )
    schedule = pd.DataFrame(
        [
            dict(season=2024, week=w, game_type="REG", home_team="A", away_team="B")
            for w in range(1, 6)
        ]
    )
    before = h.training_panel(panel, box, schedule)
    box.loc[box.week.gt(3), "passing_yards"] = 10000
    after = h.training_panel(panel, box, schedule)
    pd.testing.assert_frame_equal(h.feature_matrix(before), h.feature_matrix(after))
    assert before.loc[1, "label_passing_yards"] != after.loc[1, "label_passing_yards"]
    assert before.loc[0, "label_passing_yards"] == after.loc[0, "label_passing_yards"]
    assert not any("v2" in c for c in h.FEATURES)


def test_present_missing_component_is_not_zero_and_future_roster_not_required():
    panel = pd.DataFrame(
        [
            dict(
                player_id=p,
                season=2024,
                week=3,
                origin_week=2,
                team="A",
                position="QB",
                horizon="early_week",
                base_prediction=10,
            )
            for p in ("p", "left_roster")
        ]
    )
    box = pd.DataFrame(
        [
            dict(
                player_id="p",
                season=2024,
                week=4,
                season_type="REG",
                position="QB",
                **{c: 1.0 for c in h.COMPONENTS},
            )
        ]
    )
    box.loc[0, "passing_yards"] = np.nan
    schedule = pd.DataFrame(
        [
            dict(season=2024, week=w, game_type="REG", home_team="A", away_team="B")
            for w in range(1, 6)
        ]
    )
    rows = h.training_panel(panel, box, schedule)
    assert np.isnan(rows.loc[1, "label_passing_yards"])
    assert rows.loc[rows.player_id.eq("left_roster"), "label_passing_yards"].eq(0).all()


def test_fit_excludes_evaluation_and_future_seasons(monkeypatch):
    seen = []

    class CaptureFit:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            seen.append((len(x), np.asarray(y).max()))
            return self

    monkeypatch.setattr(h, "LGBMRegressor", CaptureFit)
    frame = pd.DataFrame(
        [
            dict(
                position=p,
                season=year,
                later_games=1,
                base_prediction=2,
                **{f"label_{t}": 3 if year == 2024 else 9999 for t in ("points", *h.COMPONENTS)},
            )
            for p in h.TARGETS
            for year in (2024, 2025, 2026)
            for _ in range(110)
        ]
    )
    fit = h.fit_models(frame, before_season=2025)
    assert fit["training_through_season"] == 2024
    assert all(count == 110 and maximum <= 3 for count, maximum in seen)


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


@pytest.fixture
def live_inputs(tmp_path, monkeypatch):
    run = tmp_path / "fit"
    run.mkdir()
    (run / "horizons.py").write_bytes(Path(h.__file__).read_bytes())
    (run / "policy.json").write_text(json.dumps({"forecast_season": 2026}))
    (run / "manifest.json").write_text("{}")
    monkeypatch.setattr(h, "verify", lambda _: {})
    monkeypatch.setattr(h, "ROOT", tmp_path)
    models = {
        "training_through_season": 2025,
        "positions": {
            p: {t: ConstantModel(1) for t in ("points", *targets)}
            for p, targets in h.TARGETS.items()
        },
    }
    monkeypatch.setattr(h.joblib, "load", lambda _: models)
    data = tmp_path / "data"
    data.mkdir()
    schedule = pd.DataFrame(
        [
            dict(season=2026, week=w, game_type="REG", home_team="A", away_team="C")
            for w in (1, 2, 3)
        ]
        + [dict(season=2026, week=w, game_type="REG", home_team="B", away_team="D") for w in (1, 3)]
    )
    box = pd.DataFrame(
        [
            dict(
                player_id="qb",
                season=2026,
                week=1,
                season_type="REG",
                position="QB",
                **{c: 1.0 for c in h.COMPONENTS},
            )
        ]
    )
    document = {"current_observations": {"season": 2026, "through_week": 1}, "assets": {}}
    for name, frame in (("schedule", schedule), ("weeks", box)):
        path = data / f"{name}.parquet"
        frame.to_parquet(path)
        document["assets"][f"current/{name}.parquet"] = dict(
            object=path.name, sha256=h.digest(path)
        )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(document))
    players = pd.DataFrame(
        [
            dict(
                player_id="qb",
                position="QB",
                team="A",
                season=2026,
                week=2,
                prediction=12,
                base_prediction=10,
                is_bye=False,
            ),
            dict(
                player_id="rb",
                position="RB",
                team="B",
                season=2026,
                week=2,
                prediction=0,
                base_prediction=np.nan,
                is_bye=True,
            ),
        ]
    )
    panel = pd.DataFrame(
        [
            dict(player_id=p, target_points=np.nan, recent8_points_mean=8, v2_weekly_mean=9000)
            for p in ("qb", "rb")
        ]
    )
    return players, panel, snapshot, run


def test_live_curve_preserves_weekly_components_actuals_and_byes(live_inputs):
    result = h.build_projection(*live_inputs)
    weekly, remaining, season = (result[k][0] for k in ("weekly", "remaining", "season"))
    assert weekly["predictions"]["points"] == 12
    assert remaining["predictions"]["points"] == 12 + 11
    assert remaining["projection"] == season["projection"]
    assert season["predictions"]["points"] == pytest.approx(23 + 8.14)
    assert remaining["predictions"]["attempts"] == 2
    assert season["predictions"]["attempts"] == 3
    assert weekly["component_points"] == pytest.approx(8.14)
    assert result["weekly"][1]["scheduled_games"] == 0
    assert all(value == 0 for value in result["weekly"][1]["predictions"].values())
    assert result["remaining"][1]["predictions"]["points"] == 9
    assert [g["week"] for g in result["remaining"][1]["projection"]["future_weeks"]] == [3]


def test_live_rejects_forecast_labels_and_wrong_origin(live_inputs):
    players, panel, snapshot, run = live_inputs
    panel.loc[0, "target_points"] = 5
    with pytest.raises(ValueError, match="forecast outcomes"):
        h.build_projection(players, panel, snapshot, run)
    panel["target_points"] = np.nan
    players["week"] = 3
    with pytest.raises(ValueError, match="snapshot origin"):
        h.build_projection(players, panel, snapshot, run)


def test_official_out_zeroes_current_components_only(live_inputs):
    players, panel, snapshot, run = live_inputs
    players.loc[0, "report_out"] = 1
    players.loc[0, "prediction"] = 0
    result = h.build_projection(players, panel, snapshot, run)
    assert all(v == 0 for v in result["weekly"][0]["predictions"].values())
    assert result["remaining"][0]["predictions"]["attempts"] == 1
