import json

import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v4 import retraining, uncertainty


def history():
    rng = np.random.default_rng(4)
    rows = []
    for season in range(2021, 2026):
        for i in range(120):
            prediction = 3 + i % 18
            rows.append(
                dict(
                    row_id=season * 1000 + i,
                    player_id=str(i),
                    season=season,
                    week=i % 17 + 1,
                    position="QB",
                    horizon="early_week",
                    prediction=prediction,
                    base_prediction=prediction,
                    target_points=prediction + rng.normal(0, 4),
                    player_history_count=12,
                    report_known=0,
                    report_out=0,
                    report_questionable=0,
                    room_out_usage=0,
                )
            )
    return pd.DataFrame(rows)


def test_fit_and_retraining_reject_future_labels():
    frame = history()
    with pytest.raises(ValueError, match="earlier"):
        uncertainty.fit(frame, 2025)
    with pytest.raises(ValueError, match="future"):
        retraining.fit(frame, 2025, 4)
    contract = retraining.fit(frame[frame.season.lt(2025)], 2025, 1)
    with pytest.raises(ValueError, match="before"):
        retraining.predict(frame[frame.season.eq(2024)], contract)


def test_future_outcomes_cannot_change_earlier_calibration_choices_or_draws():
    frame = history()
    _, predictions, decisions, _ = uncertainty.evaluate(frame)
    changed = frame.copy()
    changed.loc[changed.season.eq(2025), "target_points"] += 1000
    _, altered, later_decisions, _ = uncertainty.evaluate(changed)
    assert [r for r in decisions if r["season"] <= 2025] == [
        r for r in later_decisions if r["season"] <= 2025
    ]
    np.testing.assert_array_equal(np.stack(predictions.samples), np.stack(altered.samples))


def test_distribution_apply_retains_means_and_deterministic_zero(tmp_path):
    contract = uncertainty.fit(history(), 2026, 1.15)
    (tmp_path / "uncertainty.json").write_text(json.dumps({"early_week": {"QB": contract}}))
    players = history().iloc[:4].copy()
    players["season"] = 2026
    players["forecast_horizon"] = "early_week"
    players["is_bye"] = [0, 1, 0, 0]
    players.loc[players.index[1], "prediction"] = 0
    players.loc[players.index[2], ["prediction", "report_out"]] = [0, 1]
    out = uncertainty.apply(players, tmp_path)
    draws = np.stack(out.samples)
    np.testing.assert_allclose(draws.mean(axis=1), players.prediction, atol=1e-12)
    np.testing.assert_array_equal(draws[1:3], np.zeros((2, 256)))
    assert (np.diff(draws, axis=1) >= 0).all()


def test_retraining_updates_only_from_completed_weeks():
    frame = history()
    train = frame[frame.season.lt(2025) | (frame.season.eq(2025) & frame.week.lt(8))]
    contract = retraining.fit(train, 2025, 8)
    assert contract["max_current_week"] == 7
    assert contract["current_season_n"] > 0
    prior_only = retraining.fit(train, 2025, 8, in_season=False)
    assert prior_only["current_season_n"] == 0
    test = frame[frame.season.eq(2025) & frame.week.eq(8)]
    first = retraining.predict(test, contract)
    changed = test.copy()
    changed["target_points"] = 100000
    np.testing.assert_array_equal(first, retraining.predict(changed, contract))
    assert np.max(abs(first - test.prediction)) <= 5


def test_total_uncertainty_uses_matched_origin_errors():
    frame = history()
    frame["origin_week"] = 2
    contract = uncertainty.fit_total(frame, 2026)
    live = frame.iloc[:3].copy()
    live["season"] = 2026
    draws = uncertainty.total_samples(live, contract)
    np.testing.assert_allclose(draws.mean(axis=1), live.prediction, atol=1e-12)
    live["origin_week"] = 7
    with pytest.raises(ValueError, match="No evaluated"):
        uncertainty.total_samples(live, contract)


def test_future_features_and_labels_cannot_change_earlier_retrained_forecasts():
    frame = history()
    _, distributions, _, _ = uncertainty.evaluate(frame)
    _, original, _ = retraining.evaluate(frame, distributions)
    changed = frame.copy()
    future = changed.season.eq(2025) & changed.week.ge(12)
    changed.loc[future, ["target_points", "room_out_usage"]] = [1000, 500]
    _, altered_distributions, _, _ = uncertainty.evaluate(changed)
    _, altered, _ = retraining.evaluate(changed, altered_distributions)
    earlier = original.season.lt(2025) | original.week.lt(12)
    np.testing.assert_array_equal(
        original.loc[earlier, "prediction"], altered.loc[earlier, "prediction"]
    )
    np.testing.assert_array_equal(
        np.stack(original.loc[earlier, "samples"]), np.stack(altered.loc[earlier, "samples"])
    )


def test_total_distribution_does_not_extend_one_week_out_or_bye():
    frame = history()
    frame["origin_week"] = 7
    contract = uncertainty.fit_total(frame, 2026)
    live = frame.iloc[:2].copy()
    live["season"] = 2026
    live["report_out"] = 1
    live["is_bye"] = 1
    draws = uncertainty.total_samples(live, contract)
    np.testing.assert_allclose(draws.mean(axis=1), live.prediction, atol=1e-12)
    assert (draws.std(axis=1) > 0).all()
