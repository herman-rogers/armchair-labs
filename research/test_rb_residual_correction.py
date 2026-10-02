"""Chronology, residual-target and policy-selection contracts for the RB experiment."""

import json

import numpy as np
import pytest
import role_transition_pilot as pilot
from rb_residual_correction import (
    POLICIES,
    PROTOCOL_PATH,
    adjusted,
    choose_shrinkage,
    configurations,
    eligible_windows,
    evaluate,
    nested_predictions,
)


@pytest.fixture
def protocol():
    value = json.loads(PROTOCOL_PATH.read_text())
    value["outer_test_seasons"] = [2019, 2020]
    return value


@pytest.fixture
def rows():
    rng = np.random.default_rng(19)
    return [
        {
            "player_id": str(player),
            "season": year,
            "week": week,
            "adp": 150.0,
            "last3_points": float(player / 2),
            "season_points_pg": float(player / 3),
            "observed_games": week,
            "log_adp": float(np.log(150)),
            "snap_last2": float(player / 12),
            "last3_carries": float(player),
            "last3_targets": float(player % 4),
            "snap_change": float(rng.uniform(-0.3, 0.3)),
            "snap_observations": week,
            "prior_season_games": player + 1,
            "small_prior_sample": float(player < 9),
            "next4_points": float(player * 3 + (year - 2013) * 2 + rng.normal(0, 4)),
        }
        for year in range(2013, 2022)
        for week in (3, 4, 5)
        for player in range(12)
    ]


def test_outer_outcomes_cannot_change_same_fold_predictions_or_tuning(rows, protocol):
    before, tuning_before, _ = nested_predictions(rows, protocol)
    changed = [{**r, "next4_points": 999999.0} if r["season"] >= 2019 else r for r in rows]
    after, tuning_after, _ = nested_predictions(changed, protocol)
    for left, right in zip(before, after, strict=True):
        if left["season"] == 2019:
            for policy in POLICIES:
                assert left[policy] == right[policy]
    assert [r for r in tuning_before if r["season"] == 2019] == [
        r for r in tuning_after if r["season"] == 2019
    ]


def test_later_seasons_cannot_change_any_requested_fold(rows, protocol):
    before, tuning_before, _ = nested_predictions(rows, protocol)
    before_only, tuning_only, _ = nested_predictions(
        [r for r in rows if r["season"] <= 2020], protocol
    )
    assert before == before_only
    assert tuning_before == tuning_only


def test_residual_training_targets_are_chronological_out_of_fold(rows, protocol, monkeypatch):
    corrections_seen = []

    def spy_fit(train, test, features, target="next4_points"):
        assert max(r["season"] for r in train) < min(r["season"] for r in test)
        if target == "residual":
            for row in train:
                expected_baseline = np.mean(
                    [r["next4_points"] for r in rows if r["season"] < row["season"]]
                )
                assert row["usage_ridge"] == expected_baseline
                assert row["residual"] == row["next4_points"] - expected_baseline
            corrections_seen.append(len(train))
        return np.repeat(np.mean([r[target] for r in train]), len(test))

    monkeypatch.setattr(pilot, "fit_predict", spy_fit)
    _, tuning, chronology = nested_predictions(rows, protocol)
    assert corrections_seen
    for row in chronology:
        assert row["correction_train_latest"] < row["season"]
        assert row["baseline_train_latest"] < row["season"]
    for row in tuning:
        assert max(row["validation_seasons"]) < row["season"]
    assert tuning[0]["validation_seasons"] == [2017, 2018]


def test_ties_prefer_exactly_unchanged_baseline(rows, protocol):
    constant = [{**r, "usage_ridge": float(r["player_id"]), "delta": 0.0} for r in rows]
    result = choose_shrinkage(constant, 2019, "delta", protocol)
    assert result["selected"]["weight"] == 0
    assert len(result["trials"]) == 10
    assert configurations(protocol).count((0.0, 0.0)) == 1


def test_zero_correction_and_sample_shrinkage():
    assert adjusted({"usage_ridge": 12.0}, "missing_delta", 0, 10) == 12.0
    sparse = {"usage_ridge": 12.0, "prior_season_games": 1, "observed_games": 3, "delta": 8.0}
    experienced = {**sparse, "prior_season_games": 17}
    assert adjusted(sparse, "delta", 0.5, 4) == 14
    assert adjusted(experienced, "delta", 0.5, 4) > adjusted(sparse, "delta", 0.5, 4)


def test_original_price_and_prior_sample_eligibility(rows):
    frame = [r for r in rows if r["season"] == 2019 and r["week"] == 3]
    frame[0] = {**frame[0], "adp": 100.0}
    frame[1] = {**frame[1], "adp": 300.01}
    frame[2] = {**frame[2], "last3_points": 12.0}
    frame[3] = {**frame[3], "adp": None}
    remaining = eligible_windows(frame, "all_late_price")[2019, 3]
    assert {r["player_id"] for r in remaining} == {str(i) for i in range(4, 12)}
    small = eligible_windows(frame, "small_prior_sample")[2019, 3]
    assert {r["player_id"] for r in small} == {str(i) for i in range(4, 9)}
    assert eligible_windows(frame[-2:], "all_late_price") == {}


def test_every_policy_uses_identical_windows_and_candidate_error_samples(rows, protocol):
    predictions, _, _ = nested_predictions(rows, protocol)
    summaries, errors, _ = evaluate(predictions)
    for cohort in ("all_late_price", "small_prior_sample"):
        samples = [r for r in summaries if r["cohort"] == cohort]
        assert {r["picks"] for r in samples} == {18}
        assert all([f["season"] for f in r["folds"]] == [2019, 2020] for r in samples)
        counts = {r["common_rows"] for r in errors if r["cohort"] == cohort}
        assert len(counts) == 1
        assert all(r["excluded_nonfinite_rows"] == 0 for r in errors)


def test_future_and_duplicate_rows_fail_closed(rows, protocol):
    with pytest.raises(ValueError, match="2026"):
        nested_predictions(rows + [{**rows[0], "season": 2026}], protocol)
    with pytest.raises(ValueError, match="Duplicate"):
        nested_predictions(rows + [rows[0]], protocol)


def test_tuning_requires_multiple_prior_validation_seasons(rows, protocol):
    with pytest.raises(ValueError, match="Not enough"):
        choose_shrinkage([r for r in rows if r["season"] == 2018], 2019, "delta", protocol)
