"""Aggregation preserves means, missingness, dependence, and snapshot chronology."""

import numpy as np
import pytest

from engine.metrics.weekly_distribution import correlation_matrix, forecast_snapshot, joint_draws
from tests.test_weekly_lineup import schedule, snapshot


def players():
    return [
        dict(player_id="a", position="QB", team="BUF", opponent="NYJ", prediction=20.0, std=5.0),
        dict(player_id="b", position="WR", team="BUF", opponent="NYJ", prediction=10.0, std=4.0),
    ]


def test_positive_shared_team_dependence_widens_sum():
    rows = players()
    correlated = joint_draws(rows, {("same_team", "QB", "WR"): {"correlation": 0.4}}, n=100000)
    independent = joint_draws(rows, n=100000)
    assert correlated.mean() == pytest.approx(30.0, abs=0.1)
    assert correlated.var() == pytest.approx(25 + 16 + 2 * 0.4 * 5 * 4, rel=0.025)
    assert independent.var() == pytest.approx(41, rel=0.025)


def test_matrix_is_psd_even_for_inconsistent_pair_estimates():
    rows = players() + [dict(position="RB", team="BUF", opponent="NYJ", prediction=8.0, std=4.0)]
    est = {
        ("same_team", "QB", "WR"): {"correlation": 0.95},
        ("same_team", "QB", "RB"): {"correlation": 0.95},
        ("same_team", "RB", "WR"): {"correlation": -0.95},
    }
    matrix = correlation_matrix(rows, est)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-8
    assert np.diag(matrix) == pytest.approx([1, 1, 1])


def test_samples_preserve_mean_and_bye_zero():
    row = dict(position="WR", prediction=12.0, std=2.0, samples=list(np.linspace(-3, 8, 256)))
    assert joint_draws([row], n=100000).mean() == pytest.approx(12, abs=0.05)
    assert not joint_draws([dict(position="WR", prediction=0.0, std=0.0, samples=[0.0] * 20)]).any()


def test_missing_starter_withholds_total_and_kicker_uses_previous_week():
    data = [
        dict(
            player_id="qb-allen",
            player_display_name="Josh Allen",
            season=2026,
            week=3,
            position="QB",
            team="BUF",
            prediction=20.0,
            std=5.0,
        )
    ]
    result = forecast_snapshot(
        snapshot(),
        data,
        {11: "qb-allen"},
        schedule(),
        {"special_teams": {"K": {"std": 4.0, "n": 20}}},
    )
    home, away = result["matchups"][0]["home"], result["matchups"][0]["away"]
    assert home["model_projection"] == 29
    assert home["prediction_std"] > 5
    assert home["offense_only_interval_80"] is not None
    assert home["uncertainty_status"] == "provisional_full_lineup"
    assert away["model_projection"] is None
    assert away["prediction_interval_80"] is None


def test_sparse_special_scale_withholds_full_range_but_not_mean():
    data = [dict(player_id="qb-allen", position="QB", team="BUF", prediction=20.0, std=5.0)]
    result = forecast_snapshot(
        snapshot(),
        data,
        {11: "qb-allen"},
        schedule(),
        {"special_teams": {"K": {"std": 4.0, "n": 3}}},
    )
    home = result["matchups"][0]["home"]
    assert home["model_projection"] == 29
    assert home["prediction_interval_80"] is None
    assert home["offense_only_interval_80"] is not None


def test_future_calibration_is_rejected():
    with pytest.raises(ValueError, match="forecast cutoff"):
        forecast_snapshot(
            snapshot(),
            [],
            {},
            schedule(),
            {"capture_season": 2026, "special_errors_before_week": 4},
        )


def test_runtime_reports_observed_undercoverage_and_provisional_special_teams():
    result = forecast_snapshot(snapshot(), [], {}, schedule(), {})
    note = result["range_reliability_note"]
    assert "nominal 80%" in note
    assert "synthetic offense-lineup" in note
    assert "77%" in note
    assert "K/DST uncertainty remains provisional" in note
    assert note in result["limitations"]
