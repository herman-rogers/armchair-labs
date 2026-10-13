"""Chronological uncertainty fitting never sees same-season/game outcomes."""

import pytest
from experiments.future_player_lab.campaigns.weekly_v3.lineups import (
    estimate_correlations,
    special_estimate,
)


def residual_rows(season, multiplier=1):
    return [
        dict(
            season=season,
            week=w,
            player_id=pos,
            position=pos,
            team="BUF",
            opponent="NYJ",
            prediction=10.0,
            target_points=10 + w * (multiplier if pos == "WR" else 1),
            phase="test",
        )
        for w in range(1, 12)
        for pos in ("QB", "WR")
    ]


def test_covariance_excludes_target_season():
    before = estimate_correlations(residual_rows(2020), 2021)
    after = estimate_correlations(residual_rows(2020) + residual_rows(2021, -1000), 2021)
    assert before == after
    assert before[("same_team", "QB", "WR")]["n"] == 11
    assert 0 < before[("same_team", "QB", "WR")]["correlation"] < 0.1


def test_covariance_rejects_in_sample_rows():
    row = residual_rows(2020)[0]
    row["phase"] = "train"
    with pytest.raises(ValueError, match="out-of-time"):
        estimate_correlations([row], 2021)


def test_special_estimate_ignores_same_week_scores_and_projection_by_default():
    player = dict(espn_id=1, position="K", projected_points=1000.0)
    history = [
        dict(espn_id=1, position="K", week=1, points=7.0),
        dict(espn_id=1, position="K", week=2, points=10000.0),
    ]
    result = special_estimate(player, 2, history)
    assert result["prediction"] == 7
    assert result["std"] is None
    assert special_estimate(player, 2, history, use_projection=True)["prediction"] == 1000


def test_probability_metrics_are_not_misattributed_to_v2_mean():
    from experiments.future_player_lab.campaigns.weekly_v3.lineups import metrics

    rows = [dict(prediction=10.0, v2_prediction=5.0, actual=8.0, q10=0.0, q90=20.0, crps=3.0)]
    assert metrics(rows)["crps"] == 3.0
    assert "crps" not in metrics(rows, mean="v2_prediction")
    assert "coverage80" not in metrics(rows, mean="v2_prediction")


def test_empirical_crps_matches_pairwise_definition():
    import numpy as np
    from experiments.future_player_lab.campaigns.weekly_v3.lineups import empirical_crps

    draws = np.array([0.0, 2.0, 4.0])
    expected = np.abs(draws - 3.0).mean() - 0.5 * np.abs(draws[:, None] - draws).mean()
    assert empirical_crps(draws, 3.0) == pytest.approx(expected)


def test_unknown_recorded_components_remain_unknown():
    from experiments.future_player_lab.campaigns.weekly_v3.lineups import model_scoring_actual

    stats = dict(
        rushing_yards=10, rushing_tds=0, passing_yards=None, passing_tds=1, passing_interceptions=0
    )
    assert model_scoring_actual(stats, "QB", espn_points=0) is None
    stats["passing_yards"] = 100
    assert model_scoring_actual(stats, "QB") == 9
    assert model_scoring_actual(None, "QB", espn_points=0) == 0
    assert model_scoring_actual(None, "QB", espn_points=5) is None


def test_synthetic_primary_comparison_excludes_any_proxy_v2_player():
    from experiments.future_player_lab.campaigns.weekly_v3.lineups import synthetic_experiment

    rows = [
        dict(
            season=2021,
            week=1,
            player_id=f"{pos}{i}",
            position=pos,
            team="BUF",
            opponent="NYJ",
            prediction=10 + i / 100,
            std=2.0,
            target_points=12.0,
            v2_weekly_mean=9 + i / 100,
            v2_exact_available=0 if pos == "QB" and i == 0 else 1,
        )
        for pos, n in [("QB", 10), ("RB", 20), ("WR", 30), ("TE", 20)]
        for i in range(n)
    ]
    output, report, _ = synthetic_experiment(rows)
    assert len(output) == 20
    assert report["coverage"]["lineups_with_v2_proxy"] == 2
    assert report["v2_mean"]["n"] == report["v3_joint"]["n"] == 18
    assert report["all_lineups_with_v2_proxy"]["v2_plus_history_proxy_mean"]["n"] == 20
