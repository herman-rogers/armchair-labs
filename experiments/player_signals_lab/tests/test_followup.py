import followup_closure
import numpy as np
import polars as pl
import pytest
from followup_calibration import calibration_group, interval_score, weighted_quantile
from followup_closure import signed_bounds
from followup_common import holm, predict
from followup_distributions import crps, earlier_history, joint_draws
from followup_structure import reconcile


def test_population_calibration_ignores_same_season_and_falls_back_for_small_groups():
    earlier = [
        dict(season=year, population="returner", prediction=float(i))
        for year in (2022, 2023, 2024)
        for i in range(50)
    ]
    future = [dict(season=2025, population="rookie", prediction=9999.0)] * 300
    group, source = calibration_group(
        earlier + future, "rookie", 3, [10, 20, 30], "population_volume_scaled", 2025
    )
    assert source == "global" and len(group) == 150
    assert max(r["season"] for r in group) == 2024
    group, source = calibration_group(
        earlier, "returner", 3, [10, 20, 30], "population_scaled", 2025
    )
    assert source == "population"


def test_interval_scores_penalize_misses_and_weights_change_quantiles():
    assert interval_score(5, 0, 10, 0.8) == 10
    assert interval_score(12, 0, 10, 0.8) == pytest.approx(30)
    with pytest.raises(ValueError):
        interval_score(5, 10, 0, 0.8)
    assert weighted_quantile([0, 100], 0.8, [0.9, 0.1]) == 0
    assert weighted_quantile([0, 100], 0.8, [0.1, 0.9]) == 100


def test_holm_is_monotone_and_does_not_treat_subgroups_as_primary():
    rows = [dict(slice="all", p_value=p) for p in [0.001, 0.04, 0.2]]
    rows.append(dict(slice="rookie", p_value=0.001))
    holm(rows)
    assert [r["holm_p"] for r in rows[:3]] == [0.003, 0.08, 0.2]
    assert "holm_p" not in rows[-1]


def test_quantile_and_probability_predictions_do_not_read_test_labels():
    train = [dict(x=i, y=float(i % 2)) for i in range(80)]
    test = [dict(x=12, y=0)]
    for kind, quantile in (("probability", None), ("mean", 0.9)):
        first = predict(train, test, "y", ["x"], kind=kind, quantile=quantile)
        second = predict(train, [dict(x=12, y=1e9)], "y", ["x"], kind=kind, quantile=quantile)
        np.testing.assert_array_equal(first, second)


def test_crps_matches_pairwise_definition_with_a_zero_mass():
    draws = np.array([0, 0, 0, 2, 5, 10], dtype=float)
    expected = np.abs(draws - 4).mean() - 0.5 * np.abs(draws[:, None] - draws[None, :]).mean()
    assert crps(draws, 4) == pytest.approx(expected)
    assert crps([3], 8) == 5


def test_joint_simulation_preserves_hurdle_and_earlier_residual_dependence():
    history = [dict(season=y) for y in range(2015, 2027)]
    assert [r["season"] for r in earlier_history(history, 2025)] == list(range(2020, 2025))
    errors = np.array([[-1.0, -1.0], [1.0, 1.0]])
    rng = np.random.default_rng(7)
    paired = joint_draws(1, 1, 1, errors, rng, paired=True)
    assert set(paired) <= {0, 4}  # Shared positive-case shocks, not a product of means.
    assert not joint_draws(100, 10, 0, errors, rng, paired=True).any()


def test_reconciliation_retains_residual_bucket_and_does_not_invent_shares():
    values = np.array([10.0, 30.0])
    full = reconcile(values, 100, 0.2)
    np.testing.assert_allclose(full, [20, 60])
    assert full.sum() + 100 * 0.2 == pytest.approx(100)
    np.testing.assert_allclose(reconcile(values, 100, 0.2, 0.5), (values + full) / 2)
    np.testing.assert_array_equal(reconcile([0, 0], 100, 0.2), [0, 0])


def test_signed_conformal_can_shrink_and_records_collapsed_bounds():
    assert signed_bounds(0, 10, -2) == (2, 8, False)
    assert signed_bounds(0, 10, 2) == (-2, 12, False)
    assert signed_bounds(0, 10, -6) == (5, 5, True)


def test_signed_calibration_reuses_saved_controls_without_current_season_feedback(
    tmp_path, monkeypatch
):
    from followup_calibration import metric_record

    meta = []
    for position in ("QB", "RB", "WR", "TE"):
        saved = []
        for year in range(2010, 2014):
            for i in range(40 if year < 2013 else 3):
                pid = f"{position}_{i}"
                actual = 5 if year < 2013 else [5, 500, 50000][i]
                row = dict(
                    player_id=pid,
                    position=position,
                    target="season_points",
                    model="market_control",
                    season=year,
                    population="returner",
                    prediction=5,
                    actual=actual,
                    known_available_games_cap=None,
                )
                meta.append(
                    dict(player_id=pid, forecast_season=year, known_available_games_cap=None)
                )
                for level in (0.8, 0.9):
                    for method in ("global_scaled", "quantile_uncalibrated"):
                        value = metric_record(row, method, level, 0, 10, 5, "global", 120)
                        value["baseline_score"] = value["score"]
                        saved.append(value)
        pl.DataFrame(saved).write_parquet(tmp_path / f"{position}_quantiles.parquet")

    class SampleInputs:
        def panel(self):
            return meta

    monkeypatch.setattr(
        followup_closure, "artifact", lambda inputs, run, name: tmp_path / name.split("/")[-1]
    )
    root = tmp_path / "result"
    root.mkdir()
    followup_closure.signed_calibration(SampleInputs(), root)
    result = pl.read_parquet(root / "signed_calibration.parquet")
    signed = result.filter(pl.col("method").str.starts_with("signed_"))
    assert signed.height == 4 * 3 * 2 * 3
    assert signed["signed_correction"].unique().to_list() == [-5.0]
    assert signed["lower"].unique().to_list() == [5.0]
    assert result["baseline_score"].null_count() == 0
