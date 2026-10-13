import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.interval_research import methods
from experiments.future_player_lab.campaigns.interval_research.run import (
    baseline_diagnosis,
    evaluate,
    paired_diagnostics,
)


def panel(n=250, year=2020):
    return pd.DataFrame(
        dict(
            row_id=np.arange(n),
            season=year,
            week=1,
            position="QB",
            prediction=7.0,
            target_points=np.linspace(-1, 15, n),
            q05=-5.0,
            q10=-3.0,
            q90=17.0,
            q95=19.0,
            player_history_count=32,
            phase="evaluation",
            player_id=[f"p{i}" for i in range(n)],
        )
    )


@pytest.mark.parametrize("method", methods.FIXED)
def test_challengers_preserve_means_quantiles_and_nested_finite_intervals(method):
    past = panel()
    future = panel(year=2021)
    future["is_bye"] = 0
    future.loc[0, ["is_bye", "prediction", "q05", "q10", "q90", "q95"]] = [1, 0, 0, 0, 0, 0]
    result = methods.apply(future, methods.fit(past, 2021, method))
    pd.testing.assert_frame_equal(future, result[future.columns])
    assert (result.lower_80 <= result.upper_80).all()
    assert (result.lower_90 <= result.lower_80).all()
    assert (result.upper_90 >= result.upper_80).all()
    assert np.isfinite(result[["lower_80", "upper_80", "lower_90", "upper_90"]]).all().all()
    assert result.loc[0, "lower_90"] == result.loc[0, "upper_90"] == 0


def test_signed_shrinkage_can_reduce_overwide_ranges():
    past = panel()
    future = panel(year=2021)
    original = methods.apply(future, methods.fit(past, 2021, "v31"))
    signed = methods.apply(future, methods.fit(past, 2021, "scaled_signed"))
    assert (signed.upper_80 - signed.lower_80).mean() < (
        original.upper_80 - original.lower_80
    ).mean()
    assert methods.components(signed, 80)["covered"].mean() >= 0.8


def test_endpoint_order_statistic_is_stable_at_exact_integer_ranks():
    values = np.arange(99, dtype=float)
    assert methods.order_stat(values, (1 - 0.8) / 2, lower=True) == 9
    assert methods.order_stat(values, 1 - (1 - 0.8) / 2) == 89


def test_asymmetric_correction_moves_only_the_tail_that_needs_it():
    past = panel().assign(target_points=np.linspace(0, 10, 250), q10=1.0, q90=20.0)
    future = past.assign(season=2021)
    adjusted = methods.apply(future, methods.fit(past, 2021, "tail_additive", levels=(80,)))
    assert abs(adjusted.lower_80.iloc[0] - 1) < 0.1
    assert adjusted.upper_80.iloc[0] < 10
    assert methods.components(adjusted, 80)["covered"].mean() >= 0.8


def test_future_and_in_sample_calibration_rejected_and_recent_window_is_real():
    with pytest.raises(ValueError, match="precede"):
        methods.fit(panel(year=2021), 2021, "tail_scaled")
    with pytest.raises(ValueError, match="out-of-time"):
        methods.fit(panel().assign(phase="train"), 2021, "tail_scaled")
    past = pd.concat([panel(year=2018).assign(target_points=1000), panel(year=2020)])
    recent = methods.fit(past, 2021, "tail_scaled_recent")
    assert recent["n"] == 250 and recent["max_season"] == 2020
    with pytest.raises(ValueError, match="match"):
        methods.apply(panel(year=2022), recent)


def test_sparse_fine_groups_fall_back_to_position():
    past = panel().assign(prediction=np.linspace(0, 12, 250))
    future = panel(year=2021).assign(prediction=0.5)
    result = methods.apply(future, methods.fit(past, 2021, "tail_scaled_fine"))
    assert set(result.group_80) == {"QB"}
    assert set(result.support_80) == {250}


def test_selection_guard_rejects_cheap_but_undercovering_intervals():
    # 77% exact zero, 23% tiny misses: cheaper score but below the preset 78% guard.
    base = panel().assign(target_points=np.r_[np.zeros(192), np.full(58, 0.01)])
    cheap = base.assign(method="v3", lower_80=0.0, upper_80=0.0)
    wide = base.assign(method="v31", lower_80=-1.0, upper_80=1.0)
    validation = pd.concat([cheap, wide], ignore_index=True)
    assert methods.select(validation, 2021, levels=(80,))["choices"]["QB/5_to_10"]["method"] == "v3"
    assert (
        methods.select(validation, 2021, levels=(80,), guarded=True)["choices"]["QB/5_to_10"][
            "method"
        ]
        == "v31"
    )
    with pytest.raises(ValueError, match="forecast-season"):
        methods.select(validation.assign(season=2021), 2021)


def test_full_chronology_cannot_use_evaluation_year_outcomes_for_bounds_or_policy():
    frame = pd.concat(
        [
            panel(year=year).assign(row_id=np.arange(250) + year * 1000)
            for year in range(2018, 2026)
        ],
        ignore_index=True,
    )
    before, receipts = evaluate(frame)
    frame.loc[frame.season.eq(2025), "target_points"] = 9999
    after, _ = evaluate(frame)
    keys = ["row_id", "method"]
    fields = ["lower_80", "upper_80", "lower_90", "upper_90", "chosen_method"]
    pd.testing.assert_frame_equal(
        before.set_index(keys)[fields].sort_index(), after.set_index(keys)[fields].sort_index()
    )
    assert all(r["max_validation_season"] < r["before_season"] for r in receipts["selection"])


def test_cost_decomposition_and_season_pairing_are_consistent():
    frame = panel().assign(season=2021, lower_80=-3.0, upper_80=17.0, method="v3")
    wider = frame.assign(method="v31", lower_80=-4.0, upper_80=18.0)
    combined = pd.concat([frame, wider], ignore_index=True)
    diagnosis = baseline_diagnosis(combined)
    row = diagnosis[diagnosis.cohort.eq("all")].iloc[0]
    assert row.net_score_cost == pytest.approx(
        row.added_width - row.saved_below_penalty - row.saved_above_penalty
    )
    assert row.net_score_cost == pytest.approx(2.0)
    combined["q05"], combined["q95"] = -5.0, 19.0
    combined["lower_90"], combined["upper_90"] = -5.0, 19.0
    comparisons, _ = paired_diagnostics(combined)
    score = comparisons[comparisons.cohort.eq("all") & comparisons.level.eq(80)].iloc[0]
    assert score.delta_interval_score == score.ci95_lower == score.ci95_upper == -2
