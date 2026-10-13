import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v31_points import methods


def sample(year, n=240):
    rng = np.random.default_rng(year)
    base = rng.uniform(1, 15, n)
    active = rng.random(n) < 0.7
    frame = pd.DataFrame(
        dict(
            row_id=np.arange(n) + year * 1000,
            season=year,
            week=1,
            position="QB",
            player_id=[f"p{x % 60}" for x in range(n)],
            base_prediction=base,
            target_points=np.where(active, base + rng.normal(0, 3, n), 0),
            appeared=active,
            player_history_count=20,
            recent4_appearance_rate=0.7,
            recent8_points_mean=base,
            weeks_since_appearance=1,
            is_bye=0,
        )
    )
    for key in methods.UNCERTAINTY:
        frame[key] = rng.uniform(1, 3, n)
    for key in methods.MEMBERS:
        frame[f"member_{key}"] = base + rng.normal(0, 1, n)
    return frame


@pytest.fixture(scope="module")
def fitted():
    return methods.fit(sample(2020), 2021, ["recent8_points_mean", "recent4_appearance_rate"])


def test_future_outcomes_and_identity_features_are_rejected():
    with pytest.raises(ValueError, match="earlier"):
        methods.fit(sample(2021), 2021, ["base_prediction"])
    for forbidden in ("target_points", "appeared", "player_id"):
        with pytest.raises(ValueError, match="allowlist"):
            methods.matrix(sample(2020), [forbidden])


def test_inference_does_not_read_labels_and_every_candidate_is_finite(fitted):
    frame = sample(2021)
    expected, _ = methods.predict(fitted, frame)
    frame["target_points"], frame["appeared"] = 99999, False
    changed, _ = methods.predict(fitted, frame)
    without, diagnostic = methods.predict(fitted, frame.drop(columns=["target_points", "appeared"]))
    assert set(expected) == set(methods.FIXED)
    for key in methods.FIXED:
        np.testing.assert_array_equal(expected[key], changed[key])
        np.testing.assert_array_equal(expected[key], without[key])
        assert np.isfinite(expected[key]).all()
    assert np.any(expected["residual_uncertainty"] != frame.base_prediction)
    for label in ("basic", "uncertainty"):
        p = diagnostic[f"p_appearance_{label}"]
        assert ((p >= 0) & (p <= 1)).all()
        np.testing.assert_allclose(
            expected[f"mixture_{label}"], p * diagnostic[f"active_mean_{label}"]
        )


def test_byes_and_cold_start_players_are_supported(fitted):
    frame = sample(2021)
    frame.loc[0, "is_bye"] = 1
    frame["player_id"] = "unseen"
    result, _ = methods.predict(fitted, frame)
    assert all(v[0] == 0 for v in result.values())
    np.testing.assert_allclose(result["bias_group_100"], result["bias_player_24"])


def test_mixture_uses_appearance_not_nonzero_score_and_preserves_negative_scores():
    frame = sample(2020)
    frame["appeared"] = True
    frame["target_points"] = -2.0
    bundle = methods.fit(frame, 2021, ["base_prediction"])
    prediction, diagnostics = methods.predict(bundle, sample(2021))
    np.testing.assert_allclose(diagnostics["p_appearance_basic"], 1)
    np.testing.assert_allclose(prediction["mixture_basic"], -2)
    frame["target_points"] = 0.0
    bundle = methods.fit(frame, 2021, ["base_prediction"])
    _, diagnostics = methods.predict(bundle, sample(2021))
    np.testing.assert_allclose(diagnostics["p_appearance_basic"], 1)
    frame["appeared"] = False
    frame["target_points"] = 1.0
    with pytest.raises(ValueError, match="zero"):
        methods.fit(frame, 2021, ["base_prediction"])


def test_blend_weights_are_convex_and_ablations_use_matched_inputs(fitted):
    for key in ("global_weights", "convex_weights"):
        assert (fitted[key] >= -1e-10).all()
        assert fitted[key].sum() == pytest.approx(1)
    assert set(fitted["full"]) - set(fitted["basic"]) == set(methods.UNCERTAINTY)
    frame = sample(2021)
    result, _ = methods.predict(fitted, frame)
    members = frame[[f"member_{m}" for m in methods.MEMBERS]].to_numpy()
    for name in (
        "blend_equal",
        "blend_global",
        "blend_convex",
        "blend_conditional_basic",
        "blend_conditional_uncertainty",
    ):
        assert (result[name] >= members.min(axis=1) - 1e-8).all()
        assert (result[name] <= members.max(axis=1) + 1e-8).all()


def test_selector_cannot_use_future_and_guard_protects_qb_low_group():
    base = sample(2020, n=400).assign(
        base_prediction=np.r_[np.full(150, 3.0), np.full(250, 0.2)],
        target_points=0.0,
        prediction=2.0,
        method="v31",
    )
    candidate = base.assign(
        method="bias_global", prediction=np.r_[np.full(150, 2.1), np.zeros(250)]
    )
    history = pd.concat([base, candidate], ignore_index=True)
    assert methods.choose(history, 2021)["method"] == "bias_global"
    assert methods.choose(history, 2021, guarded=True)["method"] == "v31"
    with pytest.raises(ValueError, match="earlier"):
        methods.choose(history.assign(season=2021), 2021)
    with pytest.raises(ValueError, match="identical"):
        methods.choose(history.iloc[:-1], 2021)


def test_residual_intervals_use_past_candidate_errors_not_current_outcomes():
    past = sample(2020).assign(prediction=5.0)
    future = sample(2021).assign(prediction=7.0)
    a = methods.residual_intervals(past, future, 2021)
    b = methods.residual_intervals(past, future.assign(target_points=99999), 2021)
    columns = ["lower_80", "upper_80", "lower_90", "upper_90"]
    pd.testing.assert_frame_equal(a[columns], b[columns])
    assert (a.lower_90 <= a.lower_80).all()
    assert (a.upper_90 >= a.upper_80).all()
    with pytest.raises(ValueError, match="earlier"):
        methods.residual_intervals(past.assign(season=2021), future, 2021)


def test_bias_variance_decomposition_and_ranking_metrics():
    from experiments.future_player_lab.campaigns.weekly_v31_points.run import metric_values

    frame = sample(2021).assign(
        prediction=3.0, lower_80=-5.0, upper_80=15.0, lower_90=-8.0, upper_90=18.0
    )
    score = metric_values(frame)
    assert score["mse"] == pytest.approx(score["error_variance"] + score["bias"] ** 2)


def test_unchanged_endpoint_shift_is_exact_zero_before_feature_scaling():
    from experiments.future_player_lab.campaigns.weekly_v31_points.run import tail_shift

    rng = np.random.default_rng(45)
    lower, upper = rng.normal(size=1000), rng.uniform(5, 25, size=1000)
    np.testing.assert_array_equal(tail_shift(lower, upper, lower, upper), np.zeros(1000))
    np.testing.assert_allclose(tail_shift(lower, upper, lower + 2, upper + 4), 3.0)


def test_full_pipeline_cannot_use_2025_outcomes_for_2021_to_2025_forecasts(tmp_path):
    import json

    from experiments.future_player_lab.campaigns.weekly_v31_points.run import evaluate

    frame = pd.concat([sample(year) for year in range(2018, 2026)], ignore_index=True)
    frame["player_name"], frame["team"], frame["label_observed"] = "Example", "ABC", True
    for lower, upper in (("q10", "q90"), ("q05", "q95")):
        frame[lower], frame[upper] = -5.0, 25.0
    for level in (80, 90):
        frame[f"v31_lower_{level}"], frame[f"v31_upper_{level}"] = -10.0, 30.0
    a, b = tmp_path / "before", tmp_path / "after"
    a.mkdir()
    b.mkdir()
    first, _ = evaluate(frame, ["recent8_points_mean"], a)
    frame.loc[frame.season.eq(2025) & frame.appeared, "target_points"] += 50
    second, _ = evaluate(frame, ["recent8_points_mean"], b)
    keys = ["row_id", "method"]
    fields = ["prediction", "lower_80", "upper_80", "lower_90", "upper_90", "chosen_method"]
    pd.testing.assert_frame_equal(
        first.set_index(keys)[fields].sort_index(), second.set_index(keys)[fields].sort_index()
    )
    decisions = [json.loads((path / "selection.json").read_text()) for path in (a, b)]
    assert [r for r in decisions[0] if r["year"] <= 2025] == [
        r for r in decisions[1] if r["year"] <= 2025
    ]
