from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.data import (
    CORE,
    TARGETS,
    build_panel,
    labels,
    summarize,
    validate_candidates,
)
from experiments.future_player_lab.evaluate import interval, ranking, select
from experiments.future_player_lab.models import MultiTarget, Transform, recipes


def fixture_tables():
    features = pl.DataFrame(
        {
            "player_id": ["p", "q"],
            "forecast_season": [2020, 2020],
            "forecast_cutoff_date": ["2020-09-01", "2020-09-01"],
            "source_season": [2019, 2019],
            "position": ["WR", "WR"],
            "player_population": ["returning", "rookie"],
            "player_display_name": ["Player", "Quiet"],
        }
    )
    outcomes = features.select("player_id", "forecast_season", "forecast_cutoff_date").with_columns(
        pl.lit(True).alias("outcome_complete")
    )
    from experiments.future_player_lab.data import MEASURES

    weeks = []
    for year, week, amount in [(2019, 1, 10), (2020, 1, 20), (2020, 5, 999)]:
        weeks.append(
            {
                "player_id": "p",
                "season": year,
                "week": week,
                "season_type": "REG",
                **{name: float(amount) for name in MEASURES},
            }
        )
    weeks = pl.DataFrame(weeks)
    usage = weeks.select("player_id", "season", "week").with_columns(pl.lit(0.5).alias("snap_pct"))
    return {
        "preseason_features": features,
        "season_outcomes": outcomes,
        "nfl_player_weeks": weeks,
        "nfl_weekly_usage": usage,
    }


def test_future_outcomes_cannot_change_features_or_membership():
    tables = fixture_tables()
    config = {
        "positions": ["WR"],
        "evaluation_years": [2020],
        "history_windows": [4, 0],
        "sequence_length": 3,
        "horizons": ["season", "next_week"],
        "origins": [4],
    }

    def quiet(*args, **kwargs):
        pass

    before = build_panel(tables, config, quiet)
    tables["nfl_player_weeks"] = tables["nfl_player_weeks"].with_columns(
        pl.when((pl.col("season") == 2020) & (pl.col("week") == 5))
        .then(999999)
        .otherwise(pl.col("league_points"))
        .alias("league_points")
    )
    after = build_panel(tables, config, quiet)
    np.testing.assert_equal(before[1], after[1])
    assert before[0].equals(after[0])
    assert not np.array_equal(before[2], after[2])
    assert before[0].height == 4  # Includes the player with no observations in either period.
    assert np.all(before[2][before[0]["player_id"].to_numpy() == "q"] == 0)


def test_cutoff_includes_only_completed_week_and_excludes_future_week():
    tables = fixture_tables()
    config = {
        "positions": ["WR"],
        "evaluation_years": [2020],
        "history_windows": [0],
        "sequence_length": 1,
        "horizons": ["next_week"],
        "origins": [4],
    }
    panel, x, y, _, names, _ = build_panel(tables, config, lambda *a, **kw: None)
    assert x[0, names.index("sequence:1:league_points")] == 20
    assert y[0, 0] == 999
    assert panel["history_max_time"][0] < panel["cutoff_time"][0]


def test_missing_workload_is_not_zero_or_missing_points():
    block = np.ones((2, len(CORE)))
    block[0, CORE.index("targets")] = np.nan
    result = labels(block, "WR")
    assert np.isnan(result[1])
    assert result[0] == 2
    assert np.all(labels(np.empty((0, len(CORE))), "WR") == 0)


def test_binary_and_sparse_features_survive_training_transform():
    x = np.array([[0, np.nan], [1, np.nan], [0, np.nan], [0, 4]], dtype=float)
    transform = Transform().fit(x)
    assert 0 in transform.active
    transformed = transform.transform(x)
    assert transformed.shape[1] == len(transform.active) + 2
    assert np.isfinite(transformed).all()
    before = transform.mean.copy()
    transform.transform(np.array([[1e9, 1e9]]))
    np.testing.assert_equal(before, transform.mean)


def test_selector_never_uses_current_or_future_losses():
    history = {
        "a": {2020: {"mse": 1}, 2021: {"mse": 1}, 2022: {"mse": 1e9}},
        "b": {2020: {"mse": 2}, 2021: {"mse": 2}, 2022: {"mse": 0}},
    }
    order, years, _ = select(history, 2022, 2)
    assert order[0] == "a" and years == [2020, 2021]
    with pytest.raises(ValueError):
        select(history, 2020, 2)


def test_ranking_is_within_origin():
    actual = np.array([1, 2, 100, 200])
    pred = np.array([100, 200, 1, 2])
    assert ranking(actual, pred, np.array([1, 1, 2, 2]))["spearman"] == pytest.approx(1)


def test_missing_history_has_no_fabricated_efficiency():
    result = summarize(np.empty((0, 2)), np.array([]), 12, [4])
    assert np.isnan(result[:6]).all()
    assert (result[6:] == 0).all()


def test_interval_warmup_is_explicit():
    assert np.isnan(interval(np.array([1, 2]), 0.2))
    assert interval(np.ones(50), 0.2) == 1


def test_future_evidence_date_and_duplicate_forecast_are_rejected():
    frame = fixture_tables()["preseason_features"]
    future = frame.with_columns(pl.lit("2020-10-01").alias("availability_latest_known_on"))
    with pytest.raises(ValueError, match="Future evidence"):
        validate_candidates(future)
    with pytest.raises(ValueError, match="Duplicate forecast"):
        validate_candidates(pl.concat([frame, frame]))


def test_future_college_records_cannot_change_inputs():
    tables = fixture_tables()
    tables["college_identity_links"] = pl.DataFrame(
        {"college_id": ["c"], "player_id": ["q"], "status": ["linked"]}
    )
    tables["college_annual"] = pl.DataFrame(
        {
            "college_id": ["c", "c"],
            "season": [2019, 2020],
            "season_end_date": ["2020-01-15", "2021-01-15"],
            "receiving_yards": [500.0, 10000.0],
        }
    )
    config = {
        "positions": ["WR"],
        "evaluation_years": [2020],
        "history_windows": [0],
        "sequence_length": 1,
        "horizons": ["season"],
        "origins": [4],
    }
    _, x, _, _, names, _ = build_panel(tables, config, lambda *a, **kw: None)
    assert x[1, names.index("college:1:receiving_yards")] == 500
    assert np.isnan(x[1, names.index("college:2:receiving_yards")])


def test_hurdle_averages_uncertain_participation_instead_of_using_future_role():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(100, 5))
    y = np.ones((100, len(TARGETS)))
    y[:50] = 0
    recipe = recipes({"seed": 1, "families": ["hurdle"], "trials_per_family": 1})[0]
    model = MultiTarget(recipe, 1).fit(x, y, np.ones(100))
    prediction = model.predict(np.zeros((1, 5)))
    assert model.participation is not None
    assert 0 < prediction[0, 4] < 1


def test_unknown_model_family_fails_instead_of_silently_using_ridge():
    with pytest.raises(ValueError, match="Unknown model family"):
        recipes({"seed": 1, "families": ["typo"], "trials_per_family": 1})


def test_unlabeled_inference_never_fabricates_zero_outcomes():
    tables = fixture_tables()
    tables["season_outcomes"] = tables["season_outcomes"].with_columns(
        pl.lit(False).alias("outcome_complete")
    )
    config = {
        "positions": ["WR"],
        "evaluation_years": [2020],
        "history_windows": [0],
        "sequence_length": 1,
        "horizons": ["season"],
        "origins": [4],
    }
    with pytest.raises(ValueError, match="No eligible candidates"):
        build_panel(tables, config)
    frame, x, y, _, _, _ = build_panel(tables, config, include_unlabeled=True)
    assert frame.height == len(x) == 2
    assert np.isnan(y).all()


def test_inference_does_not_require_placeholder_outcome_rows():
    tables = fixture_tables()
    tables["season_outcomes"] = tables["season_outcomes"].clear()
    config = {
        "positions": ["WR"],
        "evaluation_years": [2020],
        "history_windows": [0],
        "sequence_length": 1,
        "horizons": ["season"],
        "origins": [4],
    }
    frame, _, y, _, _, _ = build_panel(tables, config, include_unlabeled=True)
    assert frame.height == 2 and np.isnan(y).all()


@pytest.mark.parametrize(
    "family",
    [
        "ridge",
        "hist",
        "extra",
        "svr",
        "kernel",
        "mlp",
        "pls",
        "interactions",
        "neural_tree",
        "hurdle",
    ],
)
def test_all_model_families_handle_unknown_labels(family):
    rng = np.random.default_rng(44)
    x = rng.normal(size=(90, 12)).astype(np.float32)
    x[::3, 1] = np.nan
    y = rng.normal(size=(90, len(TARGETS)))
    y[:30, 1] = np.nan
    recipe = recipes({"seed": 44, "families": [family], "trials_per_family": 1})[0]
    recipe = replace(
        recipe, params={**recipe.params, "iterations": 5, "epochs": 3, "components": 3}
    )
    model = MultiTarget(recipe, 44).fit(x[:70], y[:70], np.ones(70))
    pred = model.predict(x[70:])
    assert pred.shape == (20, len(TARGETS))
    assert np.isfinite(pred).all()
    assert (pred[:, [1, 3, 4]] >= 0).all()
    assert len(model.learners) == 2
