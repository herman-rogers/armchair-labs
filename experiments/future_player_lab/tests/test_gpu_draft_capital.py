"""Contracts for the isolated draft-event extension and matched comparisons."""

import shutil

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.gpu_lab.data import Dataset
from experiments.future_player_lab.gpu_lab.draft_capital_study import (
    ADDED,
    aligned_forecasts,
    draft_features,
    extend_dataset,
    fit_fold,
    parse_positive_integer,
    score,
)
from experiments.future_player_lab.gpu_lab.run import digest, verify_checkpoint


def sources():
    panel = pl.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "year": [2018] * 3,
            "origin": [0, 2, 2],
            "candidate_basis": ["dated draft / exact ID"] * 3,
        }
    )
    combine = pl.DataFrame(
        {
            "player_id": ["a", "c"],
            "combine_year": [2018, 2019],
            "draft_year": [2018.0, 2019.0],
            "draft_round": [1.0, 2.0],
            "combine_pick": [12.0, 40.0],
        }
    )
    rosters = pl.DataFrame(
        {
            "player_id": ["a", "b", "b"],
            "season": [2018] * 3,
            "week": [1, 2, 3],
            "roster_pick": [100.0, 50.0, 90.0],
        }
    )
    return panel, combine, rosters


def test_schema_parse_preserves_unknowns():
    parsed = parse_positive_integer(pl.DataFrame({"pick": [" 12 ", "50.0", None, ""]}), "pick")
    assert parsed["pick"].to_list() == [12.0, 50.0, None, None]
    assert (
        parse_positive_integer(pl.DataFrame({"pick": [1, 2]}), "pick")["pick"].dtype == pl.Float64
    )


@pytest.mark.parametrize("value", ["unknown", "0", "-1", "1.5", "NaN"])
def test_invalid_event_values_fail(value):
    with pytest.raises(ValueError, match="positive integer"):
        parse_positive_integer(pl.DataFrame({"pick": [value]}), "pick")


def test_as_of_gate_and_future_mutation_invariance():
    panel, combine, rosters = sources()
    result = draft_features(panel, combine, rosters)
    assert result[ADDED[0]].to_list() == [12.0, 50.0, None]
    assert result[ADDED[1]].to_list() == [1.0, None, None]
    assert result[ADDED[2]].to_list() == [0.0, None, None]
    changed_combine = combine.with_columns(
        pl.when(pl.col("combine_year") > 2018)
        .then(200.0)
        .otherwise(pl.col("combine_pick"))
        .alias("combine_pick")
    )
    changed_rosters = rosters.with_columns(
        pl.when(pl.col("week") > 2)
        .then(222.0)
        .otherwise(pl.col("roster_pick"))
        .alias("roster_pick")
    )
    assert (
        draft_features(panel, changed_combine, changed_rosters)
        .select(ADDED)
        .equals(result.select(ADDED))
    )


def test_past_conflicts_and_cross_source_conflicts_fail():
    panel, combine, rosters = sources()
    conflict = rosters.with_columns(pl.lit(1).alias("week"))
    with pytest.raises(ValueError, match="Conflicting"):
        draft_features(panel, combine, conflict)
    later = panel.with_columns(pl.lit(2).alias("origin"))
    with pytest.raises(ValueError, match="disagree"):
        draft_features(later, combine, rosters)


def test_extension_preserves_raw_columns_population_and_label_masks():
    panel, combine, rosters = sources()
    x = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    targets = np.ones((3, 7))
    targets[1, 2] = np.nan
    base = Dataset(
        panel,
        x,
        ["raw_a", "raw_b"],
        ["raw", "raw"],
        targets,
        np.ones(3),
        np.zeros((3, 7)),
        pl.DataFrame(),
        {},
        {},
    )
    result, _ = extend_dataset(base, combine, rosters)
    np.testing.assert_array_equal(result.x[:, :2], base.x)
    np.testing.assert_array_equal(result.targets, targets)
    assert result.names == ["raw_a", "raw_b", *ADDED]
    assert base.names == ["raw_a", "raw_b"] and base.x.shape == (3, 2)
    assert result.panel.equals(base.panel)


def forecasts():
    return pl.DataFrame(
        {
            "horizon": ["season"] * 2,
            "year": [2025] * 2,
            "origin": [0] * 2,
            "player_id": ["a", "b"],
            "target": ["points"] * 2,
            "actual": [1.0, float("nan")],
            "prediction": [2.0, 3.0],
            "weeks": [18.0, 18.0],
            "training_max_year": [2024] * 2,
        }
    )


def test_matched_rows_require_identical_keys_and_unknown_masks():
    f = forecasts()
    assert aligned_forecasts(f.reverse(), f).height == 2
    with pytest.raises(ValueError, match="row keys"):
        aligned_forecasts(f.head(1), f)
    with pytest.raises(ValueError, match="actual"):
        aligned_forecasts(f.with_columns(pl.col("actual").fill_nan(0.0)), f)
    with pytest.raises(ValueError, match="Nonfinite"):
        aligned_forecasts(f.with_columns(pl.lit(float("nan")).alias("prediction")), f)


def test_real_cpu_fit_resume_and_final_bundle(tmp_path):
    panel = pl.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(12)],
            "year": [2024] * 8 + [2025] * 4,
            "origin": [0] * 12,
            "candidate_basis": ["past NFL observation"] * 12,
            "horizon": ["season"] * 12,
            "weeks": [18.0] * 12,
        }
    )
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    targets = np.arange(84, dtype=float).reshape(12, 7)
    targets[8, 2] = np.nan
    dataset = Dataset(
        panel,
        x,
        ["a", "b"],
        ["raw", "raw"],
        targets,
        np.full(12, 18.0),
        np.zeros((12, 7)),
        pl.DataFrame(),
        {},
        {},
    )
    recipe = {
        "id": "fixture",
        "family": "lightgbm",
        "device": "cpu",
        "seed": 3,
        "representation": {"kind": "raw"},
        "params": {"n_estimators": 2, "num_leaves": 3, "min_child_samples": 1},
    }
    config = {"threads": 1, "years": [2025]}
    study = tmp_path / "study"
    fold = study / "season/fits/fixture/2025"
    fit_fold(dataset, recipe, 2025, config, fold)
    verify_checkpoint(fold)
    before = digest(fold / "complete.json")
    fit_fold(dataset, recipe, 2025, config, fold)
    assert digest(fold / "complete.json") == before
    predictions = pl.read_parquet(fold / "predictions.parquet")
    assert predictions.height == 32
    assert (
        predictions.filter(
            (pl.col("player_id") == "p8") & pl.col("target").is_in(["rushing_tds", "points"])
        )["actual"]
        .is_nan()
        .all()
    )
    fit_fold(dataset, recipe, 2026, config, tmp_path / "final", final=True)
    verify_checkpoint(tmp_path / "final")
    assert len(list((tmp_path / "final").glob("*.joblib"))) == 9
    main = tmp_path / "main"
    shutil.copytree(fold, main / "season/fits/fixture/2025")
    panel.select("player_id", "year", "origin").with_columns(
        pl.when(pl.col("player_id") == "p9").then(2025.0).otherwise(None).alias("draft_year")
    ).write_parquet(study / "season/draft_features.parquet")
    score(
        study,
        main,
        {**config, "recipes": [recipe], "horizons": ["season"], "evaluation_years": [2025]},
    )
    comparison = pl.read_parquet(study / "analysis/controlled_comparisons.parquet")
    rookie = comparison.filter(
        (pl.col("scope") == "evaluation")
        & (pl.col("cohort") == "known_current_draft_year")
        & (pl.col("target") == "points")
    )
    assert rookie["n"].to_list() == [1]
    assert rookie["mse_improvement_pct"].to_list() == [0.0]
