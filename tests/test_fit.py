"""Walk-forward fitted ranker: learns weights, never sees its own outcome."""

from __future__ import annotations

import polars as pl
import pytest

from patron.metrics.fit import (
    FitConfig,
    ModelSpec,
    apply_fitted_models,
    fit_ridge,
    fit_walk_forward,
    models_for_season,
    summarize_fitted_models,
)


def synthetic() -> pl.DataFrame:
    """Outcome = 2 × signal + 1 × other for 2004–2010, with a pending 2011 fold."""
    rows = []
    for season in range(2004, 2012):
        complete = season < 2011
        for index in range(40):
            signal = float(index % 10)
            other = float((index * 7) % 5)
            rows.append(
                {
                    "forecast_season": season,
                    "outcome_complete": complete,
                    "position": "WR" if index % 2 else "RB",
                    "signal": signal,
                    "other": other,
                    "noise": float((index * 13) % 3),
                    "expected_games": 16.0,
                    "actual_games": 16.0 if complete else None,
                    "actual_ppg": 2.0 * signal + other + (0.01 * season) if complete else None,
                }
            )
    return pl.DataFrame(rows)


def test_ridge_recovers_the_generating_weights() -> None:
    frame = synthetic().filter(pl.col("outcome_complete"))
    model = fit_ridge(frame.to_dicts(), ("signal", "other", "noise"), "actual_ppg", 1e-6)

    assert model is not None
    predicted = [model.predict(row) for row in frame.head(5).to_dicts()]
    assert predicted == pytest.approx(frame.head(5)["actual_ppg"].to_list(), abs=0.05)
    # Standardized weights: signal (sd ~2.87, slope 2) dominates other (sd ~1.4, slope 1).
    weights = dict(zip(model.features, model.coefficients, strict=True))
    assert weights["signal"] > weights["other"] > 0
    assert abs(weights["noise"]) < 0.05


def test_walk_forward_scores_every_fold_only_from_earlier_outcomes() -> None:
    config = FitConfig(
        features=("signal", "other", "noise"),
        target="actual_ppg",
        ridge_lambda=1e-6,
        min_train_folds=3,
        min_position_rows=10,
        positions=("RB", "WR"),
    )
    fitted, models = fit_walk_forward(synthetic(), config)

    by_season = fitted.group_by("forecast_season").agg(
        pl.col("fitted_ppg").is_null().all().alias("all_null"),
        pl.col("fitted_season_points").max().alias("max_pts"),
    )
    seasons = {row["forecast_season"]: row for row in by_season.to_dicts()}
    assert seasons[2004]["all_null"] and seasons[2006]["all_null"]  # fewer than 3 prior folds
    assert not seasons[2007]["all_null"]
    assert not seasons[2011]["all_null"]  # the pending fold is scored from all completed
    assert seasons[2011]["max_pts"] == pytest.approx(
        16.0 * fitted.filter(pl.col("forecast_season") == 2011)["fitted_ppg"].max()
    )

    scored = fitted.filter(pl.col("outcome_complete") & pl.col("fitted_ppg").is_not_null())
    error = (scored["fitted_ppg"] - scored["actual_ppg"]).abs().mean()
    assert error < 0.1

    train_ranges = {
        (m["forecast_season"], m["position"]): m["train_seasons"]
        for m in models
        if m["model"] == "fitted_ppg"
    }
    assert train_ranges[(2007, "RB")] == [2004, 2006]
    assert train_ranges[(2011, "WR")] == [2004, 2010]
    assert all(m["scope"] == "position" for m in models)

    summary = [entry for entry in summarize_fitted_models(models) if entry["model"] == "fitted_ppg"]
    assert {entry["position"] for entry in summary} == {"RB", "WR"}
    assert summary[0]["latest_forecast_season"] == 2011
    assert summary[0]["refits"] == 5
    assert summary[0]["coefficient_sd_across_refits"]["signal"] < 0.05


def test_missing_features_are_skipped_not_fatal() -> None:
    frame = synthetic()
    fitted, models = fit_walk_forward(frame, FitConfig(features=("absent",)))
    assert not [m for m in models if m["model"] == "fitted_ppg"]
    assert "fitted_ppg" not in fitted.columns
    assert "fitted_season_points" not in fitted.columns  # product needs fitted_ppg


def test_serialized_pending_models_reproduce_the_fold_scores() -> None:
    """Live-board parity: re-applying the stored artifact gives the parquet's numbers."""
    from patron.metrics.fit import apply_fitted_models, models_for_season

    config = FitConfig(
        features=("signal", "other", "noise"),
        ridge_lambda=0.5,
        min_train_folds=3,
        min_position_rows=10,
        positions=("RB", "WR"),
    )
    fitted, models = fit_walk_forward(synthetic(), config)
    pending = fitted.filter(pl.col("forecast_season") == 2011)
    reapplied = apply_fitted_models(
        pending.drop(list(config.output_columns)),
        models_for_season(models, 2011),
        config,
    )
    assert reapplied["fitted_ppg"].to_list() == pytest.approx(
        pending["fitted_ppg"].to_list(), abs=1e-9
    )
    assert reapplied["fitted_season_points"].to_list() == pytest.approx(
        pending["fitted_season_points"].to_list(), abs=1e-9
    )


def test_report_only_model_is_fitted_but_not_applied_to_live_board() -> None:
    spec = ModelSpec(
        name="research",
        features=("signal",),
        lambda_grid=(1.0,),
        apply_live=False,
    )
    config = FitConfig(
        models=(spec,), min_train_folds=3, min_position_rows=10, positions=("RB", "WR")
    )
    fitted, records = fit_walk_forward(synthetic(), config)
    pending = fitted.filter(pl.col("forecast_season") == 2011)

    assert pending["research"].is_not_null().all()
    live = apply_fitted_models(pending.drop("research"), models_for_season(records, 2011), config)
    assert live["research"].is_null().all()


def test_artifact_contract_rejects_mismatched_models() -> None:
    from patron.metrics.fit import FittedArtifactError, build_fit_artifact, check_fit_artifact

    config = FitConfig(features=("signal",), positions=("RB", "WR"))
    frame = synthetic().with_columns(pl.lit("2011-08-30T12:00:00Z").alias("depth_chart_date"))
    artifact = build_fit_artifact(config, frame, "08-31")
    assert artifact["pending_forecast_season"] == 2011
    assert artifact["depth_chart_latest"] == "2011-08-30T12:00:00Z"

    check_fit_artifact(artifact, config, 2011, "08-31", "2011-08-30T12:00:00Z")
    with pytest.raises(FittedArtifactError, match="fingerprint"):
        check_fit_artifact(artifact, FitConfig(features=("other",)), 2011, "08-31", None)
    with pytest.raises(FittedArtifactError, match="fitted for 2011"):
        check_fit_artifact(artifact, config, 2012, "08-31", None)
    with pytest.raises(FittedArtifactError, match="cutoff"):
        check_fit_artifact(artifact, config, 2011, "09-05", None)
    with pytest.raises(FittedArtifactError, match="snapshot"):
        check_fit_artifact(artifact, config, 2011, "08-31", "2011-03-14T07:32:09Z")
    with pytest.raises(FittedArtifactError, match="no fitted-model artifact"):
        check_fit_artifact({}, config, 2011, "08-31", None)
