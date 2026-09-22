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


def test_walk_forward_stack_uses_only_earlier_out_of_fold_predictions() -> None:
    """A later ridge may safely learn from an earlier model's historical forecasts."""
    base = ModelSpec(
        name="base",
        features=("signal",),
        lambda_grid=(1e-6,),
    )
    stack = ModelSpec(
        name="stack",
        features=("base", "other"),
        require_features=("base",),
        lambda_grid=(1e-6,),
    )
    config = FitConfig(
        models=(base, stack),
        min_train_folds=3,
        min_position_rows=10,
        positions=("RB", "WR"),
    )

    fitted, models = fit_walk_forward(synthetic(), config)
    scored = {
        row["forecast_season"]: (row["base"], row["stack"])
        for row in fitted.group_by("forecast_season")
        .agg(
            pl.col("base").is_not_null().any(),
            pl.col("stack").is_not_null().any(),
        )
        .to_dicts()
    }

    assert scored[2007] == (True, False)
    assert scored[2009] == (True, False)
    assert scored[2010] == (True, True)
    assert scored[2011] == (True, True)
    first_stack = next(m for m in models if m["model"] == "stack")
    assert first_stack["forecast_season"] == 2010
    assert first_stack["train_seasons"] == [2007, 2009]


def test_position_selector_uses_configured_source_with_fallback() -> None:
    rb = ModelSpec(name="rb_model", features=("signal",), lambda_grid=(1e-6,))
    wr = ModelSpec(name="wr_model", features=("other",), lambda_grid=(1e-6,))
    selector = ModelSpec(
        name="hybrid",
        kind="position_select",
        by_position=(("RB", "rb_model"), ("WR", "wr_model")),
        fallback="rb_model",
        apply_live=False,
    )
    config = FitConfig(
        models=(rb, wr, selector),
        min_train_folds=3,
        min_position_rows=10,
        positions=("RB", "WR"),
    )

    fitted, _ = fit_walk_forward(synthetic(), config)
    scored = fitted.filter(pl.col("forecast_season") == 2011)

    assert scored.filter(pl.col("position") == "RB")["hybrid"].to_list() == pytest.approx(
        scored.filter(pl.col("position") == "RB")["rb_model"].to_list()
    )
    assert scored.filter(pl.col("position") == "WR")["hybrid"].to_list() == pytest.approx(
        scored.filter(pl.col("position") == "WR")["wr_model"].to_list()
    )


def test_coalesce_combines_mutually_exclusive_population_models() -> None:
    frame = synthetic().with_columns(
        pl.when(pl.col("position") == "RB").then(pl.col("signal")).alias("rookie"),
        pl.when(pl.col("position") == "WR").then(pl.col("other")).alias("returner"),
    )
    config = FitConfig(
        models=(
            ModelSpec(
                name="combined",
                kind="coalesce",
                factors=("rookie", "returner"),
                apply_live=False,
            ),
        ),
        positions=("RB", "WR"),
    )

    fitted, _ = fit_walk_forward(frame, config)

    assert fitted.filter(pl.col("position") == "RB")["combined"].to_list() == pytest.approx(
        fitted.filter(pl.col("position") == "RB")["rookie"].to_list()
    )
    assert fitted.filter(pl.col("position") == "WR")["combined"].to_list() == pytest.approx(
        fitted.filter(pl.col("position") == "WR")["returner"].to_list()
    )


def test_adaptive_selector_uses_only_prior_fold_ranking_results() -> None:
    frame = synthetic().with_columns(
        pl.col("actual_ppg").alias("good_ranker"),
        (-pl.col("actual_ppg")).alias("bad_ranker"),
    )
    selector = ModelSpec(
        name="adaptive",
        kind="adaptive_select",
        target="actual_ppg",
        factors=("bad_ranker", "good_ranker"),
        fallback="bad_ranker",
        selection_top_k=(("RB", 2), ("WR", 2)),
        min_train_folds=3,
        apply_live=False,
    )
    config = FitConfig(
        models=(selector,), min_train_folds=3, min_position_rows=10, positions=("RB", "WR")
    )

    fitted, records = fit_walk_forward(frame, config)
    pending = fitted.filter(pl.col("forecast_season") == 2011)

    assert pending["adaptive"].to_list() == pytest.approx(pending["good_ranker"].to_list())
    latest = [
        entry
        for entry in summarize_fitted_models(records)
        if entry["model"] == "adaptive" and entry["position"] == "RB"
    ][0]
    assert latest["selected_source"] == "good_ranker"
    assert latest["selection_folds"] == 7


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


def test_required_features_restrict_training_and_scoring() -> None:
    """A feature that exists only from some season trains and scores only where present."""
    from patron.metrics.fit import ModelSpec

    frame = synthetic().with_columns(
        pl.when(pl.col("forecast_season") >= 2008)
        .then(pl.col("signal") * 3.0)
        .otherwise(None)
        .alias("market")
    )
    spec = ModelSpec(
        name="fitted_market",
        features=("signal", "market"),
        require_features=("market",),
        min_train_folds=2,
        lambda_grid=(1e-6,),
    )
    config = FitConfig(
        models=(spec,), min_train_folds=3, positions=("RB", "WR"), min_position_rows=10
    )
    fitted, models = fit_walk_forward(frame, config)
    by_season = {
        row["forecast_season"]: row["scored"]
        for row in fitted.group_by("forecast_season")
        .agg(pl.col("fitted_market").is_not_null().any().alias("scored"))
        .to_dicts()
    }
    # Market rows exist 2008+; two prior market folds are needed, so 2010 is the first
    # scored season, and pre-market rows are never scored.
    assert not by_season[2007] and not by_season[2009]
    assert by_season[2010] and by_season[2011]
    assert all(m["train_seasons"][0] >= 2008 for m in models)


def test_prefix_feature_pool_is_selected_inside_each_outer_training_window() -> None:
    frame = synthetic().with_columns(
        (pl.col("actual_ppg") * 2.0).alias("rich_good"),
        (pl.col("noise") % 3).alias("rich_noise"),
    )
    spec = ModelSpec(
        name="selected_rich",
        features=("other",),
        feature_prefixes=("rich_",),
        max_features=2,
        lambda_grid=(1.0,),
    )
    config = FitConfig(
        models=(spec,), min_train_folds=3, min_position_rows=10, positions=("RB", "WR")
    )

    fitted, models = fit_walk_forward(frame, config)

    assert fitted.filter(pl.col("forecast_season") == 2011)["selected_rich"].is_not_null().all()
    latest = max(
        (
            model
            for model in models
            if model["model"] == "selected_rich" and model["position"] == "RB"
        ),
        key=lambda model: model["forecast_season"],
    )
    assert "rich_good" in latest["coefficients"]
    assert len(latest["coefficients"]) <= 2
