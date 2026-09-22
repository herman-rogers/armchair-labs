from __future__ import annotations

import polars as pl

from patron.metrics.discovery import (
    DISCOVERY_COLUMNS,
    DiscoveryConfig,
    automated_feature_pool,
    build_weekly_discovery_features,
    fit_discovery_walk_forward,
)


def test_weekly_features_recover_late_growth_and_add_random_convolutions() -> None:
    rows = []
    for week in range(1, 19):
        rows.append(
            {
                "season": 2025,
                "week": week,
                "player_id": "ascending",
                "position": "WR",
                "fantasy_points_ppr": float(week),
                "targets": week,
                "carries": 0,
                "attempts": 0,
                "rushing_yards": 0,
                "receiving_yards": week * 10,
            }
        )
    result = build_weekly_discovery_features(
        pl.DataFrame(rows),
        [2026],
        config=DiscoveryConfig(random_kernels_per_signal=2),
    )

    assert result["forecast_season"].item() == 2026
    assert result["ts_fantasy_points_ppr_slope"].item() > 0
    assert result["ts_targets_late_delta"].item() > 0
    assert any(name.startswith("conv_fantasy_points_ppr") for name in result.columns)
    assert not any(name.startswith("_ts_") for name in result.columns)


def _synthetic_predictions() -> pl.DataFrame:
    rows = []
    for season in range(2004, 2013):
        for index in range(12):
            signal = float(index + (season - 2004) * 0.1)
            actual = 30.0 + signal * 12.0
            rows.append(
                {
                    "player_id": f"p{index}",
                    "forecast_season": season,
                    "source_season": season - 1,
                    "position": "QB",
                    "outcome_complete": season < 2012,
                    "actual_season_points": actual if season < 2012 else 0.0,
                    "actual_availability_value": actual if season < 2012 else 0.0,
                    "actual_ppg": actual / 17 if season < 2012 else None,
                    "ppg": signal,
                    "season_pts": signal * 17,
                    "games": 17.0,
                    "age_at_season": 24.0 + index / 10,
                    "fitted_season_points": actual * 0.9 + index % 3 if season >= 2007 else None,
                    "fitted_two_stage": actual * 0.88 + index % 2 if season >= 2007 else None,
                    "market_ecr_score": float(12 - index),
                }
            )
    return pl.DataFrame(rows)


def test_automated_pool_rejects_outcomes_fitted_market_and_handcrafted_outputs() -> None:
    frame = _synthetic_predictions()
    pool = automated_feature_pool(frame)

    assert "ppg" in pool
    assert "age_at_season" in pool
    assert "actual_season_points" not in pool
    assert "fitted_season_points" not in pool
    assert "market_ecr_score" not in pool


def test_walk_forward_predictions_do_not_change_when_future_outcomes_change() -> None:
    frame = _synthetic_predictions()
    config = DiscoveryConfig(
        min_train_folds=2,
        min_position_rows=12,
        inner_validation_folds=1,
        feature_counts=(2, 4),
        temporal_feature_counts=(2,),
        ridge_alphas=(1.0,),
        latent_components=2,
        latent_clusters=2,
        symbolic_base_features=2,
        symbolic_features=3,
    )
    first, _ = fit_discovery_walk_forward(frame, config=config, top_k={"QB": 4})
    changed = frame.with_columns(
        pl.when(pl.col("forecast_season") >= 2011)
        .then(pl.col("actual_season_points") * -1000)
        .otherwise(pl.col("actual_season_points"))
        .alias("actual_season_points")
    )
    second, _ = fit_discovery_walk_forward(changed, config=config, top_k={"QB": 4})

    left = first.filter(pl.col("forecast_season") == 2010).select(DISCOVERY_COLUMNS[:6])
    right = second.filter(pl.col("forecast_season") == 2010).select(DISCOVERY_COLUMNS[:6])
    assert left.equals(right, null_equal=True)
    assert first.filter(pl.col("forecast_season") == 2010)["discovery_linear"].is_not_null().all()
