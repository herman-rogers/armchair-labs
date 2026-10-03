"""Prospective forecasts are frozen before outcomes and graded only afterward."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from engine.board.pipeline import build_adaptive_board
from engine.config.league import get_league
from engine.metrics.backtest import MetricReportConfig
from engine.metrics.prospective import (
    SNAPSHOT_COLUMNS,
    ProspectiveGradePending,
    frozen_snapshot,
    grade_frozen_forecast,
    snapshot_sha256,
)


def _pending_predictions() -> pl.DataFrame:
    rows = []
    for position in ("QB", "RB", "WR", "TE"):
        for suffix, good in (("a", True), ("b", False)):
            score = 100.0 if good else 1.0
            inverse = 1.0 if good else 100.0
            rows.append(
                {
                    "forecast_season": 2026,
                    "player_id": f"{position}-{suffix}",
                    "player_display_name": f"{position} {suffix}",
                    "position": position,
                    "games": 17,
                    "historical_ppg_prior": inverse,
                    "ppg": inverse,
                    "season_pts": inverse,
                    "market_ecr_score": inverse,
                    "market_snapshot": "2026-08-30",
                    "preseason_rostered": 1.0,
                    "preseason_reserve": 0.0,
                    "cutoff_transaction_matched": 1.0,
                    "fitted_season_points": inverse,
                    "fitted_two_stage": inverse,
                    "fitted_adaptive_ppg_hybrid": score,
                    "fitted_adaptive_season_hybrid": score,
                    "outcome_complete": False,
                }
            )
    return pl.DataFrame(rows)


def _completed_outcomes() -> pl.DataFrame:
    rows = []
    for position in ("QB", "RB", "WR", "TE"):
        for suffix, good in (("a", True), ("b", False)):
            actual = 100.0 if good else 0.0
            rows.append(
                {
                    "forecast_season": 2026,
                    "player_id": f"{position}-{suffix}",
                    "actual_ppg": actual,
                    "actual_vor": actual,
                    "actual_games": 17.0 if good else 0.0,
                    "actual_season_points": actual,
                    "actual_availability_value": actual,
                    "outcome_complete": True,
                }
            )
    return pl.DataFrame(rows)


def _grading_config() -> MetricReportConfig:
    config = MetricReportConfig.from_config()
    ranking = replace(
        config.ranking,
        top_k={position: 1 for position in ("QB", "RB", "WR", "TE")},
        pool={position: 2 for position in ("QB", "RB", "WR", "TE")},
        overall={
            **config.ranking.overall,
            "k": 4,
            "pool": 8,
            "min_games": 0,
            "replacement_ranks": {position: 2 for position in ("QB", "RB", "WR", "TE")},
        },
    )
    return replace(config, ranking=ranking)


def test_frozen_snapshot_is_minimal_sorted_and_pending() -> None:
    snapshot = frozen_snapshot(_pending_predictions().reverse(), 2026)

    assert snapshot.columns == list(SNAPSHOT_COLUMNS)
    assert snapshot["player_id"].to_list() == sorted(snapshot["player_id"].to_list())
    assert len(snapshot_sha256(snapshot)) == 64


def test_grade_refuses_to_peek_before_the_declared_date() -> None:
    snapshot = frozen_snapshot(_pending_predictions(), 2026)

    with pytest.raises(ProspectiveGradePending, match="cannot be graded before"):
        grade_frozen_forecast(
            snapshot,
            _completed_outcomes(),
            _grading_config(),
            2026,
            expected_sha256=snapshot_sha256(snapshot),
            grade_not_before=date(2027, 1, 15),
            today=date(2026, 8, 30),
        )


def test_grade_uses_frozen_rankers_and_fresh_outcomes() -> None:
    def expand(frame: pl.DataFrame) -> pl.DataFrame:
        return pl.concat(
            [
                frame.with_columns((pl.col("player_id") + pl.lit(f"-{i}")).alias("player_id"))
                for i in range(15)
            ]
        )

    snapshot = frozen_snapshot(expand(_pending_predictions()), 2026)
    outcomes = expand(_completed_outcomes())
    report = grade_frozen_forecast(
        snapshot,
        outcomes,
        _grading_config(),
        2026,
        expected_sha256=snapshot_sha256(snapshot),
        grade_not_before=date(2027, 1, 15),
        today=date(2027, 1, 15),
    )
    # The mutated k=4 test config is ignored in favor of frozen overall k=60.
    assert report["frozen_rows"] == 120
    assert report["passes_primary_gates"] is True
    assert report["overall"]["adaptive_hit_rate"] > report["overall"]["incumbent_hit_rate"]
    assert report["overall"]["adaptive_hit_rate"] > report["overall"]["ecr_hit_rate"]
    assert all(row["k"] == 60 for row in report["ranking_results"] if row["position"] == "ALL")


def test_adaptive_board_ranks_the_live_pool_on_frozen_scores() -> None:
    pending = _pending_predictions()
    snapshot = frozen_snapshot(pending, 2026)
    board = pending.select("player_id", "player_display_name", "position", "games").with_columns(
        pl.lit(0.0).alias("override_delta"),
        pl.lit("v2").alias("metric_version"),
    )

    ranked = build_adaptive_board(
        board,
        snapshot,
        get_league(),
        selected_sources={position: "frozen_source" for position in ("QB", "RB", "WR", "TE")},
    )
    qb = ranked.filter(pl.col("position") == "QB").sort("v2_position_rank").to_dicts()

    assert ranked["metric_version"].unique().to_list() == ["adaptive"]
    assert qb[0]["player_id"] == "QB-a"
    assert qb[0]["v2_rank_key"] == "adaptive_season_points"
    assert qb[0]["adaptive_selected_source"] == "frozen_source"
    assert qb[0]["season_equivalent_ppg"] == pytest.approx(100 / 17)
