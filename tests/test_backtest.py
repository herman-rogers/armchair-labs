"""Metric report statistics stay configurable and season-aware."""

from __future__ import annotations

import polars as pl

from patron.metrics.backtest import (
    AnalysisWindow,
    MetricDefinition,
    MetricReportConfig,
    RankingConfig,
    TargetDefinition,
    _assessment,
    _depth_chart_before,
    _finalize_entries,
    analyze_market_disagreements,
    analyze_predictions,
    analyze_rankings,
    build_metric_report,
    render_metric_report_markdown,
)


def test_sparse_market_baseline_cannot_win_on_a_favorable_subset() -> None:
    def fold(season: int, hit_rate: float) -> dict[str, float | int | None]:
        return {
            "forecast_season": season,
            "hit_rate": hit_rate,
            "ndcg": hit_rate,
            "pool_spearman": hit_rate,
            "top_k_actual_mean": hit_rate,
            "ideal_top_k_actual_mean": 1.0,
        }

    def entry(name: str, role: str, folds: list[dict[str, float | int | None]]) -> dict:
        return {
            "ranker": name,
            "role": role,
            "hit_rate": sum(float(f["hit_rate"] or 0) for f in folds) / len(folds),
            "ndcg": sum(float(f["ndcg"] or 0) for f in folds) / len(folds),
            "pool_spearman": 0.0,
            "top_k_actual_mean": 0.0,
            "ideal_top_k_actual_mean": 1.0,
            "fold_results": folds,
        }

    finalized = _finalize_entries(
        {
            "production": entry("production", "baseline", [fold(2020, 0.5), fold(2021, 0.5)]),
            "market": entry("market", "baseline", [fold(2021, 0.9)]),
            "candidate": entry("candidate", "candidate", [fold(2020, 0.6), fold(2021, 0.6)]),
        }
    )
    candidate = next(row for row in finalized if row["ranker"] == "candidate")

    assert candidate["best_baseline"] == "production"
    assert candidate["hit_rate_lift"] == 0.1


def report_config() -> MetricReportConfig:
    return MetricReportConfig(
        title="Test report",
        forecast_seasons=(2023, 2024, 2025),
        history_seasons=3,
        depth_chart_cutoff="08-31",
        minimum_sample=4,
        baseline_metric="prior",
        targets=(TargetDefinition("actual_ppg", "Actual PPG", "Outcome"),),
        metrics=(
            MetricDefinition(
                key="signal",
                label="Signal",
                group="Test",
                description="Perfectly ordered signal",
                targets=("actual_ppg",),
                prediction_target="actual_ppg",
            ),
            MetricDefinition(
                key="prior",
                label="Prior",
                group="Baseline",
                description="Baseline",
                targets=("actual_ppg",),
            ),
        ),
    )


def predictions() -> pl.DataFrame:
    rows = []
    for season in (2023, 2024):
        for index in range(1, 6):
            rows.append(
                {
                    "forecast_season": season,
                    "outcome_complete": True,
                    "position": "WR",
                    "signal": float(index),
                    "prior": float(6 - index if season == 2023 else index % 2),
                    "actual_ppg": float(index),
                }
            )
    rows.append(
        {
            "forecast_season": 2025,
            "outcome_complete": False,
            "position": "WR",
            "signal": 10.0,
            "prior": 10.0,
            "actual_ppg": None,
        }
    )
    return pl.DataFrame(rows)


def test_analysis_keeps_fold_results_and_incremental_signal() -> None:
    results, models = analyze_predictions(predictions(), report_config())
    signal = next(row for row in results if row["metric"] == "signal" and row["position"] == "ALL")

    assert signal["n"] == 10
    assert signal["folds"] == 2
    assert signal["spearman"] == 1.0
    assert signal["partial_spearman"] is not None
    assert [fold["forecast_season"] for fold in signal["fold_results"]] == [2023, 2024]
    assert models[0]["mae"] == 0.0


def test_report_distinguishes_completed_and_pending_forecasts() -> None:
    report = build_metric_report(predictions(), report_config())

    assert report["data_summary"]["completed_forecasts"] == [2023, 2024]
    assert report["data_summary"]["pending_forecasts"] == [2025]
    assert report["metrics"][0]["available"] is True
    assert report["metrics"][0]["coverage"] == 1.0


def test_assessment_marks_repeatable_wrong_sign_as_harmful() -> None:
    assert _assessment(-0.2, -0.15, 0.75, 100, 4, 20, "positive") == "harmful"
    assert _assessment(-0.2, -0.15, 0.75, 100, 4, 20, "negative") == "useful"


def test_depth_chart_cutoff_parses_iso_timestamps() -> None:
    depth = pl.DataFrame(
        {
            "dt": ["2025-08-30T12:00:00Z", "2025-09-02T12:00:00Z"],
            "team": ["OLD", "LATE"],
            "gsis_id": ["player", "player"],
            "pos_abb": ["WR", "WR"],
            "pos_name": ["Wide Receiver", "Wide Receiver"],
            "pos_rank": [2, 1],
        }
    )

    result = _depth_chart_before(depth, 2025, "08-31")

    assert result is not None
    assert result.to_dicts()[0]["current_team"] == "OLD"


def test_windows_and_source_start_dates_limit_evidence() -> None:
    config = report_config()
    gated_metric = MetricDefinition(
        key="signal",
        label="Signal",
        group="Test",
        description="Available in the second fold",
        targets=("actual_ppg",),
        positions=("WR",),
        available_from_forecast=2024,
    )
    config = MetricReportConfig(
        **{
            **config.__dict__,
            "metrics": (gated_metric,),
            "analysis_windows": (
                AnalysisWindow("long", "Long", 2023, 2024),
                AnalysisWindow("recent", "Recent", 2024, 2024),
            ),
        }
    )

    results, _ = analyze_predictions(predictions(), config)

    assert {row["window"] for row in results} == {"long", "recent"}
    assert all(row["n"] == 5 for row in results)
    assert all(
        [fold["forecast_season"] for fold in row["fold_results"]] == [2024] for row in results
    )


def ranking_predictions() -> pl.DataFrame:
    """Two folds of eight WRs. `good` orders them correctly; `bad` reverses them."""
    rows = []
    for season in (2023, 2024):
        for index in range(1, 9):
            rows.append(
                {
                    "forecast_season": season,
                    "outcome_complete": True,
                    "position": "WR",
                    "good": float(index),
                    "bad": float(9 - index),
                    "actual_ppg": float(index),
                    "actual_season_points": float(index) * 10.0,
                }
            )
    return pl.DataFrame(rows)


def ranking_config() -> MetricReportConfig:
    return MetricReportConfig(
        title="Ranking test",
        forecast_seasons=(2023, 2024, 2025),
        history_seasons=3,
        depth_chart_cutoff="08-31",
        minimum_sample=4,
        baseline_metric="good",
        targets=(
            TargetDefinition("actual_ppg", "Actual PPG", "Outcome"),
            TargetDefinition("actual_season_points", "Season points", "Outcome", True),
        ),
        metrics=(),
        ranking=RankingConfig(
            baselines=("good",),
            candidates=("bad", "missing"),
            targets=("actual_season_points",),
            top_k={"WR": 3},
            pool={"WR": 6},
        ),
    )


def test_ranking_scores_top_k_against_the_best_baseline() -> None:
    results = analyze_rankings(ranking_predictions(), ranking_config())
    by_ranker = {row["ranker"]: row for row in results}

    assert set(by_ranker) == {"good", "bad"}  # `missing` is not a column and is skipped
    good, bad = by_ranker["good"], by_ranker["bad"]
    assert good["role"] == "baseline" and good["hit_rate"] == 1.0 and good["ndcg"] == 1.0
    assert good["beats_baseline"] is None
    assert bad["role"] == "candidate"
    assert bad["hit_rate"] == 0.0
    assert bad["hit_rate_lift"] == -1.0
    assert bad["best_baseline"] == "good"
    assert bad["folds_beating_baseline"] == 0
    assert bad["beats_baseline"] is False
    assert bad["pool_spearman"] == -1.0
    assert [fold["forecast_season"] for fold in bad["fold_results"]] == [2023, 2024]
    assert bad["top_k_actual_mean"] == 20.0 and bad["ideal_top_k_actual_mean"] == 70.0


def test_ranking_verdict_renders_in_markdown() -> None:
    report = build_metric_report(ranking_predictions(), ranking_config())
    markdown = render_metric_report_markdown(report)

    assert report["ranking_results"]
    assert "Does v2 beat the naive baseline?" in markdown
    assert "WR | bad | candidate | 2 | 0.000" in markdown
    assert "loses" in markdown


def test_market_disagreements_use_the_common_pool_and_score_two_round_calls() -> None:
    rows = []
    for season in (2023, 2024):
        for index in range(1, 9):
            rows.append(
                {
                    "player_id": f"{season}-{index}",
                    "forecast_season": season,
                    "outcome_complete": True,
                    "position": "WR",
                    "games": 17.0,
                    "model": float(index),
                    "market": float(9 - index),
                    "actual_availability_value": float(index),
                }
            )
    cfg = MetricReportConfig(
        title="moneyball",
        forecast_seasons=(2023, 2024, 2025),
        history_seasons=3,
        depth_chart_cutoff="08-31",
        minimum_sample=4,
        baseline_metric="market",
        targets=(TargetDefinition("actual_availability_value", "Value", "", True),),
        metrics=(),
        analysis_windows=(AnalysisWindow("market", "Market", 2023, 2024),),
        ranking=RankingConfig(
            baselines=("market",),
            candidates=("model",),
            targets=("actual_availability_value",),
            top_k={"WR": 3},
            pool={"WR": 6},
            market_baseline="market",
            market_price_ranker="market",
            disagreement_round_size=1,
            disagreement_rounds=2,
            overall={
                "k": 3,
                "pool": 6,
                "target": "actual_availability_value",
                "replacement_ranks": {"WR": 3},
                "min_games": 0,
                "season_games": 17,
            },
        ),
    )

    result = analyze_market_disagreements(pl.DataFrame(rows), cfg)[0]

    assert result["common_players_mean"] == 8.0
    assert result["hit_rate_lift"] == 1.0
    assert result["model_only_hits"] == 6
    assert result["market_only_hits"] == 0
    assert result["contrarian_precision"] == 1.0
    assert result["missed_value_capture_rate"] == 1.0
    assert result["net_swap_value"] > 0
    assert result["value_capture_bands"][0] == {
        "rank_gap": 0,
        "calls": 6,
        "hits": 6,
        "precision": 1.0,
        "actual_value_sum": 42.0,
        "false_positive_cost": 0.0,
    }
    assert result["bullish_correct_rate"] == 1.0


def test_directional_improvement_is_not_counted_as_captured_draft_value() -> None:
    rows = []
    market_order = ["a", "b", "c", "d", "e", "f"]
    model_order = ["c", "d", "a", "b", "e", "f"]
    actual_order = ["c", "a", "d", "b", "e", "f"]
    for player_id in market_order:
        rows.append(
            {
                "player_id": player_id,
                "forecast_season": 2024,
                "outcome_complete": True,
                "position": "WR",
                "games": 17.0,
                "market": float(7 - market_order.index(player_id)),
                "model": float(7 - model_order.index(player_id)),
                "actual_availability_value": float(7 - actual_order.index(player_id)),
            }
        )
    cfg = MetricReportConfig(
        title="value, not direction",
        forecast_seasons=(2024,),
        history_seasons=3,
        depth_chart_cutoff="08-31",
        minimum_sample=2,
        baseline_metric="market",
        targets=(TargetDefinition("actual_availability_value", "Value", "", True),),
        metrics=(),
        analysis_windows=(AnalysisWindow("market", "Market", 2024, 2024),),
        ranking=RankingConfig(
            baselines=("market",),
            candidates=("model",),
            targets=("actual_availability_value",),
            market_price_ranker="market",
            disagreement_round_size=1,
            disagreement_rounds=1,
            overall={
                "k": 2,
                "pool": 6,
                "target": "actual_availability_value",
                "replacement_ranks": {"WR": 2},
                "min_games": 0,
                "season_games": 17,
            },
        ),
    )

    result = analyze_market_disagreements(pl.DataFrame(rows), cfg)[0]

    # D improved from market rank 4 to actual rank 3, so the old directional test
    # calls both C and D correct. Only C actually entered the draftable top-two tier.
    assert result["bullish_correct_rate"] == 1.0
    assert result["contrarian_precision"] == 0.5
    assert result["model_captured_market_misses"] == 1


def test_baselines_are_target_matched_and_wtl_is_strict() -> None:
    """A season-points outcome is compared against last season's points too, and a
    candidate that merely ties the baseline does not pass."""
    rows = []
    for season in (2023, 2024):
        for index in range(1, 9):
            rows.append(
                {
                    "forecast_season": season,
                    "outcome_complete": True,
                    "position": "WR",
                    "prior": float(index),
                    "season_pts": float(index) * 10.0,
                    "same_as_prior": float(index),
                    "actual_season_points": float(index) * 10.0,
                }
            )
    frame = pl.DataFrame(rows)
    cfg = MetricReportConfig(
        title="t",
        forecast_seasons=(2023, 2024, 2025),
        history_seasons=3,
        depth_chart_cutoff="08-31",
        minimum_sample=4,
        baseline_metric="prior",
        targets=(TargetDefinition("actual_season_points", "Pts", "", True),),
        metrics=(),
        ranking=RankingConfig.from_raw(
            {
                "baselines": {
                    "default": ["prior"],
                    "actual_season_points": ["prior", "season_pts"],
                },
                "candidates": ["same_as_prior"],
                "targets": ["actual_season_points"],
                "top_k": {"WR": 3},
                "pool": {"WR": 6},
            }
        ),
    )
    assert cfg.ranking.baselines_for("actual_season_points") == ("prior", "season_pts")
    assert cfg.ranking.baselines_for("actual_ppg") == ("prior",)

    results = {row["ranker"]: row for row in analyze_rankings(frame, cfg)}
    assert set(results) == {"prior", "season_pts", "same_as_prior"}
    tie = results["same_as_prior"]
    assert tie["hit_rate_lift"] == 0.0
    assert (tie["folds_won"], tie["folds_tied"], tie["folds_lost"]) == (0, 2, 0)
    assert tie["beats_baseline"] is False
    assert tie["hit_rate_lift_se"] == 0.0


def test_depth_chart_selection_ignores_stale_seasons() -> None:
    """A player whose last chart entry is from an old season is not 'current'."""
    depth = pl.DataFrame(
        {
            "dt": ["2019-08-31T00:00:00Z", "2025-08-31T00:00:00Z", "2026-08-29T12:00:00Z"],
            "team": ["OLD", "LASTYEAR", "NOW"],
            "gsis_id": ["stale", "lastyear", "current"],
            "pos_abb": ["RB", "RB", "RB"],
            "pos_name": ["Running Back"] * 3,
            "pos_rank": [1, 1, 1],
        }
    )
    result = _depth_chart_before(depth, 2026, "08-31")
    assert result is not None
    assert result["player_id"].to_list() == ["current"]


def test_candidates_are_scored_head_to_head_against_the_market_on_its_folds() -> None:
    rows = []
    for season in (2023, 2024, 2025):
        for index in range(1, 9):
            rows.append(
                {
                    "forecast_season": season,
                    "outcome_complete": True,
                    "position": "WR",
                    "prior": float(index),
                    "market": float(index) if season >= 2024 else None,
                    "reversed": float(9 - index),
                    "actual_season_points": float(index) * 10.0,
                }
            )
    cfg = MetricReportConfig(
        title="t",
        forecast_seasons=(2023, 2024, 2025, 2026),
        history_seasons=3,
        depth_chart_cutoff="08-31",
        minimum_sample=4,
        baseline_metric="prior",
        targets=(TargetDefinition("actual_season_points", "Pts", "", True),),
        metrics=(),
        ranking=RankingConfig.from_raw(
            {
                "baselines": ["prior", "market"],
                "candidates": ["reversed"],
                "targets": ["actual_season_points"],
                "market_baseline": "market",
                "top_k": {"WR": 3},
                "pool": {"WR": 6},
            }
        ),
    )
    rows_out = {r["ranker"]: r for r in analyze_rankings(pl.DataFrame(rows), cfg)}
    reversed_row = rows_out["reversed"]
    # Lift is against `prior` (the only baseline covering all three candidate folds)...
    assert reversed_row["best_baseline"] == "prior"
    # ...while the market comparison uses only the two market folds.
    assert reversed_row["market_baseline"] == "market"
    assert reversed_row["market_folds"] == 2
    assert reversed_row["market_lift"] == -1.0
    assert (reversed_row["market_won"], reversed_row["market_lost"]) == (0, 2)
    assert reversed_row["beats_market"] is False
