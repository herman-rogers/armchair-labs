"""Immutable prospective snapshots and grading for experimental forecasts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from patron.config.settings import CONFIG_DIR
from patron.metrics.backtest import (
    AnalysisWindow,
    MetricReportConfig,
    RankingConfig,
    analyze_rankings,
)

FROZEN_RANKERS = (
    "fitted_season_points",
    "fitted_two_stage",
    "fitted_adaptive_ppg_hybrid",
    "fitted_adaptive_season_hybrid",
)
SNAPSHOT_COLUMNS = (
    "forecast_season",
    "player_id",
    "player_display_name",
    "position",
    "games",
    "historical_ppg_prior",
    "ppg",
    "season_pts",
    "market_ecr_score",
    "market_snapshot",
    "preseason_rostered",
    "preseason_reserve",
    "cutoff_transaction_matched",
    *FROZEN_RANKERS,
)
OUTCOME_COLUMNS = (
    "forecast_season",
    "player_id",
    "actual_ppg",
    "actual_vor",
    "actual_games",
    "actual_season_points",
    "actual_availability_value",
    "outcome_complete",
)


class ProspectiveGradePending(RuntimeError):
    """The frozen forecast is intact, but its complete outcome does not exist yet."""


def frozen_snapshot(predictions: pl.DataFrame, season: int) -> pl.DataFrame:
    """Select the immutable inputs needed to grade one pending forecast later."""
    missing = sorted(set(SNAPSHOT_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"forecast snapshot is missing columns: {', '.join(missing)}")
    rows = predictions.filter(pl.col("forecast_season") == season)
    if not rows.height:
        raise ValueError(f"forecast snapshot has no rows for {season}")
    if "outcome_complete" in rows.columns and rows["outcome_complete"].any():
        raise ValueError(f"refusing to freeze {season} after outcomes were attached")
    snapshot = rows.select(SNAPSHOT_COLUMNS).sort(["player_id", "position"])
    if snapshot.select(pl.struct("forecast_season", "player_id").is_duplicated().any()).item():
        raise ValueError(f"forecast snapshot contains duplicate {season} player ids")
    return snapshot


def snapshot_bytes(snapshot: pl.DataFrame) -> bytes:
    """Canonical CSV representation used by the prospective audit digest."""
    return (
        snapshot.select(SNAPSHOT_COLUMNS)
        .sort(["player_id", "position"])
        .write_csv(float_precision=12)
        .encode()
    )


def snapshot_sha256(snapshot: pl.DataFrame) -> str:
    return hashlib.sha256(snapshot_bytes(snapshot)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_snapshot(
    snapshot_path: Path, manifest_path: Path
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load a tracked forecast only when its row count and byte digest are intact."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"no prospective manifest at {manifest_path}")
    if not snapshot_path.exists():
        raise FileNotFoundError(f"no prospective snapshot at {snapshot_path}")
    manifest: dict[str, Any] = yaml.safe_load(manifest_path.read_text())
    expected = str(manifest["pending_prediction_sha256"])
    actual = file_sha256(snapshot_path)
    if actual != expected:
        raise ValueError(f"prospective snapshot digest mismatch: {actual} != {expected}")
    snapshot = pl.read_csv(snapshot_path)
    if snapshot.height != int(manifest["pending_rows"]):
        raise ValueError(
            f"prospective snapshot has {snapshot.height} rows; "
            f"manifest requires {manifest['pending_rows']}"
        )
    seasons = snapshot["forecast_season"].unique().to_list()
    if seasons != [int(manifest["forecast_season"])]:
        raise ValueError(f"prospective snapshot seasons {seasons} do not match its manifest")
    return snapshot, manifest


def _grading_frame(
    snapshot: pl.DataFrame,
    outcomes: pl.DataFrame,
    season: int,
    *,
    grade_not_before: date,
    today: date,
) -> pl.DataFrame:
    if today < grade_not_before:
        raise ProspectiveGradePending(
            f"{season} is frozen but cannot be graded before {grade_not_before.isoformat()}"
        )
    missing = sorted(set(OUTCOME_COLUMNS) - set(outcomes.columns))
    if missing:
        raise ValueError(f"outcome input is missing columns: {', '.join(missing)}")
    actual = outcomes.filter(pl.col("forecast_season") == season).select(OUTCOME_COLUMNS)
    if not actual.height or not actual["outcome_complete"].all():
        raise ProspectiveGradePending(f"complete {season} outcomes are not available")
    raw_max_games = actual["actual_games"].max()
    max_games = float(raw_max_games) if isinstance(raw_max_games, (int, float)) else None
    if max_games is None or max_games < 17.0:
        raise ProspectiveGradePending(
            f"{season} outcomes appear partial (maximum games played is {max_games})"
        )
    if actual.select(pl.struct("forecast_season", "player_id").is_duplicated().any()).item():
        raise ValueError(f"outcome input contains duplicate {season} player ids")
    return snapshot.join(actual, on=["forecast_season", "player_id"], how="left").with_columns(
        pl.col("outcome_complete").fill_null(True),
        pl.col("actual_games").fill_null(0.0),
        pl.col("actual_season_points").fill_null(0.0),
        pl.col("actual_availability_value").fill_null(0.0),
    )


def _overall_result(rankings: list[dict[str, Any]], ranker: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rankings
            if row["position"] == "ALL"
            and row["target"] == "actual_availability_value"
            and row["ranker"] == ranker
        ),
        None,
    )


def grade_frozen_forecast(
    snapshot: pl.DataFrame,
    outcomes: pl.DataFrame,
    report_config: MetricReportConfig,
    season: int,
    *,
    expected_sha256: str,
    grade_not_before: date,
    today: date | None = None,
) -> dict[str, Any]:
    """Grade only frozen values, joining fresh data solely for the actual outcomes."""
    actual_sha256 = snapshot_sha256(snapshot)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"frozen snapshot digest mismatch: {actual_sha256} != {expected_sha256}")
    frame = _grading_frame(
        snapshot,
        outcomes,
        season,
        grade_not_before=grade_not_before,
        today=today or date.today(),
    )
    grading_path = CONFIG_DIR / f"prospective_grading_{season}.json"
    if season == 2026:
        if (
            file_sha256(grading_path)
            != "3ea599e42ebd90dfeb8268a2da6c2eaa57f7925ba55e68d6d0c7490dc2aa5a87"
        ):
            raise ValueError("Frozen grading rules changed")
        raw = json.loads(grading_path.read_text())
        ranking = RankingConfig(
            outcome_pool="scored_legacy",
            baselines=tuple(raw["baselines"]["default"]),
            baselines_by_target={
                k: tuple(v) for k, v in raw["baselines"].items() if k != "default"
            },
            candidates=FROZEN_RANKERS,
            targets=tuple(raw["targets"]),
            top_k=raw["top_k"],
            pool=raw["pool"],
            overall=raw["overall"],
            market_baseline=raw["market_baseline"],
        )
    else:
        ranking = replace(report_config.ranking, candidates=FROZEN_RANKERS)
    grading_config = replace(
        report_config,
        analysis_windows=(
            AnalysisWindow(
                key=f"prospective_{season}",
                label=f"Untouched {season} prospective forecast",
                start=season,
                end=season,
            ),
        ),
        ranking=ranking,
    )
    rankings = analyze_rankings(frame, grading_config)
    adaptive = _overall_result(rankings, "fitted_adaptive_ppg_hybrid")
    incumbent = _overall_result(rankings, "fitted_season_points")
    market = _overall_result(rankings, "market_ecr_score")
    if adaptive is None or incumbent is None or market is None:
        raise ValueError("prospective grade is missing an overall adaptive, incumbent, or ECR row")
    hit_vs_incumbent = float(adaptive["hit_rate"]) - float(incumbent["hit_rate"])
    hit_vs_ecr = float(adaptive["hit_rate"]) - float(market["hit_rate"])
    ndcg_vs_ecr = float(adaptive["ndcg"]) - float(market["ndcg"])
    gates = {
        "snapshot_digest_matches": True,
        "overall_hit_rate_beats_incumbent": hit_vs_incumbent > 0,
        "overall_hit_rate_beats_ecr": hit_vs_ecr > 0,
        "overall_ndcg_matches_or_beats_ecr": ndcg_vs_ecr >= 0,
    }
    return {
        "schema_version": 1,
        "forecast_season": season,
        "graded_at": datetime.now(UTC).isoformat(),
        "snapshot_sha256": actual_sha256,
        "frozen_rows": snapshot.height,
        "outcome_rows": frame.height,
        "overall": {
            "adaptive_hit_rate": adaptive["hit_rate"],
            "incumbent_hit_rate": incumbent["hit_rate"],
            "ecr_hit_rate": market["hit_rate"],
            "adaptive_ndcg": adaptive["ndcg"],
            "ecr_ndcg": market["ndcg"],
            "hit_rate_lift_vs_incumbent": round(hit_vs_incumbent, 4),
            "hit_rate_lift_vs_ecr": round(hit_vs_ecr, 4),
            "ndcg_lift_vs_ecr": round(ndcg_vs_ecr, 4),
        },
        "promotion_gates": gates,
        "passes_primary_gates": all(gates.values()),
        "review_required": True,
        "ranking_results": rankings,
    }
