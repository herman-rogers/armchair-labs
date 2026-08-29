"""Rolling, leakage-aware backtests for the configurable v2 metric catalog.

The forecasting model stays untouched: this module repeatedly invokes it with a
historical season window, joins the resulting preseason board to the following
season's actual league-scored results, and describes both raw and incremental
predictive power.  The metric catalog lives in ``config/metric_report.yaml`` so a
metric can be added to or removed from the report without changing this engine.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any

import polars as pl

from patron.board.builder import build_board
from patron.board.overrides import OverrideSet
from patron.config.league import LeagueConfig
from patron.config.settings import load_metric_report_config
from patron.metrics.enrichment import current_depth_chart
from patron.metrics.projection import METRIC_VERSION, ProjectionAssumptions, build_projection_board
from patron.metrics.vor import add_vor, replacement_levels


@dataclass(frozen=True)
class TargetDefinition:
    key: str
    label: str
    description: str
    missing_as_zero: bool = False


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    group: str
    description: str
    targets: tuple[str, ...]
    positions: tuple[str, ...] = ()
    prediction_target: str | None = None


@dataclass(frozen=True)
class MetricReportConfig:
    title: str
    forecast_seasons: tuple[int, ...]
    history_seasons: int
    depth_chart_cutoff: str
    minimum_sample: int
    baseline_metric: str
    targets: tuple[TargetDefinition, ...]
    metrics: tuple[MetricDefinition, ...]

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None = None) -> MetricReportConfig:
        raw = raw if raw is not None else load_metric_report_config()
        return cls(
            title=str(raw.get("title") or "V2 Metric Report"),
            forecast_seasons=tuple(int(value) for value in raw["forecast_seasons"]),
            history_seasons=int(raw.get("history_seasons", 3)),
            depth_chart_cutoff=str(raw.get("depth_chart_cutoff", "08-31")),
            minimum_sample=int(raw.get("minimum_sample", 20)),
            baseline_metric=str(raw.get("baseline_metric", "historical_ppg_prior")),
            targets=tuple(TargetDefinition(**entry) for entry in raw["targets"]),
            metrics=tuple(
                MetricDefinition(
                    **{
                        **entry,
                        "targets": tuple(entry.get("targets") or ()),
                        "positions": tuple(entry.get("positions") or ()),
                    }
                )
                for entry in raw["metrics"]
            ),
        )

    @property
    def input_seasons(self) -> tuple[int, ...]:
        first = min(self.forecast_seasons) - self.history_seasons
        # The last forecast is prospective, so its source season is the final input.
        last = max(self.forecast_seasons) - 1
        return tuple(range(first, last + 1))


def _depth_chart_before(
    depth_charts: pl.DataFrame | None,
    forecast_season: int,
    cutoff: str,
) -> pl.DataFrame | None:
    if depth_charts is None or depth_charts.height == 0:
        return None
    month, day = (int(part) for part in cutoff.split("-", maxsplit=1))
    limit = date(forecast_season, month, day)
    dated = depth_charts.with_columns(pl.col("dt").cast(pl.Date, strict=False).alias("_date"))
    eligible = dated.filter(pl.col("_date").is_not_null() & (pl.col("_date") <= limit)).drop(
        "_date"
    )
    return current_depth_chart(eligible) if eligible.height else None


def build_backtest_predictions(
    player_seasons: pl.DataFrame,
    birth_dates: pl.DataFrame,
    league: LeagueConfig,
    report_config: MetricReportConfig,
    depth_charts: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Create one v2 forecast row per returning player and forecast season."""
    folds: list[pl.DataFrame] = []
    birth_dates = birth_dates.unique(subset=["player_id"], keep="last")

    for forecast_season in report_config.forecast_seasons:
        source_season = forecast_season - 1
        history_start = source_season - report_config.history_seasons + 1
        history = player_seasons.filter(
            pl.col("season").is_between(history_start, source_season, closed="both")
        )
        if history.filter(pl.col("season") == source_season).height == 0:
            continue

        fold_config = league.model_copy(
            update={
                "seasons": list(range(history_start, source_season + 1)),
                "board_season": source_season,
                "draft_season": forecast_season,
            }
        )
        v1 = build_board(
            history,
            birth_dates,
            config=fold_config,
            override_set=OverrideSet(),
            strict_overrides=False,
        ).with_columns(pl.lit("v1").alias(METRIC_VERSION))
        projection = build_projection_board(
            history,
            v1,
            fold_config,
            assumptions=ProjectionAssumptions(),
            current_players=_depth_chart_before(
                depth_charts,
                forecast_season,
                report_config.depth_chart_cutoff,
            ),
        )

        actual = player_seasons.filter(pl.col("season") == forecast_season)
        complete = actual.height > 0
        if complete:
            levels = replacement_levels(
                actual,
                baseline_ranks=league.vor_baseline_rank,
                min_games=league.min_games_baseline,
            )
            actual = (
                add_vor(actual, levels)
                .select(
                    "player_id",
                    pl.col("ppg").alias("actual_ppg"),
                    pl.col("vor").alias("actual_vor"),
                    pl.col("games").alias("actual_games"),
                    pl.col("season_pts").alias("actual_season_points"),
                )
                .with_columns(
                    (
                        pl.col("actual_vor")
                        * (pl.col("actual_games") / league.metrics.projection_season_games).clip(
                            0.0, 1.0
                        )
                    ).alias("actual_availability_value")
                )
            )
        else:
            actual = pl.DataFrame(
                schema={
                    "player_id": pl.String,
                    "actual_ppg": pl.Float64,
                    "actual_vor": pl.Float64,
                    "actual_games": pl.Float64,
                    "actual_season_points": pl.Float64,
                    "actual_availability_value": pl.Float64,
                }
            )

        fold = (
            projection.join(actual, on="player_id", how="left")
            .with_columns(
                pl.lit(forecast_season).cast(pl.Int32).alias("forecast_season"),
                pl.lit(source_season).cast(pl.Int32).alias("source_season"),
                pl.lit(complete).alias("outcome_complete"),
                pl.col("actual_ppg").is_not_null().alias("actual_matched"),
            )
            .with_columns(
                pl.col("actual_games").fill_null(0.0),
                pl.col("actual_season_points").fill_null(0.0),
                pl.col("actual_availability_value").fill_null(0.0),
            )
        )
        folds.append(fold)

    if not folds:
        raise ValueError("no forecast folds could be built from the supplied player seasons")
    return pl.concat(folds, how="diagonal_relaxed")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _ranks(values: list[float]) -> list[float]:
    """Average ranks with stable tie handling."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss <= 0 or right_ss <= 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _partial_spearman(
    metric: list[float], target: list[float], baseline: list[float]
) -> float | None:
    metric_rank = _ranks(metric)
    target_rank = _ranks(target)
    baseline_rank = _ranks(baseline)
    metric_target = _pearson(metric_rank, target_rank)
    metric_baseline = _pearson(metric_rank, baseline_rank)
    target_baseline = _pearson(target_rank, baseline_rank)
    if metric_target is None or metric_baseline is None or target_baseline is None:
        return None
    denominator = math.sqrt(
        max(1.0 - metric_baseline**2, 0.0) * max(1.0 - target_baseline**2, 0.0)
    )
    if denominator <= 1e-12:
        return None
    return (metric_target - metric_baseline * target_baseline) / denominator


def _confidence_interval(
    correlation: float | None, count: int
) -> tuple[float | None, float | None]:
    """Approximate Fisher interval around a rank correlation."""
    if correlation is None or count <= 3:
        return None, None
    bounded = max(-0.999999, min(0.999999, correlation))
    center = math.atanh(bounded)
    radius = 1.96 / math.sqrt(count - 3)
    return math.tanh(center - radius), math.tanh(center + radius)


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _assessment(
    raw: float | None,
    partial: float | None,
    consistency: float | None,
    count: int,
    folds: int,
    minimum_sample: int,
) -> str:
    if count < minimum_sample or folds < 2 or raw is None:
        return "insufficient"
    incremental = abs(partial) if partial is not None else abs(raw)
    if abs(raw) >= 0.15 and partial is not None and abs(partial) < 0.05:
        return "redundant"
    if incremental >= 0.20 and (consistency or 0.0) >= 0.67:
        return "strong"
    if incremental >= 0.10 and (consistency or 0.0) >= 0.50:
        return "useful"
    if abs(raw) < 0.10 and (partial is None or abs(partial) < 0.10):
        return "weak"
    return "mixed"


def analyze_predictions(
    predictions: pl.DataFrame,
    report_config: MetricReportConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze configured metrics and direct projection errors."""
    records = predictions.to_dicts()
    targets = {target.key: target for target in report_config.targets}
    metric_results: list[dict[str, Any]] = []
    model_results: list[dict[str, Any]] = []

    for metric in report_config.metrics:
        if metric.key not in predictions.columns:
            continue
        positions = ("ALL", *(metric.positions or ("QB", "RB", "WR", "TE")))
        for target_key in metric.targets:
            target = targets[target_key]
            if target.key not in predictions.columns:
                continue
            for position in positions:
                eligible = [
                    row
                    for row in records
                    if row.get("outcome_complete")
                    and (position == "ALL" or row.get("position") == position)
                    and (not metric.positions or row.get("position") in metric.positions)
                ]
                pairs: list[tuple[float, float, int]] = []
                partial_rows: list[tuple[float, float, float]] = []
                for row in eligible:
                    metric_value = _number(row.get(metric.key))
                    target_value = _number(row.get(target.key))
                    if metric_value is None or target_value is None:
                        continue
                    pairs.append((metric_value, target_value, int(row["forecast_season"])))
                    baseline = _number(row.get(report_config.baseline_metric))
                    if baseline is not None:
                        partial_rows.append((metric_value, target_value, baseline))
                left = [item[0] for item in pairs]
                right = [item[1] for item in pairs]
                raw = _spearman(left, right)
                pearson = _pearson(left, right)
                partial = (
                    None
                    if metric.key == report_config.baseline_metric
                    else _partial_spearman(
                        [item[0] for item in partial_rows],
                        [item[1] for item in partial_rows],
                        [item[2] for item in partial_rows],
                    )
                )
                fold_results: list[dict[str, Any]] = []
                for season in sorted({item[2] for item in pairs}):
                    fold_pairs = [item for item in pairs if item[2] == season]
                    fold_rho = _spearman(
                        [item[0] for item in fold_pairs], [item[1] for item in fold_pairs]
                    )
                    fold_results.append(
                        {
                            "forecast_season": season,
                            "n": len(fold_pairs),
                            "spearman": _rounded(fold_rho),
                        }
                    )
                usable_folds = [
                    fold for fold in fold_results if fold["spearman"] is not None and fold["n"] >= 3
                ]
                consistency = None
                if raw is not None and usable_folds:
                    direction = 1 if raw >= 0 else -1
                    consistency = sum(
                        1 for fold in usable_folds if float(fold["spearman"]) * direction > 0
                    ) / len(usable_folds)
                low, high = _confidence_interval(raw, len(pairs))
                assessment = _assessment(
                    raw,
                    partial,
                    consistency,
                    len(pairs),
                    len(usable_folds),
                    report_config.minimum_sample,
                )
                metric_results.append(
                    {
                        "metric": metric.key,
                        "label": metric.label,
                        "group": metric.group,
                        "target": target.key,
                        "target_label": target.label,
                        "position": position,
                        "n": len(pairs),
                        "eligible": len(eligible),
                        "coverage": _rounded(len(pairs) / len(eligible) if eligible else None),
                        "folds": len(usable_folds),
                        "spearman": _rounded(raw),
                        "spearman_ci_low": _rounded(low),
                        "spearman_ci_high": _rounded(high),
                        "pearson": _rounded(pearson),
                        "partial_spearman": _rounded(partial),
                        "direction_consistency": _rounded(consistency),
                        "assessment": assessment,
                        "fold_results": fold_results,
                    }
                )

                if metric.prediction_target == target.key and pairs:
                    errors = [predicted - actual for predicted, actual, _ in pairs]
                    model_results.append(
                        {
                            "metric": metric.key,
                            "label": metric.label,
                            "target": target.key,
                            "target_label": target.label,
                            "position": position,
                            "n": len(errors),
                            "mae": _rounded(sum(abs(value) for value in errors) / len(errors)),
                            "rmse": _rounded(
                                math.sqrt(sum(value**2 for value in errors) / len(errors))
                            ),
                            "bias": _rounded(sum(errors) / len(errors)),
                            "spearman": _rounded(raw),
                        }
                    )

    return metric_results, model_results


def build_metric_report(
    predictions: pl.DataFrame,
    report_config: MetricReportConfig,
) -> dict[str, Any]:
    metric_results, model_results = analyze_predictions(predictions, report_config)
    completed = sorted(
        predictions.filter(pl.col("outcome_complete"))["forecast_season"].unique().to_list()
    )
    pending = sorted(
        predictions.filter(~pl.col("outcome_complete"))["forecast_season"].unique().to_list()
    )
    catalog: list[dict[str, Any]] = []
    completed_rows = predictions.filter(pl.col("outcome_complete"))
    for metric in report_config.metrics:
        available = metric.key in predictions.columns
        coverage = None
        if available and completed_rows.height:
            coverage = _number(completed_rows[metric.key].is_not_null().mean())
        catalog.append(
            {
                **asdict(metric),
                "targets": list(metric.targets),
                "positions": list(metric.positions),
                "available": available,
                "coverage": _rounded(coverage),
            }
        )

    assessments: dict[str, int] = {}
    for result in metric_results:
        assessments[result["assessment"]] = assessments.get(result["assessment"], 0) + 1

    return {
        "schema_version": 1,
        "title": report_config.title,
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "forecast_seasons": list(report_config.forecast_seasons),
            "input_seasons": list(report_config.input_seasons),
            "history_seasons": report_config.history_seasons,
            "depth_chart_cutoff": report_config.depth_chart_cutoff,
            "minimum_sample": report_config.minimum_sample,
            "baseline_metric": report_config.baseline_metric,
        },
        "data_summary": {
            "completed_forecasts": completed,
            "pending_forecasts": pending,
            "forecast_rows": predictions.height,
            "completed_rows": completed_rows.height,
            "returning_player_scope": True,
            "assessment_counts": assessments,
            "limitations": [
                "Only players with prior-season NFL production are forecast; rookies are "
                "outside this model.",
                "Depth charts are cut off before each season and current manual overrides "
                "are disabled.",
                "Partial rank correlation controls for the historical PPG prior, but does "
                "not prove causation.",
                "Three completed seasons are directional evidence; more historical folds "
                "will tighten conclusions.",
            ],
        },
        "targets": [asdict(target) for target in report_config.targets],
        "metrics": catalog,
        "results": metric_results,
        "model_results": model_results,
    }


def render_metric_report_markdown(report: dict[str, Any]) -> str:
    """Compact human-readable companion to the JSON/frontend report."""
    summary = report["data_summary"]
    lines = [
        f"# {report['title']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Completed forecast seasons: "
        f"{', '.join(map(str, summary['completed_forecasts'])) or 'none'}",
        f"Pending forecast seasons: {', '.join(map(str, summary['pending_forecasts'])) or 'none'}",
        f"Forecast rows: {summary['forecast_rows']}",
        "",
        "## Strongest incremental signals",
        "",
        "Metric | Target | Pos | N | Spearman | Partial vs prior | Assessment",
        "---|---|---|---:|---:|---:|---",
    ]
    ranked = sorted(
        (
            row
            for row in report["results"]
            if row["position"] == "ALL" and row["partial_spearman"] is not None
        ),
        key=lambda row: abs(row["partial_spearman"]),
        reverse=True,
    )
    for row in ranked[:30]:
        lines.append(
            f"{row['label']} | {row['target_label']} | {row['position']} | {row['n']} | "
            f"{row['spearman']:.3f} | {row['partial_spearman']:.3f} | {row['assessment']}"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("")
    return "\n".join(lines)
