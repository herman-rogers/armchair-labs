"""Rolling, leakage-aware backtests for the configurable v2 metric catalog.

The forecasting model stays untouched: this module repeatedly invokes it with a
historical season window, joins the resulting preseason board to the following
season's actual league-scored results, and describes both raw and incremental
predictive power.  The metric catalog lives in ``config/metric_report.yaml`` so a
metric can be added to or removed from the report without changing this engine.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import polars as pl

from patron.board.builder import build_board
from patron.board.overrides import OverrideSet
from patron.config.league import LeagueConfig
from patron.config.settings import load_metric_report_config
from patron.metrics.enrichment import depth_chart_as_of
from patron.metrics.fit import FitConfig, summarize_fitted_models
from patron.metrics.projection import METRIC_VERSION, ProjectionAssumptions, build_projection_board
from patron.metrics.regression import (
    EXPECTED_TDS,
    POSITION_TD_RATE,
    TD_OVER_EXPECTATION,
    add_td_over_expectation,
)
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
    available_from_forecast: int | None = None
    expected_sign: str = "positive"


@dataclass(frozen=True)
class AnalysisWindow:
    key: str
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class RankingConfig:
    """Head-to-head ranking evaluation: model outputs against model-free baselines.

    The report's central question is whether, in each backtested year, our ranking put
    the players who actually finished best at the top, and did so better than the
    rankings anyone could build from past production alone. Every ranker is scored on
    the same top-K slice per position, so the score describes the top of the board.
    """

    baselines: tuple[str, ...] = ("historical_ppg_prior", "ppg")
    # Target-matched baselines: a season-points outcome is compared against the
    # model-free season-points ranking too, not only PPG rankings.
    baselines_by_target: dict[str, tuple[str, ...]] = field(default_factory=dict)
    candidates: tuple[str, ...] = ("v2_score", "proj_ppg")
    targets: tuple[str, ...] = ("actual_season_points", "actual_ppg")
    top_k: dict[str, int] = field(default_factory=lambda: {"QB": 12, "RB": 24, "WR": 36, "TE": 12})
    pool: dict[str, int] = field(default_factory=lambda: {"QB": 24, "RB": 60, "WR": 80, "TE": 24})
    # Cross-position draft value: every ranker is turned into per-game VOR against its
    # own positional replacement (the live board's construction), then scored on one
    # top-K across all positions against availability-adjusted actual VOR.
    overall: dict[str, Any] = field(
        default_factory=lambda: {
            "k": 60,
            "pool": 120,
            "target": "actual_availability_value",
            "replacement_ranks": {"QB": 12, "RB": 25, "WR": 35, "TE": 12},
            "min_games": 8,
            "season_games": 17,
        }
    )

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> RankingConfig:
        if not raw:
            return cls()
        raw_baselines = raw.get("baselines")
        by_target: dict[str, tuple[str, ...]] = {}
        shared: tuple[str, ...] = cls().baselines
        if isinstance(raw_baselines, dict):
            shared = tuple(raw_baselines.get("default") or shared)
            by_target = {
                str(target): tuple(names)
                for target, names in raw_baselines.items()
                if target != "default"
            }
        elif raw_baselines:
            shared = tuple(raw_baselines)
        return cls(
            baselines=shared,
            baselines_by_target=by_target,
            candidates=tuple(raw.get("candidates") or cls().candidates),
            targets=tuple(raw.get("targets") or cls().targets),
            top_k={str(k): int(v) for k, v in (raw.get("top_k") or cls().top_k).items()},
            pool={str(k): int(v) for k, v in (raw.get("pool") or cls().pool).items()},
            overall={**cls().overall, **(raw.get("overall") or {})},
        )

    def baselines_for(self, target: str) -> tuple[str, ...]:
        return self.baselines_by_target.get(target, self.baselines)


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
    depth_chart_start_season: int = 2025
    analysis_windows: tuple[AnalysisWindow, ...] = ()
    ranking: RankingConfig = field(default_factory=RankingConfig)
    fit: FitConfig = field(default_factory=FitConfig)

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None = None) -> MetricReportConfig:
        raw = raw if raw is not None else load_metric_report_config()
        forecast_seasons = raw.get("forecast_seasons")
        if forecast_seasons is None:
            forecast_seasons = range(int(raw["forecast_start"]), int(raw["forecast_end"]) + 1)
        return cls(
            title=str(raw.get("title") or "V2 Metric Report"),
            forecast_seasons=tuple(int(value) for value in forecast_seasons),
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
            depth_chart_start_season=int(raw.get("depth_chart_start_season", 2025)),
            analysis_windows=tuple(
                AnalysisWindow(**entry) for entry in raw.get("analysis_windows") or ()
            ),
            ranking=RankingConfig.from_raw(raw.get("ranking")),
            fit=FitConfig.from_raw(raw.get("fit")),
        )

    @property
    def input_seasons(self) -> tuple[int, ...]:
        first = min(self.forecast_seasons) - self.history_seasons
        # The last forecast is prospective, so its source season is the final input.
        last = max(self.forecast_seasons) - 1
        return tuple(range(first, last + 1))

    @property
    def effective_windows(self) -> tuple[AnalysisWindow, ...]:
        if self.analysis_windows:
            return self.analysis_windows
        completed = self.forecast_seasons[:-1] or self.forecast_seasons
        return (
            AnalysisWindow(
                key="all",
                label="All completed forecasts",
                start=min(completed),
                end=max(completed),
            ),
        )


def _depth_chart_before(
    depth_charts: pl.DataFrame | None,
    forecast_season: int,
    cutoff: str,
) -> pl.DataFrame | None:
    return depth_chart_as_of(depth_charts, forecast_season, cutoff)


def build_backtest_predictions(
    player_seasons: pl.DataFrame,
    birth_dates: pl.DataFrame,
    league: LeagueConfig,
    report_config: MetricReportConfig,
    depth_charts: pl.DataFrame | None = None,
    market_rankings: pl.DataFrame | None = None,
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
        # Live/report parity: the live board pools positional TD rates over its own
        # three-season window, so each fold must pool over its window too rather than
        # inherit rates computed across every loaded season.
        history = add_td_over_expectation(
            history.drop(
                [
                    c
                    for c in (EXPECTED_TDS, TD_OVER_EXPECTATION, POSITION_TD_RATE)
                    if c in history.columns
                ]
            ),
            window=league.metrics.td_rate_window,
        )

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
                    pl.col("ppg_denominator_games").alias("actual_games"),
                    pl.col("season_pts").alias("actual_season_points"),
                )
                .with_columns(
                    (
                        pl.col("actual_vor").clip(lower_bound=0.0)
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
        if market_rankings is not None and market_rankings.height:
            market = market_rankings.filter(pl.col("forecast_season") == forecast_season).drop(
                "forecast_season"
            )
            fold = fold.join(market, on="player_id", how="left")
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
    denominator = math.sqrt(max(1.0 - metric_baseline**2, 0.0) * max(1.0 - target_baseline**2, 0.0))
    if denominator <= 1e-12:
        return None
    return (metric_target - metric_baseline * target_baseline) / denominator


def _fold_interval(values: list[float]) -> tuple[float | None, float | None]:
    """Empirical 2.5–97.5 percentile range across independent season folds."""
    if len(values) < 3:
        return None, None

    ordered = sorted(values)

    def percentile(proportion: float) -> float:
        index = (len(ordered) - 1) * proportion
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return percentile(0.025), percentile(0.975)


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _assessment(
    raw: float | None,
    partial: float | None,
    consistency: float | None,
    count: int,
    folds: int,
    minimum_sample: int,
    expected_sign: str,
) -> str:
    if count < minimum_sample or folds < 2 or raw is None:
        return "insufficient"
    signal = partial if partial is not None else raw
    signal_consistency = consistency if signal * raw >= 0 else 0.0
    direction = -1 if expected_sign == "negative" else 1
    if (
        expected_sign != "either"
        and signal * direction <= -0.10
        and (signal_consistency or 0.0) >= 0.50
    ):
        return "harmful"
    incremental = abs(signal)
    if abs(raw) >= 0.15 and partial is not None and abs(partial) < 0.05:
        return "redundant"
    if incremental >= 0.20 and (signal_consistency or 0.0) >= 0.67:
        return "strong"
    if incremental >= 0.10 and (signal_consistency or 0.0) >= 0.50:
        return "useful"
    if abs(raw) < 0.10 and (partial is None or abs(partial) < 0.10):
        return "weak"
    return "mixed"


def _analyze_predictions_window(
    predictions: pl.DataFrame,
    report_config: MetricReportConfig,
    window: AnalysisWindow,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze configured metrics inside one forecast-era window."""
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
                    and window.start <= int(row["forecast_season"]) <= window.end
                    and (
                        metric.available_from_forecast is None
                        or int(row["forecast_season"]) >= metric.available_from_forecast
                    )
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
                low, high = _fold_interval([float(fold["spearman"]) for fold in usable_folds])
                assessment = _assessment(
                    raw,
                    partial,
                    consistency,
                    len(pairs),
                    len(usable_folds),
                    report_config.minimum_sample,
                    metric.expected_sign,
                )
                metric_results.append(
                    {
                        "metric": metric.key,
                        "label": metric.label,
                        "group": metric.group,
                        "window": window.key,
                        "window_label": window.label,
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
                            "window": window.key,
                            "window_label": window.label,
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


def _ndcg_at_k(ranked_gains: list[float], k: int, ideal_gains: list[float]) -> float | None:
    """Normalized discounted cumulative gain over the top ``k`` of a ranking.

    Gains are the actual outcome values (clipped at zero) of the players the ranker
    put in positions 1..k; the ideal is the same outcome sorted descending.
    """
    if k <= 0 or not ideal_gains:
        return None
    discounts = [1.0 / math.log2(index + 2) for index in range(k)]
    dcg = sum(gain * discount for gain, discount in zip(ranked_gains[:k], discounts, strict=False))
    ideal = sum(gain * discount for gain, discount in zip(ideal_gains[:k], discounts, strict=False))
    if ideal <= 0:
        return None
    return dcg / ideal


def _rank_fold(
    rows: list[dict[str, Any]],
    ranker: str,
    target: str,
    k: int,
    pool: int,
) -> dict[str, Any] | None:
    """Score one ranker on one fold and position.

    ``hit_rate`` is the share of the ranker's top ``k`` who actually finished top ``k``
    on the target. ``ndcg`` weights those hits by how good the outcomes were and how
    high the ranker placed them. ``pool_spearman`` is rank correlation inside the top
    ``pool`` players by the ranker — the draftable board — so the score is not earned
    by separating starters from players who leave the league.
    """
    scored = [
        (metric, actual, row)
        for row in rows
        if (metric := _number(row.get(ranker))) is not None
        and (actual := _number(row.get(target))) is not None
    ]
    if len(scored) < k or k <= 0:
        return None
    by_ranker = sorted(scored, key=lambda item: item[0], reverse=True)
    by_actual = sorted(scored, key=lambda item: item[1], reverse=True)
    actual_top = {id(item[2]) for item in by_actual[:k]}
    hits = sum(1 for item in by_ranker[:k] if id(item[2]) in actual_top)
    ranked_gains = [max(item[1], 0.0) for item in by_ranker]
    ideal_gains = sorted(ranked_gains, reverse=True)
    draftable = by_ranker[: min(pool, len(by_ranker))]
    return {
        "n": len(scored),
        "hit_rate": hits / k,
        "ndcg": _ndcg_at_k(ranked_gains, k, ideal_gains),
        "pool_spearman": _spearman(
            [item[0] for item in draftable], [item[1] for item in draftable]
        ),
        "top_k_actual_mean": sum(item[1] for item in by_ranker[:k]) / k,
        "ideal_top_k_actual_mean": sum(item[1] for item in by_actual[:k]) / k,
    }


_ALREADY_VOR = frozenset({"adj_proj_vor", "proj_vor", "v2_rank_vor", "v2_overall_vor"})
_SEASON_SCALE = ("season_points", "season_pts")


def _overall_rows(
    rows: list[dict[str, Any]],
    ranker: str,
    overall: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach ``_overall`` = per-game VOR of ``ranker`` against its positional replacement.

    Mirrors ``board.rank``: season-scale keys are divided by the season length, the
    replacement is the baseline-rank value within the position among players with
    enough games, and keys that are already a VOR pass through.
    """
    season_games = float(overall.get("season_games", 17))
    min_games = float(overall.get("min_games", 8))
    ranks = overall.get("replacement_ranks") or {}
    scale = season_games if any(ranker.endswith(tag) for tag in _SEASON_SCALE) else 1.0
    out: list[dict[str, Any]] = []
    by_position: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_position.setdefault(str(row.get("position")), []).append(row)
    for position, group in by_position.items():
        rank = int(ranks.get(position, 0))
        values = [
            (v / scale)
            for row in group
            if (v := _number(row.get(ranker))) is not None
            and (_number(row.get("games")) or 0.0) >= min_games
        ]
        replacement = 0.0
        if rank and values:
            values.sort(reverse=True)
            replacement = values[min(rank, len(values)) - 1]
        for row in group:
            value = _number(row.get(ranker))
            if value is None:
                continue
            equivalent = value if ranker in _ALREADY_VOR else value / scale - replacement
            out.append({**row, "_overall": equivalent})
    return out


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _finalize_entries(per_ranker: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach baseline comparison, strict W–T–L, and paired uncertainty; round for output."""
    finalized: list[dict[str, Any]] = []
    baselines = [entry for entry in per_ranker.values() if entry["role"] == "baseline"]
    for entry in per_ranker.values():
        if entry["role"] == "candidate" and baselines:
            # Compare on the candidate's own folds: a walk-forward ranker
            # that needs training seasons must not be measured against a
            # baseline pooled over folds it could not score.
            candidate_seasons = {fold["forecast_season"] for fold in entry["fold_results"]}

            def _mean_on(item: dict[str, Any], key: str, seasons=candidate_seasons):
                matched = [
                    fold[key]
                    for fold in item["fold_results"]
                    if fold["forecast_season"] in seasons and fold[key] is not None
                ]
                return sum(matched) / len(matched) if matched else None

            # A sparse baseline (for example archived ECR available only since 2021)
            # cannot win by averaging a favorable subset of the candidate's folds.
            # It competes only in windows where it covers every candidate fold.
            comparable = [
                item
                for item in baselines
                if candidate_seasons <= {fold["forecast_season"] for fold in item["fold_results"]}
            ]
            comparable = comparable or baselines
            best = max(comparable, key=lambda item: _mean_on(item, "hit_rate") or 0.0)
            best_ndcg = max(comparable, key=lambda item: _mean_on(item, "ndcg") or 0.0)
            best_hit_rate = _mean_on(best, "hit_rate") or 0.0
            best_ndcg_value = _mean_on(best_ndcg, "ndcg")
            # Strict paired comparison against the single fixed best
            # baseline: a fold is won only when the candidate's hit rate
            # is higher; equal hit rates are broken on the actual points
            # the top-K produced, and remain a tie if those match too.
            base_folds = {f["forecast_season"]: f for f in best["fold_results"]}
            won = tied = lost = 0
            diffs: list[float] = []
            for fold in entry["fold_results"]:
                other = base_folds.get(fold["forecast_season"])
                if other is None:
                    continue
                diff = fold["hit_rate"] - other["hit_rate"]
                diffs.append(diff)
                if abs(diff) > 1e-9:
                    won += diff > 0
                    lost += diff < 0
                else:
                    gap = fold["top_k_actual_mean"] - other["top_k_actual_mean"]
                    if abs(gap) > 1e-9:
                        won += gap > 0
                        lost += gap < 0
                    else:
                        tied += 1
            wins = won
            lift_se = None
            if len(diffs) > 1:
                mean_diff = sum(diffs) / len(diffs)
                variance = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
                lift_se = math.sqrt(variance / len(diffs))
            entry["best_baseline"] = best["ranker"]
            entry["best_ndcg_baseline"] = best_ndcg["ranker"]
            entry["baseline_hit_rate"] = _rounded(best_hit_rate)
            entry["hit_rate_lift"] = (entry["hit_rate"] or 0.0) - best_hit_rate
            entry["ndcg_lift"] = (
                (entry["ndcg"] - best_ndcg_value)
                if entry["ndcg"] is not None and best_ndcg_value is not None
                else None
            )
            entry["folds_beating_baseline"] = wins
            entry["folds_won"] = won
            entry["folds_tied"] = tied
            entry["folds_lost"] = lost
            entry["hit_rate_lift_se"] = lift_se
            entry["hit_rate_lift_ci_low"] = (
                entry["hit_rate_lift"] - 1.96 * lift_se if lift_se is not None else None
            )
            entry["hit_rate_lift_ci_high"] = (
                entry["hit_rate_lift"] + 1.96 * lift_se if lift_se is not None else None
            )
            # Pass = positive pooled lift and more folds won than lost.
            entry["beats_baseline"] = entry["hit_rate_lift"] > 0 and won > lost
        else:
            entry["best_baseline"] = None
            entry["best_ndcg_baseline"] = None
            entry["baseline_hit_rate"] = None
            entry["hit_rate_lift"] = None
            entry["ndcg_lift"] = None
            entry["folds_beating_baseline"] = None
            entry["folds_won"] = None
            entry["folds_tied"] = None
            entry["folds_lost"] = None
            entry["hit_rate_lift_se"] = None
            entry["hit_rate_lift_ci_low"] = None
            entry["hit_rate_lift_ci_high"] = None
            entry["beats_baseline"] = None
        for key in (
            "hit_rate",
            "ndcg",
            "pool_spearman",
            "top_k_actual_mean",
            "ideal_top_k_actual_mean",
            "hit_rate_lift",
            "ndcg_lift",
            "hit_rate_lift_se",
            "hit_rate_lift_ci_low",
            "hit_rate_lift_ci_high",
        ):
            entry[key] = _rounded(entry[key])
        for fold in entry["fold_results"]:
            for key in (
                "hit_rate",
                "ndcg",
                "pool_spearman",
                "top_k_actual_mean",
                "ideal_top_k_actual_mean",
            ):
                fold[key] = _rounded(fold[key])
        finalized.append(entry)
    return finalized


def analyze_rankings(
    predictions: pl.DataFrame,
    report_config: MetricReportConfig,
) -> list[dict[str, Any]]:
    """Head-to-head top-of-board ranking quality: every ranker on identical slices.

    One record per window, target, position, and ranker. Candidate records carry the
    lift over the *best* baseline on each measure and how many folds they won, which is
    the direct answer to "should this output be the board's sort key".
    """
    ranking = report_config.ranking
    records = [row for row in predictions.to_dicts() if row.get("outcome_complete")]
    results: list[dict[str, Any]] = []

    for window in report_config.effective_windows:
        window_rows = [
            row for row in records if window.start <= int(row["forecast_season"]) <= window.end
        ]
        seasons = sorted({int(row["forecast_season"]) for row in window_rows})
        for target in ranking.targets:
            if target not in predictions.columns:
                continue
            rankers = [(name, "baseline") for name in ranking.baselines_for(target)] + [
                (name, "candidate") for name in ranking.candidates
            ]
            rankers = [(name, role) for name, role in rankers if name in predictions.columns]
            if not rankers:
                continue
            for position, k in ranking.top_k.items():
                pool = ranking.pool.get(position, k * 2)
                per_ranker: dict[str, dict[str, Any]] = {}
                for name, role in rankers:
                    folds = []
                    for season in seasons:
                        rows = [
                            row
                            for row in window_rows
                            if int(row["forecast_season"]) == season
                            and row.get("position") == position
                        ]
                        scored = _rank_fold(rows, name, target, k, pool)
                        if scored is not None:
                            folds.append({"forecast_season": season, **scored})
                    if not folds:
                        continue
                    per_ranker[name] = {
                        "ranker": name,
                        "role": role,
                        "window": window.key,
                        "window_label": window.label,
                        "target": target,
                        "position": position,
                        "k": k,
                        "pool": pool,
                        "folds": len(folds),
                        "hit_rate": _mean([fold["hit_rate"] for fold in folds]),
                        "ndcg": _mean([fold["ndcg"] for fold in folds]),
                        "pool_spearman": _mean([fold["pool_spearman"] for fold in folds]),
                        "top_k_actual_mean": _mean([fold["top_k_actual_mean"] for fold in folds]),
                        "ideal_top_k_actual_mean": _mean(
                            [fold["ideal_top_k_actual_mean"] for fold in folds]
                        ),
                        "fold_results": folds,
                    }

                results.extend(_finalize_entries(per_ranker))

        overall = ranking.overall
        target = str(overall.get("target") or "actual_availability_value")
        if target in predictions.columns:
            k = int(overall.get("k", 60))
            pool = int(overall.get("pool", 120))
            rankers = [(name, "baseline") for name in ranking.baselines_for(target)] + [
                (name, "candidate") for name in ranking.candidates
            ]
            rankers = [(name, role) for name, role in rankers if name in predictions.columns]
            per_ranker = {}
            for name, role in rankers:
                folds = []
                for season in seasons:
                    rows = [row for row in window_rows if int(row["forecast_season"]) == season]
                    scored = _rank_fold(
                        _overall_rows(rows, name, overall), "_overall", target, k, pool
                    )
                    if scored is not None:
                        folds.append({"forecast_season": season, **scored})
                if folds:
                    per_ranker[name] = {
                        "ranker": name,
                        "role": role,
                        "window": window.key,
                        "window_label": window.label,
                        "target": target,
                        "position": "ALL",
                        "k": k,
                        "pool": pool,
                        "folds": len(folds),
                        "hit_rate": _mean([f["hit_rate"] for f in folds]),
                        "ndcg": _mean([f["ndcg"] for f in folds]),
                        "pool_spearman": _mean([f["pool_spearman"] for f in folds]),
                        "top_k_actual_mean": _mean([f["top_k_actual_mean"] for f in folds]),
                        "ideal_top_k_actual_mean": _mean(
                            [f["ideal_top_k_actual_mean"] for f in folds]
                        ),
                        "fold_results": folds,
                    }
            results.extend(_finalize_entries(per_ranker))
    return results


def analyze_predictions(
    predictions: pl.DataFrame,
    report_config: MetricReportConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze every configured metric independently in each historical window."""
    metric_results: list[dict[str, Any]] = []
    model_results: list[dict[str, Any]] = []
    for window in report_config.effective_windows:
        window_metrics, window_models = _analyze_predictions_window(
            predictions, report_config, window
        )
        metric_results.extend(window_metrics)
        model_results.extend(window_models)
    return metric_results, model_results


def build_metric_report(
    predictions: pl.DataFrame,
    report_config: MetricReportConfig,
    fitted_models: list[dict[str, Any]] | None = None,
    fitted_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_results, model_results = analyze_predictions(predictions, report_config)
    ranking_results = analyze_rankings(predictions, report_config)
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
        metric_rows = completed_rows
        if metric.available_from_forecast is not None:
            metric_rows = metric_rows.filter(
                pl.col("forecast_season") >= metric.available_from_forecast
            )
        if available and metric_rows.height:
            coverage = _number(metric_rows[metric.key].is_not_null().mean())
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
        "schema_version": 2,
        "title": report_config.title,
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "forecast_seasons": list(report_config.forecast_seasons),
            "input_seasons": list(report_config.input_seasons),
            "history_seasons": report_config.history_seasons,
            "depth_chart_cutoff": report_config.depth_chart_cutoff,
            "depth_chart_start_season": report_config.depth_chart_start_season,
            "minimum_sample": report_config.minimum_sample,
            "baseline_metric": report_config.baseline_metric,
            "analysis_windows": [asdict(window) for window in report_config.effective_windows],
            "ranking": asdict(report_config.ranking),
            "fit": {
                "models": [asdict(spec) for spec in report_config.fit.models],
                "outputs": list(report_config.fit.output_columns),
                "min_train_folds": report_config.fit.min_train_folds,
                "inner_validation_folds": report_config.fit.inner_validation_folds,
            },
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
                "Depth charts from 2004–2024 use the published Week 1 ranking as an "
                "August 31 proxy because those files have no publication timestamp.",
                "Injury, route-participation, and depth-chart metrics enter only after a "
                "complete history exists for their source.",
                "Partial rank correlation controls for the historical PPG prior, but does "
                "not prove causation.",
                "Correlation ranges are empirical 2.5th–97.5th percentile ranges across "
                "season folds, avoiding the false independence of repeated player rows.",
                "Long-horizon, modern, and recent windows are reported separately because "
                "NFL roles and offensive environments change across eras.",
            ],
        },
        "targets": [asdict(target) for target in report_config.targets],
        "metrics": catalog,
        "results": metric_results,
        "model_results": model_results,
        "ranking_results": ranking_results,
        "fitted_models": fitted_models or [],
        "fitted_model_summary": summarize_fitted_models(fitted_models or []),
        "fitted_artifact": fitted_artifact or {},
    }


def _fitted_weight_lines(report: dict[str, Any]) -> list[str]:
    """Markdown tables of the latest learned weights per fitted model."""
    summary = report.get("fitted_model_summary") or []
    if not summary:
        return []
    lines = [
        "## Learned weights (walk-forward ridge, latest refit)",
        "",
        "Standardized coefficients; the value in parentheses is the coefficient's standard "
        "deviation across every yearly refit — a weight that swings is not evidence. λ is "
        "the ridge strength chosen by nested walk-forward validation for the latest refit.",
        "",
    ]
    for name in sorted({entry["model"] for entry in summary}):
        entries = [entry for entry in summary if entry["model"] == name]
        features = list(entries[0]["coefficients"])
        lines.extend(
            [
                f"### `{name}`",
                "",
                "Pos | Scope | Train rows | λ | " + " | ".join(features),
                "---|---|---:|---:|" + "|".join("---:" for _ in features),
            ]
        )
        for entry in entries:
            cells = []
            for feature in features:
                value = entry["coefficients"].get(feature)
                spread = entry["coefficient_sd_across_refits"].get(feature)
                cells.append(
                    ""
                    if value is None
                    else f"{value:+.2f}" + (f" ({spread:.2f})" if spread is not None else "")
                )
            lam = entry.get("ridge_lambda")
            lines.append(
                f"{entry['position']} | {entry['scope']} | {entry['n_train']} | "
                f"{'' if lam is None else lam} | " + " | ".join(cells)
            )
        lines.append("")
    return lines


def _ranking_verdict_lines(report: dict[str, Any]) -> list[str]:
    """Markdown table: does each model output beat the best model-free baseline?"""
    rows = report.get("ranking_results") or []
    if not rows:
        return []
    ranking = report["configuration"].get("ranking") or {}
    windows = report["configuration"].get("analysis_windows") or []
    lines = [
        "## Does v2 beat the naive baseline?",
        "",
        "Each ranker is scored on the same top-K per position (K = "
        + ", ".join(f"{pos} {k}" for pos, k in (ranking.get("top_k") or {}).items())
        + "). Hit rate is the share of the ranker's top K who actually finished top K; "
        "lift is against the single best baseline on the candidate's own folds, with the "
        "paired standard error across folds; W–T–L counts folds strictly won, tied "
        "(equal hit rate and equal top-K points), or lost. A candidate passes only with "
        "positive lift and more wins than losses.",
        "",
    ]
    for window in windows:
        window_rows = [row for row in rows if row["window"] == window["key"]]
        if not window_rows:
            continue
        for target in ranking.get("targets") or []:
            target_rows = [row for row in window_rows if row["target"] == target]
            if not target_rows:
                continue
            lines.extend(
                [
                    f"### {window['label']} — {target}",
                    "",
                    "Pos | Ranker | Role | Folds | Hit rate | NDCG | Pool Spearman | "
                    "Lift vs baseline (±SE) | W–T–L | Verdict",
                    "---|---|---|---:|---:|---:|---:|---:|---:|---",
                ]
            )
            for row in sorted(
                target_rows,
                key=lambda row: (
                    row["position"],
                    row["role"] != "baseline",
                    -(row["hit_rate"] or 0),
                ),
            ):
                lift = "" if row["hit_rate_lift"] is None else f"{row['hit_rate_lift']:+.3f}"
                if lift and row.get("hit_rate_lift_se") is not None:
                    lift += f" (±{row['hit_rate_lift_se']:.3f})"
                won = (
                    ""
                    if row.get("folds_won") is None
                    else f"{row['folds_won']}–{row['folds_tied']}–{row['folds_lost']}"
                )
                verdict = (
                    ""
                    if row["beats_baseline"] is None
                    else ("beats" if row["beats_baseline"] else "loses")
                )
                lines.append(
                    f"{row['position']} | {row['ranker']} | {row['role']} | {row['folds']} | "
                    f"{row['hit_rate']:.3f} | {(row['ndcg'] or 0):.3f} | "
                    f"{(row['pool_spearman'] or 0):.3f} | {lift} | {won} | {verdict}"
                )
            lines.append("")
    return lines


def render_metric_report_markdown(report: dict[str, Any]) -> str:
    """Compact human-readable companion to the JSON/frontend report."""
    summary = report["data_summary"]
    completed = summary["completed_forecasts"]
    completed_label = f"{completed[0]}–{completed[-1]}" if len(completed) > 1 else str(completed[0])
    windows = report["configuration"]["analysis_windows"]
    selected_window = next((window for window in windows if window["key"] == "modern"), windows[0])
    lines = [
        f"# {report['title']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Completed forecast seasons: {completed_label}",
        f"Pending forecast seasons: {', '.join(map(str, summary['pending_forecasts'])) or 'none'}",
        f"Forecast rows: {summary['forecast_rows']}",
        "",
        "Evidence windows: "
        + ", ".join(f"{window['label']} ({window['start']}–{window['end']})" for window in windows),
        "",
    ]
    lines.extend(_ranking_verdict_lines(report))
    lines.extend(_fitted_weight_lines(report))
    lines += [
        f"## Strongest incremental signals — {selected_window['label']}",
        "",
        "Metric | Target | Pos | N | Spearman | Partial vs prior | Assessment",
        "---|---|---|---:|---:|---:|---",
    ]
    ranked = sorted(
        (
            row
            for row in report["results"]
            if row["window"] == selected_window["key"]
            and row["position"] == "ALL"
            and row["partial_spearman"] is not None
            and row["assessment"] != "insufficient"
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
