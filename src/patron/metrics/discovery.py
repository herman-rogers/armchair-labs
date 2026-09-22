"""Leakage-safe automated feature discovery for report-only forecast challengers.

The production projection intentionally uses a small, hand-auditable feature set.
This module asks whether that choice leaves predictive structure on the table.  It
builds several deliberately different challengers and scores every forecast season
with models trained strictly on earlier completed seasons:

* broad ridge and shallow boosted-tree models select from cutoff-safe numeric inputs;
* weekly source-season sequences are converted to generic temporal descriptors;
* PCA and K-means produce training-only latent player archetypes;
* deterministic random convolutions expose local weekly shapes;
* symbolic residual search constructs readable nonlinear expressions; and
* an out-of-fold stack and adaptive selector combine only historical predictions.

Nothing here is consumed by the live board.  The experiment writes its own report and
prediction artifacts, leaving the immutable 2026 prospective snapshot untouched.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from patron.metrics.backtest import MetricReportConfig, analyze_rankings

DISCOVERY_COLUMNS = (
    "discovery_linear",
    "discovery_nonlinear",
    "discovery_temporal",
    "discovery_latent",
    "discovery_random_conv",
    "discovery_symbolic",
    "discovery_ppg_nonlinear",
    "discovery_ppg_temporal",
    "discovery_ppg_latent",
    "discovery_rich_weekly",
    "discovery_rich_nonlinear",
    "discovery_rich_latent",
    "discovery_rich_ppg",
    "discovery_symbolic_adaptive",
    "discovery_stack",
    "discovery_market_stack",
    "discovery_adaptive",
)

_BASE_FAMILIES = (
    "discovery_linear",
    "discovery_nonlinear",
    "discovery_temporal",
    "discovery_latent",
    "discovery_random_conv",
    "discovery_symbolic",
    "discovery_rich_weekly",
    "discovery_rich_nonlinear",
    "discovery_rich_latent",
    "discovery_symbolic_adaptive",
)

_POSITIONS = ("QB", "RB", "WR", "TE")
_TARGET = "actual_season_points"
_TEMPORAL_SIGNALS = (
    "fantasy_points_ppr",
    "opportunities",
    "targets",
    "carries",
    "attempts",
    "scrimmage_yards",
)
_TEMPORAL_SOURCE_COLUMNS = (
    "fantasy_points_ppr",
    "targets",
    "carries",
    "attempts",
    "rushing_yards",
    "receiving_yards",
)
_HANDCRAFTED_EXCLUSIONS = frozenset(
    {
        "age_factor",
        "depth_role_factor",
        "historical_repl_ppg",
        "historical_vor",
        "historical_ppg_prior",
        "individual_prior_ppg",
        "prior_branch_ppg",
        "component_proj_ppg",
        "expected_games",
        "projected_availability",
        "projection_confidence",
        "availability_confidence",
        "adj_vor",
        "override_delta",
        "early_career_score",
        "source_opportunities_pg",
        "source_opportunities_pg_sq",
        "career_opportunities_log",
        "career_pass_attempts_log",
        "contract_depth_security",
        "vacated_target_early_career",
        "vacated_carry_early_career",
        "young_speed_score",
        "young_burst_score",
        "entering_sophomore",
        "sophomore_rush_share",
    }
)
_EXCLUDED_PREFIXES = (
    "actual_",
    "fitted_",
    "proj_",
    "projected_",
    "adaptive_",
    "market_",
    "week1_proxy_",
    "_research_",
)
_EXCLUDED_COLUMNS = frozenset(
    {
        "season",
        "forecast_season",
        "source_season",
        "rank",
        "metric_version",
        "outcome_complete",
        "actual_matched",
        "actual_played",
    }
)

_STRUCTURAL_PRIMITIVES = frozenset(
    {
        "age_at_season",
        "depth_chart_rank",
        "preseason_years_exp",
        "cutoff_preseason_status_score",
        "cutoff_preseason_rostered",
        "cutoff_preseason_reserve",
        "cutoff_transaction_count",
        "team_changed",
        "team_vacated_target_share",
        "team_vacated_carry_share",
        "source_offense_snap_pct",
        "late_offense_snap_pct",
        "source_offense_snaps_pg",
        "offense_snap_pct_trend",
        "draft_round",
        "draft_pick",
        "was_drafted",
        "career_seasons_observed",
        "career_games",
        "career_opportunities",
        "career_pass_attempts",
        "combine_weight",
        "combine_speed_score",
        "combine_burst_score",
        "combine_agility_score",
        "player_experience",
        "contract_apy_cap_pct",
        "contract_years_remaining",
        "contract_guaranteed_log",
    }
)


@dataclass(frozen=True)
class DiscoveryConfig:
    """Small search budget suitable for 22 genuinely independent season folds."""

    min_train_folds: int = 5
    min_position_rows: int = 80
    inner_validation_folds: int = 3
    feature_counts: tuple[int, ...] = (24, 48, 80)
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0)
    temporal_feature_counts: tuple[int, ...] = (16, 32, 56)
    latent_components: int = 10
    latent_clusters: int = 6
    symbolic_base_features: int = 10
    symbolic_features: int = 16
    random_kernels_per_signal: int = 8
    random_seed: int = 20260830


@dataclass
class _MatrixState:
    features: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray


class _Predictor(Protocol):
    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray: ...


@dataclass
class _RidgePredictor:
    state: _MatrixState
    model: Ridge

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return self.model.predict(_transform_matrix(rows, self.state))


@dataclass
class _BoostPredictor:
    state: _MatrixState
    model: HistGradientBoostingRegressor

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return self.model.predict(_transform_matrix(rows, self.state))


@dataclass
class _LatentPredictor:
    state: _MatrixState
    pca: PCA
    clusters: KMeans
    model: Ridge
    passthrough: int

    def _latent(self, rows: list[dict[str, Any]]) -> np.ndarray:
        matrix = _transform_matrix(rows, self.state)
        components = self.pca.transform(matrix)
        distances = self.clusters.transform(components)
        return np.column_stack((matrix[:, : self.passthrough], components, distances))

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return self.model.predict(self._latent(rows))


@dataclass
class _SymbolicPredictor:
    base: str
    state: _MatrixState
    expressions: tuple[tuple[str, tuple[int, ...], float], ...]
    model: Ridge

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        matrix = _transform_matrix(rows, self.state)
        expressions = _expression_matrix(matrix, self.expressions)
        correction = self.model.predict(expressions)
        baseline = np.array([_finite(row.get(self.base)) or 0.0 for row in rows])
        return baseline + np.clip(correction, -125.0, 125.0)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def automated_feature_pool(frame: pl.DataFrame) -> tuple[str, ...]:
    """Return broad cutoff-safe primitives without outcomes or fitted composites."""
    source_primitives: set[str] = set()
    if "rushing_tds" in frame.columns and "neutral_plays_pg" in frame.columns:
        start = frame.columns.index("rushing_tds")
        end = frame.columns.index("neutral_plays_pg") + 1
        source_primitives.update(frame.columns[start:end])
    output: list[str] = []
    for name, dtype in zip(frame.columns, frame.dtypes, strict=True):
        if not dtype.is_numeric() or dtype == pl.Boolean:
            continue
        if name in _EXCLUDED_COLUMNS or name in _HANDCRAFTED_EXCLUSIONS:
            continue
        if name.startswith(_EXCLUDED_PREFIXES) or name.startswith("ts_"):
            continue
        if (
            source_primitives
            and name not in source_primitives
            and name not in _STRUCTURAL_PRIMITIVES
        ):
            continue
        output.append(name)
    return tuple(output)


def _weekly_descriptor(values: np.ndarray) -> dict[str, float]:
    weeks = np.arange(1.0, len(values) + 1.0)
    mean = float(values.mean())
    std = float(values.std())
    slope = float(np.polyfit(weeks, values, 1)[0]) if len(values) > 1 else 0.0
    centered = values - mean
    lag_denom = float(np.sqrt(np.sum(centered[:-1] ** 2) * np.sum(centered[1:] ** 2)))
    lag1 = float(np.sum(centered[:-1] * centered[1:]) / lag_denom) if lag_denom else 0.0
    total = float(np.sum(np.abs(values)))
    probabilities = np.abs(values) / total if total else np.zeros_like(values)
    positive = probabilities[probabilities > 0]
    entropy = (
        float(-np.sum(positive * np.log(positive)) / math.log(len(values)))
        if len(positive)
        else 0.0
    )
    half = len(values) // 2
    active = np.flatnonzero(values != 0)
    return {
        "mean": mean,
        "std": std,
        "cv": std / (abs(mean) + 1e-6),
        "slope": slope,
        "late_delta": float(values[half:].mean() - values[:half].mean()),
        "last4": float(values[-4:].mean()),
        "max": float(values.max()),
        "q75": float(np.quantile(values, 0.75)),
        "zero_rate": float(np.mean(values == 0)),
        "lag1": lag1,
        "entropy": entropy,
        "peak_week": float(np.argmax(values) + 1) / len(values) if np.any(values) else 0.0,
        "last_active_week": float(active[-1] + 1) / len(values) if len(active) else 0.0,
    }


def build_weekly_discovery_features(
    weeks: pl.DataFrame,
    forecast_seasons: Iterable[int],
    *,
    config: DiscoveryConfig | None = None,
) -> pl.DataFrame:
    """Convert source-season weekly stat lines to generic descriptors and sequences."""
    config = config or DiscoveryConfig()
    required = {
        "season",
        "week",
        "player_id",
        "position",
        *_TEMPORAL_SOURCE_COLUMNS,
    }
    missing = sorted(required - set(weeks.columns))
    if missing:
        raise ValueError(f"weekly discovery input missing columns: {', '.join(missing)}")
    seasons = {int(season) - 1 for season in forecast_seasons}
    frame = (
        weeks.filter(
            pl.col("season").is_in(seasons)
            & pl.col("position").is_in(_POSITIONS)
            & pl.col("week").is_between(1, 18)
        )
        .select("season", "week", "player_id", *_TEMPORAL_SOURCE_COLUMNS)
        .with_columns(
            (
                pl.col("attempts").fill_null(0)
                + pl.col("carries").fill_null(0)
                + pl.col("targets").fill_null(0)
            ).alias("opportunities"),
            (
                pl.col("rushing_yards").fill_null(0)
                + pl.col("receiving_yards").fill_null(0)
            ).alias("scrimmage_yards"),
        )
        .group_by("season", "week", "player_id")
        .agg(*(pl.col(signal).sum().alias(signal) for signal in _TEMPORAL_SIGNALS))
        .sort("season", "player_id", "week")
    )
    groups: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for row in frame.iter_rows(named=True):
        key = (int(row["season"]), str(row["player_id"]))
        signals = groups.setdefault(
            key, {signal: np.zeros(18, dtype=float) for signal in _TEMPORAL_SIGNALS}
        )
        week = int(row["week"]) - 1
        for signal in _TEMPORAL_SIGNALS:
            signals[signal][week] += float(row[signal] or 0.0)
    output: list[dict[str, object]] = []
    for (source_season, player_id), signals in groups.items():
        output_row: dict[str, object] = {
            "forecast_season": source_season + 1,
            "player_id": player_id,
        }
        for signal, values in signals.items():
            for name, value in _weekly_descriptor(values).items():
                output_row[f"ts_{signal}_{name}"] = value
            for week, value in enumerate(values, 1):
                output_row[f"_ts_{signal}_w{week:02d}"] = float(value)
        output.append(output_row)
    result = pl.DataFrame(output) if output else pl.DataFrame()
    return add_random_convolution_features(result, config=config) if result.height else result


def add_random_convolution_features(
    features: pl.DataFrame, *, config: DiscoveryConfig | None = None
) -> pl.DataFrame:
    """Add deterministic ROCKET-style max/positive-proportion sequence features."""
    config = config or DiscoveryConfig()
    rng = np.random.default_rng(config.random_seed)
    columns: dict[str, pl.Series] = {}
    for signal in _TEMPORAL_SIGNALS:
        week_columns = [f"_ts_{signal}_w{week:02d}" for week in range(1, 19)]
        if not set(week_columns).issubset(features.columns):
            continue
        sequence = features.select(week_columns).to_numpy().astype(float)
        for index in range(config.random_kernels_per_signal):
            width = int(rng.choice((3, 5, 7)))
            dilation = int(rng.choice((1, 1, 2)))
            span = (width - 1) * dilation + 1
            kernel = rng.normal(size=width)
            kernel -= kernel.mean()
            norm = np.linalg.norm(kernel)
            kernel = kernel / norm if norm else kernel
            bias = float(rng.normal(scale=0.5))
            windows = np.lib.stride_tricks.sliding_window_view(sequence, span, axis=1)
            responses = windows[:, :, ::dilation] @ kernel + bias
            prefix = f"conv_{signal}_{index:02d}"
            columns[f"{prefix}_max"] = pl.Series(np.max(responses, axis=1))
            columns[f"{prefix}_ppv"] = pl.Series(np.mean(responses > 0, axis=1))
    result = features.with_columns(*[series.alias(name) for name, series in columns.items()])
    raw = [name for name in result.columns if name.startswith("_ts_")]
    return result.drop(raw)


def _raw_matrix(rows: list[dict[str, Any]], features: Sequence[str]) -> np.ndarray:
    return np.array(
        [
            [np.nan if (value := _finite(row.get(name))) is None else value for name in features]
            for row in rows
        ],
        dtype=float,
    )


def _fit_matrix(
    rows: list[dict[str, Any]], features: Sequence[str]
) -> tuple[np.ndarray, _MatrixState]:
    matrix = _raw_matrix(rows, features)
    present = np.sum(np.isfinite(matrix), axis=0)
    medians = np.zeros(matrix.shape[1], dtype=float)
    for index in range(matrix.shape[1]):
        if present[index]:
            medians[index] = float(np.nanmedian(matrix[:, index]))
    filled = np.where(np.isfinite(matrix), matrix, medians)
    lower = np.quantile(filled, 0.005, axis=0)
    upper = np.quantile(filled, 0.995, axis=0)
    filled = np.clip(filled, lower, upper)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    state = _MatrixState(tuple(features), medians, means, scales)
    return (filled - means) / scales, state


def _transform_matrix(rows: list[dict[str, Any]], state: _MatrixState) -> np.ndarray:
    matrix = _raw_matrix(rows, state.features)
    filled = np.where(np.isfinite(matrix), matrix, state.medians)
    standardized = (filled - state.means) / state.scales
    return np.clip(standardized, -8.0, 8.0)


def _select_features(
    rows: list[dict[str, Any]], candidates: Sequence[str], target: np.ndarray, count: int
) -> tuple[tuple[str, ...], list[tuple[str, float]]]:
    if not candidates:
        return (), []
    matrix = _raw_matrix(rows, candidates)
    scores: list[tuple[str, float]] = []
    ranked_target = rankdata(target)
    for index, name in enumerate(candidates):
        column = matrix[:, index]
        present = np.isfinite(column)
        if present.mean() < 0.25 or np.unique(column[present]).size < 3:
            continue
        ranked = rankdata(column[present])
        correlation = np.corrcoef(ranked, ranked_target[present])[0, 1]
        if math.isfinite(float(correlation)):
            scores.append((name, abs(float(correlation))))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return tuple(name for name, _ in scores[:count]), scores


def _targets(rows: list[dict[str, Any]], target: str = _TARGET) -> np.ndarray:
    return np.array([_finite(row.get(target)) or 0.0 for row in rows], dtype=float)


def _rank_quality(rows: list[dict[str, Any]], predictions: np.ndarray, k: int) -> float:
    """Equal-weight hit-rate/NDCG score, averaged across validation seasons."""
    by_season: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        by_season[int(row["forecast_season"])].append(
            (float(prediction), _finite(row.get(_TARGET)) or 0.0)
        )
    scores: list[float] = []
    for values in by_season.values():
        if len(values) < k:
            continue
        predicted_order = sorted(range(len(values)), key=lambda i: values[i][0], reverse=True)
        actual_order = sorted(range(len(values)), key=lambda i: values[i][1], reverse=True)
        actual_top = set(actual_order[:k])
        hit = sum(index in actual_top for index in predicted_order[:k]) / k
        discounts = np.array([1.0 / math.log2(i + 2) for i in range(k)])
        gains = np.array([max(values[i][1], 0.0) for i in predicted_order[:k]])
        ideal = np.array([max(values[i][1], 0.0) for i in actual_order[:k]])
        denominator = float(ideal @ discounts)
        ndcg = float(gains @ discounts) / denominator if denominator else 0.0
        scores.append((hit + ndcg) / 2.0)
    return float(np.mean(scores)) if scores else -math.inf


def _inner_split(
    rows: list[dict[str, Any]], validation_folds: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seasons = sorted({int(row["forecast_season"]) for row in rows})
    validation = set(seasons[-validation_folds:])
    training = [row for row in rows if int(row["forecast_season"]) not in validation]
    checking = [row for row in rows if int(row["forecast_season"]) in validation]
    if not training:
        midpoint = max(1, len(rows) * 2 // 3)
        return rows[:midpoint], rows[midpoint:]
    return training, checking


def _fit_ridge_family(
    rows: list[dict[str, Any]],
    candidates: Sequence[str],
    config: DiscoveryConfig,
    k: int,
    *,
    feature_counts: Sequence[int] | None = None,
) -> tuple[_RidgePredictor | None, dict[str, Any]]:
    inner_train, validation = _inner_split(rows, config.inner_validation_folds)
    counts = feature_counts or config.feature_counts
    best: tuple[float, int, float] | None = None
    for count in counts:
        features, _ = _select_features(inner_train, candidates, _targets(inner_train), count)
        if not features:
            continue
        matrix, state = _fit_matrix(inner_train, features)
        for alpha in config.ridge_alphas:
            model = Ridge(alpha=alpha).fit(matrix, _targets(inner_train))
            score = _rank_quality(
                validation, model.predict(_transform_matrix(validation, state)), k
            )
            key = (score, -count, -alpha)
            if best is None or key > (best[0], -best[1], -best[2]):
                best = (score, count, alpha)
    if best is None:
        return None, {}
    _, count, alpha = best
    features, scores = _select_features(rows, candidates, _targets(rows), count)
    matrix, state = _fit_matrix(rows, features)
    model = Ridge(alpha=alpha).fit(matrix, _targets(rows))
    coefficients = sorted(
        zip(features, model.coef_, strict=True), key=lambda item: abs(float(item[1])), reverse=True
    )
    return _RidgePredictor(state, model), {
        "feature_count": len(features),
        "alpha": alpha,
        "inner_score": best[0],
        "selected_features": [name for name, _ in scores[:count]],
        "top_effects": [[name, round(float(value), 4)] for name, value in coefficients[:12]],
    }


def _fit_boost_family(
    rows: list[dict[str, Any]],
    candidates: Sequence[str],
    config: DiscoveryConfig,
    k: int,
) -> tuple[_BoostPredictor | None, dict[str, Any]]:
    inner_train, validation = _inner_split(rows, config.inner_validation_folds)
    variants = (
        (24, 7, 0.05, 10.0),
        (48, 15, 0.05, 10.0),
        (80, 15, 0.08, 30.0),
    )
    best: tuple[float, tuple[int, int, float, float]] | None = None
    for count, leaves, learning_rate, regularization in variants:
        features, _ = _select_features(inner_train, candidates, _targets(inner_train), count)
        if not features:
            continue
        matrix, state = _fit_matrix(inner_train, features)
        model = HistGradientBoostingRegressor(
            max_iter=160,
            max_leaf_nodes=leaves,
            learning_rate=learning_rate,
            l2_regularization=regularization,
            min_samples_leaf=25,
            random_state=config.random_seed,
        ).fit(matrix, _targets(inner_train))
        score = _rank_quality(validation, model.predict(_transform_matrix(validation, state)), k)
        if best is None or score > best[0]:
            best = (score, (count, leaves, learning_rate, regularization))
    if best is None:
        return None, {}
    count, leaves, learning_rate, regularization = best[1]
    features, scores = _select_features(rows, candidates, _targets(rows), count)
    matrix, state = _fit_matrix(rows, features)
    model = HistGradientBoostingRegressor(
        max_iter=160,
        max_leaf_nodes=leaves,
        learning_rate=learning_rate,
        l2_regularization=regularization,
        min_samples_leaf=25,
        random_state=config.random_seed,
    ).fit(matrix, _targets(rows))
    return _BoostPredictor(state, model), {
        "feature_count": len(features),
        "max_leaf_nodes": leaves,
        "learning_rate": learning_rate,
        "l2_regularization": regularization,
        "inner_score": best[0],
        "selected_features": [name for name, _ in scores[:count]],
    }


def _fit_latent_family(
    rows: list[dict[str, Any]], candidates: Sequence[str], config: DiscoveryConfig
) -> tuple[_LatentPredictor | None, dict[str, Any]]:
    features, scores = _select_features(
        rows, candidates, _targets(rows), max(config.feature_counts)
    )
    if len(features) < 4:
        return None, {}
    matrix, state = _fit_matrix(rows, features)
    components = min(config.latent_components, len(features), len(rows) - 1)
    clusters = min(config.latent_clusters, max(2, len(rows) // 25))
    pca = PCA(n_components=components, random_state=config.random_seed).fit(matrix)
    embedded = pca.transform(matrix)
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=config.random_seed).fit(embedded)
    distances = kmeans.transform(embedded)
    passthrough = min(12, matrix.shape[1])
    latent = np.column_stack((matrix[:, :passthrough], embedded, distances))
    model = Ridge(alpha=10.0).fit(latent, _targets(rows))
    return _LatentPredictor(state, pca, kmeans, model, passthrough), {
        "feature_count": len(features),
        "components": components,
        "clusters": clusters,
        "explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        "selected_features": [name for name, _ in scores[: len(features)]],
    }


def _candidate_expressions(width: int) -> tuple[tuple[str, tuple[int, ...], float], ...]:
    output: list[tuple[str, tuple[int, ...], float]] = []
    for index in range(width):
        output.extend(
            (
                ("identity", (index,), 0.0),
                ("square", (index,), 0.0),
                ("hinge_hi", (index,), 0.5),
                ("hinge_lo", (index,), -0.5),
            )
        )
    for left in range(width):
        for right in range(left + 1, width):
            output.extend(
                (
                    ("product", (left, right), 0.0),
                    ("ratio", (left, right), 0.0),
                    ("ratio", (right, left), 0.0),
                    ("minimum", (left, right), 0.0),
                    ("maximum", (left, right), 0.0),
                )
            )
    return tuple(output)


def _expression_matrix(
    matrix: np.ndarray, expressions: Sequence[tuple[str, tuple[int, ...], float]]
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for operation, indexes, value in expressions:
        left = matrix[:, indexes[0]]
        if operation == "identity":
            result = left
        elif operation == "square":
            result = np.sign(left) * left**2
        elif operation == "hinge_hi":
            result = np.maximum(left - value, 0.0)
        elif operation == "hinge_lo":
            result = np.minimum(left - value, 0.0)
        else:
            right = matrix[:, indexes[1]]
            if operation == "product":
                result = left * right
            elif operation == "ratio":
                result = left / (1.0 + np.abs(right))
            elif operation == "minimum":
                result = np.minimum(left, right)
            else:
                result = np.maximum(left, right)
        columns.append(np.clip(result, -12.0, 12.0))
    return np.column_stack(columns)


def _expression_name(
    expression: tuple[str, tuple[int, ...], float], features: Sequence[str]
) -> str:
    operation, indexes, value = expression
    names = [features[index] for index in indexes]
    if operation == "identity":
        return names[0]
    if operation == "square":
        return f"signed_square({names[0]})"
    if operation.startswith("hinge"):
        direction = ">" if operation == "hinge_hi" else "<"
        return f"hinge({names[0]} {direction} z{value:+.1f})"
    symbol = {"product": "*", "ratio": "/(1+abs)", "minimum": "min", "maximum": "max"}[operation]
    return f"{symbol}({names[0]}, {names[1]})"


def _fit_symbolic_family(
    rows: list[dict[str, Any]],
    candidates: Sequence[str],
    config: DiscoveryConfig,
    *,
    base: str = "fitted_season_points",
) -> tuple[_SymbolicPredictor | None, dict[str, Any]]:
    usable = [row for row in rows if _finite(row.get(base)) is not None]
    if len(usable) < config.min_position_rows:
        return None, {}
    residual = _targets(usable) - np.array([float(row[base]) for row in usable])
    features, _ = _select_features(usable, candidates, residual, config.symbolic_base_features)
    if not features:
        return None, {}
    matrix, state = _fit_matrix(usable, features)
    candidates_expr = _candidate_expressions(len(features))
    generated = _expression_matrix(matrix, candidates_expr)
    residual_rank = rankdata(residual)
    scores: list[tuple[int, float]] = []
    for index in range(generated.shape[1]):
        if np.std(generated[:, index]) < 1e-8:
            continue
        correlation = np.corrcoef(rankdata(generated[:, index]), residual_rank)[0, 1]
        if math.isfinite(float(correlation)):
            scores.append((index, abs(float(correlation))))
    scores.sort(key=lambda item: (-item[1], item[0]))
    chosen = tuple(candidates_expr[index] for index, _ in scores[: config.symbolic_features])
    expression_matrix = _expression_matrix(matrix, chosen)
    model = Ridge(alpha=30.0).fit(expression_matrix, residual)
    effects = sorted(
        zip(chosen, model.coef_, strict=True), key=lambda item: abs(float(item[1])), reverse=True
    )
    return _SymbolicPredictor(base, state, chosen, model), {
        "base": base,
        "base_features": list(features),
        "expressions": [
            [_expression_name(expression, features), round(float(coefficient), 4)]
            for expression, coefficient in effects
        ],
    }


def _fit_family(
    family: str,
    rows: list[dict[str, Any]],
    raw_features: Sequence[str],
    temporal_features: Sequence[str],
    conv_features: Sequence[str],
    rich_features: Sequence[str],
    config: DiscoveryConfig,
    k: int,
) -> tuple[_Predictor | None, dict[str, Any]]:
    if family == "discovery_linear":
        return _fit_ridge_family(rows, raw_features, config, k)
    if family == "discovery_nonlinear":
        return _fit_boost_family(rows, (*raw_features, *temporal_features), config, k)
    if family == "discovery_temporal":
        basics = tuple(
            name
            for name in ("ppg", "season_pts", "games", "age_at_season")
            if name in raw_features
        )
        return _fit_ridge_family(
            rows,
            (*basics, *temporal_features),
            config,
            k,
            feature_counts=config.temporal_feature_counts,
        )
    if family == "discovery_latent":
        return _fit_latent_family(rows, (*raw_features, *temporal_features), config)
    if family == "discovery_random_conv":
        basics = tuple(
            name
            for name in ("ppg", "season_pts", "games", "age_at_season")
            if name in raw_features
        )
        return _fit_ridge_family(
            rows,
            (*basics, *conv_features),
            config,
            k,
            feature_counts=config.temporal_feature_counts,
        )
    if family == "discovery_symbolic":
        return _fit_symbolic_family(rows, (*raw_features, *temporal_features), config)
    if family == "discovery_symbolic_adaptive":
        return _fit_symbolic_family(
            rows,
            (*raw_features, *temporal_features),
            config,
            base="fitted_adaptive_ppg_hybrid",
        )
    if family == "discovery_rich_weekly":
        basics = tuple(
            name
            for name in ("ppg", "season_pts", "games", "age_at_season")
            if name in raw_features
        )
        return _fit_ridge_family(
            rows,
            (*basics, *rich_features),
            config,
            k,
            feature_counts=(32, 64, 96),
        )
    if family == "discovery_rich_nonlinear":
        return _fit_boost_family(rows, (*raw_features, *rich_features), config, k)
    if family == "discovery_rich_latent":
        return _fit_latent_family(rows, (*raw_features, *rich_features), config)
    raise ValueError(f"unknown discovery family: {family}")


def _availability_scale(row: dict[str, Any]) -> float | None:
    return_probability = _finite(row.get("fitted_return_prob"))
    games_if_played = _finite(row.get("fitted_games_if_played"))
    if return_probability is not None and games_if_played is not None:
        return return_probability * games_if_played
    return _finite(row.get("fitted_games"))


def _fit_ppg_families(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    raw_features: Sequence[str],
    temporal_features: Sequence[str],
    rich_features: Sequence[str],
    config: DiscoveryConfig,
    k: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, np.ndarray]]]:
    """Learn player quality on played rows, then apply honest fitted availability."""
    ppg_train = []
    for row in train:
        ppg = _finite(row.get("actual_ppg"))
        if ppg is None or (_finite(row.get("actual_games")) or 0.0) <= 0:
            continue
        ppg_train.append({**row, _TARGET: ppg})
    if len(ppg_train) < config.min_position_rows:
        return [], []
    basics = tuple(
        name
        for name in ("ppg", "season_pts", "games", "age_at_season")
        if name in raw_features
    )
    definitions = (
        (
            "discovery_ppg_nonlinear",
            lambda: _fit_boost_family(
                ppg_train,
                (*raw_features, *temporal_features),
                config,
                k,
            ),
        ),
        (
            "discovery_ppg_temporal",
            lambda: _fit_ridge_family(
                ppg_train,
                (*basics, *temporal_features),
                config,
                k,
                feature_counts=config.temporal_feature_counts,
            ),
        ),
        (
            "discovery_ppg_latent",
            lambda: _fit_latent_family(
                ppg_train,
                (*raw_features, *temporal_features),
                config,
            ),
        ),
        (
            "discovery_rich_ppg",
            lambda: _fit_ridge_family(
                ppg_train,
                (*basics, *rich_features),
                config,
                k,
                feature_counts=(32, 64, 96),
            ),
        ),
    )
    records: list[dict[str, Any]] = []
    predictions: list[tuple[str, np.ndarray]] = []
    for family, fitter in definitions:
        predictor, details = fitter()
        if predictor is None:
            continue
        ppg_values = np.clip(predictor.predict(test), 0.0, None)
        availability = np.array(
            [np.nan if (value := _availability_scale(row)) is None else value for row in test]
        )
        season_points = ppg_values * availability
        season_points[~np.isfinite(season_points)] = np.nan
        predictions.append((family, season_points))
        records.append({"family": family, "n_train": len(ppg_train), **details})
    return records, predictions


def _stack_predictions(
    records: list[dict[str, Any]], config: DiscoveryConfig, top_k: dict[str, int]
) -> list[dict[str, Any]]:
    sources = (
        "fitted_season_points",
        "fitted_two_stage",
        "fitted_adaptive_ppg_hybrid",
        "discovery_linear",
        "discovery_nonlinear",
        "discovery_temporal",
        "discovery_latent",
        "discovery_random_conv",
        "discovery_symbolic",
        "discovery_ppg_nonlinear",
        "discovery_ppg_temporal",
        "discovery_ppg_latent",
        "discovery_rich_weekly",
        "discovery_rich_nonlinear",
        "discovery_rich_latent",
        "discovery_rich_ppg",
        "discovery_symbolic_adaptive",
    )
    seasons = sorted({int(row["forecast_season"]) for row in records})
    fit_records: list[dict[str, Any]] = []
    for season in seasons:
        for position in _POSITIONS:
            train = [
                row
                for row in records
                if row.get("outcome_complete")
                and int(row["forecast_season"]) < season
                and row.get("position") == position
                and _finite(row.get("fitted_season_points")) is not None
            ]
            train_folds = {int(row["forecast_season"]) for row in train}
            if len(train_folds) < config.min_train_folds or len(train) < config.min_position_rows:
                continue
            predictor, details = _fit_ridge_family(
                train,
                sources,
                config,
                top_k[position],
                feature_counts=(4, 6, len(sources)),
            )
            if predictor is None:
                continue
            test = [
                row
                for row in records
                if int(row["forecast_season"]) == season and row.get("position") == position
            ]
            values = np.clip(predictor.predict(test), 0.0, None)
            for row, value in zip(test, values, strict=True):
                row["discovery_stack"] = float(value)
            fit_records.append(
                {
                    "family": "discovery_stack",
                    "forecast_season": season,
                    "position": position,
                    "train_seasons": [min(train_folds), max(train_folds)],
                    **details,
                }
            )
    return fit_records


def _adaptive_predictions(
    records: list[dict[str, Any]], config: DiscoveryConfig, top_k: dict[str, int]
) -> list[dict[str, Any]]:
    candidates = ("fitted_season_points", *DISCOVERY_COLUMNS[:-1])
    seasons = sorted({int(row["forecast_season"]) for row in records})
    choices: list[dict[str, Any]] = []
    for season in seasons:
        for position in _POSITIONS:
            history = [
                row
                for row in records
                if row.get("outcome_complete")
                and int(row["forecast_season"]) < season
                and row.get("position") == position
            ]
            history_folds = sorted({int(row["forecast_season"]) for row in history})
            best_source = "fitted_season_points"
            best_score = -math.inf
            if len(history_folds) >= config.min_train_folds:
                for source in candidates:
                    scores: list[float] = []
                    for fold in history_folds:
                        rows = [row for row in history if int(row["forecast_season"]) == fold]
                        usable = [row for row in rows if _finite(row.get(source)) is not None]
                        if len(usable) < top_k[position]:
                            continue
                        predicted = np.array([float(row[source]) for row in usable])
                        scores.append(_rank_quality(usable, predicted, top_k[position]))
                    if len(scores) >= config.min_train_folds:
                        score = float(np.mean(scores))
                        if score > best_score + 1e-12:
                            best_source, best_score = source, score
            test = [
                row
                for row in records
                if int(row["forecast_season"]) == season and row.get("position") == position
            ]
            for row in test:
                value = _finite(row.get(best_source))
                if value is None:
                    value = _finite(row.get("fitted_season_points"))
                row["discovery_adaptive"] = value
            choices.append(
                {
                    "family": "discovery_adaptive",
                    "forecast_season": season,
                    "position": position,
                    "selected_source": best_source,
                    "selection_score": None if best_score == -math.inf else round(best_score, 4),
                    "selection_folds": len(history_folds),
                }
            )
    return choices


def add_market_discovery_predictions(
    records: list[dict[str, Any]],
    config: DiscoveryConfig,
    top_k: dict[str, int],
) -> list[dict[str, Any]]:
    """Learn corrections to dated ECR using only earlier ECR-covered folds."""
    sources = (
        "market_ecr_score",
        "market_ecr_sd",
        "fitted_adaptive_ppg_hybrid",
        "fitted_two_stage",
        "discovery_stack",
        "discovery_symbolic",
        "discovery_ppg_latent",
    )
    market_config = replace(config, inner_validation_folds=1)
    seasons = sorted({int(row["forecast_season"]) for row in records})
    fit_records: list[dict[str, Any]] = []
    for season in seasons:
        for position in _POSITIONS:
            train = [
                row
                for row in records
                if row.get("outcome_complete")
                and int(row["forecast_season"]) < season
                and row.get("position") == position
                and _finite(row.get("market_ecr_score")) is not None
            ]
            train_folds = {int(row["forecast_season"]) for row in train}
            if len(train_folds) < 2 or len(train) < config.min_position_rows:
                continue
            predictor, details = _fit_ridge_family(
                train,
                sources,
                market_config,
                top_k[position],
                feature_counts=(2, 4, len(sources)),
            )
            if predictor is None:
                continue
            test = [
                row
                for row in records
                if int(row["forecast_season"]) == season
                and row.get("position") == position
                and _finite(row.get("market_ecr_score")) is not None
            ]
            values = np.clip(predictor.predict(test), 0.0, None)
            for row, value in zip(test, values, strict=True):
                row["discovery_market_stack"] = float(value)
            fit_records.append(
                {
                    "family": "discovery_market_stack",
                    "forecast_season": season,
                    "position": position,
                    "train_seasons": [min(train_folds), max(train_folds)],
                    "n_train": len(train),
                    **details,
                }
            )
    return fit_records


def add_adaptive_residual_predictions(
    records: list[dict[str, Any]],
    frame: pl.DataFrame,
    config: DiscoveryConfig,
) -> list[dict[str, Any]]:
    """Search nonlinear residual formulas around the existing adaptive forecast."""
    raw_features = automated_feature_pool(frame)
    temporal = tuple(name for name in frame.columns if name.startswith("ts_"))
    seasons = sorted({int(row["forecast_season"]) for row in records})
    completed = {
        int(row["forecast_season"])
        for row in records
        if row.get("outcome_complete")
    }
    fit_records: list[dict[str, Any]] = []
    for season in seasons:
        train_seasons = sorted(value for value in completed if value < season)
        if len(train_seasons) < config.min_train_folds:
            continue
        for position in _POSITIONS:
            train = [
                row
                for row in records
                if row.get("outcome_complete")
                and int(row["forecast_season"]) < season
                and row.get("position") == position
            ]
            test = [
                row
                for row in records
                if int(row["forecast_season"]) == season and row.get("position") == position
            ]
            predictor, details = _fit_symbolic_family(
                train,
                (*raw_features, *temporal),
                config,
                base="fitted_adaptive_ppg_hybrid",
            )
            if predictor is None:
                continue
            values = np.clip(predictor.predict(test), 0.0, None)
            for row, value in zip(test, values, strict=True):
                row["discovery_symbolic_adaptive"] = float(value)
            fit_records.append(
                {
                    "family": "discovery_symbolic_adaptive",
                    "forecast_season": season,
                    "position": position,
                    "train_seasons": [min(train_seasons), max(train_seasons)],
                    "n_train": len(train),
                    **details,
                }
            )
    return fit_records


def fit_discovery_walk_forward(
    predictions: pl.DataFrame,
    *,
    temporal_features: pl.DataFrame | None = None,
    rich_features: pl.DataFrame | None = None,
    config: DiscoveryConfig | None = None,
    top_k: dict[str, int] | None = None,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Fit every discovery family on prior folds and score each untouched season."""
    config = config or DiscoveryConfig()
    top_k = top_k or {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
    frame = predictions.drop([name for name in DISCOVERY_COLUMNS if name in predictions.columns])
    if temporal_features is not None and temporal_features.height:
        frame = frame.join(
            temporal_features,
            on=["forecast_season", "player_id"],
            how="left",
            validate="m:1",
        )
    if rich_features is not None and rich_features.height:
        frame = frame.join(
            rich_features,
            on=["forecast_season", "player_id"],
            how="left",
            validate="m:1",
        )
    records = frame.to_dicts()
    raw_features = automated_feature_pool(frame)
    temporal = tuple(name for name in frame.columns if name.startswith("ts_"))
    convolutions = tuple(name for name in frame.columns if name.startswith("conv_"))
    rich = tuple(name for name in frame.columns if name.startswith("rich_"))
    seasons = sorted({int(row["forecast_season"]) for row in records})
    completed = {
        int(row["forecast_season"])
        for row in records
        if row.get("outcome_complete")
    }
    fit_records: list[dict[str, Any]] = []
    for season in seasons:
        train_seasons = sorted(value for value in completed if value < season)
        if len(train_seasons) < config.min_train_folds:
            continue
        for position in _POSITIONS:
            train = [
                row
                for row in records
                if row.get("outcome_complete")
                and int(row["forecast_season"]) < season
                and row.get("position") == position
            ]
            test = [
                row
                for row in records
                if int(row["forecast_season"]) == season and row.get("position") == position
            ]
            if len(train) < config.min_position_rows or not test:
                continue
            for family in _BASE_FAMILIES:
                predictor, details = _fit_family(
                    family,
                    train,
                    raw_features,
                    temporal,
                    convolutions,
                    rich,
                    config,
                    top_k[position],
                )
                if predictor is None:
                    continue
                values = np.clip(predictor.predict(test), 0.0, None)
                for row, value in zip(test, values, strict=True):
                    row[family] = float(value)
                fit_records.append(
                    {
                        "family": family,
                        "forecast_season": season,
                        "position": position,
                        "train_seasons": [min(train_seasons), max(train_seasons)],
                        "n_train": len(train),
                        **details,
                    }
                )
            ppg_records, ppg_predictions = _fit_ppg_families(
                train,
                test,
                raw_features,
                temporal,
                rich,
                config,
                top_k[position],
            )
            for family, values in ppg_predictions:
                for row, value in zip(test, values, strict=True):
                    row[family] = float(value) if math.isfinite(float(value)) else None
            for details in ppg_records:
                fit_records.append(
                    {
                        **details,
                        "forecast_season": season,
                        "position": position,
                        "train_seasons": [min(train_seasons), max(train_seasons)],
                    }
                )
    fit_records.extend(_stack_predictions(records, config, top_k))
    fit_records.extend(add_market_discovery_predictions(records, config, top_k))
    fit_records.extend(_adaptive_predictions(records, config, top_k))
    output = frame
    for name in DISCOVERY_COLUMNS:
        output = output.with_columns(
            pl.Series(name, [_finite(row.get(name)) for row in records], dtype=pl.Float64)
        )
    return output, fit_records


def _stable_features(fits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: Counter[tuple[str, str]] = Counter()
    appearances: Counter[tuple[str, str, str]] = Counter()
    for fit in fits:
        family = str(fit.get("family"))
        position = str(fit.get("position"))
        selected = fit.get("selected_features") or fit.get("base_features") or []
        totals[(family, position)] += 1
        for feature in selected:
            appearances[(family, position, str(feature))] += 1
    output = []
    for (family, position, feature), count in appearances.items():
        folds = totals[(family, position)]
        output.append(
            {
                "family": family,
                "position": position,
                "feature": feature,
                "folds_selected": count,
                "eligible_folds": folds,
                "selection_rate": round(count / folds, 4) if folds else None,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["family"],
            row["position"],
            -float(row["selection_rate"] or 0),
            row["feature"],
        ),
    )


def _paired_randomization_p(
    entry: dict[str, Any] | None, ranking_results: list[dict[str, Any]]
) -> float | None:
    if not entry or not entry.get("best_baseline"):
        return None
    baseline = next(
        (
            row
            for row in ranking_results
            if row["ranker"] == entry["best_baseline"]
            and row["window"] == entry["window"]
            and row["target"] == entry["target"]
            and row["position"] == entry["position"]
        ),
        None,
    )
    if baseline is None:
        return None
    baseline_folds = {row["forecast_season"]: row for row in baseline["fold_results"]}
    differences = np.array(
        [
            row["hit_rate"] - baseline_folds[row["forecast_season"]]["hit_rate"]
            for row in entry["fold_results"]
            if row["forecast_season"] in baseline_folds
        ],
        dtype=float,
    )
    observed = float(differences.mean()) if len(differences) else 0.0
    if len(differences) < 3 or observed <= 0:
        return None
    seed = 20260830 + sum(ord(char) for char in str(entry["ranker"]))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(20_000, len(differences)))
    null_lifts = (signs * differences).mean(axis=1)
    return round(float((1 + np.sum(null_lifts >= observed)) / (len(null_lifts) + 1)), 4)


def _promotion_gates(ranking_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in DISCOVERY_COLUMNS:
        entries = [
            row
            for row in ranking_results
            if row["ranker"] == candidate
            and row["position"] == "ALL"
            and row["target"] == "actual_availability_value"
        ]
        by_window = {row["window"]: row for row in entries}
        long = by_window.get("long_horizon") or by_window.get("long") or by_window.get("all")
        modern = by_window.get("modern") or by_window.get("recent")
        market = by_window.get("market")
        chance_p = _paired_randomization_p(long, ranking_results)
        chance_pass = chance_p is not None and chance_p <= 0.1
        long_pass = bool(long and long.get("beats_baseline") and (long.get("ndcg_lift") or 0) >= 0)
        modern_pass = bool(
            modern
            and (modern.get("hit_rate_lift") or 0) > 0
            and (modern.get("folds_won") or 0) >= (modern.get("folds_lost") or 0)
        )
        market_pass = bool(
            market
            and market.get("market_folds", 0) >= 3
            and (market.get("market_lift") or 0) > 0
            and market.get("beats_market")
        )
        output.append(
            {
                "candidate": candidate,
                "long_pass": long_pass,
                "modern_pass": modern_pass,
                "market_pass": market_pass,
                "chance_pass": chance_pass,
                "promotion_ready": long_pass and modern_pass and market_pass and chance_pass,
                "long_hit_lift": long.get("hit_rate_lift") if long else None,
                "long_ndcg_lift": long.get("ndcg_lift") if long else None,
                "modern_hit_lift": modern.get("hit_rate_lift") if modern else None,
                "market_hit_lift": market.get("market_lift") if market else None,
                "long_randomization_p": chance_p,
            }
        )
    return output


def build_discovery_report(
    predictions: pl.DataFrame,
    report_config: MetricReportConfig,
    fits: list[dict[str, Any]],
    *,
    config: DiscoveryConfig | None = None,
) -> dict[str, Any]:
    config = config or DiscoveryConfig()
    candidates = (
        "fitted_season_points",
        "fitted_two_stage",
        "fitted_adaptive_ppg_hybrid",
        *DISCOVERY_COLUMNS,
    )
    ranking = replace(report_config.ranking, candidates=candidates)
    discovery_report_config = replace(report_config, ranking=ranking)
    ranking_results = analyze_rankings(predictions, discovery_report_config)
    return {
        "schema_version": 1,
        "title": "Automated feature discovery lab",
        "generated_at": datetime.now(UTC).isoformat(),
        "production_impact": "none; report-only experiment",
        "prospective_2026_status": "immutable snapshot untouched",
        "configuration": asdict(config),
        "feature_pools": {
            "raw": len(automated_feature_pool(predictions)),
            "temporal": sum(name.startswith("ts_") for name in predictions.columns),
            "random_convolution": sum(name.startswith("conv_") for name in predictions.columns),
            "rich_weekly": sum(name.startswith("rich_") for name in predictions.columns),
        },
        "methods": {
            "discovery_linear": "nested selected ridge over minimally processed inputs",
            "discovery_nonlinear": "nested shallow histogram gradient boosting",
            "discovery_temporal": "generic weekly trend/volatility/entropy descriptors",
            "discovery_latent": "training-only PCA components and K-means distances",
            "discovery_random_conv": "deterministic ROCKET-style weekly convolutions",
            "discovery_symbolic": "ridge-regularized symbolic correction of incumbent residuals",
            "discovery_ppg_nonlinear": (
                "automatic PPG boost multiplied by prior-fold fitted availability"
            ),
            "discovery_ppg_temporal": (
                "weekly-shape PPG ridge multiplied by prior-fold fitted availability"
            ),
            "discovery_ppg_latent": (
                "latent-archetype PPG model multiplied by prior-fold fitted availability"
            ),
            "discovery_rich_weekly": "selected ridge over multiseason weekly role/context",
            "discovery_rich_nonlinear": "shallow boost over weekly role/context",
            "discovery_rich_latent": "PCA/K-means archetypes over weekly role/context",
            "discovery_rich_ppg": "weekly role/context PPG multiplied by fitted availability",
            "discovery_symbolic_adaptive": (
                "training-only symbolic residual correction around the adaptive forecast"
            ),
            "discovery_stack": "ridge stack of prior-fold out-of-fold predictions",
            "discovery_market_stack": "dated-ECR stack fitted only on earlier market folds",
            "discovery_adaptive": "position selector using only prior completed fold quality",
        },
        "search_rounds": [
            {
                "round": 1,
                "hypothesis": "broad primitives contain nonlinear thresholds and interactions",
                "candidates": ["discovery_linear", "discovery_nonlinear"],
            },
            {
                "round": 2,
                "hypothesis": "weekly shape and latent player archetypes add missing signal",
                "candidates": [
                    "discovery_temporal",
                    "discovery_random_conv",
                    "discovery_latent",
                ],
            },
            {
                "round": 3,
                "hypothesis": "quality and availability should be learned separately",
                "candidates": [
                    "discovery_ppg_nonlinear",
                    "discovery_ppg_temporal",
                    "discovery_ppg_latent",
                ],
            },
            {
                "round": 4,
                "hypothesis": "richer weekly role/context separates inactive from unused weeks",
                "candidates": [
                    "discovery_rich_weekly",
                    "discovery_rich_nonlinear",
                    "discovery_rich_latent",
                    "discovery_rich_ppg",
                ],
            },
            {
                "round": 5,
                "hypothesis": "stable residual formulas and OOF stacking improve the incumbent",
                "candidates": [
                    "discovery_symbolic",
                    "discovery_symbolic_adaptive",
                    "discovery_stack",
                    "discovery_adaptive",
                ],
            },
            {
                "round": 6,
                "hypothesis": "learned corrections to dated consensus can improve ECR",
                "candidates": ["discovery_market_stack"],
            },
        ],
        "ranking_results": ranking_results,
        "promotion_gates": _promotion_gates(ranking_results),
        "stable_features": _stable_features(fits),
        "fit_records": fits,
        "guardrails": [
            "Every transformer, selector, clusterer, expression search, and model is "
            "fitted only on seasons before the scored forecast.",
            "Outer backtest results never choose features or hyperparameters; blocked "
            "inner seasons do.",
            "Player and team identifiers, outcomes, fitted outputs, market ECR, and "
            "handcrafted projection composites are excluded from the broad raw pool.",
            "The stack trains only on historical out-of-fold base predictions.",
            "No discovery output is connected to the production board or frozen 2026 forecast.",
        ],
    }


def render_discovery_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['title']}",
        "",
        f"Generated `{report['generated_at']}`. Production impact: "
        f"**{report['production_impact']}**.",
        "",
        "## Promotion gates",
        "",
        "| candidate | long | modern | ECR | promote | long hit lift | modern hit lift | "
        "ECR hit lift | chance p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def value(row: dict[str, Any], key: str) -> str:
        number = row.get(key)
        return "—" if number is None else f"{float(number):+.3f}"

    for row in report["promotion_gates"]:
        lines.append(
            f"| `{row['candidate']}` | {'pass' if row['long_pass'] else 'fail'} | "
            f"{'pass' if row['modern_pass'] else 'fail'} | "
            f"{'pass' if row['market_pass'] else 'fail'} | "
            f"{'YES' if row['promotion_ready'] else 'no'} | {value(row, 'long_hit_lift')} | "
            f"{value(row, 'modern_hit_lift')} | {value(row, 'market_hit_lift')} | "
            f"{value(row, 'long_randomization_p')} |"
        )
    lines.extend(
        [
            "",
            "## Stable automatically selected features",
            "",
            "Selection rate is the share of eligible walk-forward refits selecting the feature.",
            "",
            "| family | pos | feature | selected | rate |",
            "|---|---:|---|---:|---:|",
        ]
    )
    stable = [row for row in report["stable_features"] if row["selection_rate"] >= 0.5]
    compact: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in stable:
        grouped[(row["family"], row["position"])].append(row)
    for group in grouped.values():
        compact.extend(group[:5])
    for row in compact:
        lines.append(
            f"| `{row['family']}` | {row['position']} | `{row['feature']}` | "
            f"{row['folds_selected']}/{row['eligible_folds']} | {row['selection_rate']:.0%} |"
        )
    lines.extend(["", "## Guardrails", ""])
    lines.extend(f"- {item}" for item in report["guardrails"])
    lines.append("")
    return "\n".join(lines)


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)
