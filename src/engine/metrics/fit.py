"""Walk-forward fitted rankers: learn feature weights from completed backtest folds.

v2's hand-set constants (season decay, branch blend, shrinkage samples, caps) cannot be
validated individually by the metric report; the report can only say whether the final
number beats the naive baseline.  This stage closes the loop the way the review asked
for: it takes the small set of signals that survived the ablations, fits per-position
ridge weights on every completed fold *strictly before* the forecast season, and
emits each fitted projection as one more candidate ranker.  The fitted blend then has
to win the same top-K test as everything else.

Walk-forward means the 2019 forecast is fitted on 2004–2018 outcomes, the 2020 forecast
on 2004–2019, and so on; the pending forecast uses every completed fold.  No row is
ever scored by a model that saw its own outcome.  Ridge strength is chosen the same
way: for each refit, candidates from ``lambda_grid`` are scored on the last few
training seasons (fitted on the seasons before those), and the winner is refitted on
the whole training window.

Models are declared as specs (``config/metric_report.yaml`` → ``fit.models``).  A
``ridge`` spec names a target and features; a ``product`` spec multiplies earlier
outputs, which is how season-point projections are composed from a per-game model and
an availability model without feeding availability into the per-game fit twice.
    ``adaptive_select`` can choose among earlier outputs using only the completed folds
    before each forecast.  The fit is deliberately small and readable so year-to-year
    stability is itself evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata

from engine.metrics.availability import constrain_games

logger = logging.getLogger(__name__)

FITTED_PPG = "fitted_ppg"
FITTED_SEASON_POINTS = "fitted_season_points"
ACTUAL_PLAYED = "actual_played"
ARTIFACT_SCHEMA_VERSION = 3
MODEL_VERSION = "ridge-walk-forward-3"

_CORE_FEATURES = (
    "historical_ppg_prior",
    "ppg",
    "age_factor",
    "depth_role_factor",
    "component_proj_ppg",
)
_DEFAULT_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
_AVAILABILITY_FEATURES = (
    "expected_games",
    "projected_availability",
    "games",
    "age_factor",
    "historical_ppg_prior",
    "depth_role_factor",
)


@dataclass(frozen=True)
class ModelSpec:
    """One fitted output column.

    ``ridge``: regress ``target`` on ``features``.  ``absent_as_zero`` trains on every
    row of a completed fold, scoring players who never appeared as zero (the honest
    season-points population); ``played_only`` trains only on players who did appear.
    ``product``: multiply the named earlier outputs. ``coalesce``: take the first
    present output, used to combine mutually exclusive rookie and returner models.
    ``clip`` bounds the prediction.
    """

    name: str
    population: str = "all"
    kind: str = "ridge"
    target: str = "actual_ppg"
    features: tuple[str, ...] = _CORE_FEATURES
    absent_as_zero: bool = False
    played_only: bool = False
    clip: tuple[float | None, float | None] = (None, None)
    lambda_grid: tuple[float, ...] = _DEFAULT_GRID
    factors: tuple[str, ...] = ()
    by_position: tuple[tuple[str, str], ...] = ()
    fallback: str | None = None
    selection_metric: str = "hit_rate"
    selection_top_k: tuple[tuple[str, int], ...] = ()
    apply_live: bool = True
    # Train and predict only on rows where these features are present (e.g. a market
    # feature that exists from 2021), instead of imputing the training mean.
    require_features: tuple[str, ...] = ()
    # Optional broad research pool. Features are ranked against the target using only
    # the outer fold's historical rows, then capped before ridge fitting.
    feature_prefixes: tuple[str, ...] = ()
    max_features: int | None = None
    # Override the global minimum number of completed folds before scoring.
    min_train_folds: int | None = None

    def __post_init__(self) -> None:
        if self.population not in {"all", "returner", "rookie"}:
            raise ValueError(f"Unknown model population: {self.population}")

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> ModelSpec:
        clip = raw.get("clip") or (None, None)
        return cls(
            name=str(raw["name"]),
            population=str(raw.get("population", "all")),
            kind=str(raw.get("kind") or "ridge"),
            target=str(raw.get("target") or "actual_ppg"),
            features=tuple(raw.get("features") or _CORE_FEATURES),
            absent_as_zero=bool(raw.get("absent_as_zero", False)),
            played_only=bool(raw.get("played_only", False)),
            clip=(
                None if clip[0] is None else float(clip[0]),
                None if clip[1] is None else float(clip[1]),
            ),
            lambda_grid=tuple(float(v) for v in (raw.get("lambda_grid") or _DEFAULT_GRID)),
            factors=tuple(raw.get("factors") or ()),
            by_position=tuple(
                (str(position), str(source))
                for position, source in (raw.get("by_position") or {}).items()
            ),
            fallback=str(raw["fallback"]) if raw.get("fallback") else None,
            selection_metric=str(raw.get("selection_metric") or "hit_rate"),
            selection_top_k=tuple(
                (str(position), int(k))
                for position, k in (raw.get("selection_top_k") or {}).items()
            ),
            apply_live=bool(raw.get("apply_live", True)),
            require_features=tuple(raw.get("require_features") or ()),
            feature_prefixes=tuple(raw.get("feature_prefixes") or ()),
            max_features=(
                int(raw["max_features"]) if raw.get("max_features") is not None else None
            ),
            min_train_folds=(
                int(raw["min_train_folds"]) if raw.get("min_train_folds") is not None else None
            ),
        )


def default_model_specs() -> tuple[ModelSpec, ...]:
    """The comparison the review asked for, as named candidate rankers.

    - ``fitted_ppg``: per-active-game PPG *without* availability as an input.
    - ``fitted_games``: calibrated games projection on the full population.
    - ``fitted_season_points`` = ``fitted_ppg × fitted_games`` (PPG × calibrated
      availability).
    - ``fitted_season_points_direct``: season points fitted directly with absent
      players scored as zero.
    - ``fitted_two_stage``: P(appears) × games given appearance × PPG.
    """
    return (
        ModelSpec(name=FITTED_PPG),
        ModelSpec(
            name="fitted_games",
            target="actual_games",
            features=_AVAILABILITY_FEATURES,
            absent_as_zero=True,
            clip=(0.0, 17.0),
        ),
        ModelSpec(name=FITTED_SEASON_POINTS, kind="product", factors=(FITTED_PPG, "fitted_games")),
        ModelSpec(
            name="fitted_season_points_direct",
            target="actual_season_points",
            features=(*_CORE_FEATURES, "expected_games", "season_pts"),
            absent_as_zero=True,
            clip=(0.0, None),
        ),
        ModelSpec(
            name="fitted_return_prob",
            target=ACTUAL_PLAYED,
            features=_AVAILABILITY_FEATURES,
            absent_as_zero=True,
            clip=(0.0, 1.0),
        ),
        ModelSpec(
            name="fitted_games_if_played",
            target="actual_games",
            features=_AVAILABILITY_FEATURES,
            played_only=True,
            clip=(1.0, 17.0),
        ),
        ModelSpec(
            name="fitted_two_stage",
            kind="product",
            factors=("fitted_return_prob", "fitted_games_if_played", FITTED_PPG),
        ),
    )


@dataclass(frozen=True)
class FitConfig:
    models: tuple[ModelSpec, ...] = field(default_factory=default_model_specs)
    min_train_folds: int = 3
    min_position_rows: int = 40
    per_position: bool = True
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE")
    inner_validation_folds: int = 3
    # Backward-compatible shorthand: a bare feature list configures ``fitted_ppg`` only.
    features: tuple[str, ...] | None = None
    target: str = "actual_ppg"
    ridge_lambda: float | None = None

    def __post_init__(self) -> None:
        if self.features is not None or self.ridge_lambda is not None:
            grid = (self.ridge_lambda,) if self.ridge_lambda is not None else _DEFAULT_GRID
            spec = ModelSpec(
                name=FITTED_PPG,
                target=self.target,
                features=self.features or _CORE_FEATURES,
                lambda_grid=grid,
            )
            games = ModelSpec(
                name="fitted_games",
                target="actual_games",
                features=("expected_games",),
                absent_as_zero=True,
                clip=(0.0, 17.0),
                lambda_grid=grid,
            )
            product = ModelSpec(
                name=FITTED_SEASON_POINTS, kind="product", factors=(FITTED_PPG, "fitted_games")
            )
            object.__setattr__(self, "models", (spec, games, product))

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> FitConfig:
        if not raw:
            return cls()
        defaults = cls()
        min_train_folds = int(raw.get("min_train_folds", defaults.min_train_folds))
        min_position_rows = int(raw.get("min_position_rows", defaults.min_position_rows))
        per_position = bool(raw.get("per_position", defaults.per_position))
        positions = tuple(str(value) for value in (raw.get("positions") or defaults.positions))
        if raw.get("models"):
            return cls(
                models=tuple(ModelSpec.from_raw(entry) for entry in raw["models"]),
                inner_validation_folds=int(
                    raw.get("inner_validation_folds", defaults.inner_validation_folds)
                ),
                min_train_folds=min_train_folds,
                min_position_rows=min_position_rows,
                per_position=per_position,
                positions=positions,
            )
        return cls(
            features=tuple(raw["features"]) if raw.get("features") else None,
            target=str(raw.get("target") or "actual_ppg"),
            ridge_lambda=float(raw["ridge_lambda"])
            if raw.get("ridge_lambda") is not None
            else None,
            min_train_folds=min_train_folds,
            min_position_rows=min_position_rows,
            per_position=per_position,
            positions=positions,
        )

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.models)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


@dataclass
class RidgeModel:
    features: tuple[str, ...]
    means: list[float]
    scales: list[float]
    coefficients: list[float]
    intercept: float
    n_train: int
    ridge_lambda: float = 1.0
    clip: tuple[float | None, float | None] = (None, None)

    def predict(self, row: dict[str, Any]) -> float:
        total = self.intercept
        for index, feature in enumerate(self.features):
            value = _number(row.get(feature))
            if value is None:
                value = self.means[index]
            total += self.coefficients[index] * (value - self.means[index]) / self.scales[index]
        low, high = self.clip
        if low is not None:
            total = max(total, low)
        if high is not None:
            total = min(total, high)
        return total

    def record(self) -> dict[str, Any]:
        """Everything needed to re-apply the model to new rows (e.g. the live board)."""
        return {
            "n_train": self.n_train,
            "ridge_lambda": self.ridge_lambda,
            "clip": list(self.clip),
            # Full precision: these are re-applied to the live board, not just displayed.
            "intercept": self.intercept,
            "coefficients": dict(zip(self.features, self.coefficients, strict=True)),
            "standardization": {
                feature: {"mean": mean, "scale": scale}
                for feature, mean, scale in zip(self.features, self.means, self.scales, strict=True)
            },
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> RidgeModel:
        features = tuple(record["coefficients"])
        standardization = record.get("standardization") or {}
        clip = record.get("clip") or [None, None]
        return cls(
            features=features,
            means=[float(standardization.get(f, {}).get("mean", 0.0)) for f in features],
            scales=[float(standardization.get(f, {}).get("scale", 1.0)) or 1.0 for f in features],
            coefficients=[float(record["coefficients"][f]) for f in features],
            intercept=float(record["intercept"]),
            n_train=int(record.get("n_train", 0)),
            ridge_lambda=float(record.get("ridge_lambda", 1.0)),
            clip=(clip[0], clip[1]),
        )


def _predict(model: RidgeModel, row: dict[str, Any], spec: ModelSpec) -> float:
    value = model.predict(row)
    if spec.target == "actual_games":
        return constrain_games(value, row)
    if spec.target in {"actual_season_points", ACTUAL_PLAYED} and row.get(
        "known_available_games_cap"
    ) == 0:
        return 0.0
    return value


def _has_required(row: dict[str, Any], spec: ModelSpec) -> bool:
    rookie = (_number(row.get("rookie_indicator")) or 0) > 0
    returning = (_number(row.get("returning_indicator")) or 0) > 0
    if spec.population == "rookie" and not rookie:
        return False
    if spec.population == "returner" and (rookie or not returning):
        return False
    return all(_number(row.get(feature)) is not None for feature in spec.require_features)


def _training_target(row: dict[str, Any], spec: ModelSpec) -> float | None:
    """The outcome a row contributes to training, or None to exclude it."""
    if not _has_required(row, spec):
        return None
    if spec.played_only and not (_number(row.get("actual_games")) or 0.0) > 0:
        return None
    value = _number(row.get(spec.target))
    if value is None and spec.absent_as_zero:
        return 0.0
    return value


def fit_ridge(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    target: str,
    ridge_lambda: float,
    spec: ModelSpec | None = None,
) -> RidgeModel | None:
    """Ridge on standardized features; intercept is the target mean and unpenalized.

    Missing feature values are imputed with the training mean, so a signal absent for
    a whole era (route data before 2016, say) simply carries no weight there.
    """
    spec = spec or ModelSpec(name=target, target=target, features=features)
    samples = [(row, y) for row in rows if (y := _training_target(row, spec)) is not None]
    if len(samples) <= len(features) + 1:
        return None
    matrix = np.array(
        [
            [np.nan if (v := _number(row.get(f))) is None else v for f in features]
            for row, _ in samples
        ],
        dtype=float,
    )
    targets = np.array([y for _, y in samples], dtype=float)
    present = np.sum(~np.isnan(matrix), axis=0)
    means = np.divide(
        np.nansum(matrix, axis=0),
        present,
        out=np.zeros(matrix.shape[1], dtype=float),
        where=present > 0,
    )
    filled = np.where(np.isnan(matrix), means, matrix)
    scales = filled.std(axis=0)
    scales = np.where(scales > 1e-6, scales, 1.0)
    standardized = (filled - means) / scales
    intercept = float(targets.mean())
    centered = targets - intercept
    gram = standardized.T @ standardized + ridge_lambda * len(samples) * np.eye(len(features))
    moment = standardized.T @ centered
    try:
        coefficients = np.linalg.solve(gram, moment)
    except np.linalg.LinAlgError:
        return None
    return RidgeModel(
        features=features,
        means=[float(v) for v in means],
        scales=[float(v) for v in scales],
        coefficients=[float(v) for v in coefficients],
        intercept=intercept,
        n_train=len(samples),
        ridge_lambda=ridge_lambda,
        clip=spec.clip,
    )


def _select_model_features(
    rows: list[dict[str, Any]],
    candidates: tuple[str, ...],
    spec: ModelSpec,
) -> tuple[str, ...]:
    """Select a bounded feature set using only the current outer training window."""
    if spec.max_features is None or len(candidates) <= spec.max_features:
        return candidates
    samples = [(row, y) for row in rows if (y := _training_target(row, spec)) is not None]
    if not samples:
        return candidates[: spec.max_features]
    targets = np.array([y for _, y in samples], dtype=float)
    ranked_target = rankdata(targets)
    scores: list[tuple[str, float]] = []
    for feature in candidates:
        values = np.array(
            [
                np.nan if (value := _number(row.get(feature))) is None else value
                for row, _ in samples
            ],
            dtype=float,
        )
        present = np.isfinite(values)
        if present.mean() < 0.25 or np.unique(values[present]).size < 3:
            continue
        correlation = np.corrcoef(rankdata(values[present]), ranked_target[present])[0, 1]
        if math.isfinite(float(correlation)):
            scores.append((feature, abs(float(correlation))))
    scores.sort(key=lambda item: (-item[1], item[0]))
    selected = tuple(feature for feature, _ in scores[: spec.max_features])
    return selected or candidates[: spec.max_features]


def _select_lambda(
    train_by_season: dict[int, list[dict[str, Any]]],
    spec: ModelSpec,
    features: tuple[str, ...],
    inner_folds: int,
) -> float:
    """Nested walk-forward: score each λ on the last training seasons, fitted before them."""
    if len(spec.lambda_grid) == 1:
        return spec.lambda_grid[0]
    seasons = sorted(train_by_season)
    validation = seasons[-inner_folds:] if len(seasons) > inner_folds else seasons[-1:]
    fitting = [s for s in seasons if s < validation[0]]
    if not fitting:
        return spec.lambda_grid[len(spec.lambda_grid) // 2]
    fit_rows = [row for s in fitting for row in train_by_season[s]]
    check_rows = [row for s in validation for row in train_by_season[s]]
    best_lambda, best_error = spec.lambda_grid[0], math.inf
    for candidate in spec.lambda_grid:
        model = fit_ridge(fit_rows, features, spec.target, candidate, spec)
        if model is None:
            continue
        errors = [
            abs(_predict(model, row, spec) - y)
            for row in check_rows
            if (y := _training_target(row, spec)) is not None
        ]
        if not errors:
            continue
        error = sum(errors) / len(errors)
        if error < best_error - 1e-12:
            best_lambda, best_error = candidate, error
    return best_lambda


def _with_played_flag(predictions: pl.DataFrame) -> pl.DataFrame:
    if "actual_games" not in predictions.columns or ACTUAL_PLAYED in predictions.columns:
        return predictions
    return predictions.with_columns(
        pl.when(pl.col("outcome_complete"))
        .then((pl.col("actual_games").fill_null(0) > 0).cast(pl.Float64))
        .otherwise(None)
        .alias(ACTUAL_PLAYED)
    )


def _selector_fold_score(
    rows: list[dict[str, Any]], source: str, target: str, k: int
) -> tuple[float, float] | None:
    """Top-k hit rate and NDCG for one past position-season.

    The ideal ranking uses every observed outcome, including players the source
    cannot score. Missing predictions therefore cannot erase actual winners.
    The selector never sees the season it is choosing for.
    """
    scored = [
        (predicted, actual, index)
        for index, row in enumerate(rows)
        if (predicted := _number(row.get(source))) is not None
        and (actual := _number(row.get(target))) is not None
    ]
    if k <= 0 or len(scored) < k:
        return None
    by_source = sorted(scored, key=lambda item: item[0], reverse=True)
    by_actual = sorted(
        [
            (0.0, actual, index)
            for index, row in enumerate(rows)
            if (actual := _number(row.get(target))) is not None
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    actual_top = {item[2] for item in by_actual[:k]}
    hit_rate = sum(item[2] in actual_top for item in by_source[:k]) / k
    discounts = [1.0 / math.log2(index + 2) for index in range(k)]
    gains = [max(item[1], 0.0) for item in by_source[:k]]
    ideal_gains = [max(item[1], 0.0) for item in by_actual[:k]]
    ideal = sum(gain * discount for gain, discount in zip(ideal_gains, discounts, strict=True))
    ndcg = (
        sum(gain * discount for gain, discount in zip(gains, discounts, strict=True)) / ideal
        if ideal > 0
        else 0.0
    )
    return hit_rate, ndcg


def _choose_adaptive_source(
    completed_by_season: dict[int, list[dict[str, Any]]],
    season: int,
    position: str,
    spec: ModelSpec,
    min_folds: int,
) -> tuple[str | None, float | None, int]:
    """Choose a source from expanding-window ranking results, never future folds."""
    top_k = dict(spec.selection_top_k).get(position)
    if top_k is None:
        return spec.fallback, None, 0
    best_source: str | None = None
    best_key = (-math.inf, -math.inf, -math.inf)
    best_score: float | None = None
    best_folds = 0
    # Compare identical historical folds. A late or sparse source cannot win by
    # skipping difficult seasons that count against its competitors.
    shared_scores: list[dict[str, tuple[float, float]]] = []
    for prior in sorted(completed_by_season):
        if prior >= season:
            continue
        rows = [row for row in completed_by_season[prior] if str(row.get("position")) == position]
        scores = {
            source: score
            for source in spec.factors
            if (score := _selector_fold_score(rows, source, spec.target, top_k)) is not None
        }
        if len(scores) == len(spec.factors):
            shared_scores.append(scores)
    for order, source in enumerate(spec.factors):
        fold_scores = [scores[source] for scores in shared_scores]
        if len(fold_scores) < min_folds:
            continue
        mean_hit = sum(score[0] for score in fold_scores) / len(fold_scores)
        mean_ndcg = sum(score[1] for score in fold_scores) / len(fold_scores)
        primary, secondary = (
            (mean_ndcg, mean_hit) if spec.selection_metric == "ndcg" else (mean_hit, mean_ndcg)
        )
        # Config order is a deterministic conservative tie-break: put the incumbent
        # first and a challenger must actually outperform it on past folds.
        key = (primary, secondary, -float(order))
        if key > best_key:
            best_key = key
            best_source = source
            best_score = primary
            best_folds = len(fold_scores)
    return best_source or spec.fallback, best_score, best_folds


def fit_walk_forward(
    predictions: pl.DataFrame,
    config: FitConfig,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Add every configured fitted output to every forecast row, walk-forward.

    Returns the augmented frame and one record per (model, forecast season, position)
    with the learned standardized coefficients and the λ chosen for that refit.
    """
    predictions = _with_played_flag(predictions)
    records = predictions.to_dicts()
    seasons = sorted({int(row["forecast_season"]) for row in records})
    completed_by_season: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        if row.get("outcome_complete"):
            completed_by_season.setdefault(int(row["forecast_season"]), []).append(row)

    outputs: dict[str, list[float | None]] = {}
    # Later ridge specs may stack earlier walk-forward outputs.  Those inputs are
    # honest out-of-fold predictions for every completed training season: the base
    # model that scored season S was itself fitted only on seasons before S.  Keeping
    # the evolving set explicit prevents a stack from accidentally referring forward
    # to an output that has not been generated yet.
    available_columns = set(predictions.columns)
    models: list[dict[str, Any]] = []
    for spec in config.models:
        logger.info("fitting %s (%s population)", spec.name, spec.population)
        if spec.kind == "adaptive_select":
            if spec.target not in predictions.columns:
                continue
            adaptive_sources = tuple(
                source for source in spec.factors if source in available_columns
            )
            if not adaptive_sources:
                continue
            selector_spec = ModelSpec(**{**asdict(spec), "factors": adaptive_sources})
            min_folds = spec.min_train_folds or config.min_train_folds
            choices: dict[tuple[int, str], str | None] = {}
            for season in seasons:
                for position in config.positions:
                    source, score, folds = _choose_adaptive_source(
                        completed_by_season,
                        season,
                        position,
                        selector_spec,
                        min_folds,
                    )
                    choices[(season, position)] = source
                    models.append(
                        {
                            "model": spec.name,
                            "kind": "adaptive_select",
                            "forecast_season": season,
                            "position": position,
                            "selected_source": source,
                            "selection_target": spec.target,
                            "selection_metric": spec.selection_metric,
                            "selection_score": score,
                            "selection_folds": folds,
                        }
                    )
            adaptive_values: list[float | None] = []
            for row in records:
                key = (int(row["forecast_season"]), str(row.get("position")))
                source = choices.get(key) or spec.fallback
                value = _number(row.get(source)) if source else None
                if value is None and spec.fallback:
                    value = _number(row.get(spec.fallback))
                adaptive_values.append(value)
                row[spec.name] = value
            outputs[spec.name] = adaptive_values
            available_columns.add(spec.name)
            continue
        if spec.kind == "position_select":
            fixed_sources = dict(spec.by_position)
            selector_values: list[float | None] = []
            for row in records:
                source = fixed_sources.get(str(row.get("position")), spec.fallback)
                value = _number(row.get(source)) if source else None
                if value is None and spec.fallback:
                    value = _number(row.get(spec.fallback))
                selector_values.append(value)
                row[spec.name] = value
            outputs[spec.name] = selector_values
            available_columns.add(spec.name)
            continue
        if spec.kind == "coalesce":
            coalesced: list[float | None] = []
            for row in records:
                value = next(
                    (
                        present
                        for source in spec.factors
                        if (present := _number(row.get(source))) is not None
                    ),
                    None,
                )
                coalesced.append(value)
                row[spec.name] = value
            outputs[spec.name] = coalesced
            available_columns.add(spec.name)
            continue
        if spec.kind == "product":
            if not all(factor in available_columns for factor in spec.factors):
                continue
            product_values: list[float | None] = []
            for row in records:
                factors = [_number(row.get(factor)) for factor in spec.factors]
                value = (
                    math.prod(float(factor) for factor in factors if factor is not None)
                    if all(factor is not None for factor in factors)
                    else None
                )
                product_values.append(value)
                row[spec.name] = value
            outputs[spec.name] = product_values
            available_columns.add(spec.name)
            continue
        features = tuple(f for f in spec.features if f in available_columns)
        prefixed = tuple(
            name
            for name in sorted(available_columns)
            if any(name.startswith(prefix) for prefix in spec.feature_prefixes)
            and name not in features
        )
        candidates = (*features, *prefixed)
        if not candidates or spec.target not in predictions.columns:
            continue
        fitted: list[float | None] = [None] * len(records)
        min_folds = spec.min_train_folds or config.min_train_folds
        for season in seasons:
            train_seasons = [
                s
                for s in completed_by_season
                if s < season and any(_has_required(row, spec) for row in completed_by_season[s])
            ]
            if len(train_seasons) < min_folds:
                continue
            train_by_season = {s: completed_by_season[s] for s in train_seasons}
            train = [row for s in train_seasons for row in completed_by_season[s]]
            pooled_features = _select_model_features(train, candidates, spec)
            pooled_lambda = _select_lambda(
                train_by_season, spec, pooled_features, config.inner_validation_folds
            )
            pooled = fit_ridge(train, pooled_features, spec.target, pooled_lambda, spec)
            by_position: dict[str, RidgeModel | None] = {}
            for position in config.positions:
                model = None
                if config.per_position:
                    subset_by_season = {
                        s: [row for row in rows if row.get("position") == position]
                        for s, rows in train_by_season.items()
                    }
                    subset = [row for rows in subset_by_season.values() for row in rows]
                    if len(subset) >= config.min_position_rows:
                        position_features = _select_model_features(subset, candidates, spec)
                        chosen_lambda = _select_lambda(
                            subset_by_season,
                            spec,
                            position_features,
                            config.inner_validation_folds,
                        )
                        model = fit_ridge(
                            subset,
                            position_features,
                            spec.target,
                            chosen_lambda,
                            spec,
                        )
                by_position[position] = model or pooled
                chosen = by_position[position]
                if chosen is not None:
                    models.append(
                        {
                            "model": spec.name,
                            "forecast_season": season,
                            "position": position,
                            "scope": "position" if model is not None else "pooled",
                            "train_seasons": [min(train_seasons), max(train_seasons)],
                            **chosen.record(),
                        }
                    )
            for index, row in enumerate(records):
                if int(row["forecast_season"]) != season:
                    continue
                model = by_position.get(str(row.get("position")), pooled)
                if model is not None and _has_required(row, spec):
                    fitted[index] = _predict(model, row, spec)
        outputs[spec.name] = fitted
        available_columns.add(spec.name)
        for index, row in enumerate(records):
            row[spec.name] = fitted[index]

    frame = predictions
    for name, values in outputs.items():
        frame = frame.with_columns(pl.Series(name, values, dtype=pl.Float64))
    return _apply_products(frame, config), models


def _apply_products(frame: pl.DataFrame, config: FitConfig) -> pl.DataFrame:
    for spec in config.models:
        if spec.kind != "product":
            continue
        if not all(factor in frame.columns for factor in spec.factors):
            continue
        expression = pl.lit(1.0)
        for factor in spec.factors:
            expression = expression * pl.col(factor)
        frame = frame.with_columns(expression.alias(spec.name))
    return frame


def summarize_fitted_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Latest weights per (model, position) plus the spread of each coefficient."""
    summary: list[dict[str, Any]] = []
    keys = sorted({(m.get("model", FITTED_PPG), m["position"]) for m in models})
    for name, position in keys:
        history = sorted(
            (m for m in models if m.get("model", FITTED_PPG) == name and m["position"] == position),
            key=lambda m: m["forecast_season"],
        )
        latest = history[-1]
        if latest.get("kind") == "adaptive_select":
            counts: dict[str, int] = {}
            for record in history:
                source = str(record.get("selected_source") or "fallback")
                counts[source] = counts.get(source, 0) + 1
            summary.append(
                {
                    "model": name,
                    "position": position,
                    "latest_forecast_season": latest["forecast_season"],
                    "kind": "adaptive_select",
                    "selected_source": latest.get("selected_source"),
                    "selection_target": latest.get("selection_target"),
                    "selection_metric": latest.get("selection_metric"),
                    "selection_score": latest.get("selection_score"),
                    "selection_folds": latest.get("selection_folds"),
                    "selected_source_counts": counts,
                    "refits": len(history),
                }
            )
            continue
        stability: dict[str, float] = {}
        for feature in latest["coefficients"]:
            values = [m["coefficients"].get(feature) for m in history]
            values = [v for v in values if v is not None]
            if len(values) > 1:
                mean = sum(values) / len(values)
                stability[feature] = round(
                    math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), 4
                )
        lambdas = sorted(
            float(value) for m in history if (value := _number(m.get("ridge_lambda"))) is not None
        )
        summary.append(
            {
                "model": name,
                "position": position,
                "latest_forecast_season": latest["forecast_season"],
                "scope": latest["scope"],
                "n_train": latest["n_train"],
                "ridge_lambda": latest.get("ridge_lambda"),
                "lambdas_chosen": lambdas,
                "intercept": round(latest["intercept"], 4),
                "coefficients": {k: round(v, 4) for k, v in latest["coefficients"].items()},
                "coefficient_sd_across_refits": stability,
                "refits": len(history),
            }
        )
    return summary


def models_for_season(
    models: list[dict[str, Any]], forecast_season: int
) -> dict[str, dict[str, RidgeModel]]:
    """Per-model, per-position models fitted for one forecast season."""
    result: dict[str, dict[str, RidgeModel]] = {}
    for record in models:
        if int(record["forecast_season"]) != forecast_season or not record.get("standardization"):
            continue
        result.setdefault(record.get("model", FITTED_PPG), {})[record["position"]] = (
            RidgeModel.from_record(record)
        )
    return result


def apply_fitted_models(
    board: pl.DataFrame,
    models: dict[str, dict[str, RidgeModel]],
    config: FitConfig,
) -> pl.DataFrame:
    """Score a live board with previously fitted models, ridge outputs then products.

    Rows whose position has no model receive nulls, so a missing fit degrades to the
    configured fallback sort key rather than a silent zero.
    """
    rows = board.to_dicts()
    frame = board
    for spec in config.models:
        if spec.kind == "adaptive_select":
            if spec.apply_live:
                raise ValueError("adaptive_select models must remain report-only")
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(spec.name))
            continue
        if spec.kind == "position_select":
            if not spec.apply_live:
                position_values: list[float | None] = [None] * len(rows)
            else:
                sources = dict(spec.by_position)
                position_values = []
                for row in rows:
                    source = sources.get(str(row.get("position")), spec.fallback)
                    value = _number(row.get(source)) if source else None
                    if value is None and spec.fallback:
                        value = _number(row.get(spec.fallback))
                    position_values.append(value)
            frame = frame.with_columns(pl.Series(spec.name, position_values, dtype=pl.Float64))
            for row, value in zip(rows, position_values, strict=True):
                row[spec.name] = value
            continue
        if spec.kind == "coalesce":
            values: list[float | None]
            if not spec.apply_live:
                values = [None] * len(rows)
            else:
                values = [
                    next(
                        (
                            present
                            for source in spec.factors
                            if (present := _number(row.get(source))) is not None
                        ),
                        None,
                    )
                    for row in rows
                ]
            frame = frame.with_columns(pl.Series(spec.name, values, dtype=pl.Float64))
            for row, value in zip(rows, values, strict=True):
                row[spec.name] = value
            continue
        if spec.kind == "product":
            if spec.apply_live and all(factor in frame.columns for factor in spec.factors):
                values = []
                for row in rows:
                    factors = [_number(row.get(factor)) for factor in spec.factors]
                    value = (
                        math.prod(float(factor) for factor in factors if factor is not None)
                        if all(factor is not None for factor in factors)
                        else None
                    )
                    values.append(value)
                    row[spec.name] = value
                frame = frame.with_columns(pl.Series(spec.name, values, dtype=pl.Float64))
            continue
        if not spec.apply_live:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(spec.name))
            continue
        by_position = models.get(spec.name, {})
        model_values: list[float | None] = [
            (
                _predict(model, row, spec)
                if (model := by_position.get(str(row.get("position")))) and _has_required(row, spec)
                else None
            )
            for row in rows
        ]
        frame = frame.with_columns(pl.Series(spec.name, model_values, dtype=pl.Float64))
        for row, value in zip(rows, model_values, strict=True):
            row[spec.name] = value
    return _apply_products(frame, config)


PRODUCTION_OUTPUTS = ("fitted_ppg", "fitted_games", "fitted_season_points")


def production_config(config: FitConfig) -> FitConfig:
    """Only explicitly approved outputs and their transitive dependencies."""
    specs = {spec.name: spec for spec in config.models}
    needed: set[str] = set()

    def include(name: str) -> None:
        if name in needed or name not in specs:
            return
        needed.add(name)
        spec = specs[name]
        for dependency in (*spec.factors, *(v for _, v in spec.by_position)):
            include(dependency)
        if spec.fallback:
            include(spec.fallback)

    for name in PRODUCTION_OUTPUTS:
        include(name)
    return replace(config, models=tuple(s for s in config.models if s.name in needed))


def fit_fingerprint(config: FitConfig) -> str:
    """Stable digest of everything that changes what a fitted model means."""
    payload = {
        "model_version": MODEL_VERSION,
        "models": [asdict(spec) for spec in config.models],
        "min_train_folds": config.min_train_folds,
        "min_position_rows": config.min_position_rows,
        "per_position": config.per_position,
        "positions": list(config.positions),
        "inner_validation_folds": config.inner_validation_folds,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def production_definitions_digest() -> str:
    """Scoring and feature construction used to train/apply the production model."""
    root = Path(__file__).parents[1]
    paths = [
        root / "config/scoring.yaml",
        root / "config/projections.yaml",
        root / "metrics/projection.py",
        root / "metrics/enrichment.py",
        root / "metrics/availability.py",
        *sorted((root / "scoring").glob("*.py")),
    ]
    payload = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_fit_artifact(
    config: FitConfig,
    predictions: pl.DataFrame,
    depth_chart_cutoff: str,
) -> dict[str, Any]:
    """Provenance the live board checks before applying a fitted model.

    A same-season artifact fitted with different specs or a different depth-chart
    cutoff or snapshot must be rejected, not silently applied.
    """
    pending = predictions.filter(~pl.col("outcome_complete"))
    pending_value = _number(pending["forecast_season"].max()) if pending.height else None
    pending_season = int(pending_value) if pending_value is not None else None
    depth_latest = None
    if pending.height and "depth_chart_date" in pending.columns:
        depth_latest = pending["depth_chart_date"].cast(pl.String).max()
    completed = predictions.filter(pl.col("outcome_complete"))["forecast_season"]
    completed_min = _number(completed.min()) if completed.len() else None
    completed_max = _number(completed.max()) if completed.len() else None
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "production_definitions_sha256": production_definitions_digest(),
        "production_training_population": "returner",
        "production_training_sha256": hashlib.sha256(
            predictions.filter(pl.col("returning_indicator") == 1)
            .select(
                sorted(
                    set(predictions.columns)
                    & {
                        "forecast_season",
                        "player_id",
                        "position",
                        "outcome_complete",
                        "actual_ppg",
                        "actual_games",
                        "returning_indicator",
                        *(
                            feature
                            for spec in production_config(config).models
                            for feature in spec.features
                        ),
                    }
                )
            )
            .sort(["forecast_season", "player_id"])
            .write_json()
            .encode()
        ).hexdigest()
        if "returning_indicator" in predictions.columns
        else None,
        "model_version": MODEL_VERSION,
        "fingerprint": fit_fingerprint(config),
        "production_fingerprint": fit_fingerprint(production_config(config)),
        "fit_config": {
            "models": [asdict(spec) for spec in config.models],
            "min_train_folds": config.min_train_folds,
            "min_position_rows": config.min_position_rows,
            "per_position": config.per_position,
            "positions": list(config.positions),
            "inner_validation_folds": config.inner_validation_folds,
        },
        "outputs": list(config.output_columns),
        "created_at": datetime.now(UTC).isoformat(),
        "pending_forecast_season": pending_season,
        "pending_rows": pending.height,
        "completed_forecast_seasons": (
            [int(completed_min), int(completed_max)]
            if completed_min is not None and completed_max is not None
            else []
        ),
        "depth_chart_cutoff": depth_chart_cutoff,
        "depth_chart_latest": depth_latest,
    }


class FittedArtifactError(ValueError):
    """The stored fitted models cannot be applied to this board."""


def check_fit_artifact(
    artifact: dict[str, Any] | None,
    config: FitConfig,
    forecast_season: int,
    depth_chart_cutoff: str,
    live_depth_latest: str | None,
) -> None:
    """Raise unless the artifact was fitted for this season, config, and inputs."""
    if not artifact:
        raise FittedArtifactError("metric report carries no fitted-model artifact")
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise FittedArtifactError(
            f"artifact schema {artifact.get('schema_version')} != {ARTIFACT_SCHEMA_VERSION}"
        )
    if artifact.get("production_fingerprint") != fit_fingerprint(production_config(config)):
        raise FittedArtifactError(
            "fitted-model fingerprint does not match the current fit configuration; "
            "run `engine metric-report`"
        )
    if artifact.get("production_definitions_sha256") != production_definitions_digest():
        raise FittedArtifactError("scoring or feature definitions changed; refit production models")
    if artifact.get("pending_forecast_season") != forecast_season:
        raise FittedArtifactError(
            f"artifact was fitted for {artifact.get('pending_forecast_season')}, "
            f"board is for {forecast_season}"
        )
    if artifact.get("depth_chart_cutoff") != depth_chart_cutoff:
        raise FittedArtifactError(
            f"artifact depth-chart cutoff {artifact.get('depth_chart_cutoff')} != "
            f"live cutoff {depth_chart_cutoff}"
        )
    stored = artifact.get("depth_chart_latest")
    if stored and live_depth_latest and str(stored)[:10] != str(live_depth_latest)[:10]:
        raise FittedArtifactError(
            f"live depth chart snapshot {str(live_depth_latest)[:10]} differs from the "
            f"snapshot the pending fold was scored on ({str(stored)[:10]}); refit"
        )
