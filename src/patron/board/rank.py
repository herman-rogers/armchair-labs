"""Per-position v2 sort key, chosen by the metric report's evidence.

The ranking table in the metric report answers, for each position, which output puts
the most eventual top-K finishers at the top of the board.  That answer is not the
same everywhere — the fitted ranker wins at QB/WR/TE while the bottom-up projection
wins at RB — so the live board sorts each position by its own configured key.

Each key is expressed as per-game points above the positional replacement level *of
that same key*: the key value is converted to a per-game equivalent (season-point keys
are divided by the season length), the replacement is the baseline-rank value of that
equivalent within the position, and ``v2_rank_vor`` is the difference plus any manual
override.  Keys that are already a VOR (``adj_proj_vor``, ``proj_vor``)
are used as-is.  ``v2_rank_vor`` orders players *within* a position
(``v2_position_rank``).

Different keys have different spreads — the bottom-up projection is wider than the
shrunk fitted ranker — so VORs from different keys must not be interleaved across
positions.  The overall board order therefore uses one scale for everyone:
``v2_overall_vor`` is the configured ``projection_overall_key`` (the fitted
season-points projection, chosen by the metric report's OVERALL test) as a per-game
equivalent above its own positional replacement, plus override, falling back to
``v2_rank_vor`` where the key is unavailable.
"""

from __future__ import annotations

import polars as pl

from patron.config.league import LeagueConfig
from patron.metrics.vor import replacement_levels

RANK_KEY = "v2_rank_key"
RANK_VALUE = "v2_rank_value"
RANK_VOR = "v2_rank_vor"
POSITION_RANK = "v2_position_rank"
OVERALL_VOR = "v2_overall_vor"

_ALREADY_VOR = frozenset({"adj_proj_vor", "proj_vor"})


def resolve_rank_keys(board: pl.DataFrame, config: LeagueConfig) -> dict[str, str]:
    """Configured key per position, falling back where the column is absent or empty."""
    metrics = config.metrics
    resolved: dict[str, str] = {}
    for position in config.vor_baseline_rank:
        key = metrics.projection_rank_key.get(position, metrics.projection_rank_fallback)
        usable = (
            key in board.columns
            and board.filter(pl.col("position") == position)[key].is_not_null().any()
        )
        resolved[position] = key if usable else metrics.projection_rank_fallback
    return resolved


def apply_rank_key(board: pl.DataFrame, config: LeagueConfig) -> pl.DataFrame:
    """Add the per-position key, its per-game equivalent, and the VOR the board sorts by."""
    metrics = config.metrics
    keys = resolve_rank_keys(board, config)
    season_games = float(metrics.projection_season_games)
    games_column = "ppg_denominator_games" if "ppg_denominator_games" in board.columns else "games"

    key_expr = pl.col("position").replace_strict(keys, default=metrics.projection_rank_fallback)
    value_expr = pl.lit(None, dtype=pl.Float64)
    for position, key in keys.items():
        raw = pl.col(key).cast(pl.Float64)
        equivalent = raw / season_games if key.endswith("season_points") else raw
        value_expr = pl.when(pl.col("position") == position).then(equivalent).otherwise(value_expr)
    frame = board.with_columns(key_expr.alias(RANK_KEY), value_expr.alias(RANK_VALUE))

    vor_positions = [position for position, key in keys.items() if key not in _ALREADY_VOR]
    replacement: dict[str, float] = {}
    if vor_positions:
        eligible = frame.filter(
            pl.col("position").is_in(vor_positions) & pl.col(RANK_VALUE).is_not_null()
        )
        replacement = replacement_levels(
            eligible,
            baseline_ranks={p: config.vor_baseline_rank[p] for p in vor_positions},
            min_games=config.min_games_baseline,
            ppg_column=RANK_VALUE,
            games_column=games_column,
        )
    override = (
        pl.col("override_delta").fill_null(0.0)
        if "override_delta" in frame.columns
        else pl.lit(0.0)
    )
    repl_expr = pl.col("position").replace_strict(
        replacement, default=None, return_dtype=pl.Float64
    )
    rank_vor = (
        pl.when(pl.col(RANK_KEY).is_in(list(_ALREADY_VOR)))
        .then(pl.col(RANK_VALUE))
        .otherwise(pl.col(RANK_VALUE) - repl_expr + override)
    )
    frame = frame.with_columns(rank_vor.alias(RANK_VOR))
    frame = frame.with_columns(
        pl.col(RANK_VOR)
        .rank(method="ordinal", descending=True)
        .over("position")
        .cast(pl.Int32)
        .alias(POSITION_RANK)
    )

    # Overall order on one scale. Keys differ in spread (the bottom-up projection is
    # wider than the shrunk fitted ranker), so their VORs must not be interleaved.
    overall = pl.col(RANK_VOR)
    overall_key = metrics.projection_overall_key
    if overall_key in frame.columns and frame[overall_key].is_not_null().any():
        raw = pl.col(overall_key).cast(pl.Float64)
        equivalent = raw / season_games if overall_key.endswith("season_points") else raw
        frame = frame.with_columns(equivalent.alias("_overall_equivalent"))
        overall_positions = [
            position
            for position in config.vor_baseline_rank
            if frame.filter(pl.col("position") == position)["_overall_equivalent"]
            .is_not_null()
            .any()
        ]
        overall_levels = replacement_levels(
            frame.filter(
                pl.col("position").is_in(overall_positions)
                & pl.col("_overall_equivalent").is_not_null()
            ),
            baseline_ranks={p: config.vor_baseline_rank[p] for p in overall_positions},
            min_games=config.min_games_baseline,
            ppg_column="_overall_equivalent",
            games_column=games_column,
        )
        overall_repl = pl.col("position").replace_strict(
            overall_levels, default=None, return_dtype=pl.Float64
        )
        overall = (
            pl.when(pl.col("_overall_equivalent").is_not_null() & overall_repl.is_not_null())
            .then(pl.col("_overall_equivalent") - overall_repl + override)
            .otherwise(pl.col(RANK_VOR))
        )
        if overall_key in _ALREADY_VOR:
            overall = pl.col(overall_key).cast(pl.Float64).fill_null(pl.col(RANK_VOR))
    frame = frame.with_columns(overall.alias(OVERALL_VOR)).sort(
        OVERALL_VOR, descending=True, nulls_last=True
    )
    if "_overall_equivalent" in frame.columns:
        frame = frame.drop("_overall_equivalent")
    return frame.with_columns(pl.int_range(1, frame.height + 1, dtype=pl.Int32).alias("rank"))
