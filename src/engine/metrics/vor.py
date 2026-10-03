"""Value over replacement.

The number the whole board sorts by, and the reason the board disagrees with consensus
in the places that matter. A tight end scoring 18.6 per game is worth more than a
running back scoring 20.6, because the best freely-available tight end scores 10.6 and
the best freely-available back scores 12.3. Positional scarcity is the whole game, and
raw PPG cannot see it.

Replacement level is the per-game production of the last startable player at each
position — QB12, RB25, WR35, TE12 in this 10-team league. Those ranks are config, not
constants: change the league size and they must be re-derived.

The eligibility gate matters more than it looks. Replacement level answers "what can I
get for free", and a player who was excellent across three games before getting hurt is
not that. Letting a small sample set the baseline inflates it, which deflates every
VOR at the position at once. Hence `min_games`, applied to the baseline pool only —
the small-sample players still appear on the board, flagged; they just do not get to
define the bar.
"""

from __future__ import annotations

import polars as pl

from engine.scoring.columns import require_columns

REPLACEMENT_PPG = "repl_ppg"
VOR = "vor"


class ReplacementLevelError(ValueError):
    """Raised when a position has no eligible players to set a baseline from."""


def replacement_levels(
    seasons: pl.DataFrame,
    baseline_ranks: dict[str, int],
    min_games: int = 8,
    ppg_column: str = "ppg",
    games_column: str = "games",
) -> dict[str, float]:
    """Replacement-level PPG for each position in `baseline_ranks`.

    Args:
        seasons: Player-season rows for a single season.
        baseline_ranks: Positional rank of the replacement player, e.g. `{"RB": 25}`.
        min_games: Games required to be eligible to set the baseline.
        ppg_column: Per-game points column.
        games_column: Games-played column.

    Returns:
        Position to replacement PPG.

    Raises:
        ReplacementLevelError: if a position has no eligible players at all.
    """
    require_columns(seasons.columns, (ppg_column, games_column, "position"), "replacement_levels")

    levels: dict[str, float] = {}
    for position, rank in baseline_ranks.items():
        pool = (
            seasons.filter((pl.col("position") == position) & (pl.col(games_column) >= min_games))
            .sort(ppg_column, descending=True)
            .get_column(ppg_column)
        )
        if pool.len() == 0:
            raise ReplacementLevelError(
                f"no {position} met the {min_games}-game minimum, so replacement level "
                "cannot be set. Check the season filter and the games threshold."
            )
        # A pool shorter than the baseline rank means the position is thinner than the
        # league is deep; the worst eligible player is then the honest replacement.
        index = min(rank, pool.len()) - 1
        levels[position] = float(pool[index])

    return levels


def add_vor(
    seasons: pl.DataFrame,
    levels: dict[str, float],
    ppg_column: str = "ppg",
) -> pl.DataFrame:
    """Add `repl_ppg` and `vor` given precomputed replacement levels.

    Players at a position with no configured baseline (kickers and defenses, which this
    league tiers rather than ranks) get a null VOR and drop out of the sort.
    """
    require_columns(seasons.columns, (ppg_column, "position"), "add_vor input")

    return seasons.with_columns(
        pl.col("position")
        .replace_strict(levels, default=None, return_dtype=pl.Float64)
        .alias(REPLACEMENT_PPG)
    ).with_columns((pl.col(ppg_column) - pl.col(REPLACEMENT_PPG)).alias(VOR))
