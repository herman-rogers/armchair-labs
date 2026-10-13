"""The nflverse player-stat columns the league scorer depends on, in one place.

The scoring rules validate their input against these lists and the test factories
build complete synthetic frames from them. The DuckDB recipes under
``engine/data/enrich/sql`` name the same fields, so a provider rename fails loudly
at build time instead of scoring quietly wrong.
"""

from __future__ import annotations

from typing import Final

PASSING_COLUMNS: Final[tuple[str, ...]] = (
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_2pt_conversions",
)

RUSHING_COLUMNS: Final[tuple[str, ...]] = (
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_2pt_conversions",
    "rushing_fumbles_lost",
)

RECEIVING_COLUMNS: Final[tuple[str, ...]] = (
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_2pt_conversions",
    "receiving_fumbles_lost",
)

FUMBLE_COLUMNS: Final[tuple[str, ...]] = ("sack_fumbles_lost",)

SPECIAL_TEAMS_COLUMNS: Final[tuple[str, ...]] = ("special_teams_tds",)

OPPORTUNITY_COLUMNS: Final[tuple[str, ...]] = ("target_share", "air_yards_share")

NFLVERSE_POINTS_COLUMN: Final[str] = "fantasy_points_ppr"

SCORING_COLUMNS: Final[tuple[str, ...]] = (
    PASSING_COLUMNS + RUSHING_COLUMNS + RECEIVING_COLUMNS + FUMBLE_COLUMNS + SPECIAL_TEAMS_COLUMNS
)

PBP_TOUCHDOWN_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "season_type",
    "touchdown",
    "yards_gained",
    "pass_touchdown",
    "rush_touchdown",
    "passer_player_id",
    "receiver_player_id",
    "rusher_player_id",
)


class MissingColumnsError(ValueError):
    """A required nflverse column is absent; named so a provider rename fails loudly."""

    def __init__(self, missing: list[str], context: str) -> None:
        self.missing = missing
        self.context = context
        super().__init__(
            f"{context} is missing required column(s): {', '.join(sorted(missing))}. "
            "If nflverse renamed them, update engine/scoring/columns.py."
        )


def require_columns(available: list[str], required: tuple[str, ...], context: str) -> None:
    missing = [name for name in required if name not in available]
    if missing:
        raise MissingColumnsError(missing, context)
