"""The nflverse columns this engine depends on, in one place.

Three consumers share these lists, which is the point of centralising them:

* the scoring and metrics functions, which validate their input against them;
* `tests/factories.py`, which builds complete synthetic frames from them;
* `tests/test_schema.py`, which asserts live nflverse still provides them.

That last one is the drift tripwire. The unofficial-API and community-dataset risks
in the plan's §8 both show up as a renamed or dropped column, and a renamed column
that silently reads as zero is the worst possible failure: the board still builds,
the numbers are just quietly wrong. Better to fail loudly at load time.
"""

from __future__ import annotations

from typing import Final

# Identity and context on every player-week row.
IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "player_id",
    "player_display_name",
    "position",
    "season",
    "week",
    "season_type",
    "team",
)

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

# A sack fumble is a lost fumble like any other and the league scores it -2.
# nflverse tracks it separately from rushing and receiving fumbles.
FUMBLE_COLUMNS: Final[tuple[str, ...]] = ("sack_fumbles_lost",)

# Kick and punt return touchdowns. Easy to overlook — the design doc's summary of the
# league sheet omits them entirely — and worth six points to a returner who happens to
# be roster-relevant. tests/test_parity.py is what surfaced the omission.
SPECIAL_TEAMS_COLUMNS: Final[tuple[str, ...]] = ("special_teams_tds",)

OPPORTUNITY_COLUMNS: Final[tuple[str, ...]] = (
    "target_share",
    "air_yards_share",
    "wopr",
)

# nflverse's own PPR total. Used as the fast path for base scoring and, in
# tests/test_parity.py, as the independent check on our component scorer.
NFLVERSE_POINTS_COLUMN: Final[str] = "fantasy_points_ppr"

KICKING_COLUMNS: Final[tuple[str, ...]] = (
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    "fg_missed",
    "fg_blocked",
    "pat_made",
)

DEFENSE_COLUMNS: Final[tuple[str, ...]] = (
    "def_sacks",
    "def_interceptions",
    "def_tds",
    "def_safeties",
    "fumble_recovery_opp",
)

#: Every column the component scorer reads.
SCORING_COLUMNS: Final[tuple[str, ...]] = (
    PASSING_COLUMNS + RUSHING_COLUMNS + RECEIVING_COLUMNS + FUMBLE_COLUMNS + SPECIAL_TEAMS_COLUMNS
)

#: Every column the skill-position pipeline reads, end to end.
SKILL_PLAYER_COLUMNS: Final[tuple[str, ...]] = (
    IDENTITY_COLUMNS + SCORING_COLUMNS + OPPORTUNITY_COLUMNS + (NFLVERSE_POINTS_COLUMN,)
)

# Play-by-play columns needed to reduce touchdowns to per-player bonus points.
# load_pbp returns ~370 columns; these are the only ones we keep.
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

# Projection-only nflverse contracts. These feeds enrich v2 without changing v1's
# historical scoring semantics.
TEAM_VOLUME_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "season_type",
    "team",
    "attempts",
    "sacks_suffered",
    "carries",
)

PBP_USAGE_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "play_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "pass_attempt",
    "rush_attempt",
    "qb_dropback",
    "qb_scramble",
    "qb_kneel",
    "yardline_100",
    "air_yards",
    "pass_touchdown",
    "rush_touchdown",
    "passer_player_id",
    "receiver_player_id",
    "rusher_player_id",
    "two_point_attempt",
)

PARTICIPATION_COLUMNS: Final[tuple[str, ...]] = (
    "nflverse_game_id",
    "play_id",
    "possession_team",
    "offense_players",
    "offense_positions",
)

DEPTH_CHART_COLUMNS: Final[tuple[str, ...]] = (
    "dt",
    "team",
    "gsis_id",
    "pos_abb",
    "pos_name",
    "pos_rank",
)

INJURY_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "team",
    "gsis_id",
    "position",
    "report_primary_injury",
    "report_status",
    "practice_primary_injury",
    "practice_status",
)

ROSTER_COLUMNS: Final[tuple[str, ...]] = ("gsis_id", "birth_date")

SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "game_type",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
)


class MissingColumnsError(ValueError):
    """Raised when a frame is missing columns the engine needs.

    Names the columns rather than failing on first use, so a nflverse rename shows up
    as one actionable message instead of a chain of null-propagation bugs.
    """

    def __init__(self, missing: list[str], context: str) -> None:
        self.missing = missing
        self.context = context
        super().__init__(
            f"{context} is missing required column(s): {', '.join(sorted(missing))}. "
            "If nflverse renamed them, update patron/scoring/columns.py."
        )


def require_columns(available: list[str], required: tuple[str, ...], context: str) -> None:
    """Assert every column in `required` is present, or raise `MissingColumnsError`."""
    missing = [column for column in required if column not in available]
    if missing:
        raise MissingColumnsError(missing, context)
