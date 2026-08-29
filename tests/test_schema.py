"""Live-schema contract tests — the drift tripwire.

nflverse is a community dataset and the columns it publishes are not a stable API.
A renamed column is the failure this project is most exposed to, and it is dangerous
precisely because it is quiet: read as null, coerced to zero, the board still builds
and the numbers are simply wrong.

These tests assert that everything `patron.scoring.columns` promises actually exists
upstream. They hit the network, so they are excluded from the default suite; run them
with `just test-all`, and run them before trusting a board built after a long gap.

A failure here means updating `patron/scoring/columns.py` to match the new names — and
checking whether the semantics changed too, not just the spelling.
"""

from __future__ import annotations

import polars as pl
import pytest

from patron.data import nflverse
from patron.scoring.columns import (
    DEFENSE_COLUMNS,
    KICKING_COLUMNS,
    NFLVERSE_POINTS_COLUMN,
    PBP_TOUCHDOWN_COLUMNS,
    SCHEDULE_COLUMNS,
    SKILL_PLAYER_COLUMNS,
)

pytestmark = pytest.mark.network

SEASON = 2025


@pytest.fixture(scope="module")
def player_weeks() -> pl.DataFrame:
    return nflverse.load_player_weeks([SEASON])


class TestPlayerStatsSchema:
    def test_every_skill_column_is_present(self, player_weeks: pl.DataFrame) -> None:
        missing = [c for c in SKILL_PLAYER_COLUMNS if c not in player_weeks.columns]
        assert not missing, f"nflverse no longer publishes: {missing}"

    def test_every_kicking_column_is_present(self, player_weeks: pl.DataFrame) -> None:
        missing = [c for c in KICKING_COLUMNS if c not in player_weeks.columns]
        assert not missing, f"nflverse no longer publishes: {missing}"

    def test_every_defense_column_is_present(self, player_weeks: pl.DataFrame) -> None:
        missing = [c for c in DEFENSE_COLUMNS if c not in player_weeks.columns]
        assert not missing, f"nflverse no longer publishes: {missing}"

    def test_the_precomputed_ppr_column_still_exists(self, player_weeks: pl.DataFrame) -> None:
        """The fast path depends on this column entirely. If it disappears, the
        component scorer has to take over as the primary path."""
        assert NFLVERSE_POINTS_COLUMN in player_weeks.columns

    def test_the_season_filter_actually_filters(self, player_weeks: pl.DataFrame) -> None:
        assert player_weeks["season_type"].unique().to_list() == ["REG"]

    def test_the_board_positions_are_populated(self, player_weeks: pl.DataFrame) -> None:
        positions = set(player_weeks["position"].unique().to_list())
        assert {"QB", "RB", "WR", "TE", "K"} <= positions

    def test_the_data_is_not_suspiciously_thin(self, player_weeks: pl.DataFrame) -> None:
        """A season of weekly rows should be in the tens of thousands. A tiny frame
        means a partial download that would silently skew every metric."""
        assert player_weeks.height > 10_000


class TestPlayByPlaySchema:
    def test_touchdown_columns_are_present(self) -> None:
        plays = nflverse.load_touchdown_plays([SEASON])
        missing = [c for c in PBP_TOUCHDOWN_COLUMNS if c not in plays.columns]
        assert not missing, f"nflverse pbp no longer publishes: {missing}"

    def test_yards_gained_is_populated_on_scoring_plays(self) -> None:
        plays = nflverse.load_touchdown_plays([SEASON])
        assert plays["yards_gained"].null_count() == 0, (
            "null yardage would silently zero out every big-play bonus"
        )

    def test_a_season_has_a_plausible_touchdown_count(self) -> None:
        plays = nflverse.load_touchdown_plays([SEASON])
        assert 1_000 < plays.height < 2_500


class TestSupportingSchemas:
    def test_roster_columns_are_present(self) -> None:
        rosters = nflverse.load_birth_dates(SEASON)
        assert set(rosters.columns) == {"player_id", "birth_date"}

    def test_birth_dates_are_mostly_populated(self) -> None:
        rosters = nflverse.load_birth_dates(SEASON)
        populated = 1 - rosters["birth_date"].null_count() / rosters.height
        assert populated > 0.90, f"only {populated:.1%} of players have a birth date"

    def test_one_row_per_player(self) -> None:
        """Rosters carry a row per player per team; a mid-season trade would otherwise
        fan the board out on join."""
        rosters = nflverse.load_birth_dates(SEASON)
        assert rosters["player_id"].n_unique() == rosters.height

    def test_schedule_columns_are_present(self) -> None:
        schedules = nflverse.load_schedules(SEASON)
        missing = [c for c in SCHEDULE_COLUMNS if c not in schedules.columns]
        assert not missing, f"nflverse schedules no longer publishes: {missing}"


class TestIdCrosswalk:
    """The ffverse crosswalk is Phase 2's answer to the name-joining tax. Verified now
    so the next phase does not discover a surprise."""

    def test_it_maps_nflverse_ids_to_espn_ids(self) -> None:
        crosswalk = nflverse.load_id_crosswalk()

        assert "gsis_id" in crosswalk.columns
        assert "espn_id" in crosswalk.columns

    def test_it_covers_a_useful_share_of_players(self) -> None:
        crosswalk = nflverse.load_id_crosswalk()
        both = crosswalk.filter(pl.col("gsis_id").is_not_null() & pl.col("espn_id").is_not_null())
        assert both.height > 1_000, (
            "too few gsis-to-espn mappings for the crosswalk to be the primary join"
        )
