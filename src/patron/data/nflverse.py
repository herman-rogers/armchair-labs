"""nflverse loading — the only module in the package that touches the network.

Everything downstream is pure DataFrame transformation, which is what lets the test
suite run offline. Keeping the boundary here, and narrow, is the point.

nflreadpy ships its own filesystem cache, so this module configures it rather than
reimplementing one. The cache is pointed at the project's data directory so that all
generated state lives in one place and can be cleared with `patron refresh --force`.

Both nflverse and the ESPN API are free community resources and the correct posture
is a quiet houseguest: cache aggressively, refresh on a slow clock, and never poll
faster than the data actually changes.
"""

from __future__ import annotations

import logging

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import update_config

from patron.config.settings import Settings, get_settings
from patron.scoring.columns import (
    DEPTH_CHART_COLUMNS,
    INJURY_COLUMNS,
    PARTICIPATION_COLUMNS,
    PBP_TOUCHDOWN_COLUMNS,
    PBP_USAGE_COLUMNS,
    ROSTER_COLUMNS,
    SCHEDULE_COLUMNS,
    SKILL_PLAYER_COLUMNS,
    TEAM_VOLUME_COLUMNS,
    require_columns,
)

logger = logging.getLogger(__name__)

_configured = False


def configure_cache(settings: Settings | None = None) -> None:
    """Point nflreadpy's cache at the project data directory. Idempotent."""
    global _configured
    if _configured:
        return

    settings = settings or get_settings()
    settings.ensure_dirs()
    update_config(
        cache_mode="filesystem",
        cache_dir=settings.nflverse_cache_dir,
        cache_duration=settings.nflverse_cache_duration,
    )
    _configured = True


def load_player_weeks(seasons: list[int], season_type: str = "REG") -> pl.DataFrame:
    """Weekly player stat lines for `seasons`, filtered to one season type.

    The workhorse table: passing, rushing and receiving components, opportunity shares,
    and nflverse's own PPR total. Validated against the column contract on the way out,
    so a nflverse rename fails here with a clear message rather than propagating nulls
    into the board.
    """
    configure_cache()
    frame = nfl.load_player_stats(seasons, summary_level="week")
    require_columns(frame.columns, SKILL_PLAYER_COLUMNS, "nflverse player stats")

    filtered = frame.filter(pl.col("season_type") == season_type)
    logger.info("loaded %s player-weeks for seasons %s (%s)", filtered.height, seasons, season_type)
    return filtered


def load_touchdown_plays(seasons: list[int]) -> pl.DataFrame:
    """Scoring plays only, with just the columns needed for the distance bonuses.

    `load_pbp` returns roughly 370 columns back to 1999. Filtering to touchdowns and
    projecting to ten columns before anything else touches it keeps a multi-season pull
    from dominating both memory and runtime.
    """
    configure_cache()

    frames: list[pl.DataFrame] = []
    for season in seasons:
        pbp = nfl.load_pbp([season])
        require_columns(pbp.columns, PBP_TOUCHDOWN_COLUMNS, f"nflverse pbp {season}")
        frames.append(pbp.filter(pl.col("touchdown") == 1).select(PBP_TOUCHDOWN_COLUMNS))
        logger.info("loaded %s touchdown plays for %s", frames[-1].height, season)

    return pl.concat(frames, how="vertical_relaxed")


def load_projection_plays(seasons: list[int]) -> pl.DataFrame:
    """Small play-by-play slice used for dropbacks and high-value opportunities."""
    configure_cache()
    frames: list[pl.DataFrame] = []
    for season in seasons:
        pbp = nfl.load_pbp([season])
        require_columns(pbp.columns, PBP_USAGE_COLUMNS, f"nflverse projection pbp {season}")
        frames.append(pbp.select(PBP_USAGE_COLUMNS))
    return pl.concat(frames, how="vertical_relaxed")


def load_team_weeks(seasons: list[int], season_type: str = "REG") -> pl.DataFrame:
    """Official nflverse team attempts and carries by week."""
    configure_cache()
    frame = nfl.load_team_stats(seasons, summary_level="week")
    require_columns(frame.columns, TEAM_VOLUME_COLUMNS, "nflverse team stats")
    return frame.filter(pl.col("season_type") == season_type).select(TEAM_VOLUME_COLUMNS)


def load_participation(seasons: list[int]) -> pl.DataFrame:
    """Play-level offensive participation used to derive route opportunities."""
    configure_cache()
    frame = nfl.load_participation(seasons)
    require_columns(frame.columns, PARTICIPATION_COLUMNS, "nflverse participation")
    return frame.select(PARTICIPATION_COLUMNS)


def load_depth_charts(seasons: list[int]) -> pl.DataFrame:
    """Published nflverse depth-chart snapshots, including GSIS identifiers."""
    configure_cache()
    frame = nfl.load_depth_charts(seasons)
    require_columns(frame.columns, DEPTH_CHART_COLUMNS, "nflverse depth charts")
    return frame.select(DEPTH_CHART_COLUMNS)


def load_injuries(seasons: list[int]) -> pl.DataFrame:
    """Official injury-report history from nflverse."""
    configure_cache()
    frame = nfl.load_injuries(seasons)
    require_columns(frame.columns, INJURY_COLUMNS, "nflverse injuries")
    return frame.select(INJURY_COLUMNS)


def load_birth_dates(season: int) -> pl.DataFrame:
    """One birth date per player, for the age curve.

    Rosters carry a row per player per team, so a mid-season trade yields duplicates;
    they are collapsed here rather than silently fanning out the board on join.
    """
    configure_cache()
    rosters = nfl.load_rosters([season])
    require_columns(rosters.columns, ROSTER_COLUMNS, "nflverse rosters")

    return (
        rosters.select(ROSTER_COLUMNS)
        .drop_nulls("gsis_id")
        .unique(subset=["gsis_id"], keep="first")
        .rename({"gsis_id": "player_id"})
    )


def load_schedules(season: int) -> pl.DataFrame:
    """Game results, for points allowed and (later) bye weeks and matchups."""
    configure_cache()
    schedules = nfl.load_schedules([season])
    require_columns(schedules.columns, SCHEDULE_COLUMNS, "nflverse schedules")
    return schedules


def load_id_crosswalk() -> pl.DataFrame:
    """The ffverse player-ID crosswalk: nflverse `gsis_id` to ESPN and other IDs.

    Not used in Phase 1, but it is the answer to the plan's §8 warning that name
    joining is the tax. ESPN uses its own player IDs and display names — "D.J. Moore"
    against "DJ Moore", suffixes, nicknames — and matching on those is how a ranked
    wire quietly omits the one player that mattered. With this table the join is on an
    ID, and name matching becomes a logged fallback rather than the primary path.
    """
    configure_cache()
    return nfl.load_ff_playerids()


def clear_cache() -> None:
    """Drop nflreadpy's cached downloads so the next load pulls fresh."""
    configure_cache()
    nfl.clear_cache()
