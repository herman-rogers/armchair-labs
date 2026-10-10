"""nflverse loading — the only module in the package that touches the network.

Used by the weekly capture (`engine.data.weekly`) and the ESPN identity crosswalk.
Historical observation tables never call this module: enrichment reads the preserved
provider files directly with DuckDB (`engine.data.enrich`).

nflreadpy ships its own filesystem cache, so this module configures it rather than
reimplementing one. The cache is pointed at the project's data directory so that all
generated state lives in one place and can be cleared with `engine clear-cache`.

nflverse is a free community resource and the correct posture is a quiet houseguest:
cache aggressively, refresh on a slow clock, and never poll faster than the data
actually changes.
"""

from __future__ import annotations

import logging

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import update_config

from engine.config.settings import Settings, get_settings
from engine.scoring.columns import (
    PBP_TOUCHDOWN_COLUMNS,
    SCHEDULE_COLUMNS,
    SKILL_PLAYER_COLUMNS,
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
    """Weekly player stat lines for `seasons`, filtered to one season type."""
    configure_cache()
    frame = nfl.load_player_stats(seasons, summary_level="week")
    require_columns(frame.columns, SKILL_PLAYER_COLUMNS, "nflverse player stats")
    filtered = frame.filter(pl.col("season_type") == season_type)
    logger.info("loaded %s player-weeks for seasons %s (%s)", filtered.height, seasons, season_type)
    return filtered


def load_touchdown_plays(seasons: list[int]) -> pl.DataFrame:
    """Scoring plays only, with just the columns needed for the distance bonuses."""
    configure_cache()
    frames: list[pl.DataFrame] = []
    for season in seasons:
        pbp = nfl.load_pbp([season])
        require_columns(pbp.columns, PBP_TOUCHDOWN_COLUMNS, f"nflverse pbp {season}")
        frames.append(pbp.filter(pl.col("touchdown") == 1).select(PBP_TOUCHDOWN_COLUMNS))
        logger.info("loaded %s touchdown plays for %s", frames[-1].height, season)
    return pl.concat(frames, how="vertical_relaxed")


def load_snap_counts(seasons: list[int]) -> pl.DataFrame:
    """Game-level offensive snap counts (PFR, available from 2013 in practice)."""
    configure_cache()
    return nfl.load_snap_counts(seasons)


def load_players() -> pl.DataFrame:
    """Canonical player identity plus draft capital and cross-source IDs."""
    configure_cache()
    return nfl.load_players()


def load_schedules(season: int) -> pl.DataFrame:
    """Game results, for completed-week detection and bye weeks."""
    configure_cache()
    schedules = nfl.load_schedules([season])
    require_columns(schedules.columns, SCHEDULE_COLUMNS, "nflverse schedules")
    return schedules


def load_id_crosswalk() -> pl.DataFrame:
    """The ffverse player-ID crosswalk: nflverse `gsis_id` to ESPN and other IDs."""
    configure_cache()
    return nfl.load_ff_playerids()


def clear_cache() -> None:
    """Drop nflreadpy's cached downloads so the next load pulls fresh."""
    configure_cache()
    nfl.clear_cache()
