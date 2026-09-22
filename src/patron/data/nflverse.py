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


def load_expected_opportunity(seasons: list[int]) -> pl.DataFrame:
    """ffopportunity's weekly expected-stat model output.

    The source estimates the value of a player's recorded opportunities; downstream
    code aggregates only source-season rows, so the next-season backtest target is
    never visible to this feature.
    """
    configure_cache()
    return nfl.load_ff_opportunity(seasons, stat_type="weekly")


def load_nextgen_stats(seasons: list[int], stat_type: str) -> pl.DataFrame:
    """Official season/week Next Gen Stats for passing, receiving, or rushing."""
    configure_cache()
    if stat_type not in {"passing", "receiving", "rushing"}:
        raise ValueError(f"unsupported Next Gen Stats type: {stat_type}")
    return nfl.load_nextgen_stats(seasons, stat_type=stat_type)


def load_fantasy_rankings() -> pl.DataFrame:
    """The dated FantasyPros ECR archive used as a preseason market baseline.

    Extended by `market_backfill`: the 2020 combined-offense pages are normalized
    into canonical page types, and 2010–2019 come from timestamped Wayback captures
    committed under ``data/static``. Every row keeps a pre-cutoff scrape date.
    """
    configure_cache()
    from patron.data import market_backfill

    return market_backfill.extend_rankings(nfl.load_ff_rankings(type="all"))


def load_participation(seasons: list[int]) -> pl.DataFrame:
    """Play-level offensive participation used to derive route opportunities."""
    configure_cache()
    frame = nfl.load_participation(seasons)
    require_columns(frame.columns, PARTICIPATION_COLUMNS, "nflverse participation")
    return frame.select(PARTICIPATION_COLUMNS)


def load_participation_flexible(seasons: list[int]) -> pl.DataFrame:
    """Participation with nullable positions for the older (2016–2022) schema.

    The rich weekly research panel resolves those missing positions from the player
    stat feed. Keeping this separate from the strict production loader makes the
    report-only compatibility behavior explicit.
    """
    configure_cache()
    frame = nfl.load_participation(seasons)
    required = tuple(name for name in PARTICIPATION_COLUMNS if name != "offense_positions")
    require_columns(frame.columns, required, "nflverse participation")
    if "offense_positions" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("offense_positions"))
    return frame.select(PARTICIPATION_COLUMNS)


def canonicalize_depth_charts(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize timestamped ESPN and legacy weekly nflverse depth-chart schemas.

    Legacy data has no publication timestamp. Week 1 is the closest reproducible
    preseason boundary and is assigned an August 31 proxy date so the backtest cutoff
    can treat every season consistently. The report labels this approximation.
    """
    if set(DEPTH_CHART_COLUMNS).issubset(frame.columns):
        return frame.select(DEPTH_CHART_COLUMNS)

    legacy = (
        "season",
        "week",
        "game_type",
        "club_code",
        "gsis_id",
        "position",
        "depth_position",
        "depth_team",
    )
    require_columns(frame.columns, legacy, "legacy nflverse depth charts")
    week_one = frame.filter((pl.col("game_type") == "REG") & (pl.col("week") == 1))
    return week_one.select(
        pl.concat_str(pl.col("season").cast(pl.String), pl.lit("-08-31T00:00:00Z")).alias("dt"),
        pl.col("club_code").alias("team"),
        "gsis_id",
        pl.col("position").alias("pos_abb"),
        pl.col("depth_position").alias("pos_name"),
        pl.col("depth_team").cast(pl.Int32, strict=False).alias("pos_rank"),
    )


def load_depth_charts(seasons: list[int]) -> pl.DataFrame:
    """Published nflverse depth charts normalized across the 2025 source change."""
    configure_cache()
    frame = nfl.load_depth_charts(seasons)
    return canonicalize_depth_charts(frame)


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


def load_weekly_rosters(seasons: list[int]) -> pl.DataFrame:
    """Week-level roster/status history used by report-only preseason experiments."""
    configure_cache()
    return nfl.load_rosters_weekly(seasons)


def load_snap_counts(seasons: list[int]) -> pl.DataFrame:
    """Game-level offensive snap counts (PFR, available from 2013 in practice)."""
    configure_cache()
    return nfl.load_snap_counts(seasons)


def load_players() -> pl.DataFrame:
    """Canonical player identity plus draft capital and cross-source IDs."""
    configure_cache()
    return nfl.load_players()


def load_contracts() -> pl.DataFrame:
    """Historical OverTheCap contracts for cutoff-safe roster-investment features."""
    configure_cache()
    return nfl.load_contracts()


def load_combine() -> pl.DataFrame:
    """Historical combine measurements for cutoff-safe athleticism experiments."""
    configure_cache()
    return nfl.load_combine()


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
    from patron.data import market_backfill

    return market_backfill.extend_crosswalk(nfl.load_ff_playerids())


def clear_cache() -> None:
    """Drop nflreadpy's cached downloads so the next load pulls fresh."""
    configure_cache()
    nfl.clear_cache()
