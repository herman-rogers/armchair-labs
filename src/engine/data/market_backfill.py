"""Dated market-price history that predates the DynastyProcess ECR archive.

The FantasyPros ECR archive loaded by `nflverse.load_fantasy_rankings` begins in
December 2019, and its per-position redraft pages only appear in late 2020 — which
is why the market baseline previously started at the 2021 forecast. Two additions
extend it back without weakening the dating standard (a snapshot must be provably
from on or before the August cutoff):

1. **2020**: the archive's summer-2020 scrapes exist but live on one combined
   ``redraft-offense`` page whose ``ecr_type`` column separates positional (``rp``)
   from overall (``ro``) ranks. `normalize_offense_pages` rewrites those rows into
   the canonical per-position and overall page types.

2. **2010–2019**: Internet Archive captures of the same FantasyPros cheatsheet
   pages, chosen as the latest crawl on or before each season's cutoff. The crawl
   timestamp is part of the capture URL, so the vintage is verifiable. The captures
   are normalized offline into ``data/static/market_ecr_backfill.csv``; each row
   carries its capture URL and the method that resolved the player's identity.

Preseason ADP (MyFantasyLeague real-league drafts and Fantasy Football Calculator
mocks) is archived the same way in ``data/static/market_adp_backfill.csv``. ADP is
drafter behavior, not expert consensus: it is the acquisition price. It is loaded
here for the market-disagreement work but is not part of the ECR baseline.
"""

from __future__ import annotations

import logging

import polars as pl

from engine.config.settings import get_settings

logger = logging.getLogger(__name__)

ECR_BACKFILL_FILE = "market_ecr_backfill.csv"
ADP_BACKFILL_FILE = "market_adp_backfill.csv"

# Canonical columns expected by `enrichment.build_market_rankings`.
_RANKING_COLUMNS = ("page_type", "id", "pos", "ecr", "sd", "scrape_date")

_OFFENSE_PAGE = "redraft-offense"
_POSITIONAL_ECR_TYPE = "rp"
_OVERALL_ECR_TYPE = "ro"


def normalize_offense_pages(rankings: pl.DataFrame) -> pl.DataFrame:
    """Rewrite combined ``redraft-offense`` rows into canonical page types.

    ``rp`` rows carry each position's own rank sequence and become
    ``redraft-{pos}``; ``ro`` rows carry the cross-position sequence and become
    ``redraft-overall``. Rows from any other page or ecr_type are dropped — the
    caller concatenates this output onto the unmodified archive.
    """
    if "ecr_type" not in rankings.columns:
        return rankings.clear()
    offense = rankings.filter(pl.col("page_type") == _OFFENSE_PAGE)
    positional = offense.filter(
        (pl.col("ecr_type") == _POSITIONAL_ECR_TYPE)
        & pl.col("pos").is_in(["QB", "RB", "WR", "TE"])
    ).with_columns(
        pl.format("redraft-{}", pl.col("pos").str.to_lowercase()).alias("page_type")
    )
    overall = offense.filter(pl.col("ecr_type") == _OVERALL_ECR_TYPE).with_columns(
        pl.lit("redraft-overall").alias("page_type")
    )
    return pl.concat([positional, overall], how="vertical_relaxed")


def load_backfill_rankings() -> pl.DataFrame | None:
    """The committed Wayback ECR backfill in archive-compatible shape, if present."""
    path = get_settings().static_dir / ECR_BACKFILL_FILE
    if not path.exists():
        return None
    frame = pl.read_csv(
        path,
        schema_overrides={
            "id": pl.String,
            "gsis_id": pl.String,
            "ecr": pl.Float64,
            "sd": pl.Float64,
            "scrape_date": pl.String,
        },
    )
    missing = [name for name in _RANKING_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    logger.info(
        "market ECR backfill: %s rows, seasons %s-%s",
        frame.height,
        frame["scrape_date"].str.slice(0, 4).min(),
        frame["scrape_date"].str.slice(0, 4).max(),
    )
    return frame


def extend_rankings(
    rankings: pl.DataFrame, backfill: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Archive rows plus the 2020 offense-page shim plus the Wayback backfill."""
    frames = [rankings]
    shim = normalize_offense_pages(rankings)
    if shim.height:
        frames.append(shim)
    if backfill is None:
        backfill = load_backfill_rankings()
    if backfill is not None and backfill.height:
        frames.append(backfill)
    if len(frames) == 1:
        return rankings
    return pl.concat(frames, how="diagonal_relaxed")


def extend_crosswalk(
    crosswalk: pl.DataFrame, backfill: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Append backfill-only identity pairs to the ffverse ID crosswalk.

    Backfill rows resolved to a GSIS id but absent from `load_ff_playerids` (mostly
    players who retired before the ffverse era) carry synthetic negative
    ``fantasypros_id`` values. Emitting the same ids here makes the standard
    fantasypros_id → gsis_id join in `build_market_rankings` work unchanged.
    """
    if backfill is None:
        backfill = load_backfill_rankings()
    if backfill is None or "gsis_id" not in backfill.columns:
        return crosswalk
    extra = (
        backfill.filter(
            pl.col("gsis_id").is_not_null()
            & pl.col("id").is_not_null()
            & ~pl.col("id").is_in(crosswalk["fantasypros_id"].cast(pl.String).to_list())
        )
        .select(
            pl.col("id").alias("fantasypros_id"),
            "gsis_id",
            pl.col("player").alias("name"),
            pl.col("pos").alias("position"),
            "team",
        )
        .unique(subset=["fantasypros_id"], keep="first")
    )
    if extra.height == 0:
        return crosswalk
    logger.info("market backfill crosswalk: %s appended identity pairs", extra.height)
    return pl.concat([crosswalk, extra], how="diagonal_relaxed")


def load_backfill_adp() -> pl.DataFrame | None:
    """Preseason ADP archive: (season, gsis_id, adp, source, format, window dates).

    Kept separate from the ECR frame because ADP is an observed acquisition price
    with its own dating semantics (a draft *window*, not a scrape date); the
    market-disagreement analysis joins it by ``forecast_season`` + ``player_id``.
    """
    path = get_settings().static_dir / ADP_BACKFILL_FILE
    if not path.exists():
        return None
    frame = pl.read_csv(
        path,
        schema_overrides={
            "gsis_id": pl.String,
            "adp": pl.Float64,
            "forecast_season": pl.Int64,
        },
    )
    logger.info(
        "market ADP backfill: %s rows, seasons %s-%s",
        frame.height,
        frame["forecast_season"].min(),
        frame["forecast_season"].max(),
    )
    return frame
