"""The artifacts a poll produces.

Three, per §5 of the plan: the ranked wire, my roster's health, and the raw material
for alerts. All of them are the same board, filtered by who owns whom — which is the
whole point of joining ESPN state onto it.

The wire is the one that wins waivers. Replacement level is recomputed from what is
actually available rather than reused from the preseason board, because "value over
replacement" only means anything if replacement means "what I can have for free right
now". In a league where half the pool is rostered, the real bar is far below the
preseason one, and using the stale number understates every pickup on the list.
"""

from __future__ import annotations

import logging

import polars as pl

from patron.board.overrides import ADJUSTED_VOR
from patron.espn.crosswalk import (
    CHANGED_TEAM,
    INJURY_STATUS,
    IS_FREE_AGENT,
    IS_MINE,
    OWNER_TEAM_NAME,
    PERCENT_OWNED,
)

logger = logging.getLogger(__name__)

WIRE_VOR = "wire_vor"
WIRE_REPLACEMENT = "wire_repl_ppg"

#: Injury tags worth surfacing on a roster report. ACTIVE and NORMAL are not news.
CONCERNING_INJURY_TAGS = frozenset(
    {"QUESTIONABLE", "DOUBTFUL", "OUT", "INJURY_RESERVE", "SUSPENSION", "DAY_TO_DAY"}
)


def wire_replacement_levels(
    tagged_board: pl.DataFrame,
    baseline_ranks: dict[str, int],
    ppg_column: str = "ppg",
) -> dict[str, float]:
    """Replacement level measured against the free-agent pool, not the preseason board.

    The best freely-available player at a position *is* replacement level, by
    definition. Preseason baselines assume an undrafted pool; once a league has drafted,
    the honest bar is whatever is still sitting there.

    Falls back to the best available player when the pool is thinner than the baseline
    rank, which is the common case at tight end.
    """
    levels: dict[str, float] = {}
    available = tagged_board.filter(pl.col(IS_FREE_AGENT))

    for position, rank in baseline_ranks.items():
        pool = (
            available.filter(pl.col("position") == position)
            .sort(ppg_column, descending=True)
            .get_column(ppg_column)
            .drop_nulls()
        )
        if pool.len() == 0:
            continue
        levels[position] = float(pool[min(rank, pool.len()) - 1])

    return levels


#: Tags meaning the player cannot be started this week. Worth keeping on the list — a
#: cheap stash is a real play in a league with an IR slot — but worth being able to hide.
UNAVAILABLE_INJURY_TAGS = frozenset({"OUT", "INJURY_RESERVE", "SUSPENSION"})


def ranked_wire(
    tagged_board: pl.DataFrame,
    baseline_ranks: dict[str, int],
    min_vor: float | None = 0.0,
    limit: int | None = 50,
    healthy_only: bool = False,
) -> pl.DataFrame:
    """Free agents worth having, ranked against what else is free.

    Ranks on value alone and leaves availability as a separate, visible column rather
    than blending the two into one score. An OUT player at the top of the list is
    information — a cheap stash — not an error, so hiding him is opt-in.

    Args:
        tagged_board: Board with ownership attached.
        baseline_ranks: Positional replacement ranks from league config.
        min_vor: Drop players below this wire-VOR. `None` keeps everything.
        limit: Cap on rows returned.
        healthy_only: Drop players who cannot be started this week.
    """
    levels = wire_replacement_levels(tagged_board, baseline_ranks)
    logger.info(
        "wire replacement: %s",
        ", ".join(f"{position} {ppg:.1f}" for position, ppg in sorted(levels.items())),
    )

    wire = tagged_board.filter(pl.col(IS_FREE_AGENT)).with_columns(
        pl.col("position")
        .replace_strict(levels, default=None, return_dtype=pl.Float64)
        .alias(WIRE_REPLACEMENT)
    )
    wire = wire.with_columns((pl.col("ppg") - pl.col(WIRE_REPLACEMENT)).alias(WIRE_VOR))

    if min_vor is not None:
        wire = wire.filter(pl.col(WIRE_VOR) >= min_vor)
    if healthy_only:
        wire = wire.filter(
            ~pl.col(INJURY_STATUS).is_in(list(UNAVAILABLE_INJURY_TAGS)).fill_null(False)
        )

    wire = wire.sort(WIRE_VOR, descending=True, nulls_last=True)
    return wire.head(limit) if limit else wire


def roster_health(
    tagged_board: pl.DataFrame,
    my_team_id: int | None = None,
) -> pl.DataFrame:
    """My roster, worst news first.

    Sorted so the rows demanding a decision surface above the ones that do not: an
    injury tag outranks a team change, which outranks a healthy starter.
    """
    mine = tagged_board.filter(pl.col(IS_MINE)) if my_team_id is not None else tagged_board
    return mine.with_columns(
        pl.col(INJURY_STATUS).is_in(list(CONCERNING_INJURY_TAGS)).fill_null(False).alias("flagged")
    ).sort(["flagged", CHANGED_TEAM, ADJUSTED_VOR], descending=[True, True, True])


def opponent_weaknesses(
    tagged_board: pl.DataFrame,
    baseline_ranks: dict[str, int],
) -> pl.DataFrame:
    """Each team's worst starting position, by the value of its best player there.

    The Dan-TE-room scan: knowing which rival is thin somewhere is what makes a trade
    offer land, and what tells you which free agent to take before they need him.
    """
    rostered = tagged_board.filter(~pl.col(IS_FREE_AGENT) & pl.col(OWNER_TEAM_NAME).is_not_null())

    best = (
        rostered.filter(pl.col("position").is_in(list(baseline_ranks)))
        .group_by([OWNER_TEAM_NAME, "position"])
        .agg(
            pl.col(ADJUSTED_VOR).max().alias("best_vor"),
            pl.col("player_display_name")
            .sort_by(ADJUSTED_VOR, descending=True)
            .first()
            .alias("best_player"),
            pl.len().alias("depth"),
        )
    )
    return best.sort([OWNER_TEAM_NAME, "best_vor"])


def unrankable_players(
    espn_players: pl.DataFrame,
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE"),
) -> pl.DataFrame:
    """ESPN players the board could not rank — rookies, and anyone with no prior tape.

    Surfaced rather than dropped. These are exactly the players the model is blind to,
    and a wire report that silently omits a hyped rookie is worse than one that says it
    cannot price him.
    """
    return (
        espn_players.filter(
            pl.col("player_id").is_null() & pl.col("position").is_in(list(positions))
        )
        .sort(PERCENT_OWNED, descending=True, nulls_last=True)
        .select(
            "player_display_name",
            "position",
            "espn_team",
            PERCENT_OWNED,
            OWNER_TEAM_NAME,
            INJURY_STATUS,
        )
    )
