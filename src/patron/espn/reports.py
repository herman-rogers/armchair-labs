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
import math
from collections.abc import Iterable

import polars as pl

from patron.board.overrides import ADJUSTED_VOR
from patron.espn.crosswalk import (
    CHANGED_TEAM,
    ESPN_ID,
    INJURY_STATUS,
    IS_FREE_AGENT,
    IS_MINE,
    OWNER_TEAM_ID,
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

# A bench is useful insurance and trade inventory, but it should never be valued like
# another starting lineup. Injured reserve does not contribute current roster strength.
BENCH_STRENGTH_WEIGHT = 0.20
BENCH_SLOTS = frozenset({"BE", "BENCH"})
IR_SLOTS = frozenset({"IR", "INJURED RESERVE"})


def _projection_ppg_column(frame: pl.DataFrame) -> str:
    """Season-equivalent V2 production, historical PPG for V1."""
    return "season_equivalent_ppg" if "season_equivalent_ppg" in frame.columns else "ppg"


def _board_value_column(frame: pl.DataFrame) -> str:
    return "v2_score" if "v2_score" in frame.columns else ADJUSTED_VOR


def wire_replacement_levels(
    tagged_board: pl.DataFrame,
    baseline_ranks: dict[str, int],
    ppg_column: str | None = None,
) -> dict[str, float]:
    """Replacement level measured against the free-agent pool, not the preseason board.

    The best freely-available player at a position *is* replacement level, by
    definition. Preseason baselines assume an undrafted pool; once a league has drafted,
    the honest bar is whatever is still sitting there.

    Falls back to the best available player when the pool is thinner than the baseline
    rank, which is the common case at tight end.
    """
    ppg_column = ppg_column or _projection_ppg_column(tagged_board)
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
    ppg_column = _projection_ppg_column(tagged_board)
    levels = wire_replacement_levels(tagged_board, baseline_ranks, ppg_column=ppg_column)
    logger.info(
        "wire replacement: %s",
        ", ".join(f"{position} {ppg:.1f}" for position, ppg in sorted(levels.items())),
    )

    wire = tagged_board.filter(pl.col(IS_FREE_AGENT)).with_columns(
        pl.col("position")
        .replace_strict(levels, default=None, return_dtype=pl.Float64)
        .alias(WIRE_REPLACEMENT)
    )
    wire = wire.with_columns((pl.col(ppg_column) - pl.col(WIRE_REPLACEMENT)).alias(WIRE_VOR))

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
    value_column = _board_value_column(tagged_board)
    return mine.with_columns(
        pl.col(INJURY_STATUS).is_in(list(CONCERNING_INJURY_TAGS)).fill_null(False).alias("flagged")
    ).sort(["flagged", CHANGED_TEAM, value_column], descending=[True, True, True])


def opponent_weaknesses(
    tagged_board: pl.DataFrame,
    baseline_ranks: dict[str, int],
) -> pl.DataFrame:
    """Each team's worst starting position, by the value of its best player there.

    The Dan-TE-room scan: knowing which rival is thin somewhere is what makes a trade
    offer land, and what tells you which free agent to take before they need him.
    """
    rostered = tagged_board.filter(~pl.col(IS_FREE_AGENT) & pl.col(OWNER_TEAM_NAME).is_not_null())

    value_column = _board_value_column(tagged_board)
    best = (
        rostered.filter(pl.col("position").is_in(list(baseline_ranks)))
        .group_by([OWNER_TEAM_NAME, "position"])
        .agg(
            pl.col(value_column).max().alias("best_vor"),
            pl.col("player_display_name")
            .sort_by(value_column, descending=True)
            .first()
            .alias("best_player"),
            pl.len().alias("depth"),
        )
    )
    return best.sort([OWNER_TEAM_NAME, "best_vor"])


def team_strengths(
    tagged_board: pl.DataFrame,
    espn_players: pl.DataFrame,
    team_ids: Iterable[int],
    *,
    season_games: int = 17,
) -> dict[int, dict[str, int | float]]:
    """Rank complete fantasy rosters using the selected metric generation.

    A matched player's holistic board value is the input: ``v2_score`` for V2 and
    adjusted VOR for V1. Current ESPN starters count fully; positive bench value counts
    at 20% as depth. Rookies with no nflverse history use ESPN projected PPG above the
    selected board's positional replacement level, which prevents a strong rookie class
    from disappearing from the league comparison merely because V2 cannot model it yet.

    ``team_score`` is a league-relative 0–10 index centered at 5.0. A 1.5-point change
    represents one league standard deviation, so the number stays useful without
    forcing the best and worst teams to artificial 10.0 and 0.0 endpoints.
    """

    ids = [int(team_id) for team_id in team_ids]
    raw_strength = {team_id: 0.0 for team_id in ids}
    scored_players = {team_id: 0 for team_id in ids}
    fallback_players = {team_id: 0 for team_id in ids}
    if not ids or espn_players.height == 0:
        return {
            team_id: {
                "team_rank": rank,
                "team_score": 5.0,
                "scored_players": 0,
                "fallback_players": 0,
            }
            for rank, team_id in enumerate(ids, start=1)
        }

    value_column = _board_value_column(tagged_board)
    ppg_column = _projection_ppg_column(tagged_board)
    replacement_column = (
        "proj_repl_ppg" if "proj_repl_ppg" in tagged_board.columns else "repl_ppg"
    )

    board_values: dict[int, float] = {}
    if ESPN_ID in tagged_board.columns:
        for row in tagged_board.select(ESPN_ID, value_column).iter_rows(named=True):
            espn_id = row.get(ESPN_ID)
            value = row.get(value_column)
            if espn_id is not None and isinstance(value, int | float) and math.isfinite(value):
                board_values[int(espn_id)] = float(value)

    replacement: dict[str, float] = {}
    if replacement_column in tagged_board.columns:
        for position, value in (
            tagged_board.select("position", replacement_column)
            .drop_nulls()
            .unique(subset=["position"])
            .iter_rows()
        ):
            if isinstance(value, int | float) and math.isfinite(value):
                replacement[str(position)] = float(value)
    elif ppg_column in tagged_board.columns:
        # Defensive fallback for small synthetic frames and legacy artifacts.
        replacement = {
            str(position): float(value)
            for position, value in tagged_board.group_by("position")
            .agg(pl.col(ppg_column).median())
            .iter_rows()
            if value is not None
        }

    for row in espn_players.iter_rows(named=True):
        owner_id = row.get(OWNER_TEAM_ID)
        position = str(row.get("position") or "").upper()
        if owner_id is None or int(owner_id) not in raw_strength or position not in replacement:
            continue

        team_id = int(owner_id)
        espn_id = row.get(ESPN_ID)
        value = board_values.get(int(espn_id)) if espn_id is not None else None
        if value is None:
            projected_points = row.get("projected_points")
            if not isinstance(projected_points, int | float) or not math.isfinite(projected_points):
                continue
            value = float(projected_points) / max(season_games, 1) - replacement[position]
            fallback_players[team_id] += 1

        slot = str(row.get("lineup_slot") or "").upper()
        if slot in IR_SLOTS:
            weight = 0.0
        elif slot in BENCH_SLOTS or not slot:
            weight = BENCH_STRENGTH_WEIGHT
            value = max(value, 0.0)
        else:
            weight = 1.0

        raw_strength[team_id] += value * weight
        scored_players[team_id] += 1

    values = list(raw_strength.values())
    mean = sum(values) / len(values)
    deviation = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    ordered = sorted(ids, key=lambda team_id: (-raw_strength[team_id], team_id))

    result: dict[int, dict[str, int | float]] = {}
    for rank, team_id in enumerate(ordered, start=1):
        z_score = (raw_strength[team_id] - mean) / deviation if deviation > 0 else 0.0
        rating = max(0.0, min(10.0, 5.0 + 1.5 * z_score))
        result[team_id] = {
            "team_rank": rank,
            "team_score": round(rating, 1),
            "scored_players": scored_players[team_id],
            "fallback_players": fallback_players[team_id],
        }
    return result


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
