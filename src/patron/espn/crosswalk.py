"""Joining ESPN's player universe to nflverse's.

The plan's §8 calls name joining "the tax", and it is right to: ESPN uses its own
player IDs and display names, nflverse keys on `gsis_id`, and the two disagree in every
direction — "D.J. Moore" against "DJ Moore", suffixes present on one side only,
nicknames. A silent join failure is the worst outcome available, because the board
still builds and simply omits the one player who mattered.

The fix is to not join on names. nflverse publishes the ffverse/DynastyProcess ID
crosswalk, which maps `espn_id` directly to `gsis_id`. Measured against this league:
of 417 skill players on rosters and the top of the wire, the crosswalk matched 82% and
the name fallback added **zero** on top. The remaining 18% are 2026 rookies and players
with no 2025 tape — they are not join failures, they are genuinely unrankable, and they
are reported as such rather than dropped.

The name fallback is kept anyway. It costs nothing, and the day the crosswalk lags a
mid-season call-up it is the difference between a missing row and a wrong one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from patron.data import nflverse
from patron.espn import teams
from patron.names import normalize

logger = logging.getLogger(__name__)

AVAILABILITY = "availability"

#: A player ESPN reports as owned by a team.
ROSTERED = "rostered"
#: A player ESPN reports and nobody owns — actually claimable.
FREE_AGENT = "free_agent"
#: A player the snapshot never mentioned. Not the same as available: he may sit below
#: the free-agent depth we pull, or not be in ESPN's player universe at all.
UNKNOWN = "unknown"

OWNER_TEAM_ID = "owner_team_id"
OWNER_TEAM_NAME = "owner_team_name"
IS_FREE_AGENT = "is_free_agent"
IS_MINE = "is_mine"
ESPN_ID = "espn_id"
ESPN_TEAM = "espn_team"
INJURY_STATUS = "injury_status"
PERCENT_OWNED = "percent_owned"
CHANGED_TEAM = "changed_team"


#: Positions the board ranks. Kickers and defenses are deliberately excluded — this
#: league tiers them rather than ranking them by VOR — so their absence from the board
#: is by design and must not be counted as a join failure.
RANKED_POSITIONS = ("QB", "RB", "WR", "TE")


@dataclass
class JoinReport:
    """What matched, what did not, and by which route.

    Exists because the §8 failure mode is silence. Every consumer of the join is
    expected to surface `unmatched`, not swallow it.

    `unmatched` counts only skill positions. Kickers and defenses are never on the
    board, so folding them in dragged a genuine 82% match rate down to a misleading
    70% and would have hidden a real regression behind expected noise.
    """

    matched_by_id: int = 0
    matched_by_name: int = 0
    unmatched: list[tuple[str, str, str]] = field(default_factory=list)
    #: Players skipped because their position is not ranked at all.
    not_ranked: int = 0

    @property
    def total(self) -> int:
        """Players the board could reasonably have matched."""
        return self.matched_by_id + self.matched_by_name + len(self.unmatched)

    @property
    def match_rate(self) -> float:
        return (self.matched_by_id + self.matched_by_name) / self.total if self.total else 0.0

    def summary(self) -> str:
        tail = f", {self.not_ranked} K/DST not ranked" if self.not_ranked else ""
        return (
            f"joined {self.matched_by_id + self.matched_by_name}/{self.total} skill players "
            f"({self.match_rate:.0%}) — {self.matched_by_id} by crosswalk id, "
            f"{self.matched_by_name} by name, {len(self.unmatched)} without prior tape{tail}"
        )


def build_espn_to_gsis() -> dict[int, str]:
    """Map ESPN player id to nflverse `gsis_id`.

    The crosswalk stores ids as strings, and some arrive as floats ("3117251.0"), so
    they are normalised through float before int rather than parsed strictly.
    """
    crosswalk = nflverse.load_id_crosswalk()
    if ESPN_ID not in crosswalk.columns or "gsis_id" not in crosswalk.columns:
        logger.error("crosswalk is missing espn_id or gsis_id; falling back to names only")
        return {}

    usable = crosswalk.filter(pl.col(ESPN_ID).is_not_null() & pl.col("gsis_id").is_not_null())
    mapping: dict[int, str] = {}
    for espn_id, gsis_id in usable.select(ESPN_ID, "gsis_id").iter_rows():
        try:
            mapping[int(float(espn_id))] = str(gsis_id)
        except (TypeError, ValueError):
            continue

    logger.info("crosswalk: %s usable espn_id -> gsis_id mappings", len(mapping))
    return mapping


def resolve_player_ids(
    espn_players: pl.DataFrame,
    board_player_ids: set[str],
    board_name_keys: dict[tuple[str, str], str],
    espn_to_gsis: dict[int, str] | None = None,
) -> tuple[pl.DataFrame, JoinReport]:
    """Attach a `player_id` (gsis) to each ESPN player row.

    Args:
        espn_players: Rows with `espn_id`, `player_display_name`, `position`.
        board_player_ids: gsis ids present on the board, used to confirm a hit.
        board_name_keys: `(normalized name, position)` to gsis id, for the fallback.
        espn_to_gsis: Crosswalk; loaded when omitted.

    Returns:
        The frame with a `player_id` column (null when unresolved), and a report.
    """
    espn_to_gsis = espn_to_gsis if espn_to_gsis is not None else build_espn_to_gsis()
    report = JoinReport()
    resolved: list[str | None] = []

    for row in espn_players.iter_rows(named=True):
        espn_id = row.get(ESPN_ID)
        name = row.get("player_display_name") or ""
        position = (row.get("position") or "").upper()

        gsis = espn_to_gsis.get(int(espn_id)) if espn_id is not None else None
        if gsis and gsis in board_player_ids:
            report.matched_by_id += 1
            resolved.append(gsis)
            continue

        by_name = board_name_keys.get((normalize(name), position))
        if by_name:
            report.matched_by_name += 1
            resolved.append(by_name)
            continue

        if position not in RANKED_POSITIONS:
            report.not_ranked += 1
        else:
            report.unmatched.append((name, position, row.get(ESPN_TEAM) or "?"))
        resolved.append(None)

    return espn_players.with_columns(pl.Series("player_id", resolved, dtype=pl.String)), report


def attach_ownership(
    board: pl.DataFrame,
    espn_players: pl.DataFrame,
    my_team_id: int | None = None,
) -> pl.DataFrame:
    """Tag every board row with who owns the player, and what ESPN says about him now.

    ESPN is authoritative for anything about the present — team, injury tag, roster
    status — while the board is authoritative for production. Where they disagree about
    team, both are kept and `changed_team` is set: a player who moved carries situational
    stats from a team he no longer plays for, which is precisely the staleness §8 warns
    about, and it should be visible on the row rather than silently averaged in.
    """
    joined = board.join(
        espn_players.filter(pl.col("player_id").is_not_null()).unique(
            subset=["player_id"], keep="first"
        ),
        on="player_id",
        how="left",
        suffix="_espn",
    )

    # Presence in the snapshot is what separates "claimable" from "never mentioned".
    # Treating absence as availability put 267 unverified players onto a 472-row wire —
    # more than half the list was players ESPN had never said anything about.
    seen = pl.col(ESPN_ID).is_not_null()
    availability = (
        pl.when(~seen)
        .then(pl.lit(UNKNOWN))
        .when(pl.col(OWNER_TEAM_ID).is_not_null())
        .then(pl.lit(ROSTERED))
        .otherwise(pl.lit(FREE_AGENT))
    )

    return joined.with_columns(
        availability.alias(AVAILABILITY),
        (seen & pl.col(OWNER_TEAM_ID).is_null()).alias(IS_FREE_AGENT),
        (pl.col(OWNER_TEAM_ID) == my_team_id).fill_null(False).alias(IS_MINE)
        if my_team_id is not None
        else pl.lit(False).alias(IS_MINE),
        pl.struct(["team", ESPN_TEAM])
        .map_elements(
            # Compared through the alias map, not by string equality: nflverse says
            # "LA"/"WAS" where ESPN says "LAR"/"WSH", and raw comparison reported
            # nineteen players as having changed teams when none had.
            lambda row: not teams.same_team(row["team"], row[ESPN_TEAM]),
            return_dtype=pl.Boolean,
        )
        .alias(CHANGED_TEAM),
    )


def board_lookups(board: pl.DataFrame) -> tuple[set[str], dict[tuple[str, str], str]]:
    """Build the id set and name index the resolver needs."""
    ids = set(board["player_id"].to_list())
    names = {
        (normalize(name), position): player_id
        for name, position, player_id in board.select(
            "player_display_name", "position", "player_id"
        ).iter_rows()
    }
    return ids, names
