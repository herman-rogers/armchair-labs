"""Optimal starting lineups, and the roster strength that follows from them.

Comparing two teams by total roster value is misleading: a team with five good
running backs cannot start five of them. The only comparison that means anything is
between the lineups each team can actually field, which means solving the league's
slot structure — dedicated slots first, then flex from whoever is left.

That structure comes from ESPN (`roster_slots` on the snapshot), not from config, so
a commissioner adding a flex changes the answer here without a code change.

Two deliberate limits, both stated rather than hidden:

Players the board cannot rank — 2026 rookies, anyone with no prior tape — score
nothing and will not be selected. A team whose best receiver is a rookie will look
thinner here than it is, so `unranked_starters` reports how many such players the
roster holds.

This is a season-strength comparison, not a weekly projection. It does not know byes,
this week's injury report, or matchup. It answers "who has the better team", which is
the question a trade or a schedule scan asks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

logger = logging.getLogger(__name__)

#: Which positions may fill a combination slot. ESPN names them by the positions they
#: accept, joined with "/", so the mapping is derivable rather than enumerated.
FLEX_SEPARATOR = "/"

#: Positions the board ranks. Kickers and defenses are tiered by this league rather
#: than ranked, so they carry no comparable value and their slots are excluded from a
#: strength comparison entirely — every team would otherwise show the same two empty
#: rows, which is noise rather than information.
RANKED_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


def rankable_slots(roster_slots: dict[str, int]) -> dict[str, int]:
    """Drop slots the board cannot value.

    A slot survives only if every position it accepts is ranked, so "RB/WR/TE" stays
    and "K" does not.
    """
    return {
        slot: count
        for slot, count in roster_slots.items()
        if set(slot.split(FLEX_SEPARATOR)) <= RANKED_POSITIONS
    }


@dataclass
class LineupSlot:
    """One filled starting position."""

    slot: str
    player_id: str | None
    player_name: str | None
    position: str | None
    value: float

    @property
    def empty(self) -> bool:
        return self.player_id is None


@dataclass
class TeamStrength:
    """A team's best available starting lineup, and what it is worth."""

    team_id: int
    team_name: str
    starters: list[LineupSlot] = field(default_factory=list)
    bench: list[LineupSlot] = field(default_factory=list)
    #: Rostered skill players the board cannot rank — 2026 rookies, anyone with no
    #: prior tape. They score nothing, so a roster leaning on them is understated here
    #: and this count is how the UI can say so.
    unranked_starters: int = 0

    @property
    def total(self) -> float:
        return sum(slot.value for slot in self.starters)

    @property
    def by_position(self) -> dict[str, float]:
        """Starting value grouped by the position actually filling each slot."""
        grouped: dict[str, float] = {}
        for slot in self.starters:
            if slot.position:
                grouped[slot.position] = grouped.get(slot.position, 0.0) + slot.value
        return grouped


def _slot_accepts(slot: str, position: str) -> bool:
    """Whether `position` may fill `slot`.

    A dedicated slot accepts its own position; a combination slot like "RB/WR/TE"
    accepts any of the positions in its name.
    """
    if slot == position:
        return True
    return FLEX_SEPARATOR in slot and position in slot.split(FLEX_SEPARATOR)


def best_lineup(
    roster: pl.DataFrame,
    roster_slots: dict[str, int],
    value_column: str,
    team_id: int = 0,
    team_name: str = "",
    unranked_count: int | None = None,
) -> TeamStrength:
    """Fill the league's starting slots with the most valuable eligible players.

    Dedicated slots are filled before combination slots. That ordering matters: a flex
    filled first could take the best running back and leave a dedicated RB slot to a
    worse one, producing a lineup the manager would never actually set.

    Args:
        roster: One team's players, with `player_id`, `player_display_name`,
            `position`, and `value_column`.
        roster_slots: Slot name to count, e.g. `{"RB": 2, "RB/WR/TE": 1}`.
        value_column: Which metric to maximise.
        team_id: Passed through to the result.
        team_name: Passed through to the result.
        unranked_count: How many rostered players the board cannot rank. Must be
            supplied by the caller: unrankable players are absent from the board
            entirely rather than present with a null value, so this cannot be
            derived from `roster`.

    Returns:
        The filled lineup, with anyone left over as bench.
    """
    roster_slots = rankable_slots(roster_slots)
    available = (
        roster.filter(pl.col(value_column).is_not_null())
        .sort(value_column, descending=True)
        .to_dicts()
    )
    unranked = (
        unranked_count
        if unranked_count is not None
        else roster.filter(pl.col(value_column).is_null()).height
    )

    # Dedicated slots first, combination slots after, so a flex cannot poach a player
    # a dedicated slot needed.
    ordered = sorted(
        roster_slots.items(),
        key=lambda item: (FLEX_SEPARATOR in item[0], item[0]),
    )

    taken: set[str] = set()
    starters: list[LineupSlot] = []

    for slot, count in ordered:
        for _ in range(count):
            pick = next(
                (
                    row
                    for row in available
                    if row["player_id"] not in taken
                    and _slot_accepts(slot, str(row.get("position") or ""))
                ),
                None,
            )
            if pick is None:
                # An unfillable slot is real information — a team with no tight end
                # starts a zero there — so it is kept rather than skipped.
                starters.append(LineupSlot(slot, None, None, None, 0.0))
                continue
            taken.add(pick["player_id"])
            starters.append(
                LineupSlot(
                    slot=slot,
                    player_id=pick["player_id"],
                    player_name=pick.get("player_display_name"),
                    position=pick.get("position"),
                    value=float(pick.get(value_column) or 0.0),
                )
            )

    bench = [
        LineupSlot(
            slot="BE",
            player_id=row["player_id"],
            player_name=row.get("player_display_name"),
            position=row.get("position"),
            value=float(row.get(value_column) or 0.0),
        )
        for row in available
        if row["player_id"] not in taken
    ]

    return TeamStrength(
        team_id=team_id,
        team_name=team_name,
        starters=starters,
        bench=bench,
        unranked_starters=unranked,
    )


def value_column_for(board: pl.DataFrame) -> str:
    """The best available strength metric on this board.

    v2 boards carry a projection; v1 boards carry last season's VOR. Preferring the
    projection matters for a schedule scan, where the question is about weeks that
    have not happened yet.
    """
    for candidate in (
        "v2_overall_vor",
        "v2_rank_vor",
        "adj_proj_vor",
        "v2_score",
        "adj_vor",
        "vor",
        "ppg",
    ):
        if candidate in board.columns:
            return candidate
    raise ValueError("board carries no usable strength column")
