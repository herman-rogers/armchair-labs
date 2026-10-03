"""Manual situational overrides.

The model's blind spots are trades, coaching changes, and injuries: everything that
happened after the last snap of the season the data came from. During draft prep those
adjustments were applied by hand to the finished board, which is why six rows of
`data/static/2026_draft_list.md` sit out of pure VOR order. This module turns that
judgement into code, with a reason attached to every entry.

An override is a delta on computed VOR, in points per game. Deltas rather than fixed
ranks deliberately: a delta composes with re-ranking, so an entry keeps working
in-season as the player pool moves around it, while a hardcoded rank breaks the first
time the wire changes.

**An override that matches nothing is an error.** That is the important design decision
here. A typo in a name would otherwise mean a player carrying an Achilles tear quietly
keeps his full ranking — the failure is invisible and it is exactly the failure the
override file exists to prevent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from engine.config.settings import load_overrides_config
from engine.names import normalize
from engine.scoring.columns import require_columns

logger = logging.getLogger(__name__)

ADJUSTED_VOR = "adj_vor"
OVERRIDE_DELTA = "override_delta"
OVERRIDE_REASON = "override_reason"


class UnmatchedOverrideError(ValueError):
    """Raised when a configured override matches no player on the board."""

    def __init__(self, unmatched: list[str]) -> None:
        self.unmatched = unmatched
        listing = "\n  ".join(sorted(unmatched))
        super().__init__(
            f"{len(unmatched)} override(s) matched no player on the board:\n  {listing}\n"
            "A silently-dropped override is how a hurt player keeps his full ranking. "
            "Fix the spelling in config/overrides.yaml, or remove the entry if the "
            "player is genuinely no longer in the pool."
        )


@dataclass(frozen=True)
class Override:
    """One manual adjustment to a player's computed value."""

    player: str
    position: str
    vor_delta: float
    reason: str

    @property
    def key(self) -> tuple[str, str]:
        return normalize(self.player), self.position.upper()


@dataclass(frozen=True)
class Removal:
    """A player taken off the board entirely."""

    player: str
    position: str
    reason: str

    @property
    def key(self) -> tuple[str, str]:
        return normalize(self.player), self.position.upper()


@dataclass(frozen=True)
class BlindSpot:
    """A draftable player the model cannot rank, for want of a usable sample.

    Carried alongside the board rather than in it: there is no VOR to adjust, so
    inventing one would be worse than showing the note.
    """

    player: str
    position: str
    team: str
    note: str


@dataclass(frozen=True)
class OverrideSet:
    overrides: list[Override] = field(default_factory=list)
    removals: list[Removal] = field(default_factory=list)
    blind_spots: list[BlindSpot] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict | None = None) -> OverrideSet:
        config = config if config is not None else load_overrides_config()
        return cls(
            overrides=[Override(**entry) for entry in config.get("overrides") or []],
            removals=[Removal(**entry) for entry in config.get("removed") or []],
            blind_spots=[BlindSpot(**entry) for entry in config.get("blind_spots") or []],
        )


def apply_overrides(
    board: pl.DataFrame,
    override_set: OverrideSet | None = None,
    strict: bool = True,
    name_column: str = "player_display_name",
    vor_column: str = "vor",
) -> pl.DataFrame:
    """Apply removals and VOR deltas, returning the board with `adj_vor`.

    Args:
        board: Ranked rows carrying `name_column`, `position`, and `vor_column`.
        override_set: Overrides to apply; loaded from config by default.
        strict: Raise `UnmatchedOverrideError` when an entry matches nobody. Turning
            this off logs instead, and should only be done when the board is
            deliberately a subset (a single position, say).
        name_column: Display-name column to match on.
        vor_column: Computed VOR column.

    Returns:
        `board` minus removed players, with `override_delta`, `override_reason`, and
        `adj_vor` columns. `adj_vor` equals `vor` for the untouched majority.
    """
    override_set = override_set or OverrideSet.from_config()
    require_columns(board.columns, (name_column, "position", vor_column), "apply_overrides input")

    keyed = board.with_columns(
        pl.struct([name_column, "position"])
        .map_elements(
            lambda row: f"{normalize(row[name_column])}|{row['position'].upper()}",
            return_dtype=pl.String,
        )
        .alias("_match_key")
    )

    present: set[str] = set(keyed["_match_key"].to_list())
    unmatched: list[str] = []

    def encode(key: tuple[str, str]) -> str:
        return f"{key[0]}|{key[1]}"

    removal_keys: list[str] = []
    for removal in override_set.removals:
        encoded = encode(removal.key)
        if encoded in present:
            removal_keys.append(encoded)
        else:
            # A removal that matches nothing is benign — the player is already absent,
            # which is the state the entry was asking for. Worth a note, not an error.
            logger.info(
                "removal for %s (%s) matched no board row; already absent",
                removal.player,
                removal.position,
            )

    kept = keyed.filter(~pl.col("_match_key").is_in(removal_keys))
    remaining: set[str] = set(kept["_match_key"].to_list())

    deltas: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for override in override_set.overrides:
        encoded = encode(override.key)
        if encoded not in remaining:
            unmatched.append(f"{override.player} ({override.position}) — {override.reason}")
            continue
        deltas[encoded] = override.vor_delta
        reasons[encoded] = override.reason

    if unmatched:
        if strict:
            raise UnmatchedOverrideError(unmatched)
        logger.warning("%s override(s) matched no board row", len(unmatched))

    adjusted = (
        kept.with_columns(
            pl.col("_match_key")
            .replace_strict(deltas, default=0.0, return_dtype=pl.Float64)
            .alias(OVERRIDE_DELTA),
            pl.col("_match_key")
            .replace_strict(reasons, default=None, return_dtype=pl.String)
            .alias(OVERRIDE_REASON),
        )
        .with_columns((pl.col(vor_column) + pl.col(OVERRIDE_DELTA)).alias(ADJUSTED_VOR))
        .drop("_match_key")
    )

    applied = sum(1 for value in deltas.values() if value)
    logger.info("applied %s override(s) and %s removal(s)", applied, len(removal_keys))
    return adjusted
