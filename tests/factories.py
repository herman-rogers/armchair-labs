"""Builders for small, complete synthetic frames.

The engine validates its inputs strictly, so a test frame must carry every column the
code under test reads even when the test only cares about two of them. These builders
fill the rest with zeros, which keeps tests readable — each one names only the stats
it is actually asserting on — without weakening that validation in production.

Deliberately no checked-in parquet: fixtures you can read in the test body are worth
more than fixtures you have to open a tool to inspect.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from patron.scoring.columns import (
    NFLVERSE_POINTS_COLUMN,
    OPPORTUNITY_COLUMNS,
    PBP_TOUCHDOWN_COLUMNS,
    SCORING_COLUMNS,
)

# Columns that must be floats even when unset, so Polars does not infer Int64 for a
# frame of zeros and then fail to concat with a real one.
_FLOAT_DEFAULTS = (
    frozenset(SCORING_COLUMNS) | frozenset(OPPORTUNITY_COLUMNS) | {NFLVERSE_POINTS_COLUMN}
)


def player_week(
    player_id: str = "00-0000001",
    name: str = "Test Player",
    position: str = "WR",
    season: int = 2025,
    week: int = 1,
    team: str = "SEA",
    season_type: str = "REG",
    **stats: Any,
) -> dict[str, Any]:
    """One player-week row with every scoring column present, zeroed unless given.

    Raises:
        KeyError: if `stats` names a column the engine does not read. Catching a typo
            like `recieving_yards` here is the whole point — silently adding an
            unknown column would let a test pass while asserting nothing.
    """
    unknown = set(stats) - set(_FLOAT_DEFAULTS)
    if unknown:
        raise KeyError(
            f"unknown stat column(s): {', '.join(sorted(unknown))}. "
            f"Known columns: {', '.join(sorted(_FLOAT_DEFAULTS))}"
        )

    row: dict[str, Any] = {
        "player_id": player_id,
        "player_display_name": name,
        "position": position,
        "season": season,
        "week": week,
        "season_type": season_type,
        "team": team,
    }
    row.update({column: 0.0 for column in _FLOAT_DEFAULTS})
    row.update({column: float(value) for column, value in stats.items()})
    return row


def player_weeks(*rows: dict[str, Any]) -> pl.DataFrame:
    """Assemble `player_week` rows into a frame."""
    return pl.DataFrame(list(rows))


class _Default:
    """Sentinel distinguishing "not specified" from an explicit null player id.

    A test modelling a dropped credit needs to pass a real `None`, which a plain
    `None` default would swallow.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<default>"


DEFAULT = _Default()


def touchdown_play(
    yards_gained: float,
    kind: str = "pass",
    season: int = 2025,
    week: int = 1,
    season_type: str = "REG",
    passer_player_id: str | None | _Default = DEFAULT,
    receiver_player_id: str | None | _Default = DEFAULT,
    rusher_player_id: str | None | _Default = DEFAULT,
    touchdown: int = 1,
) -> dict[str, Any]:
    """One scoring play, shaped like a play-by-play row.

    `kind` is "pass" or "rush" and sets the touchdown-type flags. Player IDs default
    to the ones the play type implies, so a test that cares only about yardage can
    omit them; pass an explicit `None` to model a missing ID.
    """
    if kind not in {"pass", "rush"}:
        raise ValueError(f"kind must be 'pass' or 'rush', got {kind!r}")

    is_pass = kind == "pass"

    def pick(given: str | None | _Default, fallback: str | None) -> str | None:
        return fallback if isinstance(given, _Default) else given

    if is_pass:
        passer = pick(passer_player_id, "00-0000QB1")
        receiver = pick(receiver_player_id, "00-0000WR1")
        rusher = pick(rusher_player_id, None)
    else:
        passer = pick(passer_player_id, None)
        receiver = pick(receiver_player_id, None)
        rusher = pick(rusher_player_id, "00-0000RB1")

    return {
        "season": season,
        "week": week,
        "season_type": season_type,
        "touchdown": touchdown,
        "yards_gained": float(yards_gained),
        "pass_touchdown": 1 if is_pass else 0,
        "rush_touchdown": 0 if is_pass else 1,
        "passer_player_id": passer,
        "receiver_player_id": receiver,
        "rusher_player_id": rusher,
    }


def pbp(*plays: dict[str, Any]) -> pl.DataFrame:
    """Assemble `touchdown_play` rows into a play-by-play frame."""
    schema = {
        "season": pl.Int64,
        "week": pl.Int64,
        "season_type": pl.String,
        "touchdown": pl.Int64,
        "yards_gained": pl.Float64,
        "pass_touchdown": pl.Int64,
        "rush_touchdown": pl.Int64,
        "passer_player_id": pl.String,
        "receiver_player_id": pl.String,
        "rusher_player_id": pl.String,
    }
    assert set(schema) == set(PBP_TOUCHDOWN_COLUMNS), "factory schema drifted from columns.py"
    return pl.DataFrame(list(plays), schema=schema)
