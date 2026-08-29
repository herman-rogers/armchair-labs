"""Reconciling nflverse and ESPN team abbreviations.

Two sources, two vocabularies, and they agree on thirty of thirty-two teams. The two
they do not agree on caused nineteen players to be reported as having changed teams
when they had not — a false signal precisely where a true one matters, since a real
team change is what makes a player's prior situational stats stale.

The map is deliberately tiny and explicit rather than a fuzzy match. `reconcile` exists
so a future rename fails a test instead of quietly resurrecting the phantom-trade bug.
"""

from __future__ import annotations

from typing import Final

#: nflverse abbreviation -> ESPN abbreviation, for the two that differ.
NFLVERSE_TO_ESPN: Final[dict[str, str]] = {
    "LA": "LAR",  # Rams: nflverse kept "LA" after the relocation, ESPN uses "LAR"
    "WAS": "WSH",  # Commanders
}

ESPN_TO_NFLVERSE: Final[dict[str, str]] = {
    espn: nflverse for nflverse, espn in NFLVERSE_TO_ESPN.items()
}

#: What ESPN puts in `proTeam` for a player on no roster. Not a team.
FREE_AGENT_MARKERS: Final[frozenset[str]] = frozenset({"None", "FA", "", "-"})


def to_espn(team: str | None) -> str | None:
    """Translate an nflverse abbreviation into ESPN's."""
    if team is None:
        return None
    return NFLVERSE_TO_ESPN.get(team, team)


def to_nflverse(team: str | None) -> str | None:
    """Translate an ESPN abbreviation into nflverse's, normalising the no-team markers."""
    if team is None or team in FREE_AGENT_MARKERS:
        return None
    return ESPN_TO_NFLVERSE.get(team, team)


def same_team(nflverse_team: str | None, espn_team: str | None) -> bool:
    """Whether two abbreviations refer to the same club.

    Unknowns are treated as "no disagreement": a missing team is not evidence of a
    trade, and flagging it as one would be a false positive on a real signal.
    """
    normalized = to_nflverse(espn_team)
    if nflverse_team is None or normalized is None:
        return True
    return nflverse_team == normalized


def reconcile(nflverse_teams: set[str], espn_teams: set[str]) -> tuple[set[str], set[str]]:
    """Team codes that fail to reconcile, in each direction.

    Returns `(nflverse_only, espn_only)`. Both empty means the vocabularies agree and
    the alias map above is complete.
    """
    translated = {mapped for team in nflverse_teams if (mapped := to_espn(team)) is not None}
    real_espn = {team for team in espn_teams if team not in FREE_AGENT_MARKERS}
    return translated - real_espn, real_espn - translated
