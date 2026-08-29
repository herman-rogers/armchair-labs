"""Typed view over scoring.yaml — the league's scoring rules as data.

Every number the engine scores with comes from here, so a league setting change is
a YAML edit rather than a code change. The bracket helpers below are the only place
that knows how a yardage bracket resolves to points.
"""

from __future__ import annotations

import functools

from pydantic import BaseModel

from patron.config.settings import load_scoring_config


class YardageBracket(BaseModel):
    """A yardage band worth a fixed number of points.

    `max_yards` of None means the band is open-ended at the top.
    """

    min_yards: int
    max_yards: int | None
    points: float

    def contains(self, yards: float) -> bool:
        if yards < self.min_yards:
            return False
        return self.max_yards is None or yards <= self.max_yards


class CountBracket(BaseModel):
    """A band over a count (points or yards allowed) worth a fixed number of points."""

    min: int
    max: int | None
    points: float

    def contains(self, value: float) -> bool:
        if value < self.min:
            return False
        return self.max is None or value <= self.max


def _resolve(brackets: list[YardageBracket] | list[CountBracket], value: float) -> float:
    """Points for the single bracket containing `value`, or 0.0 if none does.

    Brackets are exclusive by construction: a 55-yard touchdown scores the 50+ bonus
    only, never 40-49 as well. Returning on the first match is what enforces that.
    """
    for bracket in brackets:
        if bracket.contains(value):
            return bracket.points
    return 0.0


class PassingRules(BaseModel):
    yards_per_point: float
    touchdown: float
    interception: float
    two_point_conversion: float

    @property
    def points_per_yard(self) -> float:
        return 1.0 / self.yards_per_point


class RushingRules(BaseModel):
    points_per_yard: float
    touchdown: float
    two_point_conversion: float


class ReceivingRules(BaseModel):
    points_per_reception: float
    points_per_yard: float
    touchdown: float
    two_point_conversion: float


class MiscRules(BaseModel):
    fumble_lost: float
    special_teams_touchdown: float


class BonusRules(BaseModel):
    brackets: list[YardageBracket]

    def points_for(self, yards: float) -> float:
        """Big-play bonus for a touchdown of `yards`. Higher bracket only."""
        return _resolve(self.brackets, yards)


class KickingRules(BaseModel):
    field_goals: list[YardageBracket]
    extra_point: float
    missed_field_goal: float
    blocked_counts_as_miss: bool

    def field_goal_points(self, yards: float) -> float:
        return _resolve(self.field_goals, yards)


class DefenseRules(BaseModel):
    sack: float
    interception: float
    fumble_recovery: float
    touchdown: float
    safety: float
    points_allowed: list[CountBracket]
    yards_allowed: list[CountBracket]

    def points_allowed_points(self, points: float) -> float:
        return _resolve(self.points_allowed, points)

    def yards_allowed_points(self, yards: float) -> float:
        return _resolve(self.yards_allowed, yards)


class ScoringRules(BaseModel):
    passing: PassingRules
    rushing: RushingRules
    receiving: ReceivingRules
    misc: MiscRules
    bonuses: BonusRules
    kicking: KickingRules
    defense: DefenseRules


@functools.lru_cache(maxsize=1)
def get_rules() -> ScoringRules:
    return ScoringRules.model_validate(load_scoring_config())
