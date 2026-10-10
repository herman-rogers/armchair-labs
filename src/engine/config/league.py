"""Typed view over league.yaml."""

from __future__ import annotations

import functools

from pydantic import BaseModel, Field

from engine.config.settings import load_league_config


class LeagueConfig(BaseModel):
    # nflverse position corrections by GSIS id. Two-way players are filed at their
    # defensive position in the weekly stats and would otherwise never reach the board.
    position_overrides: dict[str, str] = Field(default_factory=dict)
    name: str
    team_count: int
    seasons: list[int]
    board_season: int
    draft_season: int
    board_positions: list[str]


@functools.lru_cache(maxsize=1)
def get_league() -> LeagueConfig:
    return LeagueConfig.model_validate(load_league_config())
