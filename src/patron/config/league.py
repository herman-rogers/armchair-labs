"""Typed view over league.yaml."""

from __future__ import annotations

import functools

from pydantic import BaseModel, Field

from patron.config.settings import load_league_config


class MetricsConfig(BaseModel):
    target_weight: float
    td_regress_down: float
    td_regress_up: float
    td_regress_up_min_opp: float
    rb_age_cliff: float
    floor_quantile: float
    td_rate_window: str = Field(pattern="^(all|season)$")


class LeagueConfig(BaseModel):
    name: str
    team_count: int
    seasons: list[int]
    board_season: int
    draft_season: int
    vor_baseline_rank: dict[str, int]
    board_positions: list[str]
    min_games_baseline: int
    min_games_board: int
    small_sample_games: int
    metrics: MetricsConfig


@functools.lru_cache(maxsize=1)
def get_league() -> LeagueConfig:
    return LeagueConfig.model_validate(load_league_config())
