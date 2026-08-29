"""Typed view over league.yaml."""

from __future__ import annotations

import functools
from typing import Self

from pydantic import BaseModel, Field, model_validator

from patron.config.settings import load_league_config


class MetricsConfig(BaseModel):
    target_weight: float
    td_regress_down: float
    td_regress_up: float
    td_regress_up_min_opp: float
    rb_age_cliff: float
    floor_quantile: float
    td_rate_window: str = Field(pattern="^(all|season)$")
    projection_season_decay: float = Field(default=0.55, gt=0, le=1)
    projection_shrinkage_games: float = Field(default=8.0, gt=0)
    projection_td_regression: float = Field(default=0.5, ge=0, le=1)
    projection_bonus_retention: float = Field(default=0.35, ge=0, le=1)
    projection_team_context_cap: float = Field(default=0.18, ge=0, le=0.5)
    projection_component_weight: float = Field(default=0.65, ge=0, le=1)
    projection_mean_weight: float = Field(default=0.75, ge=0, le=1)
    projection_floor_weight: float = Field(default=0.15, ge=0, le=1)
    projection_ceiling_weight: float = Field(default=0.10, ge=0, le=1)
    projection_season_games: int = Field(default=17, ge=1, le=25)
    projection_availability_prior: float = Field(default=0.94, ge=0.5, le=1)
    projection_availability_shrinkage_games: float = Field(default=17.0, gt=0)
    projection_route_weight: float = Field(default=0.55, ge=0, le=1)
    projection_depth_chart_cap: float = Field(default=0.18, ge=0, le=0.5)

    @model_validator(mode="after")
    def score_weights_have_mass(self) -> Self:
        total = (
            self.projection_mean_weight
            + self.projection_floor_weight
            + self.projection_ceiling_weight
        )
        if total <= 0:
            raise ValueError("at least one v2 score weight must be greater than zero")
        return self


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
