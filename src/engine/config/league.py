"""Typed view over league.yaml."""

from __future__ import annotations

import functools

from pydantic import BaseModel, Field

from engine.config.settings import load_league_config


class MetricsConfig(BaseModel):
    rb_age_cliff: float
    floor_quantile: float
    projection_season_decay: float = Field(default=0.55, gt=0, le=1)
    projection_shrinkage_games: float = Field(default=8.0, gt=0)
    projection_component_weight: float = Field(default=0.65, ge=0, le=1)
    projection_season_games: int = Field(default=17, ge=1, le=25)
    projection_availability_prior: float = Field(default=0.94, ge=0.5, le=1)
    projection_availability_shrinkage_games: float = Field(default=17.0, gt=0)
    projection_route_weight: float = Field(default=0.55, ge=0, le=1)
    projection_depth_chart_cap: float = Field(default=0.18, ge=0, le=0.5)
    projection_prior_pool: str = Field(default="all", pattern="^(all|rosterable)$")
    # Per-position v2 sort key, chosen from the metric report's ranking table. Any
    # position not listed, or whose key is unavailable on the board, uses the fallback.
    projection_rank_key: dict[str, str] = Field(default_factory=dict)
    projection_rank_fallback: str = "adj_proj_vor"
    # Cross-position order: one key for every position, expressed as per-game VOR
    # against its own positional replacement (season-scale keys are divided by the
    # season length). Chosen by the metric report's OVERALL ranking test.
    projection_overall_key: str = "fitted_season_points"
    # Roster simulation inputs (League team strength, lineup scans). Every roster-level
    # number is built from these three per-player fields, so they must be the outputs
    # the backtest chose — not the hand-built v2 fields the ranking no longer uses.
    # Each entry lists fallbacks in order; the first column present on the board wins.
    roster_mean_columns: list[str] = Field(
        default_factory=lambda: ["fitted_ppg", "proj_ppg", "ppg"]
    )
    roster_games_columns: list[str] = Field(default_factory=lambda: ["fitted_games"])
    roster_availability_columns: list[str] = Field(
        default_factory=lambda: ["projected_availability"]
    )
    roster_volatility_columns: list[str] = Field(
        default_factory=lambda: ["projected_volatility", "volatility"]
    )


class LeagueConfig(BaseModel):
    # nflverse position corrections by GSIS id. Two-way players are filed at their
    # defensive position in the weekly stats and would otherwise never reach the board.
    position_overrides: dict[str, str] = Field(default_factory=dict)
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
