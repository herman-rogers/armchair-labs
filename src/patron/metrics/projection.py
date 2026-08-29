"""v2 forward projections with explicit player uncertainty and team context.

v1 answers a retrospective question: how many league points per game did the player
score, and how far was that above positional replacement?  This module deliberately
answers a different question: what is a reasonable next-season PPG estimate?

The projection is intentionally transparent rather than presented as a trained black
box.  It combines a recency-weighted multi-year individual prior, small-sample
shrinkage, partial touchdown and big-play regression, a position-specific age curve,
and a team-context ratio.  Team context compares the latest projected environment to
the environments already embedded in the player's history.  Comparing contexts is
important: adding a raw "good quarterback" bonus would count the same quarterback
twice for a player who never changed teams.

The historical data cannot know a future trade, quarterback change, or depth-chart
promotion.  ``config/projections.yaml`` carries those assumptions, visibly and with
reasons, in the same spirit as the v1 injury overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from patron.config.league import LeagueConfig
from patron.config.settings import load_projections_config
from patron.metrics.vor import add_vor, replacement_levels
from patron.names import normalize

PROJECTED_PPG = "proj_ppg"
PROJECTED_VOR = "proj_vor"
ADJUSTED_PROJECTED_VOR = "adj_proj_vor"
METRIC_VERSION = "metric_version"


class UnmatchedProjectionError(ValueError):
    """A configured future player assumption matched no v1 board row."""

    def __init__(self, entries: list[PlayerProjectionOverride]) -> None:
        listing = ", ".join(f"{entry.player} ({entry.position})" for entry in entries)
        super().__init__(
            f"projection assumption(s) matched no player: {listing}. "
            "Fix the name/position in config/projections.yaml; silently ignoring a "
            "trade or role change would invalidate the forward rank."
        )


def _number(value: Any) -> float:
    return float(value or 0.0)


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ratio(current: float, historical: float) -> float:
    """A guarded context ratio; sparse team rows must not create huge projections."""
    if current <= 0 or historical <= 0:
        return 1.0
    return _bounded(current / historical, 0.75, 1.25)


@dataclass(frozen=True)
class PlayerProjectionOverride:
    player: str
    position: str
    projected_team: str | None = None
    opportunity_multiplier: float = 1.0
    reason: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return normalize(self.player), self.position.upper()


@dataclass(frozen=True)
class TeamProjectionOverride:
    team: str
    qb_context_multiplier: float = 1.0
    pass_volume_multiplier: float = 1.0
    rush_volume_multiplier: float = 1.0
    scoring_multiplier: float = 1.0
    target_availability_multiplier: float = 1.0
    backfield_availability_multiplier: float = 1.0
    reason: str | None = None


@dataclass(frozen=True)
class ProjectionAssumptions:
    players: dict[tuple[str, str], PlayerProjectionOverride] = field(default_factory=dict)
    teams: dict[str, TeamProjectionOverride] = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None = None) -> ProjectionAssumptions:
        raw = raw if raw is not None else load_projections_config()
        players = [PlayerProjectionOverride(**entry) for entry in raw.get("players") or []]
        teams = [
            TeamProjectionOverride(team=team.upper(), **(values or {}))
            for team, values in (raw.get("teams") or {}).items()
        ]
        return cls(
            players={entry.key: entry for entry in players},
            teams={entry.team: entry for entry in teams},
        )


@dataclass
class TeamProfile:
    games: float = 0.0
    qb_context: float = 0.0
    pass_volume: float = 0.0
    rush_volume: float = 0.0
    scoring: float = 0.0
    pass_td_rate: float = 0.0
    rush_td_rate: float = 0.0
    targets: dict[str, float] = field(default_factory=dict)
    rb_opportunity: dict[str, float] = field(default_factory=dict)

    def target_availability(self, player_id: str) -> float:
        """Share left after the largest teammate target earner is accounted for."""
        total = sum(self.targets.values())
        other = max((value for key, value in self.targets.items() if key != player_id), default=0.0)
        return 1.0 - other / total if total > 0 else 1.0

    def backfield_availability(self, player_id: str) -> float:
        total = sum(self.rb_opportunity.values())
        other = max(
            (value for key, value in self.rb_opportunity.items() if key != player_id),
            default=0.0,
        )
        return 1.0 - other / total if total > 0 else 1.0


def _team_profiles(seasons: pl.DataFrame) -> dict[tuple[int, str], TeamProfile]:
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for row in seasons.iter_rows(named=True):
        team = str(row.get("team") or "").upper()
        if not team:
            continue
        key = (int(row["season"]), team)
        bucket = grouped.setdefault(
            key,
            {
                "games": 0.0,
                "passing_yards": 0.0,
                "passing_tds": 0.0,
                "passing_interceptions": 0.0,
                "rushing_tds": 0.0,
                "targets": {},
                "carries": 0.0,
                "rb_opportunity": {},
            },
        )
        bucket["games"] = max(bucket["games"], _number(row.get("games")))
        for column in (
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "rushing_tds",
            "carries",
        ):
            bucket[column] += _number(row.get(column))

        player_id = str(row["player_id"])
        bucket["targets"][player_id] = bucket["targets"].get(player_id, 0.0) + _number(
            row.get("targets")
        )
        if row.get("position") == "RB":
            bucket["rb_opportunity"][player_id] = bucket["rb_opportunity"].get(
                player_id, 0.0
            ) + _number(row.get("wtd_opp"))

    profiles: dict[tuple[int, str], TeamProfile] = {}
    for key, values in grouped.items():
        games = values["games"] or 1.0
        profiles[key] = TeamProfile(
            games=games,
            qb_context=(
                0.04 * values["passing_yards"]
                + 4.0 * values["passing_tds"]
                - 2.0 * values["passing_interceptions"]
            )
            / games,
            pass_volume=sum(values["targets"].values()) / games,
            rush_volume=values["carries"] / games,
            scoring=(values["passing_tds"] + values["rushing_tds"]) / games,
            pass_td_rate=values["passing_tds"] / max(sum(values["targets"].values()), 1.0),
            rush_td_rate=values["rushing_tds"] / max(values["carries"], 1.0),
            targets=values["targets"],
            rb_opportunity=values["rb_opportunity"],
        )
    return profiles


def _age_factor(position: str, age: float | None) -> float:
    """Conservative position-specific heuristic, bounded until it is backtested."""
    if age is None:
        return 1.0
    if position == "RB":
        factor = 1.02 if age < 24 else 1.0 - max(age - 26.0, 0.0) * 0.035
    elif position == "WR":
        factor = 1.01 if age < 24 else 1.0 - max(age - 29.0, 0.0) * 0.02
    elif position == "TE":
        factor = 1.01 if age < 25 else 1.0 - max(age - 30.0, 0.0) * 0.018
    else:  # QB
        factor = 1.0 - max(age - 35.0, 0.0) * 0.015
    return _bounded(factor, 0.78, 1.03)


def _rosterable_rows(
    seasons: pl.DataFrame,
    config: LeagueConfig,
) -> dict[str, pl.DataFrame]:
    pools: dict[str, pl.DataFrame] = {}
    for position, baseline_rank in config.vor_baseline_rank.items():
        pools[position] = (
            seasons.filter(
                (pl.col("position") == position) & (pl.col("games") >= config.min_games_baseline)
            )
            .sort(["season", "ppg"], descending=[True, True])
            .group_by("season", maintain_order=True)
            .head(baseline_rank * 2)
        )
    return pools


def _position_means(
    pools: dict[str, pl.DataFrame],
) -> tuple[dict[str, float], dict[str, float]]:
    """Rosterable-player PPG and bonus/game priors by position."""
    ppg: dict[str, float] = {}
    bonus: dict[str, float] = {}
    for position, rows in pools.items():
        if rows.height == 0:
            ppg[position] = 0.0
            bonus[position] = 0.0
            continue
        weighted_games = rows["games"].cast(pl.Float64)
        game_total = float(weighted_games.sum())
        ppg[position] = float((rows["ppg"] * weighted_games).sum()) / game_total
        bonus[position] = float(rows["bonus_pts"].sum()) / game_total
    return ppg, bonus


def _actual_core_points(row: dict[str, Any]) -> float:
    """Scored components v2 can project explicitly; residual keeps rare events."""
    return (
        0.04 * _number(row.get("passing_yards"))
        + 4.0 * _number(row.get("passing_tds"))
        - 2.0 * _number(row.get("passing_interceptions"))
        + 0.1 * _number(row.get("rushing_yards"))
        + 6.0 * _number(row.get("rushing_tds"))
        + _number(row.get("receptions"))
        + 0.1 * _number(row.get("receiving_yards"))
        + 6.0 * _number(row.get("receiving_tds"))
        + _number(row.get("bonus_pts"))
    )


def _component_priors(
    pools: dict[str, pl.DataFrame],
    profiles: dict[tuple[int, str], TeamProfile],
    config: LeagueConfig,
) -> dict[str, dict[str, float]]:
    """Position anchors used to shrink role, efficiency, and distribution metrics."""
    priors: dict[str, dict[str, float]] = {}
    for position, frame in pools.items():
        totals: dict[str, float] = {
            "games": 0.0,
            "targets": 0.0,
            "carries": 0.0,
            "receptions": 0.0,
            "receiving_yards": 0.0,
            "rushing_yards": 0.0,
            "receiving_tds": 0.0,
            "rushing_tds": 0.0,
            "bonus_pts": 0.0,
            "passing_yards": 0.0,
            "passing_tds": 0.0,
            "passing_interceptions": 0.0,
            "target_share": 0.0,
            "carry_share": 0.0,
            "wopr": 0.0,
            "air_yards_share": 0.0,
            "floor_ratio": 0.0,
            "volatility_ratio": 0.0,
            "residual_points": 0.0,
        }
        for row in frame.iter_rows(named=True):
            decay = config.metrics.projection_season_decay ** max(
                config.board_season - int(row["season"]), 0
            )
            games = max(_number(row.get("games")), 1.0)
            game_weight = games * decay
            totals["games"] += game_weight
            for column in (
                "targets",
                "carries",
                "receptions",
                "receiving_yards",
                "rushing_yards",
                "receiving_tds",
                "rushing_tds",
                "bonus_pts",
                "passing_yards",
                "passing_tds",
                "passing_interceptions",
            ):
                totals[column] += _number(row.get(column)) * decay

            profile = profiles.get((int(row["season"]), str(row.get("team") or "").upper()))
            team_targets = sum(profile.targets.values()) if profile else 0.0
            team_carries = profile.rush_volume * profile.games if profile else 0.0
            totals["target_share"] += (
                _number(row.get("targets")) / team_targets if team_targets else 0.0
            ) * game_weight
            totals["carry_share"] += (
                _number(row.get("carries")) / team_carries if team_carries else 0.0
            ) * game_weight
            totals["wopr"] += _number(row.get("wopr")) * game_weight
            totals["air_yards_share"] += _number(row.get("air_yards_share")) * game_weight
            ppg = _number(row.get("ppg"))
            if ppg > 0:
                totals["floor_ratio"] += _number(row.get("floor")) / ppg * game_weight
                totals["volatility_ratio"] += _number(row.get("volatility")) / ppg * game_weight
            if row.get("season_pts") is not None:
                totals["residual_points"] += (
                    _number(row.get("season_pts")) - _actual_core_points(row)
                ) * decay

        games = max(totals["games"], 1.0)
        targets = max(totals["targets"], 1.0)
        carries = max(totals["carries"], 1.0)
        opportunity = max(totals["targets"] + totals["carries"], 1.0)
        priors[position] = {
            "target_share": totals["target_share"] / games,
            "carry_share": totals["carry_share"] / games,
            "wopr": totals["wopr"] / games,
            "air_yards_share": totals["air_yards_share"] / games,
            "catch_rate": totals["receptions"] / targets,
            "receiving_yards_per_target": totals["receiving_yards"] / targets,
            "rushing_yards_per_carry": totals["rushing_yards"] / carries,
            "receiving_td_rate": totals["receiving_tds"] / targets,
            "rushing_td_rate": totals["rushing_tds"] / carries,
            "bonus_per_opportunity": totals["bonus_pts"] / opportunity,
            "bonus_pg": totals["bonus_pts"] / games,
            "passing_yards_pg": totals["passing_yards"] / games,
            "passing_tds_pg": totals["passing_tds"] / games,
            "passing_interceptions_pg": totals["passing_interceptions"] / games,
            "rushing_yards_pg": totals["rushing_yards"] / games,
            "rushing_tds_pg": totals["rushing_tds"] / games,
            "floor_ratio": _bounded(totals["floor_ratio"] / games, 0.0, 1.0),
            "volatility_ratio": max(totals["volatility_ratio"] / games, 0.0),
            "residual_pg": totals["residual_points"] / games,
        }
    return priors


def _apply_team_override(
    profile: TeamProfile,
    override: TeamProjectionOverride | None,
) -> dict[str, float]:
    return {
        "qb_context": profile.qb_context * (override.qb_context_multiplier if override else 1.0),
        "pass_volume": profile.pass_volume * (override.pass_volume_multiplier if override else 1.0),
        "rush_volume": profile.rush_volume * (override.rush_volume_multiplier if override else 1.0),
        "scoring": profile.scoring * (override.scoring_multiplier if override else 1.0),
        "pass_td_rate": profile.pass_td_rate * (override.scoring_multiplier if override else 1.0),
        "rush_td_rate": profile.rush_td_rate * (override.scoring_multiplier if override else 1.0),
        "target_availability_multiplier": (
            override.target_availability_multiplier if override else 1.0
        ),
        "backfield_availability_multiplier": (
            override.backfield_availability_multiplier if override else 1.0
        ),
    }


def build_projection_board(
    player_seasons: pl.DataFrame,
    v1_board: pl.DataFrame,
    config: LeagueConfig,
    assumptions: ProjectionAssumptions | None = None,
) -> pl.DataFrame:
    """Build the v2 board while retaining every v1 metric as supporting evidence."""
    assumptions = assumptions or ProjectionAssumptions.from_config()
    metrics = config.metrics
    profiles = _team_profiles(player_seasons)
    pools = _rosterable_rows(player_seasons, config)
    position_ppg, position_bonus = _position_means(pools)
    component_priors = _component_priors(pools, profiles, config)

    board_keys = {
        (normalize(str(row["player_display_name"])), str(row["position"]).upper())
        for row in v1_board.iter_rows(named=True)
    }
    unmatched = [entry for key, entry in assumptions.players.items() if key not in board_keys]
    if unmatched:
        raise UnmatchedProjectionError(unmatched)

    histories: dict[str, list[dict[str, Any]]] = {}
    for row in player_seasons.iter_rows(named=True):
        histories.setdefault(str(row["player_id"]), []).append(row)

    projected: list[dict[str, Any]] = []
    for base in v1_board.iter_rows(named=True):
        player_id = str(base["player_id"])
        position = str(base["position"])
        history = histories.get(player_id, [base])
        override = assumptions.players.get((normalize(str(base["player_display_name"])), position))
        projected_team = str(
            (override.projected_team if override and override.projected_team else base["team"])
            or ""
        ).upper()

        weighted: list[tuple[dict[str, Any], float]] = []
        for row in history:
            years_old = max(config.board_season - int(row["season"]), 0)
            weight = metrics.projection_season_decay**years_old * max(
                _number(row.get("games")), 1.0
            )
            weighted.append((row, weight))
        total_weight = sum(weight for _, weight in weighted) or 1.0
        historical_ppg = (
            sum(_number(row.get("ppg")) * weight for row, weight in weighted) / total_weight
        )
        effective_games = sum(
            metrics.projection_season_decay ** max(config.board_season - int(row["season"]), 0)
            * _number(row.get("games"))
            for row in history
        )
        reliability = effective_games / (effective_games + metrics.projection_shrinkage_games)
        individual_prior = reliability * historical_ppg + (1.0 - reliability) * position_ppg.get(
            position, historical_ppg
        )

        latest = max(history, key=lambda row: int(row["season"]))
        latest_games = max(_number(latest.get("games")), 1.0)
        td_adjustment = 0.0
        if position != "QB":
            td_adjustment = (
                -_number(latest.get("td_over_exp"))
                * 6.0
                / latest_games
                * metrics.projection_td_regression
            )
            td_adjustment = _bounded(td_adjustment, -2.5, 2.5)

        historical_bonus_pg = (
            sum(
                (_number(row.get("bonus_pts")) / max(_number(row.get("games")), 1.0)) * weight
                for row, weight in weighted
            )
            / total_weight
        )
        bonus_adjustment = (position_bonus.get(position, 0.0) - historical_bonus_pg) * (
            1.0 - metrics.projection_bonus_retention
        )
        bonus_adjustment = _bounded(bonus_adjustment, -1.5, 1.5)

        source_context = {
            "qb_context": 0.0,
            "pass_volume": 0.0,
            "rush_volume": 0.0,
            "scoring": 0.0,
            "pass_td_rate": 0.0,
            "rush_td_rate": 0.0,
            "target_availability": 0.0,
            "backfield_availability": 0.0,
        }
        context_weight = 0.0
        for row, weight in weighted:
            profile = profiles.get((int(row["season"]), str(row.get("team") or "").upper()))
            if profile is None:
                continue
            context_weight += weight
            source_context["qb_context"] += profile.qb_context * weight
            source_context["pass_volume"] += profile.pass_volume * weight
            source_context["rush_volume"] += profile.rush_volume * weight
            source_context["scoring"] += profile.scoring * weight
            source_context["pass_td_rate"] += profile.pass_td_rate * weight
            source_context["rush_td_rate"] += profile.rush_td_rate * weight
            source_context["target_availability"] += profile.target_availability(player_id) * weight
            source_context["backfield_availability"] += (
                profile.backfield_availability(player_id) * weight
            )
        if context_weight:
            source_context = {key: value / context_weight for key, value in source_context.items()}

        target_profile = profiles.get((config.board_season, projected_team))
        target_team_override = assumptions.teams.get(projected_team)
        target_profile_known = target_profile is not None
        if target_profile is None:
            # Unknown future teams/rookies should not manufacture a context effect.
            target_profile = TeamProfile(
                qb_context=source_context["qb_context"],
                pass_volume=source_context["pass_volume"],
                rush_volume=source_context["rush_volume"],
                scoring=source_context["scoring"],
                pass_td_rate=source_context["pass_td_rate"],
                rush_td_rate=source_context["rush_td_rate"],
            )
        target = _apply_team_override(target_profile, target_team_override)
        target_availability = (
            target_profile.target_availability(player_id)
            if target_profile_known
            else source_context["target_availability"]
        ) * target["target_availability_multiplier"]
        backfield_availability = (
            target_profile.backfield_availability(player_id)
            if target_profile_known
            else source_context["backfield_availability"]
        ) * target["backfield_availability_multiplier"]
        latest_source_profile = profiles.get(
            (config.board_season, str(base.get("team") or "").upper())
        )
        latest_team_targets = (
            sum(latest_source_profile.targets.values()) if latest_source_profile else 0.0
        )
        latest_team_carries = (
            latest_source_profile.rush_volume * latest_source_profile.games
            if latest_source_profile
            else 0.0
        )
        season_target_share = (
            _number(latest.get("targets")) / latest_team_targets if latest_team_targets else None
        )
        season_carry_share = (
            _number(latest.get("carries")) / latest_team_carries if latest_team_carries else None
        )

        if position in {"WR", "TE"}:
            context_delta = (
                0.40 * (_ratio(target["qb_context"], source_context["qb_context"]) - 1.0)
                + 0.20 * (_ratio(target["pass_volume"], source_context["pass_volume"]) - 1.0)
                + 0.20 * (_ratio(target["scoring"], source_context["scoring"]) - 1.0)
                + 0.20 * (_ratio(target_availability, source_context["target_availability"]) - 1.0)
            )
        elif position == "RB":
            context_delta = (
                0.30 * (_ratio(target["scoring"], source_context["scoring"]) - 1.0)
                + 0.25 * (_ratio(target["rush_volume"], source_context["rush_volume"]) - 1.0)
                + 0.15 * (_ratio(target["pass_volume"], source_context["pass_volume"]) - 1.0)
                + 0.30
                * (_ratio(backfield_availability, source_context["backfield_availability"]) - 1.0)
            )
        else:  # QB: pass volume and team scoring stand in for supporting environment.
            context_delta = 0.50 * (
                _ratio(target["pass_volume"], source_context["pass_volume"]) - 1.0
            ) + 0.50 * (_ratio(target["scoring"], source_context["scoring"]) - 1.0)

        cap = metrics.projection_team_context_cap
        team_context_factor = 1.0 + _bounded(context_delta, -cap, cap)
        role_multiplier = override.opportunity_multiplier if override else 1.0
        age = base.get("age_at_season")
        age_factor = _age_factor(position, float(age) if age is not None else None)
        prior_projection = max(individual_prior + td_adjustment + bonus_adjustment, 0.0)
        prior_projection *= age_factor * team_context_factor * role_multiplier

        # The component branch turns role and team volume into a projected stat line.
        # Totals use season decay without multiplying by games a second time.
        def weighted_total(
            column: str,
            rows: list[tuple[dict[str, Any], float]] = weighted,
        ) -> float:
            return sum(
                _number(row.get(column)) * weight / max(_number(row.get("games")), 1.0)
                for row, weight in rows
            )

        def weighted_metric(
            column: str,
            rows: list[tuple[dict[str, Any], float]] = weighted,
            denominator: float = total_weight,
        ) -> float:
            return sum(_number(row.get(column)) * weight for row, weight in rows) / denominator

        prior = component_priors.get(position, {})
        weighted_targets = weighted_total("targets")
        weighted_carries = weighted_total("carries")
        weighted_receptions = weighted_total("receptions")
        weighted_receiving_yards = weighted_total("receiving_yards")
        weighted_rushing_yards = weighted_total("rushing_yards")
        weighted_receiving_tds = weighted_total("receiving_tds")
        weighted_rushing_tds = weighted_total("rushing_tds")
        weighted_bonus = weighted_total("bonus_pts")
        weighted_opportunity = weighted_total("wtd_opp")

        historical_target_share = 0.0
        historical_carry_share = 0.0
        share_weight = 0.0
        for row, weight in weighted:
            profile = profiles.get((int(row["season"]), str(row.get("team") or "").upper()))
            if profile is None:
                continue
            team_targets = sum(profile.targets.values())
            team_carries = profile.rush_volume * profile.games
            historical_target_share += (
                _number(row.get("targets")) / team_targets if team_targets else 0.0
            ) * weight
            historical_carry_share += (
                _number(row.get("carries")) / team_carries if team_carries else 0.0
            ) * weight
            share_weight += weight
        if share_weight:
            historical_target_share /= share_weight
            historical_carry_share /= share_weight

        target_reliability = weighted_targets / (weighted_targets + 40.0)
        carry_reliability = weighted_carries / (weighted_carries + 60.0)
        projected_target_share = target_reliability * historical_target_share + (
            1.0 - target_reliability
        ) * prior.get("target_share", historical_target_share)
        projected_carry_share = carry_reliability * historical_carry_share + (
            1.0 - carry_reliability
        ) * prior.get("carry_share", historical_carry_share)

        historical_wopr = weighted_metric("wopr")
        projected_wopr = target_reliability * historical_wopr + (
            1.0 - target_reliability
        ) * prior.get("wopr", historical_wopr)
        prior_wopr = prior.get("wopr", 0.0)
        wopr_equivalent_share = (
            projected_wopr * prior.get("target_share", projected_target_share) / prior_wopr
            if prior_wopr > 0
            else projected_target_share
        )
        # WOPR contributes air-yard role information, but only as one quarter of the
        # role estimate so target share is not counted twice at full strength.
        projected_target_share = 0.75 * projected_target_share + 0.25 * wopr_equivalent_share
        projected_target_share *= (
            _ratio(target_availability, source_context["target_availability"]) ** 0.35
        )
        projected_carry_share *= (
            _ratio(backfield_availability, source_context["backfield_availability"]) ** 0.40
        )
        projected_target_share = _bounded(projected_target_share, 0.0, 0.45)
        projected_carry_share = _bounded(projected_carry_share, 0.0, 0.85)

        pass_volume_ratio = _ratio(target["pass_volume"], source_context["pass_volume"])
        rush_volume_ratio = _ratio(target["rush_volume"], source_context["rush_volume"])
        team_target_projection = projected_target_share * target["pass_volume"]
        team_carry_projection = projected_carry_share * target["rush_volume"]
        historical_targets_pg = weighted_targets / max(effective_games, 1.0)
        historical_carries_pg = weighted_carries / max(effective_games, 1.0)
        # Raw weighted opportunity supplies a third, independent role anchor for
        # receivers; converting it back to targets keeps the units interpretable.
        opportunity_implied_targets = max(
            (weighted_opportunity - weighted_carries) / 2.2 / max(effective_games, 1.0),
            0.0,
        )

        if position == "QB":
            projected_targets = 0.0
            projected_carries = (
                reliability * historical_carries_pg
                + (1.0 - reliability) * prior.get("rushing_yards_pg", 0.0) / 4.5
            )
            projected_carries *= role_multiplier
        else:
            projected_targets = (
                0.65 * team_target_projection
                + 0.20 * historical_targets_pg * pass_volume_ratio
                + 0.15 * opportunity_implied_targets * pass_volume_ratio
            ) * role_multiplier
            projected_carries = (
                0.75 * team_carry_projection + 0.25 * historical_carries_pg * rush_volume_ratio
            ) * role_multiplier

        catch_reliability = weighted_targets / (weighted_targets + 80.0)
        carry_efficiency_reliability = weighted_carries / (weighted_carries + 120.0)
        historical_catch_rate = weighted_receptions / max(weighted_targets, 1.0)
        historical_ypt = weighted_receiving_yards / max(weighted_targets, 1.0)
        historical_ypc = weighted_rushing_yards / max(weighted_carries, 1.0)
        projected_catch_rate = catch_reliability * historical_catch_rate + (
            1.0 - catch_reliability
        ) * prior.get("catch_rate", historical_catch_rate)
        projected_ypt = catch_reliability * historical_ypt + (1.0 - catch_reliability) * prior.get(
            "receiving_yards_per_target", historical_ypt
        )
        projected_ypc = carry_efficiency_reliability * historical_ypc + (
            1.0 - carry_efficiency_reliability
        ) * prior.get("rushing_yards_per_carry", historical_ypc)

        historical_air_share = weighted_metric("air_yards_share")
        projected_air_share = target_reliability * historical_air_share + (
            1.0 - target_reliability
        ) * prior.get("air_yards_share", historical_air_share)
        player_depth = projected_air_share / max(projected_target_share, 0.01)
        position_depth = prior.get("air_yards_share", 0.0) / max(
            prior.get("target_share", 0.0), 0.01
        )
        air_yard_factor = _bounded(_ratio(player_depth, position_depth) ** 0.25, 0.90, 1.10)
        projected_ypt *= air_yard_factor

        receiving_td_reliability = weighted_targets / (weighted_targets + 120.0)
        rushing_td_reliability = weighted_carries / (weighted_carries + 140.0)
        historical_receiving_td_rate = weighted_receiving_tds / max(weighted_targets, 1.0)
        historical_rushing_td_rate = weighted_rushing_tds / max(weighted_carries, 1.0)
        projected_receiving_td_rate = (
            receiving_td_reliability * historical_receiving_td_rate
            + (1.0 - receiving_td_reliability)
            * prior.get("receiving_td_rate", historical_receiving_td_rate)
        ) * _ratio(target["pass_td_rate"], source_context["pass_td_rate"])
        projected_rushing_td_rate = (
            rushing_td_reliability * historical_rushing_td_rate
            + (1.0 - rushing_td_reliability)
            * prior.get("rushing_td_rate", historical_rushing_td_rate)
        ) * _ratio(target["rush_td_rate"], source_context["rush_td_rate"])

        projected_receptions = projected_targets * projected_catch_rate
        projected_receiving_yards = projected_targets * projected_ypt
        projected_rushing_yards = projected_carries * projected_ypc
        projected_receiving_tds = projected_targets * projected_receiving_td_rate
        projected_rushing_tds = projected_carries * projected_rushing_td_rate

        opportunity = weighted_targets + weighted_carries
        bonus_reliability = opportunity / (opportunity + 120.0)
        historical_bonus_rate = weighted_bonus / max(opportunity, 1.0)
        projected_bonus_rate = bonus_reliability * historical_bonus_rate + (
            1.0 - bonus_reliability
        ) * prior.get("bonus_per_opportunity", historical_bonus_rate)
        projected_bonus_pg = (
            (projected_targets + projected_carries) * projected_bonus_rate * air_yard_factor
        )

        historical_passing_yards_pg = weighted_total("passing_yards") / max(effective_games, 1.0)
        historical_passing_tds_pg = weighted_total("passing_tds") / max(effective_games, 1.0)
        historical_interceptions_pg = weighted_total("passing_interceptions") / max(
            effective_games, 1.0
        )
        projected_passing_yards = (
            reliability * historical_passing_yards_pg
            + (1.0 - reliability) * prior.get("passing_yards_pg", 0.0)
        ) * (1.0 + 0.50 * (pass_volume_ratio - 1.0))
        projected_passing_tds = (
            reliability * historical_passing_tds_pg
            + (1.0 - reliability) * prior.get("passing_tds_pg", 0.0)
        ) * _ratio(target["scoring"], source_context["scoring"])
        projected_interceptions = reliability * historical_interceptions_pg + (
            1.0 - reliability
        ) * prior.get("passing_interceptions_pg", 0.0)
        if position != "QB":
            projected_passing_yards = 0.0
            projected_passing_tds = 0.0
            projected_interceptions = 0.0
        else:
            projected_bonus_pg = reliability * historical_bonus_pg + (
                1.0 - reliability
            ) * prior.get("bonus_pg", historical_bonus_pg)

        historical_residual_points = sum(
            (
                _number(row.get("season_pts")) - _actual_core_points(row)
                if row.get("season_pts") is not None
                else 0.0
            )
            * weight
            / max(_number(row.get("games")), 1.0)
            for row, weight in weighted
        )
        historical_residual_pg = historical_residual_points / max(effective_games, 1.0)
        projected_residual_pg = reliability * historical_residual_pg + (
            1.0 - reliability
        ) * prior.get("residual_pg", 0.0)

        component_projection = (
            0.04 * projected_passing_yards
            + 4.0 * projected_passing_tds
            - 2.0 * projected_interceptions
            + 0.1 * projected_rushing_yards
            + 6.0 * projected_rushing_tds
            + projected_receptions
            + 0.1 * projected_receiving_yards
            + 6.0 * projected_receiving_tds
            + projected_bonus_pg
            + projected_residual_pg
        ) * age_factor
        if position == "QB":
            component_projection *= role_multiplier

        component_weight = metrics.projection_component_weight
        proj_ppg = (1.0 - component_weight) * prior_projection + component_weight * max(
            component_projection, 0.0
        )

        historical_floor_ratio = weighted_metric("floor") / max(historical_ppg, 0.1)
        historical_volatility_ratio = weighted_metric("volatility") / max(historical_ppg, 0.1)
        projected_floor_ratio = reliability * historical_floor_ratio + (
            1.0 - reliability
        ) * prior.get("floor_ratio", historical_floor_ratio)
        projected_volatility_ratio = reliability * historical_volatility_ratio + (
            1.0 - reliability
        ) * prior.get("volatility_ratio", historical_volatility_ratio)
        projected_floor = proj_ppg * _bounded(projected_floor_ratio, 0.0, 1.0)
        projected_volatility = proj_ppg * _bounded(projected_volatility_ratio, 0.0, 1.5)
        projected_ceiling = proj_ppg + 0.674 * projected_volatility

        moved = projected_team != str(base.get("team") or "").upper()
        confidence = reliability * (0.88 if moved else 1.0)
        if override and override.opportunity_multiplier != 1.0:
            confidence *= 0.92

        reasons = [
            value
            for value in (
                override.reason if override else None,
                target_team_override.reason if target_team_override else None,
            )
            if value
        ]
        projected.append(
            {
                **base,
                METRIC_VERSION: "v2",
                "projected_team": projected_team,
                "historical_ppg_prior": historical_ppg,
                "individual_prior_ppg": individual_prior,
                "prior_branch_ppg": prior_projection,
                "component_proj_ppg": component_projection,
                PROJECTED_PPG: proj_ppg,
                "projected_floor": projected_floor,
                "projected_ceiling": projected_ceiling,
                "projected_volatility": projected_volatility,
                "projection_confidence": _bounded(confidence, 0.0, 1.0),
                "effective_games": effective_games,
                "age_factor": age_factor,
                "td_regression_adjustment": td_adjustment,
                "bonus_regression_adjustment": bonus_adjustment,
                "team_context_factor": team_context_factor,
                "qb_context": target["qb_context"],
                "team_scoring_context": target["scoring"],
                "team_pass_volume": target["pass_volume"],
                "team_rush_volume": target["rush_volume"],
                "teammate_competition": 1.0
                - (backfield_availability if position == "RB" else target_availability),
                "season_target_share": season_target_share,
                "season_carry_share": season_carry_share,
                "projected_target_share": projected_target_share,
                "projected_carry_share": projected_carry_share,
                "projected_wopr": projected_wopr,
                "projected_air_yards_share": projected_air_share,
                "projected_targets_pg": projected_targets,
                "projected_carries_pg": projected_carries,
                "projected_receptions_pg": projected_receptions,
                "projected_receiving_yards_pg": projected_receiving_yards,
                "projected_rushing_yards_pg": projected_rushing_yards,
                "projected_receiving_tds_pg": projected_receiving_tds,
                "projected_rushing_tds_pg": projected_rushing_tds,
                "projected_passing_yards_pg": projected_passing_yards,
                "projected_passing_tds_pg": projected_passing_tds,
                "projected_interceptions_pg": projected_interceptions,
                "projected_bonus_pg": projected_bonus_pg,
                "air_yard_factor": air_yard_factor,
                "opportunity_multiplier": role_multiplier,
                "projection_reason": " | ".join(reasons) or None,
            }
        )

    frame = pl.DataFrame(projected, infer_schema_length=None).rename(
        {"vor": "historical_vor", "repl_ppg": "historical_repl_ppg"}
    )
    levels = replacement_levels(
        frame,
        baseline_ranks=config.vor_baseline_rank,
        min_games=config.min_games_baseline,
        ppg_column=PROJECTED_PPG,
    )
    frame = add_vor(frame, levels, ppg_column=PROJECTED_PPG).rename(
        {"repl_ppg": "proj_repl_ppg", "vor": PROJECTED_VOR}
    )
    floor_levels = replacement_levels(
        frame,
        baseline_ranks=config.vor_baseline_rank,
        min_games=config.min_games_baseline,
        ppg_column="projected_floor",
    )
    ceiling_levels = replacement_levels(
        frame,
        baseline_ranks=config.vor_baseline_rank,
        min_games=config.min_games_baseline,
        ppg_column="projected_ceiling",
    )
    frame = frame.with_columns(
        pl.col("position")
        .replace_strict(floor_levels, default=None, return_dtype=pl.Float64)
        .alias("floor_repl"),
        pl.col("position")
        .replace_strict(ceiling_levels, default=None, return_dtype=pl.Float64)
        .alias("ceiling_repl"),
    ).with_columns(
        (pl.col("projected_floor") - pl.col("floor_repl")).alias("floor_vor"),
        (pl.col("projected_ceiling") - pl.col("ceiling_repl")).alias("ceiling_vor"),
        (pl.col(PROJECTED_VOR) + pl.col("override_delta")).alias(ADJUSTED_PROJECTED_VOR),
    )
    score_weight = (
        config.metrics.projection_mean_weight
        + config.metrics.projection_floor_weight
        + config.metrics.projection_ceiling_weight
    )
    frame = frame.with_columns(
        (
            (
                config.metrics.projection_mean_weight * pl.col(PROJECTED_VOR)
                + config.metrics.projection_floor_weight * pl.col("floor_vor")
                + config.metrics.projection_ceiling_weight * pl.col("ceiling_vor")
            )
            / score_weight
            + pl.col("override_delta")
        ).alias("v2_score")
    ).sort("v2_score", descending=True)
    return frame.with_columns(pl.int_range(1, frame.height + 1, dtype=pl.Int32).alias("rank"))
