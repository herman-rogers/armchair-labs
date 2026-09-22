"""v2 forward projections with explicit player uncertainty and team context.

v1 answers a retrospective question: how many league points per game did the player
score, and how far was that above positional replacement?  This module deliberately
answers a different question: what is a reasonable next-season PPG estimate?

The projection is intentionally transparent rather than presented as a trained black
box.  It combines a recency-weighted multi-year individual prior, small-sample
shrinkage toward the all-player positional pool, a position-specific age curve, a
depth-chart role factor, and a bottom-up component stat line driven by projected team
volume.  Team volume enters as a ratio of the projected environment to the
environments already embedded in the player's history, so a player who never changed
teams is not credited twice for the same offense.  Every adjustment that failed the
walk-forward backtest — TD regression, big-play regression, the team-context scalar,
the WOPR role blend, and the teammate-competition multiplier — has been removed
(docs/v2_metrics_review.md §4, §5, §6a).

The latest nflverse depth chart supplies current team and role evidence. Facts that
remain uncertain—a future trade, quarterback change, or camp promotion—live in
``config/projections.yaml`` with visible reasons, in the same spirit as the v1 injury
overrides.
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


GAMES_PLAYED = "ppg_denominator_games"


def _games_played(row: dict[str, Any]) -> float:
    """Games the player was actually active for, not games with a stat line.

    nflverse weekly stats omit dressed players with no box-score event, so ``games``
    undercounts appearances for low-usage players (three games for a typical TE).
    v2's PPG already divides by participation-observed games; every other per-game
    rate must use the same denominator or the component branch projects targets per
    stat-game while being scored per active game.
    """
    played = _number(row.get(GAMES_PLAYED))
    return played if played > 0 else _number(row.get("games"))


def _games_column(frame: pl.DataFrame) -> str:
    return GAMES_PLAYED if GAMES_PLAYED in frame.columns else "games"


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
    pass_volume_multiplier: float = 1.0
    rush_volume_multiplier: float = 1.0
    scoring_multiplier: float = 1.0
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
    pass_volume: float = 0.0
    dropbacks: float = 0.0
    rush_volume: float = 0.0
    scoring: float = 0.0
    pass_td_rate: float = 0.0
    rush_td_rate: float = 0.0


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
                "passing_tds": 0.0,
                "attempts": 0.0,
                "team_pass_attempts": 0.0,
                "team_dropbacks": 0.0,
                "official_team_carries": 0.0,
                "rushing_tds": 0.0,
                "targets": 0.0,
                "carries": 0.0,
            },
        )
        bucket["games"] = max(
            bucket["games"],
            _number(row.get("team_games")) or _number(row.get("games")),
        )
        for column in (
            "passing_tds",
            "attempts",
            "rushing_tds",
            "carries",
            "targets",
        ):
            bucket[column] += _number(row.get(column))
        bucket["team_pass_attempts"] = max(
            bucket["team_pass_attempts"], _number(row.get("team_pass_attempts"))
        )
        bucket["team_dropbacks"] = max(bucket["team_dropbacks"], _number(row.get("team_dropbacks")))
        bucket["official_team_carries"] = max(
            bucket["official_team_carries"], _number(row.get("official_team_carries"))
        )

    profiles: dict[tuple[int, str], TeamProfile] = {}
    for key, values in grouped.items():
        games = values["games"] or 1.0
        pass_attempts = values["team_pass_attempts"] or values["attempts"]
        if pass_attempts <= 0:
            pass_attempts = values["targets"]
        dropbacks = values["team_dropbacks"] or pass_attempts
        team_carries = values["official_team_carries"] or values["carries"]
        profiles[key] = TeamProfile(
            games=games,
            pass_volume=pass_attempts / games,
            dropbacks=dropbacks / games,
            rush_volume=team_carries / games,
            scoring=(values["passing_tds"] + values["rushing_tds"]) / games,
            pass_td_rate=values["passing_tds"] / max(pass_attempts, 1.0),
            rush_td_rate=values["rushing_tds"] / max(team_carries, 1.0),
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
    """Top 2x-baseline-rank players with a full-enough season, by position."""
    pools: dict[str, pl.DataFrame] = {}
    for position, baseline_rank in config.vor_baseline_rank.items():
        pools[position] = (
            seasons.filter(
                (pl.col("position") == position)
                & (pl.col(_games_column(seasons)) >= config.min_games_baseline)
            )
            .sort(["season", "ppg"], descending=[True, True])
            .group_by("season", maintain_order=True)
            .head(baseline_rank * 2)
        )
    return pools


def _all_player_rows(
    seasons: pl.DataFrame,
    config: LeagueConfig,
) -> dict[str, pl.DataFrame]:
    """Every player-season at the position that reached the board."""
    return {
        position: seasons.filter(
            (pl.col("position") == position)
            & (pl.col(_games_column(seasons)) >= config.min_games_board)
        )
        for position in config.vor_baseline_rank
    }


def _prior_rows(seasons: pl.DataFrame, config: LeagueConfig) -> dict[str, pl.DataFrame]:
    """The player-seasons that define positional priors.

    Shrinkage pulls a small sample toward this pool's mean, so the pool must describe
    what a player with little evidence actually tends to become.  The rosterable pool
    (top starters only) says "a starter", which inflates every fringe player toward
    starter production; the all-player pool, weighted by games, says "an ordinary
    player at the position", which is the honest expectation.  Rosterable is retained
    as a configurable alternative for comparison.
    """
    if config.metrics.projection_prior_pool == "rosterable":
        return _rosterable_rows(seasons, config)
    return _all_player_rows(seasons, config)


def _position_means(pools: dict[str, pl.DataFrame]) -> dict[str, float]:
    """Games-weighted PPG prior by position from the prior pool."""
    ppg: dict[str, float] = {}
    for position, rows in pools.items():
        if rows.height == 0:
            ppg[position] = 0.0
            continue
        weighted_games = rows[_games_column(rows)].cast(pl.Float64)
        game_total = float(weighted_games.sum())
        ppg[position] = float((rows["ppg"] * weighted_games).sum()) / game_total
    return ppg


def _actual_core_points(row: dict[str, Any]) -> float:
    """Scored components v2 projects explicitly; the residual keeps everything else.

    Big-play bonus points sit in the residual deliberately: the dedicated bonus
    projection (opportunity x regressed rate x air-yard factor) tested anti-predictive
    (review §4/§5), so bonus scoring now reaches the component branch only through the
    heavily shrunk residual, like other rare scoring events.
    """
    return (
        0.04 * _number(row.get("passing_yards"))
        + 4.0 * _number(row.get("passing_tds"))
        - 2.0 * _number(row.get("passing_interceptions"))
        + 0.1 * _number(row.get("rushing_yards"))
        + 6.0 * _number(row.get("rushing_tds"))
        + _number(row.get("receptions"))
        + 0.1 * _number(row.get("receiving_yards"))
        + 6.0 * _number(row.get("receiving_tds"))
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
            "passing_yards": 0.0,
            "attempts": 0.0,
            "passing_tds": 0.0,
            "passing_interceptions": 0.0,
            "target_share": 0.0,
            "carry_share": 0.0,
            "air_yards_share": 0.0,
            "route_participation": 0.0,
            "targets_per_route_opportunity": 0.0,
            "end_zone_targets": 0.0,
            "end_zone_receiving_tds": 0.0,
            "goal_line_carries": 0.0,
            "goal_line_rushing_tds": 0.0,
            "floor_ratio": 0.0,
            "volatility_ratio": 0.0,
            "residual_points": 0.0,
        }
        for row in frame.iter_rows(named=True):
            decay = config.metrics.projection_season_decay ** max(
                config.board_season - int(row["season"]), 0
            )
            games = max(_games_played(row), 1.0)
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
                "passing_yards",
                "attempts",
                "passing_tds",
                "passing_interceptions",
                "end_zone_targets",
                "end_zone_receiving_tds",
                "goal_line_carries",
                "goal_line_rushing_tds",
            ):
                totals[column] += _number(row.get(column)) * decay

            profile = profiles.get((int(row["season"]), str(row.get("team") or "").upper()))
            team_targets = profile.pass_volume * profile.games if profile else 0.0
            team_carries = profile.rush_volume * profile.games if profile else 0.0
            totals["target_share"] += (
                _number(row.get("targets")) / team_targets if team_targets else 0.0
            ) * game_weight
            totals["carry_share"] += (
                _number(row.get("carries")) / team_carries if team_carries else 0.0
            ) * game_weight
            totals["air_yards_share"] += _number(row.get("air_yards_share")) * game_weight
            if row.get("route_participation") is not None:
                totals["route_participation"] += (
                    _number(row.get("route_participation")) * game_weight
                )
            if row.get("targets_per_route_opportunity") is not None:
                totals["targets_per_route_opportunity"] += (
                    _number(row.get("targets_per_route_opportunity")) * game_weight
                )
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
        priors[position] = {
            "target_share": totals["target_share"] / games,
            "carry_share": totals["carry_share"] / games,
            "air_yards_share": totals["air_yards_share"] / games,
            "route_participation": totals["route_participation"] / games,
            "targets_per_route_opportunity": totals["targets_per_route_opportunity"] / games,
            "catch_rate": totals["receptions"] / targets,
            "receiving_yards_per_target": totals["receiving_yards"] / targets,
            "rushing_yards_per_carry": totals["rushing_yards"] / carries,
            "receiving_td_rate": totals["receiving_tds"] / targets,
            "rushing_td_rate": totals["rushing_tds"] / carries,
            "end_zone_target_rate": totals["end_zone_targets"] / targets,
            "end_zone_conversion_rate": totals["end_zone_receiving_tds"]
            / max(totals["end_zone_targets"], 1.0),
            "non_end_zone_receiving_td_rate": max(
                totals["receiving_tds"] - totals["end_zone_receiving_tds"], 0.0
            )
            / max(totals["targets"] - totals["end_zone_targets"], 1.0),
            "goal_line_carry_rate": totals["goal_line_carries"] / carries,
            "goal_line_conversion_rate": totals["goal_line_rushing_tds"]
            / max(totals["goal_line_carries"], 1.0),
            "non_goal_line_rushing_td_rate": max(
                totals["rushing_tds"] - totals["goal_line_rushing_tds"], 0.0
            )
            / max(totals["carries"] - totals["goal_line_carries"], 1.0),
            "passing_yards_pg": totals["passing_yards"] / games,
            "passing_yards_per_attempt": totals["passing_yards"] / max(totals["attempts"], 1.0),
            "passing_td_per_attempt": totals["passing_tds"] / max(totals["attempts"], 1.0),
            "interception_per_attempt": totals["passing_interceptions"]
            / max(totals["attempts"], 1.0),
            "passing_tds_pg": totals["passing_tds"] / games,
            "passing_interceptions_pg": totals["passing_interceptions"] / games,
            "rushing_yards_pg": totals["rushing_yards"] / games,
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
        "pass_volume": profile.pass_volume * (override.pass_volume_multiplier if override else 1.0),
        "dropbacks": profile.dropbacks * (override.pass_volume_multiplier if override else 1.0),
        "rush_volume": profile.rush_volume * (override.rush_volume_multiplier if override else 1.0),
        "scoring": profile.scoring * (override.scoring_multiplier if override else 1.0),
        "pass_td_rate": profile.pass_td_rate * (override.scoring_multiplier if override else 1.0),
        "rush_td_rate": profile.rush_td_rate * (override.scoring_multiplier if override else 1.0),
    }


def _depth_role_factor(rank: int | None, cap: float) -> float:
    """Bounded automatic role signal from the latest nflverse depth chart."""
    if rank is None or rank <= 0:
        return 1.0
    raw = {1: 1.0, 2: 0.93, 3: 0.82, 4: 0.72}.get(rank, 0.65)
    return _bounded(raw, 1.0 - cap, 1.0 + cap)


def build_projection_board(
    player_seasons: pl.DataFrame,
    v1_board: pl.DataFrame,
    config: LeagueConfig,
    assumptions: ProjectionAssumptions | None = None,
    current_players: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the v2 board while retaining every v1 metric as supporting evidence."""
    assumptions = assumptions or ProjectionAssumptions.from_config()
    metrics = config.metrics
    profiles = _team_profiles(player_seasons)
    pools = _prior_rows(player_seasons, config)
    position_ppg = _position_means(pools)
    component_priors = _component_priors(pools, profiles, config)
    current_by_player = (
        {str(row["player_id"]): row for row in current_players.iter_rows(named=True)}
        if current_players is not None
        else {}
    )

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
        current = current_by_player.get(player_id, {})
        projected_team = str(
            (
                override.projected_team
                if override and override.projected_team
                else current.get("current_team") or base["team"]
            )
            or ""
        ).upper()
        depth_rank_value = current.get("depth_chart_rank")
        depth_rank = int(depth_rank_value) if depth_rank_value is not None else None
        depth_role_factor = _depth_role_factor(
            depth_rank,
            metrics.projection_depth_chart_cap,
        )

        weighted: list[tuple[dict[str, Any], float]] = []
        for row in history:
            years_old = max(config.board_season - int(row["season"]), 0)
            weight = metrics.projection_season_decay**years_old * max(_games_played(row), 1.0)
            weighted.append((row, weight))
        total_weight = sum(weight for _, weight in weighted) or 1.0
        historical_ppg = (
            sum(_number(row.get("ppg")) * weight for row, weight in weighted) / total_weight
        )
        effective_games = sum(
            metrics.projection_season_decay ** max(config.board_season - int(row["season"]), 0)
            * _games_played(row)
            for row in history
        )
        reliability = effective_games / (effective_games + metrics.projection_shrinkage_games)
        individual_prior = reliability * historical_ppg + (1.0 - reliability) * position_ppg.get(
            position, historical_ppg
        )

        latest = max(history, key=lambda row: int(row["season"]))

        source_context = {
            "pass_volume": 0.0,
            "dropbacks": 0.0,
            "rush_volume": 0.0,
            "scoring": 0.0,
            "pass_td_rate": 0.0,
            "rush_td_rate": 0.0,
        }
        context_weight = 0.0
        for row, weight in weighted:
            profile = profiles.get((int(row["season"]), str(row.get("team") or "").upper()))
            if profile is None:
                continue
            context_weight += weight
            source_context["pass_volume"] += profile.pass_volume * weight
            source_context["dropbacks"] += profile.dropbacks * weight
            source_context["rush_volume"] += profile.rush_volume * weight
            source_context["scoring"] += profile.scoring * weight
            source_context["pass_td_rate"] += profile.pass_td_rate * weight
            source_context["rush_td_rate"] += profile.rush_td_rate * weight
        if context_weight:
            source_context = {key: value / context_weight for key, value in source_context.items()}

        target_profile = profiles.get((config.board_season, projected_team))
        target_team_override = assumptions.teams.get(projected_team)
        if target_profile is None:
            # Unknown future teams/rookies should not manufacture a context effect.
            target_profile = TeamProfile(
                pass_volume=source_context["pass_volume"],
                dropbacks=source_context["dropbacks"],
                rush_volume=source_context["rush_volume"],
                scoring=source_context["scoring"],
                pass_td_rate=source_context["pass_td_rate"],
                rush_td_rate=source_context["rush_td_rate"],
            )
        target = _apply_team_override(target_profile, target_team_override)
        latest_source_profile = profiles.get(
            (config.board_season, str(base.get("team") or "").upper())
        )
        latest_team_targets = (
            latest_source_profile.pass_volume * latest_source_profile.games
            if latest_source_profile
            else 0.0
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

        manual_role_multiplier = override.opportunity_multiplier if override else 1.0
        role_multiplier = manual_role_multiplier * depth_role_factor
        age = base.get("age_at_season")
        age_factor = _age_factor(position, float(age) if age is not None else None)
        # The 2026-08-29 review ablated the prior branch: age and depth role carried
        # signal; TD regression, big-play regression, and the team-context scalar did
        # not (docs/v2_metrics_review.md §4, §6a). Only the survivors remain.
        prior_projection = max(individual_prior, 0.0) * age_factor * role_multiplier

        # The component branch turns role and team volume into a projected stat line.
        # Totals use season decay without multiplying by games a second time.
        def weighted_total(
            column: str,
            rows: list[tuple[dict[str, Any], float]] = weighted,
        ) -> float:
            return sum(
                _number(row.get(column)) * weight / max(_games_played(row), 1.0)
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
        weighted_pass_attempts = weighted_total("attempts")
        weighted_route_opportunities = weighted_total("route_opportunities")
        weighted_end_zone_targets = weighted_total("end_zone_targets")
        weighted_end_zone_tds = weighted_total("end_zone_receiving_tds")
        weighted_red_zone_targets = weighted_total("red_zone_targets")
        weighted_goal_line_carries = weighted_total("goal_line_carries")
        weighted_goal_line_tds = weighted_total("goal_line_rushing_tds")
        weighted_red_zone_carries = weighted_total("red_zone_carries")

        historical_target_share = 0.0
        historical_carry_share = 0.0
        share_weight = 0.0
        for row, weight in weighted:
            profile = profiles.get((int(row["season"]), str(row.get("team") or "").upper()))
            if profile is None:
                continue
            team_targets = profile.pass_volume * profile.games
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

        # The former WOPR role blend and teammate-competition multipliers are gone:
        # WOPR double-counted target share with air-yard information the air-yard
        # factor already carries, and "strongest teammate share" tested inert in every
        # era since 2004 (review §5, §6a). Competition enters only through the
        # depth-chart role factor and the shrunk shares themselves.
        projected_target_share = _bounded(projected_target_share, 0.0, 0.45)
        projected_carry_share = _bounded(projected_carry_share, 0.0, 0.85)

        pass_volume_ratio = _ratio(target["pass_volume"], source_context["pass_volume"])
        rush_volume_ratio = _ratio(target["rush_volume"], source_context["rush_volume"])
        team_target_projection = projected_target_share * target["pass_volume"]
        team_carry_projection = projected_carry_share * target["rush_volume"]
        historical_targets_pg = weighted_targets / max(effective_games, 1.0)
        historical_carries_pg = weighted_carries / max(effective_games, 1.0)
        route_reliability = weighted_route_opportunities / (weighted_route_opportunities + 160.0)
        historical_route_participation = weighted_metric("route_participation")
        projected_route_participation = route_reliability * historical_route_participation + (
            1.0 - route_reliability
        ) * prior.get("route_participation", historical_route_participation)
        historical_tprr = weighted_targets / max(weighted_route_opportunities, 1.0)
        projected_tprr = route_reliability * historical_tprr + (
            1.0 - route_reliability
        ) * prior.get("targets_per_route_opportunity", historical_tprr)
        projected_route_participation = _bounded(projected_route_participation, 0.0, 1.0)
        projected_tprr = _bounded(projected_tprr, 0.0, 0.45)
        projected_route_opportunities = projected_route_participation * target["dropbacks"]
        route_target_projection = projected_route_opportunities * projected_tprr
        has_route_data = weighted_route_opportunities > 0 and target["dropbacks"] > 0

        if position == "QB":
            projected_targets = 0.0
            projected_carries = (
                reliability * historical_carries_pg
                + (1.0 - reliability) * prior.get("rushing_yards_pg", 0.0) / 4.5
            )
            projected_carries *= role_multiplier
        else:
            # The former third anchor, "opportunity-implied targets", was weighted
            # opportunity minus carries over 2.2 — algebraically targets again
            # (review §5) — so its 0.15 weight folds into the target-rate term.
            baseline_target_projection = (
                0.65 * team_target_projection + 0.35 * historical_targets_pg * pass_volume_ratio
            )
            route_weight = metrics.projection_route_weight if has_route_data else 0.0
            projected_targets = (
                route_weight * route_target_projection
                + (1.0 - route_weight) * baseline_target_projection
            ) * role_multiplier
            projected_carries = (
                0.75 * team_carry_projection + 0.25 * historical_carries_pg * rush_volume_ratio
            ) * role_multiplier

        if target["pass_volume"] > 0 and position != "QB":
            projected_target_share = _bounded(
                projected_targets / target["pass_volume"],
                0.0,
                0.45,
            )

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

        high_value_target_reliability = weighted_end_zone_targets / (
            weighted_end_zone_targets + 16.0
        )
        historical_end_zone_target_rate = weighted_end_zone_targets / max(weighted_targets, 1.0)
        projected_end_zone_target_rate = (
            high_value_target_reliability * historical_end_zone_target_rate
            + (1.0 - high_value_target_reliability)
            * prior.get("end_zone_target_rate", historical_end_zone_target_rate)
        )
        historical_end_zone_conversion = weighted_end_zone_tds / max(weighted_end_zone_targets, 1.0)
        projected_end_zone_conversion = (
            high_value_target_reliability * historical_end_zone_conversion
            + (1.0 - high_value_target_reliability)
            * prior.get("end_zone_conversion_rate", historical_end_zone_conversion)
        )
        non_end_zone_targets = max(weighted_targets - weighted_end_zone_targets, 1.0)
        historical_non_end_zone_td_rate = (
            max(weighted_receiving_tds - weighted_end_zone_tds, 0.0) / non_end_zone_targets
        )
        projected_non_end_zone_td_rate = (
            receiving_td_reliability * historical_non_end_zone_td_rate
            + (1.0 - receiving_td_reliability)
            * prior.get("non_end_zone_receiving_td_rate", historical_non_end_zone_td_rate)
        )

        high_value_carry_reliability = weighted_goal_line_carries / (
            weighted_goal_line_carries + 20.0
        )
        historical_goal_line_carry_rate = weighted_goal_line_carries / max(weighted_carries, 1.0)
        projected_goal_line_carry_rate = (
            high_value_carry_reliability * historical_goal_line_carry_rate
            + (1.0 - high_value_carry_reliability)
            * prior.get("goal_line_carry_rate", historical_goal_line_carry_rate)
        )
        historical_goal_line_conversion = weighted_goal_line_tds / max(
            weighted_goal_line_carries, 1.0
        )
        projected_goal_line_conversion = (
            high_value_carry_reliability * historical_goal_line_conversion
            + (1.0 - high_value_carry_reliability)
            * prior.get("goal_line_conversion_rate", historical_goal_line_conversion)
        )
        non_goal_line_carries = max(weighted_carries - weighted_goal_line_carries, 1.0)
        historical_non_goal_line_td_rate = (
            max(weighted_rushing_tds - weighted_goal_line_tds, 0.0) / non_goal_line_carries
        )
        projected_non_goal_line_td_rate = (
            rushing_td_reliability * historical_non_goal_line_td_rate
            + (1.0 - rushing_td_reliability)
            * prior.get("non_goal_line_rushing_td_rate", historical_non_goal_line_td_rate)
        )

        projected_receptions = projected_targets * projected_catch_rate
        projected_receiving_yards = projected_targets * projected_ypt
        projected_rushing_yards = projected_carries * projected_ypc
        projected_red_zone_targets = projected_targets * (
            weighted_red_zone_targets / max(weighted_targets, 1.0)
        )
        projected_end_zone_targets = projected_targets * projected_end_zone_target_rate
        projected_goal_line_carries = projected_carries * projected_goal_line_carry_rate
        projected_red_zone_carries = projected_carries * (
            weighted_red_zone_carries / max(weighted_carries, 1.0)
        )
        high_value_receiving_tds = (
            projected_end_zone_targets * projected_end_zone_conversion
            + max(projected_targets - projected_end_zone_targets, 0.0)
            * projected_non_end_zone_td_rate
        ) * _ratio(target["pass_td_rate"], source_context["pass_td_rate"])
        high_value_rushing_tds = (
            projected_goal_line_carries * projected_goal_line_conversion
            + max(projected_carries - projected_goal_line_carries, 0.0)
            * projected_non_goal_line_td_rate
        ) * _ratio(target["rush_td_rate"], source_context["rush_td_rate"])
        receiving_high_value_weight = 0.70 if weighted_end_zone_targets > 0 else 0.0
        rushing_high_value_weight = 0.70 if weighted_goal_line_carries > 0 else 0.0
        projected_receiving_tds = (
            receiving_high_value_weight * high_value_receiving_tds
            + (1.0 - receiving_high_value_weight) * projected_targets * projected_receiving_td_rate
        )
        projected_rushing_tds = (
            rushing_high_value_weight * high_value_rushing_tds
            + (1.0 - rushing_high_value_weight) * projected_carries * projected_rushing_td_rate
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
        projected_pass_attempts = 0.0
        if position == "QB" and weighted_pass_attempts > 0:
            attempt_reliability = weighted_pass_attempts / (weighted_pass_attempts + 220.0)
            historical_ypa = weighted_total("passing_yards") / weighted_pass_attempts
            historical_pass_td_rate = weighted_total("passing_tds") / weighted_pass_attempts
            historical_interception_rate = (
                weighted_total("passing_interceptions") / weighted_pass_attempts
            )
            projected_ypa = attempt_reliability * historical_ypa + (
                1.0 - attempt_reliability
            ) * prior.get("passing_yards_per_attempt", historical_ypa)
            projected_pass_td_rate = attempt_reliability * historical_pass_td_rate + (
                1.0 - attempt_reliability
            ) * prior.get("passing_td_per_attempt", historical_pass_td_rate)
            projected_interception_rate = attempt_reliability * historical_interception_rate + (
                1.0 - attempt_reliability
            ) * prior.get("interception_per_attempt", historical_interception_rate)
            projected_pass_attempts = target["pass_volume"] * role_multiplier
            projected_passing_yards = projected_pass_attempts * projected_ypa
            projected_passing_tds = (
                projected_pass_attempts
                * projected_pass_td_rate
                * _ratio(target["pass_td_rate"], source_context["pass_td_rate"])
            )
            projected_interceptions = projected_pass_attempts * projected_interception_rate
        if position != "QB":
            projected_pass_attempts = 0.0
            projected_passing_yards = 0.0
            projected_passing_tds = 0.0
            projected_interceptions = 0.0

        historical_residual_points = sum(
            (
                _number(row.get("season_pts")) - _actual_core_points(row)
                if row.get("season_pts") is not None
                else 0.0
            )
            * weight
            / max(_games_played(row), 1.0)
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
            + projected_residual_pg
        ) * age_factor
        if position == "QB":
            component_projection *= role_multiplier

        component_weight = metrics.projection_component_weight
        proj_ppg = (1.0 - component_weight) * prior_projection + component_weight * max(
            component_projection, 0.0
        )

        availability_numerator = 0.0
        availability_denominator = 0.0
        injury_availability_denominator = 0.0
        injury_missed_equivalents = 0.0
        injury_report_weeks = 0.0
        for row in history:
            decay = metrics.projection_season_decay ** max(
                config.board_season - int(row["season"]), 0
            )
            observed_games = _number(row.get("active_games")) or _number(row.get("games"))
            possible_games = _number(row.get("team_games")) or observed_games
            availability_numerator += observed_games * decay
            availability_denominator += possible_games * decay
            # A missing marker means the caller predates explicit provenance and is
            # treated as available for backward compatibility. An explicit False is
            # historical source absence, not evidence of a perfectly healthy season.
            injury_source_available = row.get("injury_data_available") is not False
            if injury_source_available:
                injury_availability_denominator += possible_games * decay
                injury_report_weeks += _number(row.get("injury_report_weeks")) * decay
                injury_missed_equivalents += (
                    _number(row.get("out_report_weeks"))
                    + 0.5 * _number(row.get("doubtful_report_weeks"))
                    + 0.1 * _number(row.get("questionable_report_weeks"))
                ) * decay
        observed_availability = (
            availability_numerator / availability_denominator
            if availability_denominator > 0
            else metrics.projection_availability_prior
        )
        if injury_availability_denominator > 0:
            injury_availability = 1.0 - injury_missed_equivalents / max(
                injury_availability_denominator, 1.0
            )
            availability_sample = 0.75 * observed_availability + 0.25 * injury_availability
        else:
            availability_sample = observed_availability
        availability_sample = _bounded(availability_sample, 0.50, 1.0)
        availability_reliability = availability_denominator / (
            availability_denominator + metrics.projection_availability_shrinkage_games
        )
        projected_availability = (
            availability_reliability * availability_sample
            + (1.0 - availability_reliability) * metrics.projection_availability_prior
        )
        expected_games = _bounded(
            metrics.projection_season_games * projected_availability,
            1.0,
            float(metrics.projection_season_games),
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

        moved = projected_team != str(base.get("team") or "").upper()
        confidence = reliability * (0.88 if moved else 1.0)
        if override and override.opportunity_multiplier != 1.0:
            confidence *= 0.92
        if depth_role_factor != 1.0:
            confidence *= 0.95

        reasons = [
            value
            for value in (
                override.reason if override else None,
                target_team_override.reason if target_team_override else None,
            )
            if value
        ]
        if moved and not (override and override.projected_team):
            reasons.append(f"nflverse depth chart moved team to {projected_team}")
        projected.append(
            {
                **base,
                METRIC_VERSION: "v2",
                "projected_team": projected_team,
                "depth_chart_rank": depth_rank,
                "depth_chart_position": current.get("depth_chart_position"),
                "depth_chart_position_group": current.get("depth_chart_position_group"),
                "depth_chart_date": current.get("depth_chart_date"),
                "depth_role_factor": depth_role_factor,
                "historical_ppg_prior": historical_ppg,
                "individual_prior_ppg": individual_prior,
                "prior_branch_ppg": prior_projection,
                "component_proj_ppg": component_projection,
                PROJECTED_PPG: proj_ppg,
                "projected_floor": projected_floor,
                "projected_volatility": projected_volatility,
                "projection_confidence": _bounded(confidence, 0.0, 1.0),
                "availability_confidence": availability_reliability,
                "projected_availability": projected_availability,
                "expected_games": expected_games,
                "historical_injury_report_weeks": injury_report_weeks,
                "injury_missed_equivalents": injury_missed_equivalents,
                "effective_games": effective_games,
                "age_factor": age_factor,
                "team_scoring_context": target["scoring"],
                "team_pass_volume": target["pass_volume"],
                "team_dropbacks": target["dropbacks"],
                "team_rush_volume": target["rush_volume"],
                "season_target_share": season_target_share,
                "season_carry_share": season_carry_share,
                "projected_target_share": projected_target_share,
                "projected_carry_share": projected_carry_share,
                "projected_air_yards_share": projected_air_share,
                "projected_targets_pg": projected_targets,
                "projected_route_opportunities_pg": projected_route_opportunities,
                "projected_route_participation": projected_route_participation,
                "projected_targets_per_route_opportunity": projected_tprr,
                "projected_carries_pg": projected_carries,
                "projected_receptions_pg": projected_receptions,
                "projected_receiving_yards_pg": projected_receiving_yards,
                "projected_rushing_yards_pg": projected_rushing_yards,
                "projected_receiving_tds_pg": projected_receiving_tds,
                "projected_rushing_tds_pg": projected_rushing_tds,
                "projected_red_zone_targets_pg": projected_red_zone_targets,
                "projected_end_zone_targets_pg": projected_end_zone_targets,
                "projected_red_zone_carries_pg": projected_red_zone_carries,
                "projected_goal_line_carries_pg": projected_goal_line_carries,
                "projected_pass_attempts_pg": projected_pass_attempts,
                "projected_passing_yards_pg": projected_passing_yards,
                "projected_passing_tds_pg": projected_passing_tds,
                "projected_interceptions_pg": projected_interceptions,
                "air_yard_factor": air_yard_factor,
                "opportunity_multiplier": manual_role_multiplier,
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
        games_column=_games_column(frame),
    )
    frame = add_vor(frame, levels, ppg_column=PROJECTED_PPG).rename(
        {"repl_ppg": "proj_repl_ppg", "vor": PROJECTED_VOR}
    )
    # Superseded composites (v2_score, floor/ceiling VOR, availability-adjusted VOR)
    # were removed after the backtest showed the fitted rankers beat them; the board's
    # sort keys now live in ``board.rank``. ``adj_proj_vor`` remains as the projection's
    # own VOR with overrides, the fallback when no fitted model is available.
    frame = frame.with_columns(
        (pl.col(PROJECTED_VOR) + pl.col("override_delta")).alias(ADJUSTED_PROJECTED_VOR)
    ).sort(ADJUSTED_PROJECTED_VOR, descending=True, nulls_last=True)
    return frame.with_columns(pl.int_range(1, frame.height + 1, dtype=pl.Int32).alias("rank"))
