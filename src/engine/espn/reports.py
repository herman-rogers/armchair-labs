"""The artifacts a poll produces.

Three, per §5 of the plan: the ranked wire, my roster's health, and the raw material
for alerts. All of them are the same board, filtered by who owns whom — which is the
whole point of joining ESPN state onto it.

The wire is the one that wins waivers. Replacement level is recomputed from what is
actually available rather than reused from the preseason board, because "value over
replacement" only means anything if replacement means "what I can have for free right
now". In a league where half the pool is rostered, the real bar is far below the
preseason one, and using the stale number understates every pickup on the list.
"""

from __future__ import annotations

import logging
import math
import random
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import polars as pl

from engine.board.overrides import ADJUSTED_VOR
from engine.espn.crosswalk import (
    CHANGED_TEAM,
    ESPN_ID,
    INJURY_STATUS,
    IS_FREE_AGENT,
    IS_MINE,
    OWNER_TEAM_ID,
    OWNER_TEAM_NAME,
    PERCENT_OWNED,
)
from engine.metrics.projection import ADJUSTED_PROJECTED_VOR

logger = logging.getLogger(__name__)

WIRE_VOR = "wire_vor"
WIRE_REPLACEMENT = "wire_repl_ppg"

#: Injury tags worth surfacing on a roster report. ACTIVE and NORMAL are not news.
CONCERNING_INJURY_TAGS = frozenset(
    {"QUESTIONABLE", "DOUBTFUL", "OUT", "INJURY_RESERVE", "SUSPENSION", "DAY_TO_DAY"}
)

BENCH_SLOTS = frozenset({"BE", "BENCH"})
IR_SLOTS = frozenset({"IR", "INJURED RESERVE"})
FLEX_SLOTS = frozenset({"RB/WR/TE", "FLEX"})
TEAM_SIMULATIONS = 12_000


@dataclass(frozen=True)
class LineupRequirements:
    fixed: dict[str, int]
    flex: int


@dataclass(frozen=True)
class TeamPlayerProjection:
    player_id: int
    name: str
    position: str
    mean: float
    volatility: float
    availability: float
    ranking_value: float | None
    fallback: bool


def _projection_ppg_column(frame: pl.DataFrame) -> str:
    """Season-equivalent V2 production, historical PPG for V1."""
    return "season_equivalent_ppg" if "season_equivalent_ppg" in frame.columns else "ppg"


def _board_value_column(frame: pl.DataFrame) -> str:
    for candidate in (
        "v2_overall_vor",
        "v2_rank_vor",
        ADJUSTED_PROJECTED_VOR,
        ADJUSTED_VOR,
        "vor",
        "proj_ppg",
        "ppg",
    ):
        if candidate in frame.columns and frame[candidate].is_not_null().any():
            return candidate
    raise ValueError("board carries no usable ranking column")


def wire_replacement_levels(
    tagged_board: pl.DataFrame,
    baseline_ranks: dict[str, int],
    ppg_column: str | None = None,
) -> dict[str, float]:
    """Replacement level measured against the free-agent pool, not the preseason board.

    The best freely-available player at a position *is* replacement level, by
    definition. Preseason baselines assume an undrafted pool; once a league has drafted,
    the honest bar is whatever is still sitting there.

    Baseline keys select positions only; draft replacement ranks do not apply here.
    """
    ppg_column = ppg_column or _projection_ppg_column(tagged_board)
    levels: dict[str, float] = {}
    available = tagged_board.filter(pl.col(IS_FREE_AGENT))

    for position in baseline_ranks:
        pool = (
            available.filter(pl.col("position") == position)
            .sort(ppg_column, descending=True)
            .get_column(ppg_column)
            .drop_nulls()
        )
        if pool.len() == 0:
            continue
        levels[position] = float(pool[0])

    return levels


#: Tags meaning the player cannot be started this week. Worth keeping on the list — a
#: cheap stash is a real play in a league with an IR slot — but worth being able to hide.
UNAVAILABLE_INJURY_TAGS = frozenset({"OUT", "INJURY_RESERVE", "SUSPENSION"})


def ranked_wire(
    tagged_board: pl.DataFrame,
    baseline_ranks: dict[str, int],
    min_vor: float | None = None,
    limit: int | None = 50,
    healthy_only: bool = False,
) -> pl.DataFrame:
    """Free agents worth having, ranked against what else is free.

    Ranks on value alone and leaves availability as a separate, visible column rather
    than blending the two into one score. An OUT player at the top of the list is
    information — a cheap stash — not an error, so hiding him is opt-in.

    Args:
        tagged_board: Board with ownership attached.
        baseline_ranks: Positional replacement ranks from league config.
        min_vor: Drop players below this wire-VOR. `None` keeps everything.
        limit: Cap on rows returned.
        healthy_only: Drop players who cannot be started this week.
    """
    ppg_column = _projection_ppg_column(tagged_board)
    levels = wire_replacement_levels(tagged_board, baseline_ranks, ppg_column=ppg_column)
    logger.info(
        "wire replacement: %s",
        ", ".join(f"{position} {ppg:.1f}" for position, ppg in sorted(levels.items())),
    )

    wire = tagged_board.filter(pl.col(IS_FREE_AGENT)).with_columns(
        pl.col("position")
        .replace_strict(levels, default=None, return_dtype=pl.Float64)
        .alias(WIRE_REPLACEMENT)
    )
    wire = wire.with_columns((pl.col(ppg_column) - pl.col(WIRE_REPLACEMENT)).alias(WIRE_VOR))

    if min_vor is not None:
        wire = wire.filter(pl.col(WIRE_VOR) >= min_vor)
    if healthy_only:
        wire = wire.filter(
            ~pl.col(INJURY_STATUS).is_in(list(UNAVAILABLE_INJURY_TAGS)).fill_null(False)
        )

    wire = wire.sort(WIRE_VOR, descending=True, nulls_last=True)
    return wire.head(limit) if limit else wire


def roster_health(
    tagged_board: pl.DataFrame,
    my_team_id: int | None = None,
) -> pl.DataFrame:
    """My roster, worst news first.

    Sorted so the rows demanding a decision surface above the ones that do not: an
    injury tag outranks a team change, which outranks a healthy starter.
    """
    mine = tagged_board.filter(pl.col(IS_MINE)) if my_team_id is not None else tagged_board
    value_column = _board_value_column(tagged_board)
    return mine.with_columns(
        pl.col(INJURY_STATUS).is_in(list(CONCERNING_INJURY_TAGS)).fill_null(False).alias("flagged")
    ).sort(["flagged", CHANGED_TEAM, value_column], descending=[True, True, True])


def opponent_weaknesses(
    tagged_board: pl.DataFrame,
    baseline_ranks: dict[str, int],
) -> pl.DataFrame:
    """Each team's worst starting position, by the value of its best player there.

    The Dan-TE-room scan: knowing which rival is thin somewhere is what makes a trade
    offer land, and what tells you which free agent to take before they need him.
    """
    rostered = tagged_board.filter(~pl.col(IS_FREE_AGENT) & pl.col(OWNER_TEAM_NAME).is_not_null())

    value_column = _board_value_column(tagged_board)
    best = (
        rostered.filter(pl.col("position").is_in(list(baseline_ranks)))
        .group_by([OWNER_TEAM_NAME, "position"])
        .agg(
            pl.col(value_column).max().alias("best_vor"),
            pl.col("player_display_name")
            .sort_by(value_column, descending=True)
            .first()
            .alias("best_player"),
            pl.len().alias("depth"),
        )
    )
    return best.sort([OWNER_TEAM_NAME, "best_vor"])


def _lineup_requirements(espn_players: pl.DataFrame) -> LineupRequirements:
    """Infer this league's starting shape from ESPN's populated lineup slots."""
    positions = ("QB", "RB", "WR", "TE")
    owners = sorted(
        int(value) for value in espn_players[OWNER_TEAM_ID].drop_nulls().unique().to_list()
    )
    counts: dict[int, Counter[str]] = {owner: Counter() for owner in owners}
    for row in espn_players.filter(pl.col(OWNER_TEAM_ID).is_not_null()).iter_rows(named=True):
        owner = int(row[OWNER_TEAM_ID])
        slot = str(row.get("lineup_slot") or "").upper()
        if slot in positions or slot in FLEX_SLOTS:
            counts[owner][slot] += 1

    def mode_count(slot: str) -> int:
        distribution = Counter(counts[owner][slot] for owner in owners)
        if not distribution:
            return 0
        # Prefer the larger count when malformed snapshots produce an exact tie.
        return max(distribution, key=lambda value: (distribution[value], value))

    fixed = {position: mode_count(position) for position in positions}
    flex = sum(mode_count(slot) for slot in FLEX_SLOTS)
    if not any(fixed.values()) and flex == 0:
        # Sweaty Plays' known shape, used only for incomplete synthetic/legacy data.
        return LineupRequirements(fixed={"QB": 1, "RB": 2, "WR": 3, "TE": 1}, flex=1)
    return LineupRequirements(fixed=fixed, flex=flex)


def _best_lineup(
    players: list[TeamPlayerProjection],
    active: list[bool],
    requirements: LineupRequirements,
) -> tuple[set[int], bool]:
    """Highest-mean legal lineup among the players active in one scenario."""
    selected: set[int] = set()
    complete = True
    for position, needed in requirements.fixed.items():
        eligible = [
            index
            for index, player in enumerate(players)
            if active[index] and player.position == position
        ][:needed]
        selected.update(eligible)
        complete = complete and len(eligible) == needed

    flex = [
        index
        for index, player in enumerate(players)
        if active[index] and index not in selected and player.position in {"RB", "WR", "TE"}
    ][: requirements.flex]
    selected.update(flex)
    complete = complete and len(flex) == requirements.flex
    return selected, complete


def _position_volatility_ratios(tagged_board: pl.DataFrame) -> dict[str, float]:
    mean_column = "proj_ppg" if "proj_ppg" in tagged_board.columns else "ppg"
    volatility_column = (
        "projected_volatility" if "projected_volatility" in tagged_board.columns else "volatility"
    )
    if mean_column not in tagged_board.columns or volatility_column not in tagged_board.columns:
        return {}
    ratios = (
        tagged_board.filter(pl.col(mean_column) > 0)
        .with_columns((pl.col(volatility_column) / pl.col(mean_column)).alias("volatility_ratio"))
        .group_by("position")
        .agg(pl.col("volatility_ratio").median())
    )
    return {
        str(position): float(value)
        for position, value in ratios.iter_rows()
        if value is not None and math.isfinite(value)
    }


@dataclass(frozen=True)
class RosterProjectionColumns:
    """Which per-player fields feed roster-level numbers; first present column wins.

    ``mean`` is per-active-game points.  Availability is ``games / season_games`` when
    a games column is present, else the availability column.  Volatility is rescaled to
    the chosen mean by the ratio ``volatility / volatility_mean`` so a fitted mean keeps
    the historical weekly shape rather than the hand-built projection's level.
    """

    mean: tuple[str, ...] = ("fitted_ppg", "proj_ppg", "ppg")
    games: tuple[str, ...] = ("fitted_games",)
    availability: tuple[str, ...] = ("projected_availability",)
    volatility: tuple[str, ...] = ("projected_volatility", "volatility")
    volatility_mean: tuple[str, ...] = ("proj_ppg", "ppg")

    @staticmethod
    def _first(frame: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
        for name in candidates:
            if name in frame.columns and frame[name].is_not_null().any():
                return name
        return None

    def resolve(self, frame: pl.DataFrame) -> dict[str, str | None]:
        return {
            "mean": self._first(frame, self.mean),
            "games": self._first(frame, self.games),
            "availability": self._first(frame, self.availability),
            "volatility": self._first(frame, self.volatility),
            "volatility_mean": self._first(frame, self.volatility_mean),
        }


def team_strengths(
    tagged_board: pl.DataFrame,
    espn_players: pl.DataFrame,
    team_ids: Iterable[int],
    *,
    season_games: int = 17,
    fallback_availability: float = 0.94,
    columns: RosterProjectionColumns | None = None,
) -> dict[int, dict[str, int | float | str | None]]:
    """Rank rosters on the canonical board metric and simulate weekly production.

    Each scenario independently samples player availability, optimizes the legal lineup
    from every active player—including the bench—and retains the selected players'
    conditional means and variances. The law of total variance then combines ordinary
    scoring volatility with the extra risk created by absences and weaker replacements.

    The per-player inputs come from ``columns`` (config ``roster_*_columns``), so the
    roster-level numbers follow whichever projection the backtest chose rather than a
    hard-wired field. ``team_rank`` and ``team_score`` use the same canonical value as
    the player board and head-to-head comparison (V2 overall VOR when available), so a
    newly selected ranker cannot leave the league power board on an older metric.
    Expected points, floor, and risk remain separately visible simulation outputs.
    """

    ids = [int(team_id) for team_id in team_ids]
    fallback_players = {team_id: 0 for team_id in ids}
    if not ids or espn_players.height == 0:
        return {
            team_id: {
                "team_rank": rank,
                "team_score": 5.0,
                "ranking_total": 0.0,
                "expected_lineup_vor": None,
                "lineup_vor_risk": None,
                "risk_adjusted_total": None,
                "availability_floor_points": 0.0,
                "availability_spread_points": 0.0,
                "ranking_metric": "unavailable",
                "scored_players": 0,
                "fallback_players": 0,
                "expected_weekly_points": 0.0,
                "weekly_risk": 0.0,
                "weekly_floor": 0.0,
                "lineup_coverage": 0.0,
                "bench_rescue_points": 0.0,
            }
            for rank, team_id in enumerate(ids, start=1)
        }

    resolved = (columns or RosterProjectionColumns()).resolve(tagged_board)
    mean_column = resolved["mean"] or "ppg"
    games_column = resolved["games"]
    availability_column = resolved["availability"]
    volatility_column = resolved["volatility"]
    volatility_mean_column = resolved["volatility_mean"]
    ranking_column = _board_value_column(tagged_board)
    board_players: dict[int, TeamPlayerProjection] = {}
    if ESPN_ID in tagged_board.columns and mean_column in tagged_board.columns:
        wanted = [ESPN_ID, "player_display_name", "position", mean_column, ranking_column]
        wanted += [
            c
            for c in (games_column, availability_column, volatility_column, volatility_mean_column)
            if c and c in tagged_board.columns
        ]
        for row in tagged_board.select(list(dict.fromkeys(wanted))).iter_rows(named=True):
            espn_id = row.get(ESPN_ID)
            mean = row.get(mean_column)
            if espn_id is None or not isinstance(mean, int | float) or not math.isfinite(mean):
                continue
            mean = max(float(mean), 0.0)
            volatility = 0.0
            raw_volatility = row.get(volatility_column) if volatility_column else None
            if isinstance(raw_volatility, int | float) and math.isfinite(raw_volatility):
                volatility = max(float(raw_volatility), 0.0)
                # Keep the weekly shape (volatility relative to its own mean) but scale
                # it to the mean the roster is actually valued on.
                shape_mean = row.get(volatility_mean_column) if volatility_mean_column else None
                if (
                    volatility_mean_column != mean_column
                    and isinstance(shape_mean, int | float)
                    and math.isfinite(shape_mean)
                    and shape_mean > 0
                ):
                    volatility = volatility / float(shape_mean) * mean
            availability = 1.0
            games = row.get(games_column) if games_column else None
            if isinstance(games, int | float) and math.isfinite(games):
                availability = float(games) / float(season_games)
            else:
                raw_availability = row.get(availability_column) if availability_column else None
                if isinstance(raw_availability, int | float) and math.isfinite(raw_availability):
                    availability = float(raw_availability)
            board_players[int(espn_id)] = TeamPlayerProjection(
                player_id=int(espn_id),
                name=str(row.get("player_display_name") or ""),
                position=str(row.get("position") or "").upper(),
                mean=mean,
                volatility=volatility,
                availability=max(0.0, min(1.0, availability)),
                ranking_value=(
                    float(row[ranking_column])
                    if isinstance(row.get(ranking_column), int | float)
                    and math.isfinite(row[ranking_column])
                    else None
                ),
                fallback=False,
            )

    volatility_ratios = _position_volatility_ratios(tagged_board)
    fallback_availability = max(0.0, min(1.0, fallback_availability))
    requirements = _lineup_requirements(espn_players)
    rosters: dict[int, list[TeamPlayerProjection]] = {team_id: [] for team_id in ids}

    for row in espn_players.iter_rows(named=True):
        owner_id = row.get(OWNER_TEAM_ID)
        position = str(row.get("position") or "").upper()
        if (
            owner_id is None
            or int(owner_id) not in rosters
            or position not in {"QB", "RB", "WR", "TE"}
        ):
            continue

        team_id = int(owner_id)
        espn_id = row.get(ESPN_ID)
        player = board_players.get(int(espn_id)) if espn_id is not None else None
        if player is None:
            projected_points = row.get("projected_points")
            if not isinstance(projected_points, int | float) or not math.isfinite(projected_points):
                continue
            # ESPN supplies a season total, which already includes missed time.
            # Convert to a conditional mean before drawing availability; otherwise
            # the fallback prior discounts that season total a second time.
            mean = (
                max(float(projected_points) / (max(season_games, 1) * fallback_availability), 0.0)
                if fallback_availability > 0
                else 0.0
            )
            player = TeamPlayerProjection(
                player_id=int(espn_id or 0),
                name=str(row.get("player_display_name") or ""),
                position=position,
                mean=mean,
                volatility=mean * volatility_ratios.get(position, 0.45),
                availability=fallback_availability,
                # ESPN's projection is a useful fallback for weekly points, but it is
                # not on the model's VOR scale and must not silently enter Power rank.
                ranking_value=None,
                fallback=True,
            )
            fallback_players[team_id] += 1
        rosters[team_id].append(player)

    simulation_results: dict[int, dict[str, float | int]] = {}
    for team_id, roster in rosters.items():
        players = sorted(roster, key=lambda player: player.mean, reverse=True)
        baseline, _ = _best_lineup(players, [True] * len(players), requirements)
        ranked_players = sorted(
            (player for player in roster if player.ranking_value is not None),
            key=lambda player: float(player.ranking_value or 0.0),
            reverse=True,
        )
        ranked_lineup, _ = _best_lineup(ranked_players, [True] * len(ranked_players), requirements)
        ranking_total = sum(
            float(ranked_players[index].ranking_value or 0.0) for index in ranked_lineup
        )
        rng = random.Random(20_260_829 + team_id)
        conditional_mean_sum = 0.0
        conditional_mean_square_sum = 0.0
        conditional_variance_sum = 0.0
        bench_rescue_sum = 0.0
        complete_count = 0
        availability_means = []
        for _ in range(TEAM_SIMULATIONS):
            active = [rng.random() < player.availability for player in players]
            selected, complete = _best_lineup(players, active, requirements)
            # VOR already incorporates expected availability. Keep it on the
            # static power board, never multiply it by another availability draw.
            conditional_mean = sum(players[index].mean for index in selected)
            availability_means.append(conditional_mean)
            conditional_variance = sum(players[index].volatility ** 2 for index in selected)
            conditional_mean_sum += conditional_mean
            conditional_mean_square_sum += conditional_mean**2
            conditional_variance_sum += conditional_variance
            bench_rescue_sum += sum(
                players[index].mean for index in selected if index not in baseline
            )
            complete_count += int(complete)

        expected = conditional_mean_sum / TEAM_SIMULATIONS
        availability_variance = max(
            conditional_mean_square_sum / TEAM_SIMULATIONS - expected**2,
            0.0,
        )
        weekly_variance = conditional_variance_sum / TEAM_SIMULATIONS + availability_variance
        risk = math.sqrt(max(weekly_variance, 0.0))
        simulation_results[team_id] = {
            "ranking_total": ranking_total,
            "availability_floor_points": statistics.quantiles(
                availability_means, n=4, method="inclusive"
            )[0],
            "availability_spread_points": math.sqrt(availability_variance),
            "expected_weekly_points": expected,
            "weekly_risk": risk,
            "weekly_floor": max(expected - 0.674 * risk, 0.0),
            "lineup_coverage": complete_count / TEAM_SIMULATIONS,
            "bench_rescue_points": bench_rescue_sum / TEAM_SIMULATIONS,
            "scored_players": len(players),
            "fallback_players": fallback_players[team_id],
        }

    values = [float(simulation_results[team_id]["ranking_total"]) for team_id in ids]
    mean = sum(values) / len(values)
    deviation = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    ordered = sorted(
        ids,
        key=lambda team_id: (
            -float(simulation_results[team_id]["ranking_total"]),
            -float(simulation_results[team_id]["expected_weekly_points"]),
            team_id,
        ),
    )

    result: dict[int, dict[str, int | float | str | None]] = {}
    for rank, team_id in enumerate(ordered, start=1):
        expected = float(simulation_results[team_id]["expected_weekly_points"])
        ranking_total = float(simulation_results[team_id]["ranking_total"])
        z_score = (ranking_total - mean) / deviation if deviation > 0 else 0.0
        rating = max(0.0, min(10.0, 5.0 + 1.5 * z_score))
        result[team_id] = {
            **simulation_results[team_id],
            "team_rank": rank,
            "team_score": round(rating, 1),
            "ranking_total": round(ranking_total, 1),
            # Deprecated compatibility keys: there is no second VOR risk score.
            "expected_lineup_vor": None,
            "lineup_vor_risk": None,
            "risk_adjusted_total": None,
            "availability_floor_points": round(
                float(simulation_results[team_id]["availability_floor_points"]), 1
            ),
            "availability_spread_points": round(
                float(simulation_results[team_id]["availability_spread_points"]), 1
            ),
            "ranking_metric": ranking_column,
            "expected_weekly_points": round(expected, 1),
            "weekly_risk": round(float(simulation_results[team_id]["weekly_risk"]), 1),
            "weekly_floor": round(float(simulation_results[team_id]["weekly_floor"]), 1),
            "lineup_coverage": round(float(simulation_results[team_id]["lineup_coverage"]), 3),
            "bench_rescue_points": round(
                float(simulation_results[team_id]["bench_rescue_points"]), 1
            ),
        }
    return result


def unrankable_players(
    espn_players: pl.DataFrame,
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE"),
) -> pl.DataFrame:
    """ESPN players the board could not rank — rookies, and anyone with no prior tape.

    Surfaced rather than dropped. These are exactly the players the model is blind to,
    and a wire report that silently omits a hyped rookie is worse than one that says it
    cannot price him.
    """
    unrankable = (
        pl.col("espn_fallback")
        if "espn_fallback" in espn_players.columns
        else pl.col("player_id").is_null()
    )
    return (
        espn_players.filter(unrankable & pl.col("position").is_in(list(positions)))
        .sort(PERCENT_OWNED, descending=True, nulls_last=True)
        .select(
            "player_display_name",
            "position",
            "espn_team",
            PERCENT_OWNED,
            OWNER_TEAM_NAME,
            INJURY_STATUS,
        )
    )


def draft_analysis(
    tagged_board: pl.DataFrame,
    draft: list[Any],
    my_team_id: int | None = None,
    reach_threshold: int = 15,
) -> dict[str, Any]:
    """Score every draft pick against the board, with exact best-available at each pick.

    ``value_vs_board`` is pick number minus the board's overall rank (positive: the
    player went later than the board values him). ``value_vs_market`` is the same
    against ESPN's draft-room rank. ``best_available`` lists the top board players who
    had not yet been taken when the pick was made — exact, because the recap gives the
    full order. Team grades sum the VOR the roster captured and its board value.
    """
    if not draft:
        return {"picks": [], "teams": [], "ranking_metric": _board_value_column(tagged_board)}
    value_column = _board_value_column(tagged_board)
    columns = [
        c
        for c in (
            ESPN_ID,
            "player_id",
            "player_display_name",
            "position",
            "rank",
            "v2_position_rank",
            value_column,
            "espn_draft_rank",
            "rank_source",
            "market_ecr",
            "market_position",
        )
        if c in tagged_board.columns
    ]
    board = tagged_board.select(columns).filter(pl.col(ESPN_ID).is_not_null())
    by_espn = {int(row[ESPN_ID]): row for row in board.iter_rows(named=True)}
    ranked = sorted(
        (row for row in board.iter_rows(named=True) if row.get("rank") is not None),
        key=lambda row: int(row["rank"]),
    )
    taken: set[int] = set()
    picks: list[dict[str, Any]] = []
    per_team: dict[int, dict[str, Any]] = {}
    for pick in draft:
        row = by_espn.get(int(pick.espn_id))
        rank = row.get("rank") if row else None
        value = row.get(value_column) if row else None
        espn_rank = row.get("espn_draft_rank") if row else None
        available = [
            r
            for r in ranked
            if int(r[ESPN_ID]) not in taken and int(r[ESPN_ID]) != int(pick.espn_id)
        ][:3]
        vs_board = (pick.overall - int(rank)) if rank is not None else None
        vs_market = (pick.overall - int(espn_rank)) if espn_rank is not None else None
        verdict = "unrated"
        if vs_board is not None:
            verdict = (
                "steal"
                if vs_board >= reach_threshold
                else "reach"
                if vs_board <= -reach_threshold
                else "fair"
            )
        record = {
            "overall": pick.overall,
            "round": pick.round,
            "round_pick": pick.round_pick,
            "team_id": pick.team_id,
            "team_name": pick.team_name,
            "is_mine": my_team_id is not None and pick.team_id == my_team_id,
            "espn_id": pick.espn_id,
            "player_display_name": pick.player_display_name,
            "position": row.get("position") if row else None,
            "board_rank": rank,
            "position_rank": row.get("v2_position_rank") if row else None,
            "board_value": None if value is None else round(float(value), 2),
            "espn_draft_rank": espn_rank,
            "rank_source": row.get("rank_source") if row else None,
            "market_ecr": row.get("market_ecr") if row else None,
            "market_position": row.get("market_position") if row else None,
            "value_vs_board": vs_board,
            "value_vs_market": vs_market,
            "verdict": verdict,
            "best_available": [
                {
                    "player_display_name": r["player_display_name"],
                    "position": r["position"],
                    "board_rank": r["rank"],
                    "board_value": None
                    if r.get(value_column) is None
                    else round(float(r[value_column]), 2),
                }
                for r in available
            ],
            "best_available_gap": (
                None
                if not available or value is None or available[0].get(value_column) is None
                else round(float(available[0][value_column]) - float(value), 2)
            ),
        }
        picks.append(record)
        taken.add(int(pick.espn_id))
        team = per_team.setdefault(
            pick.team_id,
            {
                "team_id": pick.team_id,
                "team_name": pick.team_name,
                "is_mine": record["is_mine"],
                "picks": 0,
                "rated_picks": 0,
                "captured_value": 0.0,
                "value_vs_board": 0,
                "value_vs_market": 0,
                "steals": 0,
                "reaches": 0,
                "value_left_on_board": 0.0,
            },
        )
        team["picks"] += 1
        if value is not None:
            team["rated_picks"] += 1
            team["captured_value"] += max(float(value), 0.0)
        if vs_board is not None:
            team["value_vs_board"] += vs_board
        if vs_market is not None:
            team["value_vs_market"] += vs_market
        team["steals"] += verdict == "steal"
        team["reaches"] += verdict == "reach"
        if record["best_available_gap"] is not None and record["best_available_gap"] > 0:
            team["value_left_on_board"] += record["best_available_gap"]
    teams = sorted(per_team.values(), key=lambda t: -t["captured_value"])
    for grade, team in enumerate(teams, start=1):
        team["grade_rank"] = grade
        team["captured_value"] = round(team["captured_value"], 1)
        team["value_left_on_board"] = round(team["value_left_on_board"], 1)
    return {"picks": picks, "teams": teams, "ranking_metric": value_column}


def wire_lineup_improvements(
    wire: pl.DataFrame, board: pl.DataFrame, roster_slots: dict[str, int]
) -> pl.DataFrame:
    """Season-equivalent lineup gain from adding a player, before a specific drop.

    Uses all rostered players and legal slots. This is a preseason opportunity value,
    not a current-week recommendation, and does not presume a bench player is dropped.
    """
    from engine.espn.lineup import best_lineup

    value = _projection_ppg_column(board)
    mine = board.filter(pl.col(IS_MINE))
    if not mine.height:
        return wire.with_columns(pl.lit(None, dtype=pl.Float64).alias("lineup_improvement"))
    baseline = best_lineup(mine, roster_slots, value).total
    gains = []
    for row in wire.iter_slices(1):
        augmented = pl.concat([mine, row], how="diagonal_relaxed")
        gains.append(max(0.0, best_lineup(augmented, roster_slots, value).total - baseline))
    return wire.with_columns(pl.Series("lineup_improvement", gains, dtype=pl.Float64))
