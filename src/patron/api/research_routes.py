"""Player-level views of retained, walk-forward research forecasts and observed outcomes."""

from __future__ import annotations

import json
import math
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import polars as pl
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from patron.api.league_routes import ServiceDep, _envelope, _state
from patron.api.outlook_sources import load_outlook
from patron.api.research_sources import dataset_info, list_datasets, output_path, public_info
from patron.config.league import get_league
from patron.config.settings import CONFIG_DIR, get_settings
from patron.espn.lineup import best_lineup
from patron.metrics.backtest import MetricReportConfig
from patron.metrics.prospective import load_verified_snapshot

router = APIRouter(prefix="/api/research", tags=["research"])


@router.get("/data-catalog")
def data_catalog() -> dict:
    from patron.data.catalog import catalog_info

    try:
        return catalog_info(get_settings().data_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"Canonical data catalog is unavailable: {exc}") from exc


def prediction_path(dataset: str = "legacy") -> Path:
    return output_path(dataset, "metric_backtest_predictions.parquet")


def model_unit(name: str, specs: dict | None = None) -> str:
    if specs is None:
        specs = {spec.name: asdict(spec) for spec in MetricReportConfig.from_config().fit.models}

    def unit(key: str, seen: frozenset[str]) -> str:
        if key in seen:
            return "score"
        if key == "frozen_adaptive_2026":
            return "season points"
        spec = specs.get(key)
        if spec is None:
            return (
                "active-game PPG" if key in {"proj_ppg", "ppg", "historical_ppg_prior"} else "score"
            )
        if spec["kind"] == "ridge":
            return {
                "actual_ppg": "active-game PPG",
                "actual_games": "games",
                "actual_season_points": "season points",
                "actual_played": "probability",
            }.get(spec["target"], "score")
        units = {
            unit(factor, seen | {key})
            for factor in (*spec.get("factors", []), *(v for _, v in spec.get("by_position", [])))
        }
        if spec["kind"] == "product" and "active-game PPG" in units and "games" in units:
            return "season points"
        return next(iter(units)) if len(units) == 1 else "score"

    return unit(name, frozenset())


@lru_cache(maxsize=2)
def _catalog(path: str, modified: int, size: int, model_names: tuple[str, ...]) -> dict[str, Any]:
    frame = pl.scan_parquet(path)
    schema = frame.collect_schema()
    models = [name for name in model_names if name in schema]
    models += [name for name in ("proj_ppg", "historical_ppg_prior", "ppg") if name in schema]
    seasons = (
        frame.group_by("forecast_season")
        .agg(
            pl.col("outcome_complete").all().alias("complete"),
            *(pl.col(name).is_not_null().any().alias(name) for name in models),
            *(
                [
                    pl.col("forecast_cutoff_date")
                    .cast(pl.String)
                    .drop_nulls()
                    .unique()
                    .sort()
                    .alias("cutoff_dates")
                ]
                if "forecast_cutoff_date" in schema
                else []
            ),
        )
        .collect()
        .sort("forecast_season", descending=True)
        .to_dicts()
    )
    return {
        "saved_at": datetime.fromtimestamp(modified / 1e9, UTC).isoformat(),
        "seasons": [
            {
                "season": row["forecast_season"],
                "complete": row["complete"],
                "cutoff_dates": row.get("cutoff_dates", []),
            }
            for row in seasons
        ],
        "models": [
            {"id": name, "seasons": [row["forecast_season"] for row in seasons if row[name]]}
            for name in models
        ],
    }


@router.get("/catalog")
def catalog(dataset: str = "legacy") -> dict[str, Any]:
    source = dataset_info(dataset)
    specs = {spec["name"]: spec for spec in source.get("model_specs", [])}
    if not specs:
        specs = {spec.name: asdict(spec) for spec in MetricReportConfig.from_config().fit.models}
    path = prediction_path(dataset)
    stat = path.stat()
    data = _catalog(str(path), stat.st_mtime_ns, stat.st_size, tuple(specs))
    # The immutable experiment stays distinct from recomputed selector columns.
    models = list(data["models"])
    snapshot = get_settings().static_dir / "experimental_2026_predictions.csv"
    if (
        dataset == "legacy"
        and snapshot.exists()
        and (CONFIG_DIR / "experimental_freeze_2026.yaml").exists()
    ):
        models.append({"id": "frozen_adaptive_2026", "seasons": [2026]})
    return {
        **data,
        "dataset": public_info(source),
        "models": [{**model, "unit": model_unit(model["id"], specs)} for model in models],
    }


@router.get("/sources")
def sources() -> dict[str, Any]:
    return list_datasets()


@router.get("/report")
def report(dataset: str = "legacy") -> FileResponse:
    # Never substitute the newest unrelated production report for this dataset.
    return FileResponse(output_path(dataset, "metric_report.json"), media_type="application/json")


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def compare_rows(
    rows: list[dict[str, Any]], model: str, target: str, benchmark: str
) -> list[dict[str, Any]]:
    """Rank the complete selected population before search or missing-score filters."""
    result = []
    for row in rows:
        result.append(
            {
                "player_id": row["player_id"],
                "player_display_name": row.get("player_display_name") or row["player_id"],
                "position": row.get("position"),
                "team": row.get("projected_team") or row.get("team"),
                "population": row.get("player_population") or "unknown",
                "model_value": _finite(row.get(model)),
                "actual_value": _finite(row.get(target)) if row.get("outcome_complete") else None,
                "benchmark_value": _finite(row.get(benchmark)),
                "historical_context": row.get("historical_context")
                or {
                    "cutoff_date": row.get("forecast_cutoff_date"),
                    "roster_evidence": row.get("cutoff_state_resolution"),
                    "roster_status": row.get("cutoff_preseason_status"),
                    "prior_games": _finite(row.get("games")),
                    "prior_ppg": _finite(row.get("ppg")),
                    "prior_snap_share": _finite(row.get("source_offense_snap_pct")),
                },
            }
        )
    for value, rank in (
        ("model_value", "model_rank"),
        ("actual_value", "actual_rank"),
        ("benchmark_value", "benchmark_rank"),
    ):
        values = sorted((r[value] for r in result if r[value] is not None), reverse=True)
        ranks: dict[float, int] = {}
        for index, number in enumerate(values, 1):
            ranks.setdefault(number, index)
        for row in result:
            row[rank] = ranks.get(row[value])
    for row in result:
        row["rank_gap"] = (
            row["actual_rank"] - row["model_rank"]
            if row["actual_rank"] is not None and row["model_rank"] is not None
            else None
        )
    return sorted(result, key=lambda row: (row["model_rank"] or math.inf, row["player_id"]))


@router.get("/players")
def players(
    season: int,
    model: str = "fitted_season_points",
    target: Literal["actual_season_points", "actual_ppg"] = "actual_season_points",
    benchmark: Literal["production", "market"] = "production",
    position: Literal["ALL", "QB", "RB", "WR", "TE"] = "ALL",
    population: Literal["all", "returner", "rookie", "market_only"] = "all",
    search: str = "",
    coverage: Literal["all", "scored", "missing"] = "all",
    dataset: str = "legacy",
) -> dict[str, Any]:
    metadata = catalog(dataset)
    models = {row["id"]: row for row in metadata["models"]}
    if model not in models or season not in models[model]["seasons"]:
        raise HTTPException(422, "That research model has no saved forecasts for this season.")
    frame = pl.scan_parquet(prediction_path(dataset))
    schema = frame.collect_schema()
    benchmark_column = (
        "fitted_season_points"
        if benchmark == "production"
        else "market_overall_ecr_score"
        if position == "ALL"
        else "market_ecr_score"
    )
    selected = {
        "player_id",
        "player_display_name",
        "position",
        "projected_team",
        "team",
        "player_population",
        "outcome_complete",
        target,
        benchmark_column,
        "forecast_cutoff_date",
        "cutoff_state_resolution",
        "cutoff_preseason_status",
        "games",
        "ppg",
        "source_offense_snap_pct",
    }
    if model != "frozen_adaptive_2026":
        selected.add(model)
    data = (
        frame.filter(pl.col("forecast_season") == season)
        .select(sorted(selected & set(schema)))
        .collect()
    )
    if model == "frozen_adaptive_2026":
        snapshot, _ = load_verified_snapshot(
            get_settings().static_dir / "experimental_2026_predictions.csv",
            CONFIG_DIR / "experimental_freeze_2026.yaml",
        )
        data = data.join(
            snapshot.select("player_id", pl.col("fitted_adaptive_ppg_hybrid").alias(model)),
            on="player_id",
            how="left",
            validate="1:1",
        )
    if position != "ALL":
        data = data.filter(pl.col("position") == position)
    if population != "all":
        if "player_population" not in data.columns:
            data = data.head(0)
        else:
            data = data.filter(pl.col("player_population") == population)
    rows = compare_rows(data.to_dicts(), model, target, benchmark_column)
    total = len(rows)
    scored = sum(row["model_value"] is not None for row in rows)
    complete = next(row["complete"] for row in metadata["seasons"] if row["season"] == season)
    needle = search.strip().casefold()
    rows = [
        row
        for row in rows
        if needle in row["player_display_name"].casefold()
        and (coverage == "all" or (row["model_value"] is not None) == (coverage == "scored"))
    ]
    return {
        "season": season,
        "model": model,
        "target": target,
        "benchmark": benchmark,
        "complete": complete,
        "saved_at": metadata["saved_at"],
        "dataset": metadata.get("dataset"),
        "pool_size": total,
        "scored_players": scored,
        "players": rows,
        "ranking_basis": (
            "Raw scores within the selected position and population; ties share rank. "
            "Search does not change ranks."
        ),
        "model_unit": models[model].get("unit", model_unit(model)),
        "outcome_status": "Completed season results"
        if complete
        else "Actual results are not yet available in this saved fold.",
    }


@router.get("/league-impact")
def league_impact(
    service: ServiceDep,
    model: str = "fitted_season_points",
    qb: float = Query(1.0, ge=0.0, le=2.0),
    rb: float = Query(1.0, ge=0.0, le=2.0),
    wr: float = Query(1.0, ge=0.0, le=2.0),
    te: float = Query(1.0, ge=0.0, le=2.0),
    missing: Literal["production", "exclude"] = "production",
    dataset: str = "legacy",
) -> dict[str, Any]:
    state = _state(service, "v2")
    config = get_league()
    comparison = players(season=state.snapshot.season, model=model, dataset=dataset)
    unit = comparison["model_unit"]
    if unit not in {"season points", "active-game PPG"}:
        raise HTTPException(422, "Choose a points or PPG model for league lineup comparisons.")
    forecasts = {row["player_id"]: row["model_value"] for row in comparison["players"]}
    weights = {"QB": qb, "RB": rb, "WR": wr, "TE": te}
    results = scenario_teams(
        state.tagged_board,
        state.snapshot.teams,
        state.snapshot.roster_slots,
        forecasts,
        unit,
        weights,
        missing,
        config.metrics.projection_season_games,
    )
    return {
        **_envelope(state),
        "model": model,
        "teams": results,
        "weights": weights,
        "saved_at": comparison["saved_at"],
        "dataset": comparison.get("dataset"),
        "basis": (
            "Best legal lineup, using production value over replacement plus the change "
            "in forecast points per scheduled game. Positional replacement levels and manual "
            "adjustments stay fixed. PPG models use production expected games; missing forecasts "
            "follow the selected fallback. This deterministic scenario does not rerun the "
            "league injury simulation. Saved research folds may differ from the published "
            "production board even when the model name matches. Adjustments are temporary."
        ),
        "missing_policy": missing,
    }


def scenario_teams(board, teams, slots, forecasts, unit, weights, missing, season_games):
    """A pure what-if over current rosters; never writes forecasts or assumptions."""
    results = []
    for team in teams:
        roster = []
        modeled = fallback = uncovered = 0
        for row in board.filter(pl.col("owner_team_id") == team.team_id).to_dicts():
            if row.get("position") not in weights:
                continue
            base = _finite(row.get("forecast_season_points"))
            if "forecast_season_points" not in row:
                base = _finite(row.get("fitted_season_points"))
            base = base / season_games if base is not None else None
            baseline_vor = _finite(row.get("v2_overall_vor"))
            value = _finite(forecasts.get(row["player_id"]))
            if value is not None and unit == "active-game PPG":
                games = _finite(row.get("forecast_expected_games"))
                if "forecast_expected_games" not in row:
                    games = _finite(row.get("fitted_games"))
                value = value * games if games is not None else None
            if base is None or baseline_vor is None:
                value = None
            if value is not None:
                value /= season_games
                scenario_value = baseline_vor + value * weights[row["position"]] - base
                modeled += 1
            elif missing == "production" and baseline_vor is not None:
                scenario_value = baseline_vor
                fallback += 1
            else:
                scenario_value = None
                uncovered += 1
            roster.append(
                {
                    "player_id": str(row["player_id"]),
                    "player_display_name": row["player_display_name"],
                    "position": row["position"],
                    "baseline": baseline_vor,
                    "scenario": scenario_value,
                }
            )
        schema = {
            "player_id": pl.String,
            "player_display_name": pl.String,
            "position": pl.String,
            "baseline": pl.Float64,
            "scenario": pl.Float64,
        }
        frame = pl.DataFrame(roster, schema=schema)
        baseline = best_lineup(frame, slots, "baseline")
        scenario = best_lineup(frame, slots, "scenario")
        results.append(
            {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "wins": team.wins,
                "losses": team.losses,
                "baseline_value": baseline.total,
                "scenario_value": scenario.total,
                "value_change": scenario.total - baseline.total,
                "modeled_players": modeled,
                "fallback_players": fallback,
                "missing_players": uncovered,
                "starters": [
                    {"name": s.player_name, "slot": s.slot, "value": s.value}
                    for s in scenario.starters
                ],
            }
        )
    for value, rank in (("baseline_value", "baseline_rank"), ("scenario_value", "scenario_rank")):
        ranking = {
            v: i
            for i, v in reversed(
                list(enumerate(sorted((r[value] for r in results), reverse=True), 1))
            )
        }
        for row in results:
            row[rank] = ranking[row[value]]
    for row in results:
        row["rank_change"] = row["baseline_rank"] - row["scenario_rank"]
    return sorted(results, key=lambda row: (row["scenario_rank"], row["team_id"]))


@router.get("/current-players")
def current_players(
    service: ServiceDep,
    model: str = "fitted_season_points",
    position: Literal["ALL", "QB", "RB", "WR", "TE"] = "ALL",
    benchmark: Literal["production", "market"] = "production",
    population: Literal["all", "returner", "rookie", "market_only"] = "all",
    dataset: str = "legacy",
) -> dict[str, Any]:
    state = _state(service, "v2")
    data = players(
        season=state.snapshot.season,
        model=model,
        position=position,
        benchmark=benchmark,
        population=population,
        dataset=dataset,
    )
    # Count a player's weekly score once, even if a snapshot contains duplicate entries.
    observations = {}
    for week in state.snapshot.week_lineups:
        if week.week > state.snapshot.week:
            continue
        for entry in [*week.home_lineup, *week.away_lineup]:
            observations[(week.week, entry.espn_id)] = entry.points
    totals: dict[int, float] = {}
    for (_, espn_id), points in observations.items():
        totals[espn_id] = totals.get(espn_id, 0.0) + points
    ids = {
        row["player_id"]: int(row["espn_id"])
        for row in state.tagged_board.to_dicts()
        if row.get("espn_id") is not None
    }
    raw = [
        {
            **row,
            "player_population": row["population"],
            "outcome_complete": True,
            "actual_value": totals.get(ids.get(row["player_id"], -1)),
        }
        for row in data["players"]
    ]
    return {
        **data,
        **_envelope(state),
        "players": compare_rows(raw, "model_value", "actual_value", "benchmark_value"),
        "complete": False,
        "outcome_status": (
            "ESPN points observed in saved league lineups, including bench scores. "
            "Current week may be in progress; unobserved players stay unknown. "
            "Coverage is not the whole NFL."
        ),
    }


@router.get("/rookies")
def rookie_watch(
    service: ServiceDep,
    position: Literal["ALL", "QB", "RB", "WR", "TE"] = "ALL",
) -> dict[str, Any]:
    """Saved NFL rookie intelligence joined to league ownership by ESPN identifier."""
    season = get_league().draft_season
    path = get_settings().outputs_dir / f"rookie_watch_{season}.json"
    if not path.exists():
        raise HTTPException(404, "Rookie watch has not been built for this season yet.")
    data = json.loads(path.read_text())
    if data.get("season") != season:
        raise HTTPException(409, "Rookie watch artifact does not match the current season.")
    state = None
    # Public NFL research remains available if the private league is unavailable.
    with suppress(HTTPException):
        state = _state(service, "v2")
    league_players = (
        {p.espn_id: p for p in state.snapshot.players}
        if state and state.snapshot.season == season
        else {}
    )
    rows = []
    for row in data["players"]:
        if position != "ALL" and row["position"] != position:
            continue
        player = league_players.get(row.get("espn_id"))
        rows.append(
            {
                **row,
                "availability": "unknown"
                if player is None
                else ("free_agent" if player.owner_team_id is None else "rostered"),
                "owner_team_name": player.owner_team_name if player else None,
                "is_mine": bool(
                    player
                    and state
                    and state.snapshot.my_team_id is not None
                    and player.owner_team_id == state.snapshot.my_team_id
                ),
                "injury_status": player.injury_status if player else None,
                "espn_draft_rank": player.espn_draft_rank if player else None,
            }
        )
    generated = datetime.fromisoformat(data["saved_at"])
    return {
        **data,
        "players": rows,
        "forecast_age_hours": max(0, (datetime.now(UTC) - generated).total_seconds() / 3600),
        "ownership_available": bool(league_players),
        "ownership_stale": state.stale if state else True,
        "ownership_captured_at": state.snapshot.captured_at if state else None,
        "newer_week_possible": bool(state and state.snapshot.week > data["through_week"] + 1),
    }


@router.get("/outlook")
def player_outlook(service: ServiceDep) -> dict[str, Any]:
    """Verified in-season outlook; current league context never silently refits it."""
    season = get_league().draft_season
    data = load_outlook(season)
    state = None
    with suppress(HTTPException):
        state = _state(service, "v2")
    if state and state.snapshot.season != season:
        state = None
    players = {p.espn_id: p for p in state.snapshot.players} if state else {}
    rows = []
    for row in data["players"]:
        player = players.get(row.get("espn_id"))
        rows.append(
            {
                **row,
                "availability": "unknown"
                if player is None
                else ("free_agent" if player.owner_team_id is None else "rostered"),
                "owner_team_name": player.owner_team_name if player else None,
                "is_mine": bool(
                    player
                    and state
                    and state.snapshot.my_team_id is not None
                    and player.owner_team_id == state.snapshot.my_team_id
                ),
                "injury_status": player.injury_status if player else None,
            }
        )
    captured = datetime.fromisoformat(data["observations_saved_at"])
    return {
        **data,
        "players": rows,
        "observation_age_hours": max(0, (datetime.now(UTC) - captured).total_seconds() / 3600),
        "ownership_available": bool(players),
        "ownership_stale": state.stale if state else True,
        "ownership_captured_at": state.snapshot.captured_at if state else None,
        "newer_week_possible": bool(state and state.snapshot.week > data["through_week"] + 1),
    }
