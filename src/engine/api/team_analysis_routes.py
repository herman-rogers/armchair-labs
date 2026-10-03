"""Descriptive team correlations; no forecast or simulated win-probability claims."""

from typing import Literal

import duckdb
from fastapi import APIRouter, HTTPException, Query
from polars.exceptions import PolarsError

from engine.config.settings import get_settings
from engine.metrics.team_correlations import analyze_team
from engine.metrics.transaction_events import _CLUBS
from engine.tables.team_analysis import TableVersionChanged, team_session

router = APIRouter(prefix="/api/nextgen", tags=["Team analysis"])


def unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        409,
        f"Team observations unavailable: {exc}",
        headers={"X-Table-Catalog-Stale": "true"} if isinstance(exc, TableVersionChanged) else None,
    )


@router.get("/team-analysis/catalog")
def team_analysis_catalog():
    """The verified table version and coverage used to key browser analysis caches."""
    try:
        with team_session(get_settings().data_dir) as (_, report):
            result = report
        return result
    except (OSError, KeyError, ValueError, TypeError, PolarsError, duckdb.Error) as exc:
        raise unavailable(exc) from exc


@router.get("/team-analysis")
def team_analysis(
    team: str = "LA",
    start: int = Query(2021, ge=2013, le=2100),
    end: int | None = Query(None, ge=2013, le=2100),
    qb: str = "current",
    sample: Literal["starter", "active"] = "starter",
    basis: Literal["season", "raw"] = "season",
    players: str | None = None,
    table_version: str | None = None,
):
    if team not in _CLUBS:
        raise HTTPException(422, "Unknown NFL team")
    selected = None if players is None else [p for p in players.split(",") if p]
    if selected is not None and len(selected) > 8:
        raise HTTPException(422, "Select up to eight players")
    try:
        with team_session(get_settings().data_dir, table_version) as (db, report):
            last = report["season"] if end is None else end
            if start < report["earliest_season"] or start > last or last > report["season"]:
                raise HTTPException(422, "Invalid season range")
            # Restrict the scan to one team and period. QB/participation filters stay
            # in analyze_team: role eligibility must be established before QB filtering.
            panel = db.execute(
                "SELECT * FROM analytics.team_player_games "
                "WHERE team = ? AND season BETWEEN ? AND ? ORDER BY season, week, player_id",
                [team, start, last],
            ).pl()
            try:
                result = analyze_team(
                    panel,
                    team=team,
                    start=start,
                    end=last,
                    qb=qb,
                    sample=sample,
                    basis=basis,
                    selected=selected,
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
    except (OSError, KeyError, ValueError, TypeError, PolarsError, duckdb.Error) as exc:
        raise unavailable(exc) from exc
    return dict(
        **result,
        teams=[dict(id=k, name=v) for k, v in sorted(_CLUBS.items(), key=lambda x: x[1])],
        report=report,
    )
