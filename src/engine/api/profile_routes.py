"""One player profile across observed NFL seasons and verified college identities."""

from __future__ import annotations

import json

import polars as pl
from fastapi import APIRouter, HTTPException, Query

from engine.api.profile_sources import load_profiles
from engine.api.research_sources import _inside
from engine.config.settings import get_settings
from engine.data.profile_forecasts import approved_profile_forecasts
from engine.data.serving import read_model
from engine.metrics.player_profile import (
    ROLE_LABELS,
    before,
    career_features,
    college_history,
    nfl_seasons,
    role_band,
    role_periods,
    summarize,
    transitions,
)
from engine.tables.application import read_frame

router = APIRouter(prefix="/api/profiles", tags=["player profiles"])


@router.get("")
@read_model("profiles")
def players(
    search: str = "",
    position: str = "ALL",
    scope: str = "current",
    population: str = "all",
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    root, report = load_profiles()
    if scope not in {"current", "all", "college_linked"}:
        raise HTTPException(422, "Unknown profile scope")
    frame = read_frame(root / "players.parquet")
    if scope == "current":
        frame = frame.filter(pl.col("current_candidate"))
    elif scope == "college_linked":
        frame = frame.filter(pl.col("college_linked"))
    if population == "rookie":
        frame = frame.filter(pl.col("rookie_season") == report["season"])
    elif population == "returner":
        frame = frame.filter(pl.col("rookie_season") < report["season"])
    elif population != "all":
        raise HTTPException(422, "Unknown profile population")
    if position != "ALL":
        frame = frame.filter(pl.col("position") == position)
    if search:
        frame = frame.filter(
            pl.col("player_display_name")
            .str.to_lowercase()
            .str.contains(search.lower(), literal=True)
        )
    return {
        "report": report,
        "total": frame.height,
        "players": frame.sort("player_display_name").slice(offset, limit).to_dicts(),
    }


@router.get("/directory")
@read_model("profile-directory")
def directory():
    root, report = load_profiles()
    frame = read_frame(root / "players.parquet").sort("player_display_name")
    return {"report": report, "total": frame.height, "players": frame.to_dicts()}


@router.get("/player")
@read_model("profile")
def profile(
    player_id: str | None = None,
    college_id: str | None = None,
    season: int | None = None,
    week: int | None = Query(None, ge=0, le=18),
    scope: str = "analysis",
):
    if scope not in {"analysis", "research"}:
        raise HTTPException(422, "Unknown profile scope")
    if bool(player_id) == bool(college_id):
        raise HTTPException(422, "Supply exactly one NFL player_id or college_id")
    root, report = load_profiles()
    selected_season = season if season is not None else report["season"]
    selected_week = (
        week
        if week is not None
        else (report["through_week"] if selected_season == report["season"] else 18)
    )
    if not 2001 <= selected_season <= report["season"] or (selected_season, selected_week) > (
        report["season"],
        report["through_week"],
    ):
        raise HTTPException(422, "Profile cutoff exceeds captured observations")
    data = get_settings().data_dir
    college = _inside(data / "research", report["college_version"])
    all_links = read_frame(college / "identity_links.parquet").lazy()
    if college_id:
        links = all_links.filter(pl.col("college_id") == college_id).collect()
        if not links.height:
            raise HTTPException(404, "Unknown college identity")
        if links["status"][0] == "linked":
            player_id = links["player_id"][0]
    else:
        links = all_links.filter(
            (pl.col("player_id") == player_id) & (pl.col("status") == "linked")
        ).collect()
    identity_frame = (
        read_frame(root / "players.parquet")
        .lazy()
        .filter(pl.col("player_id") == player_id)
        .collect()
        if player_id
        else pl.DataFrame()
    )
    if not identity_frame.height and not college_id:
        raise HTTPException(404, "Unknown NFL profile")
    identity = (
        identity_frame.to_dicts()[0]
        if identity_frame.height
        else {
            "player_id": None,
            "player_display_name": links["college_name"][0],
            "position": None,
            "college_linked": links["status"][0] == "linked",
            "rookie_season": None,
        }
    )
    college_ids = links["college_id"].to_list()
    college_rows = college_history(
        read_frame(college / "college_seasons.parquet")
        .lazy()
        .filter(pl.col("college_id").is_in(college_ids))
        .collect()
        .to_dicts(),
        selected_season,
    )
    weeks = (
        before(
            read_frame(root / "nfl_weeks.parquet")
            .lazy()
            .filter(pl.col("player_id") == player_id)
            .collect()
            .to_dicts(),
            selected_season,
            selected_week,
        )
        if player_id
        else []
    )
    seasons = nfl_seasons(weeks, selected_season, selected_week)
    complete_selected = selected_season < report["season"] and selected_week == 18
    if complete_selected:
        for row in seasons:
            row["partial"] = False
    periods = role_periods(weeks, selected_season, selected_week)
    role_history = [
        {
            "role": role,
            "label": label,
            **summarize([r for r in weeks if role_band(r.get("snap_share")) == role]),
        }
        for role, label in ROLE_LABELS.items()
    ]
    injuries = (
        read_frame(root / "injuries.parquet")
        .lazy()
        .filter(pl.col("player_id") == player_id)
        .collect()
        .to_dicts()
        if player_id
        else []
    )
    injuries = before(injuries, selected_season, selected_week)
    tracking = []
    if player_id and report["schema_version"] >= 2:
        tracking = (
            read_frame(root / "tracking_seasons.parquet")
            .lazy()
            .filter(
                (pl.col("player_id") == player_id)
                & (pl.col("season") < selected_season + int(complete_selected))
            )
            .sort("season")
            .collect()
            .to_dicts()
        )
    try:
        approved_forecasts, analysis_version = approved_profile_forecasts(
            data, report, player_id, selected_season, selected_week
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"Approved profile forecasts unavailable: {exc}") from exc
    archived_forecasts = scope == "research"
    forecast_rows, translation, development, current = [], [], [], None
    if archived_forecasts:
        historical = _inside(data / "research", report["history"]["version"])
        forecasts = (
            read_frame(historical / "outputs/metric_backtest_predictions.parquet")
            .lazy()
            .filter(
                (pl.col("player_id") == player_id) & (pl.col("forecast_season") <= selected_season)
            )
            .collect()
            if player_id
            else pl.DataFrame()
        )
        forecast_fields = [
            "forecast_season",
            "forecast_cutoff_date",
            "fitted_ppg",
            "fitted_games",
            "fitted_season_points",
            "fitted_nextgen_season_points",
            "market_snapshot",
            "market_ecr",
            "cutoff_preseason_team",
            "cutoff_preseason_status",
            "cutoff_state_resolution",
            "cutoff_source_url",
            "cutoff_evidence_clause",
            "combine_speed_score",
            "combine_burst_score",
        ]
        # Week zero means preseason, but has no day cutoff: omit preseason forecasts
        # for that year instead of pretending an August forecast was known in January.
        if selected_week == 0 and forecasts.height:
            forecasts = forecasts.filter(pl.col("forecast_season") < selected_season)
        forecast_rows = (
            forecasts.select([c for c in forecast_fields if c in forecasts.columns])
            .sort("forecast_season")
            .to_dicts()
            if forecasts.height
            else []
        )
        translation = (
            read_frame(college / "nfl_predictions.parquet")
            .lazy()
            .filter(
                (pl.col("player_id") == player_id)
                & (
                    pl.col("forecast_year") < selected_season
                    if selected_week == 0
                    else pl.col("forecast_year") <= selected_season
                )
            )
            .collect()
            .to_dicts()
            if player_id
            else []
        )
        for row in translation:
            # Outcomes need their complete maturation window, not just a forecast date.
            end = row["forecast_year"] + (2 if row["target"] == "nfl_first3_points" else 0)
            if end > selected_season or (end == selected_season and not complete_selected):
                row["actual"] = None
        development = (
            read_frame(college / "college_predictions.parquet")
            .lazy()
            .filter(
                pl.col("college_id").is_in(college_ids)
                & (
                    pl.col("forecast_year") < selected_season
                    if selected_week == 0
                    else pl.col("forecast_year") <= selected_season
                )
            )
            .sort("forecast_year", "position")
            .collect()
            .to_dicts()
        )
        for row in development:
            if row["forecast_year"] >= selected_season:
                row["actual"] = None
        current = None
        if (selected_season, selected_week) == (report["season"], report["through_week"]):
            outlook = json.loads(
                (_inside(data / "research", report["outlook_version"]) / "outlook.json").read_text()
            )
            current = next((r for r in outlook["players"] if r["player_id"] == player_id), None)
    missing = []
    if not college_rows:
        missing.append("No linked college seasons before this cutoff.")
    if identity.get("history_left_truncated"):
        missing.append(
            "The NFL record starts after the recorded entry year; earlier seasons are missing."
        )
    if any(r.get("snap_share") is None for r in weeks):
        missing.append("Some observed NFL weeks lack offensive snap evidence.")
    if any(not r["complete_team_season"] for r in college_rows):
        missing.append("Some college stints have incomplete coverage or unreconciled team totals.")
    if archived_forecasts and not current:
        missing.append("No current four-week outlook at this profile cutoff.")
    if any(r["status"] != "linked" for r in links.to_dicts()):
        missing.append(
            "College identity has no accepted NFL link; candidate matches are not merged."
        )
    return {
        "version": report["version"],
        "identity": identity,
        "cutoff": {"season": selected_season, "week": selected_week},
        "available_through": {"season": report["season"], "week": report["through_week"]},
        "sources": {
            "gold": report.get("gold", {}).get("version"),
            "history": report["history"]["version"],
            "college": report["college_version"],
            "outlook": report["outlook_version"],
            "analysis": analysis_version,
            "observations_saved_at": report["observations_saved_at"],
        },
        "links": links.to_dicts(),
        "college_seasons": college_rows,
        "nfl_seasons": seasons,
        "career": summarize(weeks),
        "features": career_features(
            weeks,
            selected_season + int(complete_selected),
            0 if complete_selected else selected_week,
        ),
        "role_history": role_history,
        "role_periods": periods,
        "transitions": [
            event
            for event in transitions(seasons, college_rows, identity)
            if event["season"] <= selected_season
        ],
        "injury_reports": injuries,
        "tracking_seasons": tracking,
        "tracking": report.get("tracking"),
        "approved_forecasts": approved_forecasts,
        "preseason_forecasts": forecast_rows,
        "college_forecasts": translation,
        "college_development_forecasts": development,
        "current_outlook": current,
        "forecast_scope": "archived_research" if scope == "research" else "approved_analysis",
        "missing_evidence": missing,
        "metrics": report["metrics"],
        "limits": report["limits"],
    }


@router.get("/similar")
@read_model("similar")
def similar_players(
    player_id: str, season: int | None = None, week: int | None = Query(None, ge=0, le=18)
):
    from engine.metrics.career_similarity import career_neighbors

    root, report = load_profiles()
    selected_season = season if season is not None else report["season"]
    selected_week = (
        week
        if week is not None
        else (report["through_week"] if selected_season == report["season"] else 18)
    )
    if not 2001 <= selected_season <= report["season"] or (selected_season, selected_week) > (
        report["season"],
        report["through_week"],
    ):
        raise HTTPException(422, "Similarity cutoff exceeds captured observations")
    identities = read_frame(root / "players.parquet")
    if player_id not in identities["player_id"]:
        raise HTTPException(404, "Unknown NFL profile")
    complete_through = (
        selected_season
        if selected_season < report["season"] and selected_week == 18
        else selected_season - 1
    )
    result = career_neighbors(
        identities, read_frame(root / "nfl_weeks.parquet"), player_id, complete_through
    )
    return {
        "version": report["version"],
        "cutoff": {"season": selected_season, "week": selected_week},
        **result,
    }
