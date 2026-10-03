"""Read-only college pathways, audited identities, and unpromoted backtests."""

from __future__ import annotations

import polars as pl
from fastapi import APIRouter, HTTPException, Query

from engine.api.college_sources import load_college
from engine.api.outlook_sources import load_outlook
from engine.api.pagination import sorted_page
from engine.data.frames import read_frame

router = APIRouter(prefix="/api/research/college", tags=["college research"])


@router.get("")
def report():
    _, payload = load_college()
    return payload


@router.get("/players")
def players(
    season: int | None = None,
    position: str = "ALL",
    search: str = "",
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = 0,
    sort: str = "",
):
    root, info = load_college()
    frame = read_frame(root / "nfl_cohort.parquet")
    if season is not None:
        frame = frame.filter(pl.col("forecast_year") == season)
    if position != "ALL":
        frame = frame.filter(pl.col("position") == position)
    if search:
        frame = frame.filter(
            pl.col("player_display_name")
            .str.to_lowercase()
            .str.contains(search.lower(), literal=True)
        )
    prediction = read_frame(root / "nfl_predictions.parquet")
    for target, prefix in [
        ("nfl_year1_points", "points"),
        ("nfl_year1_games", "games"),
        ("nfl_first3_points", "first3"),
    ]:
        frame = frame.join(
            prediction.filter(pl.col("target") == target).select(
                "player_id",
                "forecast_year",
                *[
                    pl.col(model).alias(f"{prefix}_{model}")
                    for model in ["draft_only", "college_only", "college_plus_draft"]
                ],
            ),
            on=["player_id", "forecast_year"],
            how="left",
            validate="1:1",
        )
    return {
        "version": info["version"],
        "total": frame.height,
        "players": sorted_page(
            frame,
            sort=sort,
            allowed={
                "player_display_name",
                "forecast_year",
                "position",
                "college_team",
                "link_methods",
                "points_college_only",
                "points_draft_only",
                "points_college_plus_draft",
                "nfl_year1_points",
                "eligible",
                "draft_pick",
                "latest_receiving_yards",
            },
            default="-points_college_plus_draft",
            tie=["player_id", "forecast_year"],
            offset=offset,
            limit=limit,
        ).to_dicts(),
    }


@router.get("/identities")
def identities(
    search: str = "",
    status: str = "all",
    position: str = "ALL",
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = 0,
    sort: str = "",
):
    root, _ = load_college()
    frame = read_frame(root / "identity_links.parquet")
    if position != "ALL":
        matches = (
            pl.scan_parquet(root / "college_seasons.parquet")
            .filter(pl.col("college_position") == position)
            .select("college_id")
            .unique()
            .collect()
        )
        frame = frame.join(matches, on="college_id", how="semi")
    if search:
        frame = frame.filter(
            pl.col("college_name").str.to_lowercase().str.contains(search.lower(), literal=True)
        )
    if status != "all":
        if status not in {"linked", "review", "unmatched"}:
            raise HTTPException(422, "Unknown identity status")
        frame = frame.filter(pl.col("status") == status)
    frame = frame.with_columns(
        pl.col("status")
        .replace_strict({"linked": 0, "review": 1, "unmatched": 2}, default=3)
        .alias("_priority")
    )
    return {
        "total": frame.height,
        "players": sorted_page(
            frame,
            sort=sort,
            allowed={
                "college_name",
                "college_id",
                "first_college_season",
                "last_college_season",
                "status",
                "nfl_name",
                "method",
            },
            default="college_name",
            tie=["college_id"],
            offset=offset,
            limit=limit,
        )
        .drop("_priority")
        .to_dicts(),
    }


@router.get("/career")
def career(player_id: str | None = None, college_id: str | None = None):
    if bool(player_id) == bool(college_id):
        raise HTTPException(422, "Supply exactly one NFL player_id or college_id")
    root, info = load_college()
    links = read_frame(root / "identity_links.parquet")
    if player_id:
        matches = links.filter((pl.col("player_id") == player_id) & (pl.col("status") == "linked"))
    else:
        matches = links.filter(pl.col("college_id") == college_id)
        if matches.height and matches["status"][0] == "linked":
            player_id = matches["player_id"][0]
    ids = matches["college_id"].to_list()
    college = read_frame(root / "college_seasons.parquet").filter(pl.col("college_id").is_in(ids))
    forecasts = read_frame(root / "nfl_predictions.parquet").filter(
        pl.col("player_id") == player_id
    )
    nfl = read_frame(root / "inputs/nfl_seasons.parquet").filter(pl.col("player_id") == player_id)
    college_predictions = read_frame(root / "college_predictions.parquet").filter(
        pl.col("college_id").is_in(ids)
    )
    known_candidate = False
    if player_id:
        known_candidate = (
            read_frame(root / "nfl_cohort.parquet").filter(pl.col("player_id") == player_id).height
            > 0
        )
    if not matches.height and not nfl.height and not known_candidate:
        raise HTTPException(404, "No college/NFL career found for this identity")
    current = None
    outlook_status = "No linked current outlook"
    if player_id:
        try:
            outlook = load_outlook(info["current_forecast_year"])
            current = next((r for r in outlook["players"] if r["player_id"] == player_id), None)
            outlook_status = (
                f"Separate {outlook['version']}; through Week {outlook['through_week']}"
            )
        except HTTPException as exc:
            outlook_status = f"Current outlook unavailable: {exc.detail}"
    return {
        "version": info["version"],
        "links": matches.to_dicts(),
        "college_seasons": college.sort("season", "team_id").to_dicts(),
        "nfl_seasons": nfl.sort("season").to_dicts(),
        "nfl_forecasts": forecasts.sort("forecast_year", "target").to_dicts(),
        "college_forecasts": college_predictions.sort("forecast_year", "position").to_dicts(),
        "current_outlook": current,
        "current_outlook_status": outlook_status,
        "warning": "College/preseason forecasts and current NFL outlook have different "
        "cutoffs and horizons; do not average them.",
    }
