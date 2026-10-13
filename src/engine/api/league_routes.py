"""League tools on ESPN observations; forecasts come from whichever campaign is asked for."""

from dataclasses import asdict
from typing import Annotated

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query

from engine.config.settings import get_settings
from engine.espn.observations import weekly_matchups
from engine.espn.service import LeagueService, NotAuthenticatedError, get_service
from engine.tables import observations
from engine.tables.application import connect

router = APIRouter(prefix="/api/league", tags=["league"])

ServiceDep = Annotated[LeagueService, Depends(get_service)]


def snapshot_observations(service, *, force=False):
    try:
        return service.observations(force=True) if force else service.observations()
    except NotAuthenticatedError as exc:
        raise HTTPException(401, str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, str(exc)) from exc


def _tables(exc: Exception) -> HTTPException:
    return HTTPException(409, f"Query tables unavailable: {exc}")


@router.get("/status")
def status(service: ServiceDep):
    return service.status()


@router.post("/refresh")
def refresh(service: ServiceDep):
    from engine.data.refresh import refresh_local

    snapshot, stale, age = snapshot_observations(service, force=True)
    try:
        refresh_local()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(503, f"ESPN refreshed; {exc}") from exc
    return {"captured_at": snapshot.captured_at, "stale": stale, "age_seconds": round(age)}


@router.get("")
def league(service: ServiceDep):
    """Ownership, rosters, transactions and draft, linked to NFL identities."""
    from engine.espn.attention import annotate_players, reviewed_news, roster_attention
    from engine.espn.crosswalk import board_lookups, resolve_player_ids

    snapshot, stale, age = snapshot_observations(service)
    try:
        with connect() as db:
            crosswalk = observations.espn_ids(db)
            candidates = observations.candidates(db, snapshot.season)
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        raise _tables(exc) from exc
    import polars as pl

    ids, names = board_lookups(candidates)
    resolved, report = resolve_player_ids(snapshot.to_frame(), ids, names, crosswalk)
    resolved = resolved.with_columns(
        (pl.col("owner_team_id") == snapshot.my_team_id).fill_null(False).alias("is_mine"),
        pl.when(pl.col("owner_team_id").is_not_null())
        .then(pl.lit("rostered"))
        .otherwise(pl.lit("free_agent"))
        .alias("availability"),
    )
    fields = [
        c
        for c in (
            "player_id",
            "espn_id",
            "player_display_name",
            "position",
            "owner_team_name",
            "owner_team_id",
            "is_mine",
            "availability",
            "injury_status",
            "lineup_slot",
            "espn_team",
        )
        if c in resolved.columns
    ]
    news, news_warning = reviewed_news(get_settings().data_dir, snapshot.season)
    return dict(
        season=snapshot.season,
        week=snapshot.week,
        stale=stale,
        age_seconds=round(age),
        league_name=snapshot.league_name,
        captured_at=snapshot.captured_at,
        my_team_id=snapshot.my_team_id,
        regular_season_weeks=snapshot.regular_season_weeks,
        transactions=[asdict(row) for row in snapshot.transactions],
        draft=[{**asdict(row), "player_id": crosswalk.get(row.espn_id)} for row in snapshot.draft],
        unmatched=report.unmatched,
        teams=[
            dict(
                team_id=t.team_id,
                team_name=t.team_name,
                wins=t.wins,
                losses=t.losses,
                is_mine=t.team_id == snapshot.my_team_id,
                faab_remaining=t.faab_remaining,
                division_name=t.division_name,
                team_abbrev=t.team_abbrev,
                schedule=[asdict(row) for row in t.schedule],
            )
            for t in snapshot.teams
        ],
        players=annotate_players(resolved.select(fields).to_dicts(), snapshot, news),
        attention_notices=roster_attention(snapshot),
        injury_news_warning=news_warning,
    )


@router.get("/matchups")
def matchups(service: ServiceDep, week: int | None = Query(None, ge=1, le=25)):
    snapshot, stale, age = snapshot_observations(service)
    return {
        **weekly_matchups(snapshot, week or snapshot.week),
        "stale": stale,
        "age_seconds": round(age),
    }


def _campaign_rankings(campaign: str, horizon: str):
    from engine.api.chairstats_routes import store
    from engine.chairstats import views
    from engine.chairstats.store import CampaignNotFound

    try:
        return views.rankings(store(), campaign, horizon=horizon)
    except CampaignNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, "Campaign artifacts are unavailable or invalid.") from exc


@router.get("/forecasts")
def forecasts(service: ServiceDep, campaign: str, horizon: str = "remaining"):
    """This week's lineup forecasts from sealed weekly or horizon models."""
    from engine.espn.forecast_log import record_forecast
    from engine.metrics.weekly_lineup import weekly_forecast

    snapshot, stale, age = snapshot_observations(service)
    ranks = _campaign_rankings(campaign, horizon) if horizon != "weekly" else None
    try:
        with connect() as db:
            ids = observations.espn_ids(db)
            schedule = observations.schedule(db, snapshot.season)
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        raise _tables(exc) from exc
    if horizon == "weekly":
        from engine.api.chairstats_routes import store
        from engine.chairstats.store import CampaignNotFound
        from engine.chairstats.weekly import weekly_inputs
        from engine.metrics.weekly_distribution import forecast_snapshot

        try:
            inputs = weekly_inputs(store(), campaign, snapshot.season, snapshot.week)
            players = inputs.pop("players")
            calibration = inputs.pop("calibration")
            result = forecast_snapshot(snapshot, players, ids, schedule, calibration)
            limitations = result.get("limitations", []) + inputs.pop("limitations", [])
            result.update(inputs, limitations=list(dict.fromkeys(limitations)))
        except CampaignNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(
                409, "Weekly forecast artifacts are unavailable or invalid."
            ) from exc
    else:
        result = weekly_forecast(snapshot, ranks, ids, schedule)
    record = record_forecast(get_settings().outputs_dir / "weekly_forecasts", snapshot, result)
    return dict(record, stale=stale, age_seconds=round(age))


@router.get("/team-strength")
def strength(service: ServiceDep, campaign: str, horizon: str = "remaining"):
    from engine.metrics.roster_correlations import attach_roster_correlations
    from engine.metrics.team_strength import team_strength
    from engine.tables.team_analysis import team_session

    snapshot, stale, age = snapshot_observations(service)
    issue = None
    try:
        model = forecasts(service, campaign, horizon)
    except HTTPException as exc:
        model, issue = None, str(exc.detail)
    result = team_strength(snapshot, model)
    try:
        with team_session(get_settings().data_dir) as (db, report):
            attach_roster_correlations(result, snapshot, db, report)
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        for team in result["teams"]:
            team["correlation_risk"] = dict(status="unavailable", issue=str(exc))
    return dict(result, stale=stale, age_seconds=round(age), forecast_issue=issue)


@router.get("/forecast-accuracy")
def forecast_accuracy(service: ServiceDep):
    from engine.espn.forecast_log import accuracy

    snapshot, stale, age = snapshot_observations(service)
    return dict(
        accuracy(snapshot, get_settings().outputs_dir / "weekly_forecasts"),
        stale=stale,
        age_seconds=round(age),
    )
