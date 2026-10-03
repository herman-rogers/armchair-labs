"""Current analysis only; archived experiments require explicit research scope."""

from __future__ import annotations

import csv
import io
from dataclasses import asdict

import polars as pl
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from engine.api.league_routes import ServiceDep
from engine.config.settings import get_settings
from engine.data.nextgen import eligible, load_analysis, resolve
from engine.data.ranking_views import (
    MARKET_COLUMNS,
    RANK_COLUMNS,
    baseline_ranks,
    market_references,
    with_ranks,
)
from engine.data.releases import load_gold
from engine.data.serving import read_model
from engine.espn.attention import annotate_players, reviewed_news, roster_attention
from engine.espn.crosswalk import board_lookups, resolve_player_ids
from engine.espn.service import NotAuthenticatedError
from engine.metrics.evidence_diagnostics import passing_workload_diagnostics
from engine.tables.application import read_frame, read_json

router = APIRouter(prefix="/api/nextgen", tags=["NextGen"])


def release():
    try:
        return load_analysis(get_settings().data_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"Current NextGen analysis unavailable: {exc}") from exc


def json_records(path):
    return read_json(path)


@router.get("/catalog")
def catalog():
    root, manifest = release()
    incidents = read_json(root / "incidents.json")
    entries = json_records(root / "registry.json")
    active = [
        r
        for r in entries
        if any(eligible(r, use=use, incidents=incidents)[0] for use in r["allowed_uses"])
    ]
    return dict(
        report=read_json(root / "report.json"),
        entries=active,
        archived_count=len(entries) - len(active),
        incidents=incidents,
        version=manifest["version"],
        gold=manifest["gold"],
    )


@router.get("/registry")
def registry(
    scope: str = "analysis",
    search: str = "",
    kind: str = "all",
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    if scope not in {"analysis", "research"}:
        raise HTTPException(422, "Unknown evidence scope")
    root, manifest = release()
    entries = json_records(root / "registry.json")
    incidents = read_json(root / "incidents.json")
    if scope == "analysis":
        entries = [
            r
            for r in entries
            if any(eligible(r, use=use, incidents=incidents)[0] for use in r["allowed_uses"])
        ]
    entries = [
        r
        for r in entries
        if (kind == "all" or r["kind"] == kind)
        and search.casefold() in (r["label"] + " " + r["reason"]).casefold()
    ]
    return dict(
        version=manifest["version"],
        scope=scope,
        total=len(entries),
        entries=entries[offset : offset + limit],
    )


def attach_market(frame, manifest, root):
    if "gold" not in manifest:
        return frame
    gold = load_gold(get_settings().data_dir, manifest["gold"]["version"])
    market = market_references(
        gold.read("preseason_features", columns=MARKET_COLUMNS),
        gold.manifest["current_observations"]["season"],
        read_json(root / "incidents.json"),
    )
    return frame.join(market, on="player_id", how="left", validate="m:1")


@router.get("/players")
@read_model("measurements")
def players(
    period: str = "prior",
    position: str = "ALL",
    population: str = "all",
    search: str = "",
    player_id: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    if period not in {"prior", "recent3", "career", "current"}:
        raise HTTPException(422, "Unknown observation period")
    root, manifest = release()
    frame = read_frame(root / "players.parquet").filter(pl.col("period") == period)
    if position != "ALL":
        frame = frame.filter(pl.col("position") == position)
    if population != "all":
        frame = frame.filter(pl.col("population") == population)
    if player_id is not None:
        frame = frame.filter(pl.col("player_id") == player_id)
    if search:
        frame = frame.filter(
            pl.col("player_display_name")
            .str.to_lowercase()
            .str.contains(search.lower(), literal=True)
        )
    # A known field incident masks only affected measurements, not every player or year.
    incidents = read_json(root / "incidents.json")
    columns = [
        "player_id",
        "player_display_name",
        "position",
        "population",
        "team",
        "period",
        "first_season",
        "last_season",
        "through_week",
        "coverage",
    ]
    for entry in json_records(root / "registry.json"):
        if entry["kind"] == "measurement":
            column = entry["id"].removeprefix("measurement:")
            if column in frame.columns:
                columns.append(column)
                if eligible(entry, use="descriptive", incidents=incidents)[0]:
                    columns.extend(
                        c
                        for c in (column + "_sample", column + "_observations")
                        if c in frame.columns
                    )
                else:
                    frame = frame.with_columns(pl.lit(None).alias(column))
    frame = frame.select(columns)
    ranks = baseline_ranks(root).select(
        "player_id",
        *RANK_COLUMNS,
        pl.col("prediction").alias("baseline_points"),
        pl.col("cutoff").alias("baseline_cutoff"),
        pl.col("season").alias("baseline_season"),
    )
    frame = frame.join(ranks, on="player_id", how="left", validate="1:1")
    frame = attach_market(frame, manifest, root)
    return dict(
        version=manifest["version"],
        total=frame.height,
        period=period,
        players=frame.sort("player_display_name").slice(offset, limit).to_dicts(),
    )


@router.get("/forecasts")
def forecasts(
    target: str = "season_points",
    model: str = "baseline",
    position: str = "ALL",
    horizon: str = "preseason_season",
    search: str = "",
    population: str = "all",
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    format: str = "json",
    scope: str = "analysis",
):
    root, manifest = release()
    if scope not in {"analysis", "research"} or format not in {"json", "csv"}:
        raise HTTPException(422, "Unknown output scope or format")
    entry_id = f"forecast:{model}:{target}"
    try:
        entry = resolve(
            root,
            entry_id,
            use="forecast",
            target=target,
            position=position,
            population=population,
            horizon=horizon,
        )
    except ValueError as exc:
        if scope != "research":
            raise HTTPException(409, str(exc)) from exc
        entry = next((r for r in json_records(root / "registry.json") if r["id"] == entry_id), None)
        if not entry or entry["horizon"] != horizon:
            raise HTTPException(422, "Unknown archived forecast") from exc
    frame = read_frame(root / "forecasts.parquet").filter(
        (pl.col("target") == target) & (pl.col("model") == model)
    )
    if scope == "analysis":
        frame = frame.filter(
            pl.col("position").is_in(entry["positions"])
            & pl.col("population").is_in(entry["populations"])
        )
    frame = attach_market(with_ranks(frame), manifest, root)
    if position != "ALL":
        frame = frame.filter(pl.col("position") == position)
    if population != "all":
        frame = frame.filter(pl.col("population") == population)
    if search:
        frame = frame.filter(
            pl.col("player_display_name")
            .str.to_lowercase()
            .str.contains(search.lower(), literal=True)
        )
    total = frame.height
    frame = frame.sort(
        ["prediction", "player_display_name", "player_id"],
        descending=[True, False, False],
        nulls_last=True,
    ).slice(offset, limit)
    if format == "csv":
        stream = io.StringIO()
        names = [*frame.columns, "serving_status", "analysis_version"]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in frame.to_dicts():
            writer.writerow(
                {**row, "serving_status": entry["serving"], "analysis_version": manifest["version"]}
            )
        return Response(
            stream.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="nextgen-{scope}-{target}.csv"'},
        )
    return dict(
        version=manifest["version"],
        entry=entry,
        total=total,
        scope=scope,
        forecasts=frame.to_dicts(),
        note="Saved preseason baseline, not a current rest-of-season or waiver value.",
    )


@router.get("/rookies")
def rookies(
    position: str = "ALL",
    search: str = "",
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    result = players(
        period="current",
        position=position,
        population="rookie",
        search=search,
        limit=limit,
        offset=offset,
    )
    _, manifest = release()
    gold = load_gold(get_settings().data_dir, manifest["gold"]["version"])
    season = gold.manifest["current_observations"]["season"]
    identities = {
        row["gsis_id"]: row
        for row in gold.read("players").select("gsis_id", "draft_round", "draft_pick").to_dicts()
    }
    linked = set(
        gold.read("college_identity_links").filter(pl.col("status") == "linked")["player_id"]
    )
    for row in result["players"]:
        identity = identities.get(row["player_id"], {})
        row.update(
            draft_round=identity.get("draft_round"),
            draft_pick=identity.get("draft_pick"),
            college_linked=row["player_id"] in linked,
        )
    return {
        **result,
        "season": season,
        "through_week": gold.manifest["current_observations"]["through_week"],
    }


@router.get("/evidence")
def evidence(
    target: str = "season_points",
    position: str = "ALL",
    population: str = "all",
    window: str = "all_history",
    scope: str = "analysis",
):
    if scope not in {"analysis", "research"}:
        raise HTTPException(422, "Unknown evidence scope")
    root, manifest = release()
    rows = json_records(root / "evaluations.json")
    rows = [
        r
        for r in rows
        if r["target"] == target
        and r["population"] == population
        and r["window"] == window
        and (position == "ALL" or r["position"] == position)
    ]
    if scope == "analysis":
        incidents = read_json(root / "incidents.json")
        entries = {r["id"]: r for r in json_records(root / "registry.json")}
        rows = [
            r
            for r in rows
            if eligible(
                entries.get(f"forecast:{r['model']}:{target}", {}),
                use="forecast",
                target=target,
                position=r["position"],
                population=population,
                incidents=incidents,
            )[0]
        ]
    if target == "passing_yards" and rows and "features.parquet" in manifest.get("files", {}):
        predictions = (
            read_frame(root / "predictions.parquet")
            .lazy()
            .filter(
                (pl.col("target") == target)
                & (pl.col("position") == "QB")
                & pl.col("complete")
                & (pl.col("season") >= (2019 if window == "modern" else 2007))
            )
        )
        if population != "all":
            predictions = predictions.filter(pl.col("population") == population)
        predictions = predictions.collect()
        features = read_frame(
            root / "features.parquet", columns=["player_id", "forecast_season", "x_prior_attempts"]
        )
        rows = [
            {
                **row,
                "diagnostics": passing_workload_diagnostics(
                    predictions.filter(pl.col("model") == row["model"]), features
                ),
            }
            for row in rows
        ]
    return dict(version=manifest["version"], comparisons=rows, scope=scope)


@router.get("/individual-evidence")
def individual_evidence(
    scope: str = "analysis",
    search: str = "",
    position: str = "ALL",
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    if scope != "research":
        raise HTTPException(
            409, "Individual-stat screens are research-only, not approved analysis signals"
        )
    root, manifest = release()
    frame = read_frame(root / "individual_results.parquet")
    if position != "ALL":
        frame = frame.filter(pl.col("position") == position)
    if search:
        frame = frame.filter(
            pl.col("stat").str.to_lowercase().str.contains(search.lower(), literal=True)
        )
    return dict(
        version=manifest["version"],
        source=manifest["evidence"],
        total=frame.height,
        scope="research",
        comparisons=frame.slice(offset, limit).to_dicts(),
    )


@router.get("/league")
def league(service: ServiceDep):
    """League observations only. Old board scores never enter this analysis response."""
    root, manifest = release()
    try:
        snapshot, stale, age = service.observations()
    except NotAuthenticatedError as exc:
        raise HTTPException(401, str(exc)) from exc
    except (OSError, TimeoutError) as exc:
        raise HTTPException(503, str(exc)) from exc
    gold = load_gold(get_settings().data_dir, manifest["gold"]["version"])
    identities = gold.read("players")
    crosswalk = {}
    for row in identities.select("espn_id", "gsis_id").drop_nulls().to_dicts():
        try:
            crosswalk[int(float(row["espn_id"]))] = row["gsis_id"]
        except (ValueError, TypeError):
            continue
    candidates = read_frame(root / "players.parquet").filter(pl.col("period") == "current")
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
        version=manifest["version"],
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
                schedule=[asdict(row) for row in t.schedule],
            )
            for t in snapshot.teams
        ],
        players=annotate_players(resolved.select(fields).to_dicts(), snapshot, news),
        attention_notices=roster_attention(snapshot),
        injury_news_warning=news_warning,
        decision_status=(
            "No approved trade, waiver or weekly-lineup model. "
            "Ownership and results are observations."
        ),
    )
