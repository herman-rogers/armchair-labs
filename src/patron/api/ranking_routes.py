"""Published current-season forecasts, ranks, and their evidence; no archive fallback."""

import csv
import io

import polars as pl
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from patron.api.nextgen_routes import release
from patron.config.settings import get_settings
from patron.data.frames import read_frame
from patron.data.nextgen import eligible, read_json
from patron.data.ranking_views import MARKET_COLUMNS, market_references
from patron.data.releases import load_gold
from patron.data.serving import read_model
from patron.metrics.current_rankings import HORIZONS, POSITIONS

router = APIRouter(prefix="/api/nextgen/rankings", tags=["NextGen rankings"])


def ranking_release():
    root, manifest = release()
    if manifest.get("ranking_schema_version") != 1:
        raise HTTPException(409, "Current-season NextGen rankings have not been published")
    return root, manifest


@router.get("")
@read_model("rankings")
def rankings(
    horizon: str = "rest_of_season",
    position: str = "ALL",
    population: str = "all",
    search: str = "",
    player_id: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    format: str = "json",
):
    if (
        horizon not in HORIZONS
        or position not in (*POSITIONS, "ALL")
        or format not in {"json", "csv"}
    ):
        raise HTTPException(422, "Unknown ranking horizon, position or format")
    root, manifest = ranking_release()
    frame = read_frame(root / "rankings.parquet").filter(pl.col("horizon") == horizon)
    entries = {r["id"]: r for r in read_json(root / "registry.json")}
    incidents = read_json(root / "incidents.json")
    exclusions = []
    scopes = []
    for row in frame.select("entry_id", "position", "population").unique().to_dicts():
        allowed, reason = eligible(
            entries.get(row["entry_id"], {}),
            use="ranking",
            target="league_points",
            horizon=horizon,
            position=row["position"],
            population=row["population"],
            incidents=incidents,
        )
        if allowed:
            scopes.append(
                (pl.col("entry_id") == row["entry_id"])
                & (pl.col("population") == row["population"])
            )
        else:
            exclusions.append(
                dict(position=row["position"], population=row["population"], reason=reason)
            )
    frame = frame.filter(pl.any_horizontal(scopes) if scopes else pl.lit(False))
    # Saved ranks preserve their full-population meaning, including when another scope is suspended.
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
    gold = load_gold(get_settings().data_dir, manifest["gold"]["version"])
    market = market_references(
        gold.read("preseason_features", columns=MARKET_COLUMNS),
        gold.manifest["current_observations"]["season"],
        incidents,
    )
    frame = frame.join(market, on="player_id", how="left", validate="1:1")
    frame = frame.sort(["overall_rank", "player_id"], nulls_last=True)
    report = read_json(root / "ranking_report.json")
    if format == "csv":
        output = io.StringIO()
        # Export the full filtered population; no pagination truncation.
        rows = [
            dict(
                r,
                analysis_version=manifest["version"],
                published_at=report.get("published_at", report["generated_at"]),
            )
            for r in frame.to_dicts()
        ]
        columns = frame.columns + ["analysis_version", "published_at"]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="nextgen-{horizon}.csv"',
                "X-NextGen-Analysis": manifest["version"],
            },
        )
    return dict(
        version=manifest["version"],
        report=report,
        horizon=horizon,
        total=frame.height,
        rankings=frame.slice(offset, limit).to_dicts(),
        excluded=exclusions,
    )


@router.get("/evidence")
def evidence(horizon: str = "rest_of_season", position: str = "ALL"):
    if horizon not in HORIZONS or position not in (*POSITIONS, "ALL"):
        raise HTTPException(422, "Unknown ranking scope")
    root, manifest = ranking_release()
    entries = {r["id"]: r for r in read_json(root / "registry.json")}
    incidents = read_json(root / "incidents.json")
    results = []
    for r in read_json(root / "ranking_evaluations.json"):
        if r["horizon"] != horizon or position not in {"ALL", r["position"]}:
            continue
        entry = entries.get(f"ranking:{horizon}:{r['position']}", {})
        allowed, reason = eligible(
            entry,
            use="ranking",
            target="league_points",
            horizon=horizon,
            position=r["position"],
            incidents=incidents,
        )
        results.append(
            dict(
                r,
                serving=entry.get("serving"),
                available=allowed,
                current_recipe_status=entry.get("evidence"),
                serving_reason=reason,
            )
        )
    return dict(version=manifest["version"], evaluations=results)


@router.get("/history")
def history(horizon: str = "rest_of_season"):
    """Published weekly snapshots; unavailable history never hides current rankings."""
    from patron.data.ranking_history import ranking_history

    if horizon not in HORIZONS:
        raise HTTPException(422, "Unknown ranking horizon")
    root, manifest = ranking_release()
    report = read_json(root / "ranking_report.json")
    try:
        snapshots = ranking_history(
            get_settings().data_dir, report["season"], report["through_week"], horizon
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, "Published ranking history could not be verified") from exc
    return dict(
        version=manifest["version"],
        season=report["season"],
        through_week=report["through_week"],
        horizon=horizon,
        snapshots=snapshots,
    )
