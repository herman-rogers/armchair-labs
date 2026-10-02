"""Verified QB passing references; unapproved challengers require research scope."""

from typing import Literal

import polars as pl
from fastapi import APIRouter, HTTPException, Query

from patron.api.nextgen_routes import json_records, release
from patron.data.frames import read_frame
from patron.data.nextgen import resolve

router = APIRouter(prefix="/api/nextgen/qb-passing", tags=["QB passing"])
Horizon = Literal["next_game", "next4", "rest_of_season"]
CURRENT_COLUMNS = {
    "player_id",
    "player_display_name",
    "team",
    "season",
    "through_week",
    "end_week",
    "scheduled_games",
    "horizon",
    "prediction",
    "unconstrained_prediction",
    "lower",
    "upper",
    "reference_recipe",
    "execution_ypa",
    "observed_career_ypa",
    "evidence_attempts",
    "effective_attempts",
    "prior_weight",
    "current_attempts",
    "current_yards",
    "prior_attempts",
    "prior_yards",
    "role_group",
    "roster_state",
    "roster_source",
    "roster_known_on",
    "medical_status",
    "constraint_reason",
    "constraint_source",
    "news_known_on",
    "retired_known",
    "schedule_known",
    "issued_at",
    "entry_id",
    "evidence_status",
    "production_recipe",
    "execution_reference",
    "rate_recipe",
    "rate_evidence_status",
    "rate_entry_id",
    "experiment_version",
}


def passing_release():
    root, manifest = release()
    if manifest.get("qb_passing_schema_version") != 1:
        raise HTTPException(409, "QB passing forecasts have not been published")
    return root, manifest


@router.get("")
def forecasts(
    horizon: Horizon = "rest_of_season",
    search: str = "",
    player_id: str = "",
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    root, manifest = passing_release()
    report = json_records(root / "qb_passing/report.json")
    try:
        entry = resolve(
            root,
            report.get("current_entries", {}).get(horizon, "qb_passing:reference:" + horizon),
            use="forecast",
            target="passing_yards",
            position="QB",
            horizon=horizon,
        )
        if horizon in report.get("rate_approved_horizons", []):
            resolve(
                root,
                "qb_passing:rate:" + horizon,
                use="forecast",
                target="passing_efficiency",
                position="QB",
                horizon=horizon,
            )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    frame = read_frame(root / "qb_passing/current.parquet").filter(pl.col("horizon") == horizon)
    if player_id:
        frame = frame.filter(pl.col("player_id") == player_id)
    if search:
        frame = frame.filter(
            pl.col("player_display_name")
            .str.to_lowercase()
            .str.contains(search.lower(), literal=True)
        )
    frame = frame.sort(["prediction", "player_id"], descending=[True, False])
    frame = frame.select([c for c in frame.columns if c in CURRENT_COLUMNS])
    # Research cases and challenger decisions are exposed only on the evidence endpoint.
    public_report = {
        k: v for k, v in report.items() if k not in {"examples", "decisions", "component_scores"}
    }
    return dict(
        version=manifest["version"],
        entry=entry,
        report=public_report,
        total=frame.height,
        forecasts=frame.slice(offset, limit).to_dicts(),
    )


@router.get("/evidence")
def evidence(
    horizon: Horizon = "rest_of_season",
    window: Literal["all_history", "modern"] = "all_history",
    origin: Literal["preseason", "weekly"] = "weekly",
    scope: Literal["analysis", "research"] = "analysis",
):
    root, manifest = passing_release()
    report = json_records(root / "qb_passing/report.json")
    try:
        resolve(
            root,
            report.get("current_entries", {}).get(horizon, "qb_passing:reference:" + horizon),
            use="forecast",
            target="passing_yards",
            position="QB",
            horizon=horizon,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    rows = [
        r
        for r in json_records(root / "qb_passing/evaluations.json")
        if r["horizon"] == horizon
        and r["window"] == window
        and r["origin_type"] == origin
        and (
            scope == "research"
            or r["model"] in {"prior_season", "reference"}
            or r["model"] == "production_policy"
            and horizon in report.get("production_approved_horizons", [])
        )
    ]
    report = json_records(root / "qb_passing/report.json")
    return dict(
        version=manifest["version"],
        scope=scope,
        comparisons=rows,
        decisions=[
            r
            for r in report["decisions"]
            if r["horizon"] == horizon and r["window"] == window and r["origin_type"] == origin
        ]
        if scope == "research"
        else [],
        examples=report["examples"] if scope == "research" else [],
        component_scores=components(root, horizon, window, origin) if scope == "research" else [],
        limitations=report["limitations"],
    )


@router.get("/variations")
def variations(
    target: Literal["rate", "yards", "league_points"] = "rate",
    horizon: Horizon = "rest_of_season",
    window: Literal["all_history", "modern"] = "modern",
    origin: Literal["preseason", "weekly"] = "weekly",
    scope: Literal["analysis", "research"] = "analysis",
):
    if target == "league_points" and (horizon == "next_game" or origin != "weekly"):
        raise HTTPException(422, "QB ranking evidence covers weekly next-four/remaining forecasts")
    root, manifest = passing_release()
    if manifest.get("qb_variations_schema_version") != 1:
        raise HTTPException(409, "QB variation research has not been published")
    base = root / "qb_variations"
    decisions = json_records(base / "decisions.json")
    decision = next(
        (d for d in decisions if d["target"] == target and d["horizon"] == horizon), None
    )
    if decision and decision["approved"]:
        item = (
            f"ranking:{horizon}:QB"
            if target == "league_points"
            else f"qb_passing:{'rate' if target == 'rate' else 'production'}:{horizon}"
        )
        try:
            resolve(
                root,
                item,
                use="ranking" if target == "league_points" else "forecast",
                target={
                    "rate": "passing_efficiency",
                    "yards": "passing_yards",
                    "league_points": "league_points",
                }[target],
                position="QB",
                horizon=horizon,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    rows = []
    if target != "league_points":
        for r in json_records(base / "evaluations.json"):
            if (r["target"], r["horizon"], r["window"], r["origin_type"]) != (
                target,
                horizon,
                window,
                origin,
            ):
                continue
            policy, baseline = target + "_policy", target + "_reference"
            public = (
                r["model"] == baseline
                or r["model"] == policy
                and decision
                and decision["approved"]
                and origin == "weekly"
            )
            if scope == "research" or public:
                status = (
                    "Approved policy"
                    if r["model"] == policy
                    and decision
                    and decision["approved"]
                    and origin == "weekly"
                    else "Reference"
                    if r["model"] == baseline
                    else "Research only"
                )
                rows.append(dict(r, status=status))
    else:
        # These individual combinations remain research-only even when their
        # earlier-season-selected policy qualifies for production.
        frame = read_frame(base / "ranking_predictions.parquet").filter(
            pl.col("actual").is_not_null()
            & (pl.col("horizon") == horizon)
            & (pl.col("season") >= (2019 if window == "modern" else 2011))
        )
        if frame.height:
            from patron.metrics.current_rankings import compare

            candidates = (
                [c for c in frame.columns if c.startswith("points_")] if scope == "research" else []
            )
            names = [
                "reference",
                *(["policy"] if scope == "research" or decision and decision["approved"] else []),
                *candidates,
            ]
            for name in names:
                inputs = frame.with_columns(pl.col(name).alias("policy")).to_dicts()
                s = compare(inputs, "QB")
                rows.append(
                    dict(
                        model=name,
                        status="Reference"
                        if name == "reference"
                        else "Approved policy"
                        if name == "policy" and decision and decision["approved"]
                        else "Research only",
                        n=s["n"],
                        years=s["years"],
                        mae=s["error"],
                        rmse=None,
                        improvement_pct=100 * s["improvement"] / s["baseline_error"],
                        ci_low=s["ci_low"],
                        ci_high=s["ci_high"],
                        annual=s["annual"],
                    )
                )
    return dict(
        version=manifest["version"],
        scope=scope,
        target=target,
        horizon=horizon,
        report=json_records(base / "report.json"),
        decision=decision,
        comparisons=rows,
    )


def components(root, horizon, window, origin):
    frame = read_frame(root / "qb_passing/predictions.parquet").filter(
        pl.col("actual").is_not_null()
        & (pl.col("horizon") == horizon)
        & (pl.col("season") >= (2019 if window == "modern" else 2007))
        & ((pl.col("through_week") == 0) if origin == "preseason" else (pl.col("through_week") > 0))
    )
    if not frame.height:
        return []
    rates = frame.filter(pl.col("actual_attempts") > 0).with_columns(
        (pl.col("execution_ypa") - pl.col("actual") / pl.col("actual_attempts"))
        .abs()
        .alias("error")
    )
    return [
        dict(
            horizon=horizon,
            window=window,
            origin_type=origin,
            n=frame.height,
            primary_fraction_mse=frame.select(
                ((pl.col("probability_primary") - pl.col("actual_primary")) ** 2).mean()
            ).item(),
            execution_observed_n=rates.height,
            execution_ypa_mae=rates["error"].mean(),
            execution_attempt_weighted_mae=(rates["error"] * rates["actual_attempts"]).sum()
            / rates["actual_attempts"].sum()
            if rates.height
            else None,
        )
    ]
