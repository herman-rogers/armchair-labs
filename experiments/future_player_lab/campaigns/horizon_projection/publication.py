"""Publish coherent horizons from one locked weekly origin and accepted past outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine.chairstats import CampaignStore

from ..weekly_v3.data import ROOT, SCORING, TEAM_ALIASES, score_points
from ..weekly_v31.artifacts import digest, now, verify, write
from .curve import durable_signal, future_curve
from .research import DEFAULT_RUN


def actual_points(box, season, origin):
    """Only completed regular-season weeks under the unchanged campaign scoring rule."""
    box = box[box.season.eq(season) & box.week.le(origin) & box.season_type.eq("REG")].copy()
    box = box[box.position.isin(["QB", "RB", "WR", "TE"]) & box.player_id.notna()]
    if box.duplicated(["player_id", "week"]).any():
        raise ValueError("Duplicate accepted player-week outcomes")
    box["points"] = score_points(box)
    if box.points.isna().any():
        raise ValueError("Missing scoring components in accepted past outcomes")
    return box.groupby("player_id").points.sum().to_dict()


def inputs(registry, captures, policy_run=DEFAULT_RUN):
    """Pin the same-origin anchor and the snapshot already locked by weekly capture."""
    registry, policy_run = Path(registry), Path(policy_run)
    verify(policy_run)
    curve_source = Path(__file__).with_name("curve.py")
    if digest(curve_source) != digest(policy_run / "source/curve.py"):
        raise ValueError("Projection curve differs from evaluated policy")
    early_paths = [Path(p) for p in captures if verify(p)["forecast_horizon"] == "early_week"]
    if len(early_paths) != 1:
        raise ValueError("One locked early-week capture required for horizon projection")
    early = early_paths[0]
    document = json.loads((early / "snapshot.json").read_text())
    current = document["current_observations"]
    season, origin = current["season"], current["through_week"]
    policy = json.loads((policy_run / "policy.json").read_text())
    if policy["forecast_season"] != season:
        raise ValueError("Projection policy needs a new season evaluation")
    store = CampaignStore(registry)
    store.check("opening_drive_v2")
    baseline = store.campaign("opening_drive_v2")
    anchors, references = [], {}
    for model in baseline["models"]:
        if model["horizon"] != "remaining":
            continue
        saved = store.forecast("opening_drive_v2", model["model_id"], season, origin)
        if saved["metadata"].get("snapshot", {}).get("sha256") != digest(early / "snapshot.json"):
            raise ValueError("Long-horizon anchor does not match the locked weekly snapshot")
        if pd.Timestamp(saved["metadata"]["generated_at"]) > pd.Timestamp(
            verify(early)["generated_at"]
        ):
            raise ValueError("Long-horizon anchor was generated after the locked weekly capture")
        anchors.extend(saved["rows"])
        references[model["model_id"]] = saved["artifact_sha256"]
    if len(references) != 4:
        raise ValueError("Four same-origin long-horizon anchors required")
    assets = {}
    for name in ("schedule", "weeks"):
        ref = document["assets"][f"current/{name}.parquet"]
        path = ROOT / "data" / ref["object"]
        if digest(path) != ref["sha256"]:
            raise ValueError("Locked schedule/outcome source changed")
        assets[name] = pd.read_parquet(path)
    schedule = assets["schedule"]
    schedule = schedule[schedule.season.eq(season) & schedule.game_type.eq("REG")].copy()
    for c in ("home_team", "away_team"):
        schedule[c] = schedule[c].replace(TEAM_ALIASES)
    if (
        schedule.empty
        or schedule[schedule.week.le(origin)][["home_score", "away_score"]].isna().any().any()
    ):
        raise ValueError("Past scheduled games must have accepted completed scores")
    actuals = actual_points(assets["weeks"], season, origin)
    return dict(
        season=season,
        origin=origin,
        policy=policy,
        policy_run=policy_run,
        schedule=schedule,
        actuals=actuals,
        anchors={r["player_id"]: r for r in anchors},
        early=pd.read_parquet(early / "players.parquet").set_index("player_id").to_dict("index"),
        panel=pd.read_parquet(early / "baseline_panel.parquet")
        .set_index("player_id")
        .to_dict("index"),
        provenance=dict(
            policy_manifest_sha256=digest(policy_run / "manifest.json"),
            snapshot_sha256=digest(early / "snapshot.json"),
            anchor_forecast_hashes=references,
            publication_source_sha256=digest(Path(__file__)),
        ),
    )


def projected_rows(players, context):
    """One future schedule per player supplies both totals and their denominators."""
    season, origin = context["season"], context["origin"]
    if (
        players.player_id.duplicated().any()
        or not (players.season.eq(season) & players.week.eq(origin + 1)).all()
    ):
        raise ValueError("Weekly rows must share the projection origin and unique identities")
    output = {"remaining": [], "season": []}
    fallback = []
    schedule_changes = []
    for player in players.to_dict("records"):
        pid, position = player["player_id"], player["position"]
        team = TEAM_ALIASES.get(player["team"], player["team"])
        schedule = context["schedule"]
        games = schedule[schedule.home_team.eq(team) | schedule.away_team.eq(team)]
        future = games.loc[games.week.gt(origin), "week"].tolist()
        if games.empty or games.week.duplicated().any():
            raise ValueError(f"No unambiguous team schedule for {pid}")
        scheduled = origin + 1 in future
        if scheduled == bool(player["is_bye"]):
            raise ValueError("Locked weekly bye disagrees with locked schedule")
        early = context["early"][pid]
        original = context["panel"][pid]
        anchor_row = context["anchors"].get(pid)
        if anchor_row and anchor_row.get("scheduled_games"):
            anchor = anchor_row["predictions"]["points"] / anchor_row["scheduled_games"]
            anchor_basis = "same_origin_v2_remaining"
            if anchor_row.get("team") != team or anchor_row["scheduled_games"] != len(future):
                schedule_changes.append(
                    dict(
                        player_id=pid,
                        anchor_team=anchor_row.get("team"),
                        weekly_team=team,
                        anchor_games=anchor_row["scheduled_games"],
                        projection_games=len(future),
                    )
                )
        else:
            # New weekly-only candidates use earlier history already in the locked panel.
            anchor = original.get("v2_weekly_mean")
            if anchor is None or not pd.notna(anchor):
                raise ValueError(f"No finite long-horizon fallback for {pid}")
            anchor_basis = "locked_history_proxy_not_v2"
            fallback.append(pid)
        base = early.get("base_prediction")
        signal = durable_signal(early, anchor) if scheduled and pd.notna(base) else anchor
        curve = future_curve(
            future,
            origin + 1,
            player["prediction"],
            anchor,
            signal,
            context["policy"]["half_lives"][position],
        )
        remaining = sum(g["points"] for g in curve)
        earned = float(context["actuals"].get(pid, 0.0))
        projection = dict(
            method="opening_drive_schedule_curve_v1",
            future_weeks=curve,
            remaining_points=remaining,
            actual_points_to_date=earned,
            remaining_scheduled_games=len(future),
            season_scheduled_games=len(games),
            anchor_points_per_game=anchor,
            anchor_basis=anchor_basis,
            anchor_team=anchor_row.get("team") if anchor_row else None,
            schedule_team=team,
            early_signal=signal,
            half_life=context["policy"]["half_lives"][position],
        )
        for horizon in output:
            total = remaining if horizon == "remaining" else earned + remaining
            count = len(future) if horizon == "remaining" else len(games)
            output[horizon].append(
                dict(
                    player_id=pid,
                    player_name=player.get("player_name"),
                    position=position,
                    team=team,
                    season=season,
                    origin_week=origin,
                    horizon=horizon,
                    predictions={"points": total},
                    scheduled_games=count,
                    calendar_weeks=int(schedule.week.max())
                    - (origin if horizon == "remaining" else 0),
                    points_per_scheduled_game=total / count if count else None,
                    candidate_basis="Locked weekly eligible pool; dated schedule aggregation",
                    projection=projection,
                )
            )
    for rows in output.values():
        for position in ("QB", "RB", "WR", "TE"):
            group = sorted(
                (r for r in rows if r["position"] == position),
                key=lambda r: (-r["predictions"]["points"], r["player_id"]),
            )
            for rank, row in enumerate(group, 1):
                row["rank"] = rank
    audit = dict(
        weekly_players=len(players),
        fallback_player_ids=fallback,
        anchor_only_player_ids=sorted(set(context["anchors"]) - set(players.player_id)),
        anchor_schedule_changes=schedule_changes,
        scheduled_games_basis="Locked weekly team schedule; "
        "past actual points follow player across teams",
    )
    return output, audit


def entries(registry, old, players, context, tmp, campaign, capture_hashes):
    registry, tmp = Path(registry), Path(tmp)
    projected, audit = projected_rows(players, context)
    season, origin, run = context["season"], context["origin"], context["policy_run"]
    result = []
    for horizon, rows in projected.items():
        for position in ("QB", "RB", "WR", "TE"):
            model_id = f"{position.lower()}_{horizon}"
            payload = dict(
                campaign_id=campaign,
                model_id=model_id,
                season=season,
                origin_week=origin,
                horizon=horizon,
                training_through_season=context["policy"]["training_through_season"],
                generated_at=now().isoformat(),
                forecast_kind="coherent_schedule_projection",
                rows=[r for r in rows if r["position"] == position],
                projection_provenance=context["provenance"],
                capture_hashes=capture_hashes,
                projection_audit=audit,
            )
            path = tmp / f"{model_id}.json"
            write(path, payload)
            forecasts = []
            previous = (
                next((m for m in old["models"] if m["model_id"] == model_id), None) if old else None
            )
            # Keep older forecasts, explicitly labelled; original full bundles are archived too.
            if previous:
                for ref in previous["forecasts"]:
                    if (ref["season"], ref["origin_week"]) == (season, origin):
                        continue
                    saved = json.loads((registry / campaign / ref["path"]).read_text())
                    if isinstance(saved, list):
                        saved = dict(rows=saved)
                    inherited = saved.get("forecast_kind") != "coherent_schedule_projection"
                    # Legacy artifacts have component targets; the new horizon model is points-only.
                    for row in saved["rows"]:
                        row["predictions"] = {"points": row["predictions"]["points"]}
                    if inherited:
                        saved["forecast_kind"] = "preserved_legacy_horizon"
                    saved.update(
                        campaign_id=campaign,
                        model_id=model_id,
                        horizon=horizon,
                        training_through_season=context["policy"]["training_through_season"],
                    )
                    old_path = (
                        tmp / "historical" / model_id / f"{ref['season']}_{ref['origin_week']}.json"
                    )
                    write(old_path, saved)
                    forecasts.append(
                        dict(ref, path=old_path, rolling_origin=ref["origin_week"] != 0)
                    )
            forecasts.append(
                dict(season=season, origin_week=origin, rolling_origin=True, path=path)
            )
            metrics = tmp / f"{model_id}_metrics.json"
            scores = pd.read_csv(run / "scores.csv")
            write(
                metrics,
                dict(
                    evaluation="Remaining total at origin Week 2; "
                    "adding known actuals leaves total error unchanged",
                    scores=json.loads(
                        scores[scores.position.eq(position)].to_json(orient="records")
                    ),
                    limitations=context["policy"]["limitations"],
                ),
            )
            result.append(
                dict(
                    model_id=model_id,
                    position=position,
                    horizon=horizon,
                    origin_week=0,
                    targets=["points"],
                    training_through_season=context["policy"]["training_through_season"],
                    scoring=SCORING["QB" if position == "QB" else "RB/WR/TE"],
                    metrics=metrics,
                    forecasts=forecasts,
                    artifacts=dict(
                        projection_policy=run / "policy.json",
                        projection_evaluation=run / "predictions.parquet",
                        projection_manifest=run / "manifest.json",
                        projection_source=Path(__file__).with_name("curve.py"),
                        projection_publication_source=Path(__file__),
                    ),
                )
            )
    return result
