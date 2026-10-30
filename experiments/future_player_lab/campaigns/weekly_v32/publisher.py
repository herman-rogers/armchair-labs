"""Publish v3.2 forecasts; application configuration determines the default."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from engine.chairstats import CampaignStore, publish_campaign, replace_campaign

from ..horizon_projection import publication as horizons
from ..weekly_v3.data import SCORING
from ..weekly_v31.artifacts import digest, now, verify, write
from . import uncertainty

CAMPAIGN = "opening_drive_v32"


def optional(value):
    return None if pd.isna(value) else value


def rows(frame):
    output = []
    for rank, row in enumerate(
        frame.sort_values(["prediction", "player_id"], ascending=[False, True]).to_dict("records"),
        1,
    ):
        distribution = {c: float(row[c]) for c in uncertainty.QCOLS}
        distribution.update(
            std=float(row["std"]),
            samples=list(row["samples"]),
            mean_method=row["chosen_method"],
            distribution_method="v32_prior_year_empirical_errors",
            forecast_horizon=row["forecast_horizon"],
        )
        output.append(
            dict(
                player_id=row["player_id"],
                player_name=optional(row.get("player_name")),
                position=row["position"],
                season=int(row["season"]),
                origin_week=int(row["week"]) - 1,
                team=optional(row.get("team")),
                opponent=optional(row.get("opponent")),
                predictions={"points": float(row["prediction"])},
                points_per_scheduled_game=float(row["prediction"]),
                scheduled_games=0 if row["is_bye"] else 1,
                calendar_weeks=1,
                rank=rank,
                distribution=distribution,
                participation_probability=optional(row.get("participation_probability")),
                expected_workload={
                    c: optional(row.get(f"expected_{c}"))
                    for c in ("attempts", "carries", "targets")
                },
                availability=dict(
                    status=row.get("report_status_label", "Unknown"),
                    as_of=str(row["report_usable_at"])
                    if pd.notna(row.get("report_usable_at"))
                    else None,
                    meaning="Official weekly report; participation probability "
                    "is not active-roster status",
                ),
                candidate_basis="V3 eligible pool; v3.2 injury/workload shadow forecast",
            )
        )
    return output


def publish(registry, integration, players, captures, baseline):
    registry, integration = Path(registry), Path(integration)
    store = CampaignStore(registry)
    capture_hashes = {str(p): digest(p / "manifest.json") for p in captures}
    for path in captures:
        verify(path)
    projection = horizons.inputs(registry, captures)
    old = store.campaign(CAMPAIGN) if (registry / CAMPAIGN).exists() else None
    if (
        old
        and old.get("latest_capture_hashes") == capture_hashes
        and old.get("projection_provenance") == projection["provenance"]
    ):
        store.check(CAMPAIGN)
        return dict(status="current", manifest_sha256=old["manifest_sha256"])
    season, week = int(players.season.iloc[0]), int(players.week.iloc[0])
    if players.season.nunique() != 1 or players.week.nunique() != 1:
        raise ValueError("Publication requires one forecast week")
    if set(players.position) != {"QB", "RB", "WR", "TE"}:
        raise ValueError("Incomplete v3.2 position coverage")
    generated = max(verify(p)["generated_at"] for p in captures)
    with tempfile.TemporaryDirectory(dir=registry.parent, prefix=".v32-publish-") as tmp:
        tmp = Path(tmp)
        entries = horizons.entries(
            registry, old, players, projection, tmp, CAMPAIGN, capture_hashes
        )
        timings = set(players.forecast_horizon)
        dep_horizon = "pregame_60m" if timings == {"pregame_60m"} else "early_week"
        calibration = json.loads((integration / "dependence.json").read_text())[dep_horizon]
        calibration.update(
            campaign_id=CAMPAIGN,
            campaign_name="Opening Drive v3.2 preview",
            mixed_horizons=len(timings) > 1,
            range_reliability_note="V3.2 empirical prior-year error ranges; "
            "historical validation is not prospective coverage. "
            "K/DST uncertainty is withheld. Mixed-timing lineup ranges are withheld.",
        )
        cal_path = tmp / "lineup_calibration.json"
        write(cal_path, calibration)
        point_scores = pd.read_csv(integration / "point_scores.csv")
        uncertainty_scores = pd.read_csv(integration / "uncertainty_scores.csv")
        for position, group in players.groupby("position"):
            model_id = f"{position.lower()}_weekly"
            path = tmp / f"{model_id}.json"
            write(
                path,
                dict(
                    campaign_id=CAMPAIGN,
                    model_id=model_id,
                    season=season,
                    origin_week=week - 1,
                    forecast_week=week,
                    horizon="weekly",
                    training_through_season=2025,
                    generated_at=generated,
                    forecast_kind="prospective_v32_shadow",
                    retrospective=False,
                    capture_hashes=capture_hashes,
                    rows=rows(group),
                    integration_manifest_sha256=digest(integration / "manifest.json"),
                    lineup_calibration_artifact=f"forecast_lineup_calibration_{season}_{week - 1}",
                ),
            )
            forecasts = []
            historical_calibrations = {}
            if old:
                previous = next(m for m in old["models"] if m["model_id"] == model_id)
                historical_calibrations = {
                    name: registry / CAMPAIGN / ref["path"]
                    for name, ref in previous["artifacts"].items()
                    if name.startswith("forecast_lineup_calibration_")
                }
                forecasts = [
                    dict(r, path=registry / CAMPAIGN / r["path"])
                    for r in previous["forecasts"]
                    if (r["season"], r["origin_week"]) != (season, week - 1)
                ]
            forecasts.append(
                dict(season=season, origin_week=week - 1, rolling_origin=True, path=path)
            )
            metrics_path = tmp / f"{model_id}_metrics.json"
            write(
                metrics_path,
                dict(
                    point_scores=json.loads(
                        point_scores[
                            point_scores.position.eq(position)
                            & point_scores.method.isin(["v3", "selected"])
                            & point_scores.cohort.eq("all")
                        ].to_json(orient="records")
                    ),
                    uncertainty_scores=json.loads(
                        uncertainty_scores[uncertainty_scores.position.eq(position)].to_json(
                            orient="records"
                        )
                    ),
                    prospective="Pending until captured games complete; historical years reused",
                ),
            )
            entries.append(
                dict(
                    model_id=model_id,
                    position=position,
                    horizon="weekly",
                    origin_week=0,
                    targets=["points"],
                    training_through_season=2025,
                    scoring=SCORING["QB" if position == "QB" else "RB/WR/TE"],
                    metrics=metrics_path,
                    forecasts=forecasts,
                    artifacts=dict(
                        **(
                            historical_calibrations
                            | {f"forecast_lineup_calibration_{season}_{week - 1}": cal_path}
                        ),
                        lineup_calibration=cal_path,
                        integration_manifest=integration / "manifest.json",
                        selection=integration / "selection.json",
                        uncertainty=integration / "uncertainty.json",
                        early_weights=integration / "models" / f"early_week_{position}_2026.joblib",
                        pregame_weights=integration
                        / "models"
                        / f"pregame_60m_{position}_2026.joblib",
                    ),
                )
            )
        metadata = dict(
            campaign_id=CAMPAIGN,
            name="Opening Drive v3.2 preview",
            kind="weekly_distribution",
            description="Injury/workload weekly forecasts "
            "with coherent remaining and season totals",
            generated_at=now().isoformat(),
            latest_capture_hashes=capture_hashes,
            projection_provenance=projection["provenance"],
            horizon_contract=dict(
                version=1,
                remaining="sum of future weekly curve",
                season="known actuals plus remaining",
                averages="horizon points divided by scheduled games",
            ),
            integration_manifest_sha256=digest(integration / "manifest.json"),
            status="shadow_preview",
            limitations=[
                "Research-preview designation is independent of the application default selection.",
                "Later-model gains require timely official game-week reports; "
                "provider tags alone do not qualify.",
                "Participation probability estimates offensive participation, "
                "not official active status.",
                calibration["range_reliability_note"],
                *projection["policy"]["limitations"],
                "Season per-game denominator uses the current team's full schedule; "
                "earned points follow the player across past teams.",
            ],
        )
        if old is None:
            publish_campaign(registry, metadata, entries)
        else:
            candidates = tmp / "candidates"
            publish_campaign(candidates, metadata, entries)
            replace_campaign(
                registry,
                candidates,
                CAMPAIGN,
                old["manifest_sha256"],
                registry.parent / "campaign_revisions",
            )
    result = store.check(CAMPAIGN)
    return dict(status="published", verification=result)
