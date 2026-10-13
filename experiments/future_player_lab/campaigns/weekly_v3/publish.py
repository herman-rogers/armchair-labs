"""Seal v3 weekly models alongside byte-preserved v2 horizon model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine.chairstats.store import CampaignStore, publish_campaign

from .modeling import QCOLS, QUANTILES
from .replay import rows_digest
from .report import clean

ROOT = Path(__file__).resolve().parents[4]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")
    return path


def verify_original(root=ROOT):
    directory = root / "data/campaigns/opening_drive_v2"
    before = json.loads(
        (root / "data/research/opening_drive_v3_20261011/v2_before.json").read_text()
    )
    after = {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }
    if before != after:
        raise ValueError("Original v2 artifacts changed during the experiment")
    return dict(files=len(before), manifest_sha256=before["manifest.json"], unchanged=True)


def freeze_lineup_case(run, staging):
    """Bind a captured league, identity joins and schedule to the runtime result."""
    from engine.espn.sync import LeagueSnapshot
    from engine.metrics.weekly_distribution import forecast_snapshot
    from engine.tables import observations
    from engine.tables.application import connect, query_scope

    raw = json.loads((run / "lineups/league_snapshot.json").read_text())
    snapshot = LeagueSnapshot.from_dict(raw)
    with query_scope(ROOT / "data"), connect() as db:
        identities = observations.espn_ids(db)
        schedule = observations.schedule(db, snapshot.season)
    players = []
    for pos in ("qb", "rb", "wr", "te"):
        path = staging / f"{pos}_weekly" / f"{snapshot.season}_{snapshot.week - 1}.json"
        for row in json.loads(path.read_text())["rows"]:
            players.append(
                dict(
                    **row["distribution"],
                    player_id=row["player_id"],
                    player_display_name=row.get("player_name"),
                    position=row["position"],
                    team=row.get("team"),
                    opponent=row.get("opponent"),
                    prediction=row["predictions"]["points"],
                    season=snapshot.season,
                    week=snapshot.week,
                )
            )
    calibration = json.loads((run / "lineups/calibration.json").read_text())
    expected = forecast_snapshot(snapshot, players, identities, schedule, calibration)
    case = dict(
        snapshot=raw,
        identities=identities,
        schedule=schedule,
        players=players,
        calibration=calibration,
        expected=expected,
    )
    path = staging / "lineup_replay_case.json"
    path.write_text(json.dumps(clean(case), indent=2, allow_nan=False, default=str))
    return path


def publish(run_dir, registry=None):
    run = Path(run_dir)
    registry = Path(registry) if registry else ROOT / "data/campaigns"
    original = verify_original()
    audit = json.loads((run / "audit.json").read_text())
    if audit.get("status") != "passed":
        raise ValueError("An independent passing audit is required before publishing v3")
    source_store = CampaignStore(ROOT / "data/campaigns")
    source_store.check("opening_drive_v2")
    source = source_store.campaign("opening_drive_v2")
    staging = run / "export"
    staging.mkdir(exist_ok=True)
    entries = []
    for model in source["models"]:
        entry = dict(model)
        source_dir = ROOT / "data/campaigns/opening_drive_v2"
        entry["metrics"] = source_dir / model["metrics"]["path"]
        entry["artifacts"] = {
            name: source_dir / ref["path"] for name, ref in model["artifacts"].items()
        }
        entry.setdefault("source_campaign_id", "opening_drive_v2")
        entry.setdefault("source_manifest_sha256", source["manifest_sha256"])
        entry["inherited_from_campaign_id"] = "opening_drive_v2"
        entry["inherited_from_manifest_sha256"] = source["manifest_sha256"]
        entry["forecasts"] = []
        entry["original_forecast_rows_sha256"] = {}
        for ref in model["forecasts"]:
            payload = json.loads((source_dir / ref["path"]).read_text())
            original_rows = payload if isinstance(payload, list) else payload["rows"]
            entry["original_forecast_rows_sha256"][f"{ref['season']}_{ref['origin_week']}"] = (
                rows_digest(original_rows)
            )
            if isinstance(payload, dict):
                payload.update(
                    campaign_id="opening_drive_v3", source_campaign_id="opening_drive_v2"
                )
            path = write(
                staging / model["model_id"] / f"{ref['season']}_{ref['origin_week']}.json", payload
            )
            entry["forecasts"].append({**ref, "path": path})
        entries.append(entry)
    selected = pd.read_parquet(run / "selected_predictions.parquet")
    panel = pd.read_parquet(run / "panel.parquet")
    policies = json.loads((run / "selection.json").read_text())
    cohorts = pd.read_csv(run / "cohort_scores.csv")
    now = datetime.now(UTC).isoformat()
    forecasts_total = 0
    for pos, policy in policies.items():
        model_id = f"{pos.lower()}_weekly"
        live = selected[selected.season.eq(2026) & selected.position.eq(pos)].copy()
        byes = panel[panel.season.eq(2026) & panel.position.eq(pos) & panel.is_bye.eq(1)].copy()
        if not byes.empty:
            byes["prediction"], byes["std"] = 0.0, 0.0
            byes[QCOLS] = 0.0
            byes["mean_method"], byes["distribution_method"] = "scheduled_bye", "deterministic_zero"
            live = pd.concat([live, byes], ignore_index=True)
        snapshots = []
        for week, frame in live.groupby("week"):
            rows = []
            for rank, (_, row) in enumerate(
                frame.sort_values("prediction", ascending=False).iterrows(), 1
            ):
                samples = np.interp(
                    (np.arange(256) + 0.5) / 256, QUANTILES, row[QCOLS].to_numpy(dtype=float)
                )
                distribution = {c: float(row[c]) for c in QCOLS}
                distribution.update(
                    std=float(row["std"]),
                    samples=samples.tolist(),
                    mean_method=row["mean_method"],
                    distribution_method=row["distribution_method"],
                )
                rows.append(
                    dict(
                        player_id=row.player_id,
                        player_name=row.get("player_name"),
                        position=pos,
                        season=2026,
                        origin_week=int(week) - 1,
                        team=row.get("team"),
                        opponent=row.get("opponent"),
                        predictions={"points": float(row.prediction)},
                        scheduled_games=int(row.get("scheduled_game", 1))
                        if pd.notna(row.get("scheduled_game", 1))
                        else 1,
                        calendar_weeks=1,
                        points_per_scheduled_game=float(row.prediction),
                        rank=rank,
                        distribution=distribution,
                        candidate_basis=(
                            "Eligible at observation cutoff; scheduled byes are deterministic zero"
                        ),
                    )
                )
            payload = dict(
                campaign_id="opening_drive_v3",
                model_id=model_id,
                season=2026,
                origin_week=int(week) - 1,
                forecast_week=int(week),
                horizon="weekly",
                training_through_season=2025,
                generated_at=now,
                retrospective=True,
                rows=rows,
            )
            snapshots.append(
                dict(
                    season=2026,
                    origin_week=int(week) - 1,
                    rolling_origin=int(week) > 1,
                    path=write(staging / model_id / f"2026_{int(week) - 1}.json", payload),
                )
            )
            forecasts_total += len(rows)
        metrics_path = write(
            staging / model_id / "metrics.json",
            dict(
                policy=policy,
                evaluation="2021–2025 historical development player-week predictions",
                distribution_scores=cohorts[
                    cohorts.position.eq(pos) & cohorts.method.eq(policy["distribution_method"])
                ].to_dict("records"),
                replay=json.loads((run / "replay.json").read_text()),
                audit=audit,
            ),
        )
        artifacts = dict(
            mean_model=run / "models" / f"{pos}_{policy['mean_method']}.pkl",
            distribution_model=run
            / "models"
            / f"{pos}_{policy['mean_method']}_{policy['distribution_method']}_distribution.pkl",
            lineup_calibration=run / "lineups/calibration.json",
            dataset_contract=run / "dataset.json",
            candidate_contract=run / "candidate_metadata.json",
            run_manifest=run / "manifest.json",
            forecast_matrix=run / "panel.parquet",
        )
        for path in (run / "source_snapshot").glob("*.py"):
            artifacts[f"source_{path.stem}"] = path
        entries.append(
            dict(
                model_id=model_id,
                position=pos,
                horizon="weekly",
                origin_week=0,
                targets=["points"],
                training_through_season=2025,
                scoring=next(m.get("scoring") for m in source["models"] if m["position"] == pos),
                metrics=metrics_path,
                forecasts=snapshots,
                artifacts=artifacts,
            )
        )
    lineup_case = freeze_lineup_case(run, staging)
    for entry in entries:
        if entry["horizon"] == "weekly":
            entry["artifacts"].update(
                lineup_replay_case=lineup_case,
                source_lineup_runtime=ROOT / "src/engine/metrics/weekly_distribution.py",
                source_lineup_research=Path(__file__).with_name("lineups.py"),
                independent_audit=run / "audit.json",
                lineup_research_results=run / "lineups/results.json",
            )
    metadata = dict(
        campaign_id="opening_drive_v3",
        name="Opening Drive v3",
        generated_at=now,
        kind="weekly_distribution",
        description="Weekly means and predictive distributions with preserved v2 season models",
        positions=["QB", "RB", "WR", "TE"],
        source_campaign_id="opening_drive_v2",
        source_manifest_sha256=source["manifest_sha256"],
        original_preservation=original,
        scoring=source.get("scoring", {}),
        limitations=[
            "Historical development evaluation, not a pristine holdout or prospective guarantee.",
            "Season and remaining-season models are the preserved original v2 models; "
            "weekly models are newly trained.",
            "Current forecasts reconstruct pre-week information from later captures; "
            "publication and snapshot times can be after kickoff.",
            "Player distributions are mean-centered quantile approximations with bounded "
            "1%/99% tails; they can smooth the scoring-zero mass.",
            "Nominal 80% ranges undercovered in historical evaluation: approximately "
            "74% QB, 78% WR and 77% artificial lineups; high-projection TE coverage was "
            "about 65%. Ranges are research estimates, not verified coverage guarantees.",
            "Full-lineup ranges are provisional: K/DST uncertainty transfers prior-score "
            "errors to ESPN means, and their covariance is not estimated.",
            "Offensive scoring preserves the original v2 contract and can differ from "
            "official league scoring.",
        ],
    )
    result = publish_campaign(registry, metadata, entries)
    check = CampaignStore(registry).check("opening_drive_v3")
    verify_original()
    write(
        run / "publication.json",
        dict(**check, weekly_forecasts=forecasts_total, original_preservation=original),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    print(publish(args.run_dir, args.registry))
