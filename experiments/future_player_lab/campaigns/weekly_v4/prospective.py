"""Immutable v4/v3.2 paired forecasts, replay, and future-outcome scoring."""

import argparse
import fcntl
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..weekly_v3 import data as v3_data
from ..weekly_v3.lineups import aggregate
from ..weekly_v31.artifacts import digest, now, publish_directory, seal, timestamp, verify, write
from ..weekly_v32 import uncertainty as v32_uncertainty
from ..weekly_v32.publisher import rows as weekly_rows
from ..weekly_v32.shadow import DEFAULT_LOG as BASE_LOG
from . import horizons, retraining, uncertainty
from .artifacts import BASE, DEFAULT_INTEGRATION, DEFAULT_LOG, check
from .lineup_evaluation import score_lineups


def outcomes(snapshot, registry):
    """Reuse accepted label eligibility, preserving unknowns instead of zero filling."""
    snapshot, registry = Path(snapshot), Path(registry)
    document = json.loads(snapshot.read_text())
    current = document["current_observations"]
    if document["status"] != "accepted":
        raise ValueError("V4 labels require an accepted snapshot")
    previous = v3_data.SNAPSHOT, v3_data.CAMPAIGN
    try:
        v3_data.SNAPSHOT, v3_data.CAMPAIGN = snapshot, registry / "opening_drive_v2"
        with tempfile.TemporaryDirectory(prefix=".v4-labels-") as tmp:
            panel, _, _ = v3_data.build_panel(
                tmp, start_year=current["season"], end_year=current["season"]
            )
    finally:
        v3_data.SNAPSHOT, v3_data.CAMPAIGN = previous
    return panel[panel.week.le(current["through_week"])][
        ["player_id", "season", "week", "target_points"]
    ]


def completed_training(base_log, snapshot, registry, season, week):
    candidates = []
    for path in sorted((Path(base_log) / "forecasts").glob("*/*/manifest.json")):
        receipt = verify(path.parent)
        if receipt["season"] == season and receipt["week"] < week:
            if receipt["status"] != "prospective" or timestamp(
                receipt["generated_at"]
            ) >= timestamp(receipt["deadline"]):
                raise ValueError("Retraining labels require genuine earlier forecasts")
            frame = pd.read_parquet(path.parent / "players.parquet")
            frame["training_capture"] = str(path.parent)
            frame["training_capture_sha256"] = digest(path)
            candidates.append(frame)
    if not candidates:
        return pd.DataFrame()
    labels = outcomes(snapshot, registry)
    frame = pd.concat(candidates, ignore_index=True).drop(columns="target_points", errors="ignore")
    frame = frame.merge(
        labels, on=["player_id", "season", "week"], how="left", validate="many_to_one"
    )
    return frame[frame.target_points.notna() & frame.is_bye.eq(0)].copy()


def predictions(source, integration, completed):
    contract = check(integration)
    source = Path(source)
    original = pd.read_parquet(source / "players.parquet")
    uncertainty_run = Path(contract["dependencies"]["uncertainty"]["path"])
    players = uncertainty.apply(original, uncertainty_run)
    challenger = retraining.apply(players, uncertainty_run, completed)
    panel = pd.read_parquet(source / "baseline_panel.parquet")
    projection = horizons.build_projection(
        players,
        panel,
        source / "snapshot.json",
        Path(contract["dependencies"]["horizons"]["path"]),
    )
    saved = {r["player_id"]: r for r in weekly_rows(players)}
    for row in projection["weekly"]:
        extra = saved[row["player_id"]]
        for key in (
            "distribution",
            "availability",
            "participation_probability",
            "expected_workload",
            "opponent",
        ):
            row[key] = extra.get(key)
        row["distribution"]["distribution_method"] = "v4_chronological_empirical_errors"
    for horizon in ("weekly", "remaining", "season"):
        for row in projection[horizon]:
            row["base_campaign"] = BASE
    total_contract = json.loads((Path(integration) / "horizon_uncertainty.json").read_text())
    remaining = projection["remaining"]
    frame = pd.DataFrame(
        [
            dict(
                player_id=r["player_id"],
                position=r["position"],
                season=r["season"],
                origin_week=r["origin_week"],
                prediction=r["predictions"]["points"],
            )
            for r in remaining
        ]
    )
    # A last-game total is exactly its weekly distribution. Earlier origins use
    # observed errors of fixed-origin totals, preserving across-week dependence.
    draws = np.zeros((len(frame), 256))
    multi = np.array(
        [
            r["scheduled_games"] > 1
            or (
                r["scheduled_games"] == 1
                and r["projection"]["future_weeks"][0]["week"] != r["origin_week"] + 1
            )
            for r in remaining
        ]
    )
    if multi.any():
        draws[multi] = uncertainty.total_samples(frame.loc[multi], total_contract)
    by_id = players.set_index("player_id")
    for i, row in enumerate(remaining):
        if not multi[i] and row["scheduled_games"]:
            draws[i] = np.asarray(by_id.loc[row["player_id"], "samples"])
    distributions = v32_uncertainty.frame_with_samples(frame, draws).set_index("player_id")
    for horizon in ("remaining", "season"):
        for row in projection[horizon]:
            result = distributions.loc[row["player_id"]]
            shift = row["projection"]["actual_points_to_date"] if horizon == "season" else 0.0
            row["distribution"] = {
                **{c: float(result[c]) + shift for c in v32_uncertainty.QCOLS},
                "std": float(result["std"]),
                "samples": (np.asarray(result["samples"]) + shift).tolist(),
                "distribution_method": "v4_fixed_origin_total_errors",
            }
    return players, challenger, projection


def lineups(source, players, challenger, integration):
    """Preserve baseline memberships; do not choose lineups using future outcomes."""
    source = Path(source)
    path = source / "lineups.parquet"
    if not path.exists():
        return pd.DataFrame()
    original = pd.read_parquet(path)
    if original.empty:
        return original
    contract = check(integration)
    baseline = Path(contract["dependencies"]["baseline"]["path"])
    horizon = verify(source)["forecast_horizon"]
    dep = json.loads((baseline / "dependence.json").read_text())[horizon]
    memberships = original.drop_duplicates(["design", "team_id", "player_ids"])
    result = []
    frames = {
        "v3.2": pd.read_parquet(source / "players.parquet"),
        "v4": players,
        "v4_retrained": challenger,
    }
    for method, frame in frames.items():
        mapping = frame.set_index("player_id").to_dict("index")
        for member in memberships.to_dict("records"):
            ids = member["player_ids"].split(",")
            if not set(ids) <= set(mapping):
                continue
            roster = [dict(player_id=pid, **mapping[pid]) for pid in ids]
            for row in roster:
                row["samples"] = list(row["samples"])
            prediction = aggregate(
                roster,
                v32_uncertainty.estimates(dep),
                seed=int(member["season"]) * 10000
                + int(member["week"]) * 100
                + int(member["team_id"]),
            )
            result.append(
                dict(
                    season=member["season"],
                    week=member["week"],
                    design=member["design"],
                    team_id=member["team_id"],
                    player_ids=member["player_ids"],
                    method=method,
                    **prediction,
                )
            )
    return pd.DataFrame(result)


def capture(source, destination, integration, completed):
    source, destination, integration = map(Path, (source, destination, integration))
    receipt = verify(source)
    contract = check(integration)
    if receipt["status"] != "prospective":
        raise ValueError("V4 requires a prospective v3.2 source")
    if receipt["integration_sha256"] != contract["dependencies"]["baseline"]["sha256"]:
        raise ValueError("V3.2 capture does not match the declared base integration")
    deadline = timestamp(receipt["deadline"])
    if now() >= deadline:
        raise ValueError("V4 cannot backfill a forecast after its deadline")
    players, challenger, projection = predictions(source, integration, completed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".v4-capture-") as tmp:
        staging = Path(tmp) / "capture"
        staging.mkdir()
        # Keep complete source receipt and inputs so replay never needs a mutable source.
        shutil.copytree(
            source,
            staging / "baseline",
            ignore=shutil.ignore_patterns("manifest.json", "manifest.sha256"),
        )
        shutil.copy2(source / "manifest.json", staging / "baseline_receipt.json")
        players.to_parquet(staging / "players.parquet", index=False)
        challenger.to_parquet(staging / "retrained.parquet", index=False)
        completed.to_parquet(staging / "completed_forecasts.parquet", index=False)
        write(staging / "projections.json", projection)
        lineups(source, players, challenger, integration).to_parquet(
            staging / "lineups.parquet", index=False
        )
        generated = now()
        if generated >= deadline:
            raise ValueError("V4 inference missed the prospective cutoff")
        seal(
            staging,
            dict(
                status="prospective",
                generated_at=generated.isoformat(),
                evidence_cutoff=receipt["evidence_cutoff"],
                deadline=receipt["deadline"],
                forecast_horizon=receipt["forecast_horizon"],
                season=receipt["season"],
                week=receipt["week"],
                base_campaign=BASE,
                base_capture_sha256=digest(source / "manifest.json"),
                integration=str(integration.resolve()),
                integration_sha256=digest(integration / "manifest.json"),
                current_season_training_rows=len(completed),
            ),
        )
        if now() >= deadline:
            raise ValueError("V4 publication missed the prospective cutoff")
        publish_directory(staging, destination)


def replay(directory):
    directory = Path(directory)
    receipt = verify(directory)
    if (
        not timestamp(receipt["evidence_cutoff"])
        <= timestamp(receipt["generated_at"])
        < timestamp(receipt["deadline"])
    ):
        raise ValueError("V4 capture violates its prospective cutoff")
    integration = Path(receipt["integration"])
    if digest(integration / "manifest.json") != receipt["integration_sha256"]:
        raise ValueError("V4 integration changed")
    if digest(directory / "baseline_receipt.json") != receipt["base_capture_sha256"]:
        raise ValueError("V4 baseline capture changed")
    players, challenger, projection = predictions(
        directory / "baseline",
        integration,
        pd.read_parquet(directory / "completed_forecasts.parquet"),
    )
    for name, frame in (("players", players), ("retrained", challenger)):
        saved = pd.read_parquet(directory / f"{name}.parquet")
        if saved.player_id.tolist() != frame.player_id.tolist():
            raise ValueError("Replay population differs")
        for column in ("prediction", "std", *v32_uncertainty.QCOLS):
            if not np.allclose(saved[column], frame[column], rtol=0, atol=1e-10):
                raise ValueError(f"V4 replay differs: {name}/{column}")
        if not np.allclose(np.stack(saved.samples), np.stack(frame.samples), rtol=0, atol=1e-10):
            raise ValueError("V4 replay distributions differ")
    if json.loads((directory / "projections.json").read_text()) != projection:
        raise ValueError("V4 horizon/component replay differs")
    return dict(status="verified", players=len(players), capture=str(directory))


def score(log, snapshot, registry):
    log, snapshot = Path(log), Path(snapshot)
    current = json.loads(snapshot.read_text())["current_observations"]
    paths = sorted((log / "forecasts").glob("*/*/manifest.json"))
    receipts = [verify(p.parent) for p in paths]
    key = dict(snapshot_sha256=digest(snapshot), captures={str(p.parent): digest(p) for p in paths})
    previous = log / "scorecard.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        if saved.get("inputs") == key:
            return saved
    has_labels = any(
        r["season"] == current["season"] and r["week"] <= current["through_week"] for r in receipts
    )
    labels = (
        outcomes(snapshot, registry)
        if has_labels
        else pd.DataFrame(columns=["player_id", "season", "week", "target_points"])
    )
    scores, support, lineup_rows = [], [], []
    for path, receipt in zip(paths, receipts, strict=True):
        if receipt["status"] != "prospective" or timestamp(receipt["generated_at"]) >= timestamp(
            receipt["deadline"]
        ):
            raise ValueError("Only prospective v4 forecasts may be scored")
        directory = path.parent
        actual = None
        for method, name in (
            ("v3.2", "baseline/players"),
            ("v4", "players"),
            ("v4_retrained", "retrained"),
        ):
            frame = pd.read_parquet(directory / f"{name}.parquet")
            frame = frame.drop(columns="target_points", errors="ignore").merge(
                labels, on=["player_id", "season", "week"], how="left", validate="one_to_one"
            )
            frame.loc[frame.is_bye.eq(1), "target_points"] = 0
            observed = frame[frame.is_bye.eq(0) & frame.target_points.notna()]
            if method == "v4":
                actual = frame.set_index("player_id").target_points
                support.append(
                    dict(
                        capture=str(directory),
                        season=receipt["season"],
                        week=receipt["week"],
                        horizon=receipt["forecast_horizon"],
                        observed=len(observed),
                        pending=int((frame.is_bye.eq(0) & frame.target_points.isna()).sum()),
                    )
                )
            for position, group in observed.groupby("position"):
                scores.append(
                    dict(
                        capture=str(directory),
                        method=method,
                        position=position,
                        **uncertainty.metrics(group, np.stack(group.samples)),
                    )
                )
        frozen_lineups = score_lineups(directory, actual, receipt["integration"])
        if not frozen_lineups.empty:
            frozen_lineups["capture"] = str(directory)
            lineup_rows.append(frozen_lineups)
    destination = log / "evaluations" / now().strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(scores).to_csv(destination / "scores.csv", index=False)
    pd.DataFrame(support).to_csv(destination / "support.csv", index=False)
    (pd.concat(lineup_rows, ignore_index=True) if lineup_rows else pd.DataFrame()).to_csv(
        destination / "lineups.csv", index=False
    )
    result = dict(
        status="pending" if not scores else "scored",
        inputs=key,
        observed_players=sum(r["observed"] for r in support),
        pending_players=sum(r["pending"] for r in support),
        evaluation=str(destination),
        default_changed=False,
    )
    write(destination / "summary.json", result)
    seal(destination, dict(generated_at=now().isoformat(), **key))
    write(previous, result)
    return result


def track(snapshot, registry, integration=DEFAULT_INTEGRATION, log=DEFAULT_LOG, base_log=BASE_LOG):
    from .publisher import publish

    snapshot, registry, integration, log, base_log = map(
        Path, (snapshot, registry, integration, log, base_log)
    )
    check(integration)
    current = json.loads(snapshot.read_text())["current_observations"]
    season, week = current["season"], current["through_week"] + 1
    log.mkdir(parents=True, exist_ok=True)
    with (log / ".tracking.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        early = base_log / "forecasts" / f"{season}_{week:02d}" / "early_week"
        destinations, skipped = [], []
        if early.exists():
            sources = [early, *sorted(early.parent.glob("pregame_*"))]
            completed = None
            for source in sources:
                destination = log / "forecasts" / early.parent.name / source.name
                if destination.exists():
                    receipt = verify(destination)
                    if receipt["integration_sha256"] != digest(integration / "manifest.json"):
                        raise ValueError("Origin already belongs to another v4 integration")
                elif now() >= timestamp(verify(source)["deadline"]):
                    skipped.append(dict(source=str(source), reason="missed_prospective_deadline"))
                    if source == early:
                        break
                    continue
                else:
                    if completed is None:
                        completed = completed_training(base_log, snapshot, registry, season, week)
                    capture(source, destination, integration, completed)
                destinations.append(destination)
        publication = publish(registry, integration, destinations) if destinations else None
        result = dict(
            status="tracking" if destinations else "awaiting_eligible_baseline",
            publication=publication,
            skipped=skipped,
            scorecard=score(log, snapshot, registry),
        )
        write(log / "status.json", dict(updated_at=now().isoformat(), **result))
        return result


def main():
    from ..rescore import REGISTRY, latest_raw_snapshot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("track", "score", "replay"), default="track", nargs="?")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--integration", type=Path, default=DEFAULT_INTEGRATION)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--capture-dir", type=Path)
    args = parser.parse_args()
    if args.action == "replay":
        if not args.capture_dir:
            parser.error("replay requires --capture-dir")
        result = replay(args.capture_dir)
    elif args.action == "score":
        result = score(args.log, args.snapshot or latest_raw_snapshot(), args.registry)
    else:
        result = track(
            args.snapshot or latest_raw_snapshot(), args.registry, args.integration, args.log
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
