"""Append prospective component/horizon evidence without changing frozen inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..weekly_v31.artifacts import digest, now, seal, verify, write
from .artifacts import DEFAULT_INTEGRATION, DEFAULT_LOG, check
from .horizon_evaluation import score_projection
from .prospective import score as score_weekly


def summarize(frame):
    known = frame[frame.status.eq("scored")]
    if known.empty:
        return pd.DataFrame()
    groups = ["method", "position", "horizon", "target"]
    values = [
        c
        for c in (
            "squared_error",
            "absolute_error",
            "error",
            "crps",
            "coverage80",
            "width80",
            "interval_score80",
            "coverage90",
            "interval_score90",
        )
        if c in known
    ]
    result = known.groupby(groups)[values].mean()
    result["n"] = known.groupby(groups).size()
    result["rmse"] = np.sqrt(result.pop("squared_error"))
    return result.reset_index()


def evaluate(snapshot, registry, integration=DEFAULT_INTEGRATION, log=DEFAULT_LOG):
    from .control_evaluation import capture_controls

    snapshot, registry, integration, log = map(Path, (snapshot, registry, integration, log))
    contract = check(integration)
    weekly = score_weekly(log, snapshot, registry)
    paths = sorted((log / "forecasts").glob("*/*/manifest.json"))
    frames, controls, active = [], [], set()
    for path in paths:
        directory = path.parent
        receipt = verify(directory)
        frame = score_projection(directory, snapshot)
        frame["forecast_timing"] = receipt["forecast_horizon"]
        frames.append(frame)
        players = pd.read_parquet(directory / "players.parquet")
        active.update(
            (int(r.season), int(r.week) - 1, r.player_id)
            for r in players[players.is_bye.eq(0)].itertuples()
        )
        control = capture_controls(directory, registry, log)
        controls.append(control)
        if control.get("path"):
            base = score_projection(control["path"], snapshot)
            base["method"] = "v3.2"
            base["forecast_timing"] = receipt["forecast_horizon"]
            frames.append(base)
    evidence = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    observed = set()
    completed_weeks = set()
    if not evidence.empty:
        scored = evidence[
            evidence.method.eq("v4")
            & evidence.horizon.eq("weekly")
            & evidence.target.eq("points")
            & evidence.status.eq("scored")
        ]
        observed = {(int(r.season), int(r.origin_week), r.player_id) for r in scored.itertuples()}
        observed &= active
        completed_weeks = {(season, origin + 1) for season, origin, _ in observed}
    policy = json.loads((integration / "release_policy.json").read_text())
    minimum = policy["prospective_minimum"]
    blockers = []
    if len(completed_weeks) < minimum["completed_weeks"]:
        blockers.append("Insufficient completed prospective weeks")
    if len(observed) < minimum["observed_player_weeks"]:
        blockers.append("Insufficient unique paired prospective player-weeks")
    bridge = pd.read_csv(
        Path(contract["dependencies"]["horizons"]["path"]) / "v32_bridge_scores.csv"
    )
    bridge = bridge.pivot_table(index="position", columns="method", values="rmse", aggfunc="mean")
    regressions = bridge.index[bridge.v4 > bridge.v32_bridge].tolist()
    if regressions:
        blockers.append(
            "Historical bridge RMSE regressions require explicit review: " + ", ".join(regressions)
        )
    if (
        evidence.empty
        or not ((evidence.horizon.eq("remaining")) & evidence.status.eq("scored")).any()
    ):
        blockers.append("Whole-season prospective outcomes have not matured")
    destination = log / "review_evaluations" / now().strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    evidence.to_parquet(destination / "target_predictions.parquet", index=False)
    if not evidence.empty:
        summarize(evidence).to_csv(destination / "target_scores.csv", index=False)
        evidence.groupby(["method", "horizon", "target", "status"]).size().rename("n").to_csv(
            destination / "support.csv"
        )
    result = dict(
        status="prospective_candidate",
        default_changed=False,
        automatic_promotion=False,
        unique_observed_player_weeks=len(observed),
        unique_pending_player_weeks=len(active - observed),
        completed_prospective_weeks=len(completed_weeks),
        historical_regression_positions=regressions,
        review_blockers=blockers,
        weekly_evaluation=weekly,
        horizon_component_evaluation=str(destination),
        controls=controls,
        limitations=[
            "Early-week whole-total residual calibration transfers to pregame means "
            "without separate timing validation",
            "Bye-origin horizon fallback is functional but not separately historically validated",
            "Direct point and component forecasts are separate estimates",
        ],
    )
    write(destination / "review.json", result)
    seal(
        destination,
        dict(
            generated_at=now().isoformat(),
            snapshot_sha256=digest(snapshot),
            captures={str(p.parent): digest(p) for p in paths},
            evaluation_sources={
                p.name: digest(p) for p in Path(__file__).parent.glob("*evaluation.py")
            },
        ),
    )
    write(log / "review.json", result)
    return result


if __name__ == "__main__":
    from ..rescore import REGISTRY, latest_raw_snapshot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--integration", type=Path, default=DEFAULT_INTEGRATION)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.snapshot or latest_raw_snapshot(), args.registry, args.integration, args.log
            ),
            indent=2,
        )
    )
