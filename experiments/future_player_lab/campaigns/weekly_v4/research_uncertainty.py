"""Seal a reproducible uncertainty and residual-retraining experiment for v4."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from ..weekly_v3 import lineups
from ..weekly_v31.artifacts import digest, now, publish_directory, seal, verify, write
from ..weekly_v32 import uncertainty as base
from ..weekly_v32.run import DEFAULT_RUN as BASE_RUN
from . import retraining, uncertainty

DEFAULT_RUN = BASE_RUN.parent / "opening_drive_v4_uncertainty_20261027_002"


def evaluate_lineups(predictions, history, panel):
    context = panel.drop_duplicates("row_id").set_index("row_id")
    history = history.copy()
    history["opponent"] = history.row_id.map(context.opponent)
    predictions = predictions.copy()
    predictions["opponent"] = predictions.row_id.map(context.opponent)
    rows = []
    for (horizon, season), group in predictions.groupby(["horizon", "season"]):
        prior = history[history.horizon.eq(horizon) & history.season.lt(season)]
        dependence = base.estimates(base.dependence(prior, int(season)))
        for week, part in group.groupby("week"):
            controls = part[part.distribution_method.eq("v32")]
            rosters = base.draft(controls, "prediction")
            for method, frame in part.groupby("distribution_method"):
                lookup = frame.set_index("player_id").to_dict("index")
                for index, ids in enumerate(rosters):
                    roster = [dict(player_id=p, **lookup[p]) for p in ids]
                    actual = sum(p["target_points"] for p in roster)
                    score = lineups.aggregate(
                        roster,
                        dependence,
                        seed=int(season * 10000 + week * 10 + index),
                        actual=actual,
                    )
                    rows.append(
                        dict(
                            horizon=horizon,
                            season=int(season),
                            week=int(week),
                            lineup=index,
                            method=method,
                            actual=actual,
                            player_ids=",".join(ids),
                            **score,
                        )
                    )
        print(f"Lineups {horizon} {season}", flush=True)
    table = pd.DataFrame(rows)
    scores = [
        dict(horizon=h, season=int(s), method=m, **lineups.metrics(g.to_dict("records")))
        for (h, s, m), g in table.groupby(["horizon", "season", "method"])
    ]
    return table, pd.DataFrame(scores)


def run(output=DEFAULT_RUN, research=BASE_RUN):
    output, research = Path(output), Path(research)
    if output.exists():
        raise FileExistsError("Use a new immutable uncertainty run")
    verify(research)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v4-uncertainty-", dir=output.parent) as tmp:
        stage = Path(tmp) / "run"
        stage.mkdir()
        write(
            stage / "protocol.json",
            dict(
                version="v4_uncertainty_and_inseason_residual_1",
                baseline="opening_drive_v32",
                scales=list(uncertainty.SCALES),
                selection="Equal-year CRPS, strictly earlier years",
                chronology="2021 seed; 2022 validation; 2023–25 evaluation; 2026 frozen fits",
                point_policy="Uncertainty leaves point means unchanged",
                retraining="Fixed ridge residual correction; prior 3 seasons plus completed "
                "earlier current-season weeks weighted 5; alpha 100; correction clipped to +/-5",
                controls=["v32 frozen", "v4 calibration-only", "prior-only residual ridge"],
                prospective="Immutable pregame forecast first, label only after accepted results; "
                "no default change, no retrospective backfill",
                lineup="Matched v32-ranked artificial offense rosters; identical earlier-error "
                "dependence; K/DST and mixed timing withheld",
                promotion="Manual review; future forecasts required. Reused historical years "
                "cannot establish prospective performance.",
                limitations=[
                    "Historical source publication vintages are reconstructed",
                    "Only 3 evaluation seasons; subgroup support varies",
                    "Residual adaptation is not a full base-model refit",
                    "No automatic retraining or default promotion",
                ],
            ),
        )
        history = pd.read_parquet(
            research / "predictions.parquet", filters=[("method", "==", "selected")]
        )
        history.to_parquet(stage / "history.parquet", index=False)
        print("Evaluating player distributions", flush=True)
        scores, predictions, selections, contracts = uncertainty.evaluate(history)
        scores.to_csv(stage / "scores.csv", index=False)
        predictions.to_parquet(stage / "predictions.parquet", index=False)
        write(stage / "selection.json", selections)
        write(stage / "uncertainty.json", contracts)
        print("Evaluating in-season residual candidate", flush=True)
        retrain_scores, retrain_predictions, fits = retraining.evaluate(history, predictions)
        retrain_scores.to_csv(stage / "retraining_scores.csv", index=False)
        retrain_predictions.to_parquet(stage / "retraining_predictions.parquet", index=False)
        write(stage / "retraining_fit_receipts.json", fits)
        write(
            stage / "retraining_contract.json",
            dict(
                forecast_season=2026,
                features=list(retraining.FEATURES),
                alpha=retraining.ALPHA,
                in_season_weight=5,
                max_correction=5,
                prior_seasons=3,
                default_changed=False,
                policy="isolated_in_season_residual_ridge",
            ),
        )
        rows, lineup_scores = evaluate_lineups(
            predictions, history, pd.read_parquet(research / "panel.parquet")
        )
        rows.to_parquet(stage / "lineup_predictions.parquet", index=False)
        lineup_scores.to_csv(stage / "lineup_scores.csv", index=False)
        for filename in ("uncertainty.py", "retraining.py", "research_uncertainty.py"):
            source = Path(__file__).with_name(filename)
            (stage / "source").mkdir(exist_ok=True)
            shutil.copy2(source, stage / "source" / filename)
        seal(
            stage,
            dict(
                status="v4_research_candidate",
                generated_at=now().isoformat(),
                research_run=str(research.resolve()),
                baseline_manifest_sha256=digest(research / "manifest.json"),
                forecast_season=2026,
                default_changed=False,
            ),
        )
        publish_directory(stage, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    print(run(args.output))
