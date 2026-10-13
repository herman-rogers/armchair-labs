"""Evaluate interval-only v3.1 corrections and freeze the 2026 challenger."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from engine.chairstats import CampaignStore

from ..weekly_v3 import rescore as v3_rescore
from ..weekly_v3.data import ROOT, SCORING
from ..weekly_v3.modeling import QCOLS
from ..weekly_v3.run import DEFAULT_OUTPUT as V3_RUN
from . import calibration
from .artifacts import digest, now, publish_directory, seal, write

DEFAULT_RUN = ROOT / "experiments/future_player_lab/runs/opening_drive_v31_20261013_001"
DEFAULT_LOG = ROOT / "data/outputs/opening_drive_v31_prospective"
DEFAULT_REPORT = ROOT / "data/outputs/opening_drive_v31_comparison_2026"


def summarize(frame, method, *, adjusted=False, levels=(80, 90)):
    records = []
    for position, group in frame.groupby("position"):
        cohorts = {"all": group}
        if position != "LINEUP":
            cohorts.update(
                {
                    name: group[group.prediction.map(calibration.workload).eq(name)]
                    for name in ("under_5", "5_to_10", "10_plus")
                }
            )
        else:
            for design, part in group.groupby("design"):
                cohorts[design] = part
            for stacked, part in group.groupby("has_qb_receiver_stack"):
                cohorts[f"stack_{stacked}"] = part
        for name, part in cohorts.items():
            for season, subset in [("all", part), *list(part.groupby("season"))]:
                records.append(
                    dict(
                        position=position,
                        cohort=name,
                        season=str(season),
                        method=method,
                        **calibration.score(subset, adjusted=adjusted, levels=levels),
                    )
                )
    return pd.DataFrame(records)


def evaluate(frame, *, levels=(80, 90), first_year=2021, min_rows=200):
    folds, receipts = [], []
    for year in range(first_year, 2026):
        test = frame[frame.season.eq(year)]
        past = frame[frame.season.lt(year)]
        for policy in calibration.POLICIES:
            bundle = calibration.fit(past, year, policy=policy, levels=levels, min_rows=min_rows)
            predicted = calibration.apply(test, bundle)
            predicted["interval_method"] = policy
            folds.append(predicted)
            receipts.append(bundle)
    predictions = pd.concat(folds, ignore_index=True)
    baseline = frame[frame.season.between(first_year, 2025)]
    scores = [summarize(baseline, "v3", levels=levels)]
    for policy in calibration.POLICIES:
        scores.append(
            summarize(
                predictions[predictions.interval_method.eq(policy)],
                policy,
                adjusted=True,
                levels=levels,
            )
        )
    return predictions, pd.concat(scores, ignore_index=True), receipts


def report(directory, players, lineups):
    def table(frame):
        view = frame[frame.season.eq("all")]
        cols = ["position", "cohort", "method", "n", "coverage_80", "width_80", "interval_score_80"]
        return view[cols].to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0)

    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Opening Drive v3.1</title><style>
body{font:16px system-ui;color:#152738;background:#f0f3f5;margin:0}
main{max-width:1200px;margin:auto;padding:32px}section{background:white;padding:24px;margin:24px 0}
p,li{line-height:1.6;max-width:960px}table{border-collapse:collapse;font-size:13px}
td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}
a{color:#087e8b}</style><main><h1>Opening Drive v3.1</h1>
<p>Past-only interval recalibration and prospective tracking. The v3 point forecasts,
quantiles, standard deviations and joint-draw distributions remain frozen.</p>
<section><h2>How to read this experiment</h2><p>The designated challenger is
<b>position_workload</b>: corrections pooled by position and predicted points (below 5,
5–10, 10+), falling back to position with fewer than 200 prior observations.
The position-only method is a diagnostic control. Neither is selected using these results.</p>
<p>Each forecast season uses only the previous five seasons of out-of-time errors.
The width-scaled correction expands intervals using a finite-sample order statistic;
it never changes predicted points. The 90% interval encloses the 80% interval.
Lower interval score is better: it penalizes both unnecessary width and missed outcomes.</p>
<p>These years were already used for development. Temporal dependence and drift mean
these corrections are not coverage guarantees. The adjusted bounds are intervals,
not replacement P10/P90 quantiles; full-distribution CRPS is unchanged.</p></section>
<section><h2>Player intervals, 2021–2025</h2><p>All eligible observed scheduled
player-weeks, including players without exact v2 forecasts. Byes are excluded.</p>
<div class="scroll">"""
    page += table(players)
    page += """</div></section><section><h2>Artificial offense lineups, 2022–2025</h2>
<p>Direct 80% interval correction of the original v3 joint lineup draws. Two fixed
roster selection designs and stacked/unstacked diagnostics. 2021 supplies the first
lineup calibration history; it is not included in this comparison. Each season uses
only earlier lineup outcomes. Lineup observations overlap, so counts are not independent
sample sizes. This does not validate K/DST or real-league coverage.</p><div class="scroll">"""
    page += table(lineups)
    page += """</div></section><section><h2>Prospective evaluation</h2>
<p>The tracking workflow locks the first complete paired v3/v3.1 forecast for each
week before 00:00 UTC on the first NFL game date. Inputs, model artifacts, calibration
and predictions are checksummed and archived together. Late captures are rejected;
missing outcomes remain pending. Artificial lineups retain their pregame membership.</p>
<p><a href="../opening_drive_v31_prospective/scorecard/index.html">
Live prospective scorecard</a></p>
<p>Historical evidence alone does not promote the challenger. Review prospective
coverage, width and interval score together; point accuracy must remain identical.</p>
</section><section><h2>Downloads and provenance</h2><p>
<a href="player_scores.csv">Player metrics</a> · <a href="lineup_scores.csv">Lineup metrics</a> ·
<a href="manifest.json">Run receipt</a></p><p>Method reference:
<a href="https://arxiv.org/abs/1905.03222">Conformalized Quantile Regression</a>.
This implementation uses a chronological, width-normalized adaptation.</p>
</section></main></html>"""
    (directory / "index.html").write_text(page)


def run(v3_run=V3_RUN, output=DEFAULT_RUN, registry=ROOT / "data/campaigns"):
    v3_run, output, registry = Path(v3_run), Path(output), Path(registry)
    if output.exists():
        raise FileExistsError("Use a fresh run directory; sealed v3.1 runs are immutable")
    original = json.loads((v3_run / "manifest.json").read_text())
    selected = v3_run / "selected_predictions.parquet"
    if digest(selected) != original["artifact_hashes"][selected.name]:
        raise ValueError("Original v3 prediction checksum mismatch")
    players = pd.read_parquet(selected)
    if players.duplicated(["season", "week", "player_id"]).any():
        raise ValueError("Duplicate v3 player weeks")
    if not (players.calibration_max_year < players.season).all():
        raise ValueError("Original distributions must be out of time")
    lineup_path = v3_run / "lineups/synthetic_lineups.parquet"
    lineups = pd.read_parquet(lineup_path).rename(columns={"actual": "target_points"})
    lineups["position"] = "LINEUP"
    store = CampaignStore(registry)
    store.check("opening_drive_v3")
    baseline = store.campaign("opening_drive_v3")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v31-", dir=output.parent) as tmp:
        staging = Path(tmp) / output.name
        staging.mkdir()
        player_predictions, player_scores, player_folds = evaluate(players)
        lineup_predictions, lineup_scores, lineup_folds = evaluate(
            lineups, levels=(80,), first_year=2022, min_rows=100
        )
        pd.testing.assert_series_equal(
            player_predictions.prediction.reset_index(drop=True),
            player_predictions.merge(
                players[["row_id", "prediction"]], on="row_id", validate="many_to_one"
            ).prediction_y,
            check_names=False,
        )
        player_predictions.to_parquet(staging / "player_predictions.parquet", index=False)
        lineup_predictions.to_parquet(staging / "lineup_predictions.parquet", index=False)
        player_scores.to_csv(staging / "player_scores.csv", index=False)
        lineup_scores.to_csv(staging / "lineup_scores.csv", index=False)
        write(staging / "fold_calibrators.json", dict(players=player_folds, lineups=lineup_folds))
        calibrators = dict(
            players=calibration.fit(players[players.season.lt(2026)], 2026),
            lineups=calibration.fit(lineups, 2026, policy="position", min_rows=100, levels=(80,)),
        )
        write(staging / "calibrators.json", calibrators)
        sources = staging / "source"
        sources.mkdir()
        for path in sorted(Path(__file__).parent.glob("*.py")):
            shutil.copy2(path, sources / path.name)
        frozen = {}
        for entry in baseline["models"]:
            if entry["horizon"] != "weekly":
                continue
            local = {}
            for name in (
                "mean_model",
                "distribution_model",
                "source_modeling",
                "source_data",
                "run_manifest",
                "lineup_calibration",
                "source_lineup_runtime",
                "source_lineup_research",
            ):
                ref = entry["artifacts"][name]
                source = registry / "opening_drive_v3" / ref["path"]
                if digest(source) != ref["sha256"]:
                    raise ValueError("Frozen baseline artifact changed")
                relative = f"baseline/{entry['position']}/{name}{source.suffix}"
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                local[name] = dict(path=relative, sha256=digest(target))
            relative = f"baseline/{entry['position']}/source_rescore.py"
            shutil.copy2(Path(v3_rescore.__file__), staging / relative)
            local["source_rescore"] = dict(path=relative, sha256=digest(staging / relative))
            frozen[entry["position"]] = local
        if set(frozen) != {"QB", "RB", "WR", "TE"}:
            raise ValueError("Four frozen weekly models required")
        write(staging / "baseline_models.json", frozen)
        report(staging, player_scores, lineup_scores)
        seal(
            staging,
            dict(
                version="3.1",
                campaign_id="opening_drive_v31",
                generated_at=now().isoformat(),
                forecast_season=2026,
                packages={
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "pyarrow", "scipy")
                },
                designated_policy="position_workload",
                status="research_challenger",
                scoring=SCORING,
                original_v3_manifest_sha256=digest(v3_run / "manifest.json"),
                baseline_campaign_manifest_sha256=digest(
                    registry / "opening_drive_v3/manifest.json"
                ),
                inputs={str(selected): digest(selected), str(lineup_path): digest(lineup_path)},
                point_models_refit=False,
                mean_max_difference=0.0,
                original_distribution_quantiles=list(QCOLS),
                evaluation="Reused historical development years; no automatic promotion",
            ),
        )
        publish_directory(staging, output)
    return output


def export_report(run_dir, destination):
    from .artifacts import verify

    verify(run_dir)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "player_scores.csv", "lineup_scores.csv", "manifest.json"):
        shutil.copy2(Path(run_dir) / name, destination / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-run", type=Path, default=V3_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--registry", type=Path, default=ROOT / "data/campaigns")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run(args.v3_run, args.output, args.registry)
    export_report(result, args.report)
    print(result)
