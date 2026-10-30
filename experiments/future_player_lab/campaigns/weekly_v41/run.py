"""Run the complete v4.1 retrain in a fresh directory and judge it against controls.

Stages, all fitted in this run from the frozen source data and accepted snapshot:
1. weekly base mean (v3 method + market lines)            -> weekly.py
2. injury/participation/workload layer (v3.2 method)      -> weekly_v32.run.evaluate
3. weekly distributions (v3.2 empirical errors, v4 scale)  -> weekly_v4.uncertainty
4. horizons/components (v4 method + availability data)     -> horizons.py
Earlier versions' saved predictions are read only as evaluation controls.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..weekly_v31.artifacts import digest, now, seal, write
from ..weekly_v32 import run as v32_run
from ..weekly_v4 import horizons as v4_horizons
from ..weekly_v4 import uncertainty as v4_uncertainty
from . import data, horizons, judging, weekly

RUNS = data.ROOT / "experiments/future_player_lab/runs"
CONTROLS = dict(
    v3=RUNS / "opening_drive_v3_20261011_002",
    v32=RUNS / "opening_drive_v32_20261013_002",
    v4_horizons=RUNS / "opening_drive_v4_horizons_20261027_003",
    v32_bridge=RUNS / "opening_drive_v32_horizons_20261014_002",
)
LINE_NOISE = dict(spread_sd=1.0, total_sd=1.0)


def protocol(snapshot):
    return dict(
        version="opening_drive_v41_1",
        created_at=now().isoformat(),
        lineage_rule="Full retrain from reused methods; no earlier trained model or saved "
        "prediction is an input. Earlier versions appear only as evaluation controls.",
        snapshot=str(snapshot),
        stages={
            "weekly_base": "weekly_v3.modeling estimators and fixed tree settings; candidates "
            f"{list(weekly.CANDIDATES)}; per-position selection on 2018-20 equal-year RMSE",
            "injury_workload": "weekly_v32 modeling/selection (15 methods, prior-two-year "
            "selector) on v4.1 base predictions; early_week and pregame_60m arms",
            "uncertainty": "weekly_v32 mean-bin empirical errors with weekly_v4 prior-year "
            "CRPS scale selection",
            "horizons": "weekly_v4 labels/targets/tree settings; anchor is v4.1 base; added "
            "availability/workload/injury/market features",
        },
        market="nflverse schedule spread_line/total_line (historical closing lines). Early-week "
        "historical rows therefore have an information advantage; a line-noise robustness arm "
        f"perturbs spread and total ({LINE_NOISE}) to approximate early-week line error.",
        judging=judging.__doc__,
        evaluation_years=list(range(2021, 2026)),
        limitations=[
            "2021-25 were used in earlier development; retrospective, not an untouched test",
            "Historical early-week lines are closing lines; prospective captures are the honest test",
            "Historical injury reports are sparse retrospective reconstructions",
        ],
    )


def line_noise_arm(panel, features, choice, years=range(2021, 2026)):
    """Selected learned stage-1 method re-predicted with perturbed market lines."""
    rng = np.random.default_rng(20261027)
    rows = []
    for position, local in panel.groupby("position"):
        method = choice[position]["mean_method"]
        if method not in weekly.CANDIDATES or method == "recent":
            continue
        for year in years:
            train = local[local.season.lt(year) & local.target_points.notna()]
            test = local[local.season.eq(year)].copy()
            model = weekly.fit(train, features, method, year)
            spread = test.market_spread + rng.normal(0, LINE_NOISE["spread_sd"], len(test))
            total = test.market_total + rng.normal(0, LINE_NOISE["total_sd"], len(test))
            test["market_spread"], test["market_total"] = spread, total
            test["market_implied"] = (total + spread) / 2
            test["market_opp_implied"] = (total - spread) / 2
            rows.append(
                pd.DataFrame(
                    dict(
                        row_id=test.row_id.to_numpy(),
                        position=position,
                        season=year,
                        method="stage1_line_noise",
                        prediction=weekly.predict(model, test),
                    )
                )
            )
    return pd.concat(rows, ignore_index=True)


def control_weekly():
    v3 = pd.read_parquet(CONTROLS["v3"] / "selected_predictions.parquet")
    v3 = v3[v3.season.between(2021, 2025)][["player_id", "season", "week", "prediction"]]
    v3["method"], v3["horizon"] = "control_v3", "early_week"
    v32 = pd.read_parquet(CONTROLS["v32"] / "predictions.parquet")
    v32 = v32[v32.method.eq("selected")][["player_id", "season", "week", "horizon", "prediction"]]
    v32["method"] = "control_v32"
    return pd.concat([v3, v32], ignore_index=True)


def judge(stage, frame, base, stage1, line_noise, totals):
    keys = ["player_id", "season", "week"]
    context = frame[frame.horizon.eq("early_week")].set_index("row_id")
    reference = context.recent8_points_mean.fillna(context.prior_points_mean)
    # Weekly comparisons, per timing arm.
    selected = base[base.method.eq("selected")][[*keys, "row_id", "horizon", "prediction", "target_points"]]
    selected = selected.assign(method="v41")
    s1 = stage1.drop(columns="season").merge(context[[*keys, "target_points"]], left_on="row_id", right_index=True)
    weekly_parts = [selected]
    for horizon in data.HORIZONS:
        for name, source in (
            ("v41_base", s1[s1.method.isin(weekly.CANDIDATES)]),
            ("v41_no_market", s1[s1.method.eq("ablation_no_market")]),
            ("v41_line_noise", line_noise.drop(columns="season").merge(context[[*keys, "target_points"]], left_on="row_id", right_index=True)),
        ):
            part = source.copy()
            if name == "v41_base":
                chosen = json.loads((stage / "stage1_selection.json").read_text())
                part = pd.concat(
                    [part[part.position.eq(p) & part.method.eq(c["mean_method"])] for p, c in chosen.items()]
                )
            weekly_parts.append(part.assign(method=name, horizon=horizon)[[*keys, "row_id", "horizon", "prediction", "target_points", "method"]])
    ours = pd.concat(weekly_parts, ignore_index=True)
    ours = ours[ours.season.between(2021, 2025)]
    ours["reference_points"] = ours.row_id.map(reference)
    ours["position"] = ours.row_id.map(context.position)
    labels = ours.drop_duplicates(keys)[[*keys, "target_points", "reference_points", "position"]]
    controls = control_weekly().merge(labels, on=keys, how="inner")
    allrows = pd.concat([ours.drop(columns="row_id"), controls], ignore_index=True)
    comparisons = []
    for horizon, part in allrows.groupby("horizon"):
        pairs = [("v41", "control_v32"), ("v41", "v41_base"), ("v41_base", "v41_no_market"),
                 ("v41_line_noise", "v41_no_market"), ("v41_base", "v41_line_noise")]
        if horizon == "early_week":
            pairs.append(("v41", "control_v3"))
        for a, b in pairs:
            if {a, b} <= set(part.method):
                comparisons.append(judging.paired(part, a, b).assign(horizon=horizon))
    weekly_cmp = pd.concat(comparisons, ignore_index=True)
    floor, _ = judging.oracle_floor(
        context.reset_index().assign(reference_points=reference.to_numpy())
    )
    weekly_cmp = judging.gap_closed(weekly_cmp, floor)
    # Horizon comparisons.
    hkeys = ["player_id", "season", "origin_week"]
    v4 = pd.read_parquet(CONTROLS["v4_horizons"] / "horizon_predictions.parquet")
    v4 = v4[v4.method.isin(["v4"])].assign(method="control_v4")
    bridge = pd.read_parquet(CONTROLS["v32_bridge"] / "predictions.parquet")
    bridge = bridge[bridge.method.eq("v32_selected_decay")].assign(method="control_v32_bridge")
    t = totals[totals.season.between(2021, 2025)]
    lab = t[t.method.eq("v41")][[*hkeys, "target_points", "reference_points", "position"]]
    hcontrols = [
        c[[*hkeys, "prediction"]].assign(method=c.method.iloc[0]).merge(lab, on=hkeys)
        for c in (v4, bridge)
    ]
    horizon_frame = pd.concat([t.drop(columns="row_id"), *hcontrols], ignore_index=True)
    horizon_cmp = pd.concat(
        [
            judging.paired(horizon_frame, a, b, group=("position",))
            for a, b in (("v41", "control_v4"), ("v41", "v41_fixed_origin"), ("v41", "control_v32_bridge"))
        ],
        ignore_index=True,
    )
    by_origin = judging.paired(horizon_frame, "v41", "control_v4", group=("position", "origin_week"))
    return weekly_cmp, floor, horizon_cmp, by_origin


def run(output, snapshot=data.DEFAULT_SNAPSHOT, fresh=False):
    """Checkpointed: each stage reuses its outputs in `output` unless `fresh`."""
    output, snapshot = Path(output), Path(snapshot)
    if fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for path in CONTROLS.values():
        if not (path / "manifest.json").exists():
            raise FileNotFoundError(f"Missing evaluation control {path}")
    stage = output
    for name in ("manifest.json", "manifest.sha256"):
        (stage / name).unlink(missing_ok=True)
    write(stage / "protocol.json", protocol(snapshot))
    shutil.rmtree(stage / "source", ignore_errors=True)
    shutil.copytree(Path(__file__).parent, stage / "source", ignore=shutil.ignore_patterns("__pycache__"))
    log = lambda m: print(m, flush=True)  # noqa: E731

    # Stages 0-1: panel and weekly base.
    if (stage / "panel.parquet").exists() and (stage / "stage1_line_noise.parquet").exists():
        frame = pd.read_parquet(stage / "panel.parquet")
        features = json.loads((stage / "dataset.json").read_text())["features"]
        stage1 = pd.read_parquet(stage / "stage1_predictions.parquet")
        noise = pd.read_parquet(stage / "stage1_line_noise.parquet")
        log("reused panel and stage 1")
    else:
        frame, features, receipt, audit = data.prepare(snapshot)
        write(stage / "dataset.json", {k: v for k, v in receipt.items() if k != "audit"})
        early = frame[frame.horizon.eq("early_week")]
        stage1, final1 = weekly.out_of_time(early, features, log=log)
        choice = weekly.select(stage1, early[["row_id", "target_points"]])
        write(stage / "stage1_selection.json", choice)
        stage1.to_parquet(stage / "stage1_predictions.parquet", index=False)
        joblib.dump(final1, stage / "stage1_models_2026.joblib", compress=3)
        base = weekly.base_predictions(stage1, choice)
        frame["base_prediction"] = frame.row_id.map(base)
        frame = frame[frame.base_prediction.notna()].copy()
        frame.to_parquet(stage / "panel.parquet", index=False)
        noise = line_noise_arm(early, features, choice)
        noise.to_parquet(stage / "stage1_line_noise.parquet", index=False)

    # Stage 2: v3.2 injury/workload methods on the v4.1 base.
    if (stage / "predictions.parquet").exists():
        forecasts = pd.read_parquet(stage / "predictions.parquet")
        log("reused stage 2")
    else:
        shutil.rmtree(stage / "models", ignore_errors=True)
        forecasts, diagnostic, budgets = v32_run.evaluate(frame, features, stage)
        diagnostic.to_parquet(stage / "components.parquet", index=False)
        budgets.to_csv(stage / "redistribution_budgets.csv", index=False)
        point_scores, ranks = v32_run.scores(forecasts)
        point_scores.to_csv(stage / "stage2_scores.csv", index=False)
        ranks.to_csv(stage / "rankings.csv", index=False)
        forecasts.to_parquet(stage / "predictions.parquet", index=False)

    # Stage 3: weekly distributions from v4.1's own out-of-time errors.
    history = forecasts[forecasts.method.eq("selected") & forecasts.target_points.notna()].copy()
    history = history.merge(
        frame[["row_id", "horizon", "is_bye"]].drop_duplicates(["row_id", "horizon"]),
        on=["row_id", "horizon"],
        how="left",
    )
    # Deterministic zeros only where the selected mean is exactly zero.
    history["report_out"] = (history.report_out.eq(1) & history.prediction.eq(0)).astype(float)
    u_scores, _, u_decisions, u_final = v4_uncertainty.evaluate(history)
    u_scores.to_csv(stage / "uncertainty_scores.csv", index=False)
    write(stage / "uncertainty_decisions.json", u_decisions)
    write(stage / "uncertainty.json", u_final)

    # Stage 4: horizons and components.
    if (stage / "horizon_predictions.parquet").exists():
        totals = pd.read_parquet(stage / "horizon_predictions.parquet")
        hpanel = pd.read_parquet(stage / "horizon_panel.parquet")
        log("reused stage 4")
    else:
        keep = list(dict.fromkeys([
            "row_id", "player_id", "player_name", "position", "season", "week", "origin_week",
            "team", "horizon", "base_prediction", *horizons.FEATURES,
        ]))
        hpanel = v4_horizons.training_panel(
            frame[[c for c in keep if c in frame]],
            pd.read_parquet(data.v3_data.HISTORY / "box.parquet"),
            pd.read_parquet(data.v3_data.HISTORY / "schedule.parquet"),
        )
        hpanel.to_parquet(stage / "horizon_panel.parquet", index=False)
        selected = forecasts[forecasts.method.eq("selected") & forecasts.horizon.eq("early_week")]
        totals, components = horizons.evaluate(hpanel, selected.set_index("row_id").prediction, log=log)
        components.to_parquet(stage / "component_predictions.parquet", index=False)
        joblib.dump(horizons.fit_models(hpanel, 2026), stage / "horizon_models_2026.joblib", compress=3)
        totals.to_parquet(stage / "horizon_predictions.parquet", index=False)
    v4_horizons.metric_rows(totals[totals.season.ge(2021)]).to_csv(stage / "horizon_scores.csv", index=False)
    total_history = totals[totals.method.eq("v41") & totals.season.ge(2021)]
    write(stage / "total_uncertainty.json", v4_uncertainty.fit_total(total_history, 2026))
    v4_uncertainty.evaluate_totals(totals[totals.season.ge(2020)]).to_csv(
        stage / "total_uncertainty_scores.csv", index=False
    )

    # Judging.
    weekly_cmp, floor, horizon_cmp, by_origin = judge(stage, frame, forecasts, stage1, noise, totals)
    weekly_cmp.to_csv(stage / "judging_weekly.csv", index=False)
    floor.to_csv(stage / "judging_noise_floor.csv", index=False)
    horizon_cmp.to_csv(stage / "judging_horizons.csv", index=False)
    by_origin.to_csv(stage / "judging_horizons_by_origin.csv", index=False)

    seal(
        stage,
        dict(
            kind="opening_drive_v41_research",
            completed_at=now().isoformat(),
            snapshot_sha256=digest(snapshot),
            controls={k: digest(v / "manifest.json") for k, v in CONTROLS.items()},
            packages={
                p: importlib.metadata.version(p)
                for p in ("numpy", "pandas", "lightgbm", "xgboost", "scikit-learn", "joblib")
            },
        ),
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=data.DEFAULT_SNAPSHOT)
    parser.add_argument("--fresh", action="store_true", help="Discard checkpoints first")
    args = parser.parse_args()
    print(run(args.output, args.snapshot, args.fresh))
