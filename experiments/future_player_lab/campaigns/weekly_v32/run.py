"""Run the v3.2 injury/workload investigation against the unchanged v3 baseline."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..interval_research.run import inventory
from ..weekly_v31.artifacts import digest, now, publish_directory, seal, write
from ..weekly_v31.run import DEFAULT_LOG
from ..weekly_v31.run import DEFAULT_RUN as V31_RUN
from . import data, modeling

DEFAULT_RUN = data.ROOT / "experiments/future_player_lab/runs/opening_drive_v32_20261013_002"
DEFAULT_REPORT = data.DATA / "outputs/opening_drive_v32_2026"


def protocol():
    return dict(
        version="v3.2_injury_workload_2",
        revision="002: missing-snap label support uses the frozen ID crosswalk, "
        "not membership in any season's snap observations. Initial 001 retained.",
        created_at=now().isoformat(),
        baseline="v3",
        v31="Paused; all sealed artifacts retained; no v3.1 fitted model reused",
        methods=list(modeling.METHODS),
        selector="Earlier two years, equal-year RMSE, per position",
        fit="Earlier five years; 2015+ base predictions; validation 2019–20; evaluation 2021–25",
        horizons=dict(
            early_week="Original v3 cutoff",
            pregame_60m="One hour before individual game kickoff; later information than v3",
        ),
        report_matching="Exact player/team/season/week; timestamp <= cutoff; max age 7 days",
        injury_history="Source modification timestamps; not original publication vintages",
        current_capture="Usable from collection time only; no retrospective backfill",
        participation="Positive offensive snaps or opportunity/points; unknown when unsupported; "
        "not official active status",
        workload="Expected attempts/carries/targets conditional on offensive participation",
        point_conversion="Linear conditional points at predicted active workloads, "
        "multiplied once by P(participation)",
        teammate="Other players' pregame reports/usage; 50%/100% budget shortfall transfer",
        budgets="V3 eligible team pool; baseline no-injury expected workload; recipients P>=0.5",
        out_rule="Fresh same-game Out forces zero; v3_out_rule isolates this constraint",
        source_latency="Strictly earlier scheduled weeks; assumes next-week box/snap availability; "
        "not archived original publication vintages",
        model_parameters=dict(
            trees=140, learning_rate=0.04, leaves=10, depth=4, min_rows=80, l2=25
        ),
        metrics=[
            "point RMSE/MAE/bias",
            "weekly top-k points",
            "participation Brier/log loss",
            "conditional/unconditional workload RMSE",
            "budget conservation",
        ],
        bootstrap="5000 season resamples, five years, descriptive, no multiplicity correction",
        limitations=[
            "Reused historical development years",
            "Incomplete historical practice trajectories and official inactive lists",
            "Later pregame arm has an information advantage over original v3",
            "Provider injury tags do not prove game-day activity",
            "K/DST, injury diagnoses, API promotion and full joint lineup simulation excluded",
        ],
        references=[
            "https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html",
            "https://nflreadr.nflverse.com/articles/dictionary_injuries.html",
        ],
    )


def point_frame(test, values, method, horizon):
    columns = [
        "row_id",
        "player_id",
        "player_name",
        "season",
        "week",
        "position",
        "team",
        "target_points",
        "base_prediction",
        "player_history_count",
        "report_known",
        "report_out",
        "report_questionable",
        "room_out_usage",
    ]
    result = test[columns].copy()
    result["prediction"], result["method"], result["chosen_method"], result["horizon"] = (
        values,
        method,
        method,
        horizon,
    )
    return result


def evaluate(frame, features, directory):
    forecasts, decisions, diagnostics, budgets, receipts = [], [], [], [], []
    (directory / "models").mkdir()
    for horizon, local in frame[frame.season.le(2025)].groupby("horizon"):
        previous = []
        for year in range(2019, 2026):
            pieces, bundles, predicted, info = [], {}, [], []
            for position, part in local.groupby("position"):
                train, test = part[part.season.lt(year)], part[part.season.eq(year)].copy()
                bundle = modeling.fit(train, year, features)
                bundles[position] = bundle
                safe = test.drop(
                    columns=[
                        c
                        for c in test
                        if c.startswith("label_")
                        or c in ("target_points", "appeared", "actual_team")
                    ]
                )
                values, detail = modeling.predict(bundle, safe)
                path = directory / "models" / f"{horizon}_{position}_{year}.joblib"
                joblib.dump(bundle, path, compress=3)
                if year == 2025:
                    replay, _ = modeling.predict(joblib.load(path), safe)
                    for name in values:
                        np.testing.assert_array_equal(replay[name], values[name])
                receipts.append(
                    dict(
                        horizon=horizon,
                        position=position,
                        year=year,
                        n=bundle["n"],
                        max_training_year=bundle["max_year"],
                        features=bundle["features"],
                    )
                )
                pieces.append(test)
                predicted.append(pd.DataFrame(values, index=test.index))
                info.append(detail)
            test = pd.concat(pieces, ignore_index=True)
            values = pd.concat(predicted, ignore_index=True)
            detail = {
                stage: {
                    key: np.concatenate([x[stage][key] for x in info])
                    for key in ("probability", "active_workload")
                }
                for stage in modeling.STAGES
            }
            p = detail["team"]["probability"]
            before = (
                detail["no_injury"]["active_workload"] * detail["no_injury"]["probability"][:, None]
            )
            after = detail["team"]["active_workload"] * p[:, None]
            for name, fraction in (("redistribution_half", 0.5), ("redistribution_full", 1.0)):
                adjusted, transfers = modeling.redistribute(test, before, after, p, fraction)
                active_workload = np.divide(
                    adjusted, p[:, None], out=np.zeros_like(adjusted), where=p[:, None] > 0
                )
                points = np.zeros(len(test))
                for position, indexes in test.groupby("position").groups.items():
                    idx = np.asarray(indexes)
                    points[idx] = modeling.points_from_workload(
                        bundles[position], test.loc[idx], p[idx], active_workload[idx]
                    )
                values[name] = points
                if year >= 2021:
                    budgets.extend(dict(horizon=horizon, method=name, **r) for r in transfers)
                detail[name] = dict(probability=p, active_workload=active_workload)
            if set(values) != set(modeling.METHODS) or not np.isfinite(values.to_numpy()).all():
                raise ValueError("Incomplete or nonfinite candidate predictions")
            candidates = pd.concat(
                [
                    point_frame(test, values[name].to_numpy(), name, horizon)
                    for name in modeling.METHODS
                ],
                ignore_index=True,
            )
            if year >= 2021:
                history = pd.concat(previous, ignore_index=True)
                forecasts.append(candidates)
                for position, part in history.groupby("position"):
                    chosen = modeling.choose(part, year)
                    decisions.append(
                        dict(
                            horizon=horizon,
                            position=position,
                            year=year,
                            method=chosen,
                            max_validation_year=year - 1,
                        )
                    )
                    f = candidates[
                        candidates.position.eq(position) & candidates.method.eq(chosen)
                    ].copy()
                    f["method"] = "selected"
                    forecasts.append(f)
                for stage, d in detail.items():
                    f = test[
                        [
                            "row_id",
                            "position",
                            "season",
                            "week",
                            "label_played",
                            *[f"label_{c}" for c in data.OPPORTUNITIES],
                        ]
                    ].copy()
                    f["probability"], f["stage"], f["horizon"] = d["probability"], stage, horizon
                    for j, c in enumerate(data.OPPORTUNITIES):
                        f[f"active_{c}"] = d["active_workload"][:, j]
                        f[f"expected_{c}"] = d["probability"] * d["active_workload"][:, j]
                    diagnostics.append(f)
            previous.append(candidates)
            print(
                f"{horizon} {year}: {len(test):,} rows × {len(values.columns)} methods", flush=True
            )
        history = pd.concat(previous, ignore_index=True)
        for position, part in local.groupby("position"):
            bundle = modeling.fit(part, 2026, features)
            joblib.dump(
                bundle, directory / "models" / f"{horizon}_{position}_2026.joblib", compress=3
            )
            decisions.append(
                dict(
                    horizon=horizon,
                    position=position,
                    year=2026,
                    method=modeling.choose(history[history.position.eq(position)], 2026),
                    max_validation_year=2025,
                )
            )
    write(directory / "fit_receipts.json", receipts)
    write(directory / "selection.json", decisions)
    return (
        pd.concat(forecasts, ignore_index=True),
        pd.concat(diagnostics, ignore_index=True),
        pd.DataFrame(budgets),
    )


def cohorts(frame):
    yield "all", frame
    yield "projected_5_plus", frame[frame.base_prediction.ge(5)]
    yield "current_report_known", frame[frame.report_known.eq(1)]
    yield "reported_out", frame[frame.report_out.eq(1)]
    yield "questionable", frame[frame.report_questionable.eq(1)]
    yield "teammate_out_workload", frame[frame.room_out_usage.gt(0)]
    yield "history_under_8", frame[frame.player_history_count.lt(8)]


def scores(frame):
    rows, rankings = [], []
    for (horizon, position, method), group in frame.groupby(["horizon", "position", "method"]):
        for cohort, part in cohorts(group):
            part = part[part.target_points.notna()]
            if part.empty:
                continue
            for year, f in [("all", part), *list(part.groupby("season"))]:
                error = f.prediction - f.target_points
                rows.append(
                    dict(
                        horizon=horizon,
                        position=position,
                        method=method,
                        cohort=cohort,
                        season=str(year),
                        n=len(f),
                        rmse=float(np.sqrt(np.mean(error**2))),
                        mae=float(np.mean(np.abs(error))),
                        bias=float(error.mean()),
                    )
                )
        k = 10 if position in ("QB", "TE") else 20
        for (year, week), f in group.groupby(["season", "week"]):
            chosen = f.sort_values(["prediction", "player_id"], ascending=[False, True]).head(k)
            rankings.append(
                dict(
                    horizon=horizon,
                    position=position,
                    method=method,
                    season=int(year),
                    week=int(week),
                    k=k,
                    selected_points=float(chosen.target_points.sum()),
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(rankings)


def diagnostic_scores(frame):
    rows = []
    for (horizon, position, stage), group in frame.groupby(["horizon", "position", "stage"]):
        for year, f in [("all", group), *list(group.groupby("season"))]:
            f = f[f.label_played.notna()]
            p = np.clip(f.probability.to_numpy(), 1e-8, 1 - 1e-8)
            y = f.label_played.to_numpy()
            row = dict(
                horizon=horizon,
                position=position,
                stage=stage,
                season=str(year),
                n=len(f),
                brier=float(np.mean((p - y) ** 2)),
                log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                expected_participation=float(p.mean()),
                observed_participation=float(y.mean()),
            )
            for c in data.OPPORTUNITIES:
                for kind, subset in (("active", f[f.label_played.eq(1)]), ("expected", f)):
                    known = subset[f"label_{c}"].notna()
                    e = subset.loc[known, f"{kind}_{c}"] - subset.loc[known, f"label_{c}"]
                    row[f"{kind}_{c}_rmse"] = float(np.sqrt(np.mean(e**2))) if len(e) else None
            rows.append(row)
    return pd.DataFrame(rows)


def comparisons(scores):
    annual = scores[scores.season.ne("all")]
    rows = []
    for (horizon, position, method, cohort), part in annual.groupby(
        ["horizon", "position", "method", "cohort"]
    ):
        if method == "v3":
            continue
        base = (
            annual[
                annual.horizon.eq(horizon)
                & annual.position.eq(position)
                & annual.cohort.eq(cohort)
                & annual.method.eq("v3")
            ]
            .set_index("season")
            .loc[part.season]
        )
        d = part.rmse.to_numpy() - base.rmse.to_numpy()
        boot = np.random.default_rng(20261013).choice(d, (5000, len(d))).mean(axis=1)
        rows.append(
            dict(
                horizon=horizon,
                position=position,
                method=method,
                cohort=cohort,
                equal_year_delta_rmse=float(d.mean()),
                seasons_improved=int((d < 0).sum()),
                ci95_low=float(np.quantile(boot, 0.025)),
                ci95_high=float(np.quantile(boot, 0.975)),
            )
        )
    return pd.DataFrame(rows)


def run(output=DEFAULT_RUN, report_dir=DEFAULT_REPORT):
    from .report import render

    output, report_dir = Path(output), Path(report_dir)
    if output.exists() or report_dir.exists():
        raise FileExistsError("Use fresh experiment and report directories")
    roots = [data.V3_RUN, V31_RUN, DEFAULT_LOG, data.DATA / "campaigns/opening_drive_v3"]
    roots += list(
        (data.ROOT / "experiments/future_player_lab/runs").glob("opening_drive_v31_points_*")
    )
    preserved = inventory(roots)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v32-") as tmp:
        staging = Path(tmp) / "run"
        staging.mkdir()
        write(staging / "protocol.json", protocol())
        shutil.copytree(
            Path(__file__).parent, staging / "source", ignore=shutil.ignore_patterns("__pycache__")
        )
        frame, features, reports, audit, sources = data.prepare()
        frame.to_parquet(staging / "panel.parquet", index=False)
        reports.to_parquet(staging / "dated_reports.parquet", index=False)
        audit.to_csv(staging / "source_timestamps.csv", index=False)
        write(staging / "source_audit.json", audit.attrs)
        write(staging / "source_hashes.json", sources)
        coverage = (
            frame.groupby(["horizon", "season"])
            .agg(
                rows=("row_id", "size"),
                report_known=("report_known", "sum"),
                practice_known=("practice_known", "sum"),
                participation_label_known=("label_played", "count"),
                snap_history_known=("prior4_snaps", "count"),
            )
            .reset_index()
        )
        coverage.to_csv(staging / "coverage.csv", index=False)
        forecasts, diagnostic, budgets = evaluate(frame, features, staging)
        forecasts.to_parquet(staging / "predictions.parquet", index=False)
        diagnostic.to_parquet(staging / "components.parquet", index=False)
        budgets.to_csv(staging / "redistribution_budgets.csv", index=False)
        point_scores, ranks = scores(forecasts)
        point_scores.to_csv(staging / "scores.csv", index=False)
        ranks.to_csv(staging / "rankings.csv", index=False)
        comparisons(point_scores).to_csv(staging / "comparisons.csv", index=False)
        diagnostic_scores(diagnostic).to_csv(staging / "component_scores.csv", index=False)
        example = forecasts[forecasts.method.eq("selected") & forecasts.room_out_usage.gt(0)].copy()
        example["point_change"] = example.prediction - example.base_prediction
        example.assign(magnitude=example.point_change.abs()).nlargest(100, "magnitude").drop(
            columns="magnitude"
        ).to_csv(staging / "examples.csv", index=False)
        render(staging)
        if inventory(roots) != preserved:
            raise ValueError("Protected v3/v3.1 artifacts changed")
        if any(digest(Path(path)) != expected for path, expected in sources.items()):
            raise ValueError("Research inputs changed during fitting")
        write(
            staging / "preservation.json",
            dict(files=len(preserved), unchanged=True, sha256=preserved),
        )
        seal(
            staging,
            dict(
                status="completed_v3.2_research",
                completed_at=now().isoformat(),
                baseline="v3",
                deployed=False,
                packages={
                    p: importlib.metadata.version(p)
                    for p in ("numpy", "pandas", "lightgbm", "scikit-learn", "joblib")
                },
            ),
        )
        publish_directory(staging, output)
    report_dir.mkdir(parents=True)
    for path in [
        output / "index.html",
        *output.glob("*.csv"),
        *[
            output / name
            for name in ("protocol.json", "selection.json", "source_audit.json", "manifest.json")
        ],
    ]:
        shutil.copy2(path, report_dir / path.name)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(run(args.output, args.report))
