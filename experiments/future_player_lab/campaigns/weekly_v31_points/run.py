"""Sealed v3.1 point-forecast experiment; no serving or prospective-log mutations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..interval_research import methods as intervals
from ..interval_research.run import inventory
from ..weekly_v3.data import ROOT
from ..weekly_v3.run import DEFAULT_OUTPUT as V3_RUN
from ..weekly_v31.artifacts import digest, now, publish_directory, seal, verify, write
from ..weekly_v31.run import DEFAULT_LOG
from ..weekly_v31.run import DEFAULT_RUN as V31_RUN
from . import methods

DEFAULT_RUN = ROOT / "experiments/future_player_lab/runs/opening_drive_v31_points_20261013_002"
DEFAULT_REPORT = ROOT / "data/outputs/opening_drive_v31_points_2026"


def protocol():
    return dict(
        version="v3.1_point_feedback_research_2",
        revision="Initial 001 retained; 002 fixes cancellation noise in zero tail shifts and "
        "adds basic half-strength residual/mixture controls after inspecting 001. "
        "Selectors keep the original objective and guard; candidate set expands by two.",
        created_at=now().isoformat(),
        question="Does feeding past forecast errors and uncertainty into models improve points?",
        fixed_candidates=list(methods.FIXED),
        selectors=list(methods.SELECTORS),
        designated_policy="select_guarded",
        evaluation_years=list(range(2021, 2026)),
        training="Prior five available seasons, 2018 onward; per position; frozen pregame inputs",
        validation="2019 onward; selectors use only two earlier validation years",
        objective="Select minimum equal-year RMSE; MAE, bias, rankings and coverage are secondary",
        guard="Prior RMSE at base-projected >=5 and QB 1–5 cannot exceed v3 by over 1%; n>=100",
        tree=dict(trees=100, learning_rate=0.04, leaves=7, depth=3, min_rows=100, l2=30),
        blend_error_trees=60,
        ridge_penalty=1000,
        priors=dict(group=[24, 100, 400], player=[8, 24, 64]),
        mixture="P(box-score appearance) times E(points | appearance); "
        "zero and negative active scores retained",
        mixture_limit="Appearance is a box-score proxy, not official active status or clearance",
        ablations=["ridge", "residual", "direct", "mixture", "blend_conditional"],
        uncertainty_inputs=list(methods.UNCERTAINTY),
        blend_members=list(methods.MEMBERS),
        blend_controls="Equal, inverse historical MSE, covariance-aware convex; "
        "conditional MSE weights half-shrunk to convex",
        intervals="Changed means use same-method earlier out-of-time residual quantiles; "
        "fixed base workload bins; n>=100",
        interval_control="v3_residual_intervals: v3 mean with matched residual interval recipe",
        ranking="Weekly top 10 QB/TE and top 20 RB/WR realized points on identical universes; "
        "not actual roster start/sit",
        uncertainty="5000 paired season bootstrap draws; descriptive, five seasons, "
        "no multiple-test adjustment",
        limitations=[
            "Reused development years; original v3 policy selected on 2018–2020",
            "No assertion that every possible estimator or hyperparameter has been exhausted",
            "Frozen v3 snapshot; final fits are research artifacts, not current forecasts",
            "No new full-distribution or joint-lineup simulator; K/DST excluded",
            "Chronological selectors; retrospective best individual methods are exploratory",
        ],
        references=[
            "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html",
            "https://lightgbm.readthedocs.io/en/latest/Parameters.html",
        ],
    )


def tail_shift(old_lower, old_upper, new_lower, new_upper):
    # Subtract endpoints first: a no-op must be EXACTLY zero, not cancellation noise
    # that a linear model's feature scaler can amplify into enormous predictions.
    return ((new_lower - old_lower) + (new_upper - old_upper)) / 2


def prepare():
    manifest = json.loads((V3_RUN / "manifest.json").read_text())
    paths = [
        "panel.parquet",
        "selected_predictions.parquet",
        "mean_predictions.parquet",
        "dataset.json",
    ]
    for name in paths:
        if digest(V3_RUN / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"V3 input changed: {name}")
    panel = pd.read_parquet(V3_RUN / "panel.parquet")
    selected = pd.read_parquet(V3_RUN / "selected_predictions.parquet")
    selected = selected[selected.season.le(2025)].copy()
    if not selected.calibration_max_year.lt(selected.season).all():
        raise ValueError("Base uncertainty is not out of time")
    features = json.loads((V3_RUN / "dataset.json").read_text())["features"]
    if set(features) & methods.FORBIDDEN:
        raise ValueError("Invalid raw feature allowlist")
    if selected.row_id.duplicated().any():
        raise ValueError("Duplicate base forecasts")
    raw = panel.loc[selected.row_id].copy()
    for key in ("season", "week", "player_id", "position", "target_points"):
        np.testing.assert_array_equal(raw[key].to_numpy(), selected[key].to_numpy())
    for key in [*features, "appeared", "label_observed", "scheduled_game"]:
        if key not in selected:
            selected[key] = raw[key].to_numpy()
    if not selected.scheduled_game.eq(1).all():
        raise ValueError("Study must use scheduled games only")
    mean = pd.read_parquet(V3_RUN / "mean_predictions.parquet")
    mean = mean[mean.mean_method.isin(methods.MEMBERS) & mean.row_id.isin(selected.row_id)]
    wide = mean.pivot(index="row_id", columns="mean_method", values="prediction")
    for member in methods.MEMBERS:
        selected[f"member_{member}"] = wide.loc[selected.row_id, member].to_numpy()
    annotated, receipts = [], []
    for year, current in selected.groupby("season", sort=True):
        past = selected[selected.season.lt(year)]
        v31_fit = intervals.fit(past, int(year), "v31")
        fine_fit = intervals.fit(past, int(year), "tail_scaled_fine")
        adjusted = intervals.apply(current, v31_fit)
        fine = intervals.apply(current, fine_fit)
        current = current.copy()
        for level in (80, 90):
            for side in ("lower", "upper"):
                current[f"v31_{side}_{level}"] = adjusted[f"{side}_{level}"].to_numpy()
        current["width80"] = current.q90 - current.q10
        current["width90"] = current.q95 - current.q05
        current["median_offset"] = current.q50 - current.prediction
        current["skew80"] = current.q90 + current.q10 - 2 * current.prediction
        current["pad80"] = adjusted.upper_80.to_numpy() - current.q90
        current["pad90"] = adjusted.upper_90.to_numpy() - current.q95
        current["tail_shift80"] = tail_shift(
            current.q10.to_numpy(),
            current.q90.to_numpy(),
            fine.lower_80.to_numpy(),
            fine.upper_80.to_numpy(),
        )
        current["tail_width80"] = fine.upper_80.to_numpy() - fine.lower_80.to_numpy()
        annotated.append(current)
        receipts.append(dict(year=int(year), v31=v31_fit, fine=fine_fit))
    result = pd.concat(annotated, ignore_index=True).rename(
        columns={"prediction": "base_prediction"}
    )
    locked = pd.read_parquet(V31_RUN / "player_predictions.parquet")
    locked = locked[locked.interval_method.eq("position_workload")].set_index("row_id")
    overlap = result[result.row_id.isin(locked.index)].set_index("row_id")
    for level in (80, 90):
        for side in ("lower", "upper"):
            np.testing.assert_array_equal(
                overlap[f"v31_{side}_{level}"], locked.loc[overlap.index, f"{side}_{level}"]
            )
    return result, features, receipts


def forecast_frame(test, values, method):
    keys = [
        "row_id",
        "player_id",
        "player_name",
        "position",
        "season",
        "week",
        "team",
        "target_points",
        "base_prediction",
        "player_history_count",
        "appeared",
    ]
    out = test[keys].copy()
    out["prediction"], out["method"], out["chosen_method"] = values, method, method
    for level, pair in ((80, ("q10", "q90")), (90, ("q05", "q95"))):
        for side, q in zip(("lower", "upper"), pair, strict=True):
            out[f"{side}_{level}"] = test[q if method == "v3" else f"v31_{side}_{level}"].to_numpy()
    return out


def evaluate(frame, features, directory):
    outputs, receipts, selections, mixture = [], [], [], []
    (directory / "models").mkdir()
    replay_count = 0
    for position, local in frame.groupby("position", sort=True):
        previous = []
        for year in range(2019, 2026):
            train, test = local[local.season.lt(year)], local[local.season.eq(year)]
            bundle = methods.fit(train, year, features)
            values, diagnostic = methods.predict(
                bundle, test.drop(columns=["target_points", "appeared", "label_observed"])
            )
            path = directory / "models" / f"{position}_{year}.joblib"
            joblib.dump(bundle, path, compress=3)
            receipt = {
                k: bundle[k]
                for k in ("year", "position", "min_year", "max_year", "n", "basic", "full")
            }
            receipt.update(
                global_weights=bundle["global_weights"].tolist(),
                convex_weights=bundle["convex_weights"].tolist(),
            )
            receipts.append(receipt)
            if year == 2025:
                loaded, _ = methods.predict(
                    joblib.load(path),
                    test.drop(columns=["target_points", "appeared", "label_observed"]),
                )
                for method in methods.FIXED:
                    np.testing.assert_array_equal(values[method], loaded[method])
                    replay_count += 1
            candidates = []
            history = pd.concat(previous, ignore_index=True) if previous else pd.DataFrame()
            for method, prediction in values.items():
                current = forecast_frame(test, prediction, method)
                if year >= 2021 and method not in ("v3", "v31"):
                    own_history = history[history.method.eq(method)].reset_index(drop=True)
                    current = methods.residual_intervals(own_history, current, year)
                candidates.append(current)
            candidates = pd.concat(candidates, ignore_index=True)
            if year >= 2021:
                outputs.append(candidates)
                for guarded in (False, True):
                    policy = methods.choose(history, year, guarded=guarded)
                    selections.append(dict(position=position, **policy))
                    chosen = candidates[candidates.method.eq(policy["method"])].copy()
                    chosen["method"] = "select_guarded" if guarded else "select_rmse"
                    outputs.append(chosen)
                diagnostics = test[
                    ["row_id", "position", "season", "week", "appeared", "target_points"]
                ].copy()
                for key, value in diagnostic.items():
                    diagnostics[key] = value
                mixture.append(diagnostics)
            previous.append(candidates)
            print(
                f"{position} {year}: {len(test):,} forecasts, {len(values)} fixed methods",
                flush=True,
            )
        final = methods.fit(local, 2026, features)
        joblib.dump(final, directory / "models" / f"{position}_2026.joblib", compress=3)
        receipts.append(
            {
                k: final[k]
                for k in ("year", "position", "min_year", "max_year", "n", "basic", "full")
            }
        )
        history = pd.concat(previous, ignore_index=True)
        for guarded in (False, True):
            selections.append(
                dict(position=position, **methods.choose(history, 2026, guarded=guarded))
            )
    prediction = pd.concat(outputs, ignore_index=True)
    if prediction.duplicated(["row_id", "method"]).any():
        raise ValueError("Duplicate method forecast identities")
    baseline = prediction[prediction.method.eq("v31")]
    for method, part in prediction.groupby("method"):
        if set(part.row_id) != set(baseline.row_id):
            raise ValueError(f"Unmatched evaluation rows: {method}")
    write(directory / "fit_receipts.json", receipts)
    write(directory / "selection.json", selections)
    write(directory / "replay.json", dict(status="passed", exact_method_replays=replay_count))
    return prediction, pd.concat(mixture, ignore_index=True)


def cohorts(frame):
    yield "all", frame
    yield "projected_5_plus", frame[frame.base_prediction.ge(5)]
    for name, part in frame.groupby(
        frame.base_prediction.map(lambda x: methods.group_name(x, True))
    ):
        yield str(name), part
    yield "history_under_8", frame[frame.player_history_count.lt(8)]
    yield "history_32_plus", frame[frame.player_history_count.ge(32)]
    # Outcome-conditioned diagnostics are never training cohorts or selection guards.
    yield "observed_appearance", frame[frame.appeared.astype(bool)]
    yield "observed_nonappearance", frame[~frame.appeared.astype(bool)]


def metric_values(frame):
    e = frame.prediction.to_numpy() - frame.target_points.to_numpy()
    row = dict(
        n=len(frame),
        rmse=float(np.sqrt(np.mean(e**2))),
        mae=float(np.mean(np.abs(e))),
        bias=float(e.mean()),
        error_variance=float(e.var()),
        mse=float(np.mean(e**2)),
    )
    for level in (80, 90):
        values = intervals.components(frame, level)
        row.update(
            {
                f"{key}_{level}": float(values[key].mean())
                for key in ("width", "interval_score", "covered")
            }
        )
    return row


def summarize(frame):
    rows, paired, ranking = [], [], []
    for (position, method), group in frame.groupby(["position", "method"]):
        for cohort, part in cohorts(group):
            part = part[part.target_points.notna()]
            if part.empty:
                continue
            for season, subset in [("all", part), *list(part.groupby("season"))]:
                rows.append(
                    dict(
                        position=position,
                        method=method,
                        cohort=cohort,
                        season=str(season),
                        **metric_values(subset),
                    )
                )
        k = 10 if position in ("QB", "TE") else 20
        for (season, week), part in group.groupby(["season", "week"]):
            part = part[part.target_points.notna()]
            if len(part) < k:
                continue
            ordered = part.sort_values(["prediction", "player_id"], ascending=[False, True])
            actual = float(ordered.head(k).target_points.sum())
            oracle = float(part.nlargest(k, "target_points").target_points.sum())
            ranking.append(
                dict(
                    position=position,
                    method=method,
                    season=int(season),
                    week=int(week),
                    k=k,
                    selected_points=actual,
                    oracle_points=oracle,
                    regret=oracle - actual,
                    spearman=float(part.prediction.corr(part.target_points, method="spearman")),
                )
            )
    scores = pd.DataFrame(rows)
    season_scores = scores[scores.season.ne("all")]
    for (position, method, cohort), part in season_scores.groupby(["position", "method", "cohort"]):
        if method == "v31":
            continue
        base = (
            season_scores[
                season_scores.position.eq(position)
                & season_scores.method.eq("v31")
                & season_scores.cohort.eq(cohort)
            ]
            .set_index("season")
            .loc[part.season]
        )
        differences = part.rmse.to_numpy() - base.rmse.to_numpy()
        rng = np.random.default_rng(20261013)
        boot = rng.choice(differences, size=(5000, len(differences))).mean(axis=1)
        paired.append(
            dict(
                position=position,
                method=method,
                cohort=cohort,
                equal_year_delta_rmse=float(differences.mean()),
                seasons=len(part),
                seasons_improved=int((differences < 0).sum()),
                ci95_low=float(np.quantile(boot, 0.025)),
                ci95_high=float(np.quantile(boot, 0.975)),
            )
        )
    # Matched ablations isolate added uncertainty from a different model architecture.
    ablations = []
    pairs = [
        (family, f"{family}_basic", f"{family}_uncertainty")
        for family in ("ridge", "residual", "direct", "mixture", "blend_conditional")
    ] + [
        (f"{family}_half", f"{family}_basic_half", f"{family}_uncertainty_half")
        for family in ("residual", "mixture")
    ]
    for family, basic, uncertainty in pairs:
        for (position, cohort), group in scores[scores.season.eq("all")].groupby(
            ["position", "cohort"]
        ):
            a = group[group.method.eq(basic)].iloc[0]
            b = group[group.method.eq(uncertainty)].iloc[0]
            ablations.append(
                dict(
                    family=family,
                    position=position,
                    cohort=cohort,
                    rmse_basic=float(a.rmse),
                    rmse_uncertainty=float(b.rmse),
                    delta_rmse=float(b.rmse - a.rmse),
                    delta_mae=float(b.mae - a.mae),
                )
            )
    return scores, pd.DataFrame(paired), pd.DataFrame(ranking), pd.DataFrame(ablations)


def mixture_scores(frame):
    rows = []
    for (position, season), group in frame.groupby(["position", "season"]):
        y = group.appeared.astype(float).to_numpy()
        for label in ("basic", "uncertainty"):
            p = np.clip(group[f"p_appearance_{label}"].to_numpy(), 1e-8, 1 - 1e-8)
            rows.append(
                dict(
                    position=position,
                    season=int(season),
                    method=f"mixture_{label}",
                    n=len(group),
                    observed_appearance=float(y.mean()),
                    predicted_appearance=float(p.mean()),
                    brier=float(np.mean((p - y) ** 2)),
                    log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                )
            )
    return pd.DataFrame(rows)


def run(output=DEFAULT_RUN, report_dir=DEFAULT_REPORT):
    from .report import render

    output, report_dir = Path(output), Path(report_dir)
    if output.exists() or report_dir.exists():
        raise FileExistsError("Use fresh output and report directories")
    roots = [V3_RUN, V31_RUN, DEFAULT_LOG, ROOT / "data/campaigns/opening_drive_v3"]
    preserved = inventory(roots)
    verify(V31_RUN)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v31-points-", dir=output.parent) as tmp:
        staging = Path(tmp) / "run"
        staging.mkdir()
        write(staging / "protocol.json", protocol())
        shutil.copytree(
            Path(__file__).parent, staging / "source", ignore=shutil.ignore_patterns("__pycache__")
        )
        # Snapshot imported local implementation dependencies for audit/reproduction.
        for package in ("interval_research", "weekly_v31"):
            shutil.copytree(
                Path(__file__).parent.parent / package,
                staging / "dependencies" / package,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        frame, features, calibration = prepare()
        frame.to_parquet(staging / "inputs.parquet", index=False)
        write(staging / "uncertainty_features.json", calibration)
        predictions, mixture = evaluate(frame, features, staging)
        predictions.to_parquet(staging / "predictions.parquet", index=False)
        mixture.to_parquet(staging / "mixture_predictions.parquet", index=False)
        tables = summarize(predictions)
        for name, table in zip(
            ("scores", "comparisons", "rankings", "ablations"), tables, strict=True
        ):
            table.to_csv(staging / f"{name}.csv", index=False)
        mixture_scores(mixture).to_csv(staging / "mixture_scores.csv", index=False)
        # Changed forecasts are descriptive examples, never used to select rules.
        examples = predictions[predictions.method.eq("select_guarded")].copy()
        examples["point_change"] = examples.prediction - examples.base_prediction
        examples["absolute_change"] = examples.point_change.abs()
        examples.nlargest(100, "absolute_change").drop(columns="absolute_change").to_csv(
            staging / "examples.csv", index=False
        )
        render(staging)
        if preserved != inventory(roots):
            raise ValueError("Protected artifacts changed during experiment")
        write(
            staging / "preservation.json",
            dict(files=len(preserved), unchanged=True, sha256=preserved),
        )
        seal(
            staging,
            dict(
                status="completed_v3.1_point_research",
                completed_at=now().isoformat(),
                point_predictions_changed=True,
                deployed=False,
                rows=int(predictions.row_id.nunique()),
                methods=int(predictions.method.nunique()),
                v3_manifest_sha256=digest(V3_RUN / "manifest.json"),
                v31_manifest_sha256=digest(V31_RUN / "manifest.json"),
                packages={
                    p: importlib.metadata.version(p)
                    for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy", "joblib")
                },
            ),
        )
        publish_directory(staging, output)
    export(output, report_dir)
    return output


def export(output, report_dir):
    report_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "index.html",
        "scores.csv",
        "comparisons.csv",
        "rankings.csv",
        "ablations.csv",
        "mixture_scores.csv",
        "examples.csv",
        "protocol.json",
        "selection.json",
        "manifest.json",
        "replay.json",
    ):
        shutil.copy2(output / name, report_dir / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(run(args.output, args.report))
