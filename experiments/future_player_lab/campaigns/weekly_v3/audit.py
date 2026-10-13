"""Independent read-only acceptance audit of the v3 research and model bundle.

Recomputes selection and paired errors rather than accepting report prose. Model
replay necessarily uses the saved estimator API; numeric distribution moments and
score checks below are separately implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN = ROOT / "experiments/future_player_lab/runs/opening_drive_v3_20261011_002"
PROBABILITIES = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
QCOLS = [f"q{round(p * 100):02d}" for p in PROBABILITIES]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile_audit(frame):
    """Moments of the published linearly interpolated inverse CDF with flat tails."""
    q = frame[QCOLS].to_numpy(dtype=float)
    v = np.column_stack([q[:, 0], q, q[:, -1]])
    widths = np.diff(np.r_[0.0, PROBABILITIES, 1.0])
    first = np.sum(widths * (v[:, :-1] + v[:, 1:]) / 2, axis=1)
    second = np.sum(widths * (v[:, :-1] ** 2 + v[:, :-1] * v[:, 1:] + v[:, 1:] ** 2) / 3, axis=1)
    std = np.sqrt(np.maximum(second - first**2, 0))
    return dict(
        n=len(frame),
        finite=bool(np.isfinite(q).all()),
        ordered=bool((np.diff(q, axis=1) >= -1e-10).all()),
        std_finite_nonnegative=bool(np.isfinite(frame["std"]).all() and (frame["std"] >= 0).all()),
        mean_max_error=float(np.max(np.abs(first - frame.prediction))),
        std_max_error=float(np.max(np.abs(std - frame["std"]))),
        materially_inconsistent_std=int((np.abs(std - frame["std"]) > 1e-6).sum()),
    )


def independent_scores(frame):
    f = frame[frame.target_points.notna()]
    y = f.target_points.to_numpy()
    e = f.prediction.to_numpy() - y
    out = dict(
        n=len(f),
        rmse=float(np.sqrt(np.mean(e**2))),
        mae=float(np.mean(np.abs(e))),
        bias=float(e.mean()),
    )
    if set(QCOLS).issubset(f.columns):
        residual = y[:, None] - f[QCOLS].to_numpy()
        loss = np.where(residual >= 0, residual * PROBABILITIES, residual * (PROBABILITIES - 1))
        extended = np.column_stack([loss[:, 0], loss, loss[:, -1]])
        out["crps_quantile"] = float(
            np.mean(
                np.sum(
                    np.diff(np.r_[0, PROBABILITIES, 1]) * (extended[:, 1:] + extended[:, :-1]),
                    axis=1,
                )
            )
        )
        out["coverage_80"] = float(((y >= f.q10) & (y <= f.q90)).mean())
        out["width_80"] = float((f.q90 - f.q10).mean())
    return out


def paired_scores(selected):
    exact = selected[
        selected.season.between(2021, 2025)
        & selected.v2_exact_available.eq(1)
        & selected.target_points.notna()
    ]
    results = []
    for position in ["ALL", "QB", "RB", "WR", "TE"]:
        f = exact if position == "ALL" else exact[exact.position == position]
        e2 = f.v2_weekly_mean.to_numpy() - f.target_points.to_numpy()
        e3 = f.prediction.to_numpy() - f.target_points.to_numpy()
        results.append(
            dict(
                position=position,
                n=len(f),
                v2_rmse=float(np.sqrt(np.mean(e2**2))),
                v3_rmse=float(np.sqrt(np.mean(e3**2))),
                v2_mae=float(np.abs(e2).mean()),
                v3_mae=float(np.abs(e3).mean()),
                mse_reduction_percent=float(100 * (1 - np.mean(e3**2) / np.mean(e2**2))),
            )
        )
    return results


def verify_frozen_baselines(panel):
    """Recompute published 2021–25 horizon totals / original scheduled games."""
    schedule = pd.read_parquet(
        ROOT
        / "experiments/future_player_lab/runs/opening_drive_v1_001/inputs/history/schedule.parquet"
    )
    teams = pd.concat(
        [
            schedule.rename(columns={"home_team": "team"}),
            schedule.rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )
    scheduled = set(zip(teams.season, teams.week, teams.team, strict=True))
    baselines = {}
    for position in ["QB", "RB", "WR", "TE"]:
        for horizon, origin in [("season", 0), ("remaining", 2)]:
            f = pd.read_parquet(
                ROOT
                / f"data/campaigns/opening_drive_v2/{position.lower()}_{horizon}"
                / "artifacts/evaluation_predictions.parquet"
            )
            f = f[(f.target == "points") & (f.recipe == "ensemble")]
            for row in f.itertuples():
                n = int(
                    (
                        (teams.season == row.year)
                        & (teams.week > origin)
                        & (teams.team == row.team)
                    ).sum()
                )
                baselines[(row.year, origin, row.player_id)] = (row.prediction / n, row.team)
    exact = panel[panel.season.between(2021, 2025) & panel.v2_exact_available.eq(1)]
    errors = []
    for row in exact.itertuples():
        value, team = baselines[(row.season, 0 if row.week <= 2 else 2, row.player_id)]
        expected = value if (row.season, row.week, team) in scheduled else 0.0
        errors.append(abs(expected - row.v2_weekly_mean))
    return dict(n=len(errors), max_error=max(errors, default=0.0))


def verify_scoring_and_absence(panel, dataset):
    """Recompute labels from frozen raw components independently of data.py."""
    columns = [
        "player_id",
        "season",
        "week",
        "position",
        "team",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "rushing_yards",
        "rushing_tds",
        "receiving_yards",
        "receptions",
        "receiving_tds",
    ]
    historical = pd.read_parquet(
        ROOT / "experiments/future_player_lab/runs/opening_drive_v1_001/inputs/history/box.parquet",
        columns=columns,
    )
    snapshot = json.loads(
        (ROOT / "data/raw/snapshots/weekly_20261008_w4_r3/manifest.json").read_text()
    )
    current = pd.read_parquet(
        ROOT / "data" / snapshot["assets"]["current/weeks.parquet"]["object"], columns=columns
    )
    current = current[
        (current.season == dataset["current_season"])
        & (current.week <= dataset["current_through_week"])
    ]
    box = pd.concat([historical, current], ignore_index=True)
    box = box[box.position.isin(["QB", "RB", "WR", "TE"])]
    qb = (
        box.passing_yards * 0.04
        + box.passing_tds * 4
        - box.passing_interceptions * 2
        + box.rushing_yards * 0.1
        + box.rushing_tds * 6
    )
    skill = (
        (box.rushing_yards + box.receiving_yards) * 0.1
        + box.receptions
        + (box.rushing_tds + box.receiving_tds) * 6
    )
    box["raw_points"] = np.where(box.position.eq("QB"), qb, skill)
    appeared = panel[panel.appeared].merge(
        box[["player_id", "season", "week", "raw_points"]],
        on=["player_id", "season", "week"],
        how="left",
        validate="1:1",
    )
    points_match = np.allclose(appeared.target_points, appeared.raw_points, equal_nan=True)
    # All historical schedule rows in this capture are final; next current week
    # must have no zero labels invented from absent production.
    absence = panel[~panel.appeared & panel.target_points.notna()]
    absence_valid = (
        absence.target_points.eq(0)
        & (
            (absence.season < dataset["current_season"])
            | (absence.week <= dataset["current_through_week"])
        )
    ).all()
    return dict(
        raw_score_matches=bool(points_match),
        appeared_n=len(appeared),
        observed_absence_zeros_valid=bool(absence_valid),
        absence_n=len(absence),
    )


def audit(run_dir=DEFAULT_RUN):
    from .modeling import predict_distribution, predict_mean

    run = Path(run_dir)
    checks = []
    warnings = []

    def check(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))

    needed = [
        "manifest.json",
        "run_signature.json",
        "dataset.json",
        "panel.parquet",
        "mean_predictions.parquet",
        "distribution_predictions.parquet",
        "selected_predictions.parquet",
        "selection.json",
        "replay.json",
        "lineups/results.json",
        "lineups/calibration.json",
    ]
    missing = [name for name in needed if not (run / name).exists()]
    if missing:
        receipt = dict(
            status="pending", missing=missing, generated_at=datetime.now(UTC).isoformat()
        )
        (run / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    dataset = json.loads((run / "dataset.json").read_text())
    signature = json.loads((run / "run_signature.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    panel = pd.read_parquet(run / "panel.parquet")
    panel["row_id"] = np.arange(len(panel))
    features = dataset["features"]
    check(
        "dataset_checksum",
        digest(run / "panel.parquet") == dataset["panel_sha256"] == signature["panel_sha256"],
    )
    check("dataset_manifest_checksum", digest(run / "dataset.json") == signature["dataset_sha256"])
    failures = [
        p for p, h in dataset["source_hashes"].items() if not Path(p).exists() or digest(p) != h
    ]
    check("sealed_input_bytes", not failures, mismatches=failures)
    original = ROOT / "data/campaigns/opening_drive_v2"
    before = json.loads(
        (ROOT / "data/research/opening_drive_v3_20261011/v2_before.json").read_text()
    )
    after = {str(p.relative_to(original)): digest(p) for p in original.rglob("*") if p.is_file()}
    check(
        "original_v2_unchanged",
        before == after,
        files=len(before),
        manifest_sha256=before["manifest.json"],
    )
    changed = [
        name
        for name, h in signature["source_sha256"].items()
        if digest(Path(__file__).with_name(name)) != h
        or digest(run / "source_snapshot" / name) != h
    ]
    check("training_source_contract", not changed, mismatches=changed)
    changed = [
        name
        for name, h in manifest["artifact_hashes"].items()
        if not (run / name).exists() or digest(run / name) != h
    ]
    check("training_artifact_hashes", not changed, mismatches=changed)
    forbidden = {
        "target_points",
        "actual_team",
        "appeared",
        "label_observed",
        "team_changed_in_outcome",
        "baseline_team",
        "row_id",
    }
    check(
        "feature_allowlist",
        not forbidden.intersection(features)
        and all(pd.api.types.is_numeric_dtype(panel[c]) for c in features),
        n_features=len(features),
    )
    check("unique_player_weeks", not panel.duplicated(["season", "week", "player_id"]).any())
    check(
        "source_wide_target_missingness_policy",
        dataset.get("source_wide_missing_target_years") == list(range(2003, 2009)),
        masked_rows=dataset.get("source_wide_target_masked_rows"),
    )
    future = panel[
        (panel.season == dataset["current_season"]) & (panel.week > dataset["current_through_week"])
    ]
    check(
        "future_labels_withheld",
        future.target_points.isna().all() and (~future.label_observed).all(),
        n=len(future),
    )
    check(
        "known_schedule_mapping",
        not ((panel.scheduled_game == 0) & (panel.is_bye == 0)).any(),
        unknown_rows=int(((panel.scheduled_game == 0) & (panel.is_bye == 0)).sum()),
    )
    scoring = verify_scoring_and_absence(panel, dataset)
    check(
        "raw_scoring_and_no_future_absence_zeros",
        scoring["raw_score_matches"] and scoring["observed_absence_zeros_valid"],
        **scoring,
    )
    baseline = verify_frozen_baselines(panel)
    check("exact_v2_means_and_byes", baseline["max_error"] < 1e-10, **baseline)
    mean = pd.read_parquet(run / "mean_predictions.parquet")
    dist = pd.read_parquet(run / "distribution_predictions.parquet")
    selected = pd.read_parquet(run / "selected_predictions.parquet")
    selection = json.loads((run / "selection.json").read_text())
    metadata = json.loads((run / "candidate_metadata.json").read_text())
    expected = panel[panel.scheduled_game.eq(1) & panel.season.ge(2015)]
    good = True
    for (pos, season, _method), g in mean.groupby(["position", "season", "mean_method"]):
        want = expected[(expected.position == pos) & (expected.season == season)]
        good &= len(g) == len(want) and set(g.row_id) == set(want.row_id)
    check("mean_all_weekly_origins", good, rows=len(mean))
    actual = panel.set_index("row_id").target_points.reindex(mean.row_id).to_numpy()
    check(
        "prediction_outcomes_match_panel",
        np.allclose(actual, mean.target_points.to_numpy(), equal_nan=True),
    )
    check(
        "calibration_precedes_season",
        (dist.calibration_max_year < dist.season).all(),
        max_calibration_season=int(dist.calibration_max_year.max()),
    )
    recomputed = []
    for (pos, season, method), g in mean.groupby(["position", "season", "mean_method"]):
        if 2018 <= season <= 2020:
            recomputed.append(
                dict(position=pos, season=season, method=method, **independent_scores(g))
            )
    means = pd.DataFrame(recomputed)
    selected_checks = []
    excluded = metadata.get("selection_excluded_mean_methods", ["v2", "lgb_correction_full"])
    if isinstance(excluded, dict):
        excluded = list(excluded)
    for pos, spec in selection.items():
        subset = means[(means.position == pos) & ~means.method.isin(excluded)]
        chosen = subset.groupby("method").rmse.mean().sort_values().index[0]
        scores = []
        local = dist[
            (dist.position == pos)
            & (dist.mean_method == spec["mean_method"])
            & dist.season.between(2018, 2020)
        ]
        for (year, method), g in local.groupby(["season", "distribution_method"]):
            scores.append(dict(season=year, method=method, **independent_scores(g)))
        chosen_dist = (
            pd.DataFrame(scores).groupby("method").crps_quantile.mean().sort_values().index[0]
        )
        selected_checks.append(
            dict(
                position=pos,
                expected_mean=chosen,
                saved_mean=spec["mean_method"],
                expected_distribution=chosen_dist,
                saved_distribution=spec["distribution_method"],
            )
        )
    check(
        "selection_only_2018_2020",
        all(
            r["expected_mean"] == r["saved_mean"]
            and r["expected_distribution"] == r["saved_distribution"]
            for r in selected_checks
        ),
        policies=selected_checks,
    )
    quantiles = quantile_audit(dist)
    check(
        "distribution_moments_and_order",
        quantiles["finite"]
        and quantiles["ordered"]
        and quantiles["std_finite_nonnegative"]
        and quantiles["mean_max_error"] < 1e-7
        and quantiles["std_max_error"] < 1e-6,
        **quantiles,
    )
    replay = []
    for pos, spec in selection.items():
        current = panel[
            (panel.season == 2026) & (panel.position == pos) & panel.scheduled_game.eq(1)
        ].copy()
        with (run / "models" / f"{pos}_{spec['mean_method']}.pkl").open("rb") as file:
            mean_model = pickle.load(file)
        with (
            run
            / "models"
            / f"{pos}_{spec['mean_method']}_{spec['distribution_method']}_distribution.pkl"
        ).open("rb") as file:
            distribution_model = pickle.load(file)
        current["prediction"] = predict_mean(mean_model, current)
        q, std = predict_distribution(distribution_model, current)
        saved = selected.set_index("row_id").loc[current.row_id]
        errors = [
            np.max(np.abs(saved.prediction.to_numpy() - current.prediction.to_numpy())),
            np.max(np.abs(saved[QCOLS].to_numpy() - q)),
            np.max(np.abs(saved["std"].to_numpy() - std)),
        ]
        replay.append(
            dict(
                position=pos,
                n=len(current),
                max_error=float(max(errors)),
                fit_year=mean_model["year"],
                calibration_max_year=distribution_model["calibration_max_year"],
            )
        )
    check(
        "independent_saved_model_replay",
        all(
            r["max_error"] < 1e-8 and r["fit_year"] == 2026 and r["calibration_max_year"] < 2026
            for r in replay
        ),
        rows=replay,
    )
    cal = json.loads((run / "lineups/calibration.json").read_text())
    check(
        "lineup_calibration_clock",
        cal["covariance_before_season"] == 2026
        and cal["special_errors_before_week"] <= dataset["current_prediction_week"],
        covariance_before_season=cal["covariance_before_season"],
        special_errors_before_week=cal["special_errors_before_week"],
    )
    lineup_results = json.loads((run / "lineups/results.json").read_text())
    synthetic = pd.read_parquet(run / "lineups/synthetic_lineups.parquet")
    expanded = synthetic[["design", "season", "week", "team_id", "player_ids"]].copy()
    expanded["player_ids"] = expanded.player_ids.str.split(",")
    expanded = expanded.explode("player_ids").merge(
        selected[["player_id", "season", "week", "v2_exact_available"]],
        left_on=["player_ids", "season", "week"],
        right_on=["player_id", "season", "week"],
        how="left",
        validate="m:1",
    )
    unsupported = expanded[~expanded.v2_exact_available.eq(1)]
    affected = unsupported[["design", "season", "week", "team_id"]].drop_duplicates()
    has_support_flag = "v2_exact_all" in synthetic
    check(
        "synthetic_v2_support_explicit",
        has_support_flag and int((~synthetic.v2_exact_all).sum()) == len(affected)
        if has_support_flag
        else False,
        lineups=len(synthetic),
        affected_lineups=len(affected),
        proxy_player_allocations=len(unsupported),
    )
    calibration_diagnostics = []
    evaluation = selected[selected.season.between(2021, 2025)]
    for pos, group in evaluation.groupby("position"):
        for cohort, subset in [
            ("all", group),
            ("projected_10_plus", group[group.prediction >= 10]),
        ]:
            calibration_diagnostics.append(
                dict(position=pos, cohort=cohort, **independent_scores(subset))
            )
    cov = {(r["position"], r["cohort"]): r["coverage_80"] for r in calibration_diagnostics}
    synthetic_cov = lineup_results["synthetic"]["v3_joint"]["coverage80"]
    hybrid = lineup_results["league"]["metrics"]["v3_hybrid"]
    dependence = lineup_results["synthetic"]["paired_season_bootstrap"][
        "joint_minus_independent_crps"
    ]
    dep_low, dep_high = dependence["ci95"]
    dep_difference = dependence["season_equal_mean_difference"]
    warnings.append(
        f"Joint-minus-independent CRPS difference {dep_difference:.6f}, "
        f"five-season bootstrap95% interval [{dep_low:.6f}, {dep_high:.6f}]. "
        + (
            "The interval spans zero; predictive advantage from covariance is unestablished."
            if dep_low <= 0 <= dep_high
            else "Few-season retrospective inference remains exploratory."
        )
    )
    warnings.extend(
        [
            (
                f"Selected nominal80% intervals: QB overall coverage {cov[('QB', 'all')]:.1%}; "
                f"TE projected10+ coverage {cov[('TE', 'projected_10_plus')]:.1%}. "
                "Overall coverage hides workload-specific errors; ranges are research estimates, "
                "not validated calibrated intervals."
            ),
            (
                f"Artificial offensive-lineup nominal80% coverage is {synthetic_cov:.1%}. "
                f"Reconstructed2026 mixed-lineup coverage is {hybrid['coverage80']:.1%} "
                f"on {hybrid['interval_n']} supported team-weeks; these small retrospective "
                "samples do not establish calibrated team uncertainty."
            ),
        ]
    )
    warnings.extend(
        [
            (
                "2018–20 baseline inputs are explicitly identified persistence proxies, "
                "not exact v2 forecasts; direct v2 and correction challengers are excluded "
                "from production selection."
            ),
            (
                "2021–25 were reused in earlier research; these results are development "
                "evidence, not pristine holdout performance."
            ),
            (
                "Continuous mean-centered quantile distributions can smooth scoring-zero "
                "mass and produce negative lower tails; bounded extreme tails and standard "
                "deviations are approximation choices."
            ),
            (
                "Historical v2 is fixed at preseason/Week2 while v3 incorporates each "
                "preceding completed week; the comparison measures weekly updating plus "
                "model differences, not architecture alone."
            ),
            (
                "Current snapshots were captured after some Week5 games began; current "
                "forecasts are retrospective cutoff reconstructions."
            ),
            (
                "K/DST uncertainty transfer to ESPN means and K/DST covariance remain "
                "provisional; coverage claims require compatible complete mixed-scoring "
                "lineups."
            ),
        ]
    )
    receipt = dict(
        status="passed" if all(c["passed"] for c in checks) else "failed",
        generated_at=datetime.now(UTC).isoformat(),
        checks=checks,
        warnings=warnings,
        paired_comparison=paired_scores(selected),
        selected_calibration_diagnostics=calibration_diagnostics,
        lineup_calibration_diagnostics={
            "synthetic": lineup_results["synthetic"],
            "league_metrics": lineup_results["league"]["metrics"],
        },
        data_tests="experiments/future_player_lab/tests/test_weekly_v3_data.py",
        auditor_source_sha256=digest(__file__),
    )
    (run / "audit.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = audit(args.run_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
