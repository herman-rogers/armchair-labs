"""Research-only opening-QB job probabilities and conditional season workloads."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from artifact_inputs import require_audited_version
from data_integrity_audit import digest
from preseason_role_workload import ROLE, interval
from sklearn.linear_model import LogisticRegression
from value_capture_audit import finite, model_values, rank_ids, score, summarize

from engine.metrics.outlook import fit, unique

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research/qb_role_evidence_20260923.csv"
DESIGN = ROOT / "research/qb_role_transition_design.md"
USAGE = (*ROLE, "pass_efficiency", "rush_efficiency")
NEWS = ("evidence_starter", "evidence_competition", "evidence_backup")
MARKET = ("log_ecr", "inverse_ecr")
SPECS = {
    "usage": USAGE,
    "evidence": (*USAGE, *NEWS),
    "market": (*USAGE, *MARKET),
    "market_evidence": (*USAGE, *MARKET, *NEWS),
}
TARGETS = ("season_points", "attempts", "carries", "offensive_games")
MODELS = tuple(f"{kind}_{spec}" for kind in ("direct", "mixture") for spec in SPECS)
POINT_MODELS = ("saved_v2", "pilot_recent", "pilot_role", *MODELS)


def in_review(row):
    return (
        row["position"] == "QB"
        and row["player_population"] == "returner"
        and finite(row.get("role_games"))
        and row["role_games"] < 10
        and finite(row.get("market_ecr"))
        and row["market_ecr"] <= 32
    )


def load_evidence(path):
    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    for r in records:
        r["season"] = int(r["season"])
        for key in ("published_on", "captured_on"):
            date.fromisoformat(r[key])
        if r["event_on"]:
            date.fromisoformat(r["event_on"])
        if r["state"] not in ("starter", "competition", "backup"):
            raise ValueError("Invalid evidence state")
        if r["captured_on"] < max(r["published_on"], r["event_on"] or r["published_on"]):
            raise ValueError("Capture precedes evidence")
        if not r["source_url"].startswith("https://") or not r["summary"]:
            raise ValueError("Missing source or interpretation")
    return records


def resolve_evidence(row, records, *, strict_day=False):
    cutoff = row["forecast_cutoff_date"]
    available = (
        [
            r
            for r in records
            if (r["season"], r["player_id"]) == (row["season"], row["player_id"])
            and (
                max(r["published_on"], r.get("event_on") or r["published_on"]) < cutoff
                if strict_day
                else max(r["published_on"], r.get("event_on") or r["published_on"]) <= cutoff
            )
        ]
        if in_review(row)
        else []
    )
    if not available:
        return "unknown", "missing_or_late" if in_review(row) else "outside_review", []
    latest = max(max(r["published_on"], r.get("event_on") or r["published_on"]) for r in available)
    latest_rows = [
        r
        for r in available
        if max(r["published_on"], r.get("event_on") or r["published_on"]) == latest
    ]
    states = {r["state"] for r in latest_rows}
    if len(states) != 1:
        return "unknown", "conflict", latest_rows
    return next(iter(states)), "dated_evidence", latest_rows


def attach_evidence(rows, records, *, strict_day=False):
    result = []
    for original in rows:
        r = original.copy()
        state, resolution, sources = resolve_evidence(r, records, strict_day=strict_day)
        r.update(
            review_cohort=in_review(r),
            evidence_state=state,
            evidence_resolution=resolution,
            evidence_urls=[s["source_url"] for s in sources],
            evidence_dates=[s["published_on"] for s in sources],
        )
        for label in ("starter", "competition", "backup"):
            r[f"evidence_{label}"] = float(state == label)
        ecr = r.get("market_ecr")
        r["log_ecr"] = float(np.log(ecr)) if finite(ecr) and ecr > 0 else None
        r["inverse_ecr"] = 1 / ecr if finite(ecr) and ecr > 0 else None
        result.append(r)
    return result


def opening_starters(schedule):
    """Outcomes only: each team's first scheduled REG game, even if not Week 1."""
    unique(schedule, ["game_id"])
    first = {}
    for r in schedule.filter(pl.col("game_type") == "REG").sort("gameday").to_dicts():
        for side in ("home", "away"):
            key = (r["season"], r[f"{side}_team"])
            first.setdefault(key, r[f"{side}_qb_id"])
    if any(pid is None for pid in first.values()):
        raise ValueError("Unknown first-game starting QB")
    result = defaultdict(set)
    for (year, _), pid in first.items():
        result[year].add(pid)
    return result


def enrich_history(rows, raw, snaps, schedule, scoring):
    """Historical efficiency features and strictly separate outcome labels."""
    keys = ["player_id", "season", "week"]
    raw = raw.filter((pl.col("season_type") == "REG") & pl.col("player_id").is_not_null())
    unique(raw, keys)
    unique(snaps, keys)
    annual = defaultdict(lambda: defaultdict(float))
    for r in raw.to_dicts():
        a = annual[r["player_id"], r["season"]]

        def val(k, r=r):
            return float(r.get(k) or 0)

        a["attempts"] += val("attempts")
        a["carries"] += val("carries")
        a["pass_points"] += (
            val("passing_yards") / scoring["passing"]["yards_per_point"]
            + val("passing_tds") * scoring["passing"]["touchdown"]
            + val("passing_interceptions") * scoring["passing"]["interception"]
            + val("passing_2pt_conversions") * scoring["passing"]["two_point_conversion"]
            + val("sack_fumbles_lost") * scoring["misc"]["fumble_lost"]
        )
        a["rush_points"] += (
            val("rushing_yards") * scoring["rushing"]["points_per_yard"]
            + val("rushing_tds") * scoring["rushing"]["touchdown"]
            + val("rushing_2pt_conversions") * scoring["rushing"]["two_point_conversion"]
            + val("rushing_fumbles_lost") * scoring["misc"]["fumble_lost"]
        )
    participation = defaultdict(lambda: [0, 0])
    for r in snaps.to_dicts():
        counts = participation[r["player_id"], r["season"]]
        counts[0] += (r["offense_snaps"] or 0) > 0
        counts[1] += (r["offense_pct"] or 0) >= 0.5
    starters = opening_starters(schedule)
    result = []
    for row in rows:
        r = row.copy()
        for field in ("attempts", "carries", "pass_points", "rush_points"):
            r[f"prior3_{field}"] = sum(
                annual[r["player_id"], y][field] for y in range(r["season"] - 3, r["season"])
            )
        r["actual_opening_starter"] = int(r["player_id"] in starters[r["season"]])
        r["actual_offensive_games"], r["actual_role_games"] = participation[
            r["player_id"], r["season"]
        ]
        result.append(r)
    return result


def add_efficiency(train, test):
    """Exposure-weighted training priors; held-out outcomes never set shrinkage."""
    train, test = [r.copy() for r in train], [r.copy() for r in test]
    priors = {}
    for label, denominator, strength in (("pass", "attempts", 200), ("rush", "carries", 60)):
        den = sum(r[f"prior3_{denominator}"] for r in train)
        if den <= 0:
            raise ValueError("No earlier efficiency exposures")
        prior = sum(r[f"prior3_{label}_points"] for r in train) / den
        priors[label] = prior
        for r in train + test:
            r[f"{label}_efficiency"] = (r[f"prior3_{label}_points"] + strength * prior) / (
                r[f"prior3_{denominator}"] + strength
            )
    return train, test, priors


def probabilities(train, test, features):
    # Reuse the rigorously train-only numerical transform from the ridge helper.
    transform = fit(train, features, "actual_opening_starter")
    if transform is None:
        raise ValueError("Insufficient probability training rows")

    def matrix(rows):
        x = np.array([[r.get(f) for f in features] for r in rows], dtype=float)
        missing = ~np.isfinite(x)
        x = np.column_stack((np.where(missing, transform.fill, x), missing.astype(float)))
        return (x - transform.center) / transform.scale

    y = np.array([r["actual_opening_starter"] for r in train])
    if len(set(y)) != 2:
        raise ValueError("Both opening roles required in training")
    model = LogisticRegression(C=1, max_iter=5000, random_state=20260923)
    model.fit(matrix(train), y)
    if int(model.n_iter_[0]) >= 5000:
        raise ValueError("Logistic fit did not converge")
    return model.predict_proba(matrix(test))[:, 1]


def mixture(probability, starter, backup):
    return probability * starter + (1 - probability) * backup


def predict(rows):
    result, folds = [], []
    for year in range(2019, 2026):
        train = [r for r in rows if r["eligible"] and r["season"] < year]
        test = [r for r in rows if r["eligible"] and r["season"] == year]
        if not test:
            continue
        train, test, priors = add_efficiency(train, test)
        counts = Counter(r["actual_opening_starter"] for r in train)
        if min(counts.values()) < 30 or len(counts) != 2:
            raise ValueError("Insufficient conditional training rows")
        folds.append(
            dict(
                season=year,
                train=len(train),
                test=len(test),
                last_train_season=max(r["season"] for r in train),
                starter_train=counts[1],
                backup_train=counts[0],
                efficiency_priors=priors,
            )
        )
        for target in TARGETS:
            for role in (0, 1):
                fitted = fit(
                    [r for r in train if r["actual_opening_starter"] == role],
                    USAGE,
                    f"actual_{target}",
                )
                if fitted is None:
                    raise ValueError("Missing conditional target")
                for r, value in zip(test, fitted.predict(test), strict=True):
                    r[f"conditional_{role}_{target}"] = max(float(value), 0.0)
        for spec, features in SPECS.items():
            for r, probability in zip(test, probabilities(train, test, features), strict=True):
                r[f"{spec}_starter_probability"] = float(probability)
            for target in TARGETS:
                fitted = fit(train, features, f"actual_{target}")
                if fitted is None:
                    raise ValueError("Missing direct target")
                for r, value in zip(test, fitted.predict(test), strict=True):
                    r[f"direct_{spec}_{target}"] = max(float(value), 0.0)
                    r[f"mixture_{spec}_{target}"] = mixture(
                        r[f"{spec}_starter_probability"],
                        r[f"conditional_1_{target}"],
                        r[f"conditional_0_{target}"],
                    )
        result.extend(test)
    return result, folds


def season_stats(values):
    a = np.asarray(values, dtype=float)
    return dict(
        mean=float(a.mean()),
        bootstrap_95=interval(values),
        seasons=len(values),
        leave_one_season_out_range=[
            float(min(np.delete(a, i).mean() for i in range(len(a)))),
            float(max(np.delete(a, i).mean() for i in range(len(a)))),
        ]
        if len(a) > 1
        else None,
    )


def cohorts(rows):
    return {
        "all": rows,
        "review": [r for r in rows if r["review_cohort"]],
        "prior_box_games_under10": [r for r in rows if r["games"] < 10],
        "review_starter": [
            r for r in rows if r["review_cohort"] and r["evidence_state"] == "starter"
        ],
        "review_competition": [
            r for r in rows if r["review_cohort"] and r["evidence_state"] == "competition"
        ],
        "review_unknown": [
            r for r in rows if r["review_cohort"] and r["evidence_state"] == "unknown"
        ],
        "starter_under8_realized_role_games": [
            r for r in rows if r["evidence_state"] == "starter" and r["actual_role_games"] < 8
        ],
    }


def diagnostics(rows):
    result = {}
    for label, group in cohorts(rows).items():
        if not group:
            continue
        years = sorted({r["season"] for r in group})
        point, workloads, calibration, paired = {}, {}, {}, {}
        for target in TARGETS:
            models = POINT_MODELS if target == "season_points" else MODELS
            table = {}
            for model in models:
                yearly = [
                    float(
                        np.mean(
                            [
                                abs(r[f"{model}_{target}"] - r[f"actual_{target}"])
                                for r in group
                                if r["season"] == y
                            ]
                        )
                    )
                    for y in years
                ]
                table[model] = {
                    "mae": float(np.mean(yearly)),
                    "yearly_mae": dict(zip(years, yearly, strict=True)),
                }
            if target == "season_points":
                point = table
            else:
                workloads[target] = table
        for left, right in (
            ("mixture_market_evidence", "mixture_market"),
            ("direct_market_evidence", "direct_market"),
            ("mixture_market_evidence", "direct_market"),
            ("mixture_market_evidence", "saved_v2"),
            ("direct_market", "saved_v2"),
        ):
            # Positive differences mean the challenger has LOWER error.
            paired[f"{left}_vs_{right}"] = season_stats(
                [point[right]["yearly_mae"][y] - point[left]["yearly_mae"][y] for y in years]
            )
        for spec in SPECS:
            ps = np.array([r[f"{spec}_starter_probability"] for r in group])
            ys = np.array([r["actual_opening_starter"] for r in group])
            bins = []
            for lo in np.arange(0, 1, 0.2):
                mask = (ps >= lo) & (ps < lo + 0.2 + (1e-10 if lo >= 0.8 else 0))
                if mask.any():
                    bins.append(
                        dict(
                            lower=float(lo),
                            n=int(mask.sum()),
                            predicted=float(ps[mask].mean()),
                            observed=float(ys[mask].mean()),
                        )
                    )
            clipped = np.clip(ps, 1e-9, 1 - 1e-9)
            calibration[spec] = dict(
                brier=float(np.mean((ps - ys) ** 2)),
                log_loss=float(-np.mean(ys * np.log(clipped) + (1 - ys) * np.log(1 - clipped))),
                mean_probability=float(ps.mean()),
                actual_fraction=float(ys.mean()),
                bins=bins,
            )
        result[label] = dict(
            n=len(group),
            season_counts=dict(Counter(r["season"] for r in group)),
            points=point,
            workloads=workloads,
            calibration=calibration,
            paired_point_improvement=paired,
        )
    return result


def market_evaluation(predictions, full):
    lookup = {(r["player_id"], r["season"]): r for r in predictions}
    summaries, misses = {}, []
    for model in POINT_MODELS:
        folds = []
        for year in range(2019, 2026):
            rows = [
                r.copy()
                for r in full
                if r["forecast_season"] == year
                and r["outcome_complete"]
                and all(
                    finite(r.get(k))
                    for k in (
                        "fitted_season_points",
                        "market_overall_ecr_score",
                        "actual_availability_value",
                    )
                )
            ]
            for r in rows:
                new = lookup.get((r["player_id"], year))
                r["challenger"] = (
                    new[f"{model}_season_points"] if new else r["fitted_season_points"]
                )
            market = rank_ids(rows, {r["player_id"]: r["market_overall_ecr_score"] for r in rows})
            order = rank_ids(rows, model_values(rows, "challenger"))
            actual = rank_ids(rows, {r["player_id"]: r["actual_availability_value"] for r in rows})
            folds.append({"season": year, **score(rows, order, market, 60)})
            for r in rows:
                pid = r["player_id"]
                if pid in set(actual[:60]) - set(market[:60]):
                    misses.append(
                        dict(
                            model=model,
                            season=year,
                            player_id=pid,
                            name=r["player_display_name"],
                            position=r["position"],
                            prior_games=r["games"],
                            recovered=pid in order[:60],
                            modeled=(pid, year) in lookup,
                        )
                    )
        summaries[model] = {
            **summarize(folds, 60),
            "missed_cohorts": {
                label: dict(misses=len(group), recovered=sum(r["recovered"] for r in group))
                for label, group in {
                    "prior_under10": [
                        r for r in misses if r["model"] == model and r["prior_games"] < 10
                    ],
                    "RB": [r for r in misses if r["model"] == model and r["position"] == "RB"],
                    "QB": [r for r in misses if r["model"] == model and r["position"] == "QB"],
                }.items()
            },
        }
    return summaries, misses


def audit_adp_windows(windows, cutoffs):
    """A final-preseason aggregate cannot be backdated to an earlier forecast."""
    audited = []
    for original in windows:
        r = original.copy()
        r["forecast_cutoff_date"] = cutoffs[r["forecast_season"]]
        match = re.match(r"\d{4}-\d{2}-\d{2}\.\.(\d{4}-\d{2}-\d{2})", r["window"])
        r["window_end"] = match.group(1) if match else None
        r["admissible"] = bool(r["window_end"] and r["window_end"] <= r["forecast_cutoff_date"])
        r["reason"] = (
            "dated_window_within_cutoff"
            if r["admissible"]
            else "window_ends_after_cutoff"
            if r["window_end"]
            else "no_certified_end_at_or_before_cutoff"
        )
        audited.append(r)
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history, pilot, output = args.history.resolve(), args.pilot.resolve(), args.output.resolve()
    provenance = require_audited_version(history)
    if output.exists() or not output.is_relative_to(ROOT / "data/research"):
        raise ValueError("Output must be a new directory under data/research")
    pilot_report = json.loads((pilot / "report.json").read_text())
    if pilot_report["history"] != provenance:
        raise ValueError("Pilot uses different historical version")
    for name in ("features.parquet", "predictions.parquet"):
        if digest(pilot / name) != pilot_report["output_sha256"][name]:
            raise ValueError("Pilot artifact hash mismatch")
    protected = {name: digest(ROOT / name) for name in pilot_report["protected_artifact_sha256"]}
    source_paths = [
        pilot / name for name in ("report.json", "features.parquet", "predictions.parquet")
    ]
    pred_path = history / "outputs/metric_backtest_predictions.parquet"
    source_paths += [pred_path, EVIDENCE, DESIGN]
    full_frame = pl.read_parquet(pred_path)
    frame = pl.read_parquet(pilot / "features.parquet").filter(pl.col("position") == "QB")
    frame = frame.join(
        full_frame.select("player_id", "forecast_season", "market_ecr"),
        on=["player_id", "forecast_season"],
        validate="1:1",
    )
    index = {
        p: set(pl.scan_parquet(p).collect_schema())
        for p in (history / "cache/nflverse").glob("*.parquet")
    }

    def sources(required):
        paths = sorted(p for p, cols in index.items() if set(required) <= cols)
        if not paths:
            raise ValueError(f"Missing frozen source: {required}")
        source_paths.extend(paths)
        return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")

    raw = sources(["player_id", "fantasy_points_ppr", "sack_fumbles_lost"])
    players = sources(["gsis_id", "pfr_id", "rookie_season"])
    identities = players.select(pl.col("gsis_id").alias("player_id"), "pfr_id").drop_nulls()
    unique(identities, ["pfr_id"])
    snaps = sources(["pfr_player_id", "offense_snaps", "offense_pct"])
    snaps = snaps.filter(
        (pl.col("game_type") == "REG") & pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    )
    snaps = snaps.join(
        identities, left_on="pfr_player_id", right_on="pfr_id", how="inner", validate="m:1"
    )
    schedule = sources(["game_id", "home_qb_id", "away_qb_id", "gameday"])
    schedule = schedule.filter(pl.col("season").is_between(2014, 2025))
    scoring_path = history / "implementation/engine/config/scoring.yaml"
    source_paths.append(scoring_path)
    rows = enrich_history(
        frame.to_dicts(), raw, snaps, schedule, yaml.safe_load(scoring_path.read_text())
    )
    evidence = load_evidence(EVIDENCE)
    cohort_keys = {(r["season"], r["player_id"]) for r in rows if in_review(r)}
    if any((r["season"], r["player_id"]) not in cohort_keys for r in evidence):
        raise ValueError("Evidence outside the fixed review cohort")
    rows = attach_evidence(rows, evidence)
    pilot_lookup = {
        (r["season"], r["player_id"]): r
        for r in pl.read_parquet(pilot / "predictions.parquet").to_dicts()
    }
    for r in rows:
        r["saved_v2_season_points"] = r["fitted_season_points"]
        p = pilot_lookup.get((r["season"], r["player_id"]), {})
        r["pilot_recent_season_points"] = p.get("recent_usage_season_points")
        r["pilot_role_season_points"] = p.get("role_summary_season_points")
    predictions, folds = predict(rows)
    strict_predictions, _ = predict(attach_evidence(rows, evidence, strict_day=True))
    market, misses = market_evaluation(predictions, full_frame.to_dicts())
    for key in ("calls", "hits", "market_alternative_hits", "net_value_per_fold", "missed_cohorts"):
        actual = market["saved_v2"][key]
        if key == "missed_cohorts":
            actual = {k: actual[k] for k in ("prior_under10", "RB")}
        if actual != pilot_report["market"]["saved_v2"][key]:
            raise ValueError(f"Saved V2 reconciliation failed: {key}")
    adp_path = history / "static/market_adp_backfill.csv"
    source_paths.append(adp_path)
    adp = pl.read_csv(adp_path).filter(pl.col("forecast_season").is_between(2019, 2025))
    adp_windows = (
        adp.select("forecast_season", "source", "window")
        .unique()
        .sort("forecast_season", "source")
        .to_dicts()
    )
    adp_windows = audit_adp_windows(
        adp_windows, {r["season"]: r["forecast_cutoff_date"] for r in rows}
    )
    report = dict(
        generated_at=datetime.now(UTC).isoformat(),
        research_only=True,
        history=provenance,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(source_paths))},
        implementation_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                ROOT / "research/preseason_role_workload.py",
                ROOT / "research/value_capture_audit.py",
                ROOT / "research/artifact_inputs.py",
                ROOT / "research/data_integrity_audit.py",
                ROOT / "src/engine/metrics/outlook.py",
                ROOT / "research/test_qb_role_transition.py",
            )
        },
        package_versions={name: version(name) for name in ("numpy", "polars", "scikit-learn")},
        design=dict(
            specifications=SPECS,
            targets=TARGETS,
            conditional_features=USAGE,
            ridge_penalty=10,
            logistic_C=1,
            pass_shrinkage_attempts=200,
            rush_shrinkage_carries=60,
            evidence_cutoff="end of cutoff day; strict prior-day sensitivity also saved",
        ),
        coverage=dict(
            feature_rows=len(rows),
            eligible_rows=sum(r["eligible"] for r in rows),
            test_rows=len(predictions),
            review_rows=len(cohort_keys),
            review_train=sum(r["review_cohort"] and r["season"] < 2019 for r in rows),
            review_test=sum(r["review_cohort"] for r in predictions),
            review_states=dict(Counter(r["evidence_state"] for r in rows if r["review_cohort"])),
            review_test_states=dict(
                Counter(r["evidence_state"] for r in predictions if r["review_cohort"])
            ),
            evidence_records=len(evidence),
            late_records=sum(
                max(e["published_on"], e["event_on"] or e["published_on"])
                > next(
                    r["forecast_cutoff_date"]
                    for r in rows
                    if (r["season"], r["player_id"]) == (e["season"], e["player_id"])
                )
                for e in evidence
            ),
            unresolved=[
                dict(season=r["season"], player_id=r["player_id"], name=r["player_display_name"])
                for r in rows
                if r["review_cohort"] and r["evidence_state"] == "unknown"
            ],
        ),
        folds=folds,
        diagnostics=diagnostics(predictions),
        strict_prior_day=diagnostics(strict_predictions),
        market=market,
        adp=dict(
            status="not_used",
            admissible_windows=sum(r["admissible"] for r in adp_windows),
            windows=adp_windows,
        ),
        limitations=[
            "Reused 2019-2025 seasons are exploratory, not independent validation.",
            "Manual evidence coverage is incomplete; unknown rows stay in the cohort.",
            "Current page publication dates do not prove unchanged historical text.",
            "Competition merges favored and trailing contenders; verbal strength is uncalibrated.",
            "Evidence is collected only for the fixed ECR<=32 low-prior-role cohort.",
            "Opening job is not season-long retention or injury availability.",
            "Conditional ridge experts can extrapolate implausibly for novel starter profiles.",
            "Shrinkage strength and architecture are unvalidated fixed choices.",
            "Source capture is retrospective; no future test outcomes enter fits.",
            "Seven season clusters limit intervals; no correction for multiple comparisons.",
            "QB returners only; RB/TE roles and rookie forecasts are not tested.",
            "Top-60 swaps are not roster-constrained acquisition profit.",
        ],
        lamar_2019=next(
            r for r in predictions if r["player_id"] == "00-0034796" and r["season"] == 2019
        ),
        protected_artifact_sha256=protected,
        protected_unchanged=all(digest(ROOT / name) == sha for name, sha in protected.items()),
    )
    if not report["protected_unchanged"]:
        raise ValueError("Protected artifacts changed during research")
    json.dumps(report, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    for name in report["implementation_sha256"]:
        target = output / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copy2(EVIDENCE, output / EVIDENCE.name)
    shutil.copy2(DESIGN, output / DESIGN.name)
    for name, values in (
        ("features", rows),
        ("predictions", predictions),
        ("strict_prior_day_predictions", strict_predictions),
        ("market_misses", misses),
    ):
        pl.DataFrame(values, infer_schema_length=None).write_parquet(output / f"{name}.parquet")
    review_columns = [
        "season",
        "player_id",
        "player_display_name",
        "forecast_cutoff_date",
        "market_ecr",
        "role_games",
        "games",
        "evidence_state",
        "evidence_resolution",
    ]
    pl.DataFrame([r for r in rows if r["review_cohort"]]).select(review_columns).write_csv(
        output / "review_cohort.csv"
    )
    report["output_sha256"] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "coverage": report["coverage"],
                "point_mae": {
                    cohort: {m: round(v["mae"], 3) for m, v in d["points"].items()}
                    for cohort, d in report["diagnostics"].items()
                    if cohort in ("all", "review")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
