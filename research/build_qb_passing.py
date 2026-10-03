"""Run the fixed QB passing protocol, preserving forecasts and all rejected models."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from engine.data.nextgen import load_analysis
from engine.data.releases import digest, identifier, load_gold, reference, write_json
from engine.espn.attention import reviewed_news
from engine.metrics.qb_passing import (
    CHALLENGERS,
    HORIZONS,
    MODELS,
    REFERENCES,
    build_panel,
    choose_recipe,
    compile_events,
    fit_fold,
    news_known_at_issue,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def evaluate(predictions):
    evaluations, decisions = [], []
    for horizon in HORIZONS:
        for origin_type in ("preseason", "weekly"):
            for window, first in (("all_history", 2007), ("modern", 2019)):
                rows = [
                    r
                    for r in predictions
                    if r["actual"] is not None
                    and r["season"] >= first
                    and r["horizon"] == horizon
                    and (r["through_week"] == 0) == (origin_type == "preseason")
                ]
                for model in (*MODELS, "reference", "challenger_policy"):
                    summary = summarize(rows, model)
                    summary.update(horizon=horizon, origin_type=origin_type, window=window)
                    intervals = [r for r in rows if r["lower"] is not None]
                    summary["interval_coverage"] = (
                        float(np.mean([r["lower"] <= r["actual"] <= r["upper"] for r in intervals]))
                        if intervals
                        else None
                    )
                    summary["interval_n"] = len(intervals)
                    summary["interval_width"] = (
                        float(np.mean([r["upper"] - r["lower"] for r in intervals]))
                        if intervals
                        else None
                    )
                    # These intervals describe the reference only, never every candidate.
                    summary["interval_model"] = "reference"
                    summary["groups"] = []
                    for field, values in (
                        ("sample_group", ("sparse_history", "established")),
                        ("role_group", ("current_low", "current_substantial")),
                    ):
                        for group in values:
                            subset = [r for r in rows if r[field] == group]
                            if subset:
                                group_intervals = [r for r in subset if r["lower"] is not None]
                                group_summary = dict(group=group, **summarize(subset, model))
                                group_summary.update(
                                    interval_n=len(group_intervals),
                                    interval_coverage=float(
                                        np.mean(
                                            [
                                                r["lower"] <= r["actual"] <= r["upper"]
                                                for r in group_intervals
                                            ]
                                        )
                                    )
                                    if group_intervals
                                    else None,
                                )
                                summary["groups"].append(group_summary)
                    evaluations.append(summary)
                policy = next(
                    r
                    for r in evaluations
                    if r["horizon"] == horizon
                    and r["origin_type"] == origin_type
                    and r["window"] == window
                    and r["model"] == "challenger_policy"
                )
                baseline = next(
                    r
                    for r in evaluations
                    if r["horizon"] == horizon
                    and r["origin_type"] == origin_type
                    and r["window"] == window
                    and r["model"] == "reference"
                )
                passes = (
                    policy["mse_gain"] >= 0.02 * baseline["mse"]
                    and policy["ci_low"] > 0
                    and policy["mae"] <= 1.02 * baseline["mae"]
                    and all(
                        g["mse_gain"] >= -0.1 * (g["mse"] + g["mse_gain"])
                        for g in policy["groups"]
                        if g["n"] >= 100 and g["years"] >= 3
                    )
                )
                decisions.append(
                    dict(
                        horizon=horizon,
                        origin_type=origin_type,
                        window=window,
                        retrospective_checks_passed=bool(passes),
                        serving="shadow",
                        reason="Prospective review pending; retrospective checks "
                        + ("passed" if passes else "not passed"),
                    )
                )
    return evaluations, decisions


def build(data, version, sources, analysis_version=None):
    base_ref = (
        reference(data / "research" / identifier(analysis_version)) if analysis_version else None
    )
    base, manifest = load_analysis(data, base_ref)
    gold = load_gold(data, manifest["gold"]["version"])
    root = data / "research" / identifier(version)
    root.mkdir(parents=True, exist_ok=False)
    now = datetime.now(UTC).isoformat()
    observations = gold.manifest["current_observations"]
    season, through = observations["season"], observations["through_week"]
    shutil.copyfile(ROOT / "research/qb_passing_run_protocol.md", root / "protocol.md")
    for name in (
        "src/engine/metrics/qb_passing.py",
        "research/build_qb_passing.py",
        "src/engine/metrics/experimental.py",
        "src/engine/metrics/transaction_events.py",
        "src/engine/metrics/current_rankings.py",
        "src/engine/metrics/nextgen.py",
        "pyproject.toml",
    ):
        path = root / "implementation" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    supplements = json.loads((sources / "events.json").read_text())
    shutil.copytree(sources, root / "sources")
    for event in supplements:
        if digest(root / "sources" / event["source_file"]) != event["sha256"]:
            raise ValueError("Supplementary evidence changed")
    write_json(
        root / "started.json",
        dict(
            started_at=now,
            protocol_sha256=digest(root / "protocol.md"),
            source_analysis=reference(base),
            gold=gold.ref,
        ),
    )
    features = gold.read("preseason_features")
    identities = gold.read("players")
    events = compile_events(
        gold.read("nfl_transactions"),
        identities,
        features.filter(pl.col("position") == "QB"),
        supplements,
    )
    write_json(root / "events.json", events)
    cols = [
        "player_id",
        "player_display_name",
        "position",
        "season_type",
        "season",
        "week",
        "team",
        "attempts",
        "passing_yards",
    ]
    weeks = pl.concat(
        [gold.read("nfl_player_weeks").select(cols), gold.read("current_weeks").select(cols)],
        how="vertical_relaxed",
    ).unique(["player_id", "season", "week"])
    schedule = pl.concat([gold.read("nfl_schedule"), gold.read("current_schedule")]).unique(
        "game_id"
    )
    panel = build_panel(
        features,
        weeks,
        schedule,
        identities,
        events,
        season=season,
        through_week=through,
        current_issue_date=now[:10],
        absences=gold.read("absence_events").to_dicts(),
    )
    pl.DataFrame(panel, infer_schema_length=None).write_parquet(root / "panel.parquet")
    print(
        f"Panel: {len(panel):,} player/origin/horizon rows; {len(events):,} dated events",
        flush=True,
    )
    predictions, folds = [], []
    for year in range(2007, season + 1):
        train = [r for r in panel if r["season"] < year and r["actual"] is not None]
        test = [r for r in panel if r["season"] == year]
        print(f"QB {year}: training {len(train):,}; forecasting {len(test):,}", flush=True)
        fits, components = fit_fold(train, test)
        earlier = [r for r in predictions if r["season"] < year and r["actual"] is not None]
        selections = {}
        for h in HORIZONS:
            for origin in ("preseason", "weekly"):
                selections[h, origin] = (
                    choose_recipe(earlier, h, origin, REFERENCES, "career_reference"),
                    choose_recipe(earlier, h, origin, CHALLENGERS, "conditional_boost"),
                )
        residuals = defaultdict(list)
        for r in earlier:
            value = (r["actual"] - r["reference"]) / r["scheduled_games"]
            residuals[r["horizon"], r["role_group"]].append(value)
            residuals[r["horizon"], "all"].append(value)
        quantiles = {k: np.quantile(v, [0.1, 0.9]) for k, v in residuals.items() if len(v) >= 100}
        folds.append(
            dict(
                season=year,
                train_first_year=min(r["season"] for r in train),
                train_last_year=max(r["season"] for r in train),
                training_rows=len(train),
                candidate_rows=len(test),
                selections={"/".join(k): v for k, v in selections.items()},
            )
        )
        for i, r in enumerate(test):
            origin = "preseason" if r["through_week"] == 0 else "weekly"
            ref, challenge = selections[r["horizon"], origin]
            values = {m: float(fits[m][i]) for m in MODELS}
            interval_key = (r["horizon"], r["role_group"])
            q = quantiles.get(interval_key, quantiles.get((r["horizon"], "all")))
            bounds = (
                np.maximum(q * r["scheduled_games"] + values[ref], 0).tolist()
                if q is not None
                else [None, None]
            )
            if r["allowed_fraction"] == 0:
                bounds = [0.0, 0.0]
            predictions.append(
                dict(
                    **{k: v for k, v in r.items() if not k.startswith("x_")},
                    **values,
                    reference=values[ref],
                    challenger_policy=values[challenge],
                    reference_recipe=ref,
                    challenger_recipe=challenge,
                    lower=bounds[0],
                    upper=bounds[1],
                    probability_primary=float(components["primary"]["probability"][i])
                    * float(r["allowed_fraction"] > 0),
                    probability_reserve=float(components["reserve"]["probability"][i])
                    * float(r["allowed_fraction"] > 0),
                    conditional_primary_attempts=float(components["primary"]["attempts"][i]),
                    conditional_primary_yards=float(components["primary"]["yards"][i]),
                )
            )
    pl.DataFrame(predictions, infer_schema_length=None).write_parquet(root / "predictions.parquet")
    write_json(root / "folds.json", folds)
    print("Evaluating matched chronological forecasts", flush=True)
    evaluations, decisions = evaluate(predictions)
    write_json(root / "evaluations.json", evaluations)
    current = [
        dict(r) for r in predictions if r["season"] == season and r["through_week"] == through
    ]
    news, warning = reviewed_news(data, season)
    write_json(root / "news.json", dict(captured_at=now, warning=warning, observations=news))
    for r in current:
        relevant = [
            n
            for n in news
            if n.get("player_id") == r["player_id"]
            and news_known_at_issue(n, now)
            and n.get("season_ending_reported")
            and (n.get("earliest_affected_week") or 1) <= r["through_week"] + 1
        ]
        r.update(
            prediction=r["reference"],
            unconstrained_prediction=r["reference"],
            evidence_status="reference",
            entry_id="qb_passing:reference:" + r["horizon"],
            issued_at=now,
            observations_saved_at=observations["saved_at"],
            constraint_reason=None,
            constraint_source=None,
            news_known_on=None,
        )
        if relevant:
            note = max(relevant, key=lambda n: n["known_on"])
            r.update(
                prediction=0.0,
                lower=0.0,
                upper=0.0,
                constraint_reason=note["summary"],
                constraint_source=note["source_url"],
                news_known_on=note["known_on"],
                medical_status=note["evidence_status"],
            )
        # Never expose retrospective target labels or unapproved challenger scores here.
        for key in list(r):
            if key.startswith("actual") or key in {
                *MODELS,
                "challenger_policy",
                "challenger_recipe",
                "probability_primary",
                "probability_reserve",
                "conditional_primary_attempts",
                "conditional_primary_yards",
            }:
                del r[key]
    pl.DataFrame(current, infer_schema_length=None).write_parquet(root / "current.parquet")
    examples = [
        r
        for r in predictions
        if r["through_week"] == 0
        and r["horizon"] == "rest_of_season"
        and (r["player_display_name"], r["season"])
        in {
            ("Tom Brady", 2009),
            ("Tom Brady", 2023),
            ("Lamar Jackson", 2019),
            ("Patrick Mahomes", 2018),
        }
    ]
    component_scores = []
    for h in HORIZONS:
        rows = [
            r
            for r in predictions
            if r["actual"] is not None and r["horizon"] == h and r["through_week"] > 0
        ]
        attempts = [r for r in rows if r["actual_attempts"] > 0]
        component_scores.append(
            dict(
                horizon=h,
                n=len(rows),
                primary_fraction_mse=float(
                    np.mean([(r["probability_primary"] - r["actual_primary"]) ** 2 for r in rows])
                ),
                execution_observed_n=len(attempts),
                execution_ypa_mae=float(
                    np.mean(
                        [
                            abs(r["execution_ypa"] - r["actual"] / r["actual_attempts"])
                            for r in attempts
                        ]
                    )
                ),
                execution_attempt_weighted_mae=float(
                    np.average(
                        [
                            abs(r["execution_ypa"] - r["actual"] / r["actual_attempts"])
                            for r in attempts
                        ],
                        weights=[r["actual_attempts"] for r in attempts],
                    )
                ),
            )
        )
    report = dict(
        version=version,
        generated_at=now,
        season=season,
        through_week=through,
        observations_saved_at=observations["saved_at"],
        training_history=f"2004–{season - 1}; passing observations from 2001",
        evaluation_history=f"2007–{season - 1}",
        modern_history=f"2019–{season - 1}",
        models=list(MODELS),
        current_rows=len(current),
        candidates=len({r["player_id"] for r in current}),
        prospective_started_at=now,
        prospective_status="pending_future_outcomes",
        serving="reference",
        decisions=decisions,
        examples=examples,
        component_scores=component_scores,
        source_analysis=reference(base),
        events=len(events),
        limitations=[
            "Retrospective provider snapshots; original historical publication vintages "
            "are unavailable.",
            "Workload states are passing-attempt categories, not medical availability "
            "or confirmed starts.",
            "Injury reports without trustworthy publication dates are excluded; "
            "unknown stays unknown.",
            "Supplementary retirements are reviewed examples, not an exhaustive census; "
            "official transactions add general coverage.",
            "Residual ranges target 80% historical coverage; coverage is measured, not guaranteed.",
            "Reference forecasts are not a validated challenger or a fantasy-point ranking.",
            "Challengers require prospective review; all inspected historical years "
            "remain retrospective.",
            "Schedule/team are frozen at issue; unknown teams use a labelled league-week fallback.",
        ],
    )
    write_json(root / "report.json", report)
    write_json(
        root / "manifest.json",
        dict(
            version=version,
            kind="qb_passing_research",
            status="complete",
            gold=gold.ref,
            source_analysis=reference(base),
            created_at=now,
            files={str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
        ),
    )
    print(
        json.dumps(dict(version=version, current_rows=len(current), decisions=decisions), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--analysis-version", help="Verified staged analysis for a new data release"
    )
    parser.add_argument(
        "--sources", type=Path, default=ROOT / "data/research/qb_passing_sources_20260924_r1"
    )
    args = parser.parse_args()
    build(args.data, args.version, args.sources, args.analysis_version)
