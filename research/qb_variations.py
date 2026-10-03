"""Run the frozen QB variation matrix and cross-fitted fantasy bridge."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from engine.data.nextgen import load_analysis
from engine.data.releases import digest, identifier, load_gold, reference, write_json
from engine.metrics.current_rankings import compare
from engine.metrics.qb_passing import build_panel
from engine.metrics.qb_variations import (
    KEY,
    adjust_q,
    augment_panel,
    bridge_candidates,
    choose_bridge,
    fit_production,
    fit_rates,
    paired,
    score,
)

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = ("next_game", "next4", "rest_of_season")


def select(annual, horizon, origin, prefix, default):
    rows = [
        r
        for r in annual
        if r["horizon"] == horizon and r["origin_type"] == origin and r["model"].startswith(prefix)
    ]
    if len({r["season"] for r in rows}) < 3:
        return default
    names = {r["model"] for r in rows}
    return min(names, key=lambda m: (np.mean([r["mse"] for r in rows if r["model"] == m]), m))


def evaluate(predictions, ranking_predictions, rate_names, yard_names):
    evaluations, decisions = [], []
    for target, names, policy, baseline in (
        ("rate", rate_names, "rate_policy", "rate_reference"),
        ("yards", yard_names, "yards_policy", "yards_reference"),
    ):
        for h in HORIZONS:
            for origin in ("preseason", "weekly"):
                rows = [
                    r
                    for r in predictions
                    if r["actual"] is not None
                    and r["horizon"] == h
                    and (r["through_week"] == 0) == (origin == "preseason")
                ]
                views = {}
                for window, first in (("all_history", 2007), ("modern", 2019)):
                    subset = [r for r in rows if r["season"] >= first]
                    for name in [*names, policy]:
                        result = paired(subset, name, baseline, target)
                        result.update(target=target, horizon=h, origin_type=origin, window=window)
                        evaluations.append(result)
                        if name == policy:
                            views[window] = result
                if origin == "preseason":
                    continue
                reasons, groups = [], []
                for window, view in views.items():
                    if view["improvement_pct"] < (0.5 if target == "rate" else 2):
                        reasons.append(window + " improvement below required size")
                    if view["ci_low"] <= 0:
                        reasons.append(window + " interval includes no improvement")
                    if view["mae"] > view["baseline_mae"] * 1.02:
                        reasons.append(window + " MAE guardrail failed")
                    subset = [
                        r for r in rows if r["season"] >= (2019 if window == "modern" else 2007)
                    ]
                    for field, values in (
                        ("sample_group", ("sparse_history", "established")),
                        ("role_group", ("current_low", "current_substantial")),
                    ):
                        for group in values:
                            selected = [r for r in subset if r[field] == group]
                            if target == "rate":
                                selected = [r for r in selected if r["actual_attempts"] > 0]
                            if not selected:
                                continue
                            result = dict(
                                paired(selected, policy, baseline, target),
                                window=window,
                                group=group,
                            )
                            groups.append(result)
                            if (
                                result["n"] >= 100
                                and result["years"] >= 3
                                and result["improvement_pct"] < -10
                            ):
                                reasons.append(f"{window} {group} guardrail failed")
                decisions.append(
                    dict(target=target, horizon=h, groups=groups, reasons=reasons, **views)
                )
    for h in ("rest_of_season", "next4"):
        rows = [
            r
            for r in ranking_predictions
            if r["horizon"] == h and r["actual"] is not None and r["season"] >= 2011
        ]
        a, m = compare(rows, "QB"), compare([r for r in rows if r["season"] >= 2019], "QB")
        reasons = []
        if m["improvement"] < max(0.5, 0.01 * m["baseline_error"]):
            reasons.append("Modern fantasy improvement below required size")
        for label, view in (("all_history", a), ("modern", m)):
            if view["ci_low"] <= 0:
                reasons.append(label + " fantasy interval includes no improvement")
            if (
                view["improvement"] < 0
                or view["mse_improvement"] < 0
                or view["capture_improvement"] < 0
            ):
                reasons.append(label + " fantasy error or top-10 capture guardrail failed")
        cohorts = {}
        for name, subset in (
            ("rookies", [r for r in rows if r["season"] >= 2019 and r["population"] == "rookie"]),
            ("small_prior", [r for r in rows if r["season"] >= 2019 and r["prior_weeks"] < 10]),
        ):
            view = compare(subset, "QB")
            cohorts[name] = view
            if (
                view["n"] >= 100
                and view["years"] >= 5
                and view["improvement"] < -max(1, 0.05 * view["baseline_error"])
            ):
                reasons.append(name + " fantasy guardrail failed")
        decisions.append(
            dict(
                target="league_points",
                horizon=h,
                reasons=reasons,
                modern=m,
                all_history=a,
                cohorts=cohorts,
            )
        )
    adjust_q(decisions)
    for d in decisions:
        if d["q_value"] > 0.05:
            d["reasons"].append("Eight-policy multiple-comparison gate failed")
        if d["modern"]["years"] < 7 or d["all_history"]["years"] < 8:
            d["reasons"].append("Insufficient evaluation years")
        d["approved"] = not d["reasons"]
        d["reason"] = (
            "; ".join(d["reasons"])
            if d["reasons"]
            else "Nested chronological policy passed target-specific retrospective checks"
        )
    return evaluations, decisions


def ranking_only_rows(extra, rank, origin):
    extra = [
        r for r in extra if r["player_id"] == rank["player_id"] and r["season"] == rank["season"]
    ]
    rows = [
        dict(r, _ranking_only=True)
        for r in extra
        if r["through_week"] == origin and r["horizon"] == "rest_of_season"
    ]
    if not rows and rank["scheduled_games"] == 0:
        # Older 17-week seasons have no passing forecast at a Week 17 origin.
        templates = [r for r in extra if r["horizon"] == "rest_of_season"]
        if not templates:
            raise ValueError("Missing QB evidence for a completed historical season")
        template = max(templates, key=lambda r: r["through_week"])
        rows = [
            dict(
                template,
                through_week=origin,
                end_week=rank["end_week"],
                scheduled_games=0,
                x_horizon_games=0,
                _ranking_only=True,
            )
        ]
    return rows


def run(data, version, analysis_version=None):
    ref = reference(data / "research" / identifier(analysis_version)) if analysis_version else None
    base, manifest = load_analysis(data, ref)
    gold = load_gold(data, manifest["gold"]["version"])
    root = data / "research" / identifier(version)
    root.mkdir(parents=True, exist_ok=False)
    now = datetime.now(UTC).isoformat()
    shutil.copyfile(ROOT / "research/qb_variations_protocol.md", root / "protocol.md")
    for name in (
        "src/engine/metrics/qb_variations.py",
        "research/qb_variations.py",
        "src/engine/metrics/current_rankings.py",
        "src/engine/config/scoring.yaml",
    ):
        out = root / "implementation" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, out)
    write_json(
        root / "started.json",
        dict(
            started_at=now,
            source_analysis=reference(base),
            gold=gold.ref,
            protocol_sha256=digest(root / "protocol.md"),
        ),
    )
    season = gold.manifest["current_observations"]["season"]
    source_panel = pl.read_parquet(base / "qb_passing/panel.parquet").to_dicts()
    ranks = pl.read_parquet(base / "ranking_panel.parquet").to_dicts()
    columns = [
        "player_id",
        "player_display_name",
        "position",
        "season_type",
        "season",
        "week",
        "team",
        "attempts",
        "passing_yards",
        "completions",
        "passing_tds",
        "passing_interceptions",
        "passing_epa",
        "passing_cpoe",
        "rushing_yards",
        "rushing_tds",
        "carries",
        "league_points",
    ]
    weeks = pl.concat(
        [gold.read("nfl_player_weeks").select(columns), gold.read("current_weeks").select(columns)],
        how="vertical_relaxed",
    ).unique(["player_id", "season", "week"])
    keys = {(r["player_id"], r["season"], r["through_week"]) for r in source_panel}
    missing = {
        (r["player_id"], r["season"], r["through_week"]): r
        for r in ranks
        if r["position"] == "QB" and (r["player_id"], r["season"], r["through_week"]) not in keys
    }
    identities = gold.read("players")
    names = dict(identities.select("gsis_id", "display_name").iter_rows())
    # Snap-only observations already in the frozen ranking pool are eligible at
    # its cutoff. Construct their passing inputs without adding them to an earlier pool.
    for (pid, year, origin), rank in missing.items():
        feature = pl.DataFrame(
            [
                dict(
                    player_id=pid,
                    player_display_name=names[pid],
                    position="QB",
                    forecast_season=year,
                    forecast_cutoff_date=f"{year}-09-01",
                    player_population=rank["population"],
                    cutoff_preseason_team=rank["team"],
                    cutoff_preseason_status="unknown",
                    cutoff_state_resolution="unknown",
                )
            ]
        )
        extra = build_panel(
            feature,
            weeks,
            pl.concat([gold.read("nfl_schedule"), gold.read("current_schedule")]).unique("game_id"),
            identities,
            json.loads((base / "qb_passing/events.json").read_text()),
            season=season,
            through_week=origin,
            current_issue_date=now[:10],
            absences=gold.read("absence_events").to_dicts(),
        )
        source_panel.extend(ranking_only_rows(extra, rank, origin))
    panel, rank_panel = augment_panel(source_panel, weeks, gold.read("preseason_features"), ranks)
    panel = [r for r in panel if not r.get("_ranking_only")]
    pl.DataFrame(panel + rank_panel, infer_schema_length=None).write_parquet(root / "panel.parquet")
    old = {
        tuple(r[k] for k in KEY): r
        for r in pl.read_parquet(base / "qb_passing/predictions.parquet").to_dicts()
    }
    old_ranks = {
        (r["player_id"], r["season"], r["horizon"]): r
        for r in pl.read_parquet(base / "ranking_predictions.parquet")
        .filter(pl.col("position") == "QB")
        .to_dicts()
    }
    rank_templates = {
        (r["player_id"], r["season"], r["horizon"]): r for r in ranks if r["position"] == "QB"
    }
    predictions, annual, folds, rank_predictions = [], [], [], []
    coefficient = (
        1
        / yaml.safe_load((ROOT / "src/engine/config/scoring.yaml").read_text())["passing"][
            "yards_per_point"
        ]
    )
    for year in range(2007, season + 1):
        train = [r for r in panel if r["season"] < year and r["actual"] is not None]
        regular = [r for r in panel if r["season"] == year]
        ranking = [r for r in rank_panel if r["season"] == year]
        test = regular + ranking
        print(f"QB variations {year}: train {len(train):,}, forecast {len(test):,}", flush=True)
        rates = fit_rates(train, test)
        rate_names = list(rates)
        selections = {}
        selected_rates = []
        for r in test:
            h = r["horizon"].removeprefix("ranking_")
            origin = "preseason" if not r["through_week"] else "weekly"
            if (h, origin) not in selections:
                selections[h, origin] = dict(
                    rate=select(annual, h, origin, "rate_", "rate_reference"),
                    yards=select(annual, h, origin, "yards_", "yards_reference"),
                    stack=select(
                        annual,
                        h,
                        origin,
                        ("yards_profile", "yards_enriched"),
                        "yards_profile_boost_15",
                    ),
                )
            selected_rates.append(rates[selections[h, origin]["rate"]][len(selected_rates)])
        production = fit_production(train, test, np.array(selected_rates))
        yard_names = ["yards_reference", "yards_old_direct", "yards_old_conditional"] + [
            k for k in production if k.startswith("yards_")
        ]
        output = []
        for i, r in enumerate(test):
            row = dict(r)
            row.update({m: float(v[i]) for m, v in rates.items()})
            row.update({m: float(v[i]) for m, v in production.items()})
            h, origin = (
                r["horizon"].removeprefix("ranking_"),
                "preseason" if not r["through_week"] else "weekly",
            )
            selected = selections[h, origin]
            if i < len(regular):
                saved = old[tuple(r[k] for k in KEY)]
                row.update(
                    yards_reference=saved["reference"],
                    yards_old_direct=saved["direct_boost"],
                    yards_old_conditional=saved["conditional_boost"],
                )
                row["yards_policy"] = row[selected["yards"]]
            row.update(
                rate_policy=selected_rates[i],
                rate_recipe=selected["rate"],
                yards_recipe=selected["yards"],
                stack_rate=selected_rates[i],
                stack_yards_pg=row[selected["stack"]] / max(r["scheduled_games"], 1),
                stack_attempts_pg=row["attempts_enriched_boost_15"] / max(r["scheduled_games"], 1),
                stack_yards_recipe=selected["stack"],
            )
            output.append(row)
        # Cross-fitted rows contain only predictions issued without their target season.
        stack_train = [
            r for r in predictions if r["season"] < year and r["actual_points"] is not None
        ]
        rank_test = output[len(regular) :]
        for r in rank_test:
            key = r["player_id"], year, r["horizon"].removeprefix("ranking_")
            original = old_ranks.get(key, {})
            template = rank_templates[key]
            r["reference"] = original.get("reference", 0.0)
            r["ranking_role"] = template["role_group"]
        bridge = bridge_candidates(stack_train, rank_test, coefficient) if stack_train else {}
        for h in ("rest_of_season", "next4"):
            earlier = [r for r in rank_predictions if r["season"] < year and r["horizon"] == h]
            choices = {
                role: choose_bridge(earlier, role, list(bridge))
                for role in ("observed_workload", "limited_workload")
            }
            for i, r in enumerate(rank_test):
                if r["horizon"] != "ranking_" + h:
                    continue
                original = old_ranks[r["player_id"], year, h]
                recipe = choices[r["ranking_role"]]
                values = {m: float(v[i]) for m, v in bridge.items()}
                if not bridge:
                    values = {}
                rank_predictions.append(
                    dict(
                        original,
                        **values,
                        policy=values.get(recipe, r["reference"]),
                        selected_recipe=recipe,
                    )
                )
        # First bridge year has no fitted candidates; complete its columns with reference.
        if bridge:
            for r in rank_predictions:
                for name in bridge:
                    if name not in r:
                        r[name] = r["reference"]
        predictions.extend(output[: len(regular)])
        for h in HORIZONS:
            for origin in ("preseason", "weekly"):
                subset = [
                    r
                    for r in output[: len(regular)]
                    if r["horizon"] == h and (r["through_week"] == 0) == (origin == "preseason")
                ]
                if year < season:
                    for target, names in (("rate", rate_names), ("yards", yard_names)):
                        for name in names:
                            s = score(subset, name, target)
                            annual.append(
                                dict(
                                    s["annual"][0],
                                    model=name,
                                    target=target,
                                    horizon=h,
                                    origin_type=origin,
                                )
                            )
        folds.append(
            dict(
                season=year,
                training_first_year=min(r["season"] for r in train),
                training_last_year=year - 1,
                stack_training_first_year=min((r["season"] for r in stack_train), default=None),
                stack_training_last_year=max((r["season"] for r in stack_train), default=None),
                training_rows=len(train),
                forecast_rows=len(test),
                selections={"/".join(k): v for k, v in selections.items()},
            )
        )
        pl.DataFrame(output, infer_schema_length=None).write_parquet(root / f"fold_{year}.parquet")
        write_json(root / "folds.json", folds)
    print("Evaluating frozen policies and all rejected variations", flush=True)
    pl.DataFrame(predictions, infer_schema_length=None).write_parquet(root / "predictions.parquet")
    pl.DataFrame(rank_predictions, infer_schema_length=None).write_parquet(
        root / "ranking_predictions.parquet"
    )
    write_json(root / "annual.json", annual)
    evaluations, decisions = evaluate(predictions, rank_predictions, rate_names, yard_names)
    write_json(root / "evaluations.json", evaluations)
    write_json(root / "decisions.json", decisions)
    report = dict(
        version=version,
        generated_at=now,
        completed_at=datetime.now(UTC).isoformat(),
        season=season,
        through_week=gold.manifest["current_observations"]["through_week"],
        source_analysis=reference(base),
        rate_variations=len(rate_names),
        yard_variations=len(yard_names),
        ranking_variations=len(bridge),
        evaluation_years=[2007, season - 1],
        ranking_policy_years=[2011, season - 1],
        passing_point_coefficient=coefficient,
        approved_scopes=[d["target"] + ":" + d["horizon"] for d in decisions if d["approved"]],
        limitations=[
            "Retrospective development; previously inspected history is not an untouched holdout.",
            "Historical inputs are reconstructed, not complete publication-vintage snapshots.",
            "Overlapping weekly forecasts are correlated; uncertainty is clustered by season.",
            "No demonstrated market-edge or prospective accuracy claim.",
            "Rate, total production and fantasy ranking approvals are separate.",
        ],
    )
    write_json(root / "report.json", report)
    write_json(
        root / "manifest.json",
        dict(
            version=version,
            kind="qb_variations_research",
            status="complete",
            source_analysis=reference(base),
            gold=gold.ref,
            files={str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
        ),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--analysis-version", help="Verified staged analysis for a new data release"
    )
    args = parser.parse_args()
    run(args.data, args.version, args.analysis_version)
