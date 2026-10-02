"""Evaluate current-season rankings and prepare a new immutable analysis release."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from patron.data.nextgen import load_analysis
from patron.data.releases import digest, identifier, load_gold, reference, write_json
from patron.espn.attention import reviewed_news
from patron.metrics.current_rankings import (
    BASELINES,
    HORIZONS,
    POSITIONS,
    apply_constraints,
    choose_by_role,
    evaluate_scopes,
    fit_candidates,
    make_panel,
    rank_frame,
)

ROOT = Path(__file__).resolve().parents[1]


def build(data: Path, version: str, analysis_version: str | None = None):
    ref = reference(data / "research" / identifier(analysis_version)) if analysis_version else None
    base, base_manifest = load_analysis(data, ref)
    gold = load_gold(data, base_manifest["gold"]["version"])
    profile_root = data / "research" / base_manifest["profiles"]["version"]
    root = data / "research" / identifier(version)
    if root.exists():
        raise ValueError("Ranking releases are immutable; choose a new version")
    shutil.copytree(base, root)
    (root / "protocol.md").rename(root / "base_protocol.md")
    shutil.copy2(ROOT / "research/nextgen_rankings_protocol.md", root / "protocol.md")
    # A copied manifest must never make an in-progress extension appear complete.
    (root / "manifest.json").unlink()
    observations = gold.manifest["current_observations"]
    season, cutoff = observations["season"], observations["through_week"]
    schedule = pl.concat([gold.read("nfl_schedule"), gold.read("current_schedule")]).unique(
        "game_id"
    )
    panel = make_panel(
        gold.read("preseason_features"),
        pl.read_parquet(profile_root / "nfl_weeks.parquet"),
        schedule,
        gold.read("players"),
        gold.read("college_annual"),
        gold.read("college_identity_links"),
        season=season,
        through_week=cutoff,
    )
    pl.DataFrame(panel, infer_schema_length=None).write_parquet(root / "ranking_panel.parquet")
    predictions, folds = [], []
    for position in POSITIONS:
        for horizon in HORIZONS:
            scoped = [r for r in panel if r["position"] == position and r["horizon"] == horizon]
            earlier = []
            for year in sorted({r["season"] for r in scoped if r["season"] >= 2007}):
                train = [r for r in scoped if r["season"] < year and r["actual"] is not None]
                test = [r for r in scoped if r["season"] == year]
                print(
                    f"Rankings {position} {horizon} {year}: train={len(train)} test={len(test)}",
                    flush=True,
                )
                selections = choose_by_role(earlier, position)
                fits = fit_candidates(train, test)
                fold = dict(
                    position=position,
                    horizon=horizon,
                    season=year,
                    training_first_year=min(r["season"] for r in train),
                    training_last_year=max(r["season"] for r in train),
                    selection_last_year=max((r["season"] for r in earlier), default=None),
                    training_rows=len(train),
                    candidate_rows=len(test),
                    selections_by_role={
                        k: {"reference": v[0], "recipe": v[1]} for k, v in selections.items()
                    },
                )
                folds.append(fold)
                output = []
                for i, row in enumerate(test):
                    reference_recipe, selected = selections[row["role_group"]]
                    values = {model: v[i] for model, v in fits.items()}
                    output.append(
                        dict(
                            **{k: v for k, v in row.items() if not k.startswith(("x_", "e_"))},
                            **values,
                            policy=values[selected],
                            reference=values[reference_recipe],
                            selected_recipe=selected,
                            reference_recipe=reference_recipe,
                        )
                    )
                predictions.extend(output)
                if year < season:
                    earlier.extend(output)
    scopes = evaluate_scopes(predictions)
    by_scope = {(s["position"], s["horizon"]): s for s in scopes}
    current = []
    for r in predictions:
        if r["season"] != season:
            continue
        s = by_scope[r["position"], r["horizon"]]
        approved = s["approved_challenger_policy"] and r["selected_recipe"] not in BASELINES
        recipe = r["selected_recipe"] if approved else r["reference_recipe"]
        current.append(
            dict(
                **{
                    k: r[k]
                    for k in (
                        "player_id",
                        "player_display_name",
                        "position",
                        "population",
                        "team",
                        "season",
                        "through_week",
                        "horizon",
                        "end_week",
                        "scheduled_games",
                        "schedule_known",
                        "prior_weeks",
                        "current_weeks",
                        "current_points",
                    )
                },
                prediction=r[recipe],
                recipe=recipe,
                evidence_status="validated_forecast" if approved else "reference",
                evidence_reason=s["reason"]
                if not approved
                else "Chronological selection policy passed the scoped publication checks",
                entry_id=f"ranking:{r['horizon']}:{r['position']}",
            )
        )
    now = datetime.now(UTC).isoformat()
    news, news_warning = reviewed_news(data, season)
    write_json(
        root / "ranking_availability.json",
        dict(captured_at=now, warning=news_warning, observations=news),
    )
    current = apply_constraints(current, news, now[:10])
    ranking = rank_frame(pl.DataFrame(current, infer_schema_length=None))
    ranking.write_parquet(root / "rankings.parquet")
    pl.DataFrame(predictions, infer_schema_length=None).write_parquet(
        root / "ranking_predictions.parquet"
    )
    write_json(root / "ranking_folds.json", folds)
    write_json(root / "ranking_evaluations.json", scopes)
    registry = [
        r for r in json.loads((root / "registry.json").read_text()) if r["kind"] != "ranking"
    ]
    dependencies = [
        "nfl_player_weeks." + c
        for c in (
            "league_points",
            "passing_yards",
            "rushing_yards",
            "receiving_yards",
            "attempts",
            "carries",
            "targets",
            "receptions",
            "passing_tds",
            "rushing_tds",
            "receiving_tds",
            "offense_snaps",
            "offense_pct",
        )
    ]
    dependencies += [
        "preseason_features",
        "players",
        "college_annual",
        "college_identity_links",
        "nfl_schedule",
        "current_schedule",
        "current_weeks",
        "current_snaps",
    ]
    for scope in scopes:
        subset = [
            r
            for r in current
            if r["position"] == scope["position"] and r["horizon"] == scope["horizon"]
        ]
        validated = any(r["evidence_status"] == "validated_forecast" for r in subset)
        registry.append(
            dict(
                id=f"ranking:{scope['horizon']}:{scope['position']}",
                label=f"NextGen {scope['position']} · {scope['horizon'].replace('_', ' ')}",
                kind="ranking",
                target="league_points",
                horizon=scope["horizon"],
                unit="points",
                positions=[scope["position"]],
                populations=sorted({r["population"] for r in subset}),
                validity="verified",
                evidence="retrospectively_validated" if validated else "reference_baseline",
                serving="approved" if validated else "baseline",
                allowed_uses=["forecast", "ranking"],
                dependencies=dependencies,
                reason=scope["reason"]
                if not validated
                else (
                    "Current-season forecast ordering passed nested chronological checks; "
                    "no market-edge or FAAB claim"
                ),
                decision_date=now[:10],
                analysis_version=version,
                through_week=cutoff,
            )
        )
    # Retain every old model disposition. This extension approves only its own new scopes.
    write_json(root / "registry.json", registry)
    ranking_report = dict(
        season=season,
        through_week=cutoff,
        generated_at=now,
        observations_saved_at=observations["saved_at"],
        horizons=list(HORIZONS),
        candidate_players=ranking["player_id"].n_unique(),
        validated_scopes=[
            s["position"] + ":" + s["horizon"] for s in scopes if s["approved_challenger_policy"]
        ],
        training_first_year=2004,
        forecast_evaluation_years=[2007, season - 1],
        selection_policy_evaluation_years=[2011, season - 1],
        source_analysis=reference(base),
        availability_warning=news_warning,
        rank_meaning=(
            "Predicted league points within the stated horizon; positional ranks, "
            "not FAAB or scarcity-adjusted draft values"
        ),
        limitations=[
            "Retrospective validation; historical source publication vintages are incomplete.",
            "No demonstrated advantage over current expert consensus or waiver prices.",
            "References remain explicit wherever challenger checks fail.",
            "Current injury constraints are dated reports, not a validated medical model.",
            "New roles without observed workload remain uncertain; "
            "college/tracking coverage is partial.",
            "Reference calibration design was informed by the preserved first ranking run.",
            "Role-group recipe design was informed by earlier runs; no prospective claim.",
        ],
    )
    write_json(root / "ranking_report.json", ranking_report)
    report = json.loads((root / "report.json").read_text())
    report.update(version=version, registry_entries=len(registry), rankings=ranking_report)
    write_json(root / "report.json", report)
    for source in (
        Path(__file__),
        ROOT / "src/patron/metrics/current_rankings.py",
        ROOT / "research/nextgen_rankings_protocol.md",
    ):
        destination = root / "implementation" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = dict(
        base_manifest,
        version=version,
        generated_at=now,
        ranking_schema_version=1,
        source_analysis=reference(base),
        files={str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
    )
    manifest.pop("ranking_research", None)
    write_json(root / "manifest.json", manifest)
    print(json.dumps(dict(report=ranking_report, decisions=scopes), indent=2), flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--analysis", help="Verified base analysis, including an unpublished new-gold build"
    )
    args = parser.parse_args()
    build(ROOT / "data", args.version, args.analysis)
