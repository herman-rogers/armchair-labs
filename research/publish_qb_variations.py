"""Package only approved QB variation policies, retaining every rejected result."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from patron.data.catalog import publish_catalog
from patron.data.nextgen import load_analysis
from patron.data.releases import digest, identifier, inside, reference, write_json
from patron.data.retirements import carry_retirements
from patron.metrics.current_rankings import rank_frame
from patron.metrics.qb_passing import summarize
from patron.metrics.qb_variations import policy_ranges

ROOT = Path(__file__).resolve().parents[1]


def package(data, source_version, version, publish=False, analysis_version=None):
    source = data / "research" / identifier(source_version)
    research = json.loads((source / "manifest.json").read_text())
    if research.get("kind") != "qb_variations_research" or research.get("status") != "complete":
        raise ValueError("Incomplete QB variation experiment")
    for name, expected in research["files"].items():
        if digest(inside(source, name)) != expected:
            raise ValueError("QB experiment changed: " + name)
    ref = reference(data / "research" / identifier(analysis_version)) if analysis_version else None
    base, old_manifest = load_analysis(data, ref)
    if research["source_analysis"] != reference(base):
        raise ValueError("QB experiment and current delivery have different source analyses")
    decisions = json.loads((source / "decisions.json").read_text())
    approved = {(d["target"], d["horizon"]): d for d in decisions if d["approved"]}
    for d in approved.values():
        if d["reasons"] or d["q_value"] > 0.05:
            raise ValueError("Inconsistent QB approval")
    root = data / "research" / identifier(version)
    shutil.copytree(base, root)
    (root / "manifest.json").unlink()
    if (root / "qb_variations").exists():
        shutil.rmtree(root / "qb_variations")
    shutil.copytree(source, root / "qb_variations")
    now = datetime.now(UTC).isoformat()
    experiment = json.loads((source / "report.json").read_text())
    year, cutoff = experiment["season"], experiment["through_week"]
    forecasts = pl.read_parquet(source / "predictions.parquet")
    if forecasts.filter(
        (pl.col("season") == year)
        & (pl.col("actual").is_not_null() | pl.col("actual_points").is_not_null())
    ).height:
        raise ValueError("Current QB forecasts contain future outcome labels")
    for fold in json.loads((source / "folds.json").read_text()):
        if fold["training_last_year"] >= fold["season"] or (
            fold["stack_training_last_year"] is not None
            and fold["stack_training_last_year"] >= fold["season"]
        ):
            raise ValueError("QB variation training crossed a forecast cutoff")
    calibrated = policy_ranges(forecasts.to_dicts())
    pl.DataFrame(calibrated, infer_schema_length=None).write_parquet(
        root / "qb_variations/ranges.parquet"
    )
    ranges = {
        (r["player_id"], r["horizon"]): r
        for r in calibrated
        if r["season"] == year and r["through_week"] == cutoff
    }
    historical = forecasts.filter(pl.col("actual").is_not_null()).to_dicts()
    new_current = {
        (r["player_id"], r["horizon"]): r
        for r in forecasts.filter(
            (pl.col("season") == year) & (pl.col("through_week") == cutoff)
        ).to_dicts()
    }
    current = pl.read_parquet(root / "qb_passing/current.parquet").to_dicts()
    entries = {}
    for r in current:
        new = new_current[r["player_id"], r["horizon"]]
        h = r["horizon"]
        r.update(
            execution_reference=r["execution_ypa"],
            rate_recipe="rate_reference",
            rate_evidence_status="reference",
            production_recipe=r["reference_recipe"],
            rate_entry_id=None,
            experiment_version=source_version,
        )
        if ("rate", h) in approved:
            r.update(
                execution_ypa=new["rate_policy"],
                rate_recipe=new["rate_recipe"],
                rate_evidence_status="retrospectively_validated",
                rate_entry_id=f"qb_passing:rate:{h}",
            )
        if ("yards", h) in approved:
            r.update(
                prediction=new["yards_policy"],
                unconstrained_prediction=new["yards_policy"],
                production_recipe=new["yards_recipe"],
                evidence_status="retrospectively_validated",
                entry_id=f"qb_passing:production:{h}",
            )
            interval = ranges[r["player_id"], h]
            r.update(lower=interval["lower"], upper=interval["upper"])
        # Preserve the same dated retirement/absence constraints on the new forecast.
        if r.get("constraint_reason") or r.get("retired_known"):
            r.update(prediction=0.0, lower=0.0, upper=0.0)
        r["issued_at"] = now
        entries[h] = r["entry_id"]
    if len(current) != len(new_current):
        raise ValueError("Current QB candidate coverage differs")
    pl.DataFrame(current, infer_schema_length=None).write_parquet(
        root / "qb_passing/current.parquet"
    )
    passing_report = json.loads((root / "qb_passing/report.json").read_text())
    passing_report.update(
        generated_at=now,
        current_entries=entries,
        variation_research=source_version,
        production_approved_horizons=[h for t, h in approved if t == "yards"],
        rate_approved_horizons=[h for t, h in approved if t == "rate"],
        ranking_approved_horizons=[h for t, h in approved if t == "league_points"],
        serving="scoped_policies",
        limitations=experiment["limitations"]
        + [
            "Medical availability is separate dated evidence, not a predicted diagnosis.",
            "Passing forecasts use team games; fantasy next-four rankings use calendar weeks.",
            "Original comparisons remain preserved; variations evaluate the new selected policies.",
        ],
    )
    write_json(root / "qb_passing/report.json", passing_report)
    # Keep the original comparisons, adding only approved production policies to analysis.
    evaluations = json.loads((root / "qb_passing/evaluations.json").read_text())
    for h in passing_report["production_approved_horizons"]:
        for window, first in (("all_history", 2007), ("modern", 2019)):
            rows = [
                dict(r, reference=r["yards_reference"], production_policy=r["yards_policy"])
                for r in historical
                if r["horizon"] == h and r["season"] >= first and r["through_week"] > 0
            ]
            result = summarize(rows, "production_policy")
            intervals = [
                r
                for r in calibrated
                if r["horizon"] == h
                and r["through_week"] > 0
                and first <= r["season"] < year
                and r["lower"] is not None
            ]
            result.update(
                horizon=h,
                origin_type="weekly",
                window=window,
                groups=[],
                interval_coverage=float(
                    np.mean([r["lower"] <= r["actual"] <= r["upper"] for r in intervals])
                ),
                interval_width=float(np.mean([r["upper"] - r["lower"] for r in intervals])),
                interval_n=len(intervals),
                evidence="retrospectively_validated",
            )
            evaluations.append(result)
    write_json(root / "qb_passing/evaluations.json", evaluations)
    ranking_current = pl.read_parquet(root / "rankings.parquet").to_dicts()
    rank_forecasts = pl.read_parquet(source / "ranking_predictions.parquet")
    rank_map = {
        (r["player_id"], r["horizon"]): r
        for r in rank_forecasts.filter(pl.col("season") == year).to_dicts()
    }
    changed = 0
    for r in ranking_current:
        if r["position"] != "QB" or ("league_points", r["horizon"]) not in approved:
            continue
        new = rank_map[r["player_id"], r["horizon"]]
        changed += abs(r["unconstrained_prediction"] - new["policy"]) > 1e-8
        r.update(
            unconstrained_prediction=new["policy"],
            prediction=new["policy"] if r["rank_eligible"] else 0.0,
            recipe=new["selected_recipe"],
            evidence_status="validated_forecast",
            evidence_reason=approved["league_points", r["horizon"]]["reason"],
        )
    rank_frame(pl.DataFrame(ranking_current, infer_schema_length=None)).write_parquet(
        root / "rankings.parquet"
    )
    rank_evaluations = json.loads((root / "ranking_evaluations.json").read_text())
    for i, row in enumerate(rank_evaluations):
        d = approved.get(("league_points", row["horizon"])) if row["position"] == "QB" else None
        if d:
            rank_evaluations[i] = dict(
                row,
                modern=d["modern"],
                all_history=d["all_history"],
                cohorts=d["cohorts"],
                q_value=d["q_value"],
                approved_challenger_policy=True,
                reason=d["reason"],
                experiment=source_version,
            )
    write_json(root / "ranking_evaluations.json", rank_evaluations)
    approved_rank_horizons = [h for t, h in approved if t == "league_points"]
    if approved_rank_horizons:
        previous = pl.read_parquet(root / "ranking_predictions.parquet").filter(
            ~((pl.col("position") == "QB") & pl.col("horizon").is_in(approved_rank_horizons))
        )
        replacement = rank_forecasts.filter(pl.col("horizon").is_in(approved_rank_horizons))
        pl.concat([previous, replacement], how="diagonal_relaxed").write_parquet(
            root / "ranking_predictions.parquet"
        )
        bridge_folds = {r["season"]: r for r in json.loads((source / "folds.json").read_text())}
        ranking_folds = json.loads((root / "ranking_folds.json").read_text())
        for fold in ranking_folds:
            if fold["position"] == "QB" and fold["horizon"] in approved_rank_horizons:
                fold["passing_bridge"] = bridge_folds[fold["season"]]
                fold["passing_bridge_source"] = source_version
        write_json(root / "ranking_folds.json", ranking_folds)
    rank_report = json.loads((root / "ranking_report.json").read_text())
    rank_report.update(
        published_at=now,
        qb_passing_integration=[
            dict(horizon=d["horizon"], approved=d["approved"], reason=d["reason"])
            for d in decisions
            if d["target"] == "league_points"
        ],
        qb_passing_research=source_version,
    )
    rank_report["validated_scopes"] = sorted(
        set(rank_report["validated_scopes"])
        | {"QB:" + h for t, h in approved if t == "league_points"}
    )
    write_json(root / "ranking_report.json", rank_report)
    registry = json.loads((root / "registry.json").read_text())
    deps = [
        "nfl_player_weeks.passing_yards",
        "nfl_player_weeks.attempts",
        "nfl_player_weeks.league_points",
        "nfl_player_weeks.completions",
        "nfl_player_weeks.passing_tds",
        "nfl_player_weeks.passing_interceptions",
        "nfl_player_weeks.passing_epa",
        "nfl_player_weeks.passing_cpoe",
        "nfl_player_weeks.rushing_yards",
        "nfl_player_weeks.rushing_tds",
        "nfl_player_weeks.carries",
        "preseason_features",
        "preseason_features.ngs_cpoe",
        "players",
        "college_annual",
        "college_identity_links",
        "nfl_transactions",
        "nfl_schedule",
        "current_schedule",
        "current_weeks",
        "absence_events",
        "qb_passing.events",
        "qb_passing.news",
    ]
    for d in decisions:
        target, h = d["target"], d["horizon"]
        if target == "league_points":
            if d["approved"]:
                entry = next(e for e in registry if e["id"] == f"ranking:{h}:QB")
                entry.update(
                    evidence="retrospectively_validated",
                    serving="approved",
                    reason=d["reason"],
                    dependencies=deps,
                    experiment=source_version,
                )
            continue
        name = "rate" if target == "rate" else "production"
        registry.append(
            dict(
                id=f"qb_passing:{name}:{h}",
                label=f"QB passing {name} · {h}",
                kind="qb_rate" if target == "rate" else "qb_forecast",
                target="passing_efficiency" if target == "rate" else "passing_yards",
                horizon=h,
                positions=["QB"],
                populations=["all"],
                validity="verified",
                evidence="retrospectively_validated" if d["approved"] else "inconclusive_research",
                serving="approved" if d["approved"] else "archive",
                allowed_uses=["forecast"] if d["approved"] else [],
                dependencies=deps,
                reason=d["reason"],
                experiment=source_version,
            )
        )
    registry.append(
        dict(
            id="study:" + source_version,
            label="QB passing variation matrix",
            kind="study",
            target="multiple",
            horizon="mixed",
            positions=["QB"],
            validity="verified",
            evidence="preserved_research",
            serving="archive",
            allowed_uses=[],
            dependencies=[],
            path="research/" + source_version,
            source_manifest_sha256=digest(source / "manifest.json"),
            reason="Complete search retained. Only scoped, approved policies enter analysis.",
        )
    )
    for failed, reason in (
        (
            "qb_variations_20260924_r1",
            "Stopped before fitting: snap-only ranking candidate coverage needed correction.",
        ),
        (
            "qb_variations_20260924_r2",
            "Stopped in first fold: empty tracking columns needed training-only filtering.",
        ),
    ):
        if (data / "research" / failed).exists():
            registry.append(
                dict(
                    id="study:" + failed,
                    label=failed,
                    kind="study",
                    target="multiple",
                    horizon="mixed",
                    positions=["QB"],
                    validity="quarantined",
                    evidence="incomplete_run",
                    serving="archive",
                    allowed_uses=[],
                    dependencies=[],
                    path="research/" + failed,
                    reason=reason,
                )
            )
    registry.append(
        dict(
            id="study:" + old_manifest["version"],
            label=old_manifest["version"],
            kind="study",
            target="multiple",
            horizon="mixed",
            positions=["QB"],
            validity="verified",
            evidence="superseded_delivery",
            serving="archive",
            allowed_uses=[],
            dependencies=[],
            path="research/" + old_manifest["version"],
            source_manifest_sha256=reference(base)["manifest_sha256"],
            reason="Superseded delivery preserved for reproduction; excluded from analysis.",
        )
    )
    for staged in sorted((data / "research").glob("nextgen_qb_variations_*/manifest.json")):
        if staged.parent.name in {version, old_manifest["version"]}:
            continue
        registry.append(
            dict(
                id="study:" + staged.parent.name,
                label=staged.parent.name,
                kind="study",
                target="multiple",
                horizon="mixed",
                positions=["QB"],
                validity="verified",
                evidence="superseded_packaging",
                serving="archive",
                allowed_uses=[],
                dependencies=[],
                path=str(staged.parent.relative_to(data)),
                source_manifest_sha256=digest(staged),
                reason="Superseded delivery packaging; "
                "preserved for reproduction and excluded from current analysis.",
            )
        )
    for e in registry:
        if e["kind"] == "ranking":
            e["analysis_version"] = version
    registry = list({e["id"]: e for e in registry}.values())
    # A new week's observations must not reopen unchanged, already retired recipes.
    # Carry only exclusions, never a prior model's permission to serve predictions.
    prior_root, prior_manifest = load_analysis(data)
    legacy = ROOT / "src/patron/config/metric_report.yaml"
    reviewed_legacy = prior_root / "review_inputs/metric_report.yaml"
    if not reviewed_legacy.exists():
        reviewed_legacy = prior_root / "retirement_inputs/metric_report.yaml"
    registry = carry_retirements(
        registry,
        json.loads((prior_root / "registry.json").read_text()),
        evidence_unchanged=prior_manifest["evidence"] == old_manifest["evidence"],
        legacy_unchanged=reviewed_legacy.exists() and digest(reviewed_legacy) == digest(legacy),
        source=reference(prior_root),
    )
    (root / "retirement_inputs").mkdir(exist_ok=True)
    shutil.copyfile(legacy, root / "retirement_inputs/metric_report.yaml")
    write_json(root / "registry.json", registry)
    report = json.loads((root / "report.json").read_text())
    report["qb_passing"].update(
        serving="scoped_policies",
        variation_research=source_version,
        published_at=now,
        current_entries=entries,
    )
    report.update(
        version=version,
        registry_entries=len(registry),
        rankings=rank_report,
        qb_variations=dict(
            source_version=source_version,
            approved_scopes=experiment["approved_scopes"],
            changed_qb_point_forecasts=changed,
        ),
    )
    write_json(root / "report.json", report)
    for name in (
        "research/publish_qb_variations.py",
        "research/score_qb_variations.py",
        "src/patron/api/qb_passing_routes.py",
        "src/patron/data/nextgen.py",
        "src/patron/data/catalog.py",
        "src/patron/data/retirements.py",
        "src/patron/metrics/qb_variations.py",
        "web/src/components/QBVariations.tsx",
        "web/src/components/QBPassing.tsx",
        "web/src/components/NextGenView.tsx",
        "web/src/components/NextGenRankings.tsx",
        "web/src/api/qbPassing.ts",
        "web/src/api/nextgen.ts",
        "web/tests/qb_passing_smoke.py",
        "tests/test_qb_variations.py",
        "tests/test_qb_variations_scoring.py",
        "tests/test_qb_passing_routes.py",
        "tests/test_ranking_publication.py",
        "docs/research/qb_passing_variations_2026-09-24.md",
    ):
        dest = root / "delivery" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    manifest = dict(
        old_manifest,
        version=version,
        generated_at=now,
        qb_variations_schema_version=1,
        qb_variations_research=reference(source),
        files={str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
    )
    write_json(root / "manifest.json", manifest)
    load_analysis(data, reference(root))
    if publish:
        profile = json.loads(
            (data / "research" / old_manifest["profiles"]["version"] / "report.json").read_text()
        )
        publish_catalog(
            data,
            manifest["gold"]["version"],
            dict(
                college=profile["college_version"],
                profiles=old_manifest["profiles"]["version"],
                analysis=version,
                **{f"outlook_{year}": profile["outlook_version"]},
            ),
        )
    print(
        json.dumps(
            dict(
                version=version,
                published=publish,
                approved=experiment["approved_scopes"],
                changed_qb_points=changed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--analysis-version", help="Verified staged analysis for a new data release"
    )
    args = parser.parse_args()
    package(ROOT / "data", args.source, args.version, args.publish, args.analysis_version)
