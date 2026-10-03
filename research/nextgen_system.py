"""Build a complete, immutable NextGen analysis release; publish separately."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from engine.data.nextgen import verify_evidence
from engine.data.releases import digest, identifier, load_gold, reference, write_json
from engine.metrics.nextgen import (
    COUNTERS,
    POSITIONS,
    TARGETS,
    build_panel,
    distribution,
    fit_fold,
    paired_summary,
)
from engine.metrics.player_profile import summarize
from engine.metrics.profile_tracking import TRACKING_METRICS, TRACKING_NOTE

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = """# NextGen multi-outcome release protocol

Fixed before this build's fits. All completed, corrected candidates and all earlier
training years are retained. First three candidate years seed training. Outcomes
are season totals, scoring appearances (not medical availability), appearance
probability, and opportunity-conditioned efficiency. Pending outcomes remain null.
Unknown target history and undefined efficiency denominators are not zero labels.

Serving reference: last-season observation, adjusted for nominal season length for
totals, or the earlier position/population mean when no prior observation exists.
Appearance probability always uses the earlier position/population frequency.
This is an explicit baseline, not a promoted predictor or a claimed market edge.
Ridge alpha=100 and histogram boosting (120 iterations, .05 learning rate, 15
leaves, minimum leaf 30, L2=10, 63 bins, no early stopping) are fixed challengers.
Imputation/scaling and feature variation checks use training rows only. Columns
are prior production/exposure and all-career/recent-three-year rates plus age,
draft pick, population, era, observed participation and dated absence facts.
No ECR is used: this evaluates own-data forecasts for each named outcome.

Evaluate all years and modern 2019–2025 separately by position and population.
Continuous targets: equal-season MAE with MSE guardrail. Appearance probability:
Brier score and calibration bins. Report 10,000 season-bootstrap intervals,
leave-one-year-out gains, coverage and extreme extrapolations. These are
exploratory, overlapping comparisons with no multiple-testing promotion claim.
Every challenger remains shadow regardless of the outcome; no historical winner
selection, new blend, calibrated interval or decision-utility claim is authorized.
Existing individual-stat results retain their own inputs, targets and BH-adjusted
screen. Descriptive measurements require valid sources, not predictive lift.

Preseason estimates retain their original forecast cutoff; they are not current
rest-of-season or waiver values. No 2026 outcome is evaluated. Production and
frozen archives remain untouched. Publication is one verified catalog update.
"""


def safe(value):
    return float(value) if value is not None and np.isfinite(value) else None


def measurement_registry():
    definitions = {
        **{
            c: (
                c.replace("_", " ").capitalize(),
                "points" if c == "league_points" else "yards" if "yards" in c else "count",
                "Complete recorded total for this period.",
            )
            for c in COUNTERS
        },
        "observed_weeks": (
            "Observed weeks",
            "weeks",
            "Scoring record or positive offensive snaps; not medical availability.",
        ),
        "snap_share": (
            "Offensive snap share",
            "share",
            "Mean recorded share; missing feed stays unknown.",
        ),
        "median": (
            "Median weekly points",
            "points/week",
            "Median observed scores; zeros and negatives retained.",
        ),
        "mean": ("Mean weekly points", "points/week", "Sum of observed scores / observations."),
        "std": (
            "Weekly standard deviation",
            "points/week",
            "Population SD; at least two observations.",
        ),
        "variance": (
            "Weekly variance",
            "points squared",
            "Population variance; at least two observations.",
        ),
        "cv": (
            "Coefficient of variation",
            "ratio",
            "Population SD / mean, only for a positive mean.",
        ),
        "top2_positive_share": (
            "Best-two-week share",
            "share",
            "Largest two positive scores / all positive scores; at least two observations.",
        ),
        "mean_without_top2": (
            "Mean without best two",
            "points/week",
            "Remove highest two scores; requires at least three observations.",
        ),
        "between_season_std": (
            "Between-season dispersion",
            "points/week",
            "Population SD of completed seasonal mean scoring rates; at least two seasons.",
        ),
        "passing_yards_per_attempt": (
            "Passing yards / attempt",
            "yards/attempt",
            "Ratio on paired observed weeks; sample denominator is retained.",
        ),
        "rushing_yards_per_carry": (
            "Rushing yards / carry",
            "yards/carry",
            "Ratio on paired observed weeks; sample denominator is retained.",
        ),
        "receiving_yards_per_target": (
            "Receiving yards / target",
            "yards/target",
            "Ratio on paired observed weeks; incomplete target periods are identified.",
        ),
    }
    lineage = {
        **{c: ["league_points"] for c in distribution([]) if c != "observations"},
        "between_season_std": ["league_points"],
        "observed_weeks": ["league_points", "offense_snaps"],
        "snap_share": ["offense_snaps", "offense_pct"],
        "passing_yards_per_attempt": ["passing_yards", "attempts"],
        "rushing_yards_per_carry": ["rushing_yards", "carries"],
        "receiving_yards_per_target": ["receiving_yards", "targets"],
    }
    return [
        dict(
            id="measurement:" + key,
            label=label,
            kind="measurement",
            target=None,
            horizon="observed",
            unit=unit,
            definition=definition,
            validity="verified",
            evidence="descriptive",
            serving="approved",
            allowed_uses=["descriptive"],
            positions=list(POSITIONS),
            dependencies=["nfl_player_weeks." + c for c in lineage.get(key, [key])],
            reason="Validated measurement; no predictive or medical-risk claim.",
        )
        for key, (label, unit, definition) in definitions.items()
    ]


def player_measurements(gold, profile_root):
    candidates = gold.read("preseason_features").filter(
        pl.col("forecast_season") == gold.manifest["current_observations"]["season"]
    )
    weekly = pl.read_parquet(profile_root / "nfl_weeks.parquet")
    by_player = defaultdict(list)
    for row in weekly.to_dicts():
        by_player[row["player_id"]].append(row)
    season = gold.manifest["current_observations"]["season"]
    rows = []
    for player in candidates.to_dicts():
        history = by_player[player["player_id"]]
        for period, start, end in (
            ("prior", season - 1, season - 1),
            ("recent3", season - 3, season - 1),
            ("career", 2001, season - 1),
            ("current", season, season),
        ):
            subset = [w for w in history if start <= w["season"] <= end]
            scores = [
                w["league_points"]
                for w in subset
                if w.get("stat_recorded") or (w.get("offense_snaps") or 0) > 0
            ]
            annual = defaultdict(list)
            for w in subset:
                if w.get("league_points") is not None:
                    annual[w["season"]].append(w["league_points"])
            rates = [float(np.mean(v)) for v in annual.values() if v]
            rows.append(
                dict(
                    player_id=player["player_id"],
                    player_display_name=player["player_display_name"],
                    position=player["position"],
                    population=player["player_population"],
                    team=player.get("cutoff_preseason_team"),
                    period=period,
                    first_season=min(annual) if annual else None,
                    last_season=max(annual) if annual else None,
                    through_week=gold.manifest["current_observations"]["through_week"]
                    if period == "current"
                    else 18,
                    **summarize(subset),
                    **distribution(scores),
                    between_season_std=float(np.std(rates)) if len(rates) >= 2 else None,
                    coverage="No observations"
                    if not scores
                    else "Observed records; gaps remain unknown",
                )
            )
    return rows


def archive_inventory(data: Path, current: str, gold, evidence_ref, profile_root):
    """Inventory is archival metadata, never discovery of serving-eligible models."""
    current_dependencies = {gold.root.resolve(), profile_root.resolve()}
    records = []
    for parent, kind in (
        (data / "tables/batches", "data_release"),
        (data / "gold/releases", "data_release"),
        (data / "research", "study"),
    ):
        if not parent.exists():
            continue
        for root in sorted(parent.iterdir()):
            if not root.is_dir() or root.name == current or root.resolve() in current_dependencies:
                continue
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            version = root.name
            validity, evidence = "revalidation_required", "archived"
            reason = "Preserved research/input vintage; not approved for current analysis."
            if kind == "data_release" and version.startswith("canonical_20260923_"):
                validity = "quarantined"
                reason = (
                    "Superseded gold with known invalid receiving-target counts and derivatives. "
                    "Use the corrected current gold; independent historical outcomes are preserved."
                )
            elif version in {"individual_stats_20260923_r1", "profile_history_20260923_r1"}:
                validity = "quarantined"
                reason = (
                    "Target-history defect affected inputs. Results cannot support analysis "
                    "without a corrected rerun; original artifacts are retained."
                )
            elif version == evidence_ref["version"]:
                validity, evidence = "verified", "research_only"
                reason = (
                    "Corrected completed individual-stat screen, retained as scoped research "
                    "evidence. No serving model was approved."
                )
            elif version.startswith("individual_stats_"):
                reason = (
                    "Superseded/incomplete stat run; "
                    "use the completed corrected evidence reference."
                )
            elif version.startswith("nextgen_system_"):
                reason = (
                    "Superseded NextGen build; only the published analysis reference may serve."
                )
            elif version.startswith("canonical_targets_"):
                reason = "Corrected research gold for reproduction; current catalog supersedes it."
            records.append(
                dict(
                    id=f"{kind}:{version}",
                    label=version,
                    kind=kind,
                    target=None,
                    horizon="historical_artifact",
                    unit=None,
                    positions=list(POSITIONS),
                    validity=validity,
                    evidence=evidence,
                    serving="archive",
                    allowed_uses=[],
                    dependencies=[],
                    reason=reason,
                    path=str(root.relative_to(data)),
                    source_manifest_sha256=digest(manifest_path)
                    if manifest_path.exists()
                    else None,
                    original_status=manifest.get("status"),
                    reconsideration=(
                        "Requires a new corrected run and scoped review; preserve this archive."
                    ),
                )
            )
    return records


def build(data: Path, version: str, gold_version: str, profile_version: str, evidence_version: str):
    gold = load_gold(data, gold_version)
    if "target_quality.json" not in gold.manifest["files"]:
        raise ValueError("A canonical target correction is required")
    evidence_ref = reference(data / "research" / identifier(evidence_version))
    evidence_root, evidence = verify_evidence(data, evidence_ref)
    # Preserve the old evidence reference. Only exact consumed-table equality authorizes reuse.
    old_gold = load_gold(data, evidence["gold"]["version"])
    equivalent = []
    for table in (
        "preseason_features",
        "season_outcomes",
        "nfl_player_weeks",
        "nfl_player_seasons",
        "nfl_weekly_usage",
        "nfl_snap_counts",
        "players",
        "college_annual",
        "nfl_injuries",
    ):
        keys = gold.manifest["tables"][table]["primary_key"]
        if not gold.read(table).sort(keys).equals(old_gold.read(table).sort(keys)):
            raise ValueError(f"Reused evidence requires revalidation: {table}")
        equivalent.append(table)
    profile_root = data / "research" / identifier(profile_version)
    if json.loads((profile_root / "manifest.json").read_text())["gold"] != gold.ref:
        raise ValueError("Profile dependency does not match gold")
    root = data / "research" / identifier(version)
    root.mkdir(parents=True, exist_ok=False)
    (root / "protocol.md").write_text(PROTOCOL)
    rows = build_panel(
        gold.read("preseason_features"), gold.read("season_outcomes"), gold.read("nfl_player_weeks")
    )
    columns = sorted(k for k in rows[0] if k.startswith("x_"))
    predictions = []
    fold_records = []
    for year in sorted({r["forecast_season"] for r in rows}):
        if year < 2007:
            continue
        print(f"NextGen fold {year}: all earlier candidate years", flush=True)
        for position in POSITIONS:
            train = [
                r
                for r in rows
                if r["position"] == position
                and r["forecast_season"] < year
                and r["outcome_complete"]
            ]
            test = [r for r in rows if r["position"] == position and r["forecast_season"] == year]
            if not test:
                continue
            for target, (_, unit, positions, _, _) in TARGETS.items():
                if position not in positions:
                    continue
                fits, n = fit_fold(train, test, target, columns)
                # A missing fit is a recorded abstention, never a dropped candidate.
                fitted_models = set(fits)
                for model in ("baseline", "ridge", "boost"):
                    fits.setdefault(model, np.full(len(test), np.nan))
                observed = [r[f"y_{target}"] for r in train if r[f"y_{target}"] is not None]
                lower, upper = (min(observed), max(observed)) if observed else (None, None)
                fold_records.append(
                    dict(
                        season=year,
                        position=position,
                        target=target,
                        train_rows=n,
                        first_train_year=min(
                            (r["forecast_season"] for r in train if r[f"y_{target}"] is not None),
                            default=None,
                        ),
                        candidate_history_first_year=min(r["forecast_season"] for r in train),
                        test_rows=len(test),
                    )
                )
                for model, values in fits.items():
                    for row, value, base in zip(test, values, fits["baseline"], strict=True):
                        predictions.append(
                            dict(
                                player_id=row["player_id"],
                                player_display_name=row["player_display_name"],
                                position=position,
                                population=row["player_population"],
                                season=year,
                                cutoff=row["forecast_cutoff_date"],
                                target=target,
                                unit=unit,
                                model=model,
                                prediction=safe(value),
                                baseline=safe(base),
                                actual=row[f"y_{target}"],
                                complete=row["outcome_complete"],
                                train_rows=n,
                                fitted=model in fitted_models,
                                prior_observed_weeks=row["x_prior_weeks"],
                                career_observed_weeks=row["x_career_weeks"],
                                fallback_source=(
                                    "prior_season"
                                    if row[f"prior_{target}"] is not None
                                    else "earlier_population_mean"
                                    if any(
                                        r["player_population"] == row["player_population"]
                                        and r[f"y_{target}"] is not None
                                        for r in train
                                    )
                                    else "earlier_position_mean"
                                )
                                if model == "baseline"
                                else "fitted_model",
                                uncertainty_status="no_calibrated_interval",
                                outside_training_range=bool(
                                    safe(value) is not None
                                    and lower is not None
                                    and (value < lower or value > upper)
                                ),
                                horizon="preseason_season",
                            )
                        )
    evaluations = []
    registry = measurement_registry()
    registry.extend(
        dict(
            id="measurement:" + key,
            kind="measurement",
            target=None,
            horizon="observed",
            **definition,
            validity="verified",
            evidence="descriptive",
            serving="approved",
            allowed_uses=["descriptive"],
            dependencies=["nfl_player_seasons." + key],
            reason="Verified provider season aggregate for description; no predictive claim.",
            limitations=TRACKING_NOTE,
        )
        for key, definition in TRACKING_METRICS.items()
    )
    for target, (label, unit, positions, _, _) in TARGETS.items():
        for model in ("baseline", "ridge", "boost"):
            for position in positions:
                for window, start in (("all_history", 2007), ("modern", 2019)):
                    group = [
                        r
                        for r in predictions
                        if r["target"] == target
                        and r["model"] == model
                        and r["position"] == position
                        and r["complete"]
                        and r["season"] >= start
                    ]
                    for population in ("all", "returner", "rookie", "market_only"):
                        subset = [
                            r for r in group if population == "all" or r["population"] == population
                        ]
                        stats = paired_summary(subset, probability=target == "season_appearance")
                        evaluations.append(
                            dict(
                                target=target,
                                model=model,
                                position=position,
                                population=population,
                                window=window,
                                **stats,
                            )
                        )
            registry.append(
                dict(
                    id=f"forecast:{model}:{target}",
                    label=f"{label} · {model}",
                    kind="forecast",
                    target=target,
                    horizon="preseason_season",
                    unit=unit,
                    positions=list(positions),
                    validity="verified",
                    evidence="reference_baseline" if model == "baseline" else "exploratory",
                    serving="baseline" if model == "baseline" else "shadow",
                    allowed_uses=["forecast"] if model == "baseline" else [],
                    dependencies=["nfl_player_weeks." + c for c in COUNTERS]
                    + ["preseason_features", "season_outcomes.actual_games"],
                    reason="Declared historical baseline; no market or decision advantage claimed."
                    if model == "baseline"
                    else "Fixed challenger; retrospective evaluation is not serving approval.",
                )
            )
    for target, label in (
        ("starting_status", "Starting status"),
        ("medical_availability", "Medical availability"),
        ("development", "Longer-term development"),
        ("waiver_value", "Waiver / trade value"),
    ):
        registry.append(
            dict(
                id="unavailable:" + target,
                label=label,
                kind="forecast",
                target=target,
                horizon="unvalidated",
                positions=list(POSITIONS),
                unit=None,
                validity="revalidation_required",
                evidence="untested",
                serving="archive",
                allowed_uses=[],
                dependencies=[],
                reason="No validated target-matched model; measurements remain available.",
            )
        )
    screened = pl.read_parquet(evidence_root / "individual_results.parquet")
    findings = {
        r["stat"]: r
        for r in screened.group_by("stat")
        .agg(
            pl.len().alias("comparisons"),
            ((pl.col("action") == "add") & pl.col("passes_screen"))
            .sum()
            .alias("supported_additions"),
            ((pl.col("action") == "remove") & pl.col("passes_screen"))
            .sum()
            .alias("supported_removals"),
        )
        .to_dicts()
    }
    for r in json.loads((evidence_root / "registry.json").read_text()):
        finding = findings.get(r["stat"], {})
        registry.append(
            dict(
                id="stat:" + r["stat"],
                label=r["stat"],
                kind="research_stat",
                target="season_points",
                horizon="preseason_season",
                positions=list(POSITIONS),
                unit=r["unit"],
                validity="verified"
                if r["status"] == "admitted_research_only"
                else "revalidation_required",
                evidence="candidate"
                if finding.get("supported_additions")
                else "scoped_removal_supported"
                if finding.get("supported_removals")
                else "inconclusive"
                if finding
                else "untested",
                serving="archive",
                allowed_uses=[],
                reason=r["reason"],
                definition=r["definition"],
                origins=r["origins"],
                alias_of=r["exact_alias_of"],
                dependencies=[r["dependencies"]],
                source_evidence=evidence_ref,
                source_gold=evidence["gold"],
                screen_summary=finding,
                reconsideration=(
                    "Outcome- and use-matched validation on corrected inputs is required."
                ),
            )
        )
    registry.extend(archive_inventory(data, version, gold, evidence_ref, profile_root))
    config = yaml.safe_load((ROOT / "src/engine/config/metric_report.yaml").read_text())
    for spec in config["fit"]["models"]:
        registry.append(
            dict(
                id="legacy:" + spec["name"],
                label=spec["name"],
                kind="legacy_model",
                target=spec.get("target"),
                horizon="preseason_season",
                positions=list(POSITIONS),
                unit=None,
                validity="revalidation_required",
                evidence="archived",
                serving="archive",
                allowed_uses=[],
                dependencies=["legacy_inputs"],
                reason="Archived input vintage or unvalidated recipe; retained in Research only.",
            )
        )
    measures = player_measurements(gold, profile_root)
    pl.DataFrame(measures, infer_schema_length=None).write_parquet(root / "players.parquet")
    frame = pl.DataFrame(predictions, infer_schema_length=None)
    frame.write_parquet(root / "predictions.parquet")
    frame.filter(~pl.col("complete")).write_parquet(root / "forecasts.parquet")
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(root / "features.parquet")
    # Evidence comparisons retain their original scope and all negative/null results.
    shutil.copy2(evidence_root / "individual_results.parquet", root / "individual_results.parquet")
    for item in registry:
        item["populations"] = ["returner", "rookie", "market_only"]
        item["decision_date"] = "2026-09-23"
        item["analysis_version"] = version
    write_json(root / "registry.json", registry)
    write_json(root / "evaluations.json", evaluations)
    write_json(root / "folds.json", fold_records)
    write_json(
        root / "incidents.json",
        [
            dict(
                id="targets_2003_2008",
                status="resolved",
                dependencies=["nfl_player_weeks.targets"],
                reason="Invalid counters and derivatives are unknown in this gold release.",
            )
        ],
    )
    report = dict(
        version=version,
        gold=gold.ref,
        evidence=evidence_ref,
        evidence_gold=evidence["gold"],
        exact_evidence_table_equivalence=equivalent,
        season=gold.manifest["current_observations"]["season"],
        current_observations=gold.manifest["current_observations"],
        players=len(measures) // 4,
        targets=len(TARGETS),
        registry_entries=len(registry),
        forecast_rows=frame.filter(~pl.col("complete")).height,
        evaluated_rows=frame.filter(pl.col("complete")).height,
        training="all earlier completed candidate years",
        default_model="baseline",
        challengers_promoted=False,
        evaluation_years=[2007, gold.manifest["current_observations"]["season"] - 1],
        training_first_year=min(r["forecast_season"] for r in rows),
        observation_first_year=gold.read("nfl_player_weeks")["season"].min(),
        limitations=[
            "Retrospective evidence; original source publication vintages are incomplete.",
            "Preseason baselines are not rest-of-season or decision values.",
            "Observed weeks and scoring appearances do not establish medical availability.",
            "Unknown target history and undefined denominators remain null.",
            "Multiple outcome comparisons are exploratory; no challenger is promoted.",
        ],
    )
    write_json(root / "report.json", report)
    for path in (
        Path(__file__),
        ROOT / "src/engine/metrics/nextgen.py",
        ROOT / "src/engine/data/nextgen.py",
        ROOT / "src/engine/data/target_quality.py",
        ROOT / "src/engine/metrics/player_profile.py",
        ROOT / "src/engine/metrics/profile_tracking.py",
    ):
        destination = root / "implementation" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    manifest = dict(
        schema_version=1,
        version=version,
        kind="nextgen_analysis",
        status="complete",
        generated_at=datetime.now(UTC).isoformat(),
        gold=gold.ref,
        history=gold.manifest["history"],
        evidence=evidence_ref,
        profiles=reference(profile_root),
        files={str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
    )
    write_json(root / "manifest.json", manifest)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--evidence", default="individual_stats_20260923_r4")
    args = parser.parse_args()
    build(ROOT / "data", args.version, args.gold, args.profiles, args.evidence)
