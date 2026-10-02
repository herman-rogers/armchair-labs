"""Read-only source audit; writes only separately named research artifacts.

Run: .venv/bin/python research/data_integrity_audit.py
No source refresh, application publication, or 2026 outcome evaluation occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from patron.artifacts import artifact_status, verify_draft
from patron.data.historical_evidence import load_transaction_backfill
from patron.data.market_backfill import extend_crosswalk
from patron.metrics.availability import EVIDENCE_FILE, audit_absence_constraints, load_absences
from patron.metrics.backtest import MetricReportConfig
from patron.metrics.experimental import build_contract_features, build_transaction_features
from patron.metrics.positions import repair_player_week_positions
from patron.metrics.roster_evidence import supplement_identity_names
from patron.metrics.transaction_events import normalize_transaction_sources
from patron.scoring.bonuses import extract_touchdown_bonuses
from patron.scoring.columns import PBP_TOUCHDOWN_COLUMNS
from patron.scoring.engine import score_components

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/nflverse"
DERIVED = ROOT / "data/cache/derived"
OUT = ROOT / "data/outputs"
POSITIONS = ["QB", "RB", "WR", "TE"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog():
    return {p: set(pl.scan_parquet(p).collect_schema().names()) for p in CACHE.glob("*.parquet")}


def select_sources(index, required):
    return sorted(p for p, columns in index.items() if set(required).issubset(columns))


def read_sources(paths):
    assert paths, "Required cached source is absent"
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")


def raw_weeks(index):
    paths = select_sources(index, ["player_id", "fantasy_points_ppr", "sack_fumbles_lost"])
    return read_sources(paths).filter((pl.col("season_type") == "REG") & (pl.col("season") <= 2025))


def duplicate_rows(frame, keys):
    return frame.select(pl.struct(keys).is_duplicated().sum()).item()


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    global CACHE, DERIVED, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--inputs-only", action="store_true", help="Validate before fitting models")
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    CACHE = args.data_dir / "cache/nflverse"
    DERIVED = args.data_dir / "cache/derived"
    OUT = args.data_dir / "outputs"
    repaired = (args.data_dir / "manifest.json").exists()
    index = catalog()
    prediction_path = OUT / (
        "historical_inputs.parquet" if args.inputs_only else "metric_backtest_predictions.parquet"
    )
    watched = [*index, *DERIVED.glob("*.parquet"), prediction_path]
    protected = [
        ROOT / "data/outputs/board.json",
        ROOT / "data/outputs/board_v1.json",
        ROOT / "data/outputs/board_v2.json",
        ROOT / "data/outputs/board_adaptive.json",
        ROOT / "data/static/experimental_2026_predictions.csv",
    ]
    stamps = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in watched}
    protected_hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    pred = pl.read_parquet(prediction_path)
    players = read_sources(select_sources(index, ["gsis_id", "display_name", "short_name"]))
    crosswalk_names = read_sources(select_sources(index, ["gsis_id", "fantasypros_id", "name"]))
    market_identity_path = args.data_dir / "static/market_ecr_backfill.csv"
    if market_identity_path.exists():
        crosswalk_names = extend_crosswalk(
            crosswalk_names, pl.read_csv(market_identity_path, infer_schema_length=0)
        )
    transaction_players = supplement_identity_names(players, crosswalk_names)
    weeks = raw_weeks(index)
    position_repair_count = 0
    if repaired:
        positions = pl.read_parquet(
            next(DERIVED.glob("metric_report_position_history_v1_*.parquet"))
        )
        league = yaml.safe_load(
            (args.data_dir / "implementation/patron/config/league.yaml").read_text()
        )
        weeks = repair_player_week_positions(
            weeks, positions, overrides=league.get("position_overrides")
        )
        position_repair_count = weeks.filter(pl.col("position") != pl.col("raw_position")).height
    skill = weeks.filter(pl.col("position").is_in(POSITIONS))
    scored = score_components(skill).with_columns(
        (pl.col("component_pts") - pl.col("fantasy_points_ppr")).alias("score_delta")
    )
    print("Auditing scoring and bonuses...", flush=True)
    bonus_frames, bonus_checks = [], []
    for path in select_sources(index, ["qb_dropback", "play_id", "touchdown"]):
        plays = pl.scan_parquet(path).select(PBP_TOUCHDOWN_COLUMNS).collect()
        if plays["season"].min() > 2025:
            continue
        bonuses, audit = extract_touchdown_bonuses(plays)
        bonus_frames.append(bonuses)
        bonus_checks.append({"season": int(plays["season"][0]), **asdict(audit)})
    bonuses = pl.concat(bonus_frames, how="diagonal_relaxed")
    unjoined_bonuses = bonuses.join(
        weeks.select("player_id", "season", "week"),
        on=["player_id", "season", "week"],
        how="anti",
    )
    weekly_points = weeks.join(bonuses, on=["player_id", "season", "week"], how="left")
    weekly_points = weekly_points.with_columns(
        (pl.col("fantasy_points_ppr") + pl.col("bonus_pts").fill_null(0)).alias("league_points")
    )
    totals = weekly_points.group_by("player_id", "season").agg(
        pl.col("league_points").sum().alias("recomputed_points"),
        pl.col("position").unique().alias("raw_positions"),
    )
    outcomes = (
        pred.filter(pl.col("outcome_complete"))
        .join(
            totals.rename({"season": "forecast_season"}),
            on=["player_id", "forecast_season"],
            how="left",
        )
        .with_columns(
            (pl.col("actual_season_points") - pl.col("recomputed_points").fill_null(0)).alias(
                "outcome_delta"
            )
        )
    )
    outcome_errors = outcomes.filter(pl.col("outcome_delta").abs() > 0.001)
    # This compact, exactly rescored table is the only outcome input to the pilot.
    weekly_points.select(
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
        "team",
        "carries",
        "targets",
        "receptions",
        "league_points",
        "fantasy_points_ppr",
        "bonus_pts",
    ).write_parquet(OUT / "research_audited_weekly_points.parquet")

    panel_paths = sorted(
        DERIVED.glob(
            "metric_report_rich_weekly_panel_v3_*.parquet"
            if repaired
            else "metric_report_rich_weekly_panel_v2_*.parquet"
        )
    )
    if len(panel_paths) != 1:
        raise ValueError(f"Ambiguous or missing weekly panel: {panel_paths}")
    panel_path = panel_paths[0]
    panel = pl.read_parquet(panel_path)
    coverage = (
        panel.group_by("season")
        .agg(
            pl.len().alias("rows"),
            *[pl.col(c).mean() for c in panel.columns if c.endswith("_available")],
            pl.col("position").is_null().sum().alias("unclassified_rows"),
        )
        .sort("season")
    )
    numeric_checks = {}
    for name in ["snap_pct", "route_participation", "targets_per_route", "air_yards_share"]:
        col = pl.col(name)
        numeric_checks[name] = {
            "null": panel[name].null_count(),
            "nonfinite": panel.select((col.is_not_null() & ~col.is_finite()).sum()).item(),
            "finite_below_zero": panel.select((col.is_finite() & (col < 0)).sum()).item(),
            "finite_above_one": panel.select((col.is_finite() & (col > 1)).sum()).item(),
        }
    # Signed air-yard shares can legitimately be negative or exceed one.
    # Route participation, in contrast, promises a part/whole ratio.

    print("Replaying transaction identity and same-day ordering...", flush=True)
    returners = pred.filter(
        (pl.col("player_population") == "returner") & pl.col("outcome_complete")
    )
    sources = pred.filter(pl.col("player_population") == "returner").select(
        (pl.col("forecast_season") - 1).alias("season"),
        "player_id",
        "player_display_name",
        "team",
        "position",
    )
    espn = pl.read_parquet(next(DERIVED.glob("nfl_transactions_cutoff_v1_*.parquet")))
    official = pl.read_parquet(next(DERIVED.glob("nfl_official_transactions_cutoff_v1_*.parquet")))
    transactions = pl.concat(
        [normalize_transaction_sources(espn), normalize_transaction_sources(official)],
        how="diagonal_relaxed",
    ).sort("transaction_date", "source_team", "description")
    backfill = load_transaction_backfill(args.data_dir)
    if backfill is not None:
        transactions = pl.concat([transactions, backfill], how="diagonal_relaxed").unique(
            subset=["transaction_date", "source_team", "description", "category"]
        )
    cutoffs = {
        r["forecast_season"]: r["forecast_cutoff_date"]
        for r in pred.select("forecast_season", "forecast_cutoff_date").unique().to_dicts()
    }
    candidates = pred.select("player_id", "forecast_season", "player_display_name", "position")
    replay = build_transaction_features(
        transactions,
        sources,
        transaction_players,
        sorted(cutoffs),
        "08-31",
        cutoff_by_season=cutoffs,
        trusted_sources_only=repaired,
        forecast_candidates=candidates,
    )
    reverse = build_transaction_features(
        transactions.reverse(),
        sources,
        transaction_players,
        sorted(cutoffs),
        "08-31",
        cutoff_by_season=cutoffs,
        trusted_sources_only=repaired,
        forecast_candidates=candidates,
    )
    replay_name = (
        "research_transaction_replay.parquet"
        if args.inputs_only
        else "research_final_transaction_replay.parquet"
    )
    replay.sort("forecast_season", "player_id").write_parquet(OUT / replay_name)
    joined = pred.filter(pl.col("forecast_season").is_in(sorted(cutoffs))).join(
        replay, on=["player_id", "forecast_season"], suffix="_replayed"
    )
    changes = {
        c: joined.filter(~pl.col(c).eq_missing(pl.col(c + "_replayed"))).height
        for c in [
            "cutoff_preseason_team",
            "cutoff_preseason_status",
            "cutoff_transaction_count",
            "cutoff_availability_class",
            "cutoff_preseason_status_score",
            "cutoff_state_resolution",
        ]
        if c in returners.columns
    }
    reversed_join = replay.join(reverse, on=["player_id", "forecast_season"], suffix="_reverse")
    order_sensitive = reversed_join.filter(
        pl.any_horizontal(
            ~pl.col(c).eq_missing(pl.col(c + "_reverse"))
            for c in replay.columns
            if c.startswith("cutoff_")
        )
    ).join(
        transaction_players.select(pl.col("gsis_id").alias("player_id"), "display_name"),
        on="player_id",
    )
    identity_examples = joined.filter(
        pl.col("player_display_name").str.contains("Lamar Jackson|Deebo Samuel")
        & pl.col("forecast_season").is_in([2019, 2020, 2021, 2025])
    ).select(
        "forecast_season",
        "player_display_name",
        "cutoff_preseason_team",
        "cutoff_preseason_team_replayed",
        "cutoff_preseason_status",
        "cutoff_preseason_status_replayed",
    )

    contracts = read_sources(select_sources(index, ["year_signed", "apy_cap_pct", "gsis_id"]))
    contract_features = build_contract_features(
        contracts, sorted(cutoffs), cutoff_by_season=cutoffs
    )
    contract_examples = contract_features.join(
        returners.select(
            "forecast_season", "player_id", "player_display_name", "forecast_cutoff_date"
        ),
        on=["forecast_season", "player_id"],
        how="inner",
    ).filter(
        ((pl.col("player_display_name") == "Dak Prescott") & (pl.col("forecast_season") == 2024))
        | (
            (pl.col("player_display_name") == "Ezekiel Elliott")
            & (pl.col("forecast_season") == 2019)
        )
    )
    same_year = (
        contracts.select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("year_signed").alias("forecast_season"),
        )
        .unique()
        .join(
            returners.select("player_id", "forecast_season"),
            on=["player_id", "forecast_season"],
            how="inner",
        )
    )
    snap_paths = select_sources(index, ["pfr_player_id", "offense_snaps", "offense_pct"])
    snaps = read_sources(snap_paths).filter(
        (pl.col("game_type") == "REG")
        & (pl.col("season") <= 2025)
        & pl.col("position").is_in(["QB", "RB", "WR", "TE", "FB", "HB"])
    )
    identity = players.select("pfr_id", "gsis_id").drop_nulls()
    missing_snap_ids = snaps.join(identity, left_on="pfr_player_id", right_on="pfr_id", how="anti")

    adp = pl.read_csv(ROOT / "data/static/market_adp_backfill.csv").filter(pl.col("preferred"))
    snapshot_records = []
    for path in sorted((OUT / "market_snapshots").glob("*.parquet")):
        snap = pl.read_parquet(path)
        snapshot_records.append(
            {
                "file": path.name,
                "rows": snap.height,
                "captured_at": snap["captured_at"].unique().to_list(),
                "duplicate_espn_ids": duplicate_rows(snap, ["espn_id"]),
                "nonnull_adp": snap["espn_adp"].is_not_null().sum(),
            }
        )
    frozen = verify_draft(ROOT / "data/static/draft_2026")
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "audit_implementation_sha256": digest(Path(__file__)),
        "data_dir": str(args.data_dir),
        "repaired_version": repaired,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "implementation_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                ROOT / "src/patron/metrics/experimental.py",
                ROOT / "src/patron/metrics/rich_weekly.py",
                ROOT / "src/patron/scoring/engine.py",
                ROOT / "src/patron/scoring/bonuses.py",
            ]
        },
        "prediction_sha256": digest(prediction_path),
        "weekly_points_sha256": digest(OUT / "research_audited_weekly_points.parquet"),
        "weekly_panel_sha256": digest(panel_path),
        "raw_cache_inventory": stamps,
        "protected_artifact_sha256": protected_hashes,
        "frozen_v1_sha256": digest(frozen),
        "published_status": {p.name: artifact_status(p) for p in protected[:4]},
        "production_model_present": (OUT / "production_model.json").exists(),
        "identity": {
            "historical_position_repairs": position_repair_count,
            "player_ids_duplicated": duplicate_rows(players.drop_nulls("gsis_id"), ["gsis_id"]),
            "pfr_mapping_duplicates": duplicate_rows(identity, ["pfr_id"]),
            "skill_week_duplicate_rows": duplicate_rows(skill, ["player_id", "season", "week"]),
            "prediction_duplicate_rows": duplicate_rows(pred, ["player_id", "forecast_season"]),
            "snap_rows": snaps.height,
            "snap_rows_unmatched": missing_snap_ids.height,
            "unmatched_snap_examples": missing_snap_ids.select(
                "season",
                "player",
                "pfr_player_id",
                "position",
                "offense_snaps",
            )
            .head(12)
            .to_dicts(),
        },
        "scoring": {
            "skill_week_rows": skill.height,
            "parity_mismatches": scored.filter(pl.col("score_delta").abs() > 1e-6).height,
            "maximum_parity_error": scored["score_delta"].abs().max(),
            "bonus_audits": sorted(bonus_checks, key=lambda r: r["season"]),
            "unjoined_bonus_rows": unjoined_bonuses.height,
            "unjoined_bonus_points": unjoined_bonuses["bonus_pts"].sum(),
            "saved_outcome_mismatches": outcome_errors.height,
            "saved_outcome_mismatch_groups": outcome_errors.group_by(
                "player_population",
                "position",
                "raw_positions",
            )
            .len()
            .to_dicts(),
            "saved_outcome_examples": outcome_errors.select(
                "forecast_season",
                "player_display_name",
                "actual_season_points",
                "recomputed_points",
                "outcome_delta",
            )
            .head(15)
            .to_dicts(),
        },
        "rich_panel": {
            "rows": panel.height,
            "duplicate_rows": duplicate_rows(panel, ["player_id", "season", "week"]),
            "coverage_by_season": coverage.to_dicts(),
            "numeric_checks": numeric_checks,
            "points_definition": "Standard PPR, not bonus-inclusive league points",
        },
        "transactions": {
            "all_candidate_rows_replayed": replay.height,
            "changed_from_retained": changes,
            "identity_examples": identity_examples.to_dicts(),
            "same_day_order_sensitive_rows": order_sensitive.select(
                "forecast_season",
                "display_name",
                "cutoff_preseason_team",
                "cutoff_preseason_team_reverse",
                "cutoff_preseason_status",
                "cutoff_preseason_status_reverse",
            ).to_dicts(),
            "future_event_violations": replay.filter(
                pl.col("cutoff_last_transaction_date") > pl.col("forecast_cutoff_date")
            ).height,
            "source_coverage": transactions.group_by("transaction_year", "category")
            .agg(
                pl.len().alias("events"),
                pl.col("source_team").n_unique().alias("teams"),
            )
            .sort("transaction_year", "category")
            .to_dicts(),
        },
        "contracts": {
            "same_year_player_seasons_without_exact_signing_date": same_year.height,
            "previously_leaking_players_after_repair": contract_examples.to_dicts(),
            "unverified_canonical_money": contract_features.filter(
                ~pl.col("contract_point_in_time_verified")
                & pl.col("contract_apy_cap_pct").is_not_null()
            ).height,
        },
        "prices": {
            "preferred_duplicate_keys": duplicate_rows(
                adp.drop_nulls("gsis_id"), ["forecast_season", "gsis_id"]
            ),
            "unresolved_preferred_rows": adp["gsis_id"].null_count(),
            "preferred_sources": adp.group_by("source", "scoring").len().to_dicts(),
            "espn_snapshots": snapshot_records,
            "cutoff_warning": "Final preseason MFL ADP is not an exact-date ECR comparator.",
        },
    }
    report["acceptance_checks"] = {
        "scoring_parity": report["scoring"]["parity_mismatches"] == 0,
        "outcomes_reconcile": outcome_errors.height == 0,
        "forecast_keys_unique": duplicate_rows(pred, ["player_id", "forecast_season"]) == 0,
        "weekly_keys_unique": duplicate_rows(panel, ["player_id", "season", "week"]) == 0,
        "route_ratio_bounded": numeric_checks["route_participation"]["finite_above_one"] == 0
        and numeric_checks["route_participation"]["finite_below_zero"] == 0,
        "nonfinite_signals_absent": all(v["nonfinite"] == 0 for v in numeric_checks.values()),
        "transaction_order_invariant": order_sensitive.height == 0,
        "transaction_replay_matches": all(v == 0 for v in changes.values()) if repaired else False,
        "no_future_transactions": report["transactions"]["future_event_violations"] == 0,
        "no_unverified_contract_money": report["contracts"]["unverified_canonical_money"] == 0,
    }
    if repaired:
        quarantined = [
            name
            for name in pred.columns
            if name
            in {
                "xfp_pg",
                "xfp_total",
                "expected_first_downs_pg",
                "expected_first_downs_total",
            }
            or (name.startswith("rich_") and ("_xfp_" in name or "_expected_td_" in name))
        ]
        report["acceptance_checks"].update(
            quarantined_xfp_absent=all(
                pred[name].null_count() == pred.height for name in quarantined
            ),
            no_legacy_depth_proxy=pred.filter(
                (pl.col("forecast_season") < 2025) & pl.col("depth_chart_rank").is_not_null()
            ).is_empty(),
        )
        report["quarantined_model_columns"] = quarantined
    report["accepted"] = all(report["acceptance_checks"].values())
    absence_path = args.data_dir / "static" / EVIDENCE_FILE
    if absence_path.exists() and "availability_evidence_status" not in pred.columns:
        report["acceptance_checks"]["known_absence_constraints"] = False
        report["accepted"] = False
    elif "availability_evidence_status" in pred.columns:
        game_columns = [
            "expected_games",
            *(
                spec.name
                for spec in MetricReportConfig.from_config().fit.models
                if spec.target == "actual_games"
            ),
        ]
        report["known_absences"] = audit_absence_constraints(
            pred, load_absences(absence_path), game_columns
        )
        report["acceptance_checks"]["known_absence_constraints"] = report["known_absences"][
            "accepted"
        ]
        report["accepted"] = all(report["acceptance_checks"].values())
    for p in watched:
        assert stamps[str(p)] == (p.stat().st_size, p.stat().st_mtime_ns), f"Changed source: {p}"
    assert protected_hashes == {str(p.relative_to(ROOT)): digest(p) for p in protected}
    path = OUT / (
        "historical_input_audit.json"
        if args.inputs_only
        else "data_integrity_audit_2026-09-22.json"
    )
    path.write_text(json.dumps(clean_json(report), indent=2, default=str, allow_nan=False) + "\n")
    print(f"Wrote {path}")
    print("Parity mismatches:", report["scoring"]["parity_mismatches"])
    print("Outcome mismatches:", outcome_errors.height)
    print("Transaction changes:", changes)
    print("Same-day ordering changes:", order_sensitive.height)
    print("Acceptance checks:", report["acceptance_checks"])
    if args.require_clean and not report["accepted"]:
        raise SystemExit("Historical rebuild failed integrity acceptance checks")


if __name__ == "__main__":
    main()
