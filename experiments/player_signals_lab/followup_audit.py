"""Evidence-backed boundaries for every data-limited research path, plus closure tests."""

from __future__ import annotations

import json

import followup_closure
import numpy as np
import polars as pl
from followup_common import REPO
from safety import write_json


def discrepancy(actual, reconstructed, tolerance):
    a, b = np.asarray(actual, float), np.asarray(reconstructed, float)
    valid = np.isfinite(a) & np.isfinite(b)
    differences = np.abs(a[valid] - b[valid])
    return dict(
        n=int(valid.sum()),
        tolerance=tolerance,
        matched_fraction=float((differences <= tolerance).mean()) if len(differences) else None,
        median_absolute_difference=float(np.median(differences)) if len(differences) else None,
        max_absolute_difference=float(differences.max()) if len(differences) else None,
    )


def ngs_audit(inputs):
    providers = inputs.raw_providers()
    gold = inputs.gold("nfl_player_seasons")
    results = {}
    for name, count, gold_count in (
        ("passing", "attempts", "attempts"),
        ("receiving", "targets", "targets"),
        ("rushing", "rush_attempts", "carries"),
    ):
        frame = providers[name].filter((pl.col("week") == 0) & (pl.col("season_type") == "REG"))
        joined = frame.select(
            "player_gsis_id", "season", pl.col(count).alias("provider_count")
        ).join(
            gold.select("player_id", "season", pl.col(gold_count).alias("gold_count")),
            left_on=["player_gsis_id", "season"],
            right_on=["player_id", "season"],
            how="left",
            validate="m:1",
        )
        result = dict(
            rows=frame.height,
            min_season=frame["season"].min(),
            max_season=frame["season"].max(),
            missing_gold_count=joined["gold_count"].null_count(),
            published_vs_gold_count=discrepancy(joined["provider_count"], joined["gold_count"], 0),
            per_season=frame.group_by("season").len().sort("season").to_dicts(),
        )
        if name == "passing":
            result["cpoe_identity"] = discrepancy(
                frame["completion_percentage_above_expectation"],
                frame["completion_percentage"] - frame["expected_completion_percentage"],
                0.02,
            )
            result["completion_count_identity"] = discrepancy(
                frame["completion_percentage"], 100 * frame["completions"] / frame["attempts"], 0.02
            )
        elif name == "receiving":
            result["yac_identity"] = discrepancy(
                frame["avg_yac_above_expectation"],
                frame["avg_yac"] - frame["avg_expected_yac"],
                0.02,
            )
            result["catch_count_identity"] = discrepancy(
                frame["catch_percentage"], 100 * frame["receptions"] / frame["targets"], 0.02
            )
        else:
            result["ryoe_count_identity"] = discrepancy(
                frame["rush_yards_over_expected_per_att"],
                frame["rush_yards_over_expected"] / frame["rush_attempts"],
                0.02,
            )
        results[name] = result
    return results


def source_inventory(inputs):
    manifest = json.loads(
        inputs.bind(REPO / "data/raw/snapshots/canonical_20260923_r4/manifest.json").read_text()
    )
    records = []
    for key, asset in manifest["assets"].items():
        if not key.startswith("history/cache/nflverse/") or not key.endswith(".parquet"):
            continue
        path = inputs.bind(REPO / "data" / asset["object"], asset["sha256"])
        cols = list(pl.read_parquet_schema(path))
        lower = {c.lower() for c in cols}
        records.append(
            dict(
                logical_key=key,
                object=asset["object"],
                columns=cols,
                trajectories={"x", "y"}.issubset(lower) and bool({"frame_id", "frameid"} & lower),
                primary_route_field="route" in lower,
                pressure_field="was_pressure" in lower,
            )
        )
    return dict(
        nfl_provider_tables=len(records),
        trajectory_tables=sum(r["trajectories"] for r in records),
        route_field_tables=sum(r["primary_route_field"] for r in records),
        pressure_field_tables=sum(r["pressure_field"] for r in records),
        tables=records,
        limitation=(
            "Inventory of this pinned NFL provider snapshot, not all public or proprietary "
            "data. Primary-receiver route labels do not enumerate every receiver's routes. "
            "Pressure flags alone do not identify contextual player effects or "
            "alternative target decisions."
        ),
    )


def college_audit(inputs):
    annual = inputs.gold("college_annual")
    links = inputs.gold("college_identity_links")
    rosters = []
    for year in (2004, 2014, 2025):
        path = inputs.bind(
            REPO / f"data/research/college_source_20260922_r1/raw/rosters_{year}.parquet"
        )
        frame = pl.read_parquet(path, columns=["experience_years", "draft_year"])
        rosters.append(
            dict(
                season=year,
                rows=frame.height,
                experience_observed=frame["experience_years"].is_not_null().sum(),
                later_draft_year_recorded=frame.filter(pl.col("draft_year") > year + 1).height,
            )
        )
    return dict(
        annual_rows=annual.height,
        identity_status=links.group_by("status").len().to_dicts(),
        annual_coverage=annual.group_by("season")
        .agg(
            pl.len(),
            pl.col("complete_team_season").mean().alias("complete_fraction"),
            pl.col("coverage").mean(),
        )
        .sort("season")
        .to_dicts(),
        gold_exit_fields=[
            c
            for c in annual.columns
            if any(w in c for w in ("declaration", "graduation", "eligibility", "exit_date"))
        ],
        raw_roster_audit=rosters,
        limitation=(
            "Experience fields exist, but no independently verified complete exit "
            "population/nonentry-negative ledger. Unmatched and review identities remain "
            "unknown. NFL-selected rookie feature ablations cannot estimate NFL entry "
            "probability. Raw biographies can contain later draft information."
        ),
    )


def decision_audit(inputs):
    snapshots = []
    for path in sorted((REPO / "data/research").glob("*/league_snapshot.json")):
        data = json.loads(inputs.bind(path).read_text())
        snapshots.append(
            dict(
                path=str(path.relative_to(REPO)),
                season=data.get("season"),
                week=data.get("week"),
                captured_at=data.get("captured_at"),
                draft_entries=len(data.get("draft", [])),
                transactions=len(data.get("transactions", [])),
                week_lineups=len(data.get("week_lineups", [])),
                draft_fields=sorted(data["draft"][0]) if data.get("draft") else [],
                transaction_fields=sorted(data["transactions"][0])
                if data.get("transactions")
                else [],
            )
        )
    replay = inputs.bind(
        REPO / "data/research/historical_v2_20260922_r5/outputs/research_transaction_replay.parquet"
    )
    return dict(
        league_snapshots=snapshots,
        historical_replay_fields=list(pl.read_parquet_schema(replay)),
        limitation=(
            "Real 2026 draft/transaction/lineup records exist, but they are current-season "
            "snapshots after the draft and only early in the season. They do not supply "
            "decision-time feasible alternatives and completed outcomes for the 2007–2025 "
            "forecast history. Historical transaction replay here is NFL roster evidence, "
            "not executed fantasy trades. Fixed-slot policies therefore remain cost-free "
            "diagnostics, not profit/decision-value validation."
        ),
    )


def run(inputs, root):
    report = dict(
        ngs=ngs_audit(inputs),
        tracking=source_inventory(inputs),
        college=college_audit(inputs),
        decisions=decision_audit(inputs),
    )
    injuries = inputs.gold("nfl_injuries")
    report["injury_capture"] = dict(
        year_counts=injuries.group_by("season").len().sort("season").to_dicts(),
        missing_timestamp=injuries["date_modified"].null_count(),
        missing_player_id=injuries["gsis_id"].null_count(),
        limitation=(
            "This local pinned table includes 2025 reports. Earlier prose saying injuries "
            "stop in 2024 does not describe this capture. Reported status and offensive-snap "
            "participation are not a medical surveillance outcome."
        ),
    )
    report["source_definitions"] = [
        "https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html",
        "https://nflreadr.nflverse.com/articles/dictionary_participation.html",
        "https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html",
    ]
    report["ngs_limitations"] = (
        "Published counts are useful exposure proxies, not proof that every metric uses all those "
        "plays. Numerical reconciliation is separate from tracking eligibility and original "
        "expectation-model training vintages; the latter are unavailable in this snapshot. "
        "All NGS efficacy results are retrospective."
    )
    write_json(root / "data_path_audit.json", report)
    print("source/identity/decision audits finished", flush=True)
    return followup_closure.run(inputs, root)
