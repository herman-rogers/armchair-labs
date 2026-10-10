"""Independent ledger clocks, identity support, coverage, and as-of smoke audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .archive import digest, write_json
from .build import verify
from .ledger import as_of


def audit(root: Path) -> dict:
    manifest = verify(root)
    events = pl.read_parquet(root / "events.parquet")
    pending = pl.read_parquet(root / "pending_events.parquet")
    if events["event_id"].n_unique() != events.height:
        raise ValueError("Duplicate evidence IDs")
    if events.join(pending, on="event_id", how="inner").height:
        raise ValueError("Pending evidence leaked into ledger")
    if events.filter(pl.col("usable_at") > pl.col("collected_at")).height:
        raise ValueError("Ledger contains not-yet-usable evidence")
    transactions = events.filter(pl.col("source").is_in(["espn", "official_club"]))
    expected_clock = pl.col("event_date").cast(pl.Datetime("us", "UTC")) + pl.duration(
        days=1, hours=12
    )
    if transactions.filter(pl.col("usable_at") != expected_clock).height:
        raise ValueError("Transaction clock mismatch")
    raw = pl.DataFrame(json.loads((root / "transaction_reports.json").read_text())).select(
        "raw_record_id",
        pl.col("raw_sha256").alias("source_hash"),
    )
    linked = transactions.join(raw, on="raw_record_id", how="left")
    if linked.filter(
        pl.col("source_hash").is_null() | (pl.col("source_hash") != pl.col("raw_sha256"))
    ).height:
        raise ValueError("Transaction not linked to its raw evidence")
    schedule = pl.read_csv(root / "captures/schedule.bin", infer_schema_length=10000).filter(
        pl.col("game_type") == "REG"
    )
    ends = schedule.group_by("season", "week").agg(
        pl.col("gameday").str.to_date().max().alias("last_game")
    )
    week_based = events.filter(pl.col("date_basis").str.starts_with("completed_regular_week"))
    joined = week_based.join(ends, on=["season", "week"], how="left")
    calendar_clock = pl.col("last_game").cast(pl.Datetime("us", "UTC")) + pl.duration(
        days=1, hours=12
    )
    if joined.filter(
        pl.col("last_game").is_null() | (pl.col("usable_at") != calendar_clock)
    ).height:
        raise ValueError("Weekly evidence precedes the completed-week calendar")
    # Independently verify disambiguation support, including both evidence clocks.
    supported = events.filter(pl.col("identity_evidence_id").is_not_null())
    supports = events.select(
        pl.col("event_id").alias("identity_evidence_id"),
        pl.col("player_id").alias("support_player"),
        pl.col("team").alias("support_team"),
        pl.col("usable_at").alias("support_time"),
        pl.col("collected_at").alias("support_capture"),
        pl.col("source").alias("support_source"),
    )
    joined = supported.join(supports, on="identity_evidence_id", how="left")
    if joined.filter(
        pl.col("support_time").is_null()
        | (pl.col("support_time") > pl.col("usable_at"))
        | (pl.col("support_capture") > pl.col("collected_at"))
        | (pl.col("support_player") != pl.col("player_id"))
        | (pl.col("support_source") != "weekly_roster")
    ).height:
        raise ValueError("Identity inference uses missing, wrong, or future evidence")
    coverage = json.loads((root / "source_coverage.json").read_text())
    espn = [r for r in coverage["acquisition"] if r.get("source") == "espn"]
    teamweeks = pl.read_csv(root / "team_week_coverage.csv")
    completed = teamweeks.filter(pl.col("completed_at_build"))
    holes = completed.filter(pl.col("roster_players").is_null())
    review = pl.read_parquet(root / "review_queue.parquet")
    cases = []
    for year, month, day in [(2021, 9, 9), (2022, 9, 8)]:
        cutoff = datetime(year, month, day, 12, tzinfo=UTC)
        ids = ["00-0020531", "00-0019596"]
        full = as_of(events, cutoff, mode="reconstructed", player_ids=ids)
        prior_only = as_of(
            events.filter(pl.col("usable_at") <= cutoff),
            cutoff,
            mode="reconstructed",
            player_ids=ids,
        )
        if not full.equals(prior_only):
            raise ValueError("Future evidence altered a past query")
        roster_states = dict(
            full.filter(pl.col("dimension") == "roster").select("player_id", "status").iter_rows()
        )
        if roster_states.get("00-0020531") != "RET" or roster_states.get("00-0019596") == "RET":
            raise ValueError("Brees/Brady retirement regression failed")
        observed = as_of(events, cutoff, player_ids=ids)
        if observed.filter(pl.col("status") != "UNKNOWN").height:
            raise ValueError("A later capture was treated as historical observed evidence")
        cases.extend(
            full.select("player_id", "dimension", "status", "team", "age_days")
            .with_columns(pl.lit(str(cutoff)).alias("cutoff"))
            .to_dicts()
        )
    return dict(
        manifest_sha256=digest(root / "manifest.json"),
        audit_code_sha256=digest(Path(__file__)),
        status="passed",
        events=events.height,
        players=events["player_id"].n_unique(),
        first_event=str(events["event_date"].min()),
        last_event=str(events["event_date"].max()),
        transaction_reports=manifest["transaction_reports"],
        complete_espn_years=[r["year"] for r in espn if r["status"] == "all_reported_pages"],
        incomplete_espn_years=[r for r in espn if r["status"] != "all_reported_pages"],
        completed_team_weeks=completed.height,
        missing_completed_team_weeks=holes.to_dicts(),
        pending_events=pending.height,
        context_resolved_events=supported.height,
        review_reasons=review.group_by("reason").len().sort("reason").to_dicts(),
        same_time_conflicts=manifest["conflicting_groups"],
        evidence_by_source=events.group_by("source", "dimension")
        .len()
        .sort("source", "dimension")
        .to_dicts(),
        date_bases=events.group_by("source", "date_basis")
        .len()
        .sort("source", "date_basis")
        .to_dicts(),
        regression_cases=cases,
        interpretation="Team-week coverage does not prove complete player or transaction coverage.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "events",
                    "players",
                    "completed_team_weeks",
                    "context_resolved_events",
                )
            },
            indent=2,
        )
    )
