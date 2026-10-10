"""Build and verify the reusable 2002+ availability evidence dataset.

Run ``python -m engine.data.availability_history.build --help``. Each output
directory is an immutable snapshot; use a new directory for a refresh.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from engine.data import historical_evidence, nfl_transactions
from engine.data.historical_evidence import load_transaction_backfill
from engine.metrics import transaction_events

from . import archive, ledger


def source_frame(root: Path, key: str, *, required=False):
    receipt = json.loads((root / "captures" / f"{key}.json").read_text())
    path = root / "captures" / receipt["file"]
    if archive.digest(path) != receipt["sha256"]:
        raise ValueError(f"Source changed: {key}")
    if receipt["http_status"] != 200:
        if required:
            raise ValueError(f"Required source unavailable: {key}")
        return None, receipt
    return pl.read_parquet(path), receipt


def official_reports(data: Path, first: int, last: int):
    frame = load_transaction_backfill(data)
    if frame is None:
        return [], []
    reports, bindings = [], {}
    manifest = data / "static/historical_backfill_manifest.json"
    bindings[str(manifest.resolve())] = archive.digest(manifest)
    exported = data / "static/historical_transaction_backfill.parquet"
    bindings[str(exported.resolve())] = archive.digest(exported)
    frame = frame.filter(pl.col("transaction_year").is_between(first, last))
    for row in frame.iter_rows(named=True):
        raw = data / "cache/historical_evidence/v1" / row["source_capture_file"]
        meta = raw.with_name(raw.name.removesuffix(".html.gz") + ".json")
        if str(meta.resolve()) not in bindings:
            bindings[str(meta.resolve())] = archive.digest(meta)
            bindings[str(raw.resolve())] = archive.digest(raw)
        receipt = json.loads(meta.read_text())
        reports.append(
            dict(
                event_date=str(row["transaction_date"]),
                description=row["description"],
                team=row["source_team"],
                source="official_club",
                source_url=row["source_url"],
                capture_key=str(raw.resolve()),
                raw_sha256=row["source_sha256"],
                collected_at=receipt["retrieved_at"],
                published_at=None,
            )
        )
    return reports, [dict(path=p, sha256=h) for p, h in sorted(bindings.items())]


def evidence_meta(receipt, source):
    return dict(
        source=source,
        source_url=receipt["url"],
        capture_key=receipt["key"],
        raw_sha256=receipt["sha256"],
        collected_at=receipt["collected_at"],
        published_at=None,
    )


def roster_events(frame, receipt, week_ends):
    rows, counts = [], dict(raw_rows=frame.height, undated=0, missing_identity=0, unknown_status=0)
    status_map = {
        "UFA": "CUT",
        "RFA": "CUT",
        "NWT": "CUT",
        "TRC": "CUT",
        "TRD": "CUT",
        "TRT": "CUT",
        "RSN": "RES",
        "RSR": "CUT",
    }
    states = {"ACT", "RET", "RES", "PUP", "SUS", "DEV", "CUT", "INA", "EXE", "NFI"}
    meta = evidence_meta(receipt, "weekly_roster")
    for record in frame.iter_rows(named=True):
        year, week = int(record["season"]), int(record["week"] or 0)
        end = week_ends.get((year, week)) if record.get("game_type") == "REG" else None
        if not record.get("gsis_id"):
            counts["missing_identity"] += 1
            continue
        if end is None:
            counts["undated"] += 1
            continue
        raw_status = record.get("status")
        status = status_map.get(raw_status, raw_status)
        if status not in states:
            counts["unknown_status"] += 1
            status = "UNKNOWN"
        rows.append(
            ledger.event(
                {**meta, "raw_record_id": ledger.record_id(record)},
                player_id=record["gsis_id"],
                player_name=record.get("full_name"),
                identity_basis="source_gsis_id",
                dimension="roster",
                status=status,
                team=record.get("team"),
                event_date=end,
                usable_at=ledger.next_day(end),
                date_basis="completed_regular_week_plus_next_day_noon",
                season=year,
                week=week,
                description=(
                    f"Roster status={raw_status}; detail={record.get('status_description_abbr')}"
                ),
            )
        )
    return rows, counts


def injury_events(frame, receipt, week_ends=None):
    rows, counts = [], dict(raw_rows=frame.height, undated=0, missing_identity=0, week_proxy=0)
    meta = evidence_meta(receipt, "injury_report")
    for record in frame.iter_rows(named=True):
        modified = ledger.stamp(record.get("date_modified"))
        basis = "source_modification_timestamp"
        if not record.get("gsis_id"):
            counts["missing_identity"] += 1
            continue
        if modified is None:
            kind = record.get("game_type") or record.get("season_type")
            end = (week_ends or {}).get((int(record["season"]), int(record["week"])))
            if kind == "REG" and end is not None:
                modified = ledger.next_day(end)
                basis = "completed_regular_week_proxy_no_publication_timestamp"
                counts["week_proxy"] += 1
            else:
                counts["undated"] += 1
                continue
        rows.append(
            ledger.event(
                {**meta, "raw_record_id": ledger.record_id(record)},
                player_id=record["gsis_id"],
                player_name=record.get("full_name"),
                identity_basis="source_gsis_id",
                dimension="injury_report",
                status=record.get("report_status") or "UNKNOWN",
                team=record.get("team"),
                event_date=modified.date(),
                usable_at=modified,
                # Modification is distinct from initial publication, which is unknown.
                date_basis=basis,
                season=int(record["season"]),
                week=int(record["week"]),
                description=json.dumps(record, default=str, sort_keys=True),
            )
        )
    return rows, counts


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if archive.digest(root / name) != expected:
            raise ValueError(f"Dataset changed: {name}")
    for record in manifest["captures"]:
        receipt = root / "captures" / f"{record['key']}.json"
        if archive.digest(receipt) != record["receipt_sha256"]:
            raise ValueError(f"Capture receipt changed: {record['key']}")
        if archive.digest(root / "captures" / record["file"]) != record["sha256"]:
            raise ValueError(f"Capture changed: {record['key']}")
    for item in manifest["external_inputs"]:
        if archive.digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"External evidence changed: {item['path']}")
    return manifest


def build(root: Path, data: Path, first: int, last: int):
    if (root / "manifest.json").exists():
        manifest = verify(root)
        if (first, last) != (manifest["first_year"], manifest["last_year"]):
            raise ValueError("Snapshot covers different requested years")
        print("Existing snapshot verified", flush=True)
        return manifest
    code_paths = [Path(__file__)] + [
        Path(module.__file__)
        for module in (archive, ledger, historical_evidence, nfl_transactions, transaction_events)
    ]
    code_hashes = {str(p.resolve()): archive.digest(p) for p in code_paths}
    # Fetch is resumable and validates all cached bodies before reuse.
    reports, coverage = archive.capture_sources(root / "captures", first, last)
    official, bindings = official_reports(data, first, last)
    reports.extend(official)
    for report in reports:
        report["raw_record_id"] = ledger.record_id(report)
    archive.write_json(root / "transaction_reports.json", reports)
    players, _ = source_frame(root, "players", required=True)
    players.select(
        "gsis_id",
        *[
            c
            for c in ("display_name", "football_name", "birth_date", "espn_id", "pfr_id")
            if c in players.columns
        ],
    ).write_parquet(root / "identity.parquet")
    schedule_path = root / "captures/schedule.bin"
    schedule_receipt = json.loads((root / "captures/schedule.json").read_text())
    if (
        schedule_receipt["http_status"] != 200
        or archive.digest(schedule_path) != schedule_receipt["sha256"]
    ):
        raise ValueError("Schedule unavailable or changed")
    schedule = pl.read_csv(schedule_path, infer_schema_length=10000).filter(
        pl.col("game_type") == "REG"
    )
    week_ends = {
        (int(y), int(w)): date.fromisoformat(d)
        for y, w, d in schedule.group_by("season", "week").agg(pl.col("gameday").max()).iter_rows()
    }
    all_rows, roster_frames, source_counts = [], [], []
    for year in range(first, last + 1):
        roster, rec = source_frame(root, f"roster_{year}")
        if roster is not None:
            if set(roster["season"].drop_nulls().unique()) != {year}:
                raise ValueError(f"Wrong roster source year: {year}")
            roster_frames.append(roster)
            rows, counts = roster_events(roster, rec, week_ends)
            all_rows.extend(rows)
            source_counts.append(
                dict(year=year, source="weekly_roster", **counts, usable_events=len(rows))
            )
        if year >= 2009:
            injuries, rec = source_frame(root, f"injuries_{year}")
            if injuries is not None:
                if set(injuries["season"].drop_nulls().unique()) != {year}:
                    raise ValueError(f"Wrong injury source year: {year}")
                rows, counts = injury_events(injuries, rec, week_ends)
                all_rows.extend(rows)
                source_counts.append(
                    dict(year=year, source="injury_report", **counts, usable_events=len(rows))
                )
        print(f"Normalized roster/injury year {year}", flush=True)
    if not roster_frames:
        raise ValueError("No roster sources available")
    rosters = pl.concat(roster_frames, how="diagonal_relaxed")
    names, display = ledger.identity_index(players, rosters)
    transaction_rows, rejected = ledger.transaction_events(
        reports, names, display, ledger.prior_roster_index(all_rows)
    )
    all_rows.extend(transaction_rows)
    events = (
        pl.DataFrame(all_rows, schema=ledger.SCHEMA)
        .unique("event_id")
        .sort("usable_at", "event_id")
    )
    # Ignore observations after their actual capture, e.g. a not-yet-completed week.
    future = events.filter(pl.col("usable_at") > pl.col("collected_at"))
    events = events.filter(pl.col("usable_at") <= pl.col("collected_at"))
    if events["player_id"].null_count() or events["collected_at"].null_count():
        raise ValueError("Evidence missing identity or collection timestamp")
    events.write_parquet(root / "events.parquet")
    future.write_parquet(root / "pending_events.parquet")
    pl.DataFrame(
        rejected, schema={"raw_record_id": pl.String, "reason": pl.String, "name": pl.String}
    ).write_parquet(root / "review_queue.parquet")
    archive.write_json(
        root / "ambiguous_names.json", {n: sorted(ids) for n, ids in names.items() if len(ids) > 1}
    )
    coverage_frame = (
        events.with_columns(pl.col("event_date").dt.year().alias("calendar_year"))
        .group_by("calendar_year", "source", "dimension")
        .agg(
            pl.len().alias("events"),
            pl.col("player_id").n_unique().alias("players"),
            pl.col("event_date").min().alias("first_date"),
            pl.col("event_date").max().alias("last_date"),
        )
    )
    coverage_frame.sort("calendar_year", "source", "dimension").write_csv(root / "coverage.csv")
    # A complete list of expected team-weeks exposes holes; no-event weeks are not healthy weeks.
    expected = (
        pl.concat(
            [
                schedule.select("season", "week", pl.col(c).alias("team"))
                for c in ("home_team", "away_team")
            ]
        )
        .unique()
        .filter(pl.col("season").is_between(first, last))
        .with_columns(pl.col("team").replace(ledger.TEAM_ALIASES))
    )
    calendars = pl.DataFrame(
        [
            dict(
                season=y,
                week=w,
                week_end=d,
                completed_at_build=ledger.next_day(d) <= datetime.now(UTC),
            )
            for (y, w), d in week_ends.items()
        ]
    )
    expected = expected.join(calendars, on=["season", "week"], how="left")
    observed = (
        events.filter(pl.col("source") == "weekly_roster")
        .group_by("season", "week", "team")
        .agg(pl.col("player_id").n_unique().alias("roster_players"))
    )
    expected.join(observed, on=["season", "week", "team"], how="left").sort(
        "season", "week", "team"
    ).write_csv(root / "team_week_coverage.csv")
    conflicts = (
        events.group_by("player_id", "dimension", "usable_at")
        .agg(
            pl.col("status").n_unique().alias("status_count"),
            pl.col("team").drop_nulls().n_unique().alias("team_count"),
            pl.col("event_id").alias("evidence_ids"),
        )
        .filter((pl.col("status_count") > 1) | (pl.col("team_count") > 1))
    )
    conflicts.write_parquet(root / "conflicts.parquet")
    archive.write_json(
        root / "source_coverage.json", dict(acquisition=coverage, normalization=source_counts)
    )
    captures = []
    for path in sorted((root / "captures").glob("*.json")):
        rec = json.loads(path.read_text())
        if archive.digest(root / "captures" / rec["file"]) != rec["sha256"]:
            raise ValueError(f"Capture changed: {path}")
        captures.append({**rec, "receipt_sha256": archive.digest(path)})
    manifest = dict(
        schema_version=1,
        first_year=first,
        last_year=last,
        built_at=datetime.now(UTC).isoformat(),
        events=events.height,
        players=events["player_id"].n_unique(),
        transaction_reports=len(reports),
        transaction_events=len(transaction_rows),
        review_items=len(rejected),
        conflicting_groups=conflicts.height,
        pending_events=future.height,
        captures=captures,
        external_inputs=bindings,
        files={
            p.name: archive.digest(p)
            for p in sorted(root.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
        source_code=code_hashes,
        limitations=[
            "Historical states are retrospective reconstructions, "
            "not archived publication-time vintages.",
            "Observed mode requires collection by cutoff; reconstructed mode is explicit opt-in.",
            "Reported pages are fully retrieved when flagged; "
            "source events themselves may be incomplete.",
            "Ambiguous identities and unsupported actions are retained for review, not guessed.",
            "Season-level rosters, future appearances, and career end dates "
            "never establish availability.",
            "Missing injury reports are unknown. "
            "Injury-report history is separate from roster status.",
            "Source state conflicts remain unresolved; no model eligibility rule is imposed.",
        ],
    )
    if any(archive.digest(Path(p)) != h for p, h in code_hashes.items()):
        raise ValueError("Source code changed during build; rebuild before sealing")
    archive.write_json(root / "manifest.json", manifest)
    verify(root)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "events",
                    "players",
                    "transaction_reports",
                    "transaction_events",
                    "review_items",
                    "conflicting_groups",
                    "pending_events",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--first-year", type=int, default=2002)
    parser.add_argument("--last-year", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
        print("Availability dataset verified")
    else:
        build(args.output, args.data_dir, args.first_year, args.last_year)


if __name__ == "__main__":
    main()
