"""Observation ingestion: register captured sources → enrich with DuckDB → build a table batch.

Local, single-writer Parquet pipeline. No network is used after source registration.
The batch holds observation tables only; models read them through DuckDB and
publish their own campaign bundles. `engine data refresh --upload` captures the
latest completed week and runs this end to end; `engine tables refresh --upload`
rebuilds recipes from an already built batch. Enrichment lives in
``engine.data.enrich``: one SQL recipe per table over the preserved provider files.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path

import polars as pl

from engine.config.settings import REPO_ROOT
from engine.data.enrich import enrich
from engine.data.releases import (
    digest,
    identifier,
    inside,
    load_manifest,
    load_tables,
    preserve_object,
    reference,
    write_json,
)
from engine.tables.contracts import clean_floats, table_spec

ROOT = REPO_ROOT


def now():
    return datetime.now(UTC).isoformat()


def register_sources(data: Path, args) -> Path:
    from engine.data.history import accepted_version

    history, acceptance, old_manifest = accepted_version(args.history)
    root = data / "raw/snapshots" / identifier(args.version)
    root.mkdir(parents=True, exist_ok=False)
    assets = {}

    def add(name, path, kind, expected=None):
        expected = expected or digest(path)
        assets[name] = {
            "object": preserve_object(data, path, expected),
            "sha256": expected,
            "bytes": path.stat().st_size,
            "origin_path": str(path.relative_to(data)) if path.is_relative_to(data) else str(path),
            "kind": kind,
        }

    for name, expected in old_manifest["source_sha256"].items():
        if name.startswith("cache/nflverse/"):
            kind = "provider_cache_snapshot"
        elif name.startswith("cache/historical_evidence/"):
            kind = "captured_web_evidence"
        elif name.endswith("historical_absences.json"):
            kind = "reviewed_annotation"
        elif name.startswith("cache/derived/") or name.endswith(".parquet"):
            kind = "legacy_enriched_import"
        else:
            kind = "reference_input"
        add("history/" + name, history / name, kind, expected)
    configuration = history / "implementation/engine/config"
    if not configuration.is_dir():
        # Accepted archives created before the package rename retain their paths.
        configuration = history / "implementation/patron/config"
    for path in sorted(configuration.glob("*.yaml")) if configuration.is_dir() else []:
        add("configuration/" + path.name, path, "configuration")
    add("checks/history_manifest.json", history / "manifest.json", "accepted_replay_reference")
    add("checks/history_acceptance.json", history / "acceptance.json", "accepted_replay_reference")

    college = data / "research" / identifier(args.college_source)
    cm = json.loads((college / "manifest.json").read_text())
    if cm["status"] != "captured" or cm["kind"] != "college_source":
        raise ValueError("College source is not a completed capture")
    add("college/manifest.json", college / "manifest.json", "capture_metadata")
    for spec in cm["files"]:
        add(
            "college/raw/" + spec["filename"],
            inside(college / "raw", spec["filename"]),
            "provider_cache_snapshot",
            spec["sha256"],
        )

    report_path = args.current_report.resolve()
    report = json.loads(report_path.read_text())
    snapshot = inside(report_path.parent, report["snapshot_path"])
    add("current/source_report.json", report_path, "capture_metadata")
    for name in ("weeks", "snaps", "schedule", "players"):
        filename = name + ".parquet"
        add(
            "current/" + filename,
            snapshot / filename,
            "legacy_enriched_import",
            report["source_sha256"][filename],
        )
    manifest = {
        "schema_version": 1,
        "layer": "raw",
        "version": args.version,
        "status": "accepted",
        "registered_at": now(),
        "history": acceptance["audited_input"],
        "college_source": args.college_source,
        "current_observations": {k: report[k] for k in ("season", "through_week", "saved_at")},
        "assets": assets,
        "files": {a["object"]: a["sha256"] for a in assets.values()},
        "source_kinds": dict(Counter(a["kind"] for a in assets.values())),
        "limitations": [
            "Provider cache snapshots preserve the bytes available locally, not original HTTP "
            "responses or original historical publication vintages.",
            "Legacy parsed transactions and league-scored current observations are explicitly "
            "marked enriched imports. They are not relabeled as raw provider evidence.",
            "Registration time is not publication time. Event dates and known_on remain separate "
            "from capture timestamps; missing timestamps remain unknown.",
        ],
    }
    write_json(root / "manifest.json", manifest)
    print(f"Raw snapshot: {len(assets):,} preserved assets", flush=True)
    return root


def build_tables(data: Path, version: str, enriched_version: str | None = None) -> Path:
    enriched, em = load_manifest(
        data, "enriched", reference(data / "enriched/releases" / (enriched_version or version))
    )
    if (data / "gold/releases" / identifier(version)).exists():
        raise ValueError("Release ID already belongs to a historical source batch")
    root = data / "tables/batches" / identifier(version)
    root.mkdir(parents=True, exist_ok=False)
    (root / "tables").mkdir()
    implementations = root / "implementation"
    implementations.mkdir()
    for source in (
        Path(__file__),
        ROOT / "src/engine/tables/contracts.py",
        ROOT / "src/engine/data/enrich/__init__.py",
        ROOT / "src/engine/data/target_quality.py",
        ROOT / "src/engine/data/releases.py",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
    ):
        shutil.copy2(source, implementations / source.name)
    specs = {}

    def read(name):
        return pl.read_parquet(enriched / em["tables"][name])

    def save(name, frame, keys, role, description):
        frame, nonfinite = clean_floats(frame)
        spec = table_spec(frame, keys, role, description)
        path = root / "tables" / (name + ".parquet")
        frame.sort(keys).write_parquet(path)
        specs[name] = {
            **spec,
            "path": str(path.relative_to(root)),
            "sha256": digest(path),
            "enriched_release": enriched.name,
            "nonfinite_values_to_null": nonfinite,
        }

    players = read("nfl_players").drop_nulls("gsis_id")
    # Identity dimension only: no current team/status/position or mutable experience.
    identity_columns = [
        "gsis_id",
        "display_name",
        "birth_date",
        "espn_id",
        "pfr_id",
        "nfl_id",
        "rookie_season",
        "draft_year",
        "draft_round",
        "draft_pick",
        "draft_team",
    ]
    save(
        "players",
        players.select(identity_columns),
        ["gsis_id"],
        "identity",
        "Stable NFL identities and biographical/draft fields; not historical roster states.",
    )
    for name, keys in {
        "nfl_player_weeks": ["player_id", "season", "week"],
        "nfl_player_seasons": ["player_id", "season"],
        "absence_events": ["evidence_id"],
        "college_seasons": ["college_id", "season", "team_id"],
        "college_annual": ["college_id", "season"],
        "college_identity_links": ["college_id"],
        "current_weeks": ["player_id", "season", "week"],
        "current_snaps": ["player_id", "season", "week"],
        "current_players": ["player_id"],
    }.items():
        frame = read(name)
        missing_key = frame.filter(pl.any_horizontal(pl.col(keys).is_null()))
        if missing_key.height:
            save(
                name + "_quarantine",
                missing_key.with_row_index("record_id"),
                ["record_id"],
                "quarantine",
                "Source observations missing a primary-key identity; "
                "retained without inventing IDs.",
            )
            frame = frame.drop_nulls(keys)
        save(
            name,
            frame,
            keys,
            "observations",
            "Observed records; filter by event/known date before joining a forecast cutoff.",
        )

    # Preserve source-local record grain for feeds without a clean natural key.
    # Exact duplicates are collapsed; differing records are retained, not resolved by order.
    for name in (
        "nfl_injuries",
        "nfl_snap_counts",
        "college_games",
        "college_identity_registry",
        "nfl_identity_crosswalk",
    ):
        frame = read(name).unique(maintain_order=True)
        frame = frame.with_row_index("record_id")
        save(
            name,
            frame,
            ["record_id"],
            "observations",
            "Source records with a release-local row key; unresolved evidence is retained.",
        )
    schedule_columns = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
    ]
    current_schedule = read("current_schedule").select(schedule_columns)
    save(
        "current_schedule",
        current_schedule,
        ["game_id"],
        "observations",
        "Schedule from the captured current observation release; no result fields.",
    )
    schedule = pl.concat(
        [
            read("nfl_schedule")
            .filter(pl.col("season") < em["current_observations"]["season"])
            .select(schedule_columns),
            current_schedule,
        ],
        how="diagonal_relaxed",
    ).unique()
    save(
        "nfl_schedule",
        schedule,
        ["game_id"],
        "observations",
        "Captured schedule context; scores, realized starters, weather and betting results "
        "excluded.",
    )
    current = em["current_observations"]
    for name in ("current_weeks", "current_snaps"):
        frame = read(name)
        if frame.filter(
            (pl.col("season") != current["season"]) | (pl.col("week") > current["through_week"])
        ).height:
            raise ValueError("Current observations exceed their captured cutoff")
    # Apply football invariants centrally, including every transitive target derivative.
    from engine.data.target_quality import correct_tables

    tables = {name: pl.read_parquet(root / spec["path"]) for name, spec in specs.items()}
    corrected, target_quality = correct_tables(tables)
    for name, frame in corrected.items():
        if not frame.equals(tables[name]):
            spec = specs[name]
            save(name, frame, spec["primary_key"], spec["role"], spec["description"])
    for name in ("nfl_player_weeks", "nfl_player_seasons"):
        if corrected[name].filter(pl.col("receptions") > pl.col("targets")).height:
            raise ValueError(f"Invalid target history remains in {name}")
    write_json(root / "target_quality.json", target_quality)
    write_json(
        root / "quality.json",
        {
            "checks": {
                "unique_nonnull_primary_keys": True,
                "current_observation_cutoff": True,
                "nonfinite_quantities_are_null": True,
                "known_receptions_do_not_exceed_known_targets": True,
                "transitive_target_mask": True,
            },
            "coverage_complete": False,
        },
    )
    for prefix in ("nfl", "current"):
        save(
            f"{prefix}_observed_schedule",
            read(f"{prefix}_schedule"),
            ["game_id"],
            "observations",
            "Observed game results and starting QB identities; postgame evidence only.",
        )
    manifest = {
        "schema_version": 1,
        "layer": "tables",
        "version": version,
        "status": "accepted",
        "generated_at": now(),
        "quality_passed": True,
        "input": reference(enriched),
        "history": em["history"],
        "current_observations": current,
        "environment": {"python": sys.version, "polars": package_version("polars")},
        "tables": specs,
        "files": {
            str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()
        },
        "limitations": [
            "Historical statistics are retrospective reconstructions, "
            "not original publication vintages.",
            "Cleaned tables do not imply complete source coverage.",
            "Observation tables require time filtering before joining a forecast cutoff.",
            "Provider identifiers and linked college identities "
            "are retrospective identity mappings.",
        ],
    }
    write_json(root / "manifest.json", manifest)
    load_tables(data, version)
    print(f"Table batch: {len(specs)} tables", flush=True)
    return root


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="engine data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    refresh = sub.add_parser(
        "refresh", help="Capture completed weeks, build products/tables, optionally upload"
    )
    refresh.add_argument("--version", help="Unique run ID; reuse to resume a failed run")
    refresh.add_argument("--current-report", type=Path, help="Reuse a preserved capture")
    refresh.add_argument("--force", action="store_true", help="Rebuild same-week source revisions")
    refresh.add_argument("--status", action="store_true", help="Read the most recent run status")
    refresh.add_argument(
        "--upload", action="store_true", help="Publish the validated table catalog to GCS"
    )
    from engine.tables.transport import DEFAULT_STORE

    refresh.add_argument("--store", default=DEFAULT_STORE)
    build = sub.add_parser("build")
    build.add_argument("--version", required=True)
    build.add_argument("--history", required=True)
    build.add_argument("--college-source", required=True)
    build.add_argument("--current-report", required=True, type=Path)
    sub.add_parser("verify").add_argument("--version", required=True)
    enrich_parser = sub.add_parser("enrich")
    enrich_parser.add_argument("--version", required=True)
    enrich_parser.add_argument(
        "--raw-version", help="Raw snapshot to enrich; defaults to --version"
    )
    tables = sub.add_parser("tables")
    tables.add_argument("--version", required=True)
    tables.add_argument("--enriched-version")
    args = parser.parse_args(argv)
    from engine.config.settings import get_settings

    data = get_settings().data_dir.resolve()
    if args.command == "refresh":
        from engine.data import refresh as refresh_pipeline

        if args.status:
            path = data / ".runtime/observation_refresh.json"
            print(path.read_text() if path.exists() else '{"status": "never_run"}')
        else:
            print(json.dumps(refresh_pipeline.run(args), indent=2))
        return
    identifier(args.version)
    if args.command == "verify":
        source = load_tables(data, args.version)
        print(json.dumps({"table_release": source.ref, "tables": len(source.manifest["tables"])}))
    elif args.command == "enrich":
        enrich(data, args.version, args.raw_version)
    elif args.command == "tables":
        build_tables(data, args.version, args.enriched_version)
    else:
        (data / ".runtime").mkdir(parents=True, exist_ok=True)
        with (data / ".runtime/data_pipeline.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            register_sources(data, args)
            enrich(data, args.version)
            build_tables(data, args.version)


if __name__ == "__main__":
    main()
