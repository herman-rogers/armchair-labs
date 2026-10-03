"""Supported data lifecycle: capture → validate → build → publish → GCS.

Local, single-writer Parquet pipeline. No network is used after source registration.
Products are built separately and the complete catalog is published atomically.
`engine data refresh --upload` orchestrates the full production workflow.
`engine tables refresh --upload` rebuilds recipes from already captured inputs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from engine.config.settings import REPO_ROOT
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
from engine.tables.contracts import clean_floats, coverage, split_forecasts, table_spec

ROOT = REPO_ROOT


def now():
    return datetime.now(UTC).isoformat()


def register_sources(data: Path, args) -> Path:
    from engine.api.research_sources import accepted_version

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
    for path in sorted((history / "implementation/engine/config").glob("*.yaml")):
        add("configuration/" + path.name, path, "configuration")
    for name in ("historical_inputs.parquet", "research_audited_weekly_points.parquet"):
        add(
            "checks/" + name,
            history / "outputs" / name,
            "accepted_replay_reference",
            old_manifest["output_sha256"]["outputs/" + name],
        )
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


def enrich(data: Path, version: str) -> Path:
    raw, snapshot = load_manifest(data, "raw", reference(data / "raw/snapshots" / version))
    root = data / "enriched/releases" / identifier(version)
    root.mkdir(parents=True, exist_ok=False)
    tables = root / "tables"
    tables.mkdir()
    write_json(root / "build_state.json", {"status": "building", "started_at": now()})
    implementation = root / "implementation"
    shutil.copytree(
        ROOT / "src/engine", implementation / "engine", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(Path(__file__), implementation / "data_pipeline.py")
    shutil.copy2(ROOT / "uv.lock", implementation / "uv.lock")
    shutil.copy2(ROOT / "pyproject.toml", implementation / "pyproject.toml")

    def offline(*args, **kwargs):
        raise RuntimeError("Data pipeline is offline: a required source was not captured")

    # Enrichment runs in its own subprocess; block all network access there.
    socket.socket.connect = offline  # type: ignore[method-assign]
    socket.socket.connect_ex = offline  # type: ignore[method-assign]
    with tempfile.TemporaryDirectory(prefix="engine-enrich-") as temporary:
        work = Path(temporary)
        for name, asset in snapshot["assets"].items():
            relative = name.removeprefix("history/")
            target = inside(work, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inside(data, asset["object"]), target)
        os.environ["DATA_DIR"] = str(work)
        os.environ["NFLVERSE_CACHE_DURATION"] = "3153600000"
        from engine.config import settings as configuration

        configuration.CONFIG_DIR = work / "configuration"
        configuration.get_settings.cache_clear()
        from engine.board import pipeline
        from engine.data import nflverse
        from engine.metrics.backtest import MetricReportConfig

        settings = configuration.Settings(data_dir=work, nflverse_cache_duration=3153600000)
        nflverse.configure_cache(settings)
        config = MetricReportConfig.from_config()
        print("Enrichment: replaying historical statistics and dated evidence", flush=True)
        result = pipeline.build_metric_report(
            settings=settings, report_config=config, analyze=False, enrichment_dir=tables
        )
        expected = pl.read_parquet(work / "checks/historical_inputs.parquet")
        assert_frame_equal(
            result.predictions.drop("rank").sort("forecast_season", "player_id"),
            expected.drop("rank").sort("forecast_season", "player_id"),
            check_column_order=False,
            abs_tol=1e-9,
            rel_tol=1e-9,
        )
        weekly = pl.read_parquet(tables / "nfl_player_weeks.parquet")
        expected_weekly = pl.read_parquet(work / "checks/research_audited_weekly_points.parquet")
        assert_frame_equal(
            weekly.select(expected_weekly.columns).sort("player_id", "season", "week"),
            expected_weekly.sort("player_id", "season", "week"),
            check_dtypes=False,
            abs_tol=1e-9,
            rel_tol=1e-9,
        )
        print(
            "Enrichment: forecast inputs and weekly scoring match accepted reconstruction",
            flush=True,
        )

        # Schedule context is a separate observation table, never a future-result feature.
        paths = [
            p
            for p in (work / "cache/nflverse").glob("*.parquet")
            if {"game_id", "gameday", "home_team", "away_team", "gametime"}
            <= set(pl.read_parquet_schema(p))
        ]
        if len(paths) != 1:
            raise ValueError("Expected one captured historical schedule")
        shutil.copyfile(paths[0], tables / "nfl_schedule.parquet")
        # The metric builder only needs a narrow injury projection. Gold also
        # serves player histories, so retain secondary injuries and source timestamps.
        injury_paths = [
            p
            for p in (work / "cache/nflverse").glob("*.parquet")
            if {"gsis_id", "report_primary_injury", "practice_status"}
            <= set(pl.read_parquet_schema(p))
        ]
        if not injury_paths:
            raise ValueError("Missing captured full injury reports")
        pl.concat(
            [pl.read_parquet(p) for p in sorted(injury_paths)], how="diagonal_relaxed"
        ).write_parquet(tables / "nfl_injuries.parquet")
        events = json.loads((work / "static/historical_absences.json").read_text())["events"]
        pl.DataFrame(events, infer_schema_length=None).write_parquet(
            tables / "absence_events.parquet"
        )
        for name in ("weeks", "snaps", "schedule", "players"):
            shutil.copyfile(
                work / "current" / (name + ".parquet"), tables / f"current_{name}.parquet"
            )

        from engine.data.college import normalize_season
        from engine.data.college_identity import build_links
        from engine.metrics.college_translation import annual_rows

        cm = json.loads((work / "college/manifest.json").read_text())
        seasons, games, audits = [], [], []
        for year in range(cm["start_season"], cm["through_season"] + 1):
            frame, game, audit = normalize_season(
                *[
                    pl.read_parquet(work / "college/raw" / f"{kind}_{year}.parquet")
                    for kind in ("players", "rosters", "schedule")
                ]
            )
            seasons.append(frame)
            games.append(game)
            audits.append(audit)
        college = pl.concat(seasons, how="diagonal_relaxed").sort("college_id", "season", "team_id")
        college.write_parquet(tables / "college_seasons.parquet")
        pl.concat(games, how="diagonal_relaxed").write_parquet(tables / "college_games.parquet")
        links, registry, identity_audit = build_links(
            college,
            pl.read_parquet(tables / "nfl_players.parquet"),
            pl.read_parquet(tables / "nfl_identity_crosswalk.parquet"),
            json.loads((work / "static/college_identity_overrides.json").read_text()),
        )
        links.write_parquet(tables / "college_identity_links.parquet")
        registry.write_parquet(tables / "college_identity_registry.parquet")
        pl.DataFrame(annual_rows(college), infer_schema_length=None).write_parquet(
            tables / "college_annual.parquet"
        )
        write_json(root / "college_quality.json", {"coverage": audits, "identity": identity_audit})

    write_json(root / "build_state.json", {"status": "complete", "completed_at": now()})
    checks = {
        "historical_input_replay_except_legacy_tied_rank": True,
        "audited_weekly_scoring_replay": True,
        "offline_build": True,
    }
    write_json(root / "quality.json", {"checks": checks, "coverage_complete": False})
    manifest = {
        "schema_version": 1,
        "layer": "enriched",
        "version": version,
        "status": "accepted",
        "generated_at": now(),
        "quality_passed": all(checks.values()),
        "input": reference(raw),
        "history": snapshot["history"],
        "current_observations": snapshot["current_observations"],
        "environment": {
            "python": sys.version,
            "packages": {
                name: package_version(name)
                for name in ("polars", "pyarrow", "nflreadpy", "numpy", "scikit-learn")
            },
        },
        "files": {
            str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()
        },
        "tables": {p.stem: str(p.relative_to(root)) for p in tables.glob("*.parquet")},
    }
    write_json(root / "manifest.json", manifest)
    print(f"Enriched release: {len(manifest['tables'])} tables", flush=True)
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

    features, labels, exclusions = split_forecasts(read("historical_inputs"))
    from engine.tables.contracts import FORECAST_KEY

    quarantine = pl.DataFrame(
        exclusions["quarantined_candidates"],
        schema={
            "player_id": pl.String,
            "forecast_season": pl.Int32,
            "forecast_cutoff_date": pl.Date,
            "player_display_name": pl.String,
            "player_population": pl.String,
            "reason": pl.String,
        },
    )
    save(
        "forecast_quarantine",
        quarantine,
        FORECAST_KEY,
        "quarantine",
        "Candidate folds with only later market admission evidence; excluded from features.",
    )

    save(
        "preseason_features",
        features,
        FORECAST_KEY,
        "asof_features",
        "One player / forecast season / cutoff; production statistics refer to source_season.",
    )
    save(
        "season_outcomes",
        labels,
        FORECAST_KEY,
        "labels",
        "Realized outcomes for the same candidate keys; incomplete seasons remain null.",
    )
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
        "nfl_team_seasons": ["team", "season"],
        "nfl_weekly_usage": ["player_id", "season", "week"],
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
        "nfl_transactions",
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
    for name in ("nfl_player_weeks", "nfl_player_seasons", "preseason_features"):
        if corrected[name].filter(pl.col("receptions") > pl.col("targets")).height:
            raise ValueError(f"Invalid target history remains in {name}")
    features = corrected["preseason_features"]
    write_json(root / "target_quality.json", target_quality)
    cov = coverage(features)
    write_json(root / "coverage.json", cov)
    write_json(root / "feature_policy.json", exclusions)
    write_json(
        root / "quality.json",
        {
            "checks": {
                "unique_nonnull_primary_keys": True,
                "matching_feature_label_keys": True,
                "no_future_evidence_in_preseason_features": True,
                "pending_outcomes_null": True,
                "known_absence_caps": True,
                "current_observation_cutoff": True,
                "historical_replay_in_enriched": True,
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
            "Observation tables require time filtering; "
            "only preseason_features is a forecast-cutoff view.",
            "Provider identifiers and linked college identities "
            "are retrospective identity mappings.",
        ],
    }
    write_json(root / "manifest.json", manifest)
    load_tables(data, version)
    print(
        f"Table batch: {len(specs)} tables; {features.height:,} forecast feature rows", flush=True
    )
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
    for command in ("enrich", "verify"):
        sub.add_parser(command).add_argument("--version", required=True)
    tables = sub.add_parser("tables")
    tables.add_argument("--version", required=True)
    tables.add_argument("--enriched-version")
    products = sub.add_parser("products")
    products.add_argument("--version", required=True, help="Source table batch")
    products.add_argument("--prefix", required=True, help="Create-only product version prefix")
    products.add_argument("--publish", action="store_true")
    publish = sub.add_parser("publish")
    publish.add_argument("--version", required=True, help="Source table batch")
    publish.add_argument("--prefix", required=True, help="Previously built product version prefix")
    publish.add_argument(
        "--analysis", help="Verified NextGen analysis release to publish with products"
    )
    args = parser.parse_args(argv)
    from engine.config.settings import get_settings

    data = get_settings().data_dir.resolve()
    if args.command == "refresh":
        from engine.data import refresh as refresh_pipeline

        if args.status:
            path = data / ".runtime/nextgen_refresh.json"
            print(path.read_text() if path.exists() else '{"status": "never_run"}')
        else:
            print(json.dumps(refresh_pipeline.run(args), indent=2))
        return
    identifier(args.version)
    if args.command == "verify":
        source = load_tables(data, args.version)
        print(json.dumps({"table_release": source.ref, "tables": len(source.manifest["tables"])}))
    elif args.command == "enrich":
        enrich(data, args.version)
    elif args.command == "tables":
        build_tables(data, args.version, args.enriched_version)
    elif args.command in {"products", "publish"}:
        from concurrent.futures import ThreadPoolExecutor

        from engine.data.catalog import publish_catalog
        from engine.data.player_profiles import build_profiles

        source = load_tables(data, args.version)
        season = source.manifest["current_observations"]["season"]
        names = {
            key: identifier(args.prefix + "_" + suffix)
            for key, suffix in {
                "college": "college",
                f"outlook_{season}": "outlook",
                "profiles": "profiles",
            }.items()
        }
        (data / ".runtime").mkdir(parents=True, exist_ok=True)
        with (data / ".runtime/data_pipeline.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == "products":
                commands = [
                    [
                        sys.executable,
                        str(ROOT / "research" / script),
                        "--gold",
                        args.version,
                        "--version",
                        names[key],
                        "--no-publish",
                    ]
                    for script, key in (
                        ("college_translation.py", "college"),
                        ("player_outlook.py", f"outlook_{season}"),
                    )
                ]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(subprocess.run, command, check=True) for command in commands
                    ]
                    for future in futures:
                        future.result()
                build_profiles(
                    names["profiles"],
                    season,
                    gold_version=args.version,
                    college_ref=reference(data / "research" / names["college"]),
                    outlook_ref=reference(data / "research" / names[f"outlook_{season}"]),
                    publish=False,
                )
            if args.command == "publish" or args.publish:
                if getattr(args, "analysis", None):
                    names["analysis"] = identifier(args.analysis)
                catalog = publish_catalog(data, args.version, names)
                print(json.dumps(catalog, indent=2))
            else:
                print(
                    json.dumps(
                        {"table_release": source.ref, "products": names, "published": False},
                        indent=2,
                    )
                )
    else:
        (data / ".runtime").mkdir(parents=True, exist_ok=True)
        with (data / ".runtime/data_pipeline.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            register_sources(data, args)
            subprocess.run(
                [sys.executable, "-m", "engine.data.pipeline", "enrich", "--version", args.version],
                check=True,
            )
            build_tables(data, args.version)


if __name__ == "__main__":
    main()
