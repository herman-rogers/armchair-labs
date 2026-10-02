"""Build a create-only college→NFL identity registry and temporal research release."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from artifact_inputs import require_audited_version

from patron.config.settings import get_settings
from patron.data.college import normalize_season, sha256, write_json
from patron.data.college_identity import build_links
from patron.data.releases import load_gold, load_manifest
from patron.metrics.college_translation import (
    annual_rows,
    college_backtest,
    nfl_backtest,
    nfl_cohort,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--gold")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    settings = get_settings()
    gold = load_gold(settings.data_dir, args.gold) if args.gold else None
    if gold:
        enriched, enrichment = load_manifest(settings.data_dir, "enriched", gold.manifest["input"])
        _, snapshot = load_manifest(settings.data_dir, "raw", enrichment["input"])
        args.source = settings.data_dir / "research" / snapshot["college_source"]
        args.history = settings.data_dir / "research" / gold.manifest["history"]["version"]
    elif args.source is None or args.history is None:
        parser.error("Provide --gold, or both --source and --history")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", args.version):
        parser.error("Invalid research version")
    provenance = require_audited_version(args.history)
    source = json.loads((args.source / "manifest.json").read_text())
    if source["kind"] != "college_source" or source["status"] != "captured":
        raise ValueError("College source capture is incomplete")
    # Never label the current, partially played college season complete.
    if source["through_season"] >= datetime.now(UTC).year:
        raise ValueError("This season-level study requires completed college seasons")
    for spec in source["files"]:
        path = (args.source / "raw" / spec["filename"]).resolve()
        if (
            not path.is_relative_to((args.source / "raw").resolve())
            or sha256(path) != spec["sha256"]
        ):
            raise ValueError("Captured college source changed")
    settings = get_settings()
    version = settings.data_dir / "research" / args.version
    version.mkdir(parents=True, exist_ok=False)
    inputs = version / "inputs"
    inputs.mkdir()
    protected_names = json.loads((args.history / "manifest.json").read_text())[
        "protected_artifact_sha256"
    ]
    protected = {name: sha256(ROOT / name) for name in protected_names}
    for original, name in [
        (Path(__file__), "runner.py"),
        (ROOT / "src/patron/data/college.py", "college.py"),
        (ROOT / "src/patron/data/college_identity.py", "college_identity.py"),
        (ROOT / "src/patron/metrics/college_translation.py", "college_translation.py"),
        (ROOT / "data/static/college_identity_overrides.json", "identity_overrides.json"),
        (args.source / "manifest.json", "college_source_manifest.json"),
    ]:
        shutil.copy2(original, inputs / name)
    write_json(version / "manifest.json", {"version": args.version, "status": "building"})
    season_frames, game_frames, coverage = [], [], []
    for year in [] if gold else range(source["start_season"], source["through_season"] + 1):
        frame, games, audit = normalize_season(
            *[
                pl.read_parquet(args.source / "raw" / f"{kind}_{year}.parquet")
                for kind in ["players", "rosters", "schedule"]
            ]
        )
        season_frames.append(frame)
        game_frames.append(games)
        coverage.append(audit)
        print(
            f"College {year}: {frame.height:,} player-team seasons; "
            f"{audit['complete_team_seasons']} complete teams",
            flush=True,
        )
    college = (
        gold.read("college_seasons") if gold else pl.concat(season_frames, how="diagonal_relaxed")
    ).sort("college_id", "season", "team_id")
    college.write_parquet(version / "college_seasons.parquet")
    (
        gold.read("college_games") if gold else pl.concat(game_frames, how="diagonal_relaxed")
    ).write_parquet(version / "college_games.parquet")
    del game_frames, season_frames
    files = {
        p: set(pl.read_parquet_schema(p))
        for p in (args.history / "cache/nflverse").glob("*.parquet")
    }

    def source_table(required: set[str], filename: str) -> pl.DataFrame:
        if gold:
            table = "players" if filename == "nfl_identities.parquet" else "nfl_identity_crosswalk"
            shutil.copy2(gold.path(table), inputs / filename)
            return gold.read(table)
        paths = [p for p, columns in files.items() if required <= columns]
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one accepted {filename} table")
        shutil.copy2(paths[0], inputs / filename)
        return pl.read_parquet(paths[0])

    identities = source_table(
        {"gsis_id", "display_name", "rookie_season"}, "nfl_identities.parquet"
    )
    crosswalk = source_table({"cfbref_id", "espn_id", "gsis_id"}, "nfl_crosswalk.parquet")
    overrides = json.loads((inputs / "identity_overrides.json").read_text())
    print("Resolving provider IDs and corroborated identities", flush=True)
    if gold:
        links, registry = (
            gold.read("college_identity_links"),
            gold.read("college_identity_registry"),
        )
        quality = json.loads((enriched / "college_quality.json").read_text())
        identity_audit, coverage = quality["identity"], quality["coverage"]
    else:
        links, registry, identity_audit = build_links(college, identities, crosswalk, overrides)
    links.write_parquet(version / "identity_links.parquet")
    registry.write_parquet(version / "identity_registry.parquet")
    print(
        "Identity audit:",
        {k: v for k, v in identity_audit.items() if k != "review_rows"},
        flush=True,
    )
    predictions = (
        gold.read("preseason_features").join(
            gold.read("season_outcomes"),
            on=["player_id", "forecast_season", "forecast_cutoff_date"],
            validate="1:1",
        )
        if gold
        else pl.read_parquet(args.history / "outputs/metric_backtest_predictions.parquet")
    )
    candidates = predictions.filter(pl.col("player_population") == "rookie").select(
        "player_id",
        "player_display_name",
        "position",
        "forecast_season",
        "rookie_draft_pick",
        "outcome_complete",
        "actual_season_points",
        "actual_games",
    )
    complete_through = int(candidates.filter(pl.col("outcome_complete"))["forecast_season"].max())
    candidates.write_parquet(inputs / "nfl_candidates.parquet")
    weekly = (
        gold.read("nfl_player_weeks")
        if gold
        else pl.read_parquet(args.history / "outputs/research_audited_weekly_points.parquet")
    )
    nfl_seasons = weekly.group_by("player_id", "season").agg(pl.col("league_points").sum())
    reconciled = candidates.filter(pl.col("outcome_complete")).join(
        nfl_seasons,
        left_on=["player_id", "forecast_season"],
        right_on=["player_id", "season"],
        how="left",
        validate="1:1",
    )
    if reconciled.filter(
        (pl.col("actual_season_points").fill_null(0) - pl.col("league_points").fill_null(0)).abs()
        > 1e-6
    ).height:
        raise ValueError("Accepted rookie outcomes disagree with audited weekly scoring")
    nfl_seasons.write_parquet(inputs / "nfl_seasons.parquet")
    annual = gold.read("college_annual").to_dicts() if gold else annual_rows(college)
    pl.DataFrame(annual, infer_schema_length=None).write_parquet(version / "college_annual.parquet")
    rows = nfl_cohort(annual, links, candidates, nfl_seasons, complete_through)
    cohort = pl.DataFrame(rows, infer_schema_length=None)
    cohort.write_parquet(version / "nfl_cohort.parquet")
    by_class = (
        cohort.group_by("forecast_year", "position")
        .agg(
            pl.len().alias("candidates"),
            pl.col("college_ids").list.len().gt(0).sum().alias("linked"),
            pl.col("eligible").sum().alias("eligible"),
        )
        .sort("forecast_year", "position")
        .to_dicts()
    )
    print(
        f"NFL cohort: {sum(r['eligible'] for r in rows):,}/{len(rows):,} eligible; "
        f"outcomes through {complete_through}",
        flush=True,
    )
    nfl_results, nfl_summary = nfl_backtest(rows)
    _, strict_summary = nfl_backtest(
        [
            row
            for row in rows
            if row["link_methods"] and "exact_name_school_entry_window" not in row["link_methods"]
        ]
    )
    pl.DataFrame(nfl_results, infer_schema_length=None).write_parquet(
        version / "nfl_predictions.parquet"
    )
    print("Backtesting next-college-season production", flush=True)
    college_results, college_summary = college_backtest(annual)
    pl.DataFrame(college_results, infer_schema_length=None).write_parquet(
        version / "college_predictions.parquet"
    )
    report = {
        "version": args.version,
        "saved_at": datetime.now(UTC).isoformat(),
        "status": "experimental_not_promoted",
        "history": provenance,
        **({"gold": gold.ref} if gold else {}),
        "college_start": source["start_season"],
        "college_end": source["through_season"],
        "nfl_complete_through": complete_through,
        "source_commit": source["commit"],
        "coverage": coverage,
        "identity": identity_audit,
        "cohorts": by_class,
        "nfl_backtest": nfl_summary,
        "strict_identity_sensitivity": strict_summary,
        "college_backtest": college_summary,
        "current_forecast_year": complete_through + 1,
        "limitations": [
            "Retrospectively revised provider records, not historical publication vintages.",
            "2004 is the first player-stat partition; schedules alone extend to 2001. "
            "Earlier careers are left-truncated.",
            "NFL translation is conditional on the accepted NFL-entry candidate population; "
            "no NFL-entry probability is modeled.",
            "Unlinked college athletes are unknown, NOT zero-production NFL outcomes.",
            "Only complete, reconciled final college team-seasons enter forecasts. "
            "Incomplete, unreconciled and multi-team seasons remain visible but unscored.",
            "Early unnamed stat columns are quarantined, not guessed. "
            "Non-FBS coverage is substantially thinner.",
            "College-only models exclude NFL draft capital; post-draft models include it.",
            "Three-year NFL targets must be complete before each training cutoff.",
            "College yardage forecasts are conditional on subsequent roster presence; "
            "they do not predict transfers, exits or injury.",
            "College production includes bowls; NFL targets use regular-season league scoring.",
            "NFL games is participation, not probability of medical health or a starting job.",
            "No individual confidence intervals, market edge, or production promotion is claimed.",
            "Missing NFL draft picks combine undrafted and unknown records; "
            "a missingness indicator and pick-300 sentinel are explicit baseline assumptions.",
        ],
    }
    write_json(version / "report.json", report)
    if any(sha256(ROOT / name) != expected for name, expected in protected.items()):
        raise ValueError("Protected production/reference artifact changed during research build")
    manifest = {
        "version": args.version,
        "kind": "college_nfl_research",
        "status": "complete",
        "history": provenance,
        **({"gold": gold.ref} if gold else {}),
        "source_version": source["version"],
        "source_manifest_sha256": sha256(args.source / "manifest.json"),
        "protected_artifacts_unchanged": True,
        "protected_sha256": protected,
        "source_sha256": {str(p.relative_to(version)): sha256(p) for p in inputs.iterdir()},
        "output_sha256": {
            p.name: sha256(p)
            for p in version.iterdir()
            if p.is_file() and p.name != "manifest.json"
        },
    }
    write_json(version / "manifest.json", manifest)
    pointer = settings.outputs_dir / "college_nfl.json"
    if not args.no_publish:
        temporary = pointer.with_suffix(".tmp")
        write_json(
            temporary,
            {"version": args.version, "manifest_sha256": sha256(version / "manifest.json")},
        )
        temporary.replace(pointer)
    action = "Built" if args.no_publish else "Published"
    print(f"{action} {version}; all {len(protected)} protected artifacts unchanged", flush=True)


if __name__ == "__main__":
    main()
