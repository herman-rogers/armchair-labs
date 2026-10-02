"""Offline, create-only four-week outlook study and read-only frontend publication.

Uses accepted repaired history plus an already captured all-player NFL snapshot
from rookie_watch.py. No source downloads, board rebuilds, or live model promotion.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml
from artifact_inputs import require_audited_version
from data_integrity_audit import digest

from patron.config.settings import CONFIG_DIR, get_settings
from patron.data.releases import load_gold
from patron.metrics.outlook import (
    OUTLOOK,
    POSITIONS,
    SCALE,
    USAGE,
    build_rows,
    evaluate,
    predict_fold,
    publication_rows,
    team_name,
    unique,
)
from patron.metrics.positions import canonical_positions

ROOT = Path(__file__).resolve().parents[1]


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--current-report", type=Path)
    parser.add_argument(
        "--gold", help="Canonical gold release; supplies history and current observations"
    )
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", args.version):
        parser.error("version must be a simple directory name")
    settings = get_settings()
    gold = load_gold(settings.data_dir, args.gold) if args.gold else None
    if gold:
        args.history = settings.data_dir / "research" / gold.manifest["history"]["version"]
    elif args.history is None or args.current_report is None:
        parser.error("Provide --gold, or both --history and --current-report")
    args.history = args.history.resolve()
    provenance = require_audited_version(args.history)
    current_report = (
        {**gold.manifest["current_observations"], "history": gold.manifest["history"]}
        if gold
        else json.loads(args.current_report.read_text())
    )
    if current_report["history"] != provenance:
        raise ValueError("Current snapshot and accepted historical version disagree")
    season, cutoff = current_report["season"], current_report["through_week"]
    if not 1 <= cutoff <= 18:
        raise ValueError("Outlook requires a completed regular-season cutoff from Week 1 to 18")
    if not gold:
        args.current_report = args.current_report.resolve()
        snapshot = (args.current_report.parent / current_report["snapshot_path"]).resolve()
        if not snapshot.is_relative_to(args.current_report.parent):
            raise ValueError("Current snapshot path escapes its output directory")
        for name, expected in current_report["source_sha256"].items():
            path = (snapshot / name).resolve()
            if not path.is_relative_to(snapshot) or digest(path) != expected:
                raise ValueError(f"Captured current input changed: {name}")
    frozen_scoring = args.history / "implementation/patron/config/scoring.yaml"
    if yaml.safe_load(frozen_scoring.read_text()) != yaml.safe_load(
        (CONFIG_DIR / "scoring.yaml").read_text()
    ):
        raise ValueError("Historical and current league scoring must match")
    settings = get_settings()
    version = settings.data_dir / "research" / args.version
    version.mkdir(parents=True, exist_ok=False)
    inputs = version / "inputs"
    inputs.mkdir()
    for name in ("weeks", "snaps", "schedule", "players"):
        source = gold.path(f"current_{name}") if gold else snapshot / f"{name}.parquet"
        shutil.copy2(source, inputs / f"current_{name}.parquet")
    for source in (Path(__file__), ROOT / "src/patron/metrics/outlook.py"):
        shutil.copy2(source, inputs / ("runner.py" if source == Path(__file__) else "model.py"))
    protected_manifest = json.loads((args.history / "manifest.json").read_text())
    protected = {
        name: digest(ROOT / name) for name in protected_manifest["protected_artifact_sha256"]
    }
    print(
        f"Building {args.version}: {season} through Week {cutoff}; no network or board writes",
        flush=True,
    )
    index = {
        p: set(pl.scan_parquet(p).collect_schema())
        for p in (args.history / "cache/nflverse").glob("*.parquet")
    }

    def sources(required):
        if gold:
            table = {
                "pfr_id": "players",
                "pfr_player_id": "nfl_snap_counts",
                "sack_fumbles_lost": "nfl_player_weeks",
                "gameday": "nfl_schedule",
            }
            names = [name for key, name in table.items() if key in required]
            if len(names) != 1:
                raise ValueError(f"No gold source contract for {required}")
            return gold.read(names[0])
        paths = sorted(p for p, cols in index.items() if set(required) <= cols)
        if not paths:
            raise ValueError(f"Missing frozen source: {required}")
        return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")

    ids = sources(["gsis_id", "pfr_id", "rookie_season"])
    identity = ids.select(pl.col("gsis_id").alias("player_id"), "pfr_id", "espn_id").drop_nulls(
        ["player_id", "pfr_id"]
    )
    unique(identity, ["pfr_id"])
    snap_ids = set(identity["player_id"])
    raw_snap = canonical_positions(sources(["pfr_player_id", "offense_snaps", "offense_pct"]))
    raw_snap = raw_snap.filter(
        (pl.col("game_type") == "REG") & pl.col("season").is_between(2013, season - 1)
    )
    mapped_snaps = raw_snap.join(
        identity, left_on="pfr_player_id", right_on="pfr_id", how="inner", validate="m:1"
    ).select(
        "player_id",
        "season",
        "week",
        "offense_snaps",
        "offense_pct",
        "position",
        "team",
        "player",
    )
    # Defensive identities can contain multi-team source collisions; they are
    # outside this skill-position study, not deduplicated into invented evidence.
    historical_snaps = mapped_snaps.filter(pl.col("position").is_in(POSITIONS))
    historical_points = (
        gold.read("nfl_player_weeks").select(
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
        )
        if gold
        else pl.read_parquet(args.history / "outputs/research_audited_weekly_points.parquet")
    ).filter(pl.col("season").is_between(2013, season - 1) & pl.col("player_id").is_not_null())
    raw_stats = sources(["player_id", "fantasy_points_ppr", "sack_fumbles_lost"])
    attempts = raw_stats.filter(
        (pl.col("season_type") == "REG")
        & pl.col("season").is_between(2013, season - 1)
        & pl.col("player_id").is_not_null()
    ).select("player_id", "season", "week", "attempts")
    historical_points = historical_points.join(
        attempts, on=["player_id", "season", "week"], how="left", validate="1:1"
    )
    historical_schedule = sources(
        ["game_id", "home_team", "away_team", "gameday", "game_type"]
    ).filter((pl.col("game_type") == "REG") & pl.col("season").is_between(2013, season - 1))
    pred = (
        gold.read("preseason_features")
        if gold
        else pl.read_parquet(args.history / "outputs/metric_backtest_predictions.parquet")
    )
    candidates = pred.select(
        "player_id",
        pl.col("forecast_season").alias("season"),
        "player_display_name",
        "position",
        pl.col("player_population").alias("population"),
        (
            pl.coalesce("cutoff_preseason_team", "source_team")
            if gold
            else pl.coalesce("projected_team", "team")
        ).alias("team"),
    )
    current_points = pl.read_parquet(inputs / "current_weeks.parquet")
    current_snaps = pl.read_parquet(inputs / "current_snaps.parquet")
    current_schedule = pl.read_parquet(inputs / "current_schedule.parquet")
    if current_points["week"].max() > cutoff or current_snaps["week"].max() > cutoff:
        raise ValueError("Current snapshot contains observations after its declared cutoff")
    if set(current_points["season"]) != {season} or set(current_snaps["season"]) != {season}:
        raise ValueError("Current snapshot has mixed seasons")
    # Newly observed players enter using only their current observed position; future
    # historical position labels never decide past eligibility.
    points = pl.concat(
        [historical_points, current_points.select(historical_points.columns)],
        how="diagonal_relaxed",
    )
    schedule = pl.concat([historical_schedule, current_schedule], how="diagonal_relaxed").unique()
    unique(schedule, ["game_id"])
    snap_ids |= set(current_snaps["player_id"])
    expected, observed = defaultdict(set), defaultdict(set)
    for row in schedule.filter(pl.col("game_type") == "REG").to_dicts():
        expected[row["season"], row["week"]].update(
            (team_name(row["home_team"]), team_name(row["away_team"]))
        )
    for row in raw_snap.to_dicts():
        observed[row["season"], row["week"]].add(team_name(row["team"]))
    current_teams = current_points.select("player_id", "season", "week", "team")
    for row in current_snaps.join(
        current_teams, on=["player_id", "season", "week"], how="inner", validate="1:1"
    ).to_dicts():
        observed[row["season"], row["week"]].add(team_name(row["team"]))
    snap_complete = {key for key, teams in expected.items() if teams <= observed[key]}
    point_teams = defaultdict(set)
    for row in points.select("season", "week", "team").unique().to_dicts():
        point_teams[row["season"], row["week"]].add(team_name(row["team"]))
    for key, teams in expected.items():
        if (key[0] < season or key[1] <= cutoff) and not teams <= point_teams[key]:
            raise ValueError(f"Incomplete scoring coverage at {key}")
    cutoffs = sorted({2, 4, 8, 12, cutoff})
    historical_candidates = candidates.filter(pl.col("season").is_between(2013, season - 1))
    history = build_rows(
        historical_points,
        historical_snaps,
        historical_schedule,
        historical_candidates,
        cutoffs,
        completed_seasons=set(range(2013, season)),
        snap_complete_weeks=snap_complete,
        snap_identities=snap_ids,
    )
    live = build_rows(
        current_points,
        current_snaps,
        schedule,
        candidates.filter(pl.col("season") == season),
        [cutoff],
        completed_seasons=set(),
        snap_complete_weeks=snap_complete,
        snap_identities=snap_ids,
        prior_snaps=historical_snaps,
    )
    validations = []
    for year in range(2019, season):
        print(f"Walk-forward validation: {year}", flush=True)
        for position in POSITIONS:
            for week in cutoffs:
                test = [
                    r
                    for r in history
                    if (r["season"], r["position"], r["cutoff_week"]) == (year, position, week)
                ]
                validations.extend(predict_fold(history, test))
    summaries = evaluate(validations)
    current_predictions = []
    for position in POSITIONS:
        current_predictions.extend(
            predict_fold(history, [r for r in live if r["position"] == position])
        )
    public = publication_rows(current_predictions, summaries)
    espn_ids = {r["gsis_id"]: r.get("espn_id") for r in ids.to_dicts()}
    espn_ids.update(
        {
            r["player_id"]: r.get("espn_id")
            for r in pl.read_parquet(inputs / "current_players.parquet").to_dicts()
        }
    )
    for row in public:
        value = espn_ids.get(row["player_id"])
        row["espn_id"] = int(value) if value is not None and str(value).isdigit() else None
    public.sort(
        key=lambda r: (r["forecast_next4"] is None, -(r["forecast_next4"] or 0), r["player_id"])
    )
    for position in POSITIONS:
        ranked = [
            r for r in public if r["position"] == position and r["forecast_next4"] is not None
        ]
        for rank, row in enumerate(ranked, 1):
            row["position_rank"] = rank
    saved_at = datetime.now(UTC).isoformat()
    report = {
        "version": args.version,
        "season": season,
        "through_week": cutoff,
        "horizon": [
            cutoff + 1,
            min(cutoff + 4, schedule.filter(pl.col("game_type") == "REG")["week"].max()),
        ],
        "saved_at": saved_at,
        "observations_saved_at": current_report["saved_at"],
        "history": provenance,
        **({"gold": gold.ref} if gold else {}),
        "status": "experimental_not_promoted",
        "confidence_scores_published": False,
        "players": public,
        "validation": summaries,
        "design": {
            "features": list(OUTLOOK),
            "usage_baseline_features": list(USAGE),
            "scale_features": list(SCALE),
            "cutoffs": cutoffs,
            "horizon_calendar_weeks": 4,
            "ridge_penalty": 10,
            "method": "Position/cutoff-specific ridge; earlier seasons only. Two prior seasons "
            "reserved for residual calibration. Error-scale model learns earlier-season errors.",
            "publication_policy": "Confidence scores always withheld in v1. Ranges require "
            "complete snap identity/feed and known prior sample, plus position-wide and "
            "rookie and prior-sample (<10 / ≥10 offensive weeks) gates: n≥200, ≥5 seasons, "
            "75–85% coverage, interval score no worse than usage, points MAE no worse than pace, "
            "and positive season-block 95% MAE improvement over usage.",
        },
        "coverage": {
            "current_snap_complete_weeks": sorted(w for y, w in snap_complete if y == season),
            "historical_rows": len(history),
            "validation_rows": len(validations),
            "unmatched_historical_snap_records": raw_snap.height - mapped_snaps.height,
            "out_of_scope_snap_records": mapped_snaps.height - historical_snaps.height,
        },
        "limitations": [
            "Research only; four-calendar-week totals, not remaining-season trade values. "
            "No market-price advantage is claimed.",
            "Offensive participation includes role loss and missed time; it is not medical "
            "availability or an injury probability. Missing snap identities/feeds stay unknown.",
            "Current saved health and ownership are display context only, "
            "not injury or transaction adjustments to these forecasts.",
            "Role estimate is snap share conditional on future offensive participation, "
            "not a probability of keeping a job.",
            "Opportunity/role trends need earlier observations; "
            "two-game samples cannot establish a durable role.",
            "Points already incorporate historical absences and byes. "
            "Do not multiply by the participation estimate again.",
            "Schedules account for the cutoff team's known games; "
            "future trades, injuries, matchups, and coaching changes are not known.",
            "Historical sources are retrospectively revised, not original publication-time "
            "vintages. Multiple research comparisons are exploratory.",
            "Intervals that fail validation are withheld, not relabeled as confidence. "
            "No calibrated confidence score is published.",
        ],
    }
    pl.DataFrame(history, infer_schema_length=None).write_parquet(
        version / "historical_features.parquet"
    )
    pl.DataFrame(validations, infer_schema_length=None).write_parquet(
        version / "validation_predictions.parquet"
    )
    pl.DataFrame(current_predictions, infer_schema_length=None).write_parquet(
        version / "current_research_predictions.parquet"
    )
    json_file(version / "outlook.json", report)
    after = {name: digest(ROOT / name) for name in protected}
    if protected != after:
        raise ValueError(
            "A protected artifact changed during the outlook build; publication stopped"
        )
    manifest = {
        "version": args.version,
        "status": "complete",
        "saved_at": saved_at,
        "history_path": str(args.history.relative_to(settings.data_dir)),
        "history": provenance,
        **({"gold": gold.ref} if gold else {}),
        "protected_artifacts_unchanged": True,
        "protected_sha256": protected,
        "source_sha256": {str(p.relative_to(version)): digest(p) for p in inputs.iterdir()},
        "output_sha256": {p.name: digest(p) for p in version.iterdir() if p.is_file()},
        "current_source_report_sha256": (
            gold.ref["manifest_sha256"] if gold else digest(args.current_report)
        ),
    }
    json_file(version / "manifest.json", manifest)
    pointer = {"version": args.version, "manifest_sha256": digest(version / "manifest.json")}
    destination = settings.outputs_dir / f"player_outlook_{season}.json"
    if not args.no_publish:
        temporary = destination.with_suffix(".tmp")
        json_file(temporary, pointer)
        temporary.replace(destination)
        print(f"Published research outlook pointer: {destination}", flush=True)
    print(
        json.dumps(
            {
                "players": len(public),
                "scored": sum(r["forecast_next4"] is not None for r in public),
                "published_ranges": sum(r["outcome_range"] is not None for r in public),
                "cutoff_summaries": [
                    s for s in summaries if s["cutoff_week"] == cutoff and s["cohort"] == "all"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
