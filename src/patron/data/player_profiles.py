"""Offline, versioned player-profile assembly from already verified sources."""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from patron.api.college_sources import load_college
from patron.api.outlook_sources import load_outlook_release
from patron.api.research_sources import accepted_version, digest
from patron.config.settings import get_settings
from patron.data.releases import load_gold
from patron.metrics.player_profile import KEYS, METRICS, TOTALS, career_features, unique
from patron.metrics.positions import canonical_positions
from patron.metrics.profile_tracking import (
    TRACKING_METRICS,
    TRACKING_NOTE,
    TRACKING_SCHEMA,
    tracking_seasons,
)

OUTPUTS = {
    "players.parquet",
    "nfl_weeks.parquet",
    "profile_features.parquet",
    "injuries.parquet",
    "tracking_seasons.parquet",
    "report.json",
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_weeks(points: pl.DataFrame, stats: pl.DataFrame, snaps: pl.DataFrame) -> pl.DataFrame:
    """Scoring has one owner: accepted league points. Stats only enrich that record."""
    points, stats, snaps = [
        f.with_columns(pl.col("season", "week").cast(pl.Int32)) for f in (points, stats, snaps)
    ]
    for frame in (points, stats, snaps):
        unique(frame, KEYS)
    if points.filter(pl.col("league_points").is_null()).height:
        raise ValueError("Missing audited points on a recorded scoring row")
    if snaps.filter(
        pl.col("offense_snaps").is_not_null()
        & (~pl.col("offense_snaps").is_finite() | (pl.col("offense_snaps") < 0))
    ).height:
        raise ValueError("Invalid offensive snap count in profile source")
    extra = [c for c in TOTALS if c not in points.columns and c in stats.columns]
    base = points.join(stats.select(*KEYS, *extra), on=KEYS, how="left", validate="1:1")
    base = base.with_columns(pl.lit(True).alias("stat_recorded"))
    # Positive snap-only weeks are useful observations, but missing stat quantities
    # remain unknown. A captured scoring source establishes no recorded points.
    result = base.join(snaps, on=KEYS, how="full", coalesce=True, suffix="_snap", validate="1:1")
    if "team_snap" in result.columns:
        result = result.with_columns(pl.coalesce("team", "team_snap").alias("team")).drop(
            "team_snap"
        )
    if "position_snap" in result.columns:
        result = result.with_columns(
            pl.coalesce("position", "position_snap").alias("position")
        ).drop("position_snap")
    result = result.filter(pl.col("stat_recorded").fill_null(False) | (pl.col("offense_snaps") > 0))
    result = result.with_columns(
        pl.col("stat_recorded").fill_null(False),
        pl.col("league_points").fill_null(0),
        pl.col("offense_pct").alias("snap_share"),
    ).drop("offense_pct")
    for name in TOTALS:
        if name not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    invalid = result.filter(
        pl.col("snap_share").is_not_null()
        & (~pl.col("snap_share").is_finite() | ~pl.col("snap_share").is_between(0, 1))
    )
    if invalid.height:
        raise ValueError("Invalid offensive snap share in profile source")
    for name in TOTALS:
        if result.filter(
            pl.col(name).is_not_null() & ~pl.col(name).cast(pl.Float64).is_finite()
        ).height:
            raise ValueError(f"Nonfinite profile source: {name}")
    return result.sort(KEYS)


def build_profiles(
    version: str,
    season: int,
    *,
    gold_version: str | None = None,
    college_ref: dict | None = None,
    outlook_ref: dict | None = None,
    publish: bool = True,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", version):
        raise ValueError("Profile version must be a simple directory name")
    settings = get_settings()
    gold = load_gold(settings.data_dir, gold_version) if gold_version else None
    college, study = load_college(reference_override=college_ref)
    outlook_root, outlook = load_outlook_release(season, reference_override=outlook_ref)
    history, acceptance, manifest = accepted_version(outlook["history"]["version"])
    if study["history"] != outlook["history"]:
        raise ValueError("College and outlook must use the same accepted NFL history")
    if gold and (study.get("gold") != gold.ref or outlook.get("gold") != gold.ref):
        raise ValueError("Profile dependencies must use the same gold release")
    root = settings.data_dir / "research" / version
    root.mkdir(parents=True, exist_ok=False)
    protected = {
        path: digest(Path(__file__).resolve().parents[3] / path)
        for path in manifest["protected_artifact_sha256"]
    }
    sources: dict[str, str] = {}

    def record(path: Path):
        sources[str(path.relative_to(settings.data_dir))] = digest(path)

    for dependency in (college, outlook_root):
        record(dependency / "manifest.json")
        dependency_manifest = json.loads((dependency / "manifest.json").read_text())
        for name in {
            **dependency_manifest["source_sha256"],
            **dependency_manifest["output_sha256"],
        }:
            record(dependency / name)
    # The original college capture remains part of the dependency closure.
    college_manifest = json.loads((college / "manifest.json").read_text())
    capture = settings.data_dir / "research" / college_manifest["source_version"]
    record(capture / "manifest.json")
    for spec in json.loads((capture / "manifest.json").read_text())["files"]:
        record(capture / "raw" / spec["filename"])

    paths = sorted((history / "cache/nflverse").glob("*.parquet"))
    schemas = {path: set(pl.scan_parquet(path).collect_schema()) for path in paths}

    def source(required: set[str]) -> pl.DataFrame:
        if gold:
            candidates = {
                "sack_fumbles_lost": "nfl_player_weeks",
                "pfr_player_id": "nfl_snap_counts",
                "report_primary_injury": "nfl_injuries",
            }
            names = [name for key, name in candidates.items() if key in required]
            if len(names) != 1:
                raise ValueError(f"No gold profile source contract for {required}")
            record(gold.path(names[0]))
            return gold.read(names[0])
        found = [path for path, columns in schemas.items() if required <= columns]
        if not found:
            raise ValueError(f"Missing frozen profile source: {required}")
        for path in found:
            record(path)
        return pl.concat([pl.read_parquet(path) for path in found], how="diagonal_relaxed")

    if gold:
        record(gold.root / "manifest.json")
        record(gold.path("preseason_features"))
        forecasts = gold.read("preseason_features").with_columns(
            pl.coalesce("cutoff_preseason_team", "source_team").alias("team")
        )
    else:
        forecasts = pl.read_parquet(history / "outputs/metric_backtest_predictions.parquet")
    forecasts = forecasts.filter(pl.col("forecast_season") <= season)
    identities = pl.read_parquet(college / "inputs/nfl_identities.parquet")
    current_players = outlook["players"]
    ids = set(forecasts["player_id"]) | {r["player_id"] for r in current_players}
    tracking = pl.DataFrame(schema=TRACKING_SCHEMA)
    if gold:
        record(gold.path("nfl_player_seasons"))
        tracking = tracking_seasons(
            gold.read("nfl_player_seasons").filter(
                pl.col("player_id").is_in(list(ids)) & (pl.col("season") < season)
            )
        )
    tracking.write_parquet(root / "tracking_seasons.parquet")
    points = (
        gold.read("nfl_player_weeks")
        if gold
        else pl.read_parquet(history / "outputs/research_audited_weekly_points.parquet")
    ).filter(pl.col("player_id").is_in(list(ids)) & (pl.col("season") < season))
    stats = source({"player_id", "fantasy_points_ppr", "sack_fumbles_lost"}).filter(
        (pl.col("season_type") == "REG")
        & (pl.col("season") < season)
        & pl.col("player_id").is_in(list(ids))
    )
    crosswalk = identities.select(pl.col("gsis_id").alias("player_id"), "pfr_id").drop_nulls()
    unique(crosswalk, ["pfr_id"])
    raw_snaps = canonical_positions(source({"pfr_player_id", "offense_snaps", "offense_pct"}))
    snaps = (
        raw_snaps.filter((pl.col("game_type") == "REG") & (pl.col("season") < season))
        .join(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner", validate="m:1")
        .filter(pl.col("player_id").is_in(list(ids)))
        .select(*KEYS, "offense_snaps", "offense_pct", "team", "position")
    )
    # Retain all outcome positions for eligible IDs, including position switches.
    historical = prepare_weeks(points, stats, snaps)
    current = pl.read_parquet(outlook_root / "inputs/current_weeks.parquet").filter(
        pl.col("player_id").is_in(list(ids))
    )
    current_snaps = pl.read_parquet(outlook_root / "inputs/current_snaps.parquet").filter(
        pl.col("player_id").is_in(list(ids))
    )
    if (
        current.filter(
            (pl.col("season") != season) | (pl.col("week") > outlook["through_week"])
        ).height
        or current_snaps.filter(
            (pl.col("season") != season) | (pl.col("week") > outlook["through_week"])
        ).height
    ):
        raise ValueError("Current observations exceed the profile cutoff")
    current = prepare_weeks(current, current, current_snaps)
    # A snap-only current observation need not have a team in the box score.
    team_map = {r["player_id"]: r.get("team") for r in current_players}
    current = current.with_columns(
        pl.coalesce(
            "team",
            pl.col("player_id").replace_strict(team_map, default=None, return_dtype=pl.String),
        ).alias("team")
    )
    keep = [
        *KEYS,
        "player_display_name",
        "position",
        "team",
        *TOTALS,
        "stat_recorded",
        "offense_snaps",
        "snap_share",
    ]
    weeks = pl.concat([historical.select(keep), current.select(keep)], how="diagonal_relaxed")
    unique(weeks, KEYS)
    weeks.write_parquet(root / "nfl_weeks.parquet")

    injuries = (
        source({"gsis_id", "report_primary_injury", "practice_status"})
        .filter(
            (pl.col("game_type") == "REG")
            & (pl.col("season") < season)
            & pl.col("gsis_id").is_in(list(ids))
        )
        .select(
            pl.col("gsis_id").alias("player_id"),
            "season",
            "week",
            "team",
            "report_primary_injury",
            "report_secondary_injury",
            "report_status",
            "practice_status",
            pl.col("date_modified").cast(pl.String).alias("source_modified_at"),
        )
    )
    injuries.unique().write_parquet(root / "injuries.parquet")
    by_id = defaultdict(list)
    for row in weeks.to_dicts():
        by_id[row["player_id"]].append(row)
    identity_map = {r["gsis_id"]: r for r in identities.to_dicts() if r["gsis_id"]}
    latest = {r["player_id"]: r for r in forecasts.sort("forecast_season").to_dicts()}
    latest.update({r["player_id"]: r for r in current_players})
    links = pl.read_parquet(college / "identity_links.parquet").filter(pl.col("status") == "linked")
    linked = set(links["player_id"])
    players, features = [], []
    for player_id in sorted(ids):
        identity, observed = identity_map.get(player_id, {}), latest.get(player_id, {})
        rows = by_id[player_id]
        years = sorted({r["season"] for r in rows})
        players.append(
            {
                "player_id": player_id,
                "player_display_name": observed.get("player_display_name")
                or identity.get("display_name")
                or player_id,
                "position": observed.get("position") or identity.get("position"),
                "team": observed.get("team"),
                "rookie_season": identity.get("rookie_season"),
                "birth_date": identity.get("birth_date"),
                "draft_year": identity.get("draft_year"),
                "draft_round": identity.get("draft_round"),
                "draft_pick": identity.get("draft_pick"),
                "draft_team": identity.get("draft_team"),
                "first_nfl_season": years[0] if years else None,
                "last_nfl_season": years[-1] if years else None,
                "nfl_seasons": len(years),
                "college_linked": player_id in linked,
                "current_candidate": player_id in {r["player_id"] for r in current_players},
                "history_left_truncated": bool(
                    years and identity.get("rookie_season") and years[0] > identity["rookie_season"]
                ),
            }
        )
        # Every row is computed before that forecast season, never from later data.
        for year in sorted(set(y + 1 for y in years if y < season) | {season}):
            features.append(
                {"player_id": player_id, "forecast_season": year, **career_features(rows, year, 0)}
            )
    pl.DataFrame(players, infer_schema_length=None).write_parquet(root / "players.parquet")
    pl.DataFrame(features, infer_schema_length=None).write_parquet(
        root / "profile_features.parquet"
    )
    report = {
        "version": version,
        "schema_version": 2,
        "season": season,
        "through_week": outlook["through_week"],
        "observations_saved_at": outlook["observations_saved_at"],
        "generated_at": datetime.now(UTC).isoformat(),
        "history": acceptance["audited_input"],
        **({"gold": gold.ref} if gold else {}),
        "college_version": college.name,
        "outlook_version": outlook_root.name,
        "players": len(players),
        "nfl_weeks": weeks.height,
        "profile_feature_rows": len(features),
        "college_linked_players": sum(r["college_linked"] for r in players),
        "metrics": METRICS,
        "tracking": {
            "source": "nfl_player_seasons",
            "metrics": TRACKING_METRICS,
            "note": TRACKING_NOTE,
            "rows": tracking.height,
            "available_seasons_by_metric": {
                metric: tracking.filter(pl.col(metric).is_not_null())["season"]
                .unique()
                .sort()
                .to_list()
                for metric in TRACKING_METRICS
            },
        },
        "limits": [
            (
                "Observed histories and descriptive role bands; no new forecast or composite "
                "talent score."
            ),
            (
                "NFL regular-season observations begin in 2001; college source history "
                "begins in 2004. Missing seasons are unknown."
            ),
            (
                "Historical source revisions are retained; these are cutoff-filtered "
                "reconstructions, not original publication-time snapshots."
            ),
            (
                "Participation bands use 35% and 70% offensive snaps. They are not starts, "
                "routes, injury diagnoses or proven skill changes."
            ),
            (
                "College statistics remain at school-stint grain with coverage and identity "
                "evidence. Unmatched identities are not NFL failures."
            ),
            (
                "Current and preseason forecasts retain their own horizons and source "
                "versions. They are not averaged or availability-discounted again."
            ),
        ],
    }
    write_json(root / "report.json", report)
    implementations = root / "implementation"
    implementations.mkdir()
    for path in [
        Path(__file__),
        Path(__file__).parents[1] / "metrics/player_profile.py",
        Path(__file__).parents[1] / "metrics/profile_tracking.py",
    ]:
        shutil.copy2(path, implementations / path.name)
    unchanged = all(
        digest(Path(__file__).resolve().parents[3] / name) == expected
        for name, expected in protected.items()
    )
    if not unchanged:
        raise ValueError("Protected artifact changed while profiles were built")
    # Recheck every input immediately before publishing the pointer.
    for name, expected in sources.items():
        if digest(settings.data_dir / name) != expected:
            raise ValueError(f"Profile source changed during build: {name}")
    accepted_version(report["history"]["version"])
    write_json(
        root / "manifest.json",
        {
            "version": version,
            "kind": "player_profiles",
            "schema_version": 2,
            "status": "complete",
            "history": report["history"],
            **({"gold": gold.ref} if gold else {}),
            "protected_artifacts_unchanged": unchanged,
            "source_sha256": sources,
            "output_sha256": {name: digest(root / name) for name in OUTPUTS},
            "implementation_sha256": {p.name: digest(p) for p in implementations.iterdir()},
        },
    )
    pointer = settings.outputs_dir / "player_profiles.json"
    if publish:
        temporary = pointer.with_suffix(".tmp")
        write_json(
            temporary, {"version": version, "manifest_sha256": digest(root / "manifest.json")}
        )
        temporary.replace(pointer)
    return root
