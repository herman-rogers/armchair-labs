"""Version-pinned public college data and coverage-aware season tables.

Source: sportsdataverse/cfbfastR-cfb-data (ESPN-derived, retrospectively revised).
An athlete's missing game row is NOT evidence of an injury or a game played.
All stat shares use team totals from the SAME captured games.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

REPOSITORY = "sportsdataverse/cfbfastR-cfb-data"
DATASETS = {
    "players": ("player_box", "player_box"),
    "rosters": ("cfb_rosters", "cfb_rosters"),
    "schedule": ("cfb_schedules", "cfb_schedules"),
}
STATS = {
    "passing": {
        "passingYards": "passing_yards",
        "passingTouchdowns": "passing_tds",
        "interceptions": "passing_ints",
    },
    "rushing": {
        "rushingAttempts": "carries",
        "rushingYards": "rushing_yards",
        "rushingTouchdowns": "rushing_tds",
    },
    "receiving": {
        "receptions": "receptions",
        "receivingYards": "receiving_yards",
        "receivingTouchdowns": "receiving_tds",
    },
}
STAT_COLUMNS = [v for values in STATS.values() for v in values.values()]
KEY = ["season", "game_id", "team_id", "college_id"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique(frame: pl.DataFrame, keys: list[str]) -> None:
    if frame.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError(f"Conflicting/duplicate college keys: {keys}")


def discover_source() -> dict:
    """Discover actual files, not an assumed start date or an API's advertised span."""
    root = f"https://api.github.com/repos/{REPOSITORY}"
    response = requests.get(f"{root}/commits/main", timeout=45)
    response.raise_for_status()
    commit = response.json()["sha"]
    result: dict = {"repository": REPOSITORY, "commit": commit, "files": [], "coverage": {}}
    for dataset, (directory, prefix) in DATASETS.items():
        response = requests.get(
            f"{root}/contents/cfb/{directory}/parquet", params={"ref": commit}, timeout=45
        )
        response.raise_for_status()
        years = []
        for item in response.json():
            match = re.fullmatch(rf"{prefix}_(\d{{4}})\.parquet", item["name"])
            if not match:
                continue
            season = int(match[1])
            years.append(season)
            result["files"].append(
                {
                    "dataset": dataset,
                    "season": season,
                    "path": item["path"],
                    "git_blob_sha1": item["sha"],
                    "bytes": item["size"],
                    "url": f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{item['path']}",
                }
            )
        if not years:
            raise ValueError(f"No published {dataset} partitions")
        result["coverage"][dataset] = sorted(years)
    return result


def capture_source(root: Path, inventory: dict, start: int, end: int) -> list[dict]:
    """Small bounded parallel downloads. Existing verified files can resume a capture."""
    root.mkdir(parents=True, exist_ok=True)
    expected = {(kind, year) for kind in DATASETS for year in range(start, end + 1)}
    files = [r for r in inventory["files"] if (r["dataset"], r["season"]) in expected]
    if {(r["dataset"], r["season"]) for r in files} != expected:
        raise ValueError("Requested college range contains unpublished partitions")

    def download(spec: dict) -> dict:
        path = root / f"{spec['dataset']}_{spec['season']}.parquet"
        if path.exists():
            content = path.read_bytes()
        else:
            response = requests.get(spec["url"], timeout=90)
            response.raise_for_status()
            content = response.content
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob != spec["git_blob_sha1"] or len(content) != spec["bytes"]:
            raise ValueError(f"Pinned upstream blob mismatch: {path.name}")
        frame = pl.read_parquet(io.BytesIO(content))
        if not frame.height or set(frame["season"].drop_nulls()) != {spec["season"]}:
            raise ValueError(f"Wrong/empty season partition: {path.name}")
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
        return {
            **spec,
            "filename": path.name,
            "sha256": sha256(path),
            "rows": frame.height,
            "captured_at": datetime.now(UTC).isoformat(),
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(download, files))


def numeric(column: str) -> pl.Expr:
    return (
        pl.col(column).cast(pl.String).replace({"--": None, "-": None, "": None}).cast(pl.Float64)
    )


def normalize_season(
    raw: pl.DataFrame, rosters: pl.DataFrame, schedule: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Preserve team stints and all roster members, including zero-production players.

    Production is measured only where BOTH passing and receiving box categories
    exist for a team-game. Coverage is against the independent all-division schedule,
    including bowls. Partial team-seasons are retained, flagged, and not modeled.
    """
    for frame, required in [
        (raw, {"season", "game_id", "team_id", "athlete_id", "athlete_name", "category"}),
        (
            rosters,
            {
                "season",
                "team_id",
                "athlete_id",
                "athlete_display_name",
                "position_abbreviation",
                "team_location",
                "division",
                "date_of_birth",
            },
        ),
        (schedule, {"season", "game_id", "home_id", "away_id", "completed", "start_date"}),
    ]:
        if not required <= set(frame.columns):
            raise ValueError(f"College source schema missing {required - set(frame.columns)}")
    raw_count = raw.height
    raw = raw.unique()
    raw = raw.with_columns(
        pl.col("athlete_id").cast(pl.String).alias("college_id"),
        pl.col("game_id", "team_id").cast(pl.String),
    )
    if raw.select(pl.any_horizontal(pl.col(KEY).is_null()).any()).item():
        raise ValueError("Null college player/game/team identity")
    if raw.filter(pl.col("college_id").is_in(["0", "-1", ""])).height:
        raise ValueError("Placeholder college athlete identity")
    unique(raw, KEY + ["category"])
    schedule = schedule.with_columns(pl.col("game_id", "home_id", "away_id").cast(pl.String))
    unique(schedule, ["season", "game_id"])
    finals = schedule.filter(pl.col("completed") == True)  # noqa: E712
    team_games = pl.concat(
        [
            finals.select("season", "game_id", "start_date", pl.col(f"{side}_id").alias("team_id"))
            for side in ("home", "away")
        ]
    )
    unique(team_games, ["season", "game_id", "team_id"])
    keys = ["season", "game_id", "team_id"]
    categories = raw.group_by(keys).agg(pl.col("category").unique())
    captured = (
        categories.filter(
            pl.col("category").list.contains("passing")
            & pl.col("category").list.contains("receiving")
        )
        .select(keys)
        .join(team_games, on=keys, how="inner", validate="1:1")
    )
    eligible = raw.join(captured, on=keys, how="inner", validate="m:1")
    offense = eligible.filter(pl.col("category").is_in(STATS))
    expressions = []
    for category, columns in STATS.items():
        for old, new in columns.items():
            if old not in offense.columns:
                offense = offense.with_columns(pl.lit(None, dtype=pl.String).alias(old))
            expressions.append(
                pl.when(pl.col("category") == category)
                .then(numeric(old))
                .otherwise(None)
                .alias(new)
            )
    offense = offense.with_columns(expressions)
    # Some upstream games have unnamed stat_1...stat_5 cells rather than typed
    # passing columns. Retain their raw records, but quarantine the whole team-game
    # rather than guess an undocumented column ordering or treat it as zero.
    invalid = []
    for category, mapping in STATS.items():
        subset = offense.filter(pl.col("category") == category)
        invalid.append(
            subset.filter(
                pl.any_horizontal(
                    pl.col(list(mapping.values())).is_null()
                    | ~pl.col(list(mapping.values())).is_finite()
                )
            ).select(keys)
        )
    quarantined = pl.concat(invalid).unique()
    captured = captured.join(quarantined, on=keys, how="anti")
    offense = offense.join(quarantined, on=keys, how="anti")
    player_games = offense.group_by(KEY).agg(
        pl.col("athlete_name").sort().first().alias("player_name"),
        pl.col("start_date").first().alias("game_date"),
        *[pl.col(c).sum() for c in STAT_COLUMNS],
    )
    totals = player_games.group_by(keys).agg(
        *[pl.col(c).sum().alias(f"team_{c}") for c in STAT_COLUMNS]
    )
    # Some historical boxes omit a receiver/passer or disagree after laterals or
    # revisions. Keep the reported stats, but do not call these reconciled inputs.
    totals = totals.with_columns(
        (
            (pl.col("team_passing_yards") - pl.col("team_receiving_yards")).abs().gt(0.01)
            | (pl.col("team_passing_tds") != pl.col("team_receiving_tds"))
        ).alias("reconciliation_failure")
    )
    team_seasons = (
        team_games.group_by(["season", "team_id"])
        .agg(pl.len().alias("scheduled_games"))
        .join(
            captured.group_by(["season", "team_id"]).agg(
                pl.len().alias("captured_games"),
                pl.col("start_date").max().alias("season_end_date"),
            ),
            on=["season", "team_id"],
            how="left",
            validate="1:1",
        )
        .join(
            totals.group_by(["season", "team_id"]).agg(
                *[pl.col(f"team_{c}").sum() for c in STAT_COLUMNS],
                pl.col("reconciliation_failure").sum().alias("reconciliation_failures"),
            ),
            on=["season", "team_id"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("captured_games").fill_null(0),
            (pl.col("captured_games").fill_null(0) / pl.col("scheduled_games")).alias("coverage"),
        )
    )
    roster = rosters.select(
        pl.col("athlete_id").cast(pl.String).alias("college_id"),
        "season",
        pl.col("team_id").cast(pl.String),
        pl.col("athlete_display_name").alias("player_name"),
        pl.col("position_abbreviation").alias("college_position"),
        pl.col("team_location").alias("college_team"),
        "division",
        pl.col("date_of_birth").str.slice(0, 10).alias("birth_date"),
    ).unique()
    unique(roster, ["college_id", "season", "team_id"])
    observed = player_games.group_by(["college_id", "season", "team_id"]).agg(
        pl.col("player_name").sort().first(),
        pl.len().alias("observed_stat_games"),
        *[pl.col(c).sum() for c in STAT_COLUMNS],
    )
    seasons = roster.join(
        observed, on=["college_id", "season", "team_id"], how="full", coalesce=True, validate="1:1"
    )
    seasons = seasons.with_columns(
        pl.coalesce("player_name", "player_name_right").alias("player_name"),
        pl.col("observed_stat_games").fill_null(0),
    ).drop("player_name_right")
    seasons = seasons.join(team_seasons, on=["season", "team_id"], how="left", validate="m:1")
    seasons = seasons.with_columns(
        *[
            pl.when(pl.col("captured_games") > 0)
            .then(pl.col(c).fill_null(0))
            .otherwise(None)
            .alias(c)
            for c in STAT_COLUMNS
        ],
        ((pl.col("coverage") == 1) & (pl.col("reconciliation_failures") == 0))
        .fill_null(False)
        .alias("complete_team_season"),
    )
    # Denominators include every offensive player's captured output, not just NFL survivors.
    seasons = seasons.with_columns(
        *[
            (pl.col(c) / pl.when(pl.col(f"team_{c}") > 0).then(pl.col(f"team_{c}"))).alias(
                f"share_{c}"
            )
            for c in ["receiving_yards", "receptions", "rushing_yards", "carries"]
        ]
    )
    audit = {
        "season": int(schedule["season"][0]),
        "raw_rows": raw_count,
        "identical_rows_removed": raw_count - raw.height,
        "completed_schedule_games": finals.height,
        "captured_team_games": captured.height,
        "quarantined_team_games": quarantined.height,
        "unreconciled_team_games": totals.filter(pl.col("reconciliation_failure")).height,
        "complete_team_seasons": seasons.filter(pl.col("complete_team_season"))[
            "team_id"
        ].n_unique(),
        "roster_or_stat_players": seasons["college_id"].n_unique(),
        "unrostered_stat_stints": seasons.filter(pl.col("college_position").is_null()).height,
        "teams": team_seasons.select(
            "team_id", "scheduled_games", "captured_games", "coverage", "reconciliation_failures"
        ).to_dicts(),
    }
    return seasons, player_games, audit


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
