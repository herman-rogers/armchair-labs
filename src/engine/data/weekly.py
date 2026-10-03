"""Capture completed regular-season observations independently of rookie experiments."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from engine.config.settings import get_settings
from engine.data import nflverse
from engine.data.releases import digest, write_json
from engine.metrics.positions import canonical_positions
from engine.metrics.rookies import completed_cutoff
from engine.scoring.bonuses import extract_touchdown_bonuses
from engine.scoring.engine import score_components


def mapped_snaps(raw: pl.DataFrame, identities: pl.DataFrame) -> pl.DataFrame:
    mapping = identities.select(pl.col("gsis_id").alias("player_id"), "pfr_id").drop_nulls()
    if mapping["pfr_id"].is_duplicated().any():
        raise ValueError("Ambiguous PFR snap identity")
    return (
        raw.filter(pl.col("game_type") == "REG")
        .join(mapping, left_on="pfr_player_id", right_on="pfr_id", how="inner", validate="m:1")
        .select("player_id", "season", "week", "offense_snaps", "offense_pct")
    )


def capture(season: int, *, after_week: int = 0, force: bool = False) -> Path | None:
    settings = get_settings()
    current = nflverse.load_player_weeks([season])
    schedule = nflverse.load_schedules(season)
    cutoff = completed_cutoff(schedule, current, season)
    if cutoff < after_week:
        raise ValueError("Provider coverage regressed behind the published cutoff")
    if cutoff == 0 or (cutoff == after_week and not force):
        print(f"No new completed week (latest: {cutoff}).", flush=True)
        return None
    identities = nflverse.load_players()
    current = current.filter(pl.col("week") <= cutoff)
    plays = nflverse.load_touchdown_plays([season]).filter(pl.col("week") <= cutoff)
    bonuses, _ = extract_touchdown_bonuses(plays)
    current = (
        score_components(current)
        .join(bonuses, on=["player_id", "season", "week"], how="left", validate="1:1")
        .with_columns(
            (pl.col("component_pts") + pl.col("bonus_pts").fill_null(0)).alias("league_points")
        )
    )
    raw_snaps = nflverse.load_snap_counts([season]).with_columns(
        pl.col("team").replace({"LAR": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS"})
    )
    if completed_cutoff(schedule, raw_snaps.filter(pl.col("game_type") == "REG"), season) < cutoff:
        raise ValueError(
            "Snap coverage is not complete through the latest finished week; retry later"
        )
    snaps = mapped_snaps(raw_snaps, identities).filter(pl.col("week") <= cutoff)
    # Preserve the current-player contract: new rookies supplement historical identities.
    players = (
        canonical_positions(identities)
        .filter(
            (pl.coalesce("rookie_season", "draft_year") == season)
            & pl.col("position").is_in(["QB", "RB", "WR", "TE"])
            & pl.col("gsis_id").is_not_null()
        )
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("display_name").alias("player_display_name"),
            "position",
            pl.lit(season).alias("season"),
            "draft_pick",
            pl.col("espn_id").cast(pl.Int64, strict=False),
            pl.col("latest_team").alias("team"),
        )
    )
    now = datetime.now(UTC).isoformat()
    root = settings.outputs_dir / "weekly_snapshots" / f"{season}_w{cutoff}_{now.replace(':', '-')}"
    root.mkdir(parents=True, exist_ok=False)
    for name, frame in dict(
        weeks=current, snaps=snaps, schedule=schedule, players=players, touchdown_plays=plays
    ).items():
        frame.write_parquet(root / f"{name}.parquet")
    report = root / "report.json"
    write_json(
        report,
        dict(
            season=season,
            through_week=cutoff,
            saved_at=now,
            snapshot_path=".",
            current_source="nflverse weekly statistics, schedules, touchdown plays and PFR snaps",
            source_sha256={p.name: digest(p) for p in root.glob("*.parquet")},
            implementation_sha256=digest(Path(__file__)),
        ),
    )
    return report
