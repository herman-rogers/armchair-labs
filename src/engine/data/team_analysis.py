"""Participation-backed QB/WR/TE observations from the verified current gold release."""

from functools import lru_cache

import polars as pl

from engine.data.frames import read_frame
from engine.data.releases import GoldRelease, load_manifest

TABLES = (
    "nfl_player_weeks",
    "current_weeks",
    "nfl_snap_counts",
    "current_snaps",
    "players",
    "current_players",
    "nfl_schedule",
    "current_schedule",
)
ALIASES = {"LAR": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC"}


def normalize(frame: pl.DataFrame, column: str = "team") -> pl.DataFrame:
    return frame.with_columns(pl.col(column).replace(ALIASES))


def load_panel(gold: GoldRelease) -> pl.DataFrame:
    # Revalidate paths before accessing decoded cached observations.
    # Gold schedules intentionally omit QB identities. Use their richer, pinned
    # enriched input, verified as part of the same immutable dependency closure.
    enriched, manifest = load_manifest(gold.root.parents[2], "enriched", gold.manifest["input"])
    paths = tuple(
        str(enriched / manifest["tables"][name])
        if name.endswith("schedule")
        else str(gold.path(name))
        for name in TABLES
    )
    cutoff = gold.manifest["current_observations"]
    return _panel(paths, gold.ref["manifest_sha256"], cutoff["season"], cutoff["through_week"])


@lru_cache(maxsize=2)
def _panel(paths: tuple[str, ...], digest: str, season: int, through_week: int) -> pl.DataFrame:
    from pathlib import Path

    sources = dict(zip(TABLES, paths, strict=True))

    def read(name, columns=None):
        return read_frame(Path(sources[name]), columns=columns)

    stat_cols = [
        "game_id",
        "player_id",
        "player_display_name",
        "position",
        "team",
        "season",
        "week",
        "league_points",
        "fantasy_points_ppr",
        "targets",
    ]
    weeks = normalize(
        pl.concat([read(n, stat_cols) for n in TABLES[:2]], how="vertical_relaxed")
    ).unique(["game_id", "player_id"], keep="last")
    weeks = weeks.filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") <= through_week))
    )
    schedule_cols = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "home_qb_id",
        "away_qb_id",
    ]
    schedule = pl.concat(
        [read(n, schedule_cols) for n in TABLES[-2:]], how="vertical_relaxed"
    ).unique("game_id", keep="last")
    schedule = schedule.filter(
        (pl.col("game_type") == "REG")
        & pl.col("home_score").is_not_null()
        & (pl.col("season") >= 2013)
        & (
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") <= through_week))
        )
    )
    games = normalize(
        pl.concat(
            [
                schedule.select(
                    "game_id",
                    "season",
                    "week",
                    "gameday",
                    pl.col(f"{side}_team").alias("team"),
                    pl.col(f"{side}_qb_id").alias("starting_qb_id"),
                )
                for side in ("home", "away")
            ]
        )
    )
    identities = read("players", ["gsis_id", "pfr_id"])
    crosswalk = identities.filter(pl.col("pfr_id").is_not_null()).select(
        "pfr_id", pl.col("gsis_id").alias("player_id")
    )
    snaps = normalize(read("nfl_snap_counts")).filter(pl.col("position").is_in(["QB", "WR", "TE"]))
    snaps = snaps.join(
        crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="left", validate="m:1"
    ).rename({"player": "name"})
    unresolved = snaps.filter(pl.col("player_id").is_null()).join(
        games.select("game_id", "team"), on=["game_id", "team"], how="semi"
    )
    potential = unresolved.group_by("pfr_player_id").agg(
        (pl.col("offense_pct") >= 0.5).sum().alias("starts"),
        (pl.col("position") == "QB").any().alias("quarterback"),
    )
    if potential.filter((pl.col("starts") >= 3) | pl.col("quarterback")).height:
        raise ValueError("Starter snap identities are incomplete")
    # Unlinked brief-role players cannot meet the three-start eligibility rule.
    # Never guess identities or assign an unmatched eligible player zero points.
    snaps = snaps.filter(pl.col("player_id").is_not_null())
    snap_cols = [
        "game_id",
        "season",
        "week",
        "team",
        "player_id",
        "name",
        "position",
        "offense_snaps",
        "offense_pct",
    ]
    snaps = snaps.select(snap_cols)
    current = read("current_snaps").join(
        weeks.select("player_id", "season", "week", "team", "player_display_name", "position"),
        on=["player_id", "season", "week"],
        how="left",
        validate="m:1",
    )
    roster = (
        normalize(read("current_players"))
        .select("player_id", "team", "player_display_name", "position")
        .unique("player_id")
    )
    current = current.join(roster, on="player_id", how="left", suffix="_roster").with_columns(
        *[
            pl.coalesce(c, f"{c}_roster").alias(c)
            for c in ("team", "player_display_name", "position")
        ]
    )
    current = normalize(current).join(
        games.select("game_id", "season", "week", "team"),
        on=["season", "week", "team"],
        how="inner",
    )
    current = (
        current.rename({"player_display_name": "name"})
        .filter(pl.col("position").is_in(["QB", "WR", "TE"]))
        .select(snap_cols)
    )
    panel = (
        pl.concat([snaps, current], how="vertical_relaxed")
        .unique(["game_id", "team", "player_id"], keep="last")
        .join(games, on=["game_id", "season", "week", "team"], how="inner", validate="m:1")
    )
    panel = panel.filter(pl.col("offense_snaps") > 0)
    if panel["player_id"].null_count():
        raise ValueError("Offensive snap identities are incomplete")
    panel = panel.join(
        weeks.select("game_id", "team", "player_id", "league_points"),
        on=["game_id", "team", "player_id"],
        how="left",
        validate="1:1",
    )
    # No recorded stats with verified offensive snaps is a genuine scoring zero.
    panel = panel.with_columns(pl.col("league_points").fill_null(0.0))
    panel = panel.filter(
        (pl.col("position") != "QB") | (pl.col("player_id") == pl.col("starting_qb_id"))
    )
    return panel.with_columns(
        ((pl.col("offense_pct") >= 0.5) | (pl.col("position") == "QB")).alias("starter_game")
    ).sort("season", "week", "player_id")
