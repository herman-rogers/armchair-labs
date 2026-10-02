"""Raw, cutoff-aware inputs for the distribution notebook; no prepared features."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / "data/raw/snapshots/canonical_20260923_r4/manifest.json"
LAST_COMPLETE_SEASON = 2025
KEY = ["player_id", "season", "week"]
STATS = [
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
]
META = {
    "season",
    "week",
    "play_id",
    "old_game_id",
    "jersey_number",
    "player_jersey_number",
    "game_id",
    "nflverse_game_id",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def family(name, columns):
    c = set(columns)
    if not name.startswith("history/cache/nflverse/"):
        return "college" if name.startswith("college/raw/") else "reference_or_capture"
    if "play_id" in c:
        return "plays" if "rusher_player_id" in c else "participation"
    for marker, group in [
        ("player_display_name", "box"),
        ("offense_pct", "snaps"),
        ("report_status", "injuries"),
        ("depth_team", "depth"),
        ("dt", "depth_dated"),
        ("gameday", "schedule"),
        ("display_name", "identity"),
        ("db_season", "crosswalk"),
        ("forty", "combine"),
        ("season_history", "contracts"),
        ("scrape_date", "rankings"),
        ("pass_completions_exp", "provider_models"),
        ("birth_date", "rosters"),
    ]:
        if marker in c:
            if "player_gsis_id" in c:
                return "ngs"
            return group
    if "carries" in c and "player_id" not in c:
        return "team_box"
    return "unmapped"


POLICY = {
    "box": "All numeric past player observations; IDs/clock fields excluded.",
    "team_box": "All numeric past team observations; future values are labels only.",
    "plays": "All numeric past rush/target play summaries, plus red-zone opportunities.",
    "participation": "All numeric past participation fields, joined by game/play IDs.",
    "snaps": "Past observations; exact PFR→GSIS crosswalk only.",
    "ngs": "Past weekly NGS aggregates, not frame-level player tracking; week 0 excluded.",
    "injuries": (
        "Past-week reports modified no later than that week's end; undated reports excluded."
    ),
    "depth": "Past-week depth rank only; no future or preseason Week 1 proxies.",
    "depth_dated": "Inventoried; different date-grain source, not joined to past-week panel.",
    "rosters": "Past weekly numeric/status fields; immutable birth date supplies age.",
    "identity": "Exact IDs and birth date only; current team/status/career end excluded.",
    "crosswalk": "Exact ID mapping only; current age/team/status excluded.",
    "combine": "Pre-draft measurements, used only after the recorded combine season.",
    "schedule": "Calendar and past lines. Future closing lines are sensitivity only.",
    "contracts": "Excluded: retrospective contract history lacks reliable as-of dates.",
    "rankings": "Excluded: single scrape is not a historical ranking archive.",
    "provider_models": "Inventoried, excluded: provider expected-production model outputs.",
    "college": (
        "Prior-season player stats via unique exact ESPN IDs; opaque stat_N fields excluded. "
        "Roster/schedule retained for exploration."
    ),
    "reference_or_capture": "Excluded from fits: prior outputs/configuration/current captures.",
    "unmapped": "Inventoried, excluded pending a verified identity and time contract.",
}


@lru_cache(maxsize=1)
def catalog():
    manifest = json.loads(SNAPSHOT.read_text())
    rows = []
    for name, spec in manifest["assets"].items():
        path = ROOT / "data" / spec["object"]
        columns, n = [], None
        if name.endswith(".parquet"):
            f = pq.ParquetFile(path)
            columns, n = f.schema_arrow.names, f.metadata.num_rows
        group = family(name, columns)
        rows.append(
            {
                "asset": name,
                "family": group,
                "rows": n,
                "columns": len(columns),
                "bytes": spec["bytes"],
                "sha256": spec["sha256"],
                "path": str(path),
                "policy": POLICY[group],
            }
        )
    return pl.DataFrame(rows)


def raw_preview(asset, limit=100):
    row = catalog().filter(pl.col("asset") == asset).row(0, named=True)
    if not asset.endswith(".parquet"):
        return pl.DataFrame({"note": ["Choose a parquet asset to preview its unmodified rows."]})
    return pl.scan_parquet(row["path"]).head(limit).collect()


def read_family(group, columns=None):
    frames = []
    for r in catalog().filter(pl.col("family") == group).iter_rows(named=True):
        if not r["asset"].endswith(".parquet"):
            continue
        if digest(r["path"]) != r["sha256"]:
            raise ValueError(f"Raw object changed: {r['asset']}")
        frame = pl.read_parquet(r["path"], columns=columns)
        if "season_type" in frame.columns:
            frame = frame.filter(pl.col("season_type") == "REG")
        elif "game_type" in frame.columns:
            frame = frame.filter(pl.col("game_type") == "REG")
        frames.append(frame)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def numeric(frame, keys=()):
    return [
        c
        for c, t in frame.schema.items()
        if t.is_numeric() and c not in META | set(keys) and not c.endswith("_id")
    ]


def finite(frame):
    cols = [c for c, t in frame.schema.items() if t.is_float()]
    return frame.with_columns(
        [pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in cols]
    )


def unique_observations(frame, keys):
    """Collapse identical captures; fail on contradictory values rather than choosing one."""
    frame = frame.unique()
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"Conflicting raw observations for {keys}")
    return frame


def prefixed(frame, keys, prefix):
    frame = frame.with_columns(pl.col([k for k in keys if k in {"season", "week"}]).cast(pl.Int64))
    cols = numeric(frame, keys)
    return (
        frame.select([*keys, *cols])
        .group_by(keys)
        .mean()
        .rename({c: f"{prefix}__{c}" for c in cols})
    )


@lru_cache(maxsize=1)
def load_raw():
    """Preserve every raw asset in the catalog and build a broad numeric observation panel."""
    box = finite(read_family("box")).filter(
        pl.col("player_id").is_not_null() & (pl.col("season") <= LAST_COMPLETE_SEASON)
    )
    box = unique_observations(box, KEY)
    team = finite(read_family("team_box")).filter(pl.col("season") <= LAST_COMPLETE_SEASON)
    team = unique_observations(team, ["team", "season", "week"])
    schedule = read_family("schedule").filter(pl.col("season") <= LAST_COMPLETE_SEASON)
    aliases = {"STL": "LA", "LAR": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}
    box = box.with_columns(pl.col("team").replace(aliases))
    team = team.with_columns(pl.col("team").replace(aliases))
    schedule = schedule.with_columns(pl.col("home_team", "away_team").replace(aliases))
    identity = read_family("identity")
    crosswalk = read_family("crosswalk")
    # Conflicting crosswalk IDs are withheld, never resolved by arbitrary row order.
    mapping = crosswalk.select("pfr_id", pl.col("gsis_id").alias("player_id")).drop_nulls().unique()
    mapping = mapping.filter(~pl.col("pfr_id").is_duplicated())
    player = prefixed(box, KEY, "box")
    supplemental = []
    for group in ["snaps", "ngs", "depth", "injuries", "rosters"]:
        f = read_family(group)
        if not f.height:
            continue
        if group == "snaps":
            f = f.join(mapping, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        else:
            id_col = "player_gsis_id" if group == "ngs" else "gsis_id"
            f = f.rename({id_col: "player_id"})
        f = f.filter(pl.col("player_id").is_not_null() & (pl.col("week") > 0))
        if group == "injuries":
            dates = schedule.group_by("season", "week").agg(
                pl.col("gameday").cast(pl.Date).max().alias("week_end")
            )
            f = f.with_columns(pl.col("season", "week").cast(pl.Int32)).join(
                dates, on=["season", "week"], how="left"
            )
            f = f.filter(
                pl.col("date_modified").is_not_null()
                & (pl.col("date_modified").cast(pl.Date) <= pl.col("week_end"))
            )
        # Missing source fields remain null after the join, including missing injury records.
        cat_cols = [c for c in ["report_status", "practice_status", "status"] if c in f.columns]
        if cat_cols:
            f = f.to_dummies(columns=cat_cols)
        if group == "rosters":
            # Draft/experience/physical values may evolve; lag these like other observations.
            f = f.drop([c for c in ["entry_year", "rookie_year"] if c in f.columns])
        supplemental.append(prefixed(finite(f), KEY, group))

    # Summarize every numeric observed play field, separately for rushing and receiving.
    # EPA/WP are provider-derived historical descriptors and are visible as such in names.
    plays = finite(read_family("plays"))
    participation = read_family("participation")
    if participation.height:
        pc = numeric(participation)
        p = participation.select("nflverse_game_id", "play_id", *pc).unique()
        p = p.group_by("nflverse_game_id", "play_id").mean()
        p = p.rename({c: f"participation_{c}" for c in pc})
        plays = plays.join(
            p,
            left_on=["game_id", "play_id"],
            right_on=["nflverse_game_id", "play_id"],
            how="left",
            validate="m:1",
        )
    for role, id_col in [("rush", "rusher_player_id"), ("target", "receiver_player_id")]:
        f = plays.filter(
            pl.col(id_col).is_not_null()
            & (pl.col("play_type") != "no_play")
            & (pl.col("play_deleted") == 0)
        )
        f = f.rename({id_col: "player_id"})
        means = prefixed(f, KEY, f"pbp_{role}")
        counts = f.group_by(KEY).agg(
            (pl.col("yardline_100") <= 20).sum().alias(f"pbp_{role}__redzone_count"),
            (pl.col("yardline_100") <= 5).sum().alias(f"pbp_{role}__inside5_count"),
        )
        supplemental.extend([means, counts])
    del plays
    for f in supplemental:
        f = f.with_columns(pl.col("season", "week").cast(pl.Int64))
        player = player.join(f, on=KEY, how="full", coalesce=True, validate="1:1")
    player = finite(player).filter(pl.col("season") <= LAST_COMPLETE_SEASON)
    espn = (
        crosswalk.select(pl.col("espn_id").cast(pl.String), pl.col("gsis_id").alias("player_id"))
        .drop_nulls()
        .unique()
    )
    espn = espn.filter(~pl.col("espn_id").is_duplicated())
    college_frames = []
    for r in (
        catalog()
        .filter((pl.col("family") == "college") & pl.col("asset").str.contains("/players_"))
        .iter_rows(named=True)
    ):
        if digest(r["path"]) != r["sha256"]:
            raise ValueError("College raw object changed")
        c = pl.read_parquet(r["path"]).with_columns(
            pl.col("athlete_id").cast(pl.String).alias("espn_id")
        )
        c = c.join(espn, on="espn_id", how="inner")
        keep = [
            n
            for n in c.columns
            if n
            not in {
                "athlete_id",
                "athlete_name",
                "team_id",
                "game_id",
                "season",
                "jersey",
                "category",
                "espn_id",
                "player_id",
            }
            and not n.startswith("stat_")
        ]
        c = c.select(
            "player_id",
            "season",
            *[pl.col(n).cast(pl.Float64, strict=False).alias(n) for n in keep],
        )
        college_frames.append(c)
    college = finite(pl.concat(college_frames, how="diagonal_relaxed"))
    return {
        "box": box,
        "team": team,
        "schedule": schedule,
        "player": player,
        "identity": identity,
        "mapping": mapping,
        "combine": read_family("combine"),
        "college": college,
    }


def aggregate_history(frame, key, prefix):
    cols = numeric(frame, [key])
    return frame.group_by(key).agg(
        *[pl.col(c).mean().alias(f"{prefix}__{c}") for c in cols],
        pl.len().alias(f"{prefix}__observations"),
    )


def make_panel(position="RB", horizon="week", origins=(2,), first_year=None):
    """Candidates come from past production; absent completed-period production is zero.

    Season candidates are prior-year participants plus exactly linked drafted combine entrants.
    In-season candidates also include players observed by the origin. No future roster filter.
    """
    raw = load_raw()
    box, player, team, sched = (raw[k] for k in ["box", "player", "team", "schedule"])
    if first_year is None:
        first_year = max(int(box["season"].min()), int(team["season"].min())) + 1
    schedules = sched.group_by("season").agg(pl.col("week").max().alias("end"))
    ends = dict(schedules.iter_rows())
    chunks = []
    audit = []
    entrants = raw["combine"].join(raw["mapping"], on="pfr_id", how="inner")
    for year in range(first_year, LAST_COMPLETE_SEASON + 1):
        for origin in [0] if horizon == "season" else origins:
            end = ends[year] if horizon in {"season", "remaining"} else min(origin + 1, ends[year])
            if end <= origin:
                continue
            history = box.filter(
                (pl.col("season") == year - 1)
                | ((pl.col("season") == year) & (pl.col("week") <= origin))
            )
            cand = (
                history.filter(pl.col("position") == position)
                .sort("season", "week")
                .group_by("player_id", maintain_order=True)
                .agg(pl.col("team").last(), pl.col("player_display_name").last())
            )
            rookies = (
                entrants.filter((pl.col("draft_year") == year) & (pl.col("pos") == position))
                .select(
                    "player_id",
                    pl.col("draft_team").alias("team"),
                    pl.col("player_name").alias("player_display_name"),
                )
                .drop_nulls()
                .unique("player_id")
            )
            rookies = rookies.join(cand.select("player_id"), on="player_id", how="anti")
            cand = pl.concat(
                [
                    cand.with_columns(pl.lit("past NFL observation").alias("candidate_basis")),
                    rookies.with_columns(pl.lit("dated draft / exact ID").alias("candidate_basis")),
                ]
            )
            if not cand.height:
                continue
            before = player.filter(pl.col("season") == year - 1)
            recent = player.filter((pl.col("season") == year) & (pl.col("week") <= origin))
            cand = cand.join(
                aggregate_history(before, "player_id", "prior"), on="player_id", how="left"
            )
            cand = cand.join(
                aggregate_history(recent, "player_id", "current"), on="player_id", how="left"
            )
            last4 = recent.filter(pl.col("week") > origin - 4)
            cand = cand.join(
                aggregate_history(last4, "player_id", "recent4"), on="player_id", how="left"
            )
            college_before = raw["college"].filter(pl.col("season") < year)
            cand = cand.join(
                aggregate_history(college_before, "player_id", "college"),
                on="player_id",
                how="left",
            )
            # Calendar-week sums retain injury/non-participation in the rate denominator.
            rate_history = history.filter(pl.col("season") == (year if origin else year - 1))
            exposure = origin if origin else ends[year - 1]
            sums = rate_history.group_by("player_id").agg(
                *[pl.col(s).sum().alias(f"history_{s}") for s in STATS],
                pl.col("week").n_unique().alias("history_games"),
            )
            cand = cand.join(sums, on="player_id", how="left").with_columns(
                pl.col([f"history_{s}" for s in STATS] + ["history_games"]).fill_null(0),
                pl.lit(exposure).alias("history_weeks"),
            )
            for label, tf in [
                ("team_prior", team.filter(pl.col("season") == year - 1)),
                (
                    "team_current",
                    team.filter((pl.col("season") == year) & (pl.col("week") <= origin)),
                ),
            ]:
                cand = cand.join(aggregate_history(tf, "team", label), on="team", how="left")
            target_games = sched.filter(
                (pl.col("season") == year) & (pl.col("week") > origin) & (pl.col("week") <= end)
            )
            if target_games.filter(
                pl.col("home_score").is_null() | pl.col("away_score").is_null()
            ).height:
                audit.append({"year": year, "origin": origin, "excluded_unknown": cand.height})
                continue
            fut = box.filter(
                (pl.col("season") == year) & (pl.col("week") > origin) & (pl.col("week") <= end)
            )
            # A null in a recorded stat makes that target unknown; only absence of a row is zero.
            target = fut.group_by("player_id").agg(
                *[
                    pl.when(pl.col(s).null_count() == 0)
                    .then(pl.col(s).sum())
                    .otherwise(None)
                    .alias(f"y_{s}")
                    for s in STATS
                ],
                pl.len().alias("target_records"),
            )
            cand = cand.join(target, on="player_id", how="left").with_columns(
                [
                    pl.when(pl.col("target_records").is_null())
                    .then(0)
                    .otherwise(pl.col(f"y_{s}"))
                    .alias(f"y_{s}")
                    for s in STATS
                ]
            )
            unknown = cand.filter(
                pl.any_horizontal([pl.col(f"y_{s}").is_null() for s in STATS])
            ).height
            cand = cand.drop_nulls([f"y_{s}" for s in STATS])
            audit.append({"year": year, "origin": origin, "excluded_unknown": unknown})
            future_team = (
                team.filter(
                    (pl.col("season") == year) & (pl.col("week") > origin) & (pl.col("week") <= end)
                )
                .group_by("team")
                .agg(
                    pl.col("carries").sum().alias("y_team_carries"),
                    pl.col("targets").sum().alias("y_team_targets"),
                    (pl.col("carries") + pl.col("attempts") + pl.col("sacks_suffered"))
                    .sum()
                    .alias("y_team_plays"),
                )
            )
            cand = cand.join(future_team, on="team", how="left").with_columns(
                pl.col("y_team_carries", "y_team_targets", "y_team_plays").fill_null(0),
                pl.lit(year).alias("year"),
                pl.lit(origin).alias("origin"),
                pl.lit(end - origin).alias("weeks"),
                pl.lit(position).alias("position"),
                pl.lit(horizon).alias("horizon"),
            )
            games = (
                pl.concat(
                    [
                        target_games.select(pl.col(f"{side}_team").alias("team"))
                        for side in ["home", "away"]
                    ]
                )
                .group_by("team")
                .len(name="scheduled_games")
            )
            cand = cand.join(games, on="team", how="left").with_columns(
                pl.col("scheduled_games").fill_null(0)
            )
            # Closing lines for the target period are a separately labelled oracle sensitivity.
            markets = []
            for side, sign in [("home", 1), ("away", -1)]:
                markets.append(
                    target_games.select(
                        pl.col(f"{side}_team").alias("team"),
                        (pl.col("total_line") / 2 + sign * pl.col("spread_line") / 2).alias(
                            "market_implied"
                        ),
                        (sign * pl.col("spread_line")).alias("market_spread"),
                    )
                )
            cand = cand.join(pl.concat(markets).group_by("team").mean(), on="team", how="left")
            chunks.append(cand)
    result = pl.concat(chunks, how="diagonal_relaxed")
    birth = raw["identity"].select(pl.col("gsis_id").alias("player_id"), "birth_date").unique()
    birth = birth.filter(~pl.col("player_id").is_duplicated())
    result = (
        result.join(birth, on="player_id", how="left")
        .with_columns(
            (
                (pl.date(pl.col("year"), 9, 1) - pl.col("birth_date").cast(pl.Date)).dt.total_days()
                / 365.25
            ).alias("age")
        )
        .drop("birth_date")
    )
    combine = raw["combine"].join(raw["mapping"], on="pfr_id", how="inner")
    combine = combine.sort("season").unique("player_id", keep="first")
    measures = [c for c in numeric(combine) if c not in {"draft_year", "draft_round", "draft_ovr"}]
    result = result.join(
        combine.select(
            "player_id",
            pl.col("season").alias("combine_year"),
            *[pl.col(c).alias(f"combine__{c}") for c in measures],
        ),
        on="player_id",
        how="left",
    )
    result = result.with_columns(
        [
            pl.when(pl.col("combine_year") <= pl.col("year"))
            .then(pl.col(f"combine__{c}"))
            .otherwise(None)
            .alias(f"combine__{c}")
            for c in measures
        ]
    ).drop("combine_year")
    result = result.with_columns(
        (
            0.1 * (pl.col("y_rushing_yards") + pl.col("y_receiving_yards"))
            + pl.col("y_receptions")
            + 6 * (pl.col("y_rushing_tds") + pl.col("y_receiving_tds"))
        ).alias("y_points"),
        (pl.col("y_rushing_tds") + pl.col("y_receiving_tds")).alias("y_td"),
    )
    features = [
        c
        for c in numeric(result)
        if not c.startswith("y_")
        and c not in {"target_records", "year", "market_implied", "market_spread"}
    ]
    return finite(result).sort("year", "origin", "team", "player_id"), features, pl.DataFrame(audit)


def stability(panel):
    """Descriptive association, not causal evidence or an aging prior."""
    metrics = {
        "carries/calendar week": ("history_carries", "y_carries", None),
        "targets/calendar week": ("history_targets", "y_targets", None),
        "yards/carry": ("history_rushing_yards", "y_rushing_yards", "carries"),
        "yards/target": ("history_receiving_yards", "y_receiving_yards", "targets"),
        "rush TD/carry": ("history_rushing_tds", "y_rushing_tds", "carries"),
    }
    rows = []
    for name, (a, b, den) in metrics.items():
        x = panel[a].to_numpy() / np.maximum(
            panel[f"history_{den}" if den else "history_weeks"].to_numpy(), 1
        )
        y = panel[b].to_numpy() / np.maximum(panel[f"y_{den}" if den else "weeks"].to_numpy(), 1)
        mask = np.isfinite(x + y)
        if den:
            mask &= (panel[f"history_{den}"].to_numpy() > 0) & (panel[f"y_{den}"].to_numpy() > 0)
        rows.append(
            {
                "quantity": name,
                "pairs": int(mask.sum()),
                "correlation": float(np.corrcoef(x[mask], y[mask])[0, 1]),
            }
        )
    return pl.DataFrame(rows)
