"""Outcome-free feature construction shared by historical checks and live scoring.

Keep the frozen GPU campaign untouched. Reuse its aggregation semantics and record
current-source exclusions explicitly. Future outcome labels are never constructed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from ..gpu_lab.data import add_history_views, validate_features
from ..gpu_lab.run import digest, json_write
from ..notebooks import raw_distribution_data as raw

KEY = raw.KEY


def history_inputs(cache: Path) -> dict[str, pl.DataFrame]:
    """Cache verified raw observations rather than rebuilding them for every refresh."""
    contract = {
        "snapshot": digest(raw.SNAPSHOT),
        "loader": digest(Path(raw.__file__)),
        "last_complete_season": raw.LAST_COMPLETE_SEASON,
    }
    receipt = cache / "manifest.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["contract"] != contract:
            raise ValueError("Historical source contract changed; use a new input-cache directory")
        for name, expected in saved["files"].items():
            if digest(cache / name) != expected:
                raise ValueError(f"Historical cache checksum mismatch: {name}")
        return {Path(name).stem: pl.read_parquet(cache / name) for name in saved["files"]}
    cache.mkdir(parents=True, exist_ok=False)
    print("Preparing verified historical observations", flush=True)
    tables = raw.load_raw()
    for name, frame in tables.items():
        frame.write_parquet(cache / f"{name}.parquet")
    json_write(
        receipt,
        {
            "contract": contract,
            "files": {p.name: digest(p) for p in sorted(cache.glob("*.parquet"))},
        },
    )
    return tables


def snapshot_inputs(history, manifest_path: Path, season: int, origin: int):
    """Read only explicit current captures; don't relabel historical data as current."""
    manifest = json.loads(manifest_path.read_text())
    cutoff = manifest["current_observations"]
    if manifest.get("status") != "accepted":
        raise ValueError("Current snapshot must be accepted")
    if cutoff["season"] != season or not 1 <= origin <= cutoff["through_week"]:
        raise ValueError("Requested season/week exceeds the snapshot's completed-week cutoff")
    frames, hashes = {}, {}
    for key in ("weeks", "snaps", "schedule"):
        asset = manifest["assets"][f"current/{key}.parquet"]
        path = raw.ROOT / "data" / asset["object"]
        if digest(path) != asset["sha256"]:
            raise ValueError(f"Current snapshot checksum mismatch: {key}")
        f = pl.read_parquet(path).filter(pl.col("season") == season)
        type_col = "season_type" if "season_type" in f.columns else "game_type"
        if type_col in f.columns:
            f = f.filter(pl.col(type_col) == "REG")
        if key != "schedule":
            f = f.filter(pl.col("week").is_between(1, origin))
        frames[key] = raw.finite(f)
        hashes[key] = asset["sha256"]
    completed = frames["schedule"].filter(pl.col("week") <= origin)
    if (
        completed.is_empty()
        or completed.filter(pl.col("home_score").is_null() | pl.col("away_score").is_null()).height
    ):
        raise ValueError("Snapshot cutoff includes an incomplete scheduled week")
    box = raw.unique_observations(frames["weeks"].drop_nulls("player_id"), KEY).with_columns(
        pl.col("team").replace(raw.TEAM_CODES)
    )
    snaps = raw.unique_observations(frames["snaps"].drop_nulls("player_id"), KEY)
    obs = raw.prefixed(box, KEY, "box").join(
        raw.prefixed(snaps, KEY, "snaps"), on=KEY, how="full", coalesce=True, validate="1:1"
    )
    result = dict(history)
    for name, frame in (("box", box), ("player", obs), ("schedule", frames["schedule"])):
        result[name] = pl.concat(
            [history[name].filter(pl.col("season") < season), frame], how="diagonal_relaxed"
        )
    # No current team-box capture exists. Keep it missing; player sums are not
    # interchangeable with all official team fields (rates, penalties, etc.).
    result["team"] = history["team"].filter(pl.col("season") < season)
    return result, {
        "manifest": str(manifest_path.resolve()),
        "sha256": digest(manifest_path),
        "current_observations": cutoff,
        "assets": hashes,
        "current_sources": ["box", "offensive snaps"],
        "missing_current_sources": ["plays", "NGS", "rosters", "depth", "injuries", "team box"],
    }


def add_supplements(tables, directory, season, origin):
    """Use the original loader's transforms; withhold sources without its time contract."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest["season"] != season:
        raise ValueError("Supplement capture is incomplete or belongs to another season")
    frames, audit = {}, []
    for name, spec in manifest["sources"].items():
        if spec["status"] != "available":
            audit.append({"source": name, "status": "unavailable", "reason": spec["error"]})
            continue
        path = directory / spec["path"]
        if digest(path) != spec["sha256"]:
            raise ValueError(f"Supplement checksum mismatch: {name}")
        f = pl.read_parquet(path)
        if "season" not in f.columns or "week" not in f.columns:
            audit.append(
                {"source": name, "status": "withheld", "reason": "No compatible season/week schema"}
            )
            continue
        f = f.filter((pl.col("season") == season) & pl.col("week").is_between(1, origin))
        for type_col in ("season_type", "game_type"):
            if type_col in f.columns:
                f = f.filter(pl.col(type_col) == "REG")
        if name == "injuries":
            if "date_modified" not in f.columns:
                audit.append(
                    {
                        "source": name,
                        "status": "withheld",
                        "reason": "Missing date_modified required by training contract",
                    }
                )
                continue
            dates = (
                tables["schedule"]
                .group_by("season", "week")
                .agg(pl.col("gameday").cast(pl.Date).max().alias("week_end"))
            )
            f = (
                f.with_columns(pl.col("season", "week").cast(pl.Int64))
                .join(
                    dates.with_columns(pl.col("season", "week").cast(pl.Int64)),
                    on=["season", "week"],
                    how="left",
                )
                .filter(pl.col("date_modified").cast(pl.Date) <= pl.col("week_end"))
            )
        frames[name] = raw.finite(f)
        audit.append(
            {
                "source": name,
                "status": "used",
                "rows_through_cutoff": f.height,
                "sha256": spec["sha256"],
            }
        )
    result = dict(tables)
    supplements = []
    ngs = [frame for name, frame in frames.items() if name.startswith("ngs_")]
    if ngs:
        frames["ngs"] = pl.concat(ngs, how="diagonal_relaxed")
    for name in ("ngs", "rosters", "injuries", "depth"):
        if name not in frames:
            continue
        f = frames[name].rename({"player_gsis_id" if name == "ngs" else "gsis_id": "player_id"})
        f = f.filter(pl.col("player_id").is_not_null())
        cat = [c for c in ("report_status", "practice_status", "status") if c in f.columns]
        if cat:
            f = f.to_dummies(columns=cat)
            # A known categorical observation without a particular state is zero,
            # while no observation from that source remains null after the join.
            for col in tables["player"].columns:
                prefix = f"{name}__"
                if col.startswith(prefix) and any(
                    col[len(prefix) :].startswith(c + "_") for c in cat
                ):
                    source_col = col[len(prefix) :]
                    if source_col not in f.columns:
                        f = f.with_columns(pl.lit(0).alias(source_col))
        if name == "rosters":
            f = f.drop([c for c in ("entry_year", "rookie_year") if c in f.columns])
        supplements.append(raw.prefixed(f, KEY, name))
    if "plays" in frames:
        plays = frames["plays"]
        for role, id_col in (("rush", "rusher_player_id"), ("target", "receiver_player_id")):
            f = plays.filter(
                pl.col(id_col).is_not_null()
                & (pl.col("play_type") != "no_play")
                & (pl.col("play_deleted") == 0)
            ).rename({id_col: "player_id"})
            supplements.extend(
                [
                    raw.prefixed(f, KEY, f"pbp_{role}"),
                    f.group_by(KEY).agg(
                        (pl.col("yardline_100") <= 20).sum().alias(f"pbp_{role}__redzone_count"),
                        (pl.col("yardline_100") <= 5).sum().alias(f"pbp_{role}__inside5_count"),
                    ),
                ]
            )
    current = tables["player"].filter(pl.col("season") == season)
    # Historical schema columns in the current concat are null placeholders. Replace
    # those with observed supplement values, keeping exact source feature names.
    for f in supplements:
        f = f.with_columns(pl.col("season", "week").cast(pl.Int64))
        current = current.drop([c for c in f.columns if c not in KEY and c in current.columns])
        current = current.join(f, on=KEY, how="full", coalesce=True, validate="1:1")
    result["player"] = pl.concat(
        [
            tables["player"].filter(pl.col("season") < season),
            current,
        ],
        how="diagonal_relaxed",
    )
    if "team_box" in frames:
        result["team"] = pl.concat(
            [
                tables["team"].filter(pl.col("season") < season),
                raw.unique_observations(
                    frames["team_box"], ["team", "season", "week"]
                ).with_columns(pl.col("team").replace(raw.TEAM_CODES)),
            ],
            how="diagonal_relaxed",
        )
    return result, {
        "manifest": str(manifest_path.resolve()),
        "sha256": digest(manifest_path),
        "sources": audit,
        "participation": "Current-season FTN intentionally withheld",
    }


def feature_rows(tables, position, year, origin):
    """Build remaining-season rows without looking at future production or scores."""
    if position not in {"RB", "WR", "QB", "TE"}:
        raise ValueError(f"Unsupported position: {position}")
    if not 1 <= origin <= 17:
        raise ValueError("Origin must be a completed regular-season week from 1 through 17")
    box, team, sched = (tables[k] for k in ("box", "team", "schedule"))
    season_schedule = sched.filter(pl.col("season") == year)
    end = season_schedule["week"].max()
    if end is None or end <= origin:
        raise ValueError("No remaining scheduled season")
    observations = tables["player"]
    known = raw.cutoff_player_history(observations, year, origin)
    history = box.filter(
        (pl.col("season") == year - 1) | ((pl.col("season") == year) & (pl.col("week") <= origin))
    )
    cand = (
        history.filter(pl.col("position") == position)
        .sort("season", "week")
        .group_by("player_id", maintain_order=True)
        .agg(pl.col("team").last(), pl.col("player_display_name").last())
    )
    entrants = tables["combine"].join(tables["mapping"], on="pfr_id", how="inner")
    rookies = (
        entrants.filter((pl.col("draft_year") == year) & (pl.col("pos") == position))
        .select(
            "player_id",
            pl.col("draft_team").replace(raw.TEAM_CODES).alias("team"),
            pl.col("player_name").alias("player_display_name"),
        )
        .drop_nulls(["player_id"])
        .unique("player_id")
        .join(cand.select("player_id"), on="player_id", how="anti")
    )
    cand = pl.concat(
        [
            cand.with_columns(pl.lit("past NFL observation").alias("candidate_basis")),
            rookies.with_columns(pl.lit("dated draft / exact ID").alias("candidate_basis")),
        ]
    )
    if cand.is_empty():
        raise ValueError("No eligible candidates")
    recent = known.filter(pl.col("season") == year)
    for prefix, frame in (
        ("prior", known.filter(pl.col("season") == year - 1)),
        ("current", recent),
        ("recent4", recent.filter(pl.col("week") > origin - 4)),
        ("college", tables["college"].filter(pl.col("season") < year)),
    ):
        cand = cand.join(
            raw.aggregate_history(frame, "player_id", prefix), on="player_id", how="left"
        )
    sums = (
        history.filter(pl.col("season") == year)
        .group_by("player_id")
        .agg(
            *[
                pl.when(pl.col(s).null_count() == 0)
                .then(pl.col(s).sum())
                .otherwise(None)
                .alias(f"history_{s}")
                for s in raw.STATS
            ],
            pl.col("week").n_unique().alias("history_games"),
        )
    )
    cand = cand.join(sums, on="player_id", how="left").with_columns(
        *[
            pl.when(pl.col("history_games").is_null())
            .then(0)
            .otherwise(pl.col(f"history_{s}"))
            .alias(f"history_{s}")
            for s in raw.STATS
        ],
        pl.col("history_games").fill_null(0),
        pl.lit(origin).alias("history_weeks"),
    )
    for prefix, frame in (
        ("team_prior", team.filter(pl.col("season") == year - 1)),
        (
            "team_current",
            team.filter((pl.col("season") == year) & (pl.col("week") <= origin)),
        ),
    ):
        cand = cand.join(raw.aggregate_history(frame, "team", prefix), on="team", how="left")
    cand = cand.with_columns(
        pl.lit(year).alias("year"),
        pl.lit(origin).alias("origin"),
        pl.lit(end - origin).alias("weeks"),
        pl.lit(position).alias("position"),
        pl.lit("remaining").alias("horizon"),
    )
    cand = raw.attach_scheduled_games(
        cand, season_schedule, season_schedule.filter(pl.col("week") > origin)
    )
    birth = tables["identity"].select(pl.col("gsis_id").alias("player_id"), "birth_date").unique()
    birth = birth.filter(~pl.col("player_id").is_duplicated())
    cand = (
        cand.join(birth, on="player_id", how="left")
        .with_columns(
            (
                (pl.date(year, 9, 1) - pl.col("birth_date").cast(pl.Date)).dt.total_days() / 365.25
            ).alias("age")
        )
        .drop("birth_date")
    )
    combine = entrants.sort("season").unique("player_id", keep="first")
    measures = [
        c for c in raw.numeric(combine) if c not in {"draft_year", "draft_round", "draft_ovr"}
    ]
    cand = (
        cand.join(
            combine.select(
                "player_id",
                pl.col("season").alias("combine_year"),
                *[pl.col(c).alias(f"combine__{c}") for c in measures],
            ),
            on="player_id",
            how="left",
        )
        .with_columns(
            *[
                pl.when(pl.col("combine_year") <= year)
                .then(pl.col(f"combine__{c}"))
                .otherwise(None)
                .alias(f"combine__{c}")
                for c in measures
            ]
        )
        .drop("combine_year")
    )
    return raw.finite(add_history_views(cand, observations)).sort(
        "year", "origin", "team", "player_id"
    )


def feature_matrix(panel, names):
    """Missing values are allowed; missing feature definitions are a contract error."""
    validate_features(names)
    absent = set(names) - set(panel.columns)
    if absent:
        raise ValueError(f"Missing feature definitions: {sorted(absent)}")
    x = panel.select(names).to_numpy().astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    return x
