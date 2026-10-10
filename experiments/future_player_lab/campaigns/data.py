"""Opening Drive raw features: past-only observations, explicit position targets."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from ..forecasting.data import history_inputs
from ..gpu_lab.data import add_history_views, validate_features
from ..gpu_lab.draft_capital_study import ADDED, draft_features, load_sources
from ..gpu_lab.run import digest, json_write
from ..notebooks import raw_distribution_data as raw

KEY = raw.KEY
ROOT = raw.ROOT
STATS = {
    "RB": list(raw.STATS),
    "WR": list(raw.STATS),
    "TE": list(raw.STATS),
    "QB": [
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ],
}
SCORING = {  # Long scoring formulas remain readable.
    # ruff: noqa: E501
    "RB": "PPR: 0.1*(rushing_yards+receiving_yards)+receptions+6*(rushing_tds+receiving_tds); excludes passing, fumbles and bonuses",
    "WR": "PPR: 0.1*(rushing_yards+receiving_yards)+receptions+6*(rushing_tds+receiving_tds); excludes passing, fumbles and bonuses",
    "TE": "PPR: 0.1*(rushing_yards+receiving_yards)+receptions+6*(rushing_tds+receiving_tds); excludes passing, fumbles and bonuses",
    "QB": "4-point passing TD: .04*passing_yards+4*passing_tds-2*interceptions+.1*rushing_yards+6*rushing_tds; excludes receiving, fumbles and bonuses",
}


def points(position, values):
    if position == "QB":
        return (
            0.04 * values["passing_yards"]
            + 4 * values["passing_tds"]
            - 2 * values["passing_interceptions"]
            + 0.1 * values["rushing_yards"]
            + 6 * values["rushing_tds"]
        )
    return (
        0.1 * (values["rushing_yards"] + values["receiving_yards"])
        + values["receptions"]
        + 6 * (values["rushing_tds"] + values["receiving_tds"])
    )


@lru_cache(maxsize=1)
def draft_sources():
    return load_sources()


def dated_entrants(tables, year):
    """Exact-ID combine entrants, with validated dated identity draft fallback.

    The identity file is a later capture. Only immutable draft-event fields are
    usable, and only when draft year equals the recorded combine season and is
    no later than the forecast season (all supported origins are after the draft).
    Current team/status/experience fields are never used to recover draft facts.
    Ambiguous IDs or conflicting event values are withheld, not guessed.
    """
    mapping = tables["mapping"].drop_nulls(["pfr_id", "player_id"]).unique()
    mapping = mapping.filter(
        ~pl.col("pfr_id").is_duplicated() & ~pl.col("player_id").is_duplicated()
    )
    entrants = tables["combine"].join(mapping, on="pfr_id", how="inner")
    required = {
        "gsis_id",
        "pfr_id",
        "position",
        "draft_year",
        "draft_round",
        "draft_pick",
        "draft_team",
    }
    identity = tables["identity"]
    if not required.issubset(identity.columns):
        return entrants
    records = identity.select(sorted(required)).drop_nulls(["gsis_id", "pfr_id"]).unique()
    records = records.filter(~pl.col("gsis_id").is_duplicated() & ~pl.col("pfr_id").is_duplicated())
    lookup = {(r["gsis_id"], r["pfr_id"]): r for r in records.to_dicts()}
    rows = []
    for original in entrants.to_dicts():
        row = dict(original)
        event = lookup.get((row["player_id"], row["pfr_id"]))
        if event is None or event["position"] != row["pos"]:
            rows.append(row)
            continue
        dy, rd, pick = (event[c] for c in ["draft_year", "draft_round", "draft_pick"])
        valid = all(
            isinstance(v, (int, float)) and np.isfinite(v) and int(v) == v for v in [dy, rd, pick]
        )
        valid = (
            valid
            and 1900 <= dy <= year
            and dy == row["season"]
            and 1 <= rd <= 7
            and 1 <= pick <= 300
        )
        valid = valid and bool(event["draft_team"])
        fields = {
            "draft_year": dy,
            "draft_round": rd,
            "draft_ovr": pick,
            "draft_team": event["draft_team"],
        }
        if valid:
            for column, value in fields.items():
                prior = row[column]
                if column == "draft_team":
                    equal = raw.TEAM_CODES.get(prior, prior) == raw.TEAM_CODES.get(value, value)
                else:
                    equal = prior == value
                if prior is not None and not equal:
                    valid = False
                    break
        if valid:
            for column, value in fields.items():
                if row[column] is None:
                    row[column] = value
        rows.append(row)
    return pl.DataFrame(rows, schema=entrants.schema) if rows else entrants


def base_feature_rows(tables, position, year, origin):
    """Build remaining-season rows without looking at future production or scores."""
    if position not in {"RB", "WR", "QB", "TE"}:
        raise ValueError(f"Unsupported position: {position}")
    if not 0 <= origin <= 17:
        raise ValueError("Origin must be preseason zero or completed week 1 through 17")
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
    entrants = dated_entrants(tables, year)
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
        history.filter(pl.col("season") == (year if origin else year - 1))
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
        pl.lit(
            origin if origin else sched.filter(pl.col("season") == year - 1)["week"].max()
        ).alias("history_weeks"),
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
        pl.lit("remaining" if origin else "season").alias("horizon"),
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


def feature_rows(tables, position, year, origin):
    """Outcome-free inference contract; draft events and QB persistence are explicit."""
    panel = base_feature_rows(tables, position, year, origin)
    combine, rosters, _ = draft_sources()
    extension = draft_features(panel, combine, rosters)
    panel = panel.with_columns(extension.select(ADDED))
    # Recover the same validated draft facts for admitted rookie feature rows.
    # Preserve all nonmissing established draft-source values.
    entrants = dated_entrants(tables, year)
    entrants = entrants.with_columns(
        [
            pl.lit(None, pl.Float64).alias(c)
            for c in ["draft_ovr", "draft_round"]
            if c not in entrants.columns
        ]
    )
    events = (
        entrants.filter((pl.col("season") <= year) & (pl.col("draft_year") <= year))
        .select("player_id", "draft_ovr", "draft_round", "draft_year")
        .unique()
    )
    events = events.filter(~pl.col("player_id").is_duplicated())
    panel = (
        panel.join(events, on="player_id", how="left", validate="m:1")
        .with_columns(
            pl.coalesce(ADDED[0], "draft_ovr").alias(ADDED[0]),
            pl.coalesce(ADDED[1], "draft_round").alias(ADDED[1]),
            pl.coalesce(pl.col(ADDED[2]), pl.lit(year) - pl.col("draft_year")).alias(ADDED[2]),
        )
        .drop("draft_ovr", "draft_round", "draft_year")
    )
    missing = [s for s in STATS[position] if f"history_{s}" not in panel.columns]
    if missing:
        past = tables["box"].filter(
            (pl.col("season") == (year if origin else year - 1))
            & ((pl.col("week") <= origin) if origin else pl.lit(True))
        )
        totals = past.group_by("player_id").agg(
            pl.len().alias("passing_records"),
            *[
                pl.when(pl.col(s).null_count() == 0)
                .then(pl.col(s).sum())
                .otherwise(None)
                .alias(f"history_{s}")
                for s in missing
            ],
        )
        panel = (
            panel.join(totals, on="player_id", how="left", validate="1:1")
            .with_columns(
                *[
                    pl.when(pl.col("passing_records").is_null())
                    .then(0)
                    .otherwise(pl.col(f"history_{s}"))
                    .alias(f"history_{s}")
                    for s in missing
                ]
            )
            .drop("passing_records")
        )
    return panel


def attach_targets(panel, tables, position, year, origin):
    """No recorded row is zero; a recorded unknown field remains an unknown target."""
    scheduled = tables["schedule"].filter((pl.col("season") == year) & (pl.col("week") > origin))
    if (
        scheduled.is_empty()
        or scheduled.filter(pl.col("home_score").is_null() | pl.col("away_score").is_null()).height
    ):
        raise ValueError("Cannot label an incomplete season")
    future = tables["box"].filter((pl.col("season") == year) & (pl.col("week") > origin))
    totals = future.group_by("player_id").agg(
        pl.len().alias("target_records"),
        *[
            pl.when(pl.col(s).null_count() == 0)
            .then(pl.col(s).sum())
            .otherwise(None)
            .alias(f"y_{s}")
            for s in STATS[position]
        ],
    )
    panel = panel.join(totals, on="player_id", how="left", validate="1:1").with_columns(
        *[
            pl.when(pl.col("target_records").is_null())
            .then(0)
            .otherwise(pl.col(f"y_{s}"))
            .alias(f"y_{s}")
            for s in STATS[position]
        ]
    )
    return panel.with_columns(
        points(position, {s: pl.col(f"y_{s}") for s in STATS[position]}).alias("y_points")
    )


def feature_names(panel):
    names = [
        c
        for c in raw.numeric(panel)
        if not c.startswith("y_") and c != "year" and c != "target_records"
    ]
    validate_features(names)
    return names


def matrix(panel, names):
    validate_features(names)
    missing = set(names) - set(panel.columns)
    if missing:
        raise ValueError(f"Missing feature definitions: {sorted(missing)}")
    x = panel.select(names).to_numpy().astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    return x


def load_dataset(position, horizon, cache):
    directory = Path(cache) / f"{position.lower()}_{horizon}"
    contract = {
        "source_sha256": digest(Path(__file__)),
        "raw_loader_sha256": digest(Path(raw.__file__)),
        "dependencies": {str(p.relative_to(ROOT)): digest(p) for p in dependency_paths()},
        "snapshot_sha256": digest(raw.SNAPSHOT),
        "position": position,
        "horizon": horizon,
        "first_year": 2002,
        "last_year": raw.LAST_COMPLETE_SEASON,
    }
    manifest = directory / "dataset.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if (
            saved["contract"] != contract
            or digest(directory / "panel.parquet") != saved["panel_sha256"]
        ):
            raise ValueError("Dataset contract or content changed; use a fresh cache")
        return pl.read_parquet(directory / "panel.parquet"), saved["feature_names"]
    tables = history_inputs(Path(cache) / "history")
    origin = 0 if horizon == "season" else 2
    parts = []
    for year in range(2002, raw.LAST_COMPLETE_SEASON + 1):
        part = feature_rows(tables, position, year, origin)
        parts.append(attach_targets(part, tables, position, year, origin))
        print(f"features {position}/{horizon}/{year}: {part.height} rows", flush=True)
    panel = pl.concat(parts, how="diagonal_relaxed").sort("year", "player_id")
    names = feature_names(panel)
    if position == "TE":
        from .te_features import relevant_features

        names = relevant_features(names)
    directory.mkdir(parents=True, exist_ok=False)
    panel.write_parquet(directory / "panel.parquet")
    json_write(
        manifest,
        {
            "contract": contract,
            "feature_names": names,
            "panel_sha256": digest(directory / "panel.parquet"),
            "rows": panel.height,
            "columns": len(names),
        },
    )
    return panel, names


def dependency_paths():
    return [
        Path(raw.__file__),
        Path(__file__),
        ROOT / "experiments/future_player_lab/forecasting/data.py",
        ROOT / "experiments/future_player_lab/gpu_lab/data.py",
        ROOT / "experiments/future_player_lab/gpu_lab/draft_capital_study.py",
        ROOT / "experiments/future_player_lab/gpu_lab/run.py",
        ROOT / "experiments/future_player_lab/gpu_lab/calibrated_baselines.py",
        ROOT / "experiments/future_player_lab/campaigns/te_features.py",
    ]
