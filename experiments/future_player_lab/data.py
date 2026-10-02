"""Dated observation -> feature -> future-label contracts, independent of old model code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

POSITIONS = ("QB", "RB", "WR", "TE")
TARGETS = ("points", "workload", "yards", "touchdowns", "observed_weeks")
CORE = (
    "league_points",
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)
MEASURES = CORE + (
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_cpoe",
    "sacks_suffered",
    "rushing_first_downs",
    "rushing_epa",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_epa",
    "target_share",
    "air_yards_share",
)
STATIC = (
    "age_at_season",
    "player_experience",
    "draft_pick",
    "draft_round",
    "was_drafted",
    "combine_weight",
    "depth_chart_rank",
    "cutoff_state_observed",
    "cutoff_preseason_rostered",
    "cutoff_preseason_reserve",
    "cutoff_injured_reserve",
    "cutoff_pup_nfi",
    "cutoff_suspended",
    "cutoff_reserve_other",
    "cutoff_transaction_count",
    "cutoff_transaction_recency_days",
    "cutoff_team_observed",
    "known_absence_games",
    "known_suspension_games",
    "team_changed",
    "known_vacated_target_share",
    "known_vacated_carry_share",
    "team_context_observed_share",
)
KEY = ["player_id", "forecast_season", "forecast_cutoff_date"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(root, config):
    gold = root / config["gold_root"]
    path = gold / "manifest.json"
    if digest(path) != config["gold_manifest_sha256"]:
        raise ValueError("Gold manifest changed")
    manifest = json.loads(path.read_text())
    tables, hashes = {}, {str(path.relative_to(root)): digest(path)}
    for name in (
        "preseason_features",
        "season_outcomes",
        "nfl_player_weeks",
        "nfl_weekly_usage",
        "college_annual",
        "college_identity_links",
    ):
        spec = manifest["tables"][name]
        p = gold / spec["path"]
        sha = digest(p)
        if sha != spec["sha256"]:
            raise ValueError(f"Gold table checksum mismatch: {name}")
        hashes[str(p.relative_to(root))] = sha
        tables[name] = pl.read_parquet(p)
    return tables, hashes


def numeric(frame, names):
    result = frame.select(pl.col(c).cast(pl.Float64) for c in names).to_numpy().copy()
    result[~np.isfinite(result)] = np.nan
    return result


def validate_candidates(features):
    if features.select(KEY).is_duplicated().any():
        raise ValueError("Duplicate forecast key")
    if features.filter(pl.col("source_season") >= pl.col("forecast_season")).height:
        raise ValueError("Source observations cross preseason cutoff")
    for name in (
        "availability_latest_known_on",
        "cutoff_last_transaction_date",
        "depth_chart_date",
    ):
        if name in features:
            dates = features[name].cast(pl.String).str.slice(0, 10).str.to_date(strict=False)
            if (dates > features["forecast_cutoff_date"].cast(pl.Date)).fill_null(False).any():
                raise ValueError(f"Future evidence: {name}")


def horizon_end(year, origin, horizon):
    last = 18 if year >= 2021 else 17
    if horizon in ("season", "remaining"):
        return last
    return min(last, origin + (1 if horizon == "next_week" else 4))


def summarize(history, times, origin_time, windows):
    """Only recorded observations; nulls never become measured zero features."""
    chunks = []
    for window in windows:
        h = history[-window:] if window else history
        t = times[-window:] if window else times
        finite = np.isfinite(h)
        count = finite.sum(axis=0)
        total = np.where(finite, h, 0).sum(axis=0)
        mean = np.divide(total, count, out=np.full(h.shape[1], np.nan), where=count > 0)
        centered = np.where(finite, h - mean, 0)
        std = np.sqrt(
            np.divide(
                (centered**2).sum(axis=0), count, out=np.full(h.shape[1], np.nan), where=count > 0
            )
        )
        # Per-channel slopes respect missing observations and calendar spacing.
        tx = t[:, None] - origin_time
        tm = np.divide((finite * tx).sum(axis=0), count, out=np.zeros(h.shape[1]), where=count > 0)
        dt = np.where(finite, tx - tm, 0)
        denom = (dt * dt).sum(axis=0)
        slope = np.divide(
            (dt * centered).sum(axis=0), denom, out=np.full(h.shape[1], np.nan), where=denom > 0
        )
        chunks.extend([mean, std, slope, count.astype(float)])
    return np.concatenate(chunks)


def labels(block, position):
    """No future row = zero recorded production; a null within a row stays unknown."""
    sums = {name: block[:, i].sum() for i, name in enumerate(CORE)}
    if position == "QB":
        workload, yards = sums["attempts"], sums["passing_yards"]
        td = sums["passing_tds"]
    elif position == "RB":
        workload = sums["carries"] + sums["targets"]
        yards = sums["rushing_yards"] + sums["receiving_yards"]
        td = sums["rushing_tds"] + sums["receiving_tds"]
    else:
        workload, yards, td = sums["targets"], sums["receiving_yards"], sums["receiving_tds"]
    return np.array([sums["league_points"], workload, yards, td, len(block)], dtype=float)


def build_panel(tables, config, progress=print, *, include_unlabeled=False):
    features = tables["preseason_features"]
    validate_candidates(features)
    candidates = features.join(
        tables["season_outcomes"],
        on=KEY,
        validate="1:1",
        how="left" if include_unlabeled else "inner",
    )
    candidates = candidates.with_columns(pl.col("outcome_complete").fill_null(False))
    candidates = candidates.filter(
        pl.col("position").is_in(config["positions"])
        & (pl.col("outcome_complete") | pl.lit(include_unlabeled))
        & (pl.col("forecast_season") <= max(config["evaluation_years"]))
    ).sort("forecast_season", "position", "player_id")
    if candidates.is_empty():
        raise ValueError("No eligible candidates for the requested forecast contract")
    # Population is determined by historical preseason membership, never future activity.
    weeks = tables["nfl_player_weeks"].filter(
        (pl.col("season_type") == "REG")
        & pl.col("player_id").is_in(candidates["player_id"].unique().implode())
    )
    if weeks.select("player_id", "season", "week").is_duplicated().any():
        raise ValueError("Duplicate weekly observation")
    usage = tables["nfl_weekly_usage"]
    usage_names = [
        c for c, dtype in usage.schema.items() if dtype.is_numeric() or dtype == pl.Boolean
    ]
    usage_names = [c for c in usage_names if c not in ("season", "week")]
    usage = usage.select("player_id", "season", "week", *usage_names).rename(
        {c: f"usage_{c}" for c in usage_names}
    )
    weeks = weeks.join(usage, on=["player_id", "season", "week"], how="left", validate="1:1")
    channels = list(MEASURES) + [f"usage_{c}" for c in usage_names]
    lookup = {}
    for (pid,), block in (
        weeks.sort("season", "week").partition_by("player_id", as_dict=True).items()
    ):
        times = block["season"].to_numpy() * 32 + block["week"].to_numpy()
        lookup[pid] = (times, numeric(block, channels))
    windows, length = config["history_windows"], config["sequence_length"]
    college_names, college_lookup = [], {}
    if "college_annual" in tables:
        college = tables["college_annual"]
        links = tables["college_identity_links"].filter(pl.col("status") == "linked")
        links = links.select("college_id", "player_id").drop_nulls().unique()
        college_names = [
            c
            for c, dt in college.schema.items()
            if (dt.is_numeric() or dt == pl.Boolean) and c != "season"
        ]
        college = college.join(links, on="college_id", how="inner", validate="m:1")
        for (pid,), block in college.sort("season").partition_by("player_id", as_dict=True).items():
            college_lookup[pid] = block
    names = [f"static:{c}" for c in STATIC] + [
        "static:rookie",
        "static:market_only",
        "context:year",
        "context:origin",
        "context:horizon_weeks",
        "context:history_count",
        "context:history_gap",
    ]
    groups = [n.split(":")[0] for n in names]
    for lag in (1, 2, 3):
        names.extend(f"college:{lag}:{c}" for c in college_names)
        groups.extend("college:" + c for c in college_names)
    for window in windows:
        for stat in ("mean", "std", "slope", "count"):
            names.extend(f"summary:{window}:{stat}:{c}" for c in channels)
            groups.extend(f"history:{c}" for c in channels)
    for lag in range(1, length + 1):
        names.extend(f"sequence:{lag}:{c}" for c in CORE)
        groups.extend(f"history:{c}" for c in CORE)
    rows, matrix, outcomes, baselines = [], [], [], []
    for i, row in enumerate(candidates.iter_rows(named=True)):
        year, pid = row["forecast_season"], row["player_id"]
        times, values = lookup.get(pid, (np.array([], dtype=int), np.empty((0, len(channels)))))
        college_values = np.full((3, len(college_names)), np.nan)
        if pid in college_lookup:
            block = (
                college_lookup[pid]
                .filter(
                    (pl.col("season") < year)
                    & (pl.col("season_end_date") < str(row["forecast_cutoff_date"]))
                )
                .tail(3)
            )
            if block.height:
                college_values[: block.height] = numeric(block, college_names)[::-1]
        history_cache = {}
        for horizon in config["horizons"]:
            for origin in [0] if horizon == "season" else config["origins"]:
                end = horizon_end(year, origin, horizon)
                if end <= origin:
                    continue
                cutoff = year * 32 + origin
                past = times <= cutoff
                future = (times > cutoff) & (times <= year * 32 + end)
                hist, ht = values[past], times[past]
                y = (
                    labels(values[future, : len(CORE)], row["position"])
                    if row["outcome_complete"]
                    else np.full(len(TARGETS), np.nan)
                )
                context = [row.get(c) if row.get(c) is not None else np.nan for c in STATIC]
                context += [
                    float(row["player_population"] == "rookie"),
                    float(row["player_population"] == "market_only"),
                    year,
                    origin,
                    end - origin,
                    len(hist),
                    cutoff - ht[-1] if len(ht) else np.nan,
                ]
                if origin not in history_cache:
                    seq = np.full((length, len(CORE)), np.nan)
                    n = min(length, len(hist))
                    if n:
                        seq[:n] = hist[-n:, : len(CORE)][::-1]
                    history_cache[origin] = np.r_[summarize(hist, ht, cutoff, windows), seq.ravel()]
                x = np.r_[context, college_values.ravel(), history_cache[origin]]
                # Persistence benchmark uses elapsed calendar weeks, including bye/nonappearance.
                start = year * 32 if origin else (year - 1) * 32
                b = values[(times > start) & (times <= cutoff)]
                if not origin:
                    b = values[(times > start) & (times < year * 32)]
                denom = origin or (18 if year - 1 >= 2021 else 17)
                baseline = labels(b[:, : len(CORE)], row["position"]) / denom
                rows.append(
                    {
                        "player_id": pid,
                        "name": row["player_display_name"],
                        "position": row["position"],
                        "year": year,
                        "origin": origin,
                        "horizon": horizon,
                        "end": end,
                        "exposure": end - origin,
                        "population": row["player_population"],
                        "forecast_cutoff_date": str(row["forecast_cutoff_date"]),
                        "history_max_time": int(ht[-1]) if len(ht) else None,
                        "cutoff_time": cutoff,
                    }
                )
                matrix.append(x.astype(np.float32))
                outcomes.append(y)
                baselines.append(baseline)
        if i % 2000 == 0:
            progress(f"panel: {i}/{candidates.height} players; {len(rows)} examples", flush=True)
    frame = pl.DataFrame(rows)
    x = np.asarray(matrix, dtype=np.float32)
    y = np.asarray(outcomes, dtype=np.float64)
    baseline = np.asarray(baselines, dtype=np.float64)
    return frame, x, y, baseline, names, groups
