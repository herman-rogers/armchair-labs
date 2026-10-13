"""Frozen, earlier-only weekly player panel for Opening Drive v3.

A row is admitted using prior-season participation, a dated draft event, or an
already completed current-season appearance. Future participation never admits a
row. Missing appearances are zero only for fully captured completed team games.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ...notebooks.raw_distribution_data import TEAM_CODES
from ..data import dated_entrants

ROOT = Path(__file__).resolve().parents[4]
HISTORY = ROOT / "experiments/future_player_lab/runs/opening_drive_v1_001/inputs/history"
CAMPAIGN = ROOT / "data/campaigns/opening_drive_v2"
SNAPSHOT = ROOT / "data/raw/snapshots/weekly_20261008_w4_r3/manifest.json"
AVAILABILITY = ROOT / "data/research/opening_drive_v2_sources"
DEFAULT_OUTPUT = ROOT / "data/research/opening_drive_v3_20261011/panel"
POSITIONS = ("QB", "RB", "WR", "TE")
# Sealed capture has 0–66 targets/year against >9,000 receptions in these years.
# Even apparently plausible zero rows are unsupported; mask the whole capture.
MISSING_TARGET_CAPTURE_YEARS = tuple(range(2003, 2009))
TEAM_ALIASES = {**TEAM_CODES, "ARZ": "ARI", "BLT": "BAL", "HST": "HOU", "CLV": "CLE", "SL": "LA"}
SCORING = {
    "QB": ".04 passing yards + 4 passing TD - 2 INT + .1 rush yards + 6 rush TD",
    "RB/WR/TE": ".1 (rush+receiving yards) + receptions + 6 (rush+receiving TD)",
}
STAT_COLUMNS = [
    "attempts",
    "carries",
    "targets",
    "receptions",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "passing_interceptions",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_points(box: pd.DataFrame) -> pd.Series:
    """Campaign scoring, preserving any missing required component as missing."""
    qb = (
        0.04 * box.passing_yards
        + 4 * box.passing_tds
        - 2 * box.passing_interceptions
        + 0.1 * box.rushing_yards
        + 6 * box.rushing_tds
    )
    skill = (
        0.1 * (box.rushing_yards + box.receiving_yards)
        + box.receptions
        + 6 * (box.rushing_tds + box.receiving_tds)
    )
    return skill.where(box.position != "QB", qb)


def _read_sources():
    receipt = json.loads((HISTORY / "manifest.json").read_text())
    hashes = {str(HISTORY / "manifest.json"): digest(HISTORY / "manifest.json")}
    tables = {}
    for name in ("box", "schedule", "combine", "mapping", "identity"):
        path = HISTORY / f"{name}.parquet"
        if digest(path) != receipt["files"][path.name]:
            raise ValueError(f"Historical checksum mismatch: {path}")
        hashes[str(path)] = digest(path)
        tables[name] = pl.read_parquet(path)
    snap = json.loads(SNAPSHOT.read_text())
    if snap["status"] != "accepted":
        raise ValueError("Current snapshot is not accepted")
    current_season = snap["current_observations"]["season"]
    through = snap["current_observations"]["through_week"]
    hashes[str(SNAPSHOT)] = digest(SNAPSHOT)
    for asset, table in (("weeks", "box"), ("schedule", "schedule")):
        spec = snap["assets"][f"current/{asset}.parquet"]
        path = ROOT / "data" / spec["object"]
        if digest(path) != spec["sha256"]:
            raise ValueError(f"Current snapshot checksum mismatch: {path}")
        hashes[str(path)] = digest(path)
        frame = pl.read_parquet(path).filter(pl.col("season") == current_season)
        type_col = "season_type" if "season_type" in frame.columns else "game_type"
        frame = frame.filter(pl.col(type_col) == "REG")
        if asset == "weeks":
            frame = frame.filter(pl.col("week") <= through)
        tables[table] = pl.concat([tables[table], frame], how="diagonal_relaxed")
    events = pd.DataFrame()
    path = AVAILABILITY / "events.parquet"
    if path.exists():
        manifest = json.loads((AVAILABILITY / "manifest.json").read_text())
        if digest(path) != manifest["files"][path.name]:
            raise ValueError("Availability source checksum mismatch")
        hashes[str(path)] = digest(path)
        hashes[str(AVAILABILITY / "manifest.json")] = digest(AVAILABILITY / "manifest.json")
        events = pl.read_parquet(path).to_pandas()
    hashes[str(CAMPAIGN / "manifest.json")] = digest(CAMPAIGN / "manifest.json")
    return tables, events, current_season, through, hashes


def load_v2(schedule: pd.DataFrame, hashes: dict):
    """Exact published historical ensemble totals; never recreate a newer v2 fit."""
    lookup = {}
    for position in POSITIONS:
        for horizon, origin in (("season", 0), ("remaining", 2)):
            directory = CAMPAIGN / f"{position.lower()}_{horizon}"
            path = directory / "artifacts/evaluation_predictions.parquet"
            hashes[str(path)] = digest(path)
            predictions = (
                pl.read_parquet(path)
                .filter((pl.col("target") == "points") & (pl.col("recipe") == "ensemble"))
                .to_dicts()
            )
            for r in predictions:
                games = schedule[(schedule.season == r["year"]) & (schedule.week > origin)]
                count = int(((games.home_team == r["team"]) | (games.away_team == r["team"])).sum())
                if count:
                    lookup[(r["year"], origin, r["player_id"])] = (
                        float(r["prediction"]) / count,
                        r["team"],
                        position,
                        r["player_display_name"],
                        "v2_frozen_historical_ensemble",
                    )
            for path in sorted((directory / "forecasts").glob("*.json")):
                payload = json.loads(path.read_text())
                hashes[str(path)] = digest(path)
                for r in payload["rows"]:
                    count = r.get("scheduled_games", 0)
                    if count:
                        lookup[(payload["season"], payload["origin_week"], r["player_id"])] = (
                            r["predictions"]["points"] / count,
                            r["team"],
                            position,
                            r["player_name"],
                            "v2_frozen_published_forecast",
                        )
    return lookup


def availability_lookup(events: pd.DataFrame):
    """Collapse same-time conflicting observations before any as-of lookup."""
    if events.empty:
        return {}
    grouped = events.groupby(["player_id", "available_at"], sort=False)
    states = grouped.agg(
        status=("status", "first"),
        team=("team", "first"),
        statuses=("status", "nunique"),
        teams=("team", "nunique"),
    ).reset_index()
    states.loc[states.statuses > 1, "status"] = "UNKNOWN"
    states.loc[states.teams > 1, "team"] = None
    states["conflict"] = (states.statuses > 1) | (states.teams > 1)
    states = states.sort_values("available_at")
    return {
        pid: (
            g.available_at.astype("datetime64[ns, UTC]").astype("int64").to_numpy(),
            g.to_dict("records"),
        )
        for pid, g in states.groupby("player_id", sort=False)
    }


def history_summary(records, season, week):
    """Records are exclusively completed earlier rows; sample variance uses ddof=1."""
    recent = records[-32:]
    values = np.asarray([r["points"] for r in recent], dtype=float)
    good = values[np.isfinite(values)]
    out = {
        "player_history_count": float(len(good)),
        "player_history_mean": float(good.mean()) if len(good) else np.nan,
        "player_history_variance": float(good.var(ddof=1)) if len(good) > 1 else np.nan,
        "history_missing_count": float(len(values) - len(good)),
    }
    for name, subset in (
        ("recent4", records[-4:]),
        ("recent8", records[-8:]),
        ("current", [r for r in records if r["season"] == season]),
        ("prior", [r for r in records if r["season"] == season - 1]),
    ):
        out[f"{name}_count"] = float(sum(math.isfinite(r["points"]) for r in subset))
        for stat in (
            "points",
            "attempts",
            "carries",
            "targets",
            "receptions",
            "passing_yards",
            "rushing_yards",
            "receiving_yards",
        ):
            vals = [r.get(stat, np.nan) for r in subset]
            vals = [v for v in vals if math.isfinite(v)]
            out[f"{name}_{stat}_mean"] = float(sum(vals) / len(vals)) if vals else np.nan
        vals = [r["points"] for r in subset if math.isfinite(r["points"])]
        out[f"{name}_points_variance"] = (
            float(sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / (len(vals) - 1))
            if len(vals) > 1
            else np.nan
        )
        out[f"{name}_appearance_rate"] = (
            float(sum(r["appeared"] for r in subset) / len(subset)) if subset else np.nan
        )
    out["points_trend4_vs8"] = out["recent4_points_mean"] - out["recent8_points_mean"]
    out["usage_trend4_vs8"] = sum(
        out[f"recent4_{s}_mean"] - out[f"recent8_{s}_mean"]
        for s in ("attempts", "carries", "targets")
    )
    prior_appearances = [r for r in records if r["appeared"]]
    if prior_appearances:
        last = prior_appearances[-1]
        out["weeks_since_appearance"] = float((season - last["season"]) * 18 + week - last["week"])
        out["last_points"] = last["points"]
    else:
        out["weeks_since_appearance"] = np.nan
        out["last_points"] = np.nan
    return out


def construct_panel(
    box,
    schedule,
    entrants,
    baselines,
    events=None,
    *,
    start_year=2003,
    end_year=2026,
    current_season=2026,
    through_week=4,
):
    """Pure panel builder, also used by perturbation and admission tests."""
    box = box.copy()
    box = box[box.position.isin(POSITIONS)].sort_values(["season", "week", "player_id"])
    if box.duplicated(["season", "week", "player_id"]).any():
        raise ValueError("Duplicate player-week observations")
    box["team"] = box.team.replace(TEAM_ALIASES)
    box["target_invalid"] = (
        box.targets.isna()
        | (box.targets < box.receptions)
        | (box.targets < 0)
        | box.season.isin(MISSING_TARGET_CAPTURE_YEARS)
    )
    box.loc[box.target_invalid, "targets"] = np.nan
    box["points"] = score_points(box)
    schedule = schedule.copy()
    schedule["home_team"] = schedule.home_team.replace(TEAM_ALIASES)
    schedule["away_team"] = schedule.away_team.replace(TEAM_ALIASES)
    if "game_type" in schedule:
        schedule = schedule[schedule.game_type == "REG"]
    schedule["gameday"] = pd.to_datetime(schedule.gameday, utc=True)
    observations = {
        (int(y), int(w)): g.to_dict("records") for (y, w), g in box.groupby(["season", "week"])
    }
    calendar = {}
    week_dates = {}
    complete = {}
    for (y, w), games in schedule.groupby(["season", "week"]):
        y, w = int(y), int(w)
        week_dates[(y, w)] = games.gameday.max()
        complete[(y, w)] = bool(games.home_score.notna().all() and games.away_score.notna().all())
        for game in games.to_dict("records"):
            for side, other in (("home", "away"), ("away", "home")):
                calendar[(y, w, game[f"{side}_team"])] = {
                    "opponent": game[f"{other}_team"],
                    "game_id": game["game_id"],
                    "home": float(side == "home"),
                    "gameday": game["gameday"],
                }
    # An absent player row is zero only when both opponents have captured box data.
    captured = set(zip(box.season, box.week, box.team, strict=True))
    availability = availability_lookup(events if events is not None else pd.DataFrame())
    histories = defaultdict(list)
    last_observation = {}
    opponent_history = defaultdict(list)
    rows = []
    for year in range(int(box.season.min()), end_year + 1):
        previous = {pid: r for pid, r in last_observation.items() if r["season"] == year - 1}
        candidates = dict(previous)
        for r in entrants.get(year, []):
            candidates.setdefault(r["player_id"], dict(r, season=year, week=0))
        season_games = schedule[schedule.season == year]
        if season_games.empty:
            continue
        max_week = through_week + 1 if year == current_season else int(season_games.week.max())
        for week in range(1, max_week + 1):
            now = observations.get((year, week), [])
            observed = {r["player_id"]: r for r in now}
            cutoff = (
                week_dates[(year, week - 1)] + pd.Timedelta(days=1, hours=12)
                if week > 1
                else season_games.gameday.min() + pd.Timedelta(hours=12)
            )
            next_histories = []
            for pid, known in sorted(candidates.items()):
                pos = known["position"]
                if pos not in POSITIONS:
                    continue
                state = None
                if pid in availability:
                    times, states = availability[pid]
                    ix = int(np.searchsorted(times, cutoff.value, side="right")) - 1
                    if ix >= 0 and states[ix]["available_at"].year >= year - 1:
                        state = states[ix]
                if state and state["status"] == "RET":
                    continue
                team = TEAM_ALIASES.get(known["team"], known["team"])
                # Latest dated positive membership updates stale team context; CUT
                # does not establish a new team. No current outcome team is used.
                if (
                    state
                    and pd.notna(state["team"])
                    and state["status"] not in ("CUT", "UNKNOWN", "RET_ANNOUNCED")
                ):
                    team = TEAM_ALIASES.get(state["team"], state["team"])
                game = calendar.get((year, week, team))
                team_known = team in set(season_games.home_team) | set(season_games.away_team)
                is_bye = bool(team_known and game is None)
                actual = observed.get(pid)
                complete_week = complete.get((year, week), False) and (
                    year < current_season or week <= through_week
                )
                team_captured = bool(
                    game
                    and (year, week, team) in captured
                    and (year, week, game["opponent"]) in captured
                )
                # A real appearance can identify a previously unknown or changed
                # team for labels, but never modifies the pregame feature team.
                valid = complete_week and (actual is not None or team_captured or is_bye)
                points = (
                    float(actual["points"])
                    if actual is not None and complete_week
                    else (0.0 if valid else np.nan)
                )
                features = history_summary(histories[pid], year, week)
                origin = 0 if week <= 2 else 2
                if year == current_season and week == through_week + 1:
                    origin = through_week
                base = baselines.get((year, origin, pid))
                fallback = features["recent8_points_mean"]
                if not np.isfinite(fallback):
                    fallback = features["prior_points_mean"]
                if not np.isfinite(fallback):
                    fallback = 0.0
                features.update(
                    v2_weekly_mean=(base[0] if (year, week, base[1]) in calendar else 0.0)
                    if base
                    else (0.0 if is_bye else float(fallback)),
                    v2_exact_available=float(base is not None),
                    week_index=float(week),
                    is_bye=float(is_bye),
                    is_home=game["home"] if game else np.nan,
                    rookie=float(known.get("draft_year", 0) == year),
                    draft_round=float(known.get("draft_round", np.nan) or np.nan),
                    draft_pick=float(known.get("draft_ovr", np.nan) or np.nan),
                    availability_known=float(state is not None and state["status"] != "UNKNOWN"),
                    availability_conflict=float(state["conflict"]) if state else 0.0,
                    availability_age_days=(cutoff - state["available_at"]).total_seconds() / 86400
                    if state
                    else np.nan,
                )
                for status in ("ACT", "RES", "PUP", "SUS", "DEV", "CUT", "INA", "EXE", "UNKNOWN"):
                    features[f"availability_{status.lower()}"] = float(
                        (state["status"] if state else "UNKNOWN") == status
                    )
                opp = game["opponent"] if game else None
                prev_opp = opponent_history[(opp, pos)][-8:]
                features["opponent_prior8_points_allowed"] = (
                    float(np.mean(prev_opp)) if prev_opp else np.nan
                )
                features["opponent_history_count"] = float(len(prev_opp))
                row = dict(
                    player_id=pid,
                    player_name=known.get("player_display_name", known.get("player_name", pid)),
                    position=pos,
                    season=year,
                    week=week,
                    origin_week=week - 1,
                    team=team,
                    opponent=opp,
                    game_id=game["game_id"] if game else None,
                    actual_team=actual["team"] if actual else None,
                    target_points=points,
                    label_observed=bool(np.isfinite(points)),
                    appeared=bool(actual is not None),
                    scheduled_game=int(game is not None),
                    team_changed_in_outcome=bool(actual is not None and actual["team"] != team),
                    baseline_source=base[4] if base else "earlier_history_proxy_not_v2",
                    baseline_team=base[1] if base else team,
                    feature_cutoff=cutoff.isoformat(),
                    **features,
                )
                if year >= start_year:
                    rows.append(row)
                if complete_week and (not is_bye or actual is not None):
                    rec = dict(
                        season=year, week=week, points=points, appeared=bool(actual is not None)
                    )
                    rec.update(
                        {
                            s: actual.get(s, np.nan) if actual else (0.0 if valid else np.nan)
                            for s in STAT_COLUMNS
                        }
                    )
                    next_histories.append((pid, rec))
            # Update state only after all predictions for this week are constructed.
            for pid, rec in next_histories:
                histories[pid].append(rec)
            if year < current_season or week <= through_week:
                for r in now:
                    pid = r["player_id"]
                    if pid not in candidates:
                        histories[pid].append(dict(r, appeared=True))
                    candidates[pid] = dict(candidates.get(pid, {}), **r)
                    last_observation[pid] = candidates[pid]
                sums = defaultdict(float)
                for r in now:
                    g = calendar.get((year, week, r["team"]))
                    if g and math.isfinite(r["points"]):
                        sums[(g["opponent"], r["position"])] += r["points"]
                for key, total in sums.items():
                    opponent_history[key].append(total)
        print(f"weekly panel {year}: {len(rows):,} cumulative rows", flush=True)
    panel = pd.DataFrame(rows)
    metadata = {
        "player_id",
        "player_name",
        "position",
        "season",
        "week",
        "origin_week",
        "team",
        "opponent",
        "game_id",
        "actual_team",
        "target_points",
        "label_observed",
        "appeared",
        "scheduled_game",
        "team_changed_in_outcome",
        "baseline_source",
        "baseline_team",
        "feature_cutoff",
    }
    features = [c for c in panel.columns if c not in metadata]
    # Training callers receive an explicit allowlist; diagnostic outcomes are never features.
    return panel, features


def attach_injuries(panel, injuries):
    """Latest dated report known at the forecast cutoff; undated reports withheld."""
    out = panel.copy()
    right = injuries.dropna(subset=["gsis_id", "date_modified"]).copy()
    right["date_modified"] = pd.to_datetime(right.date_modified, utc=True).astype(
        "datetime64[ns, UTC]"
    )
    right = (
        right.groupby(["gsis_id", "date_modified", "season"])
        .agg(report_status=("report_status", "first"), status_count=("report_status", "nunique"))
        .reset_index()
        .rename(columns={"gsis_id": "player_id", "season": "injury_season"})
    )
    right.loc[right.status_count > 1, "report_status"] = "Conflict"
    left = out[["player_id", "season", "feature_cutoff"]].copy()
    left["_row"] = np.arange(len(left))
    left["cutoff"] = pd.to_datetime(left.feature_cutoff, utc=True).astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(
        left.sort_values("cutoff"),
        right.sort_values("date_modified"),
        left_on="cutoff",
        right_on="date_modified",
        by="player_id",
        direction="backward",
    )
    merged = merged.sort_values("_row")
    known = merged.injury_season.to_numpy() == merged.season.to_numpy()
    out["injury_report_known"] = known.astype(float)
    age = (merged.cutoff - merged.date_modified).dt.total_seconds().to_numpy() / 86400
    out["injury_report_age_days"] = np.where(known, age, np.nan)
    for status in ("Out", "Doubtful", "Questionable", "Probable"):
        out[f"injury_{status.lower()}"] = np.where(
            known, (merged.report_status == status).to_numpy().astype(float), np.nan
        )
    return out


def build_panel(output_dir=DEFAULT_OUTPUT, start_year=2003, end_year=2026):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, events, current, through, hashes = _read_sources()
    box = tables["box"].to_pandas()
    schedule = tables["schedule"].to_pandas()
    entrants = {}
    for year in range(int(box.season.min()), end_year + 1):
        f = dated_entrants(tables, year).filter(
            (pl.col("draft_year") == year) & pl.col("pos").is_in(POSITIONS)
        )
        entrants[year] = [
            dict(
                r,
                position=r["pos"],
                team=TEAM_ALIASES.get(r["draft_team"], r["draft_team"]),
                player_display_name=r["player_name"],
            )
            for r in f.to_dicts()
        ]
    baselines = load_v2(schedule, hashes)
    panel, features = construct_panel(
        box,
        schedule,
        entrants,
        baselines,
        events,
        start_year=start_year,
        end_year=end_year,
        current_season=current,
        through_week=through,
    )
    injury_path = AVAILABILITY / "injuries.parquet"
    injury_manifest = json.loads((AVAILABILITY / "manifest.json").read_text())
    if digest(injury_path) != injury_manifest["files"][injury_path.name]:
        raise ValueError("Injury source checksum mismatch")
    hashes[str(injury_path)] = digest(injury_path)
    panel = attach_injuries(panel, pl.read_parquet(injury_path).to_pandas())
    features += [c for c in panel.columns if c.startswith("injury_")]
    invalid = box[box.position.isin(POSITIONS)].assign(
        invalid_target=lambda f: f.targets.isna() | (f.targets < f.receptions) | (f.targets < 0)
    )
    defects = (
        invalid.groupby("season")
        .agg(rows=("player_id", "size"), invalid_targets=("invalid_target", "sum"))
        .reset_index()
        .to_dict("records")
    )
    path = output_dir / "panel.parquet"
    panel.to_parquet(path, index=False)
    manifest = dict(
        schema_version=2,
        rows=len(panel),
        features=features,
        start_year=start_year,
        end_year=end_year,
        current_season=current,
        current_through_week=through,
        current_prediction_week=through + 1,
        panel_sha256=digest(path),
        source_hashes=hashes,
        source_code_sha256=digest(__file__),
        scoring=SCORING,
        invalid_target_counts_by_year=defects,
        source_wide_missing_target_years=list(MISSING_TARGET_CAPTURE_YEARS),
        source_wide_target_masked_rows=int(
            box[box.position.isin(POSITIONS)].season.isin(MISSING_TARGET_CAPTURE_YEARS).sum()
        ),
        population=(
            "Previous-season participants + exact-ID dated draft entrants + already "
            "observed current-season participants; formal dated retired excluded"
        ),
        baseline=(
            "Published v2 historical ensemble 2021-25; earlier-only persistence proxy "
            "elsewhere. Exact paired comparisons require v2_exact_available=1. Season "
            "baseline weeks1-2; frozen Week2 remaining baseline weeks3+; published Week4 "
            "rescore for live Week5."
        ),
        data_policy=(
            "Missing appearance is zero only in completely captured completed week or known "
            "bye. Missing component stays NaN. targets<receptions or negative is missing; "
            "not silently zero. Pregame team from past appearance or dated availability; "
            "actual team is diagnostic only."
        ),
        limitations=[
            "Historical source and event vintage is retrospective, not archived as-published.",
            (
                "Schedule treated as known; no closing lines, actual weather, starters, or "
                "future roster used."
            ),
            (
                "Draft-linked rookies admitted; undrafted unseen newcomers enter only after "
                "first observed week."
            ),
            (
                "Byes retained explicitly; train/evaluate scheduled_game=1 to avoid trivial "
                "bye accuracy."
            ),
            (
                "Injury reports require an explicit modification timestamp before cutoff; "
                "undated reports (including current snapshot entries) are withheld."
            ),
            (
                "Player variation uses last32 eligible non-bye weeks including captured "
                "missed appearances; histories span seasons."
            ),
            "2021-25 are reused research development years, not pristine holdout.",
            (
                "Position-specific scoring preserves v2 exclusions of fumbles, bonuses, and "
                "cross-position passing/receiving."
            ),
        ],
        audit=panel.groupby(["season", "position"])
        .agg(
            rows=("player_id", "size"),
            observed_labels=("label_observed", "sum"),
            exact_v2=("v2_exact_available", "sum"),
            outcome_team_changes=("team_changed_in_outcome", "sum"),
        )
        .reset_index()
        .to_dict("records"),
    )
    (output_dir / "dataset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return panel, features, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_panel(args.output)
    print(f"Saved {len(result[0]):,} rows and {len(result[1])} features to {args.output}")
