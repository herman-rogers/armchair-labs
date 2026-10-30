"""Fixed-origin v3 extensions with separately evaluated component forecasts.

The next game's point mean belongs to v3.2. A residual model around the saved v3
mean predicts the average of *later* games using only the origin feature row.
Component models predict weekly and later-game averages; no v2 long anchor enters
training or inference. Targets may inspect future outcomes; features never do.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ..weekly_v3.data import HISTORY, ROOT, TEAM_ALIASES, score_points
from ..weekly_v31.artifacts import digest, now, seal, verify, write
from ..weekly_v32.run import DEFAULT_RUN as BASE_RUN

DEFAULT_RUN = ROOT / "experiments/future_player_lab/runs/opening_drive_v4_horizons_20261027_003"
ORIGINS = tuple(range(1, 17))
TARGETS = {
    "QB": (
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ),
    **{
        p: (
            "carries",
            "rushing_yards",
            "rushing_tds",
            "targets",
            "receptions",
            "receiving_yards",
            "receiving_tds",
        )
        for p in ("RB", "WR", "TE")
    },
}
COMPONENTS = tuple(sorted({t for values in TARGETS.values() for t in values}))
# Explicit allowlist excludes current-game injury features and all v2 predictions.
FEATURES = (
    "base_prediction",
    "origin_week",
    "horizon_games",
    "later_games",
    "player_history_count",
    "player_history_mean",
    "player_history_variance",
    "weeks_since_appearance",
    "rookie",
    "draft_round",
    "draft_pick",
    *[
        f"{window}_{stat}"
        for window in ("recent4", "recent8", "current", "prior")
        for stat in (
            "count",
            "points_mean",
            "attempts_mean",
            "carries_mean",
            "targets_mean",
            "receptions_mean",
            "passing_yards_mean",
            "rushing_yards_mean",
            "receiving_yards_mean",
            "appearance_rate",
        )
    ],
)


def feature_matrix(frame):
    return frame.reindex(columns=FEATURES).astype(float).replace([np.inf, -np.inf], np.nan)


def schedules(frame):
    frame = frame[frame.game_type.eq("REG")].copy()
    result = {}
    for year, group in frame.groupby("season"):
        for team in set(group.home_team) | set(group.away_team):
            result[int(year), TEAM_ALIASES.get(team, team)] = sorted(
                group.loc[group.home_team.eq(team) | group.away_team.eq(team), "week"].astype(int)
            )
    return result


def training_panel(panel, box, schedule):
    """Build labels from complete season captures; preserve missing observed values.

    Missing player box rows are supported zero appearances in the complete source,
    including players who leave the origin roster. A present row with a missing
    component remains missing. Future roster membership never selects candidates.
    """
    panel = panel[
        panel.horizon.eq("early_week") & panel.origin_week.isin(ORIGINS) & panel.season.le(2025)
    ].copy()
    box = box[box.season_type.eq("REG") & box.player_id.notna()].copy()
    keys = ["player_id", "season", "week"]
    if box.duplicated(keys).any() or panel.duplicated(keys).any():
        raise ValueError("Duplicate player/week labels or origin rows")
    box["points"] = score_points(box)
    box.loc[(box.targets < box.receptions) | (box.targets < 0), "targets"] = np.nan
    lookup = {
        key: (group.week.to_numpy(), group[["points", *COMPONENTS]].to_numpy(float))
        for key, group in box.groupby(["player_id", "season"])
    }
    calendar = schedules(schedule)
    records = []
    for row in panel.to_dict("records"):
        games = calendar[row["season"], TEAM_ALIASES.get(row["team"], row["team"])]
        remaining = [w for w in games if w >= row["week"]]
        later = [w for w in remaining if w > row["week"]]
        # This panel contains scheduled rows; byes are handled live without a current game.
        if row["week"] not in remaining or not later:
            continue
        history = lookup.get((row["player_id"], row["season"]))
        for kind, weeks in ((0, [row["week"]]), (1, later)):
            record = dict(
                row, horizon_games=len(weeks), later_games=kind, remaining_games=len(remaining)
            )
            # All subsequent player games contribute, including a transfer. The
            # denominator stays the schedule known for the origin team. np.sum
            # deliberately propagates missing components from present box rows.
            totals = np.zeros(1 + len(COMPONENTS))
            if history is not None:
                observed_weeks, observed_values = history
                mask = observed_weeks > row["week"] if kind else observed_weeks == row["week"]
                totals = observed_values[mask].sum(axis=0)
            for target, total in zip(("points", *COMPONENTS), totals, strict=True):
                record[f"label_{target}"] = float(total) / len(weeks)
            records.append(record)
    return pd.DataFrame(records)


def fit_models(frame, before_season):
    """Fit fixed hyperparameters, using strictly earlier seasons only."""
    prior = frame[frame.season.lt(before_season)]
    if prior.empty:
        raise ValueError("Horizon fit requires earlier seasons")
    result = {"training_through_season": int(prior.season.max()), "positions": {}}
    for position, targets in TARGETS.items():
        group = prior[prior.position.eq(position)]
        models = {}
        for target in ("points", *targets):
            rows = group[group[f"label_{target}"].notna()]
            if target == "points":
                rows = rows[rows.later_games.eq(1)]
            if len(rows) < 100:
                raise ValueError(f"Insufficient earlier labels for {position}/{target}")
            label = rows[f"label_{target}"].to_numpy()
            if target == "points":
                label = label - rows.base_prediction.to_numpy()
            model = LGBMRegressor(
                n_estimators=70,
                num_leaves=12,
                max_depth=5,
                min_child_samples=50,
                learning_rate=0.05,
                reg_lambda=10,
                verbosity=-1,
                n_jobs=2,
                random_state=704,
                deterministic=True,
                force_col_wise=True,
            )
            model.fit(feature_matrix(rows), label)
            models[target] = model
        result["positions"][position] = models
    return result


def predictions(models, frame):
    result = pd.DataFrame(index=frame.index)
    for position, group in frame.groupby("position"):
        features = feature_matrix(group)
        for target, model in models["positions"][position].items():
            values = model.predict(features)
            if target == "points":
                values += group.base_prediction.to_numpy()
            # Yards and fantasy points can be negative. Counting events cannot.
            if target != "points" and not target.endswith("yards"):
                values = np.maximum(values, 0)
            result.loc[group.index, target] = values
    if "completions" in result:
        result["completions"] = np.minimum(result.completions, result.attempts)
    if "receptions" in result:
        result["receptions"] = np.minimum(result.receptions, result.targets)
    if "report_out" in frame:
        absent = frame.report_out.eq(1) & frame.later_games.eq(0)
        for position, targets in TARGETS.items():
            result.loc[absent & frame.position.eq(position), list(targets)] = 0.0
    return result


def metric_rows(frame, targets=("points",)):
    rows = []
    for (position, origin, method), group in frame.groupby(["position", "origin_week", "method"]):
        for year in [None, *sorted(group.season.unique())]:
            part = group if year is None else group[group.season.eq(year)]
            error = part.prediction - part.target_points
            ranks = []
            for _, cohort in part.groupby("season"):
                ranks.append(
                    cohort.nlargest(
                        10 if position in ("QB", "TE") else 20, "prediction"
                    ).target_points.mean()
                )
            rows.append(
                dict(
                    position=position,
                    origin_week=int(origin),
                    method=method,
                    season=year,
                    n=len(part),
                    rmse=float(np.sqrt((error**2).mean())),
                    mae=float(error.abs().mean()),
                    bias=float(error.mean()),
                    top_k_actual_mean=float(np.mean(ranks)),
                )
            )
    return pd.DataFrame(rows)


def run(output=DEFAULT_RUN):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    verify(BASE_RUN)
    receipt = json.loads((HISTORY / "manifest.json").read_text())
    for name in ("box.parquet", "schedule.parquet"):
        if digest(HISTORY / name) != receipt["files"][name]:
            raise ValueError(f"Changed frozen source: {name}")
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / "horizons.py")
    write(
        output / "protocol.json",
        dict(
            base="opening_drive_v32",
            point_anchor="v3 saved weekly mean plus learned future residual",
            origins=list(ORIGINS),
            evaluation_seasons=list(range(2021, 2026)),
            training="Strictly earlier seasons; fixed hyperparameters; no current-year fit",
            component_targets=TARGETS,
            feature_allowlist=FEATURES,
            promotion="None; reused development years and future prospective results required",
        ),
    )
    original = pd.read_parquet(BASE_RUN / "panel.parquet")
    frame = training_panel(
        original,
        pd.read_parquet(HISTORY / "box.parquet"),
        pd.read_parquet(HISTORY / "schedule.parquet"),
    )
    frame.to_parquet(output / "training_panel.parquet", index=False)
    weekly = pd.read_parquet(BASE_RUN / "predictions.parquet")
    weekly = weekly[weekly.horizon.eq("early_week") & weekly.method.eq("selected")]
    weekly = weekly.set_index("row_id").prediction
    records, component_records = [], []
    for year in range(2018, 2027):
        print(f"Fitting horizon/components before {year}", flush=True)
        fitted = fit_models(frame, year)
        if year == 2026:
            joblib.dump(fitted, output / "models.pkl")
            break
        test = frame[frame.season.eq(year)].copy()
        estimates = predictions(fitted, test)
        test["weekly_points"] = test.row_id.map(weekly).fillna(test.base_prediction)
        first = test[test.later_games.eq(0)].set_index("row_id")
        later = test[test.later_games.eq(1)]
        for idx, row in later.iterrows():
            current = first.loc[row.row_id]
            actual = current.label_points + row.label_points * row.horizon_games
            for method, mean in (
                ("v4", estimates.loc[idx, "points"]),
                ("v3_fixed_origin", row.base_prediction),
            ):
                records.append(
                    dict(
                        player_id=row.player_id,
                        player_name=row.player_name,
                        season=year,
                        position=row.position,
                        origin_week=int(row.origin_week),
                        remaining_games=int(row.remaining_games),
                        prediction=float(row.weekly_points + mean * row.horizon_games),
                        target_points=float(actual),
                        method=method,
                    )
                )
        if year >= 2021:
            for target in COMPONENTS:
                valid = (
                    test.position.map(lambda p, target=target: target in TARGETS[p])
                    & test[f"label_{target}"].notna()
                )
                for idx, row in test[valid].iterrows():
                    component_records.append(
                        dict(
                            season=year,
                            position=row.position,
                            origin_week=int(row.origin_week),
                            horizon="later" if row.later_games else "weekly",
                            target=target,
                            prediction=float(estimates.loc[idx, target] * row.horizon_games),
                            actual=float(row[f"label_{target}"] * row.horizon_games),
                        )
                    )
    result = pd.DataFrame(records)
    result.to_parquet(output / "horizon_predictions.parquet", index=False)
    metric_rows(result[result.season.ge(2021)]).to_csv(output / "scores.csv", index=False)
    components = pd.DataFrame(component_records)
    components.to_parquet(output / "component_predictions.parquet", index=False)
    components["error"] = components.prediction - components.actual
    component_scores = (
        components.groupby(["position", "origin_week", "horizon", "target"])
        .error.agg(
            n="size",
            rmse=lambda x: np.sqrt((x**2).mean()),
            mae=lambda x: x.abs().mean(),
            bias="mean",
        )
        .reset_index()
    )
    component_scores.to_csv(output / "component_scores.csv", index=False)
    # Exact shipped bridge comparison exists at origin 2 only. Pair it without
    # disguising later-origin persistence controls as archived v3.2 predictions.
    bridge = ROOT / "experiments/future_player_lab/runs/opening_drive_v32_horizons_20261014_002"
    verify(bridge)
    old = pd.read_parquet(bridge / "predictions.parquet")
    old = old[old.method.eq("v32_selected_decay")]
    keys = ["player_id", "season", "position", "origin_week"]
    paired = result[result.method.eq("v4")].merge(
        old[keys + ["prediction", "actual"]], on=keys, suffixes=("", "_v32"), validate="one_to_one"
    )
    if not np.allclose(paired.target_points, paired.actual, atol=1e-6):
        # Different eligible-label contracts must remain explicit, not silently pooled.
        paired["label_matches"] = np.isclose(paired.target_points, paired.actual, atol=1e-6)
    else:
        paired["label_matches"] = True
    paired.to_parquet(output / "v32_bridge_matched.parquet", index=False)
    matched = paired[paired.label_matches].copy()
    control = matched.copy()
    control["prediction"], control["method"] = control.prediction_v32, "v32_bridge"
    metric_rows(pd.concat([matched, control])).to_csv(output / "v32_bridge_scores.csv", index=False)
    write(
        output / "policy.json",
        dict(
            version=1,
            base_campaign="opening_drive_v32",
            forecast_season=2026,
            training_through_season=2025,
            evaluated_origins=list(ORIGINS),
            component_targets=TARGETS,
            lineage="v3 weekly residual extension; no v2 horizon anchor",
            current_week="Exact served v3.2 point forecast; direct learned component means",
            future_weeks="Fixed-origin learned average; no later forecasts or injury reports",
            limitations=[
                "Historical years reused in development; no prospective promotion evidence.",
                "Direct point means need not equal component-rescored points; both retained.",
                "Future team schedule held fixed; transfers are not predicted.",
                "Origins 1–16 evaluated; final-week current-only forecast reuses weekly model.",
                "Exact v3.2 bridge comparison at origin 2 only; later control is v3 persistence.",
                "Historical source vintages reconstructed; missing player rows are zero "
                "only in complete box captures.",
            ],
        ),
    )
    write(
        output / "sources.json",
        {
            str(p): digest(p)
            for p in (
                BASE_RUN / "manifest.json",
                HISTORY / "box.parquet",
                HISTORY / "schedule.parquet",
                bridge / "manifest.json",
            )
        },
    )
    if digest(__file__) != digest(output / "horizons.py"):
        raise ValueError("Horizon source changed during fitting; rerun in a fresh directory")
    seal(output, dict(kind="opening_drive_v4_horizons", generated_at=now().isoformat()))
    return output


def component_points(position, values):
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


def build_projection(players, panel, snapshot, fitted_run=DEFAULT_RUN):
    """Build all three horizons from a single immutable capture and accepted snapshot."""
    fitted_run, snapshot = Path(fitted_run), Path(snapshot)
    verify(fitted_run)
    if digest(__file__) != digest(fitted_run / "horizons.py"):
        raise ValueError("Horizon inference differs from sealed research source")
    policy = json.loads((fitted_run / "policy.json").read_text())
    models = joblib.load(fitted_run / "models.pkl")
    document = json.loads(snapshot.read_text())
    season, origin = (document["current_observations"][c] for c in ("season", "through_week"))
    if season != policy["forecast_season"] or models["training_through_season"] >= season:
        raise ValueError("Horizon policy requires a new earlier-season fit")
    if players.player_id.duplicated().any() or panel.player_id.duplicated().any():
        raise ValueError("Duplicate live identities")
    if not (players.season.eq(season) & players.week.eq(origin + 1)).all():
        raise ValueError("Live rows cross the snapshot origin")
    if "target_points" in panel and panel.target_points.notna().any():
        raise ValueError("Live feature rows contain forecast outcomes")
    assets = {}
    for name in ("schedule", "weeks"):
        ref = document["assets"][f"current/{name}.parquet"]
        path = ROOT / "data" / ref["object"]
        if digest(path) != ref["sha256"]:
            raise ValueError(f"Changed accepted {name}")
        assets[name] = pd.read_parquet(path)
    calendar = schedules(assets["schedule"])
    box = assets["weeks"]
    box = box[box.season.eq(season) & box.week.le(origin) & box.season_type.eq("REG")].copy()
    if box.duplicated(["player_id", "week"]).any():
        raise ValueError("Duplicate accepted actuals")
    box["points"] = score_points(box)
    totals = box.groupby("player_id")[["points", *COMPONENTS]].agg(
        lambda s: s.sum() if s.notna().all() else np.nan
    )
    lookup = panel.set_index("player_id").to_dict("index")
    feature_rows = []
    plans = {}
    for player in players.to_dict("records"):
        pid = player["player_id"]
        team = TEAM_ALIASES.get(player["team"], player["team"])
        games = calendar.get((season, team), [])
        future = [w for w in games if w > origin]
        current = origin + 1 in future
        if not games or bool(player.get("is_bye", False)) == current:
            raise ValueError("Player bye disagrees with accepted team schedule")
        base = player.get("base_prediction")
        if base is None or not np.isfinite(base):
            # Byes have no v3 weekly prediction; use past-only history already frozen.
            base = lookup[pid].get("recent8_points_mean")
            if base is None or not np.isfinite(base):
                base = lookup[pid].get("player_history_mean")
            if base is None or not np.isfinite(base):
                base = 0.0
        plans[pid] = (games, future, current, base)
        for later, count in ((0, int(current)), (1, len(future) - int(current))):
            feature_rows.append(
                dict(
                    lookup[pid],
                    player_id=pid,
                    position=player["position"],
                    base_prediction=base,
                    report_out=player.get("report_out", 0),
                    origin_week=origin,
                    later_games=later,
                    horizon_games=max(count, 1),
                )
            )
    features = pd.DataFrame(feature_rows)
    estimates = predictions(models, features)
    output = {h: [] for h in ("weekly", "remaining", "season")}
    for i, player in enumerate(players.to_dict("records")):
        pid, position = player["player_id"], player["position"]
        games, future, current, base = plans[pid]
        weekly_components = {
            t: float(estimates.loc[2 * i, t]) if current else 0.0 for t in TARGETS[position]
        }
        later_components = {t: float(estimates.loc[2 * i + 1, t]) for t in TARGETS[position]}
        anchor = float(estimates.loc[2 * i + 1, "points"])
        curve = [
            dict(
                week=w,
                points=float(player["prediction"]) if w == origin + 1 else anchor,
                components=weekly_components if w == origin + 1 else later_components,
                basis="saved_v32_weekly" if w == origin + 1 else "v3_learned_horizon_extension",
            )
            for w in future
        ]
        remaining = {t: sum(g["components"][t] for g in curve) for t in TARGETS[position]}
        remaining["points"] = sum(g["points"] for g in curve)
        earned = {
            t: float(totals.loc[pid, t]) if pid in totals.index else 0.0
            for t in ("points", *TARGETS[position])
        }
        if not all(np.isfinite(v) for v in earned.values()):
            raise ValueError(f"Missing accepted scoring/component actuals for {pid}")
        projection = dict(
            method="opening_drive_v4_v3_residual_components",
            future_weeks=curve,
            remaining_points=remaining["points"],
            actual_points_to_date=earned["points"],
            actual_components_to_date={t: earned[t] for t in TARGETS[position]},
            remaining_scheduled_games=len(future),
            season_scheduled_games=len(games),
            anchor_points_per_game=anchor,
            anchor_basis="v3_learned_residual",
            schedule_team=player["team"],
            base_weekly_points=base,
        )
        for horizon, count in (
            ("weekly", int(current)),
            ("remaining", len(future)),
            ("season", len(games)),
        ):
            values = (
                dict(weekly_components, points=float(player["prediction"]))
                if horizon == "weekly"
                else remaining
                if horizon == "remaining"
                else {t: remaining[t] + earned[t] for t in remaining}
            )
            row = dict(
                player_id=pid,
                player_name=player.get("player_name"),
                position=position,
                team=player["team"],
                season=season,
                origin_week=origin,
                horizon=horizon,
                predictions=values,
                scheduled_games=count,
                points_per_scheduled_game=values["points"] / count if count else None,
                component_points=component_points(position, values),
                candidate_basis="Locked v3.2 population; v3 fixed-origin component extension",
                model_lineage="opening_drive_v32 -> v3 residual horizon/components",
            )
            if horizon != "weekly":
                row["projection"] = projection
            output[horizon].append(row)
    for rows in output.values():
        for position in TARGETS:
            cohort = sorted(
                (r for r in rows if r["position"] == position),
                key=lambda r: (-r["predictions"]["points"], r["player_id"]),
            )
            for rank, row in enumerate(cohort, 1):
                row["rank"] = rank
    output["provenance"] = dict(
        policy_manifest_sha256=digest(fitted_run / "manifest.json"),
        snapshot_sha256=digest(snapshot),
        evaluated_origin=origin in ORIGINS,
        no_v2_horizon_anchor=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    print(run(parser.parse_args().output))
