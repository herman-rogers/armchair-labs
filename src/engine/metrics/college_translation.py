"""Temporal college→college and college→NFL research, not promoted forecasts.

Pre-draft excludes actual draft capital. Post-draft tests its incremental value.
NFL samples are conditional on the audited NFL-entry candidate population; an
unlinked college athlete is never fabricated as a zero-NFL-production outcome.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from engine.data.college import STAT_COLUMNS, unique

POSITIONS = ("QB", "RB", "WR", "TE")
COLLEGE_FEATURES = (
    "age",
    "college_seasons",
    "history_left_truncated",
    "history_partial",
    "latest_passing_yards",
    "latest_passing_tds",
    "latest_passing_ints",
    "latest_rushing_yards",
    "latest_rushing_tds",
    "latest_carries",
    "latest_receiving_yards",
    "latest_receiving_tds",
    "latest_receptions",
    "latest_scheduled_games",
    "latest_share_receiving_yards",
    "latest_share_receptions",
    "latest_share_rushing_yards",
    "latest_share_carries",
    "career_passing_yards",
    "career_rushing_yards",
    "career_receiving_yards",
    "best_share_receiving_yards",
    "best_share_rushing_yards",
    "receiving_yards_change",
    "rushing_yards_change",
    "passing_yards_change",
)
DRAFT_FEATURES = ("draft_capital", "was_drafted")
NFL_TARGETS = ("nfl_year1_points", "nfl_year1_games", "nfl_first3_points")
SHARES = ("share_receiving_yards", "share_receptions", "share_rushing_yards", "share_carries")


def annual_rows(seasons: pl.DataFrame) -> list[dict]:
    """Keep multi-team seasons inspectable, but withhold them from this first model."""
    rows = (
        seasons.group_by(["college_id", "season"])
        .agg(
            pl.col("player_name").drop_nulls().sort().first(),
            pl.col("college_position").drop_nulls().sort().first(),
            pl.col("college_team").drop_nulls().unique().sort().alias("teams"),
            pl.col("team_id").n_unique().alias("team_count"),
            pl.col("birth_date").drop_nulls().sort().first(),
            pl.col("complete_team_season").all().alias("complete_team_season"),
            pl.col("coverage").min(),
            pl.col("season_end_date").max(),
            pl.col("scheduled_games").sum(),
            pl.col("observed_stat_games").sum(),
            *[pl.col(c).sum() for c in STAT_COLUMNS],
            *[pl.col(c).mean() for c in SHARES],
        )
        .sort("college_id", "season")
    )
    return rows.to_dicts()


def features(history: list[dict], forecast_year: int, birth_date: str | None = None) -> dict:
    """Only seasons strictly before the decision year may supply model features."""
    prior = sorted([r for r in history if r["season"] < forecast_year], key=lambda r: r["season"])
    if not prior:
        return {"eligible": False, "ineligible_reason": "no_pre_cutoff_college_history"}
    last = prior[-1]
    complete = [r for r in prior if r["complete_team_season"] and r["team_count"] == 1]
    eligible = bool(
        last["complete_team_season"]
        and last["team_count"] == 1
        and last["season"] == forecast_year - 1
    )
    result = {
        "eligible": eligible,
        "ineligible_reason": None if eligible else "latest_season_incomplete_multi_team_or_gap",
        "last_college_season": last["season"],
        "college_team": ", ".join(last["teams"]),
        "college_seasons": len(prior),
        "history_left_truncated": int(prior[0]["season"] == 2004),
        "history_partial": int(len(complete) != len(prior)),
        "age": None,
    }
    dob = birth_date or last.get("birth_date")
    if dob:
        try:
            age = (date(forecast_year, 9, 1) - date.fromisoformat(str(dob)[:10])).days / 365.2425
            if 15 <= age <= 45:
                result["age"] = age
        except ValueError:
            pass
    for name in [*STAT_COLUMNS, "scheduled_games", *SHARES]:
        result[f"latest_{name}"] = last.get(name) if eligible else None
    for stat in ["passing_yards", "rushing_yards", "receiving_yards"]:
        # Partial careers are explicit; never describe this as complete lifetime production.
        result[f"career_{stat}"] = sum(r[stat] for r in complete)
        earlier = [r for r in complete if r["season"] == last["season"] - 1]
        result[f"{stat}_change"] = last[stat] - earlier[0][stat] if earlier and eligible else None
    for stat in ["share_receiving_yards", "share_rushing_yards"]:
        values = [r[stat] for r in complete if r[stat] is not None]
        result[f"best_{stat}"] = max(values) if values else None
    return result


def nfl_cohort(
    annual: list[dict],
    links: pl.DataFrame,
    candidates: pl.DataFrame,
    nfl_seasons: pl.DataFrame,
    complete_through: int,
) -> list[dict]:
    """Start from ALL audited NFL candidates, not just successful linked players."""
    history = defaultdict(list)
    for row in annual:
        history[row["college_id"]].append(row)
    accepted = defaultdict(list)
    for row in links.filter(pl.col("status") == "linked").to_dicts():
        accepted[row["player_id"]].append(row)
    outcomes = {(r["player_id"], r["season"]): r for r in nfl_seasons.to_dicts()}
    unique(candidates, ["player_id", "forecast_season"])
    rows = []
    for candidate in candidates.to_dicts():
        pid, year = candidate["player_id"], int(candidate["forecast_season"])
        linked = accepted.get(pid, [])
        college_history = [r for link in linked for r in history[link["college_id"]]]
        # Multiple reviewed college IDs can represent one person; overlapping
        # seasons need explicit reconciliation, not a silent sum.
        overlap = len({r["season"] for r in college_history}) != len(college_history)
        feature = features(college_history, year)
        if overlap:
            feature.update(eligible=False, ineligible_reason="overlapping_college_ids")
        pick = candidate.get("rookie_draft_pick")
        row = {
            "player_id": pid,
            "player_display_name": candidate["player_display_name"],
            "position": candidate["position"],
            "forecast_year": year,
            "college_ids": [r["college_id"] for r in linked],
            "link_methods": sorted({r["method"] for r in linked}),
            "draft_pick": pick,
            "draft_capital": -np.log(pick if pick and pick > 0 else 300),
            "was_drafted": float(pick is not None and pick > 0),
            **feature,
            "nfl_year1_points": None,
            "nfl_year1_games": None,
            "nfl_first3_points": None,
        }
        # Year-one games use accepted participation-aware outcomes, not box-row counts.
        if year <= complete_through and candidate.get("outcome_complete"):
            row["nfl_year1_points"] = float(candidate.get("actual_season_points") or 0)
            row["nfl_year1_games"] = float(candidate.get("actual_games") or 0)
        if year + 2 <= complete_through and candidate.get("outcome_complete"):
            row["nfl_first3_points"] = sum(
                outcomes.get((pid, season), {}).get("league_points", 0.0)
                for season in range(year, year + 3)
            )
        rows.append(row)
    return rows


def matrix(rows: list[dict], columns: tuple[str, ...]) -> np.ndarray:
    return np.array([[np.nan if r.get(c) is None else float(r[c]) for c in columns] for r in rows])


def ridge_prediction(train: list[dict], test: list[dict], columns: tuple[str, ...], target: str):
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=100),
    )
    model.fit(matrix(train, columns), [r[target] for r in train])
    predictions = np.maximum(model.predict(matrix(test, columns)), 0)
    if target == "nfl_year1_games":
        predictions = np.minimum(
            predictions, [17 if r["forecast_year"] >= 2021 else 16 for r in test]
        )
    return predictions


def nfl_backtest(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Expanding chronological folds; 3-year targets must mature BEFORE the fold.

    Fixed ridge penalty avoids selecting hyperparameters on the displayed test set.
    Every comparison uses identical eligible players. All candidate coverage is
    separately reported; unmatched and incomplete records are not dropped invisibly.
    """
    output = []
    eligible = [r for r in rows if r["eligible"] and r["position"] in POSITIONS]
    for target in NFL_TARGETS:
        horizon = 3 if target == "nfl_first3_points" else 1
        for position in POSITIONS:
            pool = [r for r in eligible if r["position"] == position]
            for year in sorted({r["forecast_year"] for r in pool}):
                train = [
                    r
                    for r in pool
                    if r["forecast_year"] + horizon - 1 < year and r[target] is not None
                ]
                test = [r for r in pool if r["forecast_year"] == year]
                if len(train) < 40 or len({r["forecast_year"] for r in train}) < 4:
                    continue
                predictions = {
                    "position_mean": np.repeat(np.mean([r[target] for r in train]), len(test)),
                    "draft_only": ridge_prediction(train, test, DRAFT_FEATURES, target),
                    "college_only": ridge_prediction(train, test, COLLEGE_FEATURES, target),
                    "college_plus_draft": ridge_prediction(
                        train, test, COLLEGE_FEATURES + DRAFT_FEATURES, target
                    ),
                }
                for i, row in enumerate(test):
                    output.append(
                        {
                            "player_id": row["player_id"],
                            "player_display_name": row["player_display_name"],
                            "college_ids": row["college_ids"],
                            "position": position,
                            "forecast_year": year,
                            "target": target,
                            "actual": row[target],
                            "train_n": len(train),
                            "train_latest_entry": max(r["forecast_year"] for r in train),
                            "train_latest_outcome_year": max(
                                r["forecast_year"] + horizon - 1 for r in train
                            ),
                            **{name: float(values[i]) for name, values in predictions.items()},
                        }
                    )
    summaries = summarize(
        output,
        ["position_mean", "draft_only", "college_only", "college_plus_draft"],
        baseline="draft_only",
    )
    return output, summaries


def summarize(rows: list[dict], models: list[str], baseline: str) -> list[dict]:
    result = []
    for position, target in sorted({(r["position"], r["target"]) for r in rows}):
        for era, minimum in [("all", 0), ("modern_2018_plus", 2018)]:
            pool = [
                r
                for r in rows
                if r["position"] == position
                and r["target"] == target
                and r["actual"] is not None
                and r["forecast_year"] >= minimum
            ]
            if not pool:
                continue
            folds = []
            for year in sorted({r["forecast_year"] for r in pool}):
                fold = [r for r in pool if r["forecast_year"] == year]
                folds.append(
                    {
                        "year": year,
                        "n": len(fold),
                        **{
                            m: float(np.mean([abs(r[m] - r["actual"]) for r in fold]))
                            for m in models
                        },
                    }
                )
            mae = {m: float(np.mean([abs(r[m] - r["actual"]) for r in pool])) for m in models}
            rng = np.random.default_rng(271828)
            intervals: dict[str, list[float] | None] = {}
            for model in models:
                differences = np.array([f[baseline] - f[model] for f in folds])
                if len(differences) >= 5:
                    draws = rng.choice(
                        differences, size=(2000, len(differences)), replace=True
                    ).mean(axis=1)
                    intervals[model] = [float(v) for v in np.quantile(draws, [0.025, 0.975])]
                else:
                    intervals[model] = None
            result.append(
                {
                    "position": position,
                    "target": target,
                    "era": era,
                    "n": len(pool),
                    "mae": mae,
                    "rmse": {
                        m: float(np.sqrt(np.mean([(r[m] - r["actual"]) ** 2 for r in pool])))
                        for m in models
                    },
                    "folds": folds,
                    "baseline": baseline,
                    "season_bootstrap_95_mae_improvement": intervals,
                    "status": "exploratory_not_promoted",
                }
            )
    return result


def college_backtest(annual: list[dict]) -> tuple[list[dict], list[dict]]:
    """Next-college-season yardage conditional on being rostered the following year.

    A departing player is NOT given zero college production. Continuing roster
    players with no offense can have zero, only when their team coverage is complete.
    Uses broad production categories, since early historical positions are missing.
    """
    histories = defaultdict(list)
    for row in annual:
        histories[row["college_id"]].append(row)
    rows = []
    for cid, history in histories.items():
        for next_row in history:
            year = next_row["season"]
            feature = features(history, year)
            if (
                not feature["eligible"]
                or not next_row["complete_team_season"]
                or next_row["team_count"] != 1
            ):
                continue
            for stat in ["passing_yards", "rushing_yards", "receiving_yards"]:
                if (feature.get(f"latest_{stat}") or 0) <= 0:
                    continue
                rows.append(
                    {
                        "college_id": cid,
                        "player_display_name": next_row["player_name"],
                        "position": stat,
                        "forecast_year": year,
                        "target": "next_college_yards",
                        "next_college_yards": next_row[stat],
                        **feature,
                    }
                )
    output = []
    for position in ["passing_yards", "rushing_yards", "receiving_yards"]:
        pool = [r for r in rows if r["position"] == position]
        for year in sorted({r["forecast_year"] for r in pool}):
            train = [r for r in pool if r["forecast_year"] < year]
            test = [r for r in pool if r["forecast_year"] == year]
            if len(train) < 100 or len({r["forecast_year"] for r in train}) < 4:
                continue
            predictions = ridge_prediction(train, test, COLLEGE_FEATURES, "next_college_yards")
            for row, prediction in zip(test, predictions, strict=True):
                output.append(
                    {
                        "college_id": row["college_id"],
                        "player_display_name": row["player_display_name"],
                        "position": position,
                        "forecast_year": year,
                        "target": "next_college_yards",
                        "actual": row["next_college_yards"],
                        "college_model": float(prediction),
                        "last_season": row[f"latest_{position}"],
                        "train_n": len(train),
                    }
                )
    return output, summarize(output, ["last_season", "college_model"], baseline="last_season")
