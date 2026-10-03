"""All-history, all-position profile forecasts with expanding temporal validation."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
from qb_role_transition import attach_evidence, load_evidence
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from value_capture_audit import finite, model_values, rank_ids, score
from value_capture_audit import summarize as market_summary

from engine.api.college_sources import load_college
from engine.api.profile_sources import load_profiles
from engine.api.research_sources import accepted_version, digest
from engine.metrics.outlook import team_name, unique
from engine.metrics.player_profile import role_band, summarize

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "research/profile_history_design.md"
EVIDENCE = ROOT / "research/qb_role_evidence_20260923.csv"
POSITIONS = ("QB", "RB", "WR", "TE")
ARCHITECTURES = ("ridge", "boost")
SPECS = ("basic", "career", "profile", "basic_market", "profile_market")
WINDOWS = {"all_history": (2005, 2025), "pre_modern": (2005, 2018), "modern": (2019, 2025)}
CONTEXT = (
    "cutoff_state_observed",
    "cutoff_preseason_rostered",
    "cutoff_preseason_reserve",
    "cutoff_injured_reserve",
    "cutoff_pup_nfi",
    "cutoff_suspended",
    "cutoff_reserve_other",
    "cutoff_transaction_recency_days",
    "cutoff_transaction_count",
    "known_absence_games",
    "known_available_games_cap",
    "known_suspension_games",
)
SUMMARY_FIELDS = (
    "observed_weeks",
    "snap_observations",
    "points_per_observed_week",
    "snap_share",
    "targets_per_observed_week",
    "carries_per_observed_week",
    "passing_yards_per_attempt",
    "rushing_yards_per_carry",
    "receiving_yards_per_target",
)
VOLUMES = ("attempts", "carries", "targets", "receptions", "league_points")
COLLEGE_STATS = (
    "passing_yards",
    "passing_tds",
    "passing_ints",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)
COLLEGE_SHARES = (
    "share_receiving_yards",
    "share_receptions",
    "share_rushing_yards",
    "share_carries",
)
KEYS = ["player_id", "forecast_season"]


def nominal_games(year):
    return 17 if year >= 2021 else 16


def number(value):
    if isinstance(value, bool):
        return float(value)
    return float(value) if finite(value) else None


def summary_features(observations, prefix):
    summary = summarize(observations)
    values = {f"{prefix}_{k}": summary[k] for k in SUMMARY_FIELDS}
    for field in VOLUMES:
        values[f"{prefix}_{field}"] = summary[field]
        # Partial numerator coverage cannot silently become a fully observed rate.
        values[f"{prefix}_{field}_per_week"] = (
            summary[field] / summary["observed_weeks"]
            if summary["observed_weeks"]
            and summary[f"{field}_observations"] == summary["observed_weeks"]
            else None
        )
    return values


def nfl_features(observations, candidate, identity, first_source_year):
    year = candidate["forecast_season"]
    prior = sorted(
        [r for r in observations if r["season"] < year], key=lambda r: (r["season"], r["week"])
    )
    last = [r for r in prior if r["season"] == year - 1]
    recent = [r for r in prior if r["season"] >= year - 3]
    result = summary_features(last, "basic_prior")
    result.update(summary_features(prior, "career"))
    result.update(summary_features(recent, "career_recent3"))
    rookie_year = identity.get("rookie_season")
    result["career_left_truncated"] = (
        float(rookie_year < first_source_year) if finite(rookie_year) else None
    )
    years = sorted({r["season"] for r in prior})
    result["career_seasons_observed"] = len(years)
    result["career_years_since_observation"] = year - years[-1] if years else None
    result["career_best_season_points"] = max(
        (sum(r["league_points"] for r in prior if r["season"] == y) for y in years), default=None
    )
    denominator = sum(0.55 ** (year - 1 - r["season"]) for r in recent)
    result["career_recent3_weighted_ppg"] = (
        sum(r["league_points"] * 0.55 ** (year - 1 - r["season"]) for r in recent) / denominator
        if denominator
        else None
    )
    teams = [team_name(r.get("team")) for r in prior if r.get("team")]
    result["career_observed_team_changes"] = sum(
        a != b for a, b in zip(teams, teams[1:], strict=False)
    )
    for band in ("high", "rotation", "limited", "unknown"):
        result.update(
            summary_features(
                [r for r in prior if role_band(r.get("snap_share")) == band], f"career_{band}"
            )
        )
    # Same 50% QB proxy as the separately collected role-evidence cohort.
    known_snaps = [r for r in last if r.get("snap_share") is not None]
    result["prior_majority_games"] = (
        sum(r["snap_share"] >= 0.5 for r in known_snaps) if known_snaps else None
    )
    return result


def college_features(history, candidate, identity, link_methods, nfl):
    year, cutoff = candidate["forecast_season"], candidate["forecast_cutoff_date"]
    entry = identity.get("rookie_season")
    prior = sorted(
        [
            r
            for r in history
            if r["season"] < year
            and (not finite(entry) or r["season"] < entry)
            and str(r["season_end_date"])[:10] <= cutoff
        ],
        key=lambda r: r["season"],
    )
    overlap = len({r["season"] for r in prior}) != len(prior)
    good = (
        [r for r in prior if r["complete_team_season"] and r["team_count"] == 1]
        if not overlap
        else []
    )
    last = good[-1] if good else None
    result = {
        "college_linked": float(bool(link_methods)),
        "college_seasons_observed": len(prior),
        "college_complete_seasons": len(good),
        "college_overlap": float(overlap),
        "college_partial": float(len(good) < len(prior)) if prior else None,
        "college_left_truncated": float(prior[0]["season"] == 2004) if prior else None,
        "college_name_school_link": float("exact_name_school_entry_window" in link_methods),
        "college_latest_observed_coverage": prior[-1]["coverage"] if prior else None,
        "college_years_since_last": year - last["season"] if last else None,
    }
    for field in COLLEGE_STATS:
        result[f"college_career_{field}"] = sum(r[field] for r in good) if good else None
        result[f"college_latest_{field}"] = last[field] if last else None
    for field in COLLEGE_SHARES:
        values = [r[field] for r in good if r[field] is not None]
        result[f"college_best_{field}"] = max(values) if values else None
        result[f"college_latest_{field}"] = last[field] if last else None
    exposure = {"QB": "attempts", "RB": "carries", "WR": "targets", "TE": "targets"}[
        candidate["position"]
    ]
    weight = (200 if candidate["position"] == "QB" else 100) / (
        (nfl.get(f"career_{exposure}") or 0) + (200 if candidate["position"] == "QB" else 100)
    )
    result["college_prior_weight"] = weight
    for field in ("passing_yards", "rushing_yards", "receiving_yards"):
        value = result[f"college_latest_{field}"]
        result[f"college_weighted_{field}"] = value * weight if value is not None else None
    return result


def injury_features(history, year, first_source_year):
    prior = [r for r in history if r["season"] < year]
    result = {"injury_prior_feed_available": float(year - 1 >= first_source_year)}
    for label, start in (("career", first_source_year), ("recent3", year - 3), ("prior", year - 1)):
        rows = [r for r in prior if r["season"] >= start]
        covered = year - 1 >= first_source_year
        for kind, predicate in {
            "reported": lambda r: True,
            "out_doubtful": lambda r: r.get("report_status") in ("Out", "Doubtful"),
            "limited_practice": lambda r: (
                r.get("practice_status")
                in ("Did Not Participate In Practice", "Limited Participation in Practice")
            ),
        }.items():
            result[f"injury_{label}_{kind}_weeks"] = (
                len({(r["season"], r["week"]) for r in rows if predicate(r)}) if covered else None
            )
    return result


def build_features(
    candidates, weeks, players, annual, links, injuries, evidence, *, progress=False
):
    """Keep every candidate; all observational aggregations precede the cutoff."""
    unique(candidates, KEYS)
    unique(weeks, ["player_id", "season", "week"])
    unique(players, ["player_id"])
    nfl_by_id, injury_by_id, college_by_id, linked = (defaultdict(list) for _ in range(4))
    for r in weeks.to_dicts():
        nfl_by_id[r["player_id"]].append(r)
    for r in injuries.to_dicts():
        injury_by_id[r["player_id"]].append(r)
    for r in annual.to_dicts():
        college_by_id[r["college_id"]].append(r)
    for r in links.filter(pl.col("status") == "linked").to_dicts():
        linked[r["player_id"]].append(r)
    identities = {r["player_id"]: r for r in players.to_dicts()}
    first_nfl_year = weeks["season"].min()
    first_injury_year = injuries["season"].min()
    results, last_year = [], None
    for candidate in candidates.sort("forecast_season", "player_id").to_dicts():
        year, pid = candidate["forecast_season"], candidate["player_id"]
        if progress and year != last_year:
            print(f"Building cutoff-safe profiles for {year}", flush=True)
            last_year = year
        identity = identities.get(pid, {})
        nfl = nfl_features(nfl_by_id[pid], candidate, identity, first_nfl_year)
        r = {**candidate, **nfl}
        r["profile_identity_known"] = pid in identities
        r["profile_weeks_known"] = pid in nfl_by_id
        r["season"] = year
        r["eligible"] = True  # Missing enrichment is never a row-exclusion criterion.
        r["role_games"] = nfl["prior_majority_games"]
        draft_pick = identity.get("draft_pick")
        draft_year = identity.get("draft_year")
        known_draft = (
            finite(draft_year) and draft_year <= year and finite(draft_pick) and draft_pick > 0
        )
        r.update(
            basic_age=number(candidate.get("age_at_season")),
            basic_experience=number(candidate.get("player_experience")),
            basic_season=year,
            basic_season_length=nominal_games(year),
            basic_draft_log_pick=float(np.log(draft_pick)) if known_draft else None,
            basic_draft_known=float(known_draft),
        )
        for population in ("rookie", "returner", "market_only"):
            r[f"basic_population_{population}"] = float(
                candidate["player_population"] == population
            )
        history = [c for link in linked[pid] for c in college_by_id[link["college_id"]]]
        r.update(
            college_features(
                history, candidate, identity, [link["method"] for link in linked[pid]], nfl
            )
        )
        r.update(injury_features(injury_by_id[pid], year, first_injury_year))
        for field in CONTEXT:
            r[f"context_{field}"] = number(candidate.get(field))
        for field in ("availability_latest_known_on", "cutoff_last_transaction_date"):
            if (
                candidate.get(field)
                and str(candidate[field])[:10] > candidate["forecast_cutoff_date"]
            ):
                raise ValueError(f"Future context: {pid} {year} {field}")
        for source, label in (("market_ecr", "position"), ("market_overall_ecr", "overall")):
            value = candidate.get(source)
            r[f"market_{label}_log_rank"] = (
                float(np.log(value)) if finite(value) and value > 0 else None
            )
            r[f"market_{label}_inverse_rank"] = 1 / value if finite(value) and value > 0 else None
        r["market_position_dispersion"] = number(candidate.get("market_ecr_sd"))
        r["market_overall_dispersion"] = number(candidate.get("market_overall_ecr_sd"))
        r["actual_points_per_scheduled_game"] = r["actual_season_points"] / nominal_games(year)
        results.append(r)
    with_news = attach_evidence(results, evidence)
    for r in with_news:
        for label in ("starter", "competition", "backup"):
            r[f"news_{label}"] = r[f"evidence_{label}"]
    return with_news


def specifications(rows):
    names = sorted({name for r in rows for name in r})
    basic = tuple(n for n in names if n.startswith("basic_"))
    career = tuple(n for n in names if n.startswith("career_"))
    extras = tuple(n for n in names if n.startswith(("college_", "injury_", "context_", "news_")))
    market = tuple(
        n
        for n in names
        if n.startswith(("market_position_", "market_overall_"))
        and n not in ("market_overall_ecr", "market_overall_ecr_score", "market_overall_ecr_sd")
    )
    return dict(
        basic=basic,
        career=(*basic, *career),
        profile=(*basic, *career, *extras),
        basic_market=(*basic, *market),
        profile_market=(*basic, *career, *extras, *market),
    )


def matrix(rows, columns):
    return np.array([[r.get(c) for c in columns] for r in rows], dtype=float)


def tree_matrices(train, test):
    """Remove unlearnable constant columns; preserve varying observation status.

    Older folds have entire feature families unavailable. The installed tree
    binning implementation cannot bin fewer than two distinct finite values.
    Decide this transform from training only, including missingness splits when
    a feature has just one observed value and some missing values.
    """
    left, right = [], []
    for i in range(train.shape[1]):
        missing = ~np.isfinite(train[:, i])
        count = len(np.unique(train[~missing, i]))
        if count > 1:
            left.append(train[:, i])
            right.append(test[:, i])
        elif count == 1 and missing.any():
            left.append(missing.astype(float))
            right.append((~np.isfinite(test[:, i])).astype(float))
    return (
        (np.column_stack(left), np.column_stack(right))
        if left
        else (np.empty((len(train), 0)), np.empty((len(test), 0)))
    )


def fit_predict(train, test, columns, architecture):
    if len(train) < 30:
        raise ValueError("Fewer than 30 earlier training rows")
    if max(r["forecast_season"] for r in train) >= min(r["forecast_season"] for r in test):
        raise ValueError("Training overlaps test season")
    x, xt = matrix(train, columns), matrix(test, columns)
    y = np.array([r["actual_points_per_scheduled_game"] for r in train])
    if architecture == "ridge":
        learner = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=100),
        )
    elif architecture == "boost":
        x, xt = tree_matrices(x, xt)
        learner = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=10,
            max_bins=63,
            early_stopping=False,
            random_state=20260923,
        )
    else:
        raise ValueError("Unknown architecture")
    with threadpool_limits(limits=1):
        if x.shape[1]:
            learner.fit(x, y)
            predicted = learner.predict(xt)
        else:
            predicted = np.repeat(y.mean(), len(test))
    values = np.maximum(predicted, 0) * np.array(
        [nominal_games(r["forecast_season"]) for r in test]
    )
    for i, row in enumerate(test):
        if row.get("known_available_games_cap") == 0:
            values[i] = 0
    return values


def predict(rows, specs, *, progress=False, architectures=ARCHITECTURES):
    result, folds, skipped = [], [], []
    for year in sorted({r["forecast_season"] for r in rows}):
        for position in POSITIONS:
            train = [r for r in rows if r["forecast_season"] < year and r["position"] == position]
            test = [
                r.copy() for r in rows if r["forecast_season"] == year and r["position"] == position
            ]
            if not test:
                continue
            if len(train) < 30:
                skipped.append(
                    dict(season=year, position=position, train=len(train), test=len(test))
                )
                continue
            recent = [r for r in train if r["forecast_season"] >= year - 5]
            folds.append(
                dict(
                    season=year,
                    position=position,
                    train=len(train),
                    test=len(test),
                    first_train_season=min(r["forecast_season"] for r in train),
                    last_train_season=max(r["forecast_season"] for r in train),
                    training_seasons=len({r["forecast_season"] for r in train}),
                    market_train=sum(finite(r.get("market_ecr")) for r in train),
                    five_year_train=len(recent),
                )
            )
            for architecture in architectures:
                for spec, columns in specs.items():
                    values = fit_predict(train, test, columns, architecture)
                    for r, value in zip(test, values, strict=True):
                        r[f"pred_{architecture}_{spec}"] = float(value)
                    if year >= 2019 and spec in ("profile", "profile_market"):
                        values = fit_predict(recent, test, columns, architecture)
                        for r, value in zip(test, values, strict=True):
                            r[f"pred_{architecture}_{spec}_five_year"] = float(value)
            if progress:
                print(
                    f"Predicted {year} {position}: train={len(train)}, test={len(test)}",
                    flush=True,
                )
            result.extend(test)
    return result, folds, skipped


def uncertainty(values):
    a = np.asarray(values, dtype=float)
    boot = np.random.default_rng(20260923).choice(a, (10000, len(a))).mean(axis=1)
    omitted = [float(np.delete(a, i).mean()) for i in range(len(a))] if len(a) > 1 else []
    return dict(
        mean=float(a.mean()),
        bootstrap_95=np.quantile(boot, [0.025, 0.975]).tolist(),
        seasons=len(a),
        positive_seasons=int((a > 0).sum()),
        leave_one_season_out_range=[min(omitted), max(omitted)] if omitted else None,
    )


def cohorts(rows):
    returners = [r for r in rows if r["player_population"] == "returner"]
    return {
        "all": rows,
        "returners": returners,
        "rookies": [r for r in rows if r["player_population"] == "rookie"],
        "market_only": [r for r in rows if r["player_population"] == "market_only"],
        "prior_season_under10": [
            r for r in returners if finite(r.get("games")) and r["games"] < 10
        ],
        "career_under10": [
            r
            for r in returners
            if r["career_observed_weeks"] < 10 and r["career_left_truncated"] == 0
        ],
        "short_prior_established_career": [
            r
            for r in returners
            if finite(r.get("games")) and r["games"] < 10 and r["career_observed_weeks"] >= 10
        ],
    }


def diagnostics(rows, models):
    comparisons = [
        ("ridge_career", "ridge_basic"),
        ("ridge_profile", "ridge_career"),
        ("ridge_profile", "ridge_basic"),
        ("ridge_profile_market", "ridge_basic_market"),
        ("boost_career", "boost_basic"),
        ("boost_profile", "boost_career"),
        ("boost_profile", "boost_basic"),
        ("boost_profile_market", "boost_basic_market"),
        ("boost_profile", "ridge_profile"),
        ("boost_profile_market", "saved_original_v2"),
        ("boost_profile_market", "saved_backfilled_v2"),
        ("boost_profile", "boost_profile_five_year"),
        ("boost_profile_market", "boost_profile_market_five_year"),
        ("ridge_profile", "ridge_profile_five_year"),
        ("ridge_profile_market", "ridge_profile_market_five_year"),
    ]
    result = []
    for window, (start, end) in WINDOWS.items():
        period = [r for r in rows if start <= r["forecast_season"] <= end]
        for position in ("ALL", *POSITIONS):
            pool = [r for r in period if position == "ALL" or r["position"] == position]
            for cohort, group in cohorts(pool).items():
                if not group:
                    continue
                scores, paired = {}, {}
                for model in models:
                    valid = [r for r in group if finite(r.get(f"pred_{model}"))]
                    if not valid:
                        continue
                    years = sorted({r["forecast_season"] for r in valid})
                    per_year = []
                    for y in years:
                        differences = [
                            r[f"pred_{model}"] - r["actual_season_points"]
                            for r in valid
                            if r["forecast_season"] == y
                        ]
                        per_year.append(
                            dict(
                                season=y,
                                n=len(differences),
                                mae=float(np.mean(np.abs(differences))),
                                rmse=float(np.sqrt(np.mean(np.square(differences)))),
                            )
                        )
                    scores[model] = dict(
                        n=len(valid),
                        seasons=len(years),
                        mae=float(np.mean([r["mae"] for r in per_year])),
                        rmse=float(np.mean([r["rmse"] for r in per_year])),
                        per_year=per_year,
                    )
                for challenger, baseline in comparisons:
                    valid = [
                        r
                        for r in group
                        if finite(r.get(f"pred_{challenger}")) and finite(r.get(f"pred_{baseline}"))
                    ]
                    if not valid:
                        continue
                    values = []
                    for y in sorted({r["forecast_season"] for r in valid}):
                        values.append(
                            float(
                                np.mean(
                                    [
                                        abs(r[f"pred_{baseline}"] - r["actual_season_points"])
                                        - abs(r[f"pred_{challenger}"] - r["actual_season_points"])
                                        for r in valid
                                        if r["forecast_season"] == y
                                    ]
                                )
                            )
                        )
                    paired[f"{challenger}_vs_{baseline}"] = {"n": len(valid), **uncertainty(values)}
                result.append(
                    dict(
                        window=window,
                        position=position,
                        cohort=cohort,
                        n=len(group),
                        scores=scores,
                        paired_mae_improvement=paired,
                    )
                )
    return result


def market_evaluation(predictions, models, original_misses):
    result, misses, named = {}, [], []
    for pool_name in ("all_candidates", "original_v2_pool"):
        allowed_models = (
            models
            if pool_name == "original_v2_pool"
            else [m for m in models if not m.startswith("saved_")]
        )
        for window, (start, end) in {"expanded": (2011, 2025), "modern": (2019, 2025)}.items():
            for model in allowed_models:
                folds = []
                for year in range(start, end + 1):
                    pool = [
                        r.copy()
                        for r in predictions
                        if r["forecast_season"] == year
                        and finite(r.get("market_overall_ecr_score"))
                        and finite(r.get(f"pred_{model}"))
                        and (
                            pool_name != "original_v2_pool"
                            or finite(r.get("pred_saved_original_v2"))
                        )
                    ]
                    if len(pool) < 60:
                        continue
                    market = rank_ids(
                        pool, {r["player_id"]: r["market_overall_ecr_score"] for r in pool}
                    )
                    predicted = rank_ids(pool, model_values(pool, f"pred_{model}"))
                    realized = rank_ids(
                        pool, {r["player_id"]: r["actual_availability_value"] for r in pool}
                    )
                    folds.append({"season": year, **score(pool, predicted, market, 60)})
                    for r in pool:
                        pid = r["player_id"]
                        if pid in set(realized[:60]) - set(market[:60]):
                            entry = dict(
                                pool=pool_name,
                                window=window,
                                model=model,
                                season=year,
                                player_id=pid,
                                name=r["player_display_name"],
                                position=r["position"],
                                population=r["player_population"],
                                prior_games=r["games"],
                                career_observed_weeks=r["career_observed_weeks"],
                                recovered=pid in predicted[:60],
                            )
                            misses.append(entry)
                        if (
                            pool_name == "original_v2_pool"
                            and window == "modern"
                            and (pid, year) in original_misses
                        ):
                            named.append(
                                dict(
                                    model=model,
                                    season=year,
                                    player_id=pid,
                                    name=r["player_display_name"],
                                    position=r["position"],
                                    prior_games=r["games"],
                                    career_weeks=r["career_observed_weeks"],
                                    predicted_points=r[f"pred_{model}"],
                                    actual_points=r["actual_season_points"],
                                    model_rank=predicted.index(pid) + 1,
                                    recovered=pid in predicted[:60],
                                )
                            )
                if folds:
                    own = [
                        r
                        for r in misses
                        if r["pool"] == pool_name and r["window"] == window and r["model"] == model
                    ]
                    result[f"{pool_name}:{window}:{model}"] = {
                        **market_summary(folds, 60),
                        "small_prior_returner_misses": sum(
                            r["population"] == "returner" and r["prior_games"] < 10 for r in own
                        ),
                        "small_prior_returner_recovered": sum(
                            r["population"] == "returner"
                            and r["prior_games"] < 10
                            and r["recovered"]
                            for r in own
                        ),
                        "RB_misses": sum(r["position"] == "RB" for r in own),
                        "RB_recovered": sum(r["position"] == "RB" and r["recovered"] for r in own),
                    }
    original = result["original_v2_pool:modern:saved_original_v2"]
    if (
        original["calls"],
        original["hits"],
        original["small_prior_returner_misses"],
        original["small_prior_returner_recovered"],
        original["RB_misses"],
        original["RB_recovered"],
    ) != (87, 24, 13, 0, 29, 2):
        raise ValueError("Original V2 pool does not reconcile")
    return result, misses, named


def verify_cross_version(original, latest):
    invariant = [
        *KEYS,
        "position",
        "player_population",
        "outcome_complete",
        "actual_season_points",
        "actual_games",
        "actual_availability_value",
        "market_ecr",
        "market_overall_ecr_score",
    ]
    for frame in (original, latest):
        unique(frame, KEYS)
    if not original.select(invariant).sort(KEYS).equals(latest.select(invariant).sort(KEYS)):
        raise ValueError("Candidate/outcome/market invariants differ across source versions")
    dates = original.select(*KEYS, "forecast_cutoff_date").join(
        latest.select(*KEYS, pl.col("forecast_cutoff_date").alias("latest_cutoff")),
        on=KEYS,
        validate="1:1",
    )
    if dates.filter(
        pl.col("forecast_cutoff_date").is_not_null()
        & ~pl.col("forecast_cutoff_date").eq_missing(pl.col("latest_cutoff"))
    ).height:
        raise ValueError("Existing forecast cutoffs changed across versions")


def coverage(rows):
    result = []
    for year in sorted({r["forecast_season"] for r in rows}):
        for position in POSITIONS:
            group = [r for r in rows if r["forecast_season"] == year and r["position"] == position]
            result.append(
                dict(
                    season=year,
                    position=position,
                    candidates=len(group),
                    prior_nfl_history=sum(r["career_observed_weeks"] > 0 for r in group),
                    prior_snap_history=sum(r["career_snap_observations"] > 0 for r in group),
                    college_linked=sum(r["college_linked"] == 1 for r in group),
                    college_complete_history=sum(r["college_complete_seasons"] > 0 for r in group),
                    injury_reports=sum((r["injury_career_reported_weeks"] or 0) > 0 for r in group),
                    dated_roster_state=sum(r["context_cutoff_state_observed"] == 1 for r in group),
                    known_absence=sum((r["context_known_absence_games"] or 0) > 0 for r in group),
                    job_evidence=sum(r["evidence_state"] != "unknown" for r in group),
                    positional_ecr=sum(finite(r.get("market_ecr")) for r in group),
                    overall_ecr=sum(finite(r.get("market_overall_ecr_score")) for r in group),
                )
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="historical_backfill_20260923_r2")
    parser.add_argument("--profiles", default="player_profiles_v1_20260923_r2")
    parser.add_argument("--college", default="college_nfl_v1_20260922_r5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "data/research"):
        raise ValueError("Output must be a new directory under data/research")
    print("Verifying accepted historical, profile and college dependencies", flush=True)
    history, acceptance, manifest = accepted_version(args.history)

    def reference(name):
        return dict(
            version=name, manifest_sha256=digest(ROOT / "data/research" / name / "manifest.json")
        )

    profiles, profile_report = load_profiles(reference_override=reference(args.profiles))
    college, college_report = load_college(reference_override=reference(args.college))
    if profile_report["history"] != college_report["history"]:
        raise ValueError("Profile and college source histories differ")
    old_history, old_acceptance, _ = accepted_version(profile_report["history"]["version"])
    latest = pl.read_parquet(history / "outputs/metric_backtest_predictions.parquet")
    original = pl.read_parquet(old_history / "outputs/metric_backtest_predictions.parquet")
    verify_cross_version(original, latest)
    protected = {name: digest(ROOT / name) for name in manifest["protected_artifact_sha256"]}
    protected_pointers = [
        ROOT / "data/outputs" / name for name in ("player_profiles.json", "college_nfl.json")
    ]
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in protected_pointers})
    cols = [
        *KEYS,
        "player_display_name",
        "position",
        "player_population",
        "forecast_cutoff_date",
        "games",
        "age_at_season",
        "player_experience",
        "actual_season_points",
        "actual_games",
        "actual_availability_value",
        "market_ecr",
        "market_ecr_sd",
        "market_overall_ecr",
        "market_overall_ecr_score",
        "market_overall_ecr_sd",
        "fitted_season_points",
        *CONTEXT,
        "availability_latest_known_on",
        "cutoff_last_transaction_date",
    ]
    candidates = latest.filter(
        pl.col("outcome_complete") & pl.col("position").is_in(POSITIONS)
    ).select(cols)
    candidates = candidates.with_columns(
        pl.col(
            "forecast_cutoff_date", "availability_latest_known_on", "cutoff_last_transaction_date"
        ).cast(pl.String)
    )
    candidates = candidates.join(
        original.select(*KEYS, pl.col("fitted_season_points").alias("pred_saved_original_v2")),
        on=KEYS,
        validate="1:1",
    ).rename({"fitted_season_points": "pred_saved_backfilled_v2"})
    paths = [
        history / "manifest.json",
        history / "acceptance.json",
        old_history / "manifest.json",
        old_history / "acceptance.json",
        history / "outputs/metric_backtest_predictions.parquet",
        old_history / "outputs/metric_backtest_predictions.parquet",
        profiles / "manifest.json",
        college / "manifest.json",
        DESIGN,
        EVIDENCE,
    ]

    def read(path):
        paths.append(path)
        return pl.read_parquet(path)

    rows = build_features(
        candidates,
        read(profiles / "nfl_weeks.parquet"),
        read(profiles / "players.parquet"),
        read(college / "college_annual.parquet"),
        read(college / "identity_links.parquet"),
        read(profiles / "injuries.parquet"),
        load_evidence(EVIDENCE),
        progress=True,
    )
    specs = specifications(rows)
    print("Feature counts: " + json.dumps({k: len(v) for k, v in specs.items()}), flush=True)
    predictions, folds, skipped = predict(rows, specs, progress=True)
    models = [f"{a}_{s}" for a in ARCHITECTURES for s in SPECS]
    sensitivity_models = [
        f"{a}_{s}_five_year" for a in ARCHITECTURES for s in ("profile", "profile_market")
    ]
    benchmarks = ["saved_original_v2", "saved_backfilled_v2"]
    miss_path = ROOT / "data/research/qb_role_transition_20260923_r3/market_misses.parquet"
    missed = read(miss_path).filter((pl.col("model") == "saved_v2") & (pl.col("prior_games") < 10))
    original_misses = {(r["player_id"], r["season"]) for r in missed.to_dicts()}
    print("Evaluating historical, modern, sparse-history and market comparisons", flush=True)
    market, misses, named = market_evaluation(predictions, [*models, *benchmarks], original_misses)
    implementation_paths = [
        Path(__file__),
        ROOT / "research/test_profile_history.py",
        ROOT / "research/qb_role_transition.py",
        ROOT / "research/preseason_role_workload.py",
        ROOT / "research/value_capture_audit.py",
        ROOT / "research/artifact_inputs.py",
        ROOT / "research/data_integrity_audit.py",
        ROOT / "src/engine/metrics/player_profile.py",
        ROOT / "src/engine/metrics/outlook.py",
        ROOT / "src/engine/api/profile_sources.py",
        ROOT / "src/engine/api/college_sources.py",
        ROOT / "src/engine/api/research_sources.py",
        ROOT / "src/engine/data/releases.py",
    ]
    report = dict(
        generated_at=datetime.now(UTC).isoformat(),
        research_only=True,
        status="experimental_not_promoted",
        history=acceptance["audited_input"],
        profile_history=old_acceptance["audited_input"],
        profiles=reference(args.profiles),
        college=reference(args.college),
        cross_version_invariants_verified=True,
        package_versions={name: version(name) for name in ("numpy", "polars", "scikit-learn")},
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))},
        implementation_sha256={str(p.relative_to(ROOT)): digest(p) for p in implementation_paths},
        design=dict(
            specifications=specs,
            architectures=list(ARCHITECTURES),
            windows=WINDOWS,
            target="full-season points / nominal season length; converted back to season points",
            training="all earlier accepted candidate years; no rolling window in primary models",
            five_year_sensitivity="training rows restricted; full career features unchanged",
            ridge_alpha=100,
            boost=dict(
                iterations=120,
                learning_rate=0.05,
                leaves=15,
                minimum_leaf=30,
                l2=10,
                max_bins=63,
                early_stopping=False,
            ),
        ),
        candidate_rows=len(rows),
        predicted_rows=len(predictions),
        populations=dict(Counter(r["player_population"] for r in rows)),
        folds=folds,
        unscored_initial_training_rows=skipped,
        coverage=coverage(rows),
        diagnostics=diagnostics(predictions, [*models, *benchmarks, *sensitivity_models]),
        market=market,
        original_small_sample_misses=13,
        original_miss_recoveries={
            m: sum(r["recovered"] for r in named if r["model"] == m) for m in [*models, *benchmarks]
        },
        protected_artifact_sha256=protected,
        protected_unchanged=all(digest(ROOT / name) == sha for name, sha in protected.items()),
        limitations=[
            "Candidates start in 2004; earlier NFL observations inform their features.",
            "NFL before 2001 and college before 2004 can be truncated; retain incomplete profiles.",
            "Coverage varies by source/year/player; unknown is not confirmed healthy or inactive.",
            "Identity, draft facts and historical observations may contain later source revisions.",
            "Previously inspected eras and multiple comparisons remain exploratory.",
            "Fixed model choices are unvalidated; no parameter tuning uses reported test folds.",
            "Direct points include absences; partial absence inputs are not points ceilings.",
            "The broad pool includes rookies and market-only players; legacy pools are separate.",
            "Snap bands are not routes or starting jobs. Dated QB job evidence is partial.",
            "No ADP is backdated. Top-60 acquisition value is a proxy, not realized profit.",
        ],
    )
    if not report["protected_unchanged"]:
        raise ValueError("Protected artifacts changed during experiment")
    json.dumps(report, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    for name in report["implementation_sha256"]:
        target = output / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copy2(DESIGN, output / DESIGN.name)
    shutil.copy2(EVIDENCE, output / EVIDENCE.name)
    for name, values in (
        ("features", rows),
        ("predictions", predictions),
        ("market_misses", misses),
        ("original_small_sample_misses", named),
    ):
        pl.DataFrame(values, infer_schema_length=None).write_parquet(output / f"{name}.parquet")
    pl.DataFrame(report["coverage"]).write_csv(output / "coverage.csv")
    (output / "feature_dictionary.json").write_text(json.dumps(specs, indent=2) + "\n")
    report["output_sha256"] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                output=str(output),
                candidates=len(rows),
                predictions=len(predictions),
                miss_recoveries=report["original_miss_recoveries"],
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
