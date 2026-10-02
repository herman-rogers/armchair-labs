"""Bounded QB efficiency/volume search and a chronological fantasy bridge."""

from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from patron.metrics.current_rankings import metrics, team_code

HALVES = (0.5, 1, 2, 3, 5, 9999)
PRIORS = (0, 50, 150, 400, 1000)
LEAVES = (7, 15, 31)
ALPHAS = (10, 100, 1000)
BOOST = dict(
    max_iter=120,
    learning_rate=0.05,
    min_samples_leaf=60,
    l2_regularization=10,
    max_bins=63,
    early_stopping=False,
    random_state=20260924,
)
KEY = ["player_id", "season", "through_week", "horizon"]


def rate_name(half, prior):
    return f"rate_career_h{str(half).replace('.', 'p')}_p{prior}"


def augment_panel(panel, weeks, features, ranking_panel):
    """The suffix of a season supplies labels only, including exact ranking weeks."""
    players, seasons = defaultdict(list), defaultdict(list)
    ids = {r["player_id"] for r in panel}
    for r in (
        weeks.filter(pl.col("player_id").is_in(ids) & (pl.col("season_type") == "REG"))
        .sort("season", "week")
        .to_dicts()
    ):
        players[r["player_id"]].append(r)
        seasons[r["season"]].append(r)
    enriched = {
        (r["player_id"], r["forecast_season"]): r
        for r in features.select(
            "player_id", "forecast_season", "ngs_cpoe", "team_changed"
        ).to_dicts()
    }
    college = {
        (r["player_id"], r["season"]): {k: v for k, v in r.items() if k.startswith("e_college")}
        for r in ranking_panel
    }
    leagues = {}
    for year in {r["season"] for r in panel}:
        old = [r for y, rows in seasons.items() if y < year for r in rows]
        leagues[year] = sum(r.get("passing_yards") or 0 for r in old) / max(
            1, sum(r.get("attempts") or 0 for r in old)
        )
    cache = {}
    for row in panel:
        pid, year, origin = row["player_id"], row["season"], row["through_week"]
        key = pid, year, origin
        if key not in cache:
            history = [
                r
                for r in players[pid]
                if r["season"] < year or r["season"] == year and r["week"] <= origin
            ]
            history = [r for r in history if (r.get("attempts") or 0) > 0]
            league = leagues[year]
            extra = {"rate_reference": row["execution_ypa"], "rate_league": league}

            def estimate(subset, prior, half=None, year=year, league=league):
                a = y = 0.0
                for r in subset:
                    w = 2 ** (-(year - r["season"]) / half) if half else 1.0
                    a += w * r["attempts"]
                    y += w * r["passing_yards"]
                return (y + prior * league) / (a + prior) if a + prior else league

            for half, prior in product(HALVES, PRIORS):
                extra[rate_name(half, prior)] = estimate(history, prior, half)
            for games, prior in product((4, 8, 16, 32), (150, 400)):
                extra[f"rate_last{games}_p{prior}"] = estimate(history[-games:], prior)
            for label, subset in (
                ("prior", [r for r in history if r["season"] == year - 1]),
                ("current", [r for r in history if r["season"] == year]),
            ):
                extra[f"rate_{label}_p400"] = estimate(subset, 400)
            for name in (
                rate_name(2, 400),
                rate_name(0.5, 150),
                "rate_last8_p400",
                "rate_current_p400",
                "rate_prior_p400",
            ):
                extra["f_" + name] = extra[name]
            for label, subset in (
                ("career", history),
                ("recent", history[-16:]),
                ("current", [r for r in history if r["season"] == year]),
            ):
                a = sum(r["attempts"] for r in subset)
                extra[f"f_{label}_attempts"] = np.log1p(a)
                for field in (
                    "completions",
                    "passing_tds",
                    "passing_interceptions",
                    "passing_epa",
                    "rushing_yards",
                    "rushing_tds",
                    "carries",
                ):
                    observed = [r for r in subset if r.get(field) is not None]
                    denom = sum(r["attempts"] for r in observed)
                    extra[f"f_{label}_{field}"] = (
                        sum(r[field] for r in observed) / denom if denom else None
                    )
                observed = [r for r in subset if r.get("passing_cpoe") is not None]
                denom = sum(r["attempts"] for r in observed)
                extra[f"e_{label}_cpoe"] = (
                    sum(r["passing_cpoe"] * r["attempts"] for r in observed) / denom
                    if denom
                    else None
                )
            prior_games = [r for r in players[pid] if r["season"] == year - 1][-4:]
            extra["f_prior_last4_attempts"] = (
                np.mean([r.get("attempts") or 0 for r in prior_games]) if prior_games else None
            )
            team = team_code(row["team"])
            team_rows = [
                r for r in seasons[year] if r["week"] <= origin and team_code(r["team"]) == team
            ]
            ta = sum(r.get("attempts") or 0 for r in team_rows)
            pa = sum(r.get("attempts") or 0 for r in team_rows if r["player_id"] == pid)
            extra["f_current_team_share"] = pa / ta if ta else None
            extra["f_competing_attempts"] = (ta - pa) / max(row["x_elapsed"], 1)
            profile = enriched.get((pid, year), {})
            extra["e_ngs_cpoe"] = profile.get("ngs_cpoe")
            extra["e_team_changed"] = (
                float(profile["team_changed"]) if profile.get("team_changed") is not None else None
            )
            extra.update(college.get((pid, year), {}))
            cache[key] = extra
        row.update(cache[key])
        future = [
            r for r in players[pid] if r["season"] == year and origin < r["week"] <= row["end_week"]
        ]
        row["actual_points"] = (
            sum(r.get("league_points") or 0 for r in future) if row["actual"] is not None else None
        )
    # Extra targets use the existing ranking population and its precise time interval.
    templates = {
        (r["player_id"], r["season"], r["through_week"]): r
        for r in panel
        if r["horizon"] == "rest_of_season"
    }
    ranking_rows = []
    for rank in ranking_panel:
        if rank["position"] != "QB":
            continue
        key = rank["player_id"], rank["season"], rank["through_week"]
        if key not in templates:
            raise ValueError(f"Missing frozen QB ranking candidate {key}")
        row = dict(templates[key])
        row.update(
            horizon="ranking_" + rank["horizon"],
            end_week=rank["end_week"],
            scheduled_games=rank["scheduled_games"],
            x_horizon_games=rank["scheduled_games"],
        )
        future = [
            r
            for r in players[rank["player_id"]]
            if r["season"] == rank["season"]
            and rank["through_week"] < r["week"] <= rank["end_week"]
        ]
        for target, field in (
            ("actual", "passing_yards"),
            ("actual_attempts", "attempts"),
            ("actual_points", "league_points"),
        ):
            row[target] = (
                sum(r.get(field) or 0 for r in future) if rank["actual"] is not None else None
            )
        if rank["actual"] is not None and not np.isclose(
            row["actual_points"], rank["actual"], atol=1e-8
        ):
            raise ValueError(
                f"Ranking outcome mismatch {key}: {row['actual_points']} != {rank['actual']}"
            )
        ranking_rows.append(row)
    return panel, ranking_rows


def matrix(rows, columns):
    return np.array(
        [[np.nan if r.get(c) is None else r[c] for c in columns] for r in rows], dtype=float
    )


def model(kind, value):
    if kind == "ridge":
        return make_pipeline(
            SimpleImputer(add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=value),
        )
    return HistGradientBoostingRegressor(max_leaf_nodes=value, **BOOST)


def fitted(kind, value, x, y, z, weights=None):
    # Sparse college/tracking columns can be entirely absent in early folds.
    # Discover usable columns from training only; later availability cannot decide.
    varying = [i for i in range(x.shape[1]) if len(np.unique(x[np.isfinite(x[:, i]), i])) > 1]
    if not varying:
        return np.full(len(z), np.average(y, weights=weights))
    x, z = x[:, varying], z[:, varying]
    estimator = model(kind, value)
    kwargs = (
        {"ridge__sample_weight" if kind == "ridge" else "sample_weight": weights}
        if weights is not None
        else {}
    )
    with threadpool_limits(limits=1):
        estimator.fit(x, y, **kwargs)
        return estimator.predict(z)


def columns_for(rows):
    efficiency = sorted(
        c
        for c in rows[0]
        if c.startswith("f_")
        and "team_share" not in c
        and "competing" not in c
        and "last4_attempts" not in c
    )
    profile = (
        efficiency
        + sorted(c for c in rows[0] if c.startswith("x_"))
        + ["f_prior_last4_attempts", "f_current_team_share", "f_competing_attempts"]
    )
    enriched = profile + sorted({c for r in rows for c in r if c.startswith("e_")})
    return dict(efficiency=efficiency, profile=profile, enriched=enriched)


def fit_rates(train, test):
    train = [r for r in train if r["actual_attempts"] > 0]
    y = np.array([r["actual"] / r["actual_attempts"] for r in train])
    weights = np.array([r["actual_attempts"] for r in train], dtype=float)
    for year in {r["season"] for r in train}:
        indices = np.array([r["season"] == year for r in train])
        weights[indices] /= weights[indices].sum()
    weights *= len(weights) / weights.sum()
    results = {c: np.array([r[c] for r in test]) for c in test[0] if c.startswith("rate_")}
    cols = columns_for(train)
    for group, columns in cols.items():
        x, z = matrix(train, columns), matrix(test, columns)
        candidates = [("ridge", a) for a in ALPHAS] + (
            [] if group == "efficiency" else [("boost", leaf) for leaf in LEAVES]
        )
        for kind, value in candidates:
            name = f"rate_{group}_{kind}_{value}"
            results[name] = np.clip(fitted(kind, value, x, y, z, weights), 0, 20)
            results[name + "_blend"] = 0.5 * results[name] + 0.5 * results[rate_name(2, 400)]
    return results


def fit_production(train, test, selected_rates):
    cols = columns_for(train)
    games = np.array([r["scheduled_games"] for r in test])
    available = np.array([r["allowed_fraction"] > 0 for r in test])
    y = np.array([r["actual"] / r["scheduled_games"] for r in train])
    a = np.array([r["actual_attempts"] / r["scheduled_games"] for r in train])
    results = {}
    for group in ("profile", "enriched"):
        x, z = matrix(train, cols[group]), matrix(test, cols[group])
        for leaf in LEAVES:
            name = f"yards_{group}_boost_{leaf}"
            direct = np.maximum(fitted("boost", leaf, x, y, z), 0) * games * available
            attempts = np.maximum(fitted("boost", leaf, x, a, z), 0) * games * available
            results[name] = direct
            results[name + "_component"] = attempts * selected_rates
            results[name + "_blend"] = 0.5 * direct + 0.5 * results[name + "_component"]
            results[f"attempts_{group}_boost_{leaf}"] = attempts
        if group == "enriched":
            for alpha in ALPHAS:
                results[f"yards_enriched_ridge_{alpha}"] = (
                    np.maximum(fitted("ridge", alpha, x, y, z), 0) * games * available
                )
    return results


def score(rows, field, target="yards"):
    if target == "rate":
        rows = [r for r in rows if r["actual_attempts"] > 0]
    annual = []
    for year in sorted({r["season"] for r in rows}):
        subset = [r for r in rows if r["season"] == year]
        actual = np.array(
            [
                r["actual"] / r["actual_attempts"] if target == "rate" else r["actual"]
                for r in subset
            ]
        )
        errors = np.array([r[field] for r in subset]) - actual
        weights = (
            np.array([r["actual_attempts"] for r in subset])
            if target == "rate"
            else np.ones(len(subset))
        )
        annual.append(
            dict(
                season=year,
                n=len(subset),
                exposure=float(weights.sum()),
                mse=float(np.average(errors**2, weights=weights)),
                mae=float(np.mean(np.abs(errors))),
                weighted_mae=float(np.average(np.abs(errors), weights=weights)),
            )
        )
    if not annual:
        return dict(n=0, years=0, annual=[])
    mse = float(np.mean([r["mse"] for r in annual]))
    return dict(
        n=len(rows),
        years=len(annual),
        annual=annual,
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(np.mean([r["mae"] for r in annual])),
        weighted_mae=float(np.mean([r["weighted_mae"] for r in annual])),
    )


def choose(earlier, horizon, origin, candidates, default, target):
    rows = [
        r
        for r in earlier
        if r["horizon"] == horizon and (r["through_week"] == 0) == (origin == "preseason")
    ]
    if len({r["season"] for r in rows}) < 3:
        return default
    return min(candidates, key=lambda name: (score(rows, name, target)["mse"], name))


def bridge_candidates(train, test, passing_coefficient):
    """Only earlier out-of-year upstream forecasts enter this fit."""
    if max(r["season"] for r in train) >= min(r["season"] for r in test):
        raise ValueError("The fantasy bridge requires strictly earlier cross-fitted seasons")
    cols = columns_for(train)["enriched"] + ["stack_rate", "stack_yards_pg", "stack_attempts_pg"]
    x, z = matrix(train, cols), matrix(test, cols)
    y = np.array([r["actual_points"] / r["scheduled_games"] for r in train])
    nonyard = y - passing_coefficient * np.array(
        [r["actual"] / r["scheduled_games"] for r in train]
    )
    games = np.array([r["scheduled_games"] for r in test])
    yard_points = np.array([r["stack_yards_pg"] for r in test]) * passing_coefficient
    raw = {}
    for kind, val in [("ridge", a) for a in ALPHAS] + [("boost", leaf) for leaf in LEAVES]:
        raw[f"points_{kind}_{val}"] = np.maximum(fitted(kind, val, x, y, z), 0) * games
        if kind == "boost":
            raw[f"points_component_{val}"] = (
                np.maximum(fitted(kind, val, x, nonyard, z) + yard_points, 0) * games
            )
    return {
        f"{name}_w{int(w * 100)}": np.maximum(
            w * values + (1 - w) * np.array([r["reference"] for r in test]), 0
        )
        for name, values in raw.items()
        for w in (0.25, 0.5, 0.75, 1.0)
    }


def choose_bridge(earlier, role, candidates):
    rows = [r for r in earlier if r["role_group"] == role]
    years = sorted({r["season"] for r in rows})
    if len(years) < 4:
        return "reference"
    scores = {
        m: {
            k: float(
                np.mean([metrics([r for r in rows if r["season"] == y], m, 10)[k] for y in years])
            )
            for k in ("mae", "mse", "capture")
        }
        for m in ["reference", *candidates]
    }
    b = scores["reference"]
    qualified = [
        m
        for m in candidates
        if scores[m]["mae"] <= b["mae"] - max(0.5, 0.01 * b["mae"])
        and scores[m]["mse"] <= b["mse"]
        and scores[m]["capture"] >= b["capture"]
    ]
    return min(qualified, key=lambda m: (scores[m]["mae"], m)) if qualified else "reference"


def paired(rows, model_name, comparator, target):
    p, b = score(rows, model_name, target), score(rows, comparator, target)
    gains = np.array(
        [base["mse"] - test["mse"] for base, test in zip(b["annual"], p["annual"], strict=True)]
    )
    boot = np.random.default_rng(20260924).choice(gains, (5000, len(gains))).mean(axis=1)
    p_value = None
    if len(gains) <= 10:
        flips = np.array(list(product((-1, 1), repeat=len(gains))))
        p_value = float(np.mean(np.abs((flips * gains).mean(axis=1)) >= abs(gains.mean()) - 1e-12))
    return dict(
        p,
        model=model_name,
        comparator=comparator,
        baseline_mse=b["mse"],
        baseline_mae=b["mae"],
        improvement=float(gains.mean()),
        improvement_pct=100 * float(gains.mean()) / b["mse"],
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
        p_value=p_value,
    )


def adjust_q(decisions):
    ordered = sorted(decisions, key=lambda d: d["modern"].get("p_value") or 1.0)
    running = 1.0
    for index in range(len(ordered) - 1, -1, -1):
        running = min(
            running, (ordered[index]["modern"].get("p_value") or 1.0) * len(ordered) / (index + 1)
        )
        ordered[index]["q_value"] = running


def policy_ranges(rows):
    """Earlier-year residual calibration, never the target year's residuals."""
    residuals = defaultdict(list)
    output = []
    for year in sorted({r["season"] for r in rows}):
        subset = [r for r in rows if r["season"] == year]
        quantiles = {
            key: np.quantile(values, [0.1, 0.9])
            for key, values in residuals.items()
            if len(values) >= 100
        }
        for r in subset:
            q = quantiles.get((r["horizon"], r["role_group"]), quantiles.get((r["horizon"], "all")))
            low, high = (
                np.maximum(q * r["scheduled_games"] + r["yards_policy"], 0)
                if q is not None
                else (None, None)
            )
            if r["allowed_fraction"] == 0:
                low, high = 0.0, 0.0
            output.append(
                dict(
                    player_id=r["player_id"],
                    season=year,
                    through_week=r["through_week"],
                    horizon=r["horizon"],
                    lower=None if low is None else float(low),
                    upper=None if high is None else float(high),
                    actual=r["actual"],
                )
            )
        for r in subset:
            if r["actual"] is not None:
                error = (r["actual"] - r["yards_policy"]) / r["scheduled_games"]
                residuals[r["horizon"], r["role_group"]].append(error)
                residuals[r["horizon"], "all"].append(error)
    return output
