"""Frequentist boosted-tree distribution experiments and proper scoring rules."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def crps_ensemble(y, draws, fair=True):
    """O(n m log m) CRPS; fair U-statistic for IID Monte Carlo samples.

    Empirical-CDF CRPS uses m²; the fair estimator uses m(m−1). Fair scores can
    be slightly negative for a finite sample; do not clip or mislabel them.
    """
    x = np.asarray(draws, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[1] < 2:
        raise ValueError("CRPS requires one row per observation and at least two draws")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("CRPS inputs must be finite")
    m = x.shape[1]
    spread = np.sum(np.sort(x, axis=1) * (2 * np.arange(1, m + 1) - m - 1), axis=1)
    return np.mean(np.abs(x - y[:, None]), axis=1) - spread / (m * (m - 1 if fair else m))


def negative_log_score(y, mean, alpha=0.0):
    """Analytic normalized count PMF; NB2 variance = mean + alpha * mean²."""
    y = np.asarray(y)
    mu = np.maximum(np.asarray(mean), 1e-9)
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("Count log scores require nonnegative integer observations")
    if alpha <= 1e-8:
        return -poisson.logpmf(y, mu)
    return -nbinom.logpmf(y, 1 / alpha, 1 / (1 + alpha * mu))


def estimate_dispersion(y, mu):
    opt = minimize_scalar(
        lambda a: negative_log_score(y, mu, a).mean(), bounds=(1e-7, 50), method="bounded"
    )
    return float(opt.x) if opt.fun < negative_log_score(y, mu).mean() else 0.0


def count_draws(mu, alpha, n, rng):
    mu = np.maximum(np.asarray(mu), 1e-9)[:, None]
    if alpha <= 1e-8:
        return rng.poisson(mu, size=(len(mu), n))
    return rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu), size=(len(mu), n))


def score_distribution(y, draws):
    y = np.asarray(y)
    q = np.quantile(draws, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], axis=1)
    mean = draws.mean(axis=1)
    rng = np.random.default_rng(58201)
    # Randomized finite-ensemble ranks handle mass at zero without a fake continuous PIT.
    pit = (
        (draws < y[:, None]).sum(axis=1)
        + rng.random(len(y)) * (1 + (draws == y[:, None]).sum(axis=1))
    ) / (draws.shape[1] + 1)
    return {
        "crps": crps_ensemble(y, draws),
        "empirical_crps": crps_ensemble(y, draws, False),
        "absolute_error": np.abs(y - mean),
        "squared_error": (y - mean) ** 2,
        "mean": mean,
        "p05": q[0],
        "p10": q[1],
        "p25": q[2],
        "median": q[3],
        "p75": q[4],
        "p90": q[5],
        "p95": q[6],
        "pit": pit,
        "covered50": ((y >= q[2]) & (y <= q[4])).astype(float),
        "covered80": ((y >= q[1]) & (y <= q[5])).astype(float),
        "covered90": ((y >= q[0]) & (y <= q[6])).astype(float),
        "width80": q[5] - q[1],
    }


def matrix(frame, columns):
    return frame.select(columns).to_numpy().astype(float)


@dataclass
class Fitted:
    model: object
    columns: list[str]
    constant: float | None = None

    def predict(self, frame):
        if self.constant is not None:
            return np.full(frame.height, self.constant)
        if hasattr(self.model, "booster_"):
            return self.model.booster_.predict(matrix(frame, self.columns), num_threads=1)
        return self.model.predict(matrix(frame, self.columns))


def fit_tree(
    frame,
    columns,
    target,
    rounds=80,
    engine="lightgbm",
    loss="squared_error",
    seed=19,
    weights=None,
):
    y = np.asarray(target, dtype=float)
    ok = np.isfinite(y)
    frame = frame.filter(pl.Series(ok))
    y = y[ok]
    if not len(y):
        raise ValueError("No known labels for fit")
    x = matrix(frame, columns)
    # Learn missing/constant-column eligibility from fitting data only.
    active = np.array([np.unique(v[np.isfinite(v)]).size > 1 for v in x.T])
    names = np.array(columns)[active].tolist()
    if not names or np.std(y) < 1e-10:
        return Fitted(None, names, float(np.mean(y)))
    if engine in {"lightgbm", "xgboost"}:
        packages = Path(__file__).resolve().parents[1] / ".packages"
        if packages.exists() and str(packages) not in sys.path:
            sys.path.insert(0, str(packages))
    if engine == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=rounds,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=25,
            reg_lambda=10,
            n_jobs=1,
            verbosity=-1,
            objective="poisson" if loss == "poisson" else "regression",
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
        )
    elif engine == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=rounds,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=25,
            reg_lambda=10,
            n_jobs=1,
            objective="count:poisson" if loss == "poisson" else "reg:squarederror",
            random_state=seed,
            tree_method="hist",
        )
    elif engine == "hist":
        model = HistGradientBoostingRegressor(
            max_iter=rounds,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=10,
            loss=loss,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown engine {engine}")
    w = np.asarray(weights)[ok] if weights is not None else None
    with threadpool_limits(limits=1):
        model.fit(x[:, active], y, sample_weight=w)
    return Fitted(model, names)


def residual_draws(mean, cal_mean, cal_y, n, seed, bins=4):
    """Conditional empirical residual bootstrap; donor bins use predicted mean only."""
    mean, cal_mean, cal_y = map(np.asarray, (mean, cal_mean, cal_y))
    edges = np.unique(np.quantile(cal_mean, np.linspace(0, 1, bins + 1)[1:-1]))
    ci, ti = np.digitize(cal_mean, edges), np.digitize(mean, edges)
    rng = np.random.default_rng(seed)
    out = np.empty((len(mean), n))
    for b in np.unique(ti):
        donor = (cal_y - cal_mean)[ci == b]
        if len(donor) < 15:
            donor = cal_y - cal_mean
        idx = np.flatnonzero(ti == b)
        out[idx] = mean[idx, None] + rng.choice(donor, size=(len(idx), n))
    return out


def baseline(frame):
    return (
        (
            0.1
            * (
                frame["history_rushing_yards"].to_numpy()
                + frame["history_receiving_yards"].to_numpy()
            )
            + frame["history_receptions"].to_numpy()
            + 6
            * (frame["history_rushing_tds"].to_numpy() + frame["history_receiving_tds"].to_numpy())
        )
        / frame["history_weeks"].to_numpy()
        * frame["weeks"].to_numpy()
    )


def divide(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a), dtype=float), where=np.asarray(b) > 0)


COMPONENTS = ["team_plays", "run_fraction", "target_fraction", "carry_share", "target_share"]


def component_targets(frame):
    v = {c: frame[c].to_numpy() for c in frame.columns if c.startswith("y_")}
    return {
        "team_plays": v["y_team_plays"] / frame["weeks"].to_numpy(),
        "run_fraction": divide(v["y_team_carries"], v["y_team_plays"]),
        "target_fraction": divide(v["y_team_targets"], v["y_team_plays"] - v["y_team_carries"]),
        "carry_share": divide(v["y_carries"], v["y_team_carries"]),
        "target_share": divide(v["y_targets"], v["y_team_targets"]),
    }


def efficiency_fit(train, age=False):
    """Exposure-weighted league rates and optional learned quadratic age curve."""
    rates, curves = {}, {}
    for label, num, den in [
        ("ypc", "rushing_yards", "carries"),
        ("ypr", "receiving_yards", "receptions"),
        ("catch", "receptions", "targets"),
        ("rush_td", "rushing_tds", "carries"),
        ("rec_td", "receiving_tds", "receptions"),
    ]:
        y, w = train[f"y_{num}"].to_numpy(), train[f"y_{den}"].to_numpy()
        rates[label] = float(y.sum() / max(w.sum(), 1))
        if age and label in {"ypc", "ypr"}:
            a = train["age"].to_numpy()
            ok = (w > 0) & np.isfinite(a)
            x = np.column_stack([a[ok] - 26, (a[ok] - 26) ** 2])
            reg = Ridge(alpha=100).fit(x, y[ok] / w[ok], sample_weight=w[ok])
            curves[label] = (reg, float(np.min(a[ok])), float(np.max(a[ok])))
    return rates, curves


def efficiency_predict(frame, fitted, shrinkage):
    rates, curves = fitted
    out = {}
    for label, num, den in [
        ("ypc", "rushing_yards", "carries"),
        ("ypr", "receiving_yards", "receptions"),
        ("catch", "receptions", "targets"),
        ("rush_td", "rushing_tds", "carries"),
        ("rec_td", "receiving_tds", "receptions"),
    ]:
        center = np.full(frame.height, rates[label])
        if label in curves:
            reg, low, high = curves[label]
            a = np.clip(np.nan_to_num(frame["age"].to_numpy(), nan=26), low, high) - 26
            center = reg.predict(np.column_stack([a, a * a]))
        y, w = frame[f"history_{num}"].to_numpy(), frame[f"history_{den}"].to_numpy()
        out[label] = np.where(
            w + shrinkage > 0, (y + shrinkage * center) / np.maximum(w + shrinkage, 1), center
        )
        if label in {"catch", "rush_td", "rec_td"}:
            out[label] = np.clip(out[label], 0, 1)
    return out


def fit_components(train, features, rounds, engine, seed):
    unique_team = train.unique(["year", "origin", "team"], maintain_order=True)
    team_features = [
        c
        for c in features
        if c.startswith("team_") or c in {"origin", "weeks", "market_implied", "market_spread"}
    ]
    fits = {}
    for c in COMPONENTS:
        data = unique_team if c in COMPONENTS[:3] else train
        y = component_targets(data)[c]
        # Byes have zero volume, not an observed share. Trades can produce ratios >1.
        if c != "team_plays":
            volume = data[
                "y_team_plays"
                if c in COMPONENTS[:3]
                else "y_team_carries"
                if c == "carry_share"
                else "y_team_targets"
            ].to_numpy()
            mask = (volume > 0) & (y <= 1)
            y = np.where(mask, y, np.nan)
        fits[c] = fit_tree(
            data, team_features if c in COMPONENTS[:3] else features, y, rounds, engine, seed=seed
        )
    return fits


def predict_components(frame, fitted):
    out = {}
    for c, model in fitted.items():
        p = model.predict(frame)
        out[c] = np.maximum(p, 0) if c == "team_plays" else np.clip(p, 0, 1)
    return out


def normalize_shares(frame, values):
    values = np.maximum(np.asarray(values), 0).copy()
    for g in frame.with_row_index("row").partition_by(["year", "origin", "team"]):
        idx = g["row"].to_numpy()
        total = values[idx].sum(axis=0)
        values[idx] /= np.maximum(total, 1)
    return values


def decomposition_mean(frame, components, eff):
    c, e = components, eff
    plays = c["team_plays"] * frame["weeks"].to_numpy()
    carries = plays * c["run_fraction"] * normalize_shares(frame, c["carry_share"])
    targets = (
        plays
        * (1 - c["run_fraction"])
        * c["target_fraction"]
        * normalize_shares(frame, c["target_share"])
    )
    rec = targets * e["catch"]
    td = carries * e["rush_td"] + rec * e["rec_td"]
    return 0.1 * (carries * e["ypc"] + rec * e["ypr"]) + rec + 6 * td


def simulate_decomposition(frame, components, eff, cal, cal_components, cal_eff, n, seed):
    """Shared team volume → competing multinomial shares → catches/efficiency/TDs.

    No posterior or hierarchical Bayesian model. Whole calibration residual tuples
    retain within-player component dependence. Shared team draws enforce allocation
    bounds. Horizon-level rates are sampled once per draw, retaining persistent risk.
    """
    rng = np.random.default_rng(seed)
    ct = component_targets(cal)
    team_cal = cal.with_row_index("idx").unique(["year", "origin", "team"])["idx"].to_numpy()
    donor = rng.integers(0, cal.height, (frame.height, n))
    shares = {}
    for c in ["carry_share", "target_share"]:
        err = ct[c] - cal_components[c]
        shares[c] = normalize_shares(frame, np.clip(components[c][:, None] + err[donor], 0, 1))
    # Efficiency residuals are sampled only when a denominator was actually observed.
    sampled_eff = {}
    for label, num, den in [
        ("ypc", "rushing_yards", "carries"),
        ("ypr", "receiving_yards", "receptions"),
        ("catch", "receptions", "targets"),
    ]:
        w = cal[f"y_{den}"].to_numpy()
        valid = np.flatnonzero(w > 0)
        if not len(valid):
            sampled_eff[label] = np.broadcast_to(eff[label][:, None], (frame.height, n))
            continue
        # Keep common donor when valid; otherwise draw from observed denominators.
        d = np.where(w[donor] > 0, donor, rng.choice(valid, donor.shape))
        err = divide(cal[f"y_{num}"].to_numpy(), w) - cal_eff[label]
        sampled_eff[label] = eff[label][:, None] + err[d]
    sampled_eff["catch"] = np.clip(sampled_eff["catch"], 0, 1)
    carries, targets = (
        np.zeros((frame.height, n), dtype=int),
        np.zeros((frame.height, n), dtype=int),
    )
    allocations = []
    for g in frame.with_row_index("row").partition_by(["year", "origin", "team"]):
        idx = g["row"].to_numpy()
        i = idx[0]
        d = rng.choice(team_cal, n)
        # Residual bootstrap already contains team count variability; no second Poisson layer.
        plays = np.rint(
            np.maximum(
                0,
                components["team_plays"][i] + ct["team_plays"][d] - cal_components["team_plays"][d],
            )
            * frame["weeks"][int(i)]
        ).astype(int)
        if frame["scheduled_games"][int(i)] == 0:
            plays[:] = 0
        run = np.clip(
            components["run_fraction"][i]
            + ct["run_fraction"][d]
            - cal_components["run_fraction"][d],
            0,
            1,
        )
        tf = np.clip(
            components["target_fraction"][i]
            + ct["target_fraction"][d]
            - cal_components["target_fraction"][d],
            0,
            1,
        )
        rush = rng.binomial(plays, run)
        tgts = rng.binomial(plays - rush, tf)
        for field, totals, destination in [
            ("carry_share", rush, carries),
            ("target_share", tgts, targets),
        ]:
            # Sequential binomials are a vectorized multinomial with an 'other players' bucket.
            remaining, mass = totals.copy(), np.ones(n)
            for j in idx:
                p = np.clip(divide(shares[field][j], mass), 0, 1)
                destination[j] = rng.binomial(remaining, p)
                remaining -= destination[j]
                mass -= shares[field][j]
            allocations.append(bool(np.all(destination[idx].sum(axis=0) <= totals)))
    receptions = rng.binomial(targets, sampled_eff["catch"])
    touchdowns = rng.binomial(carries, eff["rush_td"][:, None]) + rng.binomial(
        receptions, eff["rec_td"][:, None]
    )
    # A yardage efficiency draw represents a period average, not independent per-play noise.
    rushing_yards = carries * sampled_eff["ypc"]
    receiving_yards = receptions * sampled_eff["ypr"]
    points = 0.1 * (rushing_yards + receiving_yards) + receptions + 6 * touchdowns
    assert all(allocations) and np.all(receptions <= targets)
    return (
        points,
        touchdowns,
        {
            "carries": carries,
            "targets": targets,
            "receptions": receptions,
            "rushing_yards": rushing_yards,
            "receiving_yards": receiving_yards,
        },
    )


def summarize(rows):
    return (
        rows.group_by("year", "horizon", "model", "target")
        .agg(
            pl.len().alias("n"),
            pl.col("crps").mean(),
            pl.col("negative_log_score").mean(),
            pl.col("squared_error").mean().sqrt().alias("rmse"),
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("covered50", "covered80", "covered90", "width80").mean(),
        )
        .sort("horizon", "target", "year", "crps")
    )


def run_fold(
    panel,
    features,
    year=2025,
    engine="lightgbm",
    rounds=(40, 100),
    draws=500,
    seed=20260925,
    ablations=True,
    market=False,
    progress=print,
):
    """Train≤Y−3, select Y−2, refit≤Y−2, calibrate Y−1, test Y. No test-dependent fitting."""
    if any(c.startswith("y_") or c == "target_records" for c in features):
        raise ValueError("Outcomes cannot be predictors")
    if draws < 20:
        raise ValueError("Use at least 20 simulation draws")
    train = panel.filter(pl.col("year") <= year - 3)
    valid = panel.filter(pl.col("year") == year - 2)
    refit = panel.filter(pl.col("year") <= year - 2)
    cal = panel.filter(pl.col("year") == year - 1)
    test = panel.filter(pl.col("year") == year)
    if min(train.height, valid.height, cal.height, test.height) < 20:
        raise ValueError("Each chronological split needs at least 20 rows")
    predictions, selection, samples, fits = [], [], {}, {}

    def record(name, target, sample, logscore=None):
        y = test[f"y_{target}"].to_numpy()
        scores = score_distribution(y, sample)
        f = (
            test.select(
                "player_id",
                "player_display_name",
                "position",
                "year",
                "origin",
                "team",
                "horizon",
                "age",
                "candidate_basis",
            )
            .with_row_index("sample_index")
            .with_columns(
                pl.lit(name).alias("model"),
                pl.lit(target).alias("target"),
                pl.Series("actual", y),
                *[pl.Series(k, v) for k, v in scores.items()],
                pl.Series(
                    "negative_log_score",
                    logscore if logscore is not None else np.full(len(y), np.nan),
                ),
            )
        )
        predictions.append(f)
        samples[f"{name}/{target}"] = sample.astype(np.float32)

    yc = cal["y_points"].to_numpy()
    record("prior_rate", "points", residual_draws(baseline(test), baseline(cal), yc, draws, seed))
    views = {"raw_all": features}
    if ablations:
        views["without_play_detail"] = [c for c in features if "pbp_" not in c and "ngs__" not in c]
        views["opportunity_only"] = [
            c
            for c in features
            if any(
                w in c
                for w in [
                    "carries",
                    "targets",
                    "snaps__",
                    "depth__",
                    "injuries__",
                    "redzone",
                    "inside5",
                    "team_",
                    "age",
                    "weeks",
                    "origin",
                ]
            )
            and not any(w in c for w in ["yards", "tds", "epa", "points", "wpa"])
        ]
    if market:
        if panel["horizon"][0] != "week":
            raise ValueError(
                "Target closing-line sensitivity is available only for next-week forecasts"
            )
        views["closing_line_sensitivity"] = [*features, "market_implied", "market_spread"]
    for name, cols in views.items():
        progress(f"{year} {panel['horizon'][0]}: {name}")
        trials = []
        for r in rounds:
            model = fit_tree(
                train,
                cols,
                train["y_points"].to_numpy() / train["weeks"].to_numpy(),
                r,
                engine,
                seed=seed,
            )
            mt = model.predict(train) * train["weeks"].to_numpy()
            mv = model.predict(valid) * valid["weeks"].to_numpy()
            s = residual_draws(mv, mt, train["y_points"].to_numpy(), min(draws, 300), seed)
            score = float(crps_ensemble(valid["y_points"].to_numpy(), s).mean())
            trials.append((score, r))
            selection.append(
                {"year": year, "model": name, "parameter": str(r), "validation_crps": score}
            )
        best = min(trials)[1]
        final = fit_tree(
            refit,
            cols,
            refit["y_points"].to_numpy() / refit["weeks"].to_numpy(),
            best,
            engine,
            seed=seed,
        )
        mc, mt = (
            final.predict(cal) * cal["weeks"].to_numpy(),
            final.predict(test) * test["weeks"].to_numpy(),
        )
        record(name, "points", residual_draws(mt, mc, yc, draws, seed))
        fits[name] = final
    # Count-law comparison holds features fixed, calibrates NB overdispersion on Y−1 only.
    cal_td_rate = (
        cal["history_rushing_tds"].to_numpy() + cal["history_receiving_tds"].to_numpy()
    ) / cal["history_weeks"].to_numpy()
    test_td_rate = (
        test["history_rushing_tds"].to_numpy() + test["history_receiving_tds"].to_numpy()
    ) / test["history_weeks"].to_numpy()
    mc = np.maximum(cal_td_rate, 1e-9) * cal["weeks"].to_numpy()
    mt = np.maximum(test_td_rate, 1e-9) * test["weeks"].to_numpy()
    for law, a in [("Poisson", 0), ("NB", estimate_dispersion(cal["y_td"].to_numpy(), mc))]:
        record(
            f"TD_prior_rate_{law}",
            "td",
            count_draws(mt, a, draws, np.random.default_rng(seed)),
            negative_log_score(test["y_td"].to_numpy(), mt, a),
        )
    for loss in ["squared_error", "poisson"]:
        progress(f"{year}: TD {loss}")
        trials = []
        for r in rounds:
            f = fit_tree(
                train,
                features,
                train["y_td"].to_numpy() / train["weeks"].to_numpy(),
                r,
                engine,
                loss=loss,
                seed=seed,
                weights=train["weeks"].to_numpy(),
            )
            mu = np.maximum(f.predict(valid), 1e-9) * valid["weeks"].to_numpy()
            score = float(negative_log_score(valid["y_td"].to_numpy(), mu).mean())
            trials.append((score, r))
            selection.append(
                {"year": year, "model": f"TD_{loss}", "parameter": str(r), "validation_nll": score}
            )
        f = fit_tree(
            refit,
            features,
            refit["y_td"].to_numpy() / refit["weeks"].to_numpy(),
            min(trials)[1],
            engine,
            loss=loss,
            seed=seed,
            weights=refit["weeks"].to_numpy(),
        )
        mc = np.maximum(f.predict(cal), 1e-9) * cal["weeks"].to_numpy()
        mt = np.maximum(f.predict(test), 1e-9) * test["weeks"].to_numpy()
        alpha = estimate_dispersion(cal["y_td"].to_numpy(), mc)
        for law, a in [("Poisson", 0), ("NB", alpha)]:
            name = f"TD_{loss}_{law}"
            record(
                name,
                "td",
                count_draws(mt, a, draws, np.random.default_rng(seed)),
                negative_log_score(test["y_td"].to_numpy(), mt, a),
            )
            selection.append(
                {"year": year, "model": name, "parameter": "dispersion", "calibration_alpha": a}
            )
    progress(f"{year}: volume / shares / efficiency simulations")
    # Component tree count is fixed before evaluation; shrinkage is selected on validation only.
    component_rounds = min(rounds)
    cf = fit_components(train, features, component_rounds, engine, seed)
    vt, tt = predict_components(valid, cf), predict_components(train, cf)
    final_cf = fit_components(refit, features, component_rounds, engine, seed)
    cc, tc = predict_components(cal, final_cf), predict_components(test, final_cf)
    fits["components"] = final_cf
    for name, ks, age in [
        ("decomposed_unshrunk", [0], False),
        ("decomposed_shrunk", [25, 100, 400], False),
        ("decomposed_age", [25, 100, 400], True),
    ]:
        ef = efficiency_fit(train, age)
        trials = []
        for k in ks:
            ve, te = efficiency_predict(valid, ef, k), efficiency_predict(train, ef, k)
            s, _, _ = simulate_decomposition(valid, vt, ve, train, tt, te, min(draws, 300), seed)
            score = float(crps_ensemble(valid["y_points"].to_numpy(), s).mean())
            trials.append((score, k))
            selection.append(
                {"year": year, "model": name, "parameter": str(k), "validation_crps": score}
            )
        k = min(trials)[1]
        final_eff = efficiency_fit(refit, age)
        ce, te = efficiency_predict(cal, final_eff, k), efficiency_predict(test, final_eff, k)
        point, td, stats = simulate_decomposition(test, tc, te, cal, cc, ce, draws, seed)
        record(name, "points", point)
        record(name, "td", td)  # No density estimate from samples: CRPS is valid for this mixture.
        for stat, sample in stats.items():
            record(name, stat, sample)
        fits[name] = {"efficiency": final_eff, "shrinkage": k}
    rows = pl.concat(predictions, how="diagonal_relaxed")
    return {
        "rows": rows,
        "summary": summarize(rows),
        "selection": pl.DataFrame(selection),
        "draws": samples,
        "fits": fits,
        "test": test,
        "chronology": {
            "train_through": year - 3,
            "selection": year - 2,
            "refit_through": year - 2,
            "calibration": year - 1,
            "test": year,
        },
        "features": features,
        "engine": engine,
        "seed": seed,
    }


def paired_crps_intervals(rows, reference="prior_rate", repeats=1000, seed=17):
    """Paired player-cluster bootstrap; repeated origins stay together within a player.

    Descriptive uncertainty, not independent-season evidence or a multiple-testing correction.
    """
    rng = np.random.default_rng(seed)
    base = rows.filter((pl.col("model") == reference) & (pl.col("target") == "points")).select(
        "player_id", "year", "origin", "horizon", pl.col("crps").alias("base_crps")
    )
    matched = rows.filter(pl.col("target") == "points").join(
        base, on=["player_id", "year", "origin", "horizon"]
    )
    result = []
    for key, f in matched.partition_by(["model", "horizon"], as_dict=True).items():
        clusters = (
            f.with_columns((pl.col("crps") - pl.col("base_crps")).alias("delta"))
            .group_by("player_id")
            .agg(pl.col("delta").sum(), pl.len().alias("n"))
        )
        a, n = clusters["delta"].to_numpy(), clusters["n"].to_numpy()
        idx = rng.integers(0, len(a), (repeats, len(a)))
        boot = a[idx].sum(axis=1) / n[idx].sum(axis=1)
        result.append(
            {
                "model": key[0],
                "horizon": key[1],
                "delta_crps": float(a.sum() / n.sum()),
                "low95": float(np.quantile(boot, 0.025)),
                "high95": float(np.quantile(boot, 0.975)),
                "players": len(a),
                "paired_rows": f.height,
            }
        )
    return pl.DataFrame(result).sort("horizon", "delta_crps")
