"""Numerical scores, chronology, source contracts and simulation constraints."""

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.notebooks import raw_distribution_data as data
from experiments.future_player_lab.notebooks import raw_distribution_models as model
from scipy.stats import norm


def test_crps_matches_pairwise_definition_and_degenerate_absolute_error():
    draws = np.array([[0.0, 1.0, 2.0, 5.0], [-1.0, 0.0, 0.0, 8.0]])
    y = np.array([1.5, 0.0])
    pairs = np.abs(draws[:, :, None] - draws[:, None, :]).sum(axis=(1, 2))
    first = np.abs(draws - y[:, None]).mean(axis=1)
    np.testing.assert_allclose(model.crps_ensemble(y, draws), first - pairs / 24)
    np.testing.assert_allclose(model.crps_ensemble(y, draws, False), first - pairs / 32)
    np.testing.assert_allclose(model.crps_ensemble(y, np.ones((2, 20))), np.abs(y - 1))


def test_monte_carlo_crps_matches_closed_form_normal():
    rng = np.random.default_rng(91)
    y = np.array([-2.0, 0.0, 1.0])
    draws = rng.normal(size=(3, 100_000))
    exact = y * (2 * norm.cdf(y) - 1) + 2 * norm.pdf(y) - 1 / np.sqrt(np.pi)
    np.testing.assert_allclose(model.crps_ensemble(y, draws), exact, atol=0.015)


def test_count_log_scores_normalized_and_nb_poisson_limit():
    np.testing.assert_allclose(model.negative_log_score([1], [2]), [2 - np.log(2)])
    np.testing.assert_allclose(model.negative_log_score([0], [2], 0.5), [-np.log(0.25)])
    for alpha in [0, 0.2, 2]:
        y = np.arange(2000)
        probability = np.exp(-model.negative_log_score(y, np.full(len(y), 3.0), alpha))
        assert abs(probability.sum() - 1) < 1e-10
    np.testing.assert_allclose(
        model.negative_log_score([0, 2], [1, 3], 1e-10), model.negative_log_score([0, 2], [1, 3])
    )
    with pytest.raises(ValueError, match="integer"):
        model.negative_log_score([1.5], [1])


def synthetic_panel():
    rng = np.random.default_rng(77)
    rows = []
    for year in range(2017, 2026):
        for i in range(48):
            carries = rng.poisson(8)
            targets = rng.poisson(3)
            rec = rng.binomial(targets, 0.7)
            rt, ct = rng.binomial(carries, 0.04), rng.binomial(rec, 0.06)
            ry, cy = carries * rng.normal(4, 1), rec * rng.normal(8, 2)
            r = {
                "player_id": str(i),
                "player_display_name": f"Player {i}",
                "year": year,
                "team": str(i // 12),
                "position": "RB",
                "horizon": "week",
                "origin": 2,
                "weeks": 1,
                "scheduled_games": 1,
                "age": 22 + i % 12,
                "candidate_basis": "past NFL observation",
                "history_weeks": 2,
                "history_games": 2,
                "team_prior__carries": 22 + i // 12,
                "y_team_plays": 65,
                "y_team_carries": 30,
                "y_team_targets": 32,
                "y_carries": carries,
                "y_targets": targets,
                "y_receptions": rec,
                "y_rushing_yards": ry,
                "y_receiving_yards": cy,
                "y_rushing_tds": rt,
                "y_receiving_tds": ct,
                "y_td": rt + ct,
                "y_points": 0.1 * (ry + cy) + rec + 6 * (rt + ct),
            }
            for stat in data.STATS:
                r[f"history_{stat}"] = float(rng.poisson(10))
            rows.append(r)
    panel = pl.DataFrame(rows)
    features = [c for c in data.numeric(panel) if not c.startswith("y_") and c != "year"]
    return panel, features


@pytest.mark.parametrize("engine", ["hist", "lightgbm", "xgboost"])
def test_each_exposed_engine_can_fit_count_forecasts(engine):
    panel, features = synthetic_panel()
    try:
        fitted = model.fit_tree(
            panel, features, panel["y_td"].to_numpy(), rounds=2, engine=engine, loss="poisson"
        )
    except ModuleNotFoundError:
        pytest.skip(f"Optional {engine} environment is not installed")
    pred = fitted.predict(panel)
    assert np.isfinite(pred).all() and (pred > 0).all()


def test_future_labels_do_not_change_fits_selection_or_draws():
    p, features = synthetic_panel()
    kwargs = dict(
        year=2025, engine="hist", rounds=(2,), draws=25, ablations=False, progress=lambda s: None
    )
    first = model.run_fold(p, features, **kwargs)
    changed = p.with_columns(
        [
            pl.when(pl.col("year") == 2025).then(pl.col(c) * 3 + 1).otherwise(pl.col(c)).alias(c)
            for c in p.columns
            if c.startswith("y_")
        ]
    )
    second = model.run_fold(changed, features, **kwargs)
    assert first["selection"].equals(second["selection"])
    for key in first["draws"]:
        np.testing.assert_array_equal(first["draws"][key], second["draws"][key])
    assert not first["rows"]["actual"].equals(second["rows"]["actual"])
    with pytest.raises(ValueError, match="Outcomes"):
        model.run_fold(p, features + ["y_points"], **kwargs)


def test_simulation_conserves_volume_and_receptions():
    panel, _ = synthetic_panel()
    f = panel.filter(pl.col("year") == 2025)
    components = {
        "team_plays": np.full(f.height, 10.0),
        "run_fraction": np.full(f.height, 0.5),
        "target_fraction": np.full(f.height, 0.8),
        "carry_share": np.full(f.height, 0.7),
        "target_share": np.full(f.height, 0.7),
    }
    # Set calibration component labels equal to predictions to isolate allocation conservation.
    cal = f.with_columns(
        pl.lit(10).alias("y_team_plays"),
        pl.lit(5).alias("y_team_carries"),
        pl.lit(4).alias("y_team_targets"),
        pl.lit(0).alias("y_carries"),
        pl.lit(0).alias("y_targets"),
    )
    cc = model.component_targets(cal)
    eff = {
        k: np.full(f.height, v)
        for k, v in {"ypc": 4.0, "ypr": 8.0, "catch": 0.7, "rush_td": 0.05, "rec_td": 0.07}.items()
    }
    points, td, stats = model.simulate_decomposition(f, components, eff, cal, cc, eff, 200, 1)
    assert np.isfinite(points).all()
    assert np.all(stats["receptions"] <= stats["targets"])
    assert np.all(td <= stats["carries"] + stats["receptions"])
    for g in f.with_row_index("row").partition_by("team"):
        idx = g["row"].to_numpy()
        assert np.all(stats["carries"][idx].sum(axis=0) + stats["targets"][idx].sum(axis=0) <= 10)
    bye = f.with_columns(pl.lit(0).alias("scheduled_games"))
    _, td, stats = model.simulate_decomposition(bye, components, eff, cal, cc, eff, 100, 1)
    assert not td.any() and not stats["carries"].any() and not stats["targets"].any()


def test_contradictory_source_rows_fail_instead_of_silently_deduplicating():
    f = pl.DataFrame({"id": ["a", "a"], "x": [1.0, 1.0]})
    assert data.unique_observations(f, ["id"]).height == 1
    with pytest.raises(ValueError, match="Conflicting"):
        data.unique_observations(f.with_columns(pl.Series("x", [1.0, 2.0])), ["id"])


def test_residual_donors_and_scores_handle_zero_mass():
    draws = model.residual_draws(np.zeros(3), np.zeros(20), np.zeros(20), 100, 5)
    scored = model.score_distribution(np.zeros(3), draws)
    assert not scored["crps"].any()
    assert np.all(scored["covered80"] == 1)
    assert np.all((scored["pit"] > 0) & (scored["pit"] < 1))


@pytest.mark.skipif(not data.SNAPSHOT.exists(), reason="Local raw snapshot is not installed")
def test_real_raw_cutoffs_unknowns_and_absent_production(monkeypatch):
    raw = data.load_raw()
    base, features, _ = data.make_panel(first_year=2025)
    future = (pl.col("season") == 2025) & (pl.col("week") > 2)
    changed = dict(raw)
    changed["box"] = raw["box"].with_columns(
        [pl.when(future).then(pl.col(s) * 3 + 1).otherwise(pl.col(s)).alias(s) for s in data.STATS]
    )
    changed["player"] = raw["player"].with_columns(
        [
            pl.when(future).then(pl.col(c) * 3 + 1).otherwise(pl.col(c)).alias(c)
            for c in raw["player"].columns
            if c.startswith("box__")
        ]
    )
    monkeypatch.setattr(data, "load_raw", lambda: changed)
    other, names, _ = data.make_panel(first_year=2025)
    assert features == names
    assert base.select("player_id", *features).equals(other.select("player_id", *features))
    assert not base["y_points"].equals(other["y_points"])
    known = base.filter(pl.col("y_carries") > 0)["player_id"][0]
    changed = dict(raw)
    changed["box"] = raw["box"].with_columns(
        pl.when(future & (pl.col("week") == 3) & (pl.col("player_id") == known))
        .then(None)
        .otherwise(pl.col("carries"))
        .alias("carries")
    )
    unknown, _, audit = data.make_panel(first_year=2025)
    assert not unknown.filter(pl.col("player_id") == known).height
    assert audit["excluded_unknown"].sum() == 1
    changed = dict(raw)
    changed["box"] = raw["box"].filter(~(future & (pl.col("player_id") == known)))
    absent, _, _ = data.make_panel(first_year=2025)
    assert absent.filter(pl.col("player_id") == known)["y_points"][0] == 0
