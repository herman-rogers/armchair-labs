import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from data import make_panel, role_state
from evaluation import calibrated_intervals, paired_summary
from methods import add_rates, fit_pool, matrix, mix_predictions, role_mixture
from safety import new_run, sha256, snapshot, verify_source_pins

from engine.metrics.nextgen import COUNTERS

LAB = Path(__file__).resolve().parents[1]
CONFIG = json.loads((LAB / "config.json").read_text())


def test_source_pins_fail_closed_and_unrelated_workspace_changes_are_separate(tmp_path):
    for name in CONFIG["source_sha256"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen")
    pins = {name: sha256(tmp_path / name) for name in CONFIG["source_sha256"]}
    verify_source_pins(tmp_path, pins)
    (tmp_path / "current.json").write_text("changed elsewhere")
    verify_source_pins(tmp_path, pins)
    (tmp_path / "src/engine/metrics/nextgen.py").write_text("changed dependency")
    with pytest.raises(ValueError, match="Pinned source changed"):
        verify_source_pins(tmp_path, pins)
    assert snapshot([tmp_path / "deleted"], tmp_path, missing_ok=True) == {"deleted": None}


def test_run_path_is_create_only_and_rejects_escape_and_symlink(tmp_path):
    with pytest.raises(ValueError):
        new_run(tmp_path, "../../data")
    new_run(tmp_path, "trial")
    with pytest.raises(FileExistsError):
        new_run(tmp_path, "trial")
    other = tmp_path / "nested"
    other.mkdir()
    (other / "runs").symlink_to(tmp_path / "runs", target_is_directory=True)
    with pytest.raises(ValueError):
        new_run(other, "escape")


def test_write_guard_blocks_external_writes_network_processes_and_links(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    external = tmp_path / "protected"
    external.write_text("original")
    (root / "escape").symlink_to(external)
    # Irreversible hooks are tested in their own process, not the pytest process.
    code = r"""
import os, socket, subprocess, sys
from pathlib import Path
from safety import install_write_guard
root, external = map(Path, sys.argv[1:])
install_write_guard(root)
(root / "valid").write_text("works")
actions = [lambda: external.write_text("bad"),
           lambda: (root / "escape").write_text("bad"),
           lambda: external.unlink(),
           lambda: os.replace(root / "valid", external),
           lambda: (root / "link").symlink_to(external),
           lambda: subprocess.run([sys.executable, "-c", "pass"]),
           lambda: socket.socket().connect(("127.0.0.1", 9))]
for action in actions:
    try: action()
    except PermissionError: pass
    else: raise AssertionError("Guard did not block action")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(root), str(external)],
        cwd=LAB,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert external.read_text() == "original"
    assert (root / "valid").read_text() == "works"


def test_pool_only_reads_historical_predictors_and_weights_exposure():
    rng = np.random.default_rng(3)
    train = []
    for i in range(400):
        n = float([1, 10, 100, 300][i % 4])
        train.append(
            dict(
                rate_receiving=8 + rng.normal(0, np.sqrt(1 + 30 / n)),
                exposure_receiving=n,
                y_targets=i,
            )
        )
    pool = fit_pool(train, "receiving")
    assert pool["fitted"] and pool["strength"] > 0
    changed = [{**r, "y_targets": -999999} for r in train]
    assert fit_pool(changed, "receiving") == pool
    rows = [dict(rate_receiving=14.0, exposure_receiving=n) for n in (1, 300)]
    adjusted = add_rates(rows, {"receiving": pool}, "WR")
    assert abs(adjusted[0]["pooled_receiving"] - pool["mean"]) < abs(
        adjusted[1]["pooled_receiving"] - pool["mean"]
    )
    missing = add_rates(
        [dict(rate_receiving=None, exposure_receiving=None)], {"receiving": pool}, "WR"
    )[0]
    assert missing["pooled_receiving"] is None


def test_future_labels_cannot_be_selected_as_features():
    with pytest.raises(ValueError):
        matrix([{"y_targets": 100}], ["y_targets"])


def test_role_mixture_integrates_conditional_totals_and_rejects_bad_probabilities():
    assert [role_state(n) for n in (0, 1, 7, 8, 17)] == [0, 1, 1, 2, 2]
    np.testing.assert_allclose(
        mix_predictions(np.array([[0.2, 0.3, 0.5]]), np.array([[10, 200, 500]])), [312]
    )
    with pytest.raises(ValueError):
        mix_predictions(np.array([[0.5, 0.7]]), np.array([[100, 300]]))


def test_heldout_role_and_workload_labels_do_not_change_prediction():
    train = [
        dict(
            x_age=i % 17,
            y_role_state=i % 3,
            y_attempts=float(20 + 100 * (i % 3)),
            forecast_season=2004 + i // 30,
        )
        for i in range(90)
    ]
    test = [dict(x_age=9, y_role_state=0, y_attempts=0, forecast_season=2025)]
    prediction, probabilities = role_mixture(train, test, "attempts", ["x_age"], CONFIG["booster"])
    changed = [{**test[0], "y_role_state": 2, "y_attempts": 99999}]
    after, probabilities_after = role_mixture(
        train, changed, "attempts", ["x_age"], CONFIG["booster"]
    )
    np.testing.assert_array_equal(prediction, after)
    np.testing.assert_array_equal(probabilities, probabilities_after)


def test_calibration_ignores_current_and_future_labels_and_preserves_warmup():
    row = dict(
        season=2025, prediction=100, scale=10, target="targets", known_available_games_cap=None
    )
    history = [
        dict(season=year, prediction=100.0, actual=100.0 + i, scale=10.0)
        for year in (2022, 2023, 2024)
        for i in range(40)
    ]
    before = calibrated_intervals(history, row, CONFIG)
    assert before["calibration_n"] == 120
    assert before["absolute_80_lower"] is not None
    for year in (2025, 2026):
        history.append(dict(season=year, prediction=1, actual=-1e12, scale=1))
    assert calibrated_intervals(history, row, CONFIG) == before
    assert calibrated_intervals(history[:80], row, CONFIG)["absolute_80_lower"] is None


def test_paired_scores_weight_seasons_equally_not_candidate_counts():
    rows = [dict(season=2024, actual=0, prediction=1, control_prediction=2)]
    rows += [dict(season=2025, actual=0, prediction=2, control_prediction=2)] * 100
    result = paired_summary(rows, {**CONFIG, "bootstrap_draws": 100})
    assert result["mae_gain"] == 0.5
    assert result["mse_gain"] == 1.5


def test_current_outcomes_do_not_change_panel_predictors_and_unknown_rates_stay_null():
    features = pl.DataFrame(
        [
            dict(
                player_id="p",
                player_display_name="Player",
                position="WR",
                player_population="returner",
                forecast_season=2025,
                forecast_cutoff_date="2025-08-01",
                source_season=2024,
                games=1,
                market_ecr=10.0,
                market_ecr_sd=1.0,
                market_overall_ecr=20.0,
                rich_s0_route_participation_mean=0.8,
                rich_s0_route_participation_coverage=0.5,
                rich_s0_targets_per_route_mean=None,
                rich_s0_targets_per_route_coverage=None,
            )
        ]
    )
    outcomes = features.select("player_id", "forecast_season", "forecast_cutoff_date").with_columns(
        pl.lit(True).alias("outcome_complete"), pl.lit(1).alias("actual_games")
    )
    records = [
        dict(player_id="p", season=year, season_type="REG", **{c: 3 for c in COUNTERS})
        for year in (2024, 2025)
    ]
    records[0]["targets"] = None
    before = make_panel(features, outcomes, pl.DataFrame(records))[0]
    changed = deepcopy(records)
    changed[1].update(receiving_yards=99999, attempts=100)
    after = make_panel(features, outcomes, pl.DataFrame(changed))[0]

    def predictors(r):
        return {
            k: v
            for k, v in r.items()
            if k.startswith(("x_", "r_", "m_", "p_", "rate_", "exposure_"))
        }

    assert predictors(before) == predictors(after)
    assert before["rate_receiving"] is None
    assert before["p_target_earning"] is None
    assert not before["proxy_observed"]
    incomplete = outcomes.with_columns(pl.lit(False).alias("outcome_complete"))
    pending = make_panel(features, incomplete, pl.DataFrame(records))[0]
    assert pending["y_targets"] is None and pending["y_role_state"] is None
