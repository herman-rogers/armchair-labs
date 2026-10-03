"""Reporting checks use fixed numbers; no models are trained."""

import numpy as np
import pytest
from experiments.future_player_lab.notebooks.rb_accuracy import accuracy_context


def test_context_matches_hand_calculated_errors_and_scale_invariance():
    r, _ = accuracy_context([0, 100, 200], [10, 80, 210], [0, 50, 150], tolerance=10)
    assert r["mae"] == pytest.approx(40 / 3)
    assert r["wape_pct"] == pytest.approx(100 * 40 / 300)
    assert r["r2"] == pytest.approx(1 - 600 / 20000)
    assert r["mae_improvement_pct"] == pytest.approx(60)
    assert r["within_tolerance_pct"] == pytest.approx(200 / 3)
    larger, _ = accuracy_context([0, 1000, 2000], [100, 800, 2100], [0, 500, 1500], tolerance=100)
    for key in ["wape_pct", "r2", "mae_improvement_pct", "within_tolerance_pct"]:
        assert larger[key] == pytest.approx(r[key])


def test_zero_production_and_perfect_baseline_are_not_fake_accuracy():
    r, _ = accuracy_context([0, 0], [5, 10], [0, 0])
    assert r["wape_pct"] is None
    assert r["r2"] is None
    assert r["mae_improvement_pct"] is None
    assert r["zero_actual_pct"] == 100
    empty, _ = accuracy_context([0], [5], [0], positive_only=True)
    assert empty["players"] == 0
    assert empty["mae"] is None


def test_missing_baseline_uses_common_players_and_negative_skill_is_retained():
    r, mask = accuracy_context([1, 2, 3, 0], [10, 20, 30, 0], [1, np.nan, 2, 0], positive_only=True)
    assert mask.tolist() == [True, False, True, False]
    assert r["players"] == 2
    assert r["r2"] < 0
    assert r["wape_pct"] > 100
    assert r["mae_improvement_pct"] < 0


@pytest.mark.parametrize("with_availability", [False, True])
def test_prediction_detail_cell_exports_csv_with_injury_evidence(with_availability):
    import ast
    import io
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock

    import polars as pl

    path = Path(__file__).resolve().parents[1] / "notebooks/rb_stats.py"
    cell = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and {"accuracy_context", "results"} <= {arg.arg for arg in node.args.args}
    )
    cell.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), str(path), "exec"), namespace)
    forecasts = pl.DataFrame(
        {
            "name": ["A", "B", "C"],
            "population": ["veteran"] * 3,
            "actual": [0.0, 100.0, 200.0],
            "prediction": [10.0, 80.0, 210.0],
            "persistence_prediction": [0.0, 50.0, 150.0],
        }
    )
    if with_availability:
        forecasts = forecasts.with_columns(
            pl.Series("availability_evidence_ids", [["e1", "e2"], [], None]),
            pl.Series("availability_conflicts", [[], ['Review, "dated" evidence'], None]),
        )
    mo = Mock()
    namespace["_"](
        accuracy_cohort=SimpleNamespace(value="All evaluated RBs"),
        accuracy_context=accuracy_context,
        accuracy_tolerance=SimpleNamespace(value=25),
        data=SimpleNamespace(target_label=lambda _: "Carries", target_unit=lambda _: "carries"),
        inspect_stat=SimpleNamespace(value="carries"),
        mo=mo,
        pl=pl,
        px=Mock(),
        results={"carries": {"forecasts": forecasts}},
    )
    mo.download.assert_called_once()
    exported = pl.read_csv(io.StringIO(mo.download.call_args.args[0]))
    assert exported.select(forecasts.columns[:5]).equals(forecasts.select(forecasts.columns[:5]))
    assert exported["Absolute error"].to_list() == [10.0, 20.0, 10.0]
    if with_availability:
        assert exported["availability_evidence_ids"].to_list() == ["e1; e2", "", None]
        assert exported["availability_conflicts"].to_list() == [
            "",
            'Review, "dated" evidence',
            None,
        ]
        assert forecasts["availability_evidence_ids"].dtype == pl.List(pl.String)
