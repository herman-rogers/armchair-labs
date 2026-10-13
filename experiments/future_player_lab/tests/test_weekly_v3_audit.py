"""Acceptance audit catches inconsistent spread and separately recomputes scores."""

import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v3.audit import (
    PROBABILITIES,
    QCOLS,
    independent_scores,
    quantile_audit,
)


def test_quantile_audit_rejects_inconsistent_standard_deviation():
    # Q(p)=p in the interior, flat outside [.01,.99].
    q = PROBABILITIES[None, :]
    first = 0.5
    second = 0.01 * 0.01**2 + (0.99**3 - 0.01**3) / 3 + 0.01 * 0.99**2
    f = pd.DataFrame(q, columns=QCOLS)
    f["prediction"] = first
    f["std"] = np.sqrt(second - first**2)
    result = quantile_audit(f)
    assert result["std_max_error"] < 1e-10 and result["mean_max_error"] < 1e-10
    f["std"] += 1
    assert quantile_audit(f)["materially_inconsistent_std"] == 1


def test_independent_scores_dont_count_unobserved_labels_as_zero():
    f = pd.DataFrame(dict(prediction=[2.0, 3.0, 1000.0], target_points=[1.0, 5.0, np.nan]))
    score = independent_scores(f)
    assert score["n"] == 2
    assert score["mae"] == 1.5
    assert score["rmse"] == pytest.approx(np.sqrt(2.5))


def test_quantile_audit_detects_crossing():
    f = pd.DataFrame(PROBABILITIES[None, :], columns=QCOLS)
    f["q50"] = -10
    f["std"] = 1
    f["prediction"] = 0
    assert not quantile_audit(f)["ordered"]
