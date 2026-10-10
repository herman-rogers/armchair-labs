"""Chronology, opportunity denominators and persisted component arithmetic."""

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.campaigns.qb_features import augment_qb
from experiments.future_player_lab.campaigns.qb_policy import PredictionPolicy
from experiments.future_player_lab.campaigns.qb_revision import select_policy


def losses():
    return [
        {
            "year": year,
            "target": "attempts",
            "recipe": recipe,
            "mae": loss,
            "mse": loss,
            "starter_n": 20,
            "starter_mse": loss,
        }
        for year in (2018, 2019, 2020)
        for recipe, loss in (("control", 100.0), ("features", 98.0))
    ]


def test_policy_never_uses_current_or_future_losses():
    earlier = losses()
    future = [{**r, "year": 2021, "mae": -9999.0, "mse": -9999.0} for r in earlier]
    assert select_policy(earlier, "attempts", 2021)[0] == "features"
    assert select_policy(earlier + future, "attempts", 2021) == select_policy(
        earlier, "attempts", 2021
    )


def test_starter_guard_can_reject_broad_improvement():
    rows = losses()
    for r in rows:
        if r["recipe"] == "features":
            r["starter_mse"] = 101.0
    assert select_policy(rows, "attempts", 2021)[0] == "control"


def test_incomplete_selection_history_cannot_borrow_current_year():
    with pytest.raises(ValueError, match="Incomplete"):
        select_policy(losses()[1:], "attempts", 2021)


def fixtures():
    panel = pl.DataFrame([{"year": 2021, "origin": 2, "player_id": "a", "team": "NYG"}])
    box = pl.DataFrame(
        [
            {
                "season": 2020,
                "week": 1,
                "player_id": "a",
                "team": "NYG",
                "attempts": 100.0,
                "passing_yards": 600.0,
            },
            {
                "season": 2021,
                "week": 1,
                "player_id": "a",
                "team": "NYG",
                "attempts": 20.0,
                "passing_yards": 160.0,
            },
            {
                "season": 2021,
                "week": 2,
                "player_id": "a",
                "team": "NYG",
                "attempts": 0.0,
                "passing_yards": 0.0,
            },
            {
                "season": 2021,
                "week": 2,
                "player_id": "b",
                "team": "NYG",
                "attempts": 30.0,
                "passing_yards": 210.0,
            },
        ]
    )
    schedule = pl.DataFrame(
        [{"season": 2021, "week": w, "home_team": "NYG", "away_team": "DAL"} for w in (1, 2, 3)]
    )
    return panel, {"box": box, "schedule": schedule}


def test_features_are_unchanged_by_future_box_rows_and_panel_outcomes():
    panel, tables = fixtures()
    before = augment_qb(panel, tables)
    future = (
        tables["box"]
        .head(1)
        .with_columns(
            pl.lit(2021).alias("season"),
            pl.lit(3).alias("week"),
            pl.lit(999999.0).alias("passing_yards"),
        )
    )
    changed = dict(tables, box=pl.concat([tables["box"], future], how="vertical_relaxed"))
    after = augment_qb(panel.with_columns(pl.lit(999999.0).alias("y_passing_yards")), changed)
    assert before.equals(after.select(before.columns))
    row = before.row(0, named=True)
    assert row["qb__current_team_attempt_share"] == 0.4
    assert row["qb__competing_attempts_per_game"] == 15.0
    assert row["qb__last3_attempts"] == 10.0  # Explicitly observed zero is usable.
    assert row["qb__last_attempts"] == 0.0
    assert row["qb__current_attempt_coverage"] == 1.0
    assert row["qb__last8_ypa_p400"] == (760 + 400 * 6) / 520


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


def test_component_wrapper_preserves_per_week_exposure_contract():
    wrapper = PredictionPolicy(
        [{"model": Constant(20), "indices": [0]}],
        count=True,
        rate=Constant(8),
        rate_indices=[0],
        prior_index=1,
    )
    per_week = wrapper.predict(np.array([[1, 6], [2, 8]]))
    np.testing.assert_array_equal(per_week, [140, 160])
    np.testing.assert_array_equal(per_week * 16, [2240, 2560])


def test_missing_current_player_is_unknown_opportunity_in_past_and_current():
    panel, tables = fixtures()
    for year in (2021, 2026):
        candidate = panel.with_columns(
            pl.lit("missing").alias("player_id"), pl.lit(year).alias("year")
        )
        row = augment_qb(candidate, tables).row(0, named=True)
        assert row["qb__current_observation_known"] == 0
        for feature in (
            "current_team_attempt_share",
            "last3_attempts",
            "last_attempts",
            "competing_attempts_per_game",
            "current_attempts_per_team_game",
            "current_log_attempts",
            "last3_primary_share",
        ):
            assert row[f"qb__{feature}"] is None
        assert row["qb__career_ypa_h2_p400"] == pytest.approx(row["qb__league_ypa"])


@pytest.mark.parametrize("missing_kind", ["missing_row", "unknown_attempts"])
def test_partial_window_missing_or_null_is_not_observed_zero(missing_kind):
    panel, tables = fixtures()
    affected = (pl.col("player_id") == "a") & (pl.col("week") == 2)
    if missing_kind == "missing_row":
        box = tables["box"].filter(~affected)
    else:
        box = tables["box"].with_columns(
            pl.when(affected).then(None).otherwise(pl.col("attempts")).alias("attempts")
        )
    row = augment_qb(panel, dict(tables, box=box)).row(0, named=True)
    assert row["qb__current_observation_known"] == 1
    assert row["qb__current_attempt_coverage"] == 0.5
    for feature in (
        "current_log_attempts",
        "current_team_attempt_share",
        "last_attempts",
        "last3_attempts",
        "last3_primary_share",
        "competing_attempts_per_game",
        "current_attempts_per_team_game",
    ):
        assert row[f"qb__{feature}"] is None
