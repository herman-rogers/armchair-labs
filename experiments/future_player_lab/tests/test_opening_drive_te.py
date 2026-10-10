"""TE protocol must preserve chronological v2 fitting and explicit role relevance."""

import copy
import json

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.campaigns import data, run, tight_end
from experiments.future_player_lab.campaigns.te_features import relevant_features


def test_te_contract_and_ppr_baseline_keep_hybrid_rushing():
    config = json.loads(tight_end.CONFIG.read_text())
    tight_end.validate_config(config)
    assert config["targets"]["TE"] == [*data.STATS["WR"], "points"]
    panel = pl.DataFrame(
        {
            **{f"history_{s}": [0.0] for s in data.STATS["TE"]},
            "history_receptions": [5.0],
            "history_receiving_yards": [60.0],
            "history_receiving_tds": [1.0],
            "history_rushing_yards": [10.0],
            "history_rushing_tds": [1.0],
            "history_weeks": [2],
            "weeks": [16],
        }
    )
    assert run.baseline_values(panel, "TE", "points")[0] == 24 * 8
    for key in ("recipes", "selection", "validation_years", "evaluation_years"):
        broken = copy.deepcopy(config)
        broken[key] = None
        with pytest.raises(ValueError, match="preserve v2"):
            tight_end.validate_config(broken)


def test_te_mask_retains_receiver_role_and_team_passing_without_player_passing():
    keep = [
        "prior__box__targets",
        "recent8__box__receiving_yards",
        "history_carries",
        "prior__box__rushing_tds",
        "career__ngs__avg_separation",
        "current__ngs__avg_yac",
        "prior__snaps__offense_pct",
        "prior__pbp_target__redzone_count",
        "prior__pbp_target__inside5_count",
        "prior__pbp_target__participation_ngs_air_yards",
        "team_prior__passing_yards",
        "team_current__attempts",
        "college__receivingYards",
        "draft_event__overall_pick",
        "age",
        "availability__status_RET",
    ]
    drop = [
        "prior__box__passing_yards",
        "career__ngs__passer_rating",
        "prior__box__def_tackles_solo",
        "prior__box__fg_made",
        "college__passingYards",
        "prior__snaps__defense_pct",
        "prior__pbp_target__punt_attempt",
        "prior__pbp_pass__passing_yards",
        "team_prior__fg_made",
    ]
    assert relevant_features(keep + drop) == keep


def test_saved_te_forecast_has_direct_and_component_points():
    class Rate:
        def predict(self, x):
            return np.ones(len(x))

    model = dict(
        position="TE",
        horizon="season",
        origin_week=0,
        training_through_season=2025,
        feature_names=["age"],
        models={s: [dict(indices=[0], model=Rate())] for s in [*data.STATS["TE"], "points"]},
    )
    panel = pl.DataFrame(
        dict(
            year=[2026],
            origin=[0],
            weeks=[18],
            position=["TE"],
            horizon=["season"],
            age=[25.0],
        )
    )
    forecasts = run.predict(model, panel)
    assert forecasts["points"][0] == 18
    assert forecasts["component_points"][0] == pytest.approx(237.6)
    with pytest.raises(ValueError, match="Position"):
        run.predict(model, panel.with_columns(pl.lit("WR").alias("position")))
