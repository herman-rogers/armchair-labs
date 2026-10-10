"""Forecast cutoffs and per-target missingness for broad raw GPU inputs."""

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.gpu_lab import data


def test_history_views_ignore_future_observations_and_preserve_missing_channels():
    panel = pl.DataFrame({"player_id": ["a"], "year": [2024], "origin": [2]})
    observations = pl.DataFrame(
        {
            "player_id": ["a"] * 5,
            "season": [2022, 2023, 2024, 2024, 2024],
            "week": [1, 1, 1, 2, 3],
            "box__carries": [3.0, 7.0, 11.0, 15.0, 1000.0],
            "ngs__missing": [None] * 5,
        },
        schema_overrides={"ngs__missing": pl.Float64},
    )
    first = data.add_history_views(panel, observations)
    changed = observations.with_columns(
        pl.when((pl.col("season") == 2024) & (pl.col("week") == 3))
        .then(-1000.0)
        .otherwise(pl.col("box__carries"))
        .alias("box__carries")
    )
    assert first.equals(data.add_history_views(panel, changed))
    assert first["career__box__carries"][0] == 9
    assert first["prior2__box__carries"][0] == 5
    assert first["last_observed__box__carries"][0] == 15
    assert first["career__ngs__missing"][0] is None


def test_baseline_distinguishes_recorded_unknowns_from_absent_rows():
    panel = pl.DataFrame(
        {
            "player_id": ["unknown", "absent", "known"],
            "year": [2024] * 3,
            "origin": [2] * 3,
            "weeks": [16] * 3,
            "history_weeks": [2] * 3,
        }
    )
    box = pl.DataFrame(
        {
            "player_id": ["known", "unknown", "known"],
            "season": [2024] * 3,
            "week": [2, 1, 3],
            **{stat: [10.0, None if stat == "carries" else 2.0, 1000.0] for stat in data.STATS},
        }
    )
    baseline = data.prior_rate_baseline(panel, box)
    assert np.isnan(baseline[0, 0])
    np.testing.assert_array_equal(baseline[0, 1:], 16.0)
    np.testing.assert_array_equal(baseline[1], 0.0)
    np.testing.assert_array_equal(baseline[2], 80.0)


def test_feature_contract_rejects_future_fields_and_never_selects_by_scores():
    for name in ("y_points", "actual_carries", "market_spread", "target_records", "player_id"):
        with pytest.raises(ValueError, match="Outcomes"):
            data.validate_features(["prior__box__carries", name])
    data.validate_features(["prior__box__carries", "career__ngs__rushing_yards"])
    assert data.source_family("career__ngs__rushing_yards") == "ngs"
    assert data.source_family("last_observed__pbp_rush__epa") == "plays"


def test_delayed_ftn_participation_is_available_only_in_prior_seasons():
    panel = pl.DataFrame({"player_id": ["a"], "year": [2024], "origin": [2]})
    observations = pl.DataFrame(
        {
            "player_id": ["a", "a", "a"],
            "season": [2023, 2024, 2024],
            "week": [18, 1, 2],
            "pbp_rush__participation_box_count": [6.0, 8.0, 9.0],
            "box__carries": [1.0, 2.0, 3.0],
        }
    )
    result = data.add_history_views(panel, observations)
    assert result["career__pbp_rush__participation_box_count"][0] == 6
    assert result["last_observed__pbp_rush__participation_box_count"][0] is None
    assert result["career__box__carries"][0] == 2


def test_drafted_team_names_normalize_and_unknown_schedule_is_not_a_bye():
    from experiments.future_player_lab.notebooks import raw_distribution_data as raw

    candidates = pl.DataFrame(
        {"team": ["Denver Broncos", "San Diego Chargers", "Washington Redskins", "Unresolved"]}
    ).with_columns(pl.col("team").replace(raw.TEAM_CODES))
    games = pl.DataFrame({"home_team": ["DEN", "WAS"], "away_team": ["KC", "LAC"]})
    result = raw.attach_scheduled_games(candidates, games, games.head(1))
    assert result["team"].to_list() == ["DEN", "LAC", "WAS", "Unresolved"]
    assert result["scheduled_games"].to_list() == [1, 0, 0, None]
