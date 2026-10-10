import polars as pl
import pytest
from experiments.future_player_lab.campaigns.rankings_browser import actual_rankings


def observations():
    return pl.DataFrame(
        {
            "player_id": ["a", "a", "b", "c", "d", "e", "a"],
            "position": ["RB", "RB", "RB", "QB", "WR", "TE", "RB"],
            "season": [2026] * 7,
            "season_type": ["REG"] * 7,
            "week": [1, 2, 1, 1, 1, 1, 5],
            "fantasy_points_ppr": [8.1, 2.2, 10.3, 20.0, 0.0, 99.0, 100.0],
        }
    )


def test_actuals_sum_only_completed_weeks_and_rank_ties_across_observed_population():
    rows = {r["player_id"]: r for r in actual_rankings(observations(), 2026, 4)}
    assert set(rows) == {"a", "b", "c", "d"}
    assert rows["a"]["actual_points"] == 10.3
    assert rows["a"]["actual_overall_rank"] == rows["b"]["actual_overall_rank"] == 2
    assert rows["a"]["actual_position_rank"] == rows["b"]["actual_position_rank"] == 1
    assert rows["d"]["actual_points"] == 0
    assert rows["d"]["actual_overall_rank"] == 4


def test_actuals_reject_duplicate_week():
    frame = observations()
    with pytest.raises(ValueError, match="Duplicate"):
        actual_rankings(pl.concat([frame, frame.head(1)]), 2026, 4)


def test_actuals_do_not_turn_unknown_points_into_zero():
    frame = observations().with_columns(
        pl.when(pl.col("player_id") == "a")
        .then(None)
        .otherwise(pl.col("fantasy_points_ppr"))
        .alias("fantasy_points_ppr")
    )
    with pytest.raises(ValueError, match="Missing/nonfinite"):
        actual_rankings(frame, 2026, 4)


def test_actuals_reject_wrong_season():
    with pytest.raises(ValueError, match="No actual observations"):
        actual_rankings(observations(), 2025, 4)


def test_te_actual_ranks_use_campaign_population():
    rows = actual_rankings(observations(), 2026, 4, positions=["TE"])
    assert len(rows) == 1
    assert rows[0]["player_id"] == "e"
    assert rows[0]["actual_points"] == 99
    assert rows[0]["actual_position_rank"] == rows[0]["actual_overall_rank"] == 1
