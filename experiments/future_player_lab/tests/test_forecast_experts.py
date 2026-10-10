"""Expert agreement uses known matches and comparable populations, not missing-rank imputation."""

import polars as pl
import pytest
from experiments.future_player_lab.forecasting.compare import compare_rankings


def model(names):
    return pl.DataFrame(
        [
            {
                "player_id": str(i),
                "player_display_name": name,
                "position": "RB",
                "team": "LA",
                "rank": i + 1,
                "predicted_ppr": 100.0 - i,
            }
            for i, name in enumerate(names)
        ]
    )


def expert(names):
    return {
        "year": "2026",
        "position_id": "RB",
        "scoring": "PPR",
        "ranking_type_name": "ros",
        "players": [
            {
                "player_id": i,
                "player_name": name,
                "player_position_id": "RB",
                "player_team_id": "LAR",
                "rank_ecr": i + 1,
            }
            for i, name in enumerate(names)
        ],
    }


def test_rank_gaps_sign_missing_and_team_aliases():
    frame, metrics = compare_rankings(
        model(["A Jr.", "B", "C"]), expert(["B", "A", "D"]), 2026, "RB"
    )
    assert frame["rank_gap"].to_list() == [1, -1, None]
    assert frame["expert_rank"].to_list() == [2, 1, None]
    assert not frame["expert_team_differs"].any()
    assert metrics["spearman"] == pytest.approx(-1)
    assert metrics["mean_absolute_matched_rank_gap"] == 1
    assert metrics["unmatched_experts"][0]["name"] == "D"


def test_metrics_rerank_common_population_while_table_keeps_original_ranks():
    frame, metrics = compare_rankings(model(["Extra", "A", "B"]), expert(["A", "B"]), 2026, "RB")
    assert frame["rank_gap"].to_list() == [None, -1, -1]
    assert frame["matched_rank_gap"].to_list() == [None, 0, 0]
    assert metrics["spearman"] == pytest.approx(1)
    assert metrics["mean_absolute_matched_rank_gap"] == 0


def test_ambiguous_name_is_not_silently_joined_and_team_changes_are_visible():
    experts = expert(["A", "B", "John Smith"])
    experts["players"][0]["player_team_id"] = "KC"
    frame, metrics = compare_rankings(
        model(["A", "B", "John Smith", "John Smith Jr."]), experts, 2026, "RB"
    )
    assert metrics["matched"] == 2
    assert frame["expert_match"].to_list()[-2:] == ["ambiguous_name", "ambiguous_name"]
    assert metrics["team_differences"] == ["A"]


@pytest.mark.parametrize(
    "field,value", [("year", "2025"), ("scoring", "HALF"), ("ranking_type_name", "weekly")]
)
def test_wrong_season_scoring_or_horizon_is_rejected(field, value):
    experts = expert(["A", "B"])
    experts[field] = value
    with pytest.raises(ValueError, match="must match"):
        compare_rankings(model(["A", "B"]), experts, 2026, "RB")


def test_duplicate_expert_ids_are_rejected():
    experts = expert(["A", "B"])
    experts["players"][1]["player_id"] = 0
    with pytest.raises(ValueError, match="Duplicate expert IDs"):
        compare_rankings(model(["A", "B"]), experts, 2026, "RB")
