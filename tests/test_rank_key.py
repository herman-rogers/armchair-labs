"""Per-position v2 sort key and live application of fitted models."""

from __future__ import annotations

import polars as pl
import pytest

from patron.board.rank import (
    OVERALL_VOR,
    POSITION_RANK,
    RANK_KEY,
    RANK_VOR,
    apply_rank_key,
    resolve_rank_keys,
)
from patron.metrics.fit import FitConfig, RidgeModel, apply_fitted_models, models_for_season


def board() -> pl.DataFrame:
    rows = []
    for position, key_values in {
        "QB": [20.0, 18.0, 15.0],
        "RB": [16.0, 12.0, 10.0],
        "WR": [17.0 * 17, 14.0 * 17, 9.0 * 17],
        "TE": [None, None, None],
    }.items():
        for index, value in enumerate(key_values):
            rows.append(
                {
                    "player_id": f"{position}{index}",
                    "position": position,
                    "games": 17,
                    "override_delta": 1.0 if (position, index) == ("RB", 2) else 0.0,
                    "adj_proj_vor": 5.0 - index,
                    "proj_ppg": 15.0 - index,
                    "fitted_ppg": value if position == "QB" else None,
                    "fitted_season_points": value if position == "WR" else None,
                }
            )
    return pl.DataFrame(rows)


def config(league_config):
    return league_config.model_copy(
        update={
            "vor_baseline_rank": {"QB": 2, "RB": 2, "WR": 2, "TE": 2},
            "min_games_baseline": 8,
            "metrics": league_config.metrics.model_copy(
                update={
                    "projection_rank_key": {
                        "QB": "fitted_ppg",
                        "RB": "proj_ppg",
                        "WR": "fitted_season_points",
                        "TE": "fitted_season_points",
                    },
                    "projection_rank_fallback": "adj_proj_vor",
                    "projection_overall_key": "fitted_ppg",
                    "projection_season_games": 17,
                }
            ),
        }
    )


def test_each_position_ranks_on_its_own_key_and_falls_back_when_empty(league_config) -> None:
    cfg = config(league_config)
    assert resolve_rank_keys(board(), cfg) == {
        "QB": "fitted_ppg",
        "RB": "proj_ppg",
        "WR": "fitted_season_points",
        "TE": "adj_proj_vor",  # no fitted values for TE -> fallback
    }
    ranked = apply_rank_key(board(), cfg)
    rows = {row["player_id"]: row for row in ranked.iter_rows(named=True)}

    # Season-point key is converted to per-game before the replacement is subtracted:
    # WR replacement = second-best per-game equivalent (14.0), so WR0 = 17 - 14 = 3.
    assert rows["WR0"][RANK_KEY] == "fitted_season_points"
    assert rows["WR0"][RANK_VOR] == pytest.approx(3.0)
    assert rows["QB0"][RANK_VOR] == pytest.approx(20.0 - 18.0)
    # Manual override rides on top of the key's VOR.
    assert rows["RB2"][RANK_VOR] == pytest.approx((13.0 - 14.0) + 1.0)
    # Fallback keys that are already a VOR are used as-is.
    assert rows["TE0"][RANK_KEY] == "adj_proj_vor" and rows["TE0"][RANK_VOR] == 5.0
    assert ranked["rank"].to_list() == list(range(1, ranked.height + 1))
    # Within-position order follows the position's own key ...
    assert [rows[f"QB{i}"][POSITION_RANK] for i in range(3)] == [1, 2, 3]
    # ... while the overall order is on one scale: fitted_ppg where it exists (QB),
    # otherwise the position's rank VOR.
    assert rows["QB0"][OVERALL_VOR] == pytest.approx(20.0 - 18.0)
    assert rows["RB2"][OVERALL_VOR] == pytest.approx(rows["RB2"][RANK_VOR])
    assert ranked[OVERALL_VOR].is_sorted(descending=True)


def test_fitted_models_apply_to_a_live_board_from_their_record() -> None:
    model = RidgeModel(
        features=("ppg", "age_factor"),
        means=[10.0, 1.0],
        scales=[2.0, 0.1],
        coefficients=[1.5, 0.5],
        intercept=9.0,
        n_train=100,
    )
    records = [
        {"model": "fitted_ppg", "forecast_season": 2026, "position": "WR", **model.record()},
        {"model": "fitted_ppg", "forecast_season": 2025, "position": "WR", **model.record()},
    ]
    models = models_for_season(records, 2026)
    assert set(models) == {"fitted_ppg"} and set(models["fitted_ppg"]) == {"WR"}

    live = pl.DataFrame(
        {
            "position": ["WR", "RB"],
            "ppg": [12.0, 12.0],
            "age_factor": [0.9, 0.9],
            "expected_games": [16.0, 16.0],
        }
    )
    scored = apply_fitted_models(live, models, FitConfig(features=("ppg", "age_factor")))
    wr, rb = scored.to_dicts()
    assert wr["fitted_ppg"] == pytest.approx(9.0 + 1.5 * (2.0 / 2.0) + 0.5 * (-0.1 / 0.1))
    # No games model was supplied, so the product output is null rather than a guess.
    assert wr["fitted_season_points"] is None
    assert rb["fitted_ppg"] is None and rb["fitted_season_points"] is None


def test_market_only_players_are_rank_matched_and_tagged(league_config) -> None:
    """A rookie the market ranks lands where the board already places similar market ranks."""
    from patron.board.rank import OVERALL_VOR, RANK_SOURCE

    rows = []
    # Five rated WRs with market ranks 1..5 and VOR 10..2.
    for index in range(5):
        rows.append(
            {
                "player_id": f"wr{index}",
                "position": "WR",
                "games": 17,
                "override_delta": 0.0,
                "adj_proj_vor": 10.0 - 2 * index,
                "fitted_ppg": 20.0 - 2 * index,
                "fitted_season_points": (20.0 - 2 * index) * 17,
                "market_ecr": float(index + 1),
            }
        )
    # A rookie the market puts between WR2 and WR3, with no model values at all.
    rows.append(
        {
            "player_id": "rookie",
            "position": "WR",
            "games": 0,
            "override_delta": 0.0,
            "adj_proj_vor": None,
            "fitted_ppg": None,
            "fitted_season_points": None,
            "market_ecr": 2.5,
        }
    )
    # And one nobody ranks.
    rows.append({**rows[-1], "player_id": "ghost", "market_ecr": None})
    cfg = league_config.model_copy(
        update={
            "vor_baseline_rank": {"WR": 3},
            "min_games_baseline": 8,
            "metrics": league_config.metrics.model_copy(
                update={
                    "projection_rank_key": {"WR": "fitted_season_points"},
                    "projection_rank_fallback": "adj_proj_vor",
                    "projection_overall_key": "fitted_season_points",
                    "projection_season_games": 17,
                }
            ),
        }
    )
    ranked = apply_rank_key(pl.DataFrame(rows), cfg)
    by_id = {row["player_id"]: row for row in ranked.iter_rows(named=True)}

    rookie = by_id["rookie"]
    assert rookie[RANK_SOURCE] == "market" and rookie["v2_rank_key"] == "market"
    # Median of the three nearest rated players by market rank (WR2, WR3, WR1/WR4 tie
    # resolved by order): equals the value of a WR2-3 level player.
    assert by_id["wr2"][OVERALL_VOR] <= rookie[OVERALL_VOR] <= by_id["wr1"][OVERALL_VOR]
    assert rookie["v2_position_rank"] in (2, 3)
    assert rookie["rank"] < by_id["wr3"]["rank"]
    # The roster simulation inputs are borrowed on the same scale.
    assert by_id["wr2"]["fitted_ppg"] <= rookie["fitted_ppg"] <= by_id["wr1"]["fitted_ppg"]
    assert rookie["fitted_season_points"] is not None
    assert by_id["ghost"][RANK_SOURCE] is None and by_id["ghost"][OVERALL_VOR] is None
    assert by_id["ghost"]["rank"] == ranked.height  # unranked players sort last
    assert all(by_id[f"wr{i}"][RANK_SOURCE] == "model" for i in range(5))
