"""Display comparisons preserve rank scope and consensus publication cutoffs."""

from datetime import date

import polars as pl

from patron.data.ranking_views import market_references, with_ranks


def test_ranks_survive_search_and_ties_do_not_invent_order():
    ranked = with_ranks(
        pl.DataFrame(
            {
                "player_id": ["a", "b", "c", "d", "e"],
                "position": ["QB", "WR", "WR", "WR", "WR"],
                "population": ["returner", "returner", "rookie", "rookie", "rookie"],
                "prediction": [300.0, 200.0, 100.0, 100.0, None],
            }
        )
    )
    assert ranked["overall_rank"].to_list() == [1, 2, 3, 3, None]
    assert ranked.filter(pl.col("player_id") == "c")["position_rank"].item() == 2
    assert ranked.filter(pl.col("population") == "rookie")["population_rank"].to_list() == [
        1,
        1,
        None,
    ]


def test_ecr_is_original_dated_value_never_after_cutoff_or_wrong_position():
    frame = pl.DataFrame(
        {
            "player_id": ["a", "b", "c", "d"],
            "forecast_season": [2026] * 4,
            "forecast_cutoff_date": [date(2026, 8, 28)] * 4,
            "position": ["WR"] * 4,
            "market_position": ["WR", "WR", "RB", "WR"],
            "market_overall_ecr": [22.45] * 4,
            "market_ecr": [4.21] * 4,
            "market_overall_snapshot": ["2026-08-28", "2026-08-29", None, "2026-08-28"],
            "market_snapshot": ["2026-08-28", "2026-08-29", "2026-08-28", None],
        }
    )
    result = market_references(frame, 2026)
    assert result["ecr_overall"].to_list() == [22.45, None, None, 22.45]
    assert result["ecr_position"].to_list() == [4.21, None, None, None]
    assert result["ecr_position_date"].to_list() == ["2026-08-28", None, None, None]
    suspended = market_references(
        frame, 2026, [{"status": "open", "dependencies": ["preseason_features.market_ecr"]}]
    )
    assert suspended["ecr_position"].null_count() == 4


def test_search_and_export_use_same_rank_and_suspended_baseline_disappears(tmp_path, monkeypatch):
    from patron.api import nextgen_routes as routes
    from patron.data.ranking_views import baseline_ranks
    from patron.data.releases import write_json

    entry = dict(
        id="forecast:baseline:season_points",
        kind="forecast",
        model="baseline",
        target="season_points",
        horizon="preseason_season",
        validity="verified",
        serving="baseline",
        allowed_uses=["forecast"],
        positions=["WR"],
        populations=["returner"],
        dependencies=["nfl_player_weeks.league_points"],
        reason="Reference only",
    )
    write_json(tmp_path / "registry.json", [entry])
    write_json(tmp_path / "incidents.json", [])
    pl.DataFrame(
        {
            "player_id": ["a", "b"],
            "player_display_name": ["Alpha", "Beta"],
            "position": ["WR", "WR"],
            "population": ["returner", "returner"],
            "target": ["season_points"] * 2,
            "model": ["baseline"] * 2,
            "prediction": [100.0, 90.0],
        }
    ).write_parquet(tmp_path / "forecasts.parquet")
    monkeypatch.setattr(routes, "release", lambda: (tmp_path, {"version": "test"}))
    result = routes.forecasts(search="Beta", limit=1, offset=0)
    assert result["forecasts"][0]["overall_rank"] == 2
    assert baseline_ranks(tmp_path)["overall_rank"].to_list() == [1, 2]
    write_json(
        tmp_path / "incidents.json",
        [
            {
                "status": "open",
                "reason": "defect",
                "dependencies": ["nfl_player_weeks.league_points"],
            }
        ],
    )
    assert baseline_ranks(tmp_path).is_empty()
