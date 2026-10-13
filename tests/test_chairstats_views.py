"""Manifest-driven campaign views: rankings across models, per-player lookup, targets."""

import json

import pytest

from engine.chairstats import CampaignStore, publish_campaign, views
from engine.chairstats.store import CampaignNotFound


def forecast_rows(position, season, origin, names):
    return [
        {
            "player_id": f"{position.lower()}{i}",
            "player_name": name,
            "team": "SF",
            "position": position,
            "season": season,
            "origin_week": origin,
            "rank": i + 1,
            "scheduled_games": 15,
            "predictions": {"yards": 1000.0 - 100 * i, "points": 250.0 - 40 * i},
            "points_per_scheduled_game": (250.0 - 40 * i) / 15,
            "availability": {"status": "ACT", "as_of": "2026-09-22T12:00:00+00:00"},
        }
        for i, name in enumerate(names)
    ]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def store(tmp_path):
    work = tmp_path / "work"
    metrics = write(work / "metrics.json", {"rmse": 1.0})
    models = []
    for position, names in (("RB", ["Back One", "Back Two"]), ("WR", ["Wide One"])):
        model_id = f"{position.lower()}_remaining"
        models.append(
            {
                "model_id": model_id,
                "position": position,
                "horizon": "remaining",
                "origin_week": 2,
                "targets": ["yards", "points"],
                "training_through_season": 2025,
                "metrics": metrics,
                "forecasts": [
                    {
                        "season": 2026,
                        "origin_week": 2,
                        "path": write(
                            work / f"{model_id}_2026_2.json",
                            {"rows": forecast_rows(position, 2026, 2, names)},
                        ),
                    }
                ],
            }
        )
    # A model without a position: campaigns are not required to forecast players.
    models.append(
        {
            "model_id": "team_pace",
            "horizon": "season",
            "origin_week": 0,
            "targets": ["plays"],
            "training_through_season": 2025,
            "metrics": metrics,
            "forecasts": [
                {
                    "season": 2026,
                    "origin_week": 0,
                    "path": write(
                        work / "pace.json",
                        {
                            "rows": [
                                {
                                    "player_id": "SF",
                                    "season": 2026,
                                    "origin_week": 0,
                                    "predictions": {"plays": 1050.0},
                                }
                            ]
                        },
                    ),
                }
            ],
        }
    )
    root = tmp_path / "campaigns"
    publish_campaign(
        root,
        {
            "campaign_id": "fixture_v1",
            "name": "Fixture",
            "generated_at": "2026-10-10T00:00:00+00:00",
            "limitations": ["fixture"],
        },
        models,
    )
    return CampaignStore(root)


def test_summary_lists_models_horizons_and_latest_forecasts(store):
    summary = views.summary(store.campaign("fixture_v1"))
    assert summary["positions"] == ["RB", "WR"]
    assert summary["horizons"] == ["remaining", "season"]
    pace = next(m for m in summary["models"] if m["model_id"] == "team_pace")
    assert pace["position"] is None
    assert pace["latest_forecast"] == {"season": 2026, "origin_week": 0}


def test_rankings_merge_models_at_one_horizon_and_rank_by_points(store):
    result = views.rankings(store, "fixture_v1", horizon="remaining")
    names = [r["player_display_name"] for r in result["rankings"]]
    assert names == ["Back One", "Wide One", "Back Two"]
    assert [r["overall_rank"] for r in result["rankings"]] == [1, 2, 3]
    assert [r["position_rank"] for r in result["rankings"]] == [1, 1, 2]
    assert result["rankings"][0]["points_per_game"] == pytest.approx(250 / 15)
    assert result["rankings"][0]["availability_status"] == "ACT"
    assert {m["model_id"] for m in result["models"]} == {"rb_remaining", "wr_remaining"}
    assert result["limitations"] == ["fixture"]


def test_rankings_filter_by_position_search_and_player(store):
    only = views.rankings(store, "fixture_v1", horizon="remaining", position="WR")
    assert [r["player_id"] for r in only["rankings"]] == ["wr0"]
    found = views.rankings(store, "fixture_v1", horizon="remaining", search="two")
    assert [r["player_id"] for r in found["rankings"]] == ["rb1"]
    one = views.rankings(store, "fixture_v1", horizon="remaining", player_id="rb0")
    assert one["rankings"][0]["overall_rank"] == 1


def test_rankings_require_a_known_horizon_and_explicit_vintage_must_exist(store):
    with pytest.raises(CampaignNotFound):
        views.rankings(store, "fixture_v1", horizon="next4")
    empty = views.rankings(store, "fixture_v1", horizon="remaining", season=2026, origin_week=9)
    assert empty["rankings"] == [] and empty["models"] == []


def test_player_view_collects_every_model_that_forecast_the_player(store):
    result = views.player(store, "rb0")
    assert [f["model_id"] for f in result["forecasts"]] == ["rb_remaining"]
    assert result["forecasts"][0]["campaign_name"] == "Fixture"
    assert result["forecasts"][0]["candidates"] == 2
    assert views.player(store, "nobody")["forecasts"] == []


def test_targets_view_keeps_declared_target_columns(store):
    result = views.targets(store, "fixture_v1", "rb_remaining")
    assert result["model"]["targets"] == ["yards", "points"]
    assert result["rows"][0]["predictions"] == {"yards": 1000.0, "points": 250.0}
    assert result["season"] == 2026 and result["origin_week"] == 2
    pace = views.targets(store, "fixture_v1", "team_pace")
    assert pace["rows"][0]["position"] is None
    assert pace["rows"][0]["prediction"] is None


def test_rankings_span_campaigns_and_fill_missing_positions(store, tmp_path):
    """Other campaigns contribute only when explicitly requested."""
    work = tmp_path / "te"
    metrics = write(work / "metrics.json", {"rmse": 2.0})
    te_rows = forecast_rows("TE", 2026, 5, ["Tight One"])
    rb_rows = forecast_rows("RB", 2026, 5, ["Other Back"])
    publish_campaign(
        store.root,
        {
            "campaign_id": "fixture_te",
            "name": "TE fixture",
            "generated_at": "2026-10-11T00:00:00+00:00",
        },
        [
            {
                "model_id": "te_remaining",
                "position": "TE",
                "horizon": "remaining",
                "origin_week": 2,
                "targets": ["yards", "points"],
                "training_through_season": 2025,
                "metrics": metrics,
                "forecasts": [
                    {
                        "season": 2026,
                        "origin_week": 5,
                        "rolling_origin": True,
                        "path": write(work / "te.json", {"rows": te_rows}),
                    }
                ],
            },
            {
                "model_id": "rb_remaining",
                "position": "RB",
                "horizon": "remaining",
                "origin_week": 2,
                "targets": ["yards", "points"],
                "training_through_season": 2025,
                "metrics": metrics,
                "forecasts": [
                    {
                        "season": 2026,
                        "origin_week": 5,
                        "rolling_origin": True,
                        "path": write(work / "rb.json", {"rows": rb_rows}),
                    }
                ],
            },
        ],
    )
    alone = views.rankings(store, "fixture_v1", horizon="remaining")
    assert {r["position"] for r in alone["rankings"]} == {"RB", "WR"}
    merged = views.rankings(store, ["fixture_v1", "fixture_te"], horizon="remaining")
    assert {r["position"] for r in merged["rankings"]} == {"RB", "WR", "TE"}
    assert "Other Back" not in {r["player_display_name"] for r in merged["rankings"]}
    by_position = {m["position"]: m for m in merged["models"]}
    assert by_position["TE"]["campaign_id"] == "fixture_te" and by_position["TE"]["rolling_origin"]
    assert (
        by_position["RB"]["campaign_id"] == "fixture_v1" and not by_position["RB"]["rolling_origin"]
    )
    assert [c["campaign_id"] for c in merged["campaigns"]] == ["fixture_v1", "fixture_te"]
    explicit = views.rankings(store, ["fixture_te", "fixture_v1"], horizon="remaining")
    assert explicit["campaign_id"] == "fixture_te"
    assert "Other Back" in {r["player_display_name"] for r in explicit["rankings"]}
    assert "Back One" not in {r["player_display_name"] for r in explicit["rankings"]}
