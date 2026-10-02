"""QB passing delivery fails closed and keeps challengers out of analysis."""

import json

import polars as pl
import pytest
from fastapi import HTTPException

from patron.api import qb_passing_routes as routes
from patron.data.releases import write_json


@pytest.fixture
def release(tmp_path, monkeypatch):
    path = tmp_path / "qb_passing"
    path.mkdir()
    monkeypatch.setattr(
        routes, "release", lambda: (tmp_path, dict(version="test", qb_passing_schema_version=1))
    )
    write_json(
        tmp_path / "registry.json",
        [
            dict(
                id="qb_passing:reference:next_game",
                target="passing_yards",
                horizon="next_game",
                positions=["QB"],
                validity="verified",
                serving="baseline",
                allowed_uses=["forecast"],
                dependencies=["passing_yards"],
            )
        ],
    )
    write_json(tmp_path / "incidents.json", [])
    write_json(
        path / "report.json",
        dict(
            version="test",
            examples=[dict(direct_boost=200)],
            component_scores=[dict(research=True)],
            decisions=[dict(horizon="next_game", window="modern", origin_type="weekly")],
            limitations=["Reference only"],
        ),
    )
    write_json(
        path / "evaluations.json",
        [
            dict(model=model, horizon="next_game", window="modern", origin_type="weekly")
            for model in ["reference", "prior_season", "conditional_boost"]
        ],
    )
    pl.DataFrame(
        [
            dict(
                horizon="next_game",
                season=2020,
                through_week=1,
                actual=200.0,
                actual_attempts=25,
                execution_ypa=7.0,
                probability_primary=0.8,
                actual_primary=1.0,
            )
        ]
    ).write_parquet(path / "predictions.parquet")
    pl.DataFrame(
        [
            dict(
                player_id="a", player_display_name="Test QB", horizon="next_game", prediction=200.0
            ),
            dict(
                player_id="b", player_display_name="Other QB", horizon="next_game", prediction=10.0
            ),
        ]
    ).write_parquet(path / "current.parquet")
    return tmp_path


def test_analysis_uses_only_current_references_and_exact_player_filter(release):
    result = routes.forecasts(horizon="next_game", player_id="a", offset=0, limit=100)
    assert result["total"] == 1
    assert result["forecasts"][0]["player_id"] == "a"
    assert "examples" not in result["report"]
    result = routes.evidence(
        horizon="next_game", window="modern", origin="weekly", scope="analysis"
    )
    assert {r["model"] for r in result["comparisons"]} == {"reference", "prior_season"}
    assert result["examples"] == result["decisions"] == []
    research = routes.evidence(
        horizon="next_game", window="modern", origin="weekly", scope="research"
    )
    assert len(research["comparisons"]) == 3
    assert research["examples"]


def test_missing_or_suspended_release_cannot_fall_back_to_old_forecasts(release, monkeypatch):
    write_json(
        release / "incidents.json",
        [dict(status="open", dependencies=["passing_yards"], reason="bad input")],
    )
    with pytest.raises(HTTPException, match="Suspended"):
        routes.forecasts(horizon="next_game", offset=0, limit=100)
    monkeypatch.setattr(routes, "release", lambda: (release, dict(version="older")))
    with pytest.raises(HTTPException, match="not been published"):
        routes.forecasts(horizon="next_game", offset=0, limit=100)


def test_approved_policy_and_efficiency_have_separate_serving_gates(release):
    report = json.loads((release / "qb_passing/report.json").read_text())
    report.update(
        current_entries={"next_game": "qb_passing:production:next_game"},
        rate_approved_horizons=["next_game"],
    )
    write_json(release / "qb_passing/report.json", report)
    registry = json.loads((release / "registry.json").read_text())
    registry.extend(
        [
            dict(registry[0], id="qb_passing:production:next_game", serving="approved"),
            dict(
                registry[0],
                id="qb_passing:rate:next_game",
                serving="archive",
                target="passing_efficiency",
            ),
        ]
    )
    write_json(release / "registry.json", registry)
    with pytest.raises(HTTPException, match="Excluded from analysis"):
        routes.forecasts(horizon="next_game", offset=0, limit=100)
    registry[-1]["serving"] = "approved"
    write_json(release / "registry.json", registry)
    assert routes.forecasts(horizon="next_game", offset=0, limit=100)["total"] == 2


def test_failed_variations_are_research_only(release, monkeypatch):
    monkeypatch.setattr(
        routes,
        "release",
        lambda: (
            release,
            dict(version="test", qb_passing_schema_version=1, qb_variations_schema_version=1),
        ),
    )
    path = release / "qb_variations"
    path.mkdir()
    write_json(
        path / "decisions.json",
        [dict(target="rate", horizon="next_game", approved=False, reason="No improvement")],
    )
    write_json(path / "report.json", dict(version="test"))
    write_json(
        path / "evaluations.json",
        [
            dict(model=m, target="rate", horizon="next_game", window="modern", origin_type="weekly")
            for m in ["rate_reference", "rate_policy", "rate_bad_model"]
        ],
    )
    public = routes.variations(
        target="rate", horizon="next_game", window="modern", origin="weekly", scope="analysis"
    )
    assert [r["model"] for r in public["comparisons"]] == ["rate_reference"]
    research = routes.variations(
        target="rate", horizon="next_game", window="modern", origin="weekly", scope="research"
    )
    assert len(research["comparisons"]) == 3
