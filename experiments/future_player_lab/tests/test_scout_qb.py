"""Retrospective imports preserve original policy and current serving constraints."""

import json

import polars as pl
import pytest
from experiments.future_player_lab.campaigns.scout_qb import (
    archive_study,
    digest,
    normalize_forecasts,
    saved_metrics,
)


def original(origin=4, season=2026):
    return pl.DataFrame(
        [
            {
                "season": season,
                "through_week": origin,
                "player_id": "qb1",
                "player_display_name": "QB One",
                "team": "NYG",
                "scheduled_games": 13,
                "yards_recipe": "yards_profile_boost_31_component",
                "attempts_profile_boost_31": 100.0,
                "rate_policy": 7.0,
                "yards_policy": 700.0,
                "actual": 600.0,
                "actual_attempts": 100.0,
                "rate_recipe": "rate_profile_ridge_1000_blend",
                "issue_date": "2026-10-08",
                "retired_known": False,
            }
        ]
    )


def served(**changes):
    return pl.DataFrame(
        [
            {
                "player_id": "qb1",
                "season": 2026,
                "through_week": 4,
                "unconstrained_prediction": 700.0,
                "prediction": 0.0,
                "issue_date": "2026-10-08",
                "constraint_reason": "Season-ending injury",
                "constraint_source": "https://example.org/saved-source",
                "news_known_on": "2026-09-23",
                **changes,
            }
        ]
    )


def test_current_forecast_preserves_served_gate_without_inventing_attempts():
    row = normalize_forecasts(original(), season=2026, origin=4, served=served())[0]
    assert row["predictions"] == {"passing_yards": 0.0}
    assert row["saved_research_components"] == {
        "passing_yards": 700.0,
        "attempts": 100.0,
        "passing_yards_per_attempt": 7.0,
    }
    assert row["constraint_reason"] == "Season-ending injury"
    assert row["news_known_on"] == "2026-09-23"


def test_earlier_saved_policy_preserves_components():
    row = normalize_forecasts(original(2), season=2026, origin=2)[0]
    assert row["predictions"] == row["saved_research_components"]
    assert row["predictions"]["passing_yards"] == 700.0
    assert row["rank"] == 1


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"player_id": "other"}, "population"),
        ({"through_week": 3}, "cutoff"),
        ({"season": 2025}, "cutoff"),
        ({"unconstrained_prediction": 701.0}, "component policy"),
    ],
)
def test_served_rows_must_match_original_policy(changes, error):
    with pytest.raises(ValueError, match=error):
        normalize_forecasts(original(), season=2026, origin=4, served=served(**changes))


def test_import_metrics_include_original_zero_outcomes():
    history = pl.concat(
        [
            original(season=2024),
            original(season=2025).with_columns(pl.lit(0.0).alias("actual")),
        ]
    )
    metrics = saved_metrics(history, ["passing_yards"], 4)
    pooled = next(row for row in metrics["rows"] if row["year"] is None)
    assert pooled["n"] == 2
    assert pooled["mse"] == 250000.0
    assert pooled["mae"] == 400.0


def test_archival_rejects_changed_original_artifact(tmp_path):
    study = tmp_path / "research" / "qb_original"
    study.mkdir(parents=True)
    artifact = study / "predictions.txt"
    artifact.write_text("original")
    manifest = study / "manifest.json"
    manifest.write_text(json.dumps({"files": {artifact.name: digest(artifact)}}))
    ref = {"version": study.name, "manifest_sha256": digest(manifest)}
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        archive_study(tmp_path, ref, tmp_path / "archive.zip")
