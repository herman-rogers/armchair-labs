"""The 2026 experimental selectors stay frozen until their outcome is graded."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from engine.config.settings import load_metric_report_config
from engine.metrics.prospective import file_sha256


def test_2026_selector_definitions_match_prospective_freeze() -> None:
    freeze_path = (
        Path(__file__).parents[1] / "src/engine/config/experimental_freeze_2026.yaml"
    )
    freeze: dict[str, Any] = yaml.safe_load(freeze_path.read_text())
    configured = {
        entry["name"]: entry
        for entry in load_metric_report_config()["fit"]["models"]
        if entry["name"] in freeze["selectors"]
    }

    assert freeze["forecast_season"] == 2026
    assert freeze["outcomes_through"] == 2025
    assert freeze["promotion_status"] == "shadow_only"
    snapshot_path = Path(__file__).parents[1] / freeze["snapshot_file"]
    assert snapshot_path.read_text().count("\n") - 1 == freeze["pending_rows"]
    assert file_sha256(snapshot_path) == freeze["pending_prediction_sha256"]
    assert configured.keys() == freeze["selectors"].keys()
    for name, frozen in freeze["selectors"].items():
        current = configured[name]
        for field in (
            "target",
            "selection_metric",
            "selection_top_k",
            "factors",
            "fallback",
            "apply_live",
        ):
            assert current[field] == frozen[field], f"{name}.{field} changed after freeze"


def test_nextgen_challenger_is_shadow_only_and_outside_the_frozen_selectors() -> None:
    models = {
        entry["name"]: entry for entry in load_metric_report_config()["fit"]["models"]
    }
    nextgen = {name: spec for name, spec in models.items() if name.startswith("fitted_nextgen")}

    assert nextgen
    assert all(spec.get("apply_live") is False for spec in nextgen.values())
    assert "fitted_nextgen_season_points" in nextgen
    frozen_selectors = ("fitted_adaptive_ppg_hybrid", "fitted_adaptive_season_hybrid")
    assert all(
        "nextgen" not in factor
        for name in frozen_selectors
        for factor in models[name]["factors"]
    )


def test_rich_weekly_challenger_is_shadow_only_and_outside_frozen_selectors() -> None:
    models = {
        entry["name"]: entry for entry in load_metric_report_config()["fit"]["models"]
    }
    rich = {
        name: spec for name, spec in models.items() if name.startswith("fitted_rich_weekly")
    }

    assert {
        "fitted_rich_weekly_ppg",
        "fitted_rich_weekly_games",
        "fitted_rich_weekly_season_points",
        "fitted_rich_weekly_direct",
        "fitted_rich_weekly_combined",
    } <= rich.keys()
    assert all(spec.get("apply_live") is False for spec in rich.values())
    frozen_selectors = ("fitted_adaptive_ppg_hybrid", "fitted_adaptive_season_hybrid")
    assert all(
        "rich_weekly" not in factor
        for name in frozen_selectors
        for factor in models[name]["factors"]
    )
    assert "fitted_rich_adaptive_blend" not in models
    assert not any(name.startswith("fitted_adaptive_rich") for name in models)
