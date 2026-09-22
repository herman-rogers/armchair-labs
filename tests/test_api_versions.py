"""The HTTP API keeps v1 compatible while exposing an independently selectable v2."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from patron.api import app as api_module


def test_board_version_selects_the_matching_artifact(tmp_path, monkeypatch) -> None:
    v1 = [{"player_id": "p1", "player_display_name": "Historical", "position": "WR"}]
    v2 = [{"player_id": "p2", "player_display_name": "Projected", "position": "WR"}]
    adaptive = [{"player_id": "p3", "player_display_name": "Adaptive", "position": "WR"}]
    (tmp_path / "board_v1.json").write_text(json.dumps(v1))
    (tmp_path / "board_v2.json").write_text(json.dumps(v2))
    (tmp_path / "board_adaptive.json").write_text(json.dumps(adaptive))
    monkeypatch.setattr(
        api_module,
        "get_settings",
        lambda: SimpleNamespace(outputs_dir=tmp_path),
    )
    query: dict[str, Any] = {
        "position": None,
        "flag": None,
        "search": None,
        "limit": 200,
        "offset": 0,
    }
    assert api_module.board(version="v1", **query)["players"] == v1
    response = api_module.board(version="v2", **query)
    assert response["version"] == "v2"
    assert response["players"] == v2
    assert api_module.board(version="adaptive", **query)["players"] == adaptive

    status = api_module.status()
    assert status["metric_versions"] == {
        "v1": {
            "available": True,
            "player_count": 1,
            "provenance": {"state": "unverified", "artifact_id": None},
        },
        "v2": {
            "available": True,
            "player_count": 1,
            "provenance": {"state": "unverified", "artifact_id": None},
        },
        "adaptive": {
            "available": True,
            "player_count": 1,
            "provenance": {"state": "unverified", "artifact_id": None},
        },
    }
    assert status["metric_report"] == {"available": False, "built_at": None}


def test_metric_report_reads_generated_artifact(tmp_path, monkeypatch) -> None:
    report = {"schema_version": 1, "title": "Metric Report", "results": []}
    (tmp_path / "metric_report.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        api_module,
        "get_settings",
        lambda: SimpleNamespace(outputs_dir=tmp_path),
    )

    assert api_module.metric_report() == report
    assert api_module.status()["metric_report"]["available"] is True
