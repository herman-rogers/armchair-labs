"""The HTTP API keeps v1 compatible while exposing an independently selectable v2."""

from __future__ import annotations

import json
from types import SimpleNamespace

from patron.api import app as api_module


def test_board_version_selects_the_matching_artifact(tmp_path, monkeypatch) -> None:
    v1 = [{"player_id": "p1", "player_display_name": "Historical", "position": "WR"}]
    v2 = [{"player_id": "p2", "player_display_name": "Projected", "position": "WR"}]
    (tmp_path / "board_v1.json").write_text(json.dumps(v1))
    (tmp_path / "board_v2.json").write_text(json.dumps(v2))
    monkeypatch.setattr(
        api_module,
        "get_settings",
        lambda: SimpleNamespace(outputs_dir=tmp_path),
    )
    query = {"position": None, "flag": None, "search": None, "limit": 200, "offset": 0}
    assert api_module.board(version="v1", **query)["players"] == v1
    response = api_module.board(version="v2", **query)
    assert response["version"] == "v2"
    assert response["players"] == v2

    status = api_module.status()
    assert status["metric_versions"] == {
        "v1": {"available": True, "player_count": 1},
        "v2": {"available": True, "player_count": 1},
    }
