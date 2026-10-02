import json
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import HTTPException

from patron.api import college_routes as routes
from patron.api import college_sources as sources


def artifacts(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    root = tmp_path / "research" / "college_fixture"
    (root / "inputs").mkdir(parents=True)
    raw = tmp_path / "research" / "source_fixture" / "raw"
    raw.mkdir(parents=True)
    (raw / "players_2025.parquet").write_text("source fixture")
    source = {
        "version": "source_fixture",
        "kind": "college_source",
        "status": "captured",
        "files": [
            {
                "filename": "players_2025.parquet",
                "sha256": sources.digest(raw / "players_2025.parquet"),
            }
        ],
    }
    (raw.parent / "manifest.json").write_text(json.dumps(source))
    history = {"version": "accepted_history"}
    report = {"version": root.name, "status": "experimental_not_promoted", "history": history}
    for name in sources.INPUTS | sources.OUTPUTS:
        (root / name).write_text("fixture")
    (root / "report.json").write_text(json.dumps(report))
    manifest = {
        "version": root.name,
        "status": "complete",
        "kind": "college_nfl_research",
        "protected_artifacts_unchanged": True,
        "history": history,
        "source_version": "source_fixture",
        "source_manifest_sha256": sources.digest(raw.parent / "manifest.json"),
        "source_sha256": {name: sources.digest(root / name) for name in sources.INPUTS},
        "output_sha256": {name: sources.digest(root / name) for name in sources.OUTPUTS},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    pointer = outputs / "college_nfl.json"
    pointer.write_text(
        json.dumps(
            {"version": root.name, "manifest_sha256": sources.digest(root / "manifest.json")}
        )
    )
    monkeypatch.setattr(
        sources, "get_settings", lambda: SimpleNamespace(outputs_dir=outputs, data_dir=tmp_path)
    )
    monkeypatch.setattr(
        sources, "accepted_version", lambda name: (tmp_path, {"audited_input": history}, {})
    )
    return root, raw, pointer


def test_college_publication_is_verified_and_mutations_fail_closed(tmp_path, monkeypatch):
    root, _, _ = artifacts(tmp_path, monkeypatch)
    assert routes.load_college()[0] == root
    (root / "identity_links.parquet").write_text("modified")
    with pytest.raises(HTTPException, match="artifact changed") as error:
        routes.load_college()
    assert error.value.status_code == 409


def test_raw_source_and_history_remain_verified(tmp_path, monkeypatch):
    _, raw, _ = artifacts(tmp_path, monkeypatch)
    (raw / "players_2025.parquet").write_text("modified")
    with pytest.raises(HTTPException, match="raw source changed"):
        routes.load_college()


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", ".."])
def test_college_pointer_paths_cannot_escape(tmp_path, monkeypatch, name):
    _, _, pointer = artifacts(tmp_path, monkeypatch)
    pointer.write_text(json.dumps({"version": name, "manifest_sha256": "invalid"}))
    with pytest.raises(HTTPException) as error:
        routes.load_college()
    assert error.value.status_code == 409


def test_missing_college_release_has_explicit_404(tmp_path, monkeypatch):
    _, _, pointer = artifacts(tmp_path, monkeypatch)
    pointer.unlink()
    with pytest.raises(HTTPException) as error:
        routes.load_college()
    assert error.value.status_code == 404


def test_identity_search_is_literal_and_no_nfl_link_is_not_failure(tmp_path, monkeypatch):
    frame = pl.DataFrame(
        {
            "college_name": ["Player [Junior]", "Other"],
            "college_id": ["1", "2"],
            "status": ["unmatched", "review"],
            "player_id": [None, None],
        }
    )
    frame.write_parquet(tmp_path / "identity_links.parquet")
    monkeypatch.setattr(routes, "load_college", lambda: (tmp_path, {}))
    response = routes.identities(search="[Junior]", limit=100)
    assert response["total"] == 1
    assert response["players"][0]["status"] == "unmatched"
    pl.DataFrame(
        {"college_id": ["1", "1", "2"], "college_position": ["WR", "WR", "QB"]}
    ).write_parquet(tmp_path / "college_seasons.parquet")
    receivers = routes.identities(position="WR", limit=100)
    assert receivers["total"] == 1  # Multiple seasons cannot duplicate a player.
    assert receivers["players"][0]["college_id"] == "1"
    assert receivers["players"][0]["status"] == "unmatched"
    assert routes.identities(position="RB", limit=100)["total"] == 0
    with pytest.raises(HTTPException) as error:
        routes.identities(status="invalid", limit=100)
    assert error.value.status_code == 422
    with pytest.raises(HTTPException) as error:
        routes.career()
    assert error.value.status_code == 422
    college_routes = [r for r in routes.router.routes if r.path.startswith("/api/research/college")]
    assert college_routes and all(r.methods == {"GET"} for r in college_routes)
