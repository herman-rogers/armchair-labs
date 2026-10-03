import json
from types import SimpleNamespace

import polars as pl
import pytest
import requests
from research.capture_current_injuries import capture

from engine.data.releases import load_manifest, reference


def inputs(tmp_path):
    snapshot = tmp_path / "league.json"
    snapshot.write_text(
        json.dumps(
            {
                "captured_at": "2026-09-23T19:56:00+00:00",
                "season": 2026,
                "week": 3,
                "players": [
                    {
                        "espn_id": 1,
                        "player_display_name": "Test Player",
                        "position": "QB",
                        "injury_status": "OUT",
                        "espn_team": "NYG",
                    }
                ],
            }
        )
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "url": "https://example.test/injury",
                        "published_on": "2026-09-23",
                        "observations": [
                            {
                                "player_name": "Test Player",
                                "known_on": "2026-09-23",
                                "evidence_status": "reported_expectation",
                            }
                        ],
                    }
                ],
            }
        )
    )
    return snapshot, review


def test_capture_preserves_source_and_distinguishes_unknown_publication(tmp_path, monkeypatch):
    source = b'<html><div class="d3-o-article__body"><p>Test Player update</p></div></html>'
    monkeypatch.setattr(
        "research.capture_current_injuries.requests.get",
        lambda *a, **k: SimpleNamespace(
            content=source, url=a[0], status_code=200, headers={}, raise_for_status=lambda: None
        ),
    )
    snapshot, review = inputs(tmp_path)
    original = snapshot.read_bytes()
    root = capture(tmp_path, "injuries_1", snapshot, review)
    ref = reference(root)
    _, manifest = load_manifest(tmp_path, "raw", ref)
    assets = manifest["assets"]
    assert (tmp_path / assets["injuries/sources/0.html"]["object"]).read_bytes() == source
    assert assets["injuries/sources/0.html"]["published_at"] is None
    statuses = pl.read_parquet(tmp_path / assets["injuries/espn_statuses.parquet"]["object"])
    assert statuses["source_published_at"].to_list() == [None]
    assert "season_ending" not in statuses.columns
    snapshot.write_text("changed mutable snapshot")
    assert (tmp_path / assets["injuries/league_snapshot.json"]["object"]).read_bytes() == original
    with pytest.raises(FileExistsError):
        capture(tmp_path, "injuries_1", snapshot, review)
    (tmp_path / assets["injuries/sources/0.html"]["object"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        load_manifest(tmp_path, "raw", ref)


def test_failed_download_does_not_publish_snapshot(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("research.capture_current_injuries.requests.get", fail)
    snapshot, review = inputs(tmp_path)
    with pytest.raises(requests.ConnectionError):
        capture(tmp_path, "injuries_1", snapshot, review)
    assert not (tmp_path / "raw/snapshots/injuries_1/manifest.json").exists()


def test_rejects_backdated_news(tmp_path):
    snapshot, review = inputs(tmp_path)
    document = json.loads(review.read_text())
    document["sources"][0]["observations"][0]["known_on"] = "2026-09-22"
    review.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="before its source"):
        capture(tmp_path, "injuries_1", snapshot, review)
