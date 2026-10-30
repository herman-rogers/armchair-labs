"""Automatic backfill fills only missing earlier origins, from cut-off snapshots."""

import json

from experiments.future_player_lab.campaigns import backfill


def snapshot(tmp_path, through=4):
    path = tmp_path / "snap" / "manifest.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            dict(
                status="accepted",
                assets={"current/weeks.parquet": {"object": "x", "sha256": "y"}},
                current_observations=dict(season=2026, through_week=through, saved_at="t"),
            )
        )
    )
    return path


def registry(tmp_path, origins):
    root = tmp_path / "registry"
    (root / "c").mkdir(parents=True)
    forecasts = [dict(season=2026, origin_week=o, path="p") for o in origins]
    models = [dict(model_id=f"{p}_weekly", horizon="weekly", forecasts=forecasts) for p in "ab"]
    models.append(dict(model_id="a_season", horizon="season", forecasts=[]))
    (root / "c" / "manifest.json").write_text(json.dumps(dict(models=models)))
    return root


def test_derived_snapshot_only_moves_the_cutoff(tmp_path):
    source = snapshot(tmp_path)
    derived = json.loads(backfill.as_of(source, 2, tmp_path / "derived").read_text())
    original = json.loads(source.read_text())
    assert derived["current_observations"]["through_week"] == 2
    assert derived["assets"] == original["assets"]
    assert derived["retrospective"] is True


def test_missing_origins_and_limited_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill, "DERIVED", tmp_path / "derived")
    root = registry(tmp_path, [1, 4])
    assert backfill.missing_origins(root, "c", 2026, 4) == [0, 2, 3]
    seen = []

    def forecast(path):
        seen.append(json.loads(path.read_text())["current_observations"]["through_week"])
        return "ok"

    result = backfill.run("c", forecast, snapshot=snapshot(tmp_path), registry=root, limit=2)
    assert seen == [3, 2]
    assert result["remaining"] == [0]
    assert backfill.missing_origins(root, "absent", 2026, 4) == []
