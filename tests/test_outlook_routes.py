import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from patron.api import outlook_sources as sources
from patron.api import research_routes as routes


def artifacts(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    version = tmp_path / "research" / "outlook_fixture"
    inputs = version / "inputs"
    inputs.mkdir(parents=True)
    history = tmp_path / "research" / "history_fixture"
    history.mkdir()
    provenance = {"version": history.name, "manifest_sha256": "test-history"}
    monkeypatch.setattr(
        sources, "get_settings", lambda: SimpleNamespace(outputs_dir=outputs, data_dir=tmp_path)
    )
    monkeypatch.setattr(
        sources, "accepted_version", lambda name: (history, {"audited_input": provenance}, {})
    )
    for name in (
        "current_weeks.parquet",
        "current_snaps.parquet",
        "current_players.parquet",
        "current_schedule.parquet",
        "runner.py",
        "model.py",
    ):
        (inputs / name).write_text("fixture")
    for name in (
        "historical_features.parquet",
        "validation_predictions.parquet",
        "current_research_predictions.parquet",
    ):
        (version / name).write_text("fixture")
    report = dict(
        version=version.name,
        season=2026,
        through_week=2,
        observations_saved_at="2026-09-22T12:00:00+00:00",
        history=provenance,
        confidence_scores_published=False,
        players=[
            dict(player_id="a", espn_id=1, confidence_score=None, forecast_next4=20),
            dict(player_id="b", espn_id=2, confidence_score=None, forecast_next4=None),
        ],
    )
    (version / "outlook.json").write_text(json.dumps(report))
    manifest = dict(
        version=version.name,
        status="complete",
        protected_artifacts_unchanged=True,
        history_path="research/history_fixture",
        history=provenance,
        source_sha256={str(p.relative_to(version)): sources.digest(p) for p in inputs.iterdir()},
        output_sha256={p.name: sources.digest(p) for p in version.iterdir() if p.is_file()},
    )
    (version / "manifest.json").write_text(json.dumps(manifest))
    pointer = outputs / "player_outlook_2026.json"
    pointer.write_text(
        json.dumps(
            dict(version=version.name, manifest_sha256=sources.digest(version / "manifest.json"))
        )
    )
    return report, version, pointer


def test_verified_outlook_and_mutation_fail_closed(tmp_path, monkeypatch):
    report, version, _ = artifacts(tmp_path, monkeypatch)
    assert sources.load_outlook(2026) == report
    (version / "outlook.json").write_text("{}")
    with pytest.raises(HTTPException, match="artifact changed") as error:
        sources.load_outlook(2026)
    assert error.value.status_code == 409


@pytest.mark.parametrize("name", ["../history_fixture", "/tmp/anything", ".."])
def test_outlook_pointer_cannot_escape_root(tmp_path, monkeypatch, name):
    _, _, pointer = artifacts(tmp_path, monkeypatch)
    pointer.write_text(json.dumps(dict(version=name, manifest_sha256="invalid")))
    with pytest.raises(HTTPException) as error:
        sources.load_outlook(2026)
    assert error.value.status_code == 409


def test_changed_inputs_and_missing_pointer(tmp_path, monkeypatch):
    _, version, pointer = artifacts(tmp_path, monkeypatch)
    (version / "inputs/current_snaps.parquet").write_text("changed")
    with pytest.raises(HTTPException, match="artifact changed"):
        sources.load_outlook(2026)
    pointer.unlink()
    with pytest.raises(HTTPException) as error:
        sources.load_outlook(2026)
    assert error.value.status_code == 404


def test_outlook_history_must_still_be_accepted(tmp_path, monkeypatch):
    artifacts(tmp_path, monkeypatch)

    def rejected(name):
        raise HTTPException(409, "History changed")

    monkeypatch.setattr(sources, "accepted_version", rejected)
    with pytest.raises(HTTPException, match="History changed"):
        sources.load_outlook(2026)


def test_ownership_is_context_and_unmatched_is_not_free_agent(tmp_path, monkeypatch):
    report, _, _ = artifacts(tmp_path, monkeypatch)
    monkeypatch.setattr(routes, "get_league", lambda: SimpleNamespace(draft_season=2026))
    player = SimpleNamespace(
        espn_id=1, owner_team_id=None, owner_team_name=None, injury_status="OUT"
    )
    state = SimpleNamespace(
        snapshot=SimpleNamespace(
            season=2026, week=4, players=[player], my_team_id=1, captured_at="2026-09-22"
        ),
        stale=False,
    )
    monkeypatch.setattr(routes, "_state", lambda *args: state)
    response = routes.player_outlook(None)
    assert response["players"][0]["forecast_next4"] == report["players"][0]["forecast_next4"]
    assert response["players"][0]["injury_status"] == "OUT"
    assert [r["availability"] for r in response["players"]] == ["free_agent", "unknown"]
    assert response["newer_week_possible"]


def test_outlook_survives_unavailable_private_league(tmp_path, monkeypatch):
    artifacts(tmp_path, monkeypatch)
    monkeypatch.setattr(routes, "get_league", lambda: SimpleNamespace(draft_season=2026))

    def unavailable(*args):
        raise HTTPException(401, "Not signed in")

    monkeypatch.setattr(routes, "_state", unavailable)
    response = routes.player_outlook(None)
    assert not response["ownership_available"]
    assert all(r["availability"] == "unknown" for r in response["players"])
