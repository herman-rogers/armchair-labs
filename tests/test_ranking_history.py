import json

import polars as pl
import pytest

from patron.data import ranking_history as module
from patron.data.releases import digest, write_json


@pytest.fixture
def history_data(tmp_path, monkeypatch):
    (tmp_path / "catalog_history").mkdir()
    pointer = None

    def publish(version, week, rank, season=2026, serving="approved"):
        nonlocal pointer
        root = tmp_path / "research" / version
        root.mkdir(parents=True)
        write_json(root / "ranking_report.json", dict(season=season, through_week=week))
        write_json(
            root / "registry.json",
            [
                dict(
                    id="rank",
                    validity="verified",
                    serving=serving,
                    target="league_points",
                    horizon="rest_of_season",
                    allowed_uses=["ranking"],
                    positions=["QB"],
                    populations=["returner"],
                    dependencies=[],
                )
            ],
        )
        write_json(root / "incidents.json", [])
        pl.DataFrame(
            [
                dict(
                    player_id="a",
                    overall_rank=rank,
                    position_rank=rank,
                    prediction=123.5,
                    scheduled_games=15,
                    entry_id="rank",
                    horizon="rest_of_season",
                    position="QB",
                    population="returner",
                )
            ]
        ).write_parquet(root / "rankings.parquet")
        write_json(
            root / "manifest.json",
            dict(ranking_schema_version=1, files={p.name: digest(p) for p in root.iterdir()}),
        )
        catalog = dict(
            products=dict(
                analysis=dict(version=version, manifest_sha256=digest(root / "manifest.json"))
            ),
            previous_catalog_sha256=pointer,
        )
        scratch = tmp_path / "catalog.json"
        write_json(scratch, catalog)
        pointer = digest(scratch)
        (tmp_path / "catalog_history" / f"{pointer}.json").write_bytes(scratch.read_bytes())
        return root

    def verify(data, ref):
        root = data / "research" / ref["version"]
        manifest = json.loads((root / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            if digest(root / name) != expected:
                raise ValueError("Changed historical file")
        return root, manifest

    monkeypatch.setattr(module, "load_analysis", verify)
    monkeypatch.setattr(module, "current_catalog", lambda _: dict(previous_catalog_sha256=pointer))
    return tmp_path, publish


def test_uses_latest_published_revision_per_week_and_same_season(history_data):
    data, publish = history_data
    publish("last-season", 1, 99, season=2025)
    publish("week-one", 1, 8)
    publish("week-two-first", 2, 6)
    publish("week-two-final", 2, 4)
    publish("current-week", 3, 1)
    history = module.ranking_history(data, 2026, 3, "rest_of_season")
    assert [s["through_week"] for s in history] == [1, 2]
    assert history[-1]["version"] == "week-two-final"
    assert history[-1]["rankings"][0]["overall_rank"] == 4
    assert history[-1]["rankings"][0]["prediction"] == 123.5
    assert history[-1]["rankings"][0]["scheduled_games"] == 15
    assert all(not s["rankings"] for s in module.ranking_history(data, 2026, 3, "next4"))


def test_suspended_scope_is_not_replaced_with_older_same_week(history_data):
    data, publish = history_data
    publish("first", 2, 3)
    publish("suspended", 2, 1, serving="shadow")
    assert module.ranking_history(data, 2026, 3, "rest_of_season")[0]["rankings"] == []


def test_cached_history_rechecks_artifacts(history_data):
    data, publish = history_data
    root = publish("prior", 2, 4)
    assert module.ranking_history(data, 2026, 3, "rest_of_season")
    (root / "rankings.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.ranking_history(data, 2026, 3, "rest_of_season")


def test_broken_publication_chain_is_unavailable(history_data):
    data, publish = history_data
    publish("prior", 2, 4)
    next((data / "catalog_history").iterdir()).write_text("{}")
    with pytest.raises(ValueError):
        module.ranking_history(data, 2026, 3, "rest_of_season")
