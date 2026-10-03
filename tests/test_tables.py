"""Release atomicity, dependency freshness, portable SQL, and research preservation."""

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from engine.cli_tables import app
from engine.data.shared import LocalStore
from engine.tables.build import BuildContext, refresh
from engine.tables.builders import player_pair_correlations
from engine.tables.query import connect, materialize
from engine.tables.registry import read_registry
from engine.tables.research import archive, verify_archive
from engine.tables.storage import current, load_table, publish_catalog
from engine.tables.transport import fetch, publish


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    data = root / "data"
    data.mkdir()
    pl.DataFrame({"id": [1, 2], "value": [3.0, 7.0]}).write_parquet(data / "source.parquet")
    config = {
        "schema_version": 1,
        "tables": {
            "base": {
                "description": "Fixture observations",
                "grain": "one id",
                "primary_key": ["id"],
                "kind": "parquet",
                "source": "source.parquet",
                "refresh_hours": 24,
                "observation_cutoff": {"season": 2025, "through_week": 4},
            },
            "summary": {
                "description": "Fixture derived result",
                "grain": "one group",
                "primary_key": ["group_id"],
                "kind": "sql",
                "sql_file": "summary.sql",
                "dependencies": ["base"],
                "refresh_hours": 168,
            },
        },
    }
    registry = root / "registry.yaml"
    registry.write_text(yaml.safe_dump(config))
    (root / "summary.sql").write_text(
        "SELECT 1 AS group_id, sum(value) AS total FROM analytics.base"
    )
    return data, registry, config


def test_refresh_idempotence_dependency_rebuild_and_atomic_failure(setup):
    data, registry, _ = setup
    first = refresh(data, registry_path=registry)
    before = (data / "tables/current.json").read_bytes()
    assert refresh(data, registry_path=registry)["tables"] == {
        "base": "unchanged",
        "summary": "unchanged",
    }
    assert (data / "tables/current.json").read_bytes() == before
    with connect(data) as db:
        assert db.sql(
            "SELECT table_type FROM information_schema.tables "
            "WHERE table_schema='analytics' AND table_name='base'"
        ).fetchone() == ("BASE TABLE",)
        assert db.sql("SELECT total FROM analytics.summary").fetchone() == (10.0,)
    with connect(data, native=False) as db:
        assert db.sql("SELECT total FROM analytics.summary").fetchone() == (10.0,)
    # A daily dependency rebuild also rebuilds the weekly dependent.
    pl.DataFrame({"id": [1, 2], "value": [4.0, 8.0]}).write_parquet(data / "source.parquet")
    result = refresh(data, registry_path=registry, names=["base"])
    assert "built" in result["tables"]["summary"]
    assert result["catalog"] != first["catalog"]
    with connect(data) as db:
        assert db.sql("SELECT total FROM analytics.summary").fetchone() == (12.0,)
    before = (data / "tables/current.json").read_bytes()
    pl.DataFrame({"id": [1, 1], "value": [5.0, 9.0]}).write_parquet(data / "source.parquet")
    with pytest.raises(ValueError, match="duplicate primary key"):
        refresh(data, registry_path=registry)
    assert (data / "tables/current.json").read_bytes() == before
    # Failure after a valid dependency build still leaves the complete old catalog selected.
    pl.DataFrame({"id": [1, 2], "value": [6.0, 9.0]}).write_parquet(data / "source.parquet")
    (registry.parent / "summary.sql").write_text("SELECT missing_column FROM analytics.base")
    with pytest.raises(Exception, match="missing_column"):
        refresh(data, registry_path=registry)
    assert (data / "tables/current.json").read_bytes() == before


def test_schedule_manual_tables_and_code_changes(setup):
    data, registry, config = setup
    config["tables"]["manual"] = {**config["tables"]["base"], "refresh_hours": None}
    registry.write_text(yaml.safe_dump(config))
    now = datetime(2026, 1, 1, tzinfo=UTC)
    refresh(data, registry_path=registry, due=True, now=now)
    assert "manual" not in current(data)["tables"]
    assert (
        refresh(data, registry_path=registry, due=True, now=now + timedelta(hours=1))["tables"]
        == {}
    )
    (registry.parent / "summary.sql").write_text(
        "SELECT 1 AS group_id, sum(value) * 2 AS total FROM analytics.base"
    )
    result = refresh(data, registry_path=registry, due=True, now=now + timedelta(days=1))
    assert result["tables"]["base"] == "unchanged"
    assert "built" in result["tables"]["summary"]


def test_source_batch_imports_together_and_rebuilds_dependents(setup, monkeypatch):
    from types import SimpleNamespace

    from engine.tables import build

    data, registry, config = setup
    batch = SimpleNamespace(
        ref={"version": "source_v1", "manifest_sha256": "a" * 64},
        manifest={
            "tables": {"base": {}, "other": {}},
            "current_observations": {"season": 2025, "through_week": 4},
        },
        path=lambda _: data / "source.parquet",
    )
    monkeypatch.setattr(build, "load_tables", lambda _: batch)
    config["tables"]["base"].update(kind="source", source="base")
    config["tables"]["other"] = {**config["tables"]["base"], "source": "other"}
    registry.write_text(yaml.safe_dump(config))
    refresh(data, registry_path=registry, names=["base"])
    catalog = current(data)
    assert set(catalog["tables"]) == {"base", "other", "summary"}
    for name, ref in catalog["tables"].items():
        manifest = load_table(data, name, ref)[1]
        assert manifest["source_release"] == batch.ref
        assert "gold" not in manifest
        assert manifest["observation_cutoff"] == batch.manifest["current_observations"]
    batch.ref = {"version": "source_v2", "manifest_sha256": "b" * 64}
    refreshed = refresh(data, registry_path=registry, names=["base"])
    assert all("built" in status for status in refreshed["tables"].values())
    # Failed imports cannot publish a partial new batch or stale dependent.
    before = (data / "tables/current.json").read_bytes()
    pl.DataFrame({"id": [1, 1], "value": [1.0, 2.0]}).write_parquet(data / "source.parquet")
    with pytest.raises(ValueError, match="duplicate primary key"):
        refresh(data, registry_path=registry, names=["base"])
    assert (data / "tables/current.json").read_bytes() == before


def test_pinned_sessions_native_cache_and_corruption(setup):
    data, registry, _ = setup
    refresh(data, registry_path=registry)
    old = current(data)
    native = materialize(data)
    assert materialize(data) == native
    with connect(data, native=True) as db:
        assert db.sql("SELECT total FROM analytics.summary").fetchone() == (10.0,)
    with connect(data) as db:
        pl.DataFrame({"id": [1], "value": [99.0]}).write_parquet(data / "source.parquet")
        refresh(data, registry_path=registry)
        assert db.sql("SELECT total FROM analytics.summary").fetchone() == (10.0,)
    assert native != materialize(data)
    mixed = {**current(data), "tables": {**current(data)["tables"], "base": old["tables"]["base"]}}
    with pytest.raises(ValueError, match="Mixed table dependencies"):
        publish_catalog(data, mixed)
    root, _ = load_table(data, "base", current(data)["tables"]["base"])
    (root / "data.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"), connect(data, native=True):
        pass


def test_registry_rejects_cycles_missing_dependencies_and_unsafe_names(setup):
    _, registry, config = setup
    for mutation in ("cycle", "unknown", "name"):
        changed = json.loads(json.dumps(config))
        if mutation == "cycle":
            changed["tables"]["base"]["dependencies"] = ["summary"]
        elif mutation == "unknown":
            changed["tables"]["summary"]["dependencies"] = ["missing"]
        else:
            changed["tables"]["a; DROP TABLE x"] = changed["tables"].pop("base")
        registry.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError):
            read_registry(registry)


def test_research_archive_exact_originals_and_parquet(tmp_path):
    data = tmp_path / "data"
    source = tmp_path / "study"
    source.mkdir()
    csv = b"player_a,player_b,r,n\nQB,WR,0.25,12\nQB,TE,,3\n"
    (source / "pairs.csv").write_bytes(csv)
    (source / "methodology.json").write_text('{"cutoff": "2025 Week 4"}')
    result = archive(data, source, "study_001")
    assert result["tables"]["pairs.csv"]["rows"] == 2
    root = data / "research/study_001"
    assert (root / "originals/pairs.csv").read_bytes() == csv
    assert (source / "pairs.csv").read_bytes() == csv
    assert pl.read_parquet(root / "tables/pairs.parquet")["r"].to_list() == [0.25, None]
    verify_archive(root)
    with pytest.raises(ValueError, match="already exists"):
        archive(data, source, "study_001")
    (root / "tables/pairs.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="corrupt"):
        verify_archive(root)


def test_transfer_fresh_machine_offline_and_failed_fetch_keeps_selection(
    setup, tmp_path, monkeypatch
):
    data, registry, _ = setup
    refresh(data, registry_path=registry)
    source = tmp_path / "study"
    source.mkdir()
    (source / "x.csv").write_text("id,value\n1,2\n")
    archive(data, source, "study")
    remote = f"file://{tmp_path / 'remote'}"
    published = publish(
        data, "r1", store_uri=remote, cache=tmp_path / "writer-cache", research_runs=["study"]
    )
    handoff = json.loads((data / "releases/tables/current.json").read_text())
    assert handoff["release"] == "r1"
    consumer_path = tmp_path / "remote" / published["consumer_catalog"]["key"]
    consumer = json.loads(consumer_path.read_text())
    assert set(consumer["tables"]) == {"base", "summary"}
    parquet = consumer["tables"]["base"]["parquet"]
    assert pl.read_parquet(Path(parquet["uri"].removeprefix("file://")))["id"].to_list() == [1, 2]
    assert consumer["tables"]["base"]["primary_key"] == ["id"]
    assert consumer["tables"]["base"]["rows"] == 2
    fresh = tmp_path / "reader/data"
    cache = tmp_path / "reader-cache"
    ref = data / "releases/tables/r1.json"
    fetch(fresh, ref, cache=cache, include_research=True)
    with connect(fresh) as db:
        assert db.sql("SELECT total FROM analytics.summary").fetchone() == (10.0,)
    assert not (fresh / "source.parquet").exists()  # query-only replicas need no original inputs
    assert (fresh / "research/study/tables/x.parquet").exists()
    root, _ = load_table(fresh, "base", current(fresh)["tables"]["base"])
    (root / "data.parquet").unlink()
    monkeypatch.setattr(LocalStore, "get", lambda *args: pytest.fail("offline tried network"))
    fetch(fresh, ref, cache=cache, offline=True)
    assert (root / "data.parquet").exists()
    before = (fresh / "tables/current.json").read_bytes()
    altered = {**published, "table_catalog": {**published["table_catalog"], "sha256": "0" * 64}}
    bad_ref = tmp_path / "bad.json"
    bad_ref.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="not bound"):
        fetch(fresh, bad_ref, cache=cache, offline=True)
    assert (fresh / "tables/current.json").read_bytes() == before

    (root / "data.parquet").write_bytes(b"local edit")
    with pytest.raises(ValueError, match="Local file differs"):
        fetch(fresh, ref, cache=cache, offline=True)
    assert (fresh / "tables/current.json").read_bytes() == before


def test_consumer_publication_failure_does_not_advance_handoff(setup, tmp_path, monkeypatch):
    data, registry, _ = setup
    refresh(data, registry_path=registry)
    kwargs = dict(store_uri=f"file://{tmp_path / 'remote'}", cache=tmp_path / "cache")
    publish(data, "first", **kwargs)
    before = (data / "releases/tables/current.json").read_bytes()
    real_put = LocalStore.put

    def interrupted(self, key, source, spec):
        if key == "releases/second/catalog.json":
            raise OSError("consumer upload interrupted")
        return real_put(self, key, source, spec)

    monkeypatch.setattr(LocalStore, "put", interrupted)
    with pytest.raises(OSError, match="consumer upload interrupted"):
        publish(data, "second", **kwargs)
    assert (data / "releases/tables/current.json").read_bytes() == before
    assert not (data / "releases/tables/second.json").exists()
    monkeypatch.setattr(LocalStore, "put", real_put)
    publish(data, "second", **kwargs)
    assert json.loads((data / "releases/tables/current.json").read_text())["release"] == "second"
    with pytest.raises(ValueError, match="changed between build and upload"):
        publish(data, "third", expected_catalog="outdated", **kwargs)


def test_correlation_builder_matches_existing_metric_with_missing_games(tmp_path):
    from engine.metrics.team_correlations import pair_summary

    rows = []
    for pid, points in {"qb": [1, 2, 3, 8, 9, 10], "wr": [2, 0, None, 16, 18, 20]}.items():
        for i, value in enumerate(points):
            if value is not None:
                rows.append(
                    dict(
                        player_id=pid,
                        name=pid,
                        position="QB" if pid == "qb" else "WR",
                        team="LA",
                        game_id=f"g{i}",
                        season=2024 + i // 3,
                        week=i % 3 + 1,
                        gameday=f"2025-01-{i + 1:02}",
                        starter_game=True,
                        league_points=float(value),
                    )
                )
    panel = pl.DataFrame(rows)
    path = tmp_path / "panel.parquet"
    panel.write_parquet(path)
    result = player_pair_correlations(BuildContext(tmp_path, {"team_player_games": path}, {}))
    for basis in ("raw", "season"):
        row = result.filter(
            (pl.col("scope") == "pooled")
            & (pl.col("sample") == "starter")
            & (pl.col("basis") == basis)
        ).row(0, named=True)
        expected = pair_summary(panel, "qb", "wr", basis)
        assert row["n"] == expected["n"] == 5
        assert row["correlation"] == pytest.approx(expected["r"])
        assert row["covariance"] == pytest.approx(expected["covariance"])
        assert row["degrees_of_freedom"] == expected["df"]


def test_cli_query_and_export(setup, tmp_path):
    data, registry, _ = setup
    runner = CliRunner()
    built = runner.invoke(app, ["refresh", "--registry", str(registry), "--data-dir", str(data)])
    assert built.exit_code == 0, built.output
    result = runner.invoke(
        app, ["query", "SELECT * FROM analytics.summary", "--data-dir", str(data)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["total"] == 10.0
    path = tmp_path / "result.parquet"
    result = runner.invoke(
        app,
        [
            "query",
            "SELECT * FROM analytics.summary",
            "--data-dir",
            str(data),
            "--output",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert pl.read_parquet(path)["total"].to_list() == [10.0]


def test_scheduled_upload_retries_after_local_success(setup, tmp_path, monkeypatch):
    from engine.tables import transport

    data, registry, _ = setup
    runner = CliRunner()
    real_publish = transport.publish
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(args[1])
        raise OSError("simulated upload interruption")

    monkeypatch.setenv("ARMCHAIR_DATA_CACHE", str(tmp_path / "upload-cache"))
    monkeypatch.setattr(transport, "publish", interrupted)
    args = [
        "refresh",
        "--due",
        "--upload",
        "--registry",
        str(registry),
        "--data-dir",
        str(data),
        "--store",
        f"file://{tmp_path / 'remote'}",
    ]
    assert runner.invoke(app, args).exit_code != 0
    assert len(current(data)["tables"]) == 2
    monkeypatch.setattr(transport, "publish", real_publish)
    retried = runner.invoke(app, args)
    assert retried.exit_code == 0, retried.output
    assert (data / "releases/tables" / f"{calls[0]}.json").exists()
    monkeypatch.setattr(transport, "publish", interrupted)
    no_op = runner.invoke(app, args)
    assert no_op.exit_code == 0, no_op.output
    assert json.loads(no_op.output)["upload"]["status"] == "already_published"
