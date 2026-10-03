"""Application SQL, research isolation, publication and league capture parity."""

import asyncio
import json
from dataclasses import asdict

import polars as pl
import pytest

from engine.data.frames import read_frame
from engine.data.releases import digest, write_json
from engine.espn.sync import LeagueSnapshot, PlayerState, TeamState
from engine.tables import build
from engine.tables.application import query_scope, read_json
from engine.tables.league import query_documents, query_snapshot
from engine.tables.query import connect
from engine.tables.storage import current, load_table


@pytest.fixture
def application(tmp_path, monkeypatch):
    data = tmp_path / "data"
    root = data / "research/analysis_v1"
    root.mkdir(parents=True)
    frame = pl.DataFrame({"player_id": ["a", "b"], "prediction": [12.5, None]})
    frame.write_parquet(root / "rankings.parquet")
    write_json(root / "registry.json", [{"id": "one", "optional": None, "nested": [1, 2]}])
    write_json(root / "incidents.json", [])
    write_json(root / "report.json", {"season": 2026, "players": 2})
    source = {"version": "source_v1", "manifest_sha256": "a" * 64}
    write_json(
        root / "manifest.json",
        {
            "status": "complete",
            "gold": source,
            "files": {p.name: digest(p) for p in root.iterdir()},
        },
    )
    write_json(
        data / "current.json",
        {
            "schema_version": 1,
            "table_release": source,
            "products": {
                "analysis": {
                    "version": root.name,
                    "manifest_sha256": digest(root / "manifest.json"),
                }
            },
        },
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text("schema_version: 1\ntables: {}\n")
    monkeypatch.setattr(build, "DEFAULT_REGISTRY", registry)
    build.refresh(data, registry_path=registry)
    return data, root, registry, frame


def test_application_reads_native_tables_and_research_keeps_artifacts(application, monkeypatch):
    data, root, _, expected = application
    with connect(data) as db:
        assert db.sql(
            "SELECT table_type FROM information_schema.tables "
            "WHERE table_name='app_analysis_rankings'"
        ).fetchone() == ("BASE TABLE",)
        assert db.sql(
            "SELECT _record_json::JSON->>'id' FROM analytics.app_analysis_registry"
        ).fetchone() == ("one",)
    original = pl.read_parquet
    monkeypatch.setattr(pl, "read_parquet", lambda *a, **k: pytest.fail("Artifact read in app"))
    with query_scope(data):
        assert read_frame(root / "rankings.parquet").equals(expected)
        assert read_frame(
            root / "rankings.parquet", columns=["prediction"], player_id="a"
        ).rows() == [(12.5,)]
        assert read_json(root / "registry.json") == [
            {"id": "one", "optional": None, "nested": [1, 2]}
        ]
        assert read_json(root / "incidents.json") == []
        assert read_json(root / "report.json") == {"season": 2026, "players": 2}
    monkeypatch.setattr(pl, "read_parquet", original)
    assert read_frame(root / "rankings.parquet").equals(expected)


def test_stale_selection_and_changed_artifact_fail_closed(application):
    data, root, _, _ = application
    with query_scope(data):
        read_frame(root / "rankings.parquet")
    path = data / "current.json"
    catalog = json.loads(path.read_text())
    catalog["products"]["analysis"]["version"] = "new_unbuilt_release"
    write_json(path, catalog)
    with pytest.raises(ValueError, match="missing or stale"), query_scope(data):
        read_frame(root / "rankings.parquet")
    # A research read remains independent of the application query selection.
    assert read_frame(root / "rankings.parquet").height == 2


def test_cached_database_cannot_hide_changed_table(application):
    data, root, _, _ = application
    with query_scope(data):
        read_frame(root / "rankings.parquet")
    table, _ = load_table(
        data, "app_analysis_rankings", current(data)["tables"]["app_analysis_rankings"]
    )
    (table / "data.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"), query_scope(data):
        read_frame(root / "rankings.parquet")


def test_new_product_manifest_refreshes_sql_atomically(application):
    data, root, registry, _ = application
    pl.DataFrame({"player_id": ["a"], "prediction": [99.0]}).write_parquet(
        root / "rankings.parquet"
    )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["rankings.parquet"] = digest(root / "rankings.parquet")
    write_json(root / "manifest.json", manifest)
    catalog = json.loads((data / "current.json").read_text())
    catalog["products"]["analysis"]["manifest_sha256"] = digest(root / "manifest.json")
    write_json(data / "current.json", catalog)
    build.refresh(data, registry_path=registry)
    with query_scope(data):
        assert read_frame(root / "rankings.parquet")["prediction"].to_list() == [99.0]


def test_league_capture_sql_and_roundtrip_preserve_snapshot(tmp_path):
    snapshot = LeagueSnapshot(
        captured_at="2026-10-03T12:00:00+00:00",
        league_id=123,
        league_name="League",
        season=2026,
        week=4,
        my_team_id=3,
        roster_slots={"QB": 1},
        teams=[TeamState(3, "Team", "Owner", 1, 2, 145.0)],
        players=[PlayerState(1, "Player", "QB", "LA", "ACTIVE", 99.0, 80.0, 12.0, 3, "Team", "QB")],
    )
    result = query_snapshot(snapshot, tmp_path)
    assert asdict(result) == asdict(snapshot)
    league_data = tmp_path / "league/123/2026/data"
    with connect(league_data) as db:
        assert db.sql("SELECT espn_id, owner_team_id FROM analytics.league_players").fetchall() == [
            (1, 3)
        ]
        assert db.sql("SELECT count(*) FROM analytics.league_transactions").fetchone() == (0,)
    # A new capture advances league data only. Existing captures remain reproducible.
    snapshot.players[0].owner_team_id = None
    assert query_snapshot(snapshot, tmp_path).players[0].owner_team_id is None
    assert result.players[0].owner_team_id == 3
    records = [{"requested_week": 4, "nested": {"missing": None}, "items": []}]
    assert query_documents(tmp_path / "forecasts", "league_forecasts", records) == records


@pytest.mark.parametrize(
    ("path", "query", "headers", "expected"),
    [
        ("/api/nextgen/rankings", b"", [], True),
        ("/api/nextgen/league", b"", [], True),
        ("/api/profiles/player", b"", [], True),
        ("/api/status", b"", [], True),
        ("/api/nextgen/evidence", b"scope=research", [], False),
        ("/api/nextgen/evidence", b"", [(b"x-analysis-scope", b"research")], False),
        ("/api/research/college", b"", [], False),
        ("/api/league", b"scope=research", [], False),
    ],
)
def test_http_boundary_selects_sql_only_for_application(path, query, headers, expected):
    from starlette.requests import Request
    from starlette.responses import Response

    from engine.api.app import application_queries
    from engine.tables.application import active

    async def handler(request):
        assert active() is expected
        return Response("ok")

    request = Request({"type": "http", "path": path, "query_string": query, "headers": headers})
    result = asyncio.run(application_queries(request, handler))
    assert (result.headers.get("X-Data-Engine") == "duckdb") is expected
    assert not active()
