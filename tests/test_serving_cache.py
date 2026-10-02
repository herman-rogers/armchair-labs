"""Serving acceleration must preserve release, identity, and cutoff safety."""

import json
import os
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import HTTPException

from patron.data import serving
from patron.data.frames import read_frame
from patron.data.releases import atomic_json, digest
from patron.data.verification import (
    clear_verification_cache,
    observe_listing,
    read_json,
    verification_batch,
    verified,
)


@pytest.fixture(autouse=True)
def clean():
    clear_verification_cache()
    serving._responses.clear()
    yield
    clear_verification_cache()
    serving._responses.clear()


def test_verification_reuses_closure_but_detects_same_size_rewrite_and_deletion(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"value":1}')
    count = 0

    def loader():
        nonlocal count
        count += 1
        return read_json(source)

    assert verified(("test",), loader) == {"value": 1}
    returned = verified(("test",), loader)
    returned["value"] = 999
    assert verified(("test",), loader) == {"value": 1}
    assert count == 1
    before = source.stat()
    source.write_text('{"value":2}')
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert verified(("test",), loader) == {"value": 2}
    source.unlink()
    with pytest.raises(FileNotFoundError):
        verified(("test",), loader)


def test_nested_verification_tracks_symlink_retarget_and_new_audit(tmp_path):
    a, b, link = (tmp_path / n for n in ("a", "b", "link"))
    a.mkdir()
    b.mkdir()
    (a / "file.json").write_text('{"value":1}')
    (b / "file.json").write_text('{"value":2}')
    link.symlink_to(a)

    def nested():
        return verified(("child",), lambda: read_json(link / "file.json"))

    assert verified(("parent",), nested)["value"] == 1
    link.unlink()
    link.symlink_to(b)
    assert verified(("parent",), nested)["value"] == 2

    def audits():
        observe_listing(a)
        return sorted(p.name for p in a.glob("*.json"))

    assert len(verified(("audits",), audits)) == 1
    (a / "extra.json").write_text("{}")
    assert len(verified(("audits",), audits)) == 2


def test_build_batch_rejects_input_changes_before_commit(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"value":1}')
    with pytest.raises(ValueError, match="Inputs changed"), verification_batch():
        verified(("batch",), lambda: read_json(source))
        source.write_text('{"value":2}')
        # Only offline generation can use the captured value during a batch.
        assert verified(("batch",), lambda: read_json(source))["value"] == 1
    assert verified(("batch",), lambda: read_json(source))["value"] == 2


def test_parquet_cache_is_independent_and_replacement_sensitive(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"id": [1], "value": [2]}).write_parquet(path)
    first = read_frame(path)
    first.replace_column(1, pl.Series("value", [99]))
    assert read_frame(path)["value"][0] == 2
    replacement = tmp_path / "new.parquet"
    pl.DataFrame({"id": [1], "value": [3]}).write_parquet(replacement)
    replacement.replace(path)
    assert read_frame(path, columns=["value"]).to_dicts() == [{"value": 3}]


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    catalog = {
        "schema_version": 1,
        "gold": {"version": "g"},
        "products": {"analysis": {}},
        "published_at": "one",
    }
    atomic_json(tmp_path / "current.json", catalog)
    monkeypatch.setattr(serving, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    source = tmp_path / "source.json"
    source.write_text('{"allowed":true}')
    expected = digest(source)

    def validate(*args):
        if digest(source) != expected:
            raise ValueError("Source changed")

    monkeypatch.setattr(serving, "validate_sources", validate)
    calls = []

    @serving.read_model("profile-test")
    def handler(player_id="one", season=2026, scope="analysis"):
        calls.append((player_id, season, scope))
        return dict(player_id=player_id, season=season, scope=scope, calls=len(calls))

    return handler, calls, source


def test_response_cache_separates_identity_cutoff_scope_and_publication(endpoint, tmp_path):
    handler, calls, _ = endpoint
    assert handler() == handler()
    assert len(calls) == 1
    handler(player_id="two")
    handler(season=2025)
    handler(scope="research")
    assert len(calls) == 4
    catalog = read_json(tmp_path / "current.json")
    catalog["published_at"] = "two"
    atomic_json(tmp_path / "current.json", catalog)
    handler()
    assert len(calls) == 5


def test_warm_response_never_masks_changed_source(endpoint):
    handler, _, source = endpoint
    handler()
    source.write_text('{"allowed":false}')
    with pytest.raises(HTTPException, match="Source changed") as exc:
        handler()
    assert exc.value.status_code == 409


def test_published_response_loads_without_calculation_and_detects_tampering(endpoint, tmp_path):
    handler, calls, _ = endpoint
    build, binding = serving.generation(tmp_path)
    root = tmp_path / "serving/releases" / build
    root.mkdir(parents=True)
    key = serving.model_key("profile-test", dict(player_id="one", season=2026, scope="analysis"))
    artifact = root / (key + ".json")
    artifact.write_bytes(serving.encode({"precomputed": True}))
    manifest = root / "manifest.json"
    manifest.write_bytes(
        serving.encode(
            dict(
                schema_version=1, generation=build, binding=binding, models={key: digest(artifact)}
            )
        )
    )
    atomic_json(
        tmp_path / "serving/current.json", dict(generation=build, manifest_sha256=digest(manifest))
    )
    assert handler() == {"precomputed": True}
    assert not calls
    artifact.write_text(json.dumps({"precomputed": False}))
    with pytest.raises(HTTPException, match="Published read model changed"):
        handler()
