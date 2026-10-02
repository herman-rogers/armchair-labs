"""Transport must survive interruption/corruption without overwriting research data."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from patron.data import shared
from patron.data.notebook_clean import clean


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "source"
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
    folder = root / "data/raw"
    folder.mkdir(parents=True)
    (folder / "a.parquet").write_bytes(b"test table bytes")
    (folder / "duplicate.parquet").write_bytes(b"test table bytes")
    (folder / ".env").write_text("do not publish")
    runs = root / "experiments/lab/runs/one"
    runs.mkdir(parents=True)
    (runs / "model.ubj").write_bytes(b"serialized model")
    config = {
        "profiles": {
            "notebooks": {"include": ["data/raw"]},
            "archive": {"extends": ["notebooks"], "include": ["experiments/lab/runs/one"]},
        }
    }
    cache = tmp_path / "publisher-cache"
    manifest = shared.snapshot(root, config, "r1", cache, tmp_path / "snapshot.json")
    return root, cache, manifest


def published(release, tmp_path):
    root, cache, manifest = release
    store = shared.LocalStore(tmp_path / "remote")
    reference = shared.publish(manifest, store, cache, tmp_path / "published.json", workers=2)
    final = json.loads((tmp_path / "published.json").read_text())
    return store, reference, final


def test_round_trip_dedup_models_and_offline(release, tmp_path):
    store, reference, manifest = published(release, tmp_path)
    assert len(manifest["object_generations"]) == 2
    assert manifest["serialized_model_files"] == ["experiments/lab/runs/one/model.ubj"]
    assert manifest["excluded_files"] == ["data/raw/.env"]
    reference["store"] = f"file://{store.root}"
    destination, cache = tmp_path / "fresh", tmp_path / "reader-cache"
    fetched_manifest = shared.read_release(reference, cache, offline=False)
    shared.fetch(fetched_manifest, destination, cache, "notebooks", store, workers=2)
    assert not (destination / "experiments").exists()
    assert shared.verify(manifest, destination, "notebooks")["verified_files"] == 2
    (destination / "data/raw/a.parquet").unlink()
    shared.fetch(
        shared.read_release(reference, cache, offline=True), destination, cache, "notebooks", None
    )
    shared.fetch(manifest, destination, cache, "archive", store)
    model = destination / "experiments/lab/runs/one/model.ubj"
    assert model.read_bytes() == b"serialized model"
    # Editing working data cannot mutate the cache or remotely published data.
    model.write_bytes(b"edited")
    digest = manifest["files"][str(model.relative_to(destination))]["sha256"]
    assert (cache / shared.object_key(digest)).read_bytes() == b"serialized model"
    assert reference["manifest_sha256"] == shared.sha256(tmp_path / "published.json")


def test_conflicting_working_file_is_never_overwritten(release, tmp_path):
    store, _, manifest = published(release, tmp_path)
    root = tmp_path / "destination"
    p = root / "data/raw/a.parquet"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"my unsaved work")
    with pytest.raises(ValueError, match="Local file differs"):
        shared.fetch(manifest, root, tmp_path / "cache", "notebooks", store)
    assert p.read_bytes() == b"my unsaved work"
    assert not (root / "data/raw/duplicate.parquet").exists()


def test_corrupt_download_never_publishes_ready_files(release, tmp_path):
    store, _, manifest = published(release, tmp_path)
    spec = manifest["files"]["data/raw/a.parquet"]
    (store.root / shared.object_key(spec["sha256"])).write_bytes(b"corrupted")
    root = tmp_path / "fresh"
    with pytest.raises(ValueError, match="SHA-256"):
        shared.fetch(manifest, root, tmp_path / "cache", "notebooks", store)
    assert not (root / "data/raw/a.parquet").exists()
    assert not (root / "data/.shared/ready").exists()
    assert not list((tmp_path / "cache").rglob(".download-*"))


def test_resume_repairs_cache_and_skips_uploaded_objects(release, tmp_path):
    store, ref, manifest = published(release, tmp_path)
    assert shared.publish(release[2], store, release[1], tmp_path / "published.json") == ref
    root, cache = tmp_path / "fresh", tmp_path / "cache"
    spec = manifest["files"]["data/raw/a.parquet"]
    cached = cache / shared.object_key(spec["sha256"])
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"incomplete")
    shared.fetch(manifest, root, cache, "notebooks", store)
    assert shared.valid(cached, spec)


def test_failed_upload_never_publishes_manifest(release, tmp_path):
    class Fail(shared.LocalStore):
        def put(self, key, source, spec):
            if source.read_bytes() == b"serialized model":
                raise OSError("disconnected")
            return super().put(key, source, spec)

    store = Fail(tmp_path / "remote")
    with pytest.raises(OSError, match="disconnected"):
        shared.publish(release[2], store, release[1], tmp_path / "published.json", workers=1)
    assert not (store.root / "releases/r1/manifest.json").exists()
    assert not (tmp_path / "published.json").exists()
    # Existing objects are reused on retry.
    shared.publish(
        release[2], shared.LocalStore(store.root), release[1], tmp_path / "published.json"
    )


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/absolute",
        "data/../x",
        "data\\x",
        "data/a:b",
        "data//x",
        "data/./x",
        "data/x\x00",
    ],
)
def test_reject_unsafe_paths(name):
    with pytest.raises(ValueError):
        shared.relative(name)


def test_reject_symlink_and_source_code_restore(release, tmp_path):
    store, _, manifest = published(release, tmp_path)
    root = tmp_path / "fresh"
    root.mkdir()
    (root / "data").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        shared.fetch(manifest, root, tmp_path / "cache", "notebooks", store)
    manifest["files"]["src/patron/cli.py"] = next(iter(manifest["files"].values()))
    with pytest.raises(ValueError, match="outside data/run"):
        shared.validate_manifest(manifest)


def test_manifest_tamper_rejected(release, tmp_path):
    store, reference, _ = published(release, tmp_path)
    reference["store"] = f"file://{store.root}"
    (store.root / reference["manifest_key"]).write_text("{}")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        shared.read_release(reference, tmp_path / "cache", offline=False)


def test_concurrent_fetch_same_cache(release, tmp_path):
    store, _, manifest = published(release, tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            pool.submit(
                shared.fetch, manifest, tmp_path / "fresh", tmp_path / "cache", "notebooks", store
            )
            for _ in range(2)
        ]
        for result in results:
            result.result()
    shared.verify(manifest, tmp_path / "fresh", "notebooks")


def test_new_snapshot_requires_new_id_and_missing_inputs_fail(release, tmp_path):
    root, cache, _ = release
    with pytest.raises(ValueError, match="new release"):
        shared.snapshot(root, {}, "r1", cache, tmp_path / "snapshot.json")
    with pytest.raises(FileNotFoundError, match="Required"):
        shared.snapshot(
            root,
            {"profiles": {"x": {"include": ["data/missing"]}}},
            "r2",
            cache,
            tmp_path / "new.json",
        )


def test_notebook_clean_preserves_code_and_ignores_sessions(release):
    root, _, _ = release
    path = root / "example.ipynb"
    path.write_text(
        json.dumps(
            {
                "metadata": {"widgets": {"state": "result"}},
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["print(1)"],
                        "execution_count": 1,
                        "outputs": [{"text": "1"}],
                        "metadata": {"execution": {"time": "now"}},
                    },
                    {"cell_type": "markdown", "source": ["Research"], "metadata": {}},
                ],
            }
        )
    )
    before = path.read_bytes()
    assert clean(root, check=True) == ["example.ipynb"]
    assert path.read_bytes() == before
    assert clean(root) == ["example.ipynb"]
    value = json.loads(path.read_text())
    assert value["cells"][0]["source"] == ["print(1)"]
    assert value["cells"][0]["outputs"] == []
    assert value["cells"][0]["execution_count"] is None
    assert clean(root, check=True) == []


def test_sealed_bytecode_and_lockfiles_survive_cleanup(release, tmp_path):
    root, cache, _ = release
    directory = root / "data/research/sealed"
    bytecode = directory / "implementation/__pycache__/source.pyc"
    bytecode.parent.mkdir(parents=True)
    bytecode.write_bytes(b"old sealed bytes")
    (directory / "uv.lock").write_text("pinned environment")
    (directory / "manifest.json").write_text(
        json.dumps({"files": {"implementation/__pycache__/source.pyc": shared.sha256(bytecode)}})
    )
    fresh = bytecode.with_name("new.pyc")
    fresh.write_bytes(b"fresh cache")
    manifest = shared.snapshot(
        root,
        {"profiles": {"archive": {"include": ["data/research"]}}},
        "r2",
        cache,
        tmp_path / "sealed.json",
    )
    assert str(bytecode.relative_to(root)) in manifest["files"]
    assert str(fresh.relative_to(root)) in manifest["excluded_files"]
    assert "data/research/sealed/uv.lock" in manifest["files"]
    shared.validate_manifest(manifest)
