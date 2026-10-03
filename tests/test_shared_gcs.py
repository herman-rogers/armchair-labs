"""GCS-specific transfer guarantees; no live credentials or network required."""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("google.cloud.storage")

from google.api_core.exceptions import NotFound  # noqa: E402

from engine.data.shared import sha256  # noqa: E402
from engine.data.shared_gcs import GCSStore, crc32c  # noqa: E402


class Blob:
    def __init__(self):
        self.content = None
        self.calls = []
        self.metadata = {}
        self.generation = 123

    def reload(self):
        if self.content is None:
            raise NotFound("missing")

    def download_to_filename(self, filename, **kwargs):
        self.calls.append(kwargs)
        Path(filename).write_bytes(self.content)


def store_with(blob):
    store = object.__new__(GCSStore)
    store.prefix = "armchair-labs"
    store.bucket_name = "test-bucket"
    store.uploads = threading.BoundedSemaphore(4)
    requested = []

    def get_blob(name, **kwargs):
        requested.append((name, kwargs))
        return blob

    store.bucket = lambda: SimpleNamespace(blob=get_blob)
    return store, requested


def install_object(blob, command):
    source = Path(command[3])
    blob.content = source.read_bytes()
    blob.size = source.stat().st_size
    blob.crc32c = crc32c(source)
    blob.metadata = {
        "sha256": next(x.split("=", 2)[2] for x in command if x.startswith("--custom-metadata="))
    }


def test_gcloud_no_clobber_checksum_and_pinned_download(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"immutable artifact")
    spec = {"size": source.stat().st_size, "sha256": sha256(source)}
    blob = Blob()
    store, requested = store_with(blob)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command[:3] == ["gcloud", "storage", "cp"]
        assert "--no-clobber" in command
        assert not any(x.startswith("--if-generation-match") for x in command)
        assert kwargs["env"]["CLOUDSDK_STORAGE_THREAD_COUNT"] == "4"
        install_object(blob, command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("engine.data.shared_gcs.subprocess.run", run)
    assert store.put("objects/aa/file", source, spec) == "123"
    assert commands[0][4] == "gs://test-bucket/armchair-labs/objects/aa/file"
    assert store.put("objects/aa/file", source, spec) == "123"
    assert len(commands) == 1
    store.get("objects/aa/file", tmp_path / "copy", "123")
    assert requested[-1][1] == {"generation": 123}
    assert blob.calls[-1]["if_generation_match"] == 123
    assert (tmp_path / "copy").read_bytes() == source.read_bytes()
    blob.crc32c = "corrupt"
    with pytest.raises(ValueError, match="differs"):
        store.put("objects/aa/file", source, spec)


def test_concurrent_create_is_verified_before_reuse(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"concurrent identical upload")
    spec = {"size": source.stat().st_size, "sha256": sha256(source)}
    blob = Blob()
    store, _ = store_with(blob)

    def run(command, **kwargs):
        install_object(blob, command)
        return SimpleNamespace(returncode=1, stderr="precondition failed")

    monkeypatch.setattr("engine.data.shared_gcs.subprocess.run", run)
    assert store.put("objects/aa/file", source, spec) == "123"
    blob.metadata = {}
    with pytest.raises(ValueError, match="differs"):
        store.put("objects/aa/file", source, spec)


def test_failed_gcloud_upload_cannot_publish(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"missing upload")
    spec = {"size": source.stat().st_size, "sha256": sha256(source)}
    store, _ = store_with(Blob())
    monkeypatch.setattr(
        "engine.data.shared_gcs.subprocess.run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stderr="transfer failed"),
    )
    with pytest.raises(RuntimeError, match="transfer failed"):
        store.put("objects/aa/file", source, spec)


def test_gcloud_uploads_run_in_bounded_parallel(tmp_path, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    source = tmp_path / "source"
    source.write_bytes(b"parallel artifact")
    spec = {"size": source.stat().st_size, "sha256": sha256(source)}
    state = {"active": 0, "maximum": 0}
    guard = threading.Lock()
    blobs = {f"armchair-labs/objects/item-{i}": Blob() for i in range(12)}
    store, _ = store_with(Blob())
    store.bucket = lambda: SimpleNamespace(blob=lambda name: blobs[name])

    def run(command, **kwargs):
        with guard:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        try:
            time.sleep(0.03)
            install_object(blobs[command[4].removeprefix("gs://test-bucket/")], command)
            return SimpleNamespace(returncode=0, stderr="")
        finally:
            with guard:
                state["active"] -= 1

    monkeypatch.setattr("engine.data.shared_gcs.subprocess.run", run)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda i: store.put(f"objects/item-{i}", source, spec), range(12)))
    assert results == ["123"] * 12
    assert state["maximum"] == 4
    assert state["active"] == 0
