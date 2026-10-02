"""GCS-specific transfer guarantees; no live credentials or network required."""

from types import SimpleNamespace

import pytest

pytest.importorskip("google.cloud.storage")

from google.api_core.exceptions import NotFound, PreconditionFailed  # noqa: E402

from patron.data.shared import sha256  # noqa: E402
from patron.data.shared_gcs import GCSStore, crc32c  # noqa: E402


class Blob:
    def __init__(self, content=None):
        self.content = content
        self.calls = []
        self.metadata = {}
        self.generation = 123

    def reload(self):
        if self.content is None:
            raise NotFound("missing")

    def upload_from_filename(self, filename, **kwargs):
        from pathlib import Path

        self.calls.append(kwargs)
        self.content = Path(filename).read_bytes()
        self.size = len(self.content)
        self.crc32c = crc32c(Path(filename))

    def download_to_filename(self, filename, **kwargs):
        from pathlib import Path

        self.calls.append(kwargs)
        Path(filename).write_bytes(self.content)


def store_with(blob):
    # Avoid authentication discovery; use the real transport methods with fake GCS objects.
    store = object.__new__(GCSStore)
    store.prefix = "armchair-labs"
    requested = []

    def get_blob(name, **kwargs):
        requested.append((name, kwargs))
        return blob

    store.bucket = lambda: SimpleNamespace(blob=get_blob)
    return store, requested


def test_create_only_upload_remote_checksum_and_pinned_download(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"immutable artifact")
    spec = {"size": source.stat().st_size, "sha256": sha256(source)}
    blob = Blob()
    store, requested = store_with(blob)
    assert store.put("objects/aa/file", source, spec) == "123"
    assert blob.calls[0]["if_generation_match"] == 0
    assert blob.calls[0]["checksum"] == "crc32c"
    assert blob.metadata["sha256"] == spec["sha256"]
    assert store.put("objects/aa/file", source, spec) == "123"
    assert len(blob.calls) == 1  # Resume verifies and reuses the existing object.
    store.get("objects/aa/file", tmp_path / "copy", "123")
    assert requested[-1][1] == {"generation": 123}
    assert blob.calls[-1]["if_generation_match"] == 123
    assert (tmp_path / "copy").read_bytes() == source.read_bytes()
    blob.crc32c = "corrupt"
    with pytest.raises(ValueError, match="differs"):
        store.put("objects/aa/file", source, spec)


def test_concurrent_create_is_verified_before_reuse(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"concurrent identical upload")
    spec = {"size": source.stat().st_size, "sha256": sha256(source)}

    class Race(Blob):
        def upload_from_filename(self, filename, **kwargs):
            super().upload_from_filename(filename, **kwargs)
            raise PreconditionFailed("another uploader created it first")

    store, _ = store_with(Race())
    assert store.put("objects/aa/file", source, spec) == "123"
