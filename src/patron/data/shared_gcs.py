"""GCS transport with create-only writes, CRC32C and generation-pinned downloads."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import google.auth
import google.auth.credentials
import google_crc32c
from google.api_core.exceptions import NotFound
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.cloud import storage

from patron.data.shared import relative


class GcloudCredentials(google.auth.credentials.Credentials):
    """Use an existing gcloud login; no tokens or credentials are persisted by this app."""

    def __init__(self):
        super().__init__()
        self._refresh_lock = threading.Lock()

    def refresh(self, request):
        previous_token = self.token
        with self._refresh_lock:
            if self.token != previous_token and self.valid:
                return
            try:
                result = subprocess.run(
                    ["gcloud", "config", "config-helper", "--format=json"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                credential = json.loads(result.stdout)["credential"]
                self.token = credential["access_token"]
                self.expiry = datetime.fromisoformat(
                    credential["token_expiry"].replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except (OSError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
                raise RefreshError(
                    "Google Cloud login unavailable. Run gcloud auth login (or configure ADC)."
                ) from exc


def crc32c(path: Path) -> str:
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()


class GCSStore:
    def __init__(self, uri: str):
        parts = uri.removeprefix("gs://").split("/", 1)
        if not parts[0]:
            raise ValueError("Missing GCS bucket")
        self.bucket_name = parts[0]
        self.prefix = parts[1].strip("/") if len(parts) > 1 else ""
        if self.prefix:
            relative(self.prefix)
        try:
            credentials, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
            )
        except DefaultCredentialsError:
            credentials, project = GcloudCredentials(), None
        self.credentials, self.project = credentials, project
        self.local = threading.local()
        self.uploads = threading.BoundedSemaphore(4)

    def bucket(self):
        # One HTTP session/client per worker; credentials coordinate refreshes.
        if not hasattr(self.local, "client"):
            self.local.client = storage.Client(
                project=self.project or "armchair-labs", credentials=self.credentials
            )
        return self.local.client.bucket(self.bucket_name)

    def name(self, key):
        relative(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, source: Path, spec: dict) -> str:
        blob = self.bucket().blob(self.name(key))
        checksum = crc32c(source)
        try:
            blob.reload()
        except NotFound:
            # Invoke gcloud in bounded parallel workers; retain per-object SHA
            # metadata. gcloud no-clobber sets the atomic generation-zero precondition.
            env = {
                **os.environ,
                "CLOUDSDK_STORAGE_PROCESS_COUNT": "1",
                "CLOUDSDK_STORAGE_THREAD_COUNT": "4",
                "CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED": "False",
            }
            with self.uploads:
                result = subprocess.run(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        str(source),
                        f"gs://{self.bucket_name}/{self.name(key)}",
                        "--no-clobber",
                        f"--custom-metadata=sha256={spec['sha256']}",
                        "--quiet",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                )
            try:
                blob.reload()
            except NotFound:
                raise RuntimeError(
                    f"gcloud upload failed for {key}: {result.stderr[-2000:]}"
                ) from None
            # A competing create may return a nonzero exit code. Only a matching
            # remote object is accepted, regardless of gcloud's skip/exit status.
        if (
            blob.size != spec["size"]
            or blob.crc32c != checksum
            or (blob.metadata or {}).get("sha256") != spec["sha256"]
        ):
            raise ValueError(f"GCS object differs from immutable snapshot: {key}")
        return str(blob.generation)

    def get(self, key: str, destination: Path, generation: str | None = None):
        blob = self.bucket().blob(
            self.name(key), generation=int(generation) if generation else None
        )
        blob.download_to_filename(
            str(destination),
            if_generation_match=int(generation) if generation else None,
            checksum="crc32c",
            timeout=180,
        )
