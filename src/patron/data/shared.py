"""Immutable data transport: snapshot, publish, fetch, verify. No model execution.

Transport manifests wrap existing sealed datasets without modifying their bytes.
Content-addressed objects deduplicate files across releases and logical paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol

SCHEMA = 1
MODEL_SUFFIXES = {
    ".pkl",
    ".pickle",
    ".joblib",
    ".ubj",
    ".cbm",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
SKIP_PARTS = {
    "__pycache__",
    "__marimo__",
    ".ipynb_checkpoints",
    ".DS_Store",
    ".venv",
    ".notebook-venv",
    ".packages",
    ".git",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def relative(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(p in {".", ".."} for p in name.split("/"))
    ):
        raise ValueError(f"Unsafe relative path: {name!r}")
    if path.as_posix() != name or ":" in name or "\0" in name:
        raise ValueError(f"Nonportable relative path: {name!r}")
    return name


def inside(root: Path, name: str) -> Path:
    path = root / relative(name)
    # Reject symlinks, including symlink parents, even when they resolve within root.
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Symlink is not a data artifact: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {name}")
    return path


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
        raise ValueError("Release IDs must be simple names")
    return value


def object_key(digest: str) -> str:
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Invalid SHA-256")
    return f"objects/{digest[:2]}/{digest}"


def default_cache() -> Path:
    return (
        Path(os.environ.get("ARMCHAIR_DATA_CACHE", str(Path.home() / ".cache/armchair-labs")))
        .expanduser()
        .resolve()
    )


def valid(path: Path, spec: dict) -> bool:
    return path.is_file() and path.stat().st_size == spec["size"] and sha256(path) == spec["sha256"]


@contextmanager
def lock(cache: Path):
    """One publisher/fetcher per cache, with OS-released locks (Mac/Linux/Windows)."""
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".lock").open("a+b") as stream:
        if sys.platform == "win32":
            import msvcrt

            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if sys.platform == "win32":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_copy(source: Path, destination: Path, spec: dict, *, replace: bool = False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, prefix=".download-")
    temporary = Path(name)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        if not valid(temporary, spec):
            raise ValueError(f"Integrity failure: {source}")
        if replace:
            os.replace(temporary, destination)
        else:
            # Atomic create-only publication, without linking the mutable source/cache.
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_new(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
    temporary = Path(name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(
                    f"Immutable manifest already exists with different bytes: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def excluded(path: Path) -> bool:
    return (
        bool(set(path.parts) & SKIP_PARTS)
        or any(part.startswith((".tmp", ".partial")) for part in path.parts)
        or path.suffix in {".pyc", ".pyo", ".tmp"}
        or path.name.startswith(".env")
        or path.name.startswith("service-account")
        or ".secrets" in path.parts
        or ".browser-profile" in path.parts
    )


def sealed_bytecode(root: Path, path: Path) -> str | None:
    """Some legacy seals include bytecode. Keep those exact files, never fresh caches."""
    if path.suffix not in {".pyc", ".pyo"}:
        return None
    for parent in path.parents:
        if parent == root:
            break
        manifest = parent / "manifest.json"
        if not manifest.is_file():
            continue
        doc = json.loads(manifest.read_text())
        for field in ("files", "artifacts"):
            entries = doc.get(field, {})
            if isinstance(entries, dict):
                expected = entries.get(path.relative_to(parent).as_posix())
                if isinstance(expected, str) and expected == sha256(path):
                    return manifest.relative_to(root).as_posix()
    return None


def profile_roots(config: dict, profile: str, seen=()) -> list[str]:
    if profile in seen:
        raise ValueError("Profile inheritance cycle")
    spec = config["profiles"][profile]
    result = list(spec.get("include", []))
    for parent in spec.get("extends", []):
        result += profile_roots(config, parent, (*seen, profile))
    return sorted(set(result))


def progress(label: str, done: int, total: int):
    if done == 1 or done % 1000 == 0 or done == total:
        print(f"{label}: {done:,}/{total:,}", flush=True)


def verify_source_catalog_snapshot(manifest: dict, cache: Path) -> dict:
    """Bind auxiliary source selections to their entire sealed dependency chains."""
    files = manifest["files"]
    catalog_path = "data/source_catalog.json"
    if catalog_path not in files:
        return {}
    profiles = set(files[catalog_path]["profiles"])

    def bound(name, expected=None):
        spec = files.get(name)
        if spec is None or (expected is not None and spec["sha256"] != expected):
            raise ValueError(f"Source catalog dependency missing or changed: {name}")
        if not profiles <= set(spec["profiles"]):
            raise ValueError(f"Source dependency missing from a download profile: {name}")
        return spec

    def read(name, expected=None):
        spec = bound(name, expected)
        return json.loads(inside(cache, object_key(spec["sha256"])).read_text())

    sources = read(catalog_path)["sources"]
    for source_id, original_ref in sources.items():
        ref = original_ref
        for layer, directory in [
            ("gold", "releases"),
            ("enriched", "releases"),
            ("raw", "snapshots"),
        ]:
            root = f"data/{layer}/{directory}/{identifier(ref['version'])}"
            document = read(f"{root}/manifest.json", ref["manifest_sha256"])
            if document.get("source", {}).get("id") != source_id:
                raise ValueError(f"Source identity mismatch: {source_id}/{layer}")
            if document.get("status") != "accepted" or document.get("layer") != layer:
                raise ValueError(f"Unaccepted source dependency: {source_id}/{layer}")
            for relative_path, digest in document["files"].items():
                relative(relative_path)
                prefix = "data" if layer == "raw" else root
                bound(f"{prefix}/{relative_path}", digest)
            if layer != "raw":
                ref = document["input"]
    return sources


def snapshot(root: Path, config: dict, release: str, cache: Path, output: Path) -> dict:
    """Freeze selected bytes; no writes to source inputs or existing sealed manifests."""
    identifier(release)
    if output.exists():
        raise ValueError("Use a new release/output path for a new snapshot")
    paths: dict[str, set[str]] = {}
    skipped = set()
    sealed_references = {}
    for profile in config["profiles"]:
        for name in profile_roots(config, profile):
            source = inside(root, name)
            if not source.exists():
                raise FileNotFoundError(f"Required {profile} input is missing: {name}")
            candidates = [source] if source.is_file() else sorted(source.rglob("*"))
            for path in candidates:
                if path.is_symlink():
                    raise ValueError(f"Symlink in release: {path}")
                if not path.is_file():
                    continue
                name = path.relative_to(root).as_posix()
                if excluded(Path(name)):
                    reference = sealed_bytecode(root, path)
                    if reference:
                        sealed_references[name] = reference
                    else:
                        skipped.add(name)
                        continue
                relative(name)
                paths.setdefault(name, set()).add(profile)
    files = {}
    # Detect mutable collections: manifests are frozen before their subordinate files,
    # and every source stat is compared again before completing the transport manifest.
    observations = {}
    with lock(cache):
        for i, (name, profiles) in enumerate(sorted(paths.items()), 1):
            source = inside(root, name)
            before = source.stat()
            digest = sha256(source)
            spec = {"sha256": digest, "size": before.st_size}
            destination = inside(cache, object_key(digest))
            if not valid(destination, spec):
                atomic_copy(source, destination, spec, replace=True)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError(f"Input changed while snapshotting: {name}")
            observations[name] = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            files[name] = {**spec, "profiles": sorted(profiles)}
            if name in sealed_references:
                files[name]["sealed_reference"] = sealed_references[name]
            progress("Snapshot", i, len(paths))
        for name, expected in observations.items():
            stat = inside(root, name).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != expected:
                raise ValueError(f"Input changed before snapshot completed: {name}")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        manifest = {
            "schema_version": SCHEMA,
            "release": release,
            "created_at": datetime.now(UTC).isoformat(),
            "code_commit": commit,
            "source_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
            ),
            "profiles": config["profiles"],
            "files": files,
            "excluded_files": sorted(skipped),
            "serialized_model_files": [n for n in files if Path(n).suffix in MODEL_SUFFIXES],
            "limitations": [
                "Transport integrity does not establish research quality "
                "or complete source coverage.",
                "Historical failed/interrupted runs in archive remain failed/interrupted.",
                "Serialized estimators are included only when present; "
                "predictions are not fitted models.",
                "Source captures, sealed CSVs and NumPy arrays "
                "retain their original formats and hashes.",
            ],
        }
        validate_manifest(manifest)
        verify_source_catalog_snapshot(manifest, cache)
        write_new(output, json_bytes(manifest))
    return manifest


def validate_manifest(manifest: dict):
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported transport manifest")
    identifier(manifest["release"])
    objects: dict[str, int] = {}
    for name, spec in manifest["files"].items():
        relative(name)
        # The transport may restore data and run artifacts, never executable app source.
        parts = PurePosixPath(name).parts
        if not (
            parts[0] == "data"
            or (len(parts) > 3 and parts[0] == "experiments" and parts[2] == "runs")
        ):
            raise ValueError(f"Restore target is outside data/run directories: {name}")
        if excluded(Path(name)) and not (
            Path(name).suffix in {".pyc", ".pyo"}
            and spec.get("sealed_reference") in manifest["files"]
        ):
            raise ValueError(f"Forbidden release artifact: {name}")
        object_key(spec["sha256"])
        if not isinstance(spec["size"], int) or spec["size"] < 0:
            raise ValueError("Invalid object size")
        if not spec["profiles"] or not set(spec["profiles"]) <= manifest["profiles"].keys():
            raise ValueError("Unknown/empty file profiles")
        if objects.setdefault(spec["sha256"], spec["size"]) != spec["size"]:
            raise ValueError("Conflicting content hash sizes")


def selected(manifest: dict, profile: str) -> dict:
    validate_manifest(manifest)
    if profile not in manifest["profiles"]:
        raise ValueError(f"Unknown profile: {profile}")
    return {n: s for n, s in manifest["files"].items() if profile in s["profiles"]}


class Store(Protocol):
    def put(self, key: str, source: Path, spec: dict) -> str: ...
    def get(self, key: str, destination: Path, generation: str | None = None): ...


class LocalStore:
    """Same create-only contract as GCS, for offline integration/recovery tests."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def put(self, key: str, source: Path, spec: dict) -> str:
        target = inside(self.root, key)
        if target.exists():
            if not valid(target, spec):
                raise ValueError(f"Remote content collision: {key}")
        else:
            try:
                atomic_copy(source, target, spec)
            except FileExistsError:
                if not valid(target, spec):
                    raise ValueError(f"Concurrent remote content collision: {key}") from None
        return "local"

    def get(self, key: str, destination: Path, generation: str | None = None):
        shutil.copyfile(inside(self.root, key), destination)


def publish(manifest: dict, store: Store, cache: Path, output: Path, workers=16) -> dict:
    validate_manifest(manifest)
    unique = {s["sha256"]: s for s in manifest["files"].values()}
    with lock(cache):

        def upload(item):
            digest, spec = item
            path = inside(cache, object_key(digest))
            if not valid(path, spec):
                raise ValueError(f"Snapshot object is missing/corrupt: {digest}")
            return digest, store.put(object_key(digest), path, spec)

        generations = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(upload, item) for item in unique.items()]
            try:
                for i, future in enumerate(as_completed(futures), 1):
                    digest, generation = future.result()
                    generations[digest] = generation
                    progress("Publish objects", i, len(unique))
            except BaseException:
                # Leave completed immutable objects available for retry, but do not
                # start thousands of queued uploads after one object has failed.
                for future in futures:
                    future.cancel()
                raise
        final = {**manifest, "object_generations": generations}
        payload = json_bytes(final)
        # The completion manifest is written only after every object was verified/uploaded.
        fd, temporary_name = tempfile.mkstemp(dir=cache)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            spec = {"sha256": sha256(temporary), "size": len(payload)}
            key = f"releases/{manifest['release']}/manifest.json"
            generation = store.put(key, temporary, spec)
            write_new(output, payload)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "release": manifest["release"],
        "manifest_key": key,
        "manifest_sha256": spec["sha256"],
        "manifest_generation": generation,
    }


def fetch(manifest: dict, root: Path, cache: Path, profile: str, store: Store | None, workers=16):
    files = selected(manifest, profile)
    with lock(cache):
        # Preflight every destination before any workspace writes. Never overwrite local work.
        for name, spec in files.items():
            path = inside(root, name)
            if path.exists() and not valid(path, spec):
                raise ValueError(f"Local file differs; use a fresh checkout/destination: {name}")
        unique: dict[str, tuple[str, dict]] = {}
        for name, spec in files.items():
            unique.setdefault(spec["sha256"], (name, spec))

        def obtain(item):
            digest, (name, spec) = item
            target = inside(cache, object_key(digest))
            if valid(target, spec):
                return
            local = inside(root, name)
            if valid(local, spec):
                atomic_copy(local, target, spec, replace=True)
                return
            if store is None:
                raise FileNotFoundError(f"Not cached offline: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".download-")
            os.close(fd)
            temporary = Path(temp_name)
            try:
                generation = manifest.get("object_generations", {}).get(digest)
                store.get(object_key(digest), temporary, generation)
                if not valid(temporary, spec):
                    raise ValueError(f"Downloaded object failed SHA-256/size check: {name}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, _ in enumerate(pool.map(obtain, unique.items()), 1):
                progress("Fetch objects", i, len(unique))
        # Copy, never hard-link cache bytes to mutable working files.
        for i, (name, spec) in enumerate(files.items(), 1):
            path = inside(root, name)
            if not path.exists():
                atomic_copy(inside(cache, object_key(spec["sha256"])), path, spec)
            elif not valid(path, spec):
                raise ValueError(f"Local file changed during fetch: {name}")
            progress("Materialize", i, len(files))
        verify(manifest, root, profile)
        ready = root / "data/.shared/ready" / manifest["release"] / f"{profile}.json"
        write_new(
            ready,
            json_bytes(
                {
                    "release": manifest["release"],
                    "profile": profile,
                    "manifest_sha256": hashlib.sha256(json_bytes(manifest)).hexdigest(),
                }
            ),
        )


def verify(manifest: dict, root: Path, profile: str):
    files = selected(manifest, profile)
    for name, spec in files.items():
        if not valid(inside(root, name), spec):
            raise ValueError(f"Missing/corrupt materialized input: {name}")
    return {"verified_files": len(files), "profile": profile, "release": manifest["release"]}


def inventory(manifest: dict) -> dict:
    validate_manifest(manifest)
    result = {}
    for profile in manifest["profiles"]:
        files = selected(manifest, profile)
        unique = {s["sha256"]: s["size"] for s in files.values()}
        result[profile] = {
            "files": len(files),
            "unique_objects": len(unique),
            "logical_bytes": sum(s["size"] for s in files.values()),
            "unique_bytes": sum(unique.values()),
        }
    return {
        "profiles": result,
        "serialized_model_files": manifest["serialized_model_files"],
        "excluded_files": len(manifest["excluded_files"]),
    }


def make_store(uri: str):
    if uri.startswith("gs://"):
        from patron.data.shared_gcs import GCSStore

        return GCSStore(uri)
    if uri.startswith("file://"):
        return LocalStore(Path(uri[7:]))
    raise ValueError("Store must be gs://BUCKET/PREFIX or file:///absolute/path")


def read_release(reference: dict, cache: Path, *, offline: bool) -> dict:
    expected = reference["manifest_sha256"]
    object_key(expected)
    path = cache / "manifests" / f"{expected}.json"
    if path.exists() and sha256(path) == expected:
        return json.loads(path.read_text())
    if offline:
        raise FileNotFoundError("Release manifest is not cached; run data-fetch online first")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        make_store(reference["store"]).get(
            reference["manifest_key"], temporary, reference.get("manifest_generation")
        )
        if sha256(temporary) != expected:
            raise ValueError("Release manifest SHA-256 mismatch")
        manifest = json.loads(temporary.read_text())
        validate_manifest(manifest)
        if manifest["release"] != reference["release"]:
            raise ValueError("Release ID mismatch")
        os.replace(temporary, path)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path, default=default_cache())
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--profiles", type=Path, default=Path("data/releases/profiles.json"))
    snap.add_argument("--release", required=True)
    snap.add_argument("--output", type=Path, required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("--manifest", type=Path, required=True)
    pub.add_argument("--store", required=True)
    pub.add_argument("--reference", type=Path, required=True)
    pub.add_argument("--workers", type=int, default=16)
    for command in ["fetch", "verify"]:
        p = sub.add_parser(command)
        p.add_argument("--reference", type=Path, default=Path("data/releases/current.json"))
        p.add_argument("--profile", default="notebooks")
        p.add_argument("--offline", action="store_true")
        p.add_argument("--workers", type=int, default=16)
    inspect = sub.add_parser("inventory")
    inspect.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root, cache = args.root.resolve(), args.cache.resolve()
    if args.command == "snapshot":
        from patron.data.notebook_clean import clean

        if clean(root, check=True):
            raise ValueError("Clean notebook outputs first: just notebooks-clean")
        result = inventory(
            snapshot(root, json.loads(args.profiles.read_text()), args.release, cache, args.output)
        )
    elif args.command == "publish":
        manifest = json.loads(args.manifest.read_text())
        result = publish(
            manifest,
            make_store(args.store),
            cache,
            args.manifest.with_name("published.json"),
            args.workers,
        )
        result["store"] = args.store
        write_new(args.reference, json_bytes(result))
    elif args.command == "inventory":
        result = inventory(json.loads(args.manifest.read_text()))
    else:
        if not args.reference.exists():
            parser.error(
                f"No published release reference at {args.reference}; "
                "see docs/operations/shared-data.md"
            )
        reference = json.loads(args.reference.read_text())
        manifest = read_release(reference, cache, offline=args.offline)
        if args.command == "fetch":
            fetch(
                manifest,
                root,
                cache,
                args.profile,
                None if args.offline else make_store(reference["store"]),
                args.workers,
            )
        result = verify(manifest, root, args.profile)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
