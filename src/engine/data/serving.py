"""Version-bound page read models with bounded memory and optional published JSON.

The serving release is disposable: authoritative data remains in tables/products.
Every cache hit passes the source verifiers. The cache key binds the complete
catalog, implementation, endpoint and normalized arguments, including cutoffs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import OrderedDict
from functools import lru_cache, wraps
from pathlib import Path
from threading import RLock

from fastapi import HTTPException
from fastapi.params import Param

from engine.config.settings import get_settings
from engine.data.releases import current_catalog, digest, inside
from engine.data.verification import observe, read_json, revision, verified

SCHEMA_VERSION = 1
MAX_BYTES = 64 * 1024 * 1024
_responses: OrderedDict[tuple, bytes] = OrderedDict()
_lock = RLock()


@lru_cache(maxsize=4)
def _implementation(files: tuple) -> str:
    h = hashlib.sha256()
    for name, _ in files:
        h.update(name.encode())
        h.update(Path(name).read_bytes())
    return h.hexdigest()


def generation(data: Path) -> tuple[str, dict] | None:
    catalog = current_catalog(data)
    if catalog is None or "analysis" not in catalog.get("products", {}):
        return None
    source = Path(__file__).resolve().parents[1]
    files = tuple((str(p), revision(str(p))) for p in sorted(source.rglob("*.py")))
    binding = dict(
        schema_version=SCHEMA_VERSION, catalog=catalog, implementation=_implementation(files)
    )
    from engine.tables.application import active, catalog_state

    if active():
        # Never serve a direct-artifact response cache in the DuckDB namespace,
        # or let a stale/missing query catalog be hidden by a cached response.
        binding["query_catalog"] = catalog_state()["catalog"]
    return hashlib.sha256(encode(binding)).hexdigest(), binding


def encode(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode()


def model_key(name: str, arguments: dict) -> str:
    return hashlib.sha256(encode([name, arguments])).hexdigest()


def validate_sources(data: Path, name: str) -> None:
    from engine.api.profile_sources import load_profiles
    from engine.data.nextgen import load_analysis

    load_analysis(data)
    if name.startswith("profile") or name == "similar":
        load_profiles()


def published_models(data: Path, build: str) -> tuple[Path, dict] | None:
    pointer = data / "serving/current.json"

    def load():
        observe(pointer)
        if not pointer.exists():
            return None
        ref = read_json(pointer)
        # An older generation is not an alternative source for current data.
        if ref["generation"] != build:
            return None
        root = inside(data / "serving/releases", ref.get("version", build))
        if digest(root / "manifest.json") != ref["manifest_sha256"]:
            raise ValueError("Serving manifest changed")
        manifest = read_json(root / "manifest.json")
        if manifest["generation"] != build or manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Serving generation mismatch")
        return root, manifest

    return verified(("serving", str(data.absolute()), build), load)


def read_model(name: str):
    """Cache JSON handlers only; CSV/Response results are deliberately not cached."""

    def decorate(fn):
        signature = inspect.signature(fn, eval_str=True)

        @wraps(fn)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = {
                k: v.default if isinstance(v, Param) else v for k, v in bound.arguments.items()
            }
            data = fn.__globals__.get("get_settings", get_settings)().data_dir
            try:
                current = generation(data)
                if current is None:
                    return fn(**arguments)
                build, _ = current
                validate_sources(data, name)
                published = published_models(data, build)
                key = model_key(name, arguments)
                cache_key = (str(data.absolute()), build, key)
                artifact = None
                if published and key in published[1]["models"]:
                    artifact = published[0] / (key + ".json")
                    if digest(artifact) != published[1]["models"][key]:
                        raise ValueError("Published read model changed")
                with _lock:
                    if cache_key in _responses:
                        _responses.move_to_end(cache_key)
                        if current_catalog(data) != current[1]["catalog"]:
                            raise ValueError("Catalog changed while serving cached response")
                        return json.loads(_responses[cache_key])
                value = read_json(artifact) if artifact else fn(**arguments)
                # Reject a publication race even outside the HTTP catalog middleware.
                if current_catalog(data) != current[1]["catalog"]:
                    raise ValueError("Catalog changed while constructing read model")
                if isinstance(value, dict):
                    payload = encode(value)
                    size = len(payload)
                    if size <= MAX_BYTES:
                        with _lock:
                            _responses[cache_key] = payload
                            while sum(len(item) for item in _responses.values()) > MAX_BYTES:
                                _responses.popitem(last=False)
                return value
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise HTTPException(409, f"Read model unavailable: {exc}") from exc

        wrapped.__signature__ = signature
        wrapped.read_model_name = name
        return wrapped

    return decorate
