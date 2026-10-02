"""Cache successful verification while checking every dependency revision on reuse.

There is no TTL or grace period: edits, replacements, deletions, new directory
entries and symlink retargeting invalidate immediately. Cached failures are never
served. Nested verifiers contribute their complete dependency closure.
"""

from __future__ import annotations

import os
import stat
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import cast


class VerificationChanged(ValueError):
    """A dependency changed within one bounded operation."""


Revision = tuple[int, int, int, int, int, int] | None
_collector: ContextVar[dict[str, Revision] | None] = ContextVar("verification_files", default=None)
_cache: OrderedDict[tuple, tuple[dict[str, Revision], object]] = OrderedDict()
_lock = RLock()
_batch: ContextVar[dict | None] = ContextVar("verification_batch", default=None)
MAX_ENTRIES = 64


def revision(path: str) -> Revision:
    try:
        listing = path.startswith("listing:")
        s = os.lstat(path.removeprefix("listing:"))
        if stat.S_ISDIR(s.st_mode) and not listing:
            return s.st_dev, s.st_ino, s.st_mode, 0, 0, 0
        return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    except FileNotFoundError:
        return None


def observe(path: Path) -> None:
    """Record the lexical path and ancestors, including symlink targets."""
    collector = _collector.get()
    if collector is None:
        return
    absolute = Path(os.path.abspath(path))
    for entry in (absolute, *absolute.parents):
        name = str(entry)
        if name not in collector:
            collector[name] = revision(name)
            if entry.is_symlink():
                observe(entry.resolve())


def observe_listing(path: Path) -> None:
    observe(path)
    collector = _collector.get()
    if collector is not None:
        key = "listing:" + str(path.absolute())
        collector.setdefault(key, revision(key))


def read_json(path: Path):
    import json

    observe(path)
    return json.loads(path.read_text())


def verified[T](key: tuple, loader: Callable[[], T]) -> T:
    """Single-flight, bounded cache; callers receive independently mutable metadata."""
    parent = _collector.get()
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            dependencies, value = cached
            batch = _batch.get()
            if (batch is not None and batch.get(key) is dependencies) or all(
                revision(p) == expected for p, expected in dependencies.items()
            ):
                if batch is not None:
                    batch[key] = dependencies
                _cache.move_to_end(key)
                if parent is not None:
                    parent.update(dependencies)
                return cast(T, deepcopy(value))
            del _cache[key]
        dependencies = {}
        token = _collector.set(dependencies)
        try:
            value = loader()
            if not all(revision(p) == expected for p, expected in dependencies.items()):
                raise VerificationChanged("Release dependencies changed during verification")
        finally:
            _collector.reset(token)
        if parent is not None:
            parent.update(dependencies)
        _cache[key] = dependencies, deepcopy(value)
        batch = _batch.get()
        if batch is not None:
            batch[key] = dependencies
        while len(_cache) > MAX_ENTRIES:
            _cache.popitem(last=False)
        return value


def clear_verification_cache() -> None:
    with _lock:
        _cache.clear()


@contextmanager
def verification_batch():
    """Check dependencies once per operation, then recheck before exposing output."""
    if _batch.get() is not None:
        yield
        return
    dependencies = {}
    token = _batch.set(dependencies)
    try:
        yield
        closure = {}
        for files in dependencies.values():
            for path, expected in files.items():
                if path in closure and closure[path] != expected:
                    raise VerificationChanged("Inputs changed during verification scope")
                closure[path] = expected
        if not all(revision(p) == expected for p, expected in closure.items()):
            raise VerificationChanged("Inputs changed during verification scope")
    finally:
        _batch.reset(token)
