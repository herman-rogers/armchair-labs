"""Parquet cache for derived artifacts.

nflreadpy already caches raw downloads, so this layer exists for the one thing that is
expensive after the download: reducing multi-season play-by-play to per-player bonus
points. That computation reads the largest table in the project and produces a few
thousand rows, which is exactly the shape worth persisting.

Keyed by content rather than by clock. A derived artifact is stale when the seasons it
was built from change, so the key names them; `force` covers the nightly refresh, when
new games have landed under an unchanged key.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import polars as pl

from patron.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def cache_key(name: str, seasons: list[int]) -> str:
    """Filename for a derived artifact built from `seasons`."""
    span = "-".join(str(season) for season in sorted(seasons))
    return f"{name}_{span}.parquet"


def cached_frame(
    name: str,
    seasons: list[int],
    build: Callable[[], pl.DataFrame],
    force: bool = False,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Return a derived frame, building and persisting it on a miss.

    Args:
        name: Artifact name, e.g. `"bonuses"`.
        seasons: Seasons the artifact covers; part of the cache key.
        build: Callable producing the frame. Only invoked on a miss.
        force: Rebuild even on a hit. Used by the nightly refresh, when new games have
            landed under an unchanged key.
        settings: Settings override, mainly for tests.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    path: Path = settings.derived_cache_dir / cache_key(name, seasons)

    if path.exists() and not force:
        logger.debug("derived cache hit: %s", path.name)
        return pl.read_parquet(path)

    logger.info("building derived artifact: %s", path.name)
    frame = build()
    frame.write_parquet(path)
    return frame


def clear(settings: Settings | None = None) -> int:
    """Delete every derived artifact. Returns how many files were removed."""
    settings = settings or get_settings()
    if not settings.derived_cache_dir.exists():
        return 0

    removed = 0
    for path in settings.derived_cache_dir.glob("*.parquet"):
        path.unlink()
        removed += 1
    return removed
