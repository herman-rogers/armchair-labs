"""Bounded decoded Parquet cache, invalidated by file replacement or edit."""

from collections import OrderedDict
from pathlib import Path
from threading import RLock

import polars as pl

from engine.data.verification import observe, revision

_frames: OrderedDict[tuple, pl.DataFrame] = OrderedDict()
_lock = RLock()
MAX_BYTES = 128 * 1024 * 1024


def read_frame(
    path: Path, *, columns: list[str] | None = None, player_id: str | None = None
) -> pl.DataFrame:
    # Application HTTP reads use registered native DuckDB tables. Research and
    # generation jobs retain the bounded Parquet reader below.
    from engine.tables.application import active
    from engine.tables.application import read_frame as query_frame

    if active():
        return query_frame(path, columns=columns, player_id=player_id)
    observe(path)
    key = (
        str(path.absolute()),
        revision(str(path)),
        tuple(columns) if columns else None,
        player_id,
    )
    with _lock:
        if key in _frames:
            _frames.move_to_end(key)
            return _frames[key].clone()
        if player_id is None:
            frame = pl.read_parquet(path, columns=columns)
        else:
            scan = pl.scan_parquet(path).filter(pl.col("player_id") == player_id)
            frame = (scan.select(columns) if columns else scan).collect()
        if revision(str(path)) != key[1]:
            raise ValueError(f"Parquet changed while reading: {path.name}")
        if frame.estimated_size() <= MAX_BYTES:
            # Discard obsolete revisions as well as enforcing a total memory budget.
            for old in list(_frames):
                if old[0] == key[0] and old[2:] == key[2:]:
                    del _frames[old]
            _frames[key] = frame
            while sum(f.estimated_size() for f in _frames.values()) > MAX_BYTES:
                _frames.popitem(last=False)
        return frame.clone()
